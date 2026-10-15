"""From-scratch DPO training loop (no TRL / HF Trainer).

The trainable policy (fp32 weights, fp16 autocast + GradScaler) and the frozen
reference (fp16, no_grad) can live on different GPUs. Chosen and rejected rows
are scored in one forward pass per model. Validation tracks DPO loss, reward
accuracy (strict r_w > r_l) and reward margin; the best checkpoint is chosen by
validation reward accuracy.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pa2.sft import set_seed
from submission_adapter import compute_dpo_loss_from_logps, compute_response_logprobs


def _collate(rows):
    return (
        [r["prompt"] for r in rows],
        [r["chosen"] for r in rows],
        [r["rejected"] for r in rows],
    )


def pair_logps(model, tokenizer, prompts, chosen, rejected, max_length):
    """Summed response log-probs for chosen and rejected, from one concatenated forward pass."""
    logps = compute_response_logprobs(model, tokenizer, prompts + prompts, chosen + rejected, max_length)
    n = len(prompts)
    return logps[:n], logps[n:]


def dpo_step_outputs(policy, reference, tokenizer, batch, beta, max_length, use_amp):
    prompts, chosen, rejected = batch
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        pc, pr = pair_logps(policy, tokenizer, prompts, chosen, rejected, max_length)
    with torch.no_grad():
        rc, rr = pair_logps(reference, tokenizer, prompts, chosen, rejected, max_length)
    rc, rr = rc.to(pc.device), rr.to(pc.device)
    loss = compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    reward_chosen = beta * (pc - rc).detach()
    reward_rejected = beta * (pr - rr).detach()
    return loss, reward_chosen, reward_rejected


@torch.no_grad()
def evaluate_dpo(policy, reference, tokenizer, rows, beta, max_length, batch_size, use_amp):
    policy.eval()
    losses, rc_all, rr_all = [], [], []
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    for batch in tqdm(loader, desc="val", leave=False):
        loss, rc, rr = dpo_step_outputs(policy, reference, tokenizer, batch, beta, max_length, use_amp)
        losses.append(loss.item() * len(batch[0]))
        rc_all.append(rc.float().cpu())
        rr_all.append(rr.float().cpu())
    policy.train()
    rc_all, rr_all = torch.cat(rc_all), torch.cat(rr_all)
    return {
        "val_loss": sum(losses) / len(rows),
        "val_reward_accuracy": (rc_all > rr_all).float().mean().item(),
        "val_reward_margin": (rc_all - rr_all).mean().item(),
        "val_reward_chosen": rc_all.mean().item(),
        "val_reward_rejected": rr_all.mean().item(),
    }


def train_dpo(policy, reference, tokenizer, train_rows, val_rows, output_dir, cfg):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    metrics_path = output_dir / "metrics.jsonl"

    def log(record):
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    set_seed(cfg["seed"])
    device = next(policy.parameters()).device
    use_amp = device.type == "cuda" and cfg["fp16"]
    policy.config.use_cache = False
    if cfg["gradient_checkpointing"]:
        policy.gradient_checkpointing_enable()
    policy.train()
    reference.eval()
    reference.requires_grad_(False)

    loader = DataLoader(
        train_rows,
        batch_size=cfg["per_device_batch_size"],
        shuffle=True,
        collate_fn=_collate,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    accum = cfg["grad_accum"]
    total_steps = math.ceil(len(loader) * cfg["epochs"] / accum)

    params = [p for p in policy.parameters() if p.requires_grad]
    if cfg["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop(params, lr=cfg["lr"])
    else:
        optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_acc = -1.0
    step, start = 0, time.time()
    window = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}
    pbar = tqdm(total=total_steps, desc="dpo")

    def run_eval():
        nonlocal best_acc
        m = evaluate_dpo(policy, reference, tokenizer, val_rows, cfg["beta"], cfg["max_length"],
                         cfg["eval_batch_size"], use_amp)
        is_best = m["val_reward_accuracy"] > best_acc
        if is_best:
            best_acc = m["val_reward_accuracy"]
            policy.save_pretrained(output_dir / "best")
            tokenizer.save_pretrained(output_dir / "best")
        log({"step": step, **m, "is_best": is_best, "elapsed_min": (time.time() - start) / 60})
        pbar.write(f"step {step}: val_loss {m['val_loss']:.4f} | acc {m['val_reward_accuracy']:.3f} "
                   f"| margin {m['val_reward_margin']:.4f}{' (best)' if is_best else ''}")

    run_eval()
    for epoch in range(cfg["epochs"]):
        for micro, batch in enumerate(loader):
            loss, rc, rr = dpo_step_outputs(policy, reference, tokenizer, batch, cfg["beta"],
                                            cfg["max_length"], use_amp)
            scaler.scale(loss / accum).backward()
            window["loss"] += loss.item()
            window["acc"] += (rc > rr).float().mean().item()
            window["margin"] += (rc - rr).mean().item()
            window["n"] += 1

            if (micro + 1) % accum != 0 and micro + 1 != len(loader):
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            pbar.update(1)

            if step % cfg["log_every"] == 0:
                n = window["n"]
                log({"step": step, "epoch": epoch, "train_loss": window["loss"] / n,
                     "train_reward_accuracy": window["acc"] / n, "train_reward_margin": window["margin"] / n,
                     "grad_norm": float(grad_norm), "loss_scale": scaler.get_scale() if use_amp else 1.0})
                pbar.set_postfix(loss=f"{window['loss'] / n:.4f}", acc=f"{window['acc'] / n:.2f}")
                window = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}
            if step % cfg["eval_every"] == 0 and step != total_steps:
                run_eval()

    run_eval()
    policy.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    pbar.close()
    return best_acc

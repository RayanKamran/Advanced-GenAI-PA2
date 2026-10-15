"""From-scratch SFT training loop (no HF Trainer).

fp32 weights + fp16 autocast + GradScaler, gradient accumulation, clipping,
cosine schedule with warmup, periodic validation, best/final checkpoints, and
a metrics.jsonl file appended after every log/eval step.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from submission_adapter import build_sft_batch, compute_response_logprobs, compute_sft_loss


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _collate(rows):
    return [r["prompt"] for r in rows], [r["response"] for r in rows]


@torch.no_grad()
def evaluate_sft(model, tokenizer, rows, max_length, batch_size, use_amp):
    """Token-level mean response NLL over the whole validation set."""
    model.eval()
    total_nll, total_tokens = 0.0, 0
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    for prompts, responses in tqdm(loader, desc="val", leave=False):
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logps = compute_response_logprobs(model, tokenizer, prompts, responses, max_length)
        labels = build_sft_batch(tokenizer, prompts, responses, max_length)["labels"]
        total_nll += -logps.float().sum().item()
        total_tokens += int((labels[:, 1:] != -100).sum())
    model.train()
    return total_nll / max(total_tokens, 1)


def train_sft(model, tokenizer, train_rows, val_rows, output_dir, cfg):
    """Train `model` in place. `cfg` is a dict of hyperparameters (see scripts/train_sft.py)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    metrics_path = output_dir / "metrics.jsonl"

    def log(record):
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    set_seed(cfg["seed"])
    device = next(model.parameters()).device
    use_amp = device.type == "cuda" and cfg["fp16"]
    model.config.use_cache = False
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    model.train()

    loader = DataLoader(
        train_rows,
        batch_size=cfg["per_device_batch_size"],
        shuffle=True,
        collate_fn=_collate,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    accum = cfg["grad_accum"]
    total_steps = math.ceil(len(loader) * cfg["epochs"] / accum)

    # No weight decay on biases and norm weights (standard practice).
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"],
        betas=(0.9, 0.95),
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(cfg["warmup_ratio"] * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    step, running, running_n, start = 0, 0.0, 0, time.time()
    pbar = tqdm(total=total_steps, desc="sft")

    def run_eval():
        nonlocal best_val
        val_loss = evaluate_sft(model, tokenizer, val_rows, cfg["max_length"], cfg["eval_batch_size"], use_amp)
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            model.save_pretrained(output_dir / "best")
            tokenizer.save_pretrained(output_dir / "best")
        log({"step": step, "val_loss": val_loss, "best_val_loss": best_val, "is_best": is_best,
             "elapsed_min": (time.time() - start) / 60})
        pbar.write(f"step {step}: val_loss {val_loss:.4f}{' (best)' if is_best else ''}")

    run_eval()  # step-0 validation: the starting point for the plot
    for epoch in range(cfg["epochs"]):
        for micro, (prompts, responses) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = compute_sft_loss(model, tokenizer, prompts, responses, cfg["max_length"])
            scaler.scale(loss / accum).backward()
            running += loss.item()
            running_n += 1

            if (micro + 1) % accum != 0 and micro + 1 != len(loader):
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            pbar.update(1)

            if step % cfg["log_every"] == 0:
                log({"step": step, "epoch": epoch, "train_loss": running / running_n,
                     "lr": scheduler.get_last_lr()[0], "grad_norm": float(grad_norm),
                     "loss_scale": scaler.get_scale() if use_amp else 1.0})
                pbar.set_postfix(loss=f"{running / running_n:.4f}")
                running, running_n = 0.0, 0
            if step % cfg["eval_every"] == 0 and step != total_steps:
                run_eval()

    run_eval()
    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    pbar.close()
    return best_val

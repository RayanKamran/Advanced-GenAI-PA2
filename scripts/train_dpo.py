"""Train a from-scratch DPO policy from the SFT checkpoint.

Example (Kaggle, SFT notebook output attached as input):
    python scripts/train_dpo.py --dataset hh_rlhf \
        --sft_path /kaggle/input/<sft-notebook>/Advanced-GenAI-PA2/outputs/sft/best \
        --output_dir outputs/dpo_hh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pa2.data import PREFERENCE_SOURCES, load_preference_data
from pa2.dpo import train_dpo
from pa2.sft import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=sorted(PREFERENCE_SOURCES))
    p.add_argument("--sft_path", required=True, help="SFT best checkpoint (policy init and frozen reference)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_train", type=int, default=15_000)
    p.add_argument("--n_val", type=int, default=1_000)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--optimizer", choices=["rmsprop", "adamw"], default="rmsprop")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--policy_device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--reference_device", default=None, help="default: cuda:1 if available, else the policy device")
    args = p.parse_args()

    if args.reference_device is None:
        args.reference_device = "cuda:1" if torch.cuda.device_count() > 1 else args.policy_device

    cfg = vars(args).copy()
    cfg["fp16"] = not cfg.pop("no_fp16")
    cfg["effective_batch_size"] = args.per_device_batch_size * args.grad_accum
    cfg["lr_schedule"] = "constant"
    print(json.dumps(cfg, indent=2))

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.sft_path)
    train_rows, val_rows, stats = load_preference_data(
        args.dataset, tokenizer, args.n_train, args.n_val, args.max_prompt_length, args.seed
    )
    print("data:", json.dumps(stats))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_stats.json").write_text(json.dumps(stats, indent=2))

    # Policy: fp32 weights (fp16 only inside autocast). Reference: frozen fp16 copy.
    policy = AutoModelForCausalLM.from_pretrained(args.sft_path, torch_dtype=torch.float32).to(args.policy_device)
    ref_dtype = torch.float16 if args.reference_device.startswith("cuda") else torch.float32
    reference = AutoModelForCausalLM.from_pretrained(args.sft_path, torch_dtype=ref_dtype).to(args.reference_device)

    best = train_dpo(policy, reference, tokenizer, train_rows, val_rows, args.output_dir, cfg)
    print(f"done; best val reward accuracy {best:.4f}")


if __name__ == "__main__":
    main()

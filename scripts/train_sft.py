"""Train the SFT model on single-turn UltraChat data.

Example (Kaggle):
    python scripts/train_sft.py --output_dir outputs/sft
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pa2.data import load_sft_data
from pa2.sft import set_seed, train_sft


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--output_dir", default="outputs/sft")
    p.add_argument("--n_train", type=int, default=30_000)
    p.add_argument("--n_val", type=int, default=1_000)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = vars(args).copy()
    cfg["fp16"] = not cfg.pop("no_fp16")
    cfg["effective_batch_size"] = args.per_device_batch_size * args.grad_accum
    cfg["loss_reduction"] = "token-level mean of response NLL"
    print(json.dumps(cfg, indent=2))

    set_seed(args.seed)
    train_rows, val_rows = load_sft_data(args.n_train, args.n_val, args.seed)
    print(f"train {len(train_rows)} | val {len(val_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # Trainable weights stay fp32; fp16 is only used inside autocast.
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32).to(args.device)

    best = train_sft(model, tokenizer, train_rows, val_rows, args.output_dir, cfg)
    print(f"done; best val loss {best:.4f}")


if __name__ == "__main__":
    main()

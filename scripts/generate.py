"""Generate responses for a fixed eval set and save them as JSON.

Example:
    python scripts/generate.py --model_path Qwen/Qwen2.5-0.5B \
        --eval_path eval_sets/alpaca_eval_300_seed42.json \
        --generator qwen2.5-0.5b-base \
        --output_path outputs/generations/base_alpaca_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pa2.generation import generate_responses
from pa2.prompting import TEMPLATE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--generator", required=True, help="name stored in each output row")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--limit", type=int, default=None, help="only the first N prompts (sanity check)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_path)
    if out.exists() and not args.overwrite:
        print(f"{out} already exists; skipping (pass --overwrite to regenerate)")
        return

    rows = json.loads(Path(args.eval_path).read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype).to(args.device).eval()

    responses = generate_responses(
        model,
        tokenizer,
        [r["instruction"] for r in rows],
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    results = [
        {
            "id": r["id"],
            "instruction": r["instruction"],
            "output": response,
            "generator": args.generator,
            "dataset": r.get("dataset"),
            **({"category": r["category"]} if "category" in r else {}),
        }
        for r, response in zip(rows, responses)
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    config = {k: v for k, v in vars(args).items() if k != "overwrite"}
    config.update(decoding="greedy", prompt_template=TEMPLATE)
    out.with_suffix(".config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"wrote {len(results)} generations to {out}")


if __name__ == "__main__":
    main()

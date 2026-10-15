"""Create the fixed AlpacaEval subset (default: 300 prompts, seed 42)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", default="eval_sets/alpaca_eval_300_seed42.json")
    args = parser.parse_args()

    path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)

    indices = random.Random(args.seed).sample(range(len(rows)), args.n)
    subset = [
        {
            "id": f"alpaca-eval-{args.seed}-{i:04d}",
            "source_index": idx,
            "instruction": rows[idx]["instruction"],
            "dataset": "alpaca_eval",
            "source_dataset": rows[idx].get("dataset"),
        }
        for i, idx in enumerate(indices)
    ]

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(subset)} of {len(rows)} prompts to {out}")


if __name__ == "__main__":
    main()

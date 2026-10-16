"""Build a blind pairwise-comparison JSONL from two generation files.

Each row randomly places the two models' outputs in positions A and B. The
mapping from position to model is kept under "hidden" and is never shown to the
judge.

Example:
    python scripts/build_pairs.py \
        --target outputs/generations/dpo_hh_hh_eval.json \
        --baseline outputs/generations/sft_hh_eval.json \
        --output_path outputs/eval/dpo_hh_vs_sft_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, help="generations of the model being evaluated (e.g. DPO)")
    p.add_argument("--baseline", required=True, help="generations of the baseline (e.g. SFT)")
    p.add_argument("--output_path", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    target = json.loads(Path(args.target).read_text(encoding="utf-8"))
    baseline = {r["id"]: r for r in json.loads(Path(args.baseline).read_text(encoding="utf-8"))}
    rng = random.Random(args.seed)

    rows = []
    for t in target:
        b = baseline[t["id"]]
        assert b["instruction"] == t["instruction"], f"prompt mismatch for {t['id']}"
        target_is_a = rng.random() < 0.5
        a, bb = (t, b) if target_is_a else (b, t)
        rows.append({
            "pair_id": t["id"],
            "instruction": t["instruction"],
            "response_a": a["output"],
            "response_b": bb["output"],
            "dataset": t.get("dataset"),
            "category": t.get("category"),
            "hidden": {"model_a": a["generator"], "model_b": bb["generator"],
                       "target": t["generator"], "baseline": b["generator"]},
        })

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_target_a = sum(r["hidden"]["model_a"] == r["hidden"]["target"] for r in rows)
    print(f"wrote {len(rows)} pairs to {out} (target in position A: {n_target_a})")


if __name__ == "__main__":
    main()

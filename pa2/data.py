"""Dataset loading: single-turn instruction/response pairs for SFT."""

from __future__ import annotations

from datasets import load_dataset

from pa2.prompting import format_prompt

ULTRACHAT = "HuggingFaceH4/ultrachat_200k"


def _single_turn(example):
    """Keep only the first user -> assistant exchange of a conversation."""
    messages = example["messages"]
    if len(messages) < 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return None
    instruction, response = messages[0]["content"].strip(), messages[1]["content"].strip()
    if not instruction or not response:
        return None
    return {"instruction": instruction, "response": response}


def load_sft_data(n_train=30_000, n_val=1_000, seed=42):
    """Return (train, val) lists of {"prompt", "response", "instruction"} dicts.

    Train comes from ultrachat_200k/train_sft and validation from test_sft, each
    shuffled with `seed` before taking the first n rows.
    """
    splits = {}
    for name, split, n in (("train", "train_sft", n_train), ("val", "test_sft", n_val)):
        ds = load_dataset(ULTRACHAT, split=split).shuffle(seed=seed)
        rows = []
        for example in ds:
            row = _single_turn(example)
            if row is not None:
                row["prompt"] = format_prompt(row["instruction"])
                rows.append(row)
                if len(rows) == n:
                    break
        splits[name] = rows
    return splits["train"], splits["val"]

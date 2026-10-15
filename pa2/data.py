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


# ---------------------------------------------------------------------------
# Preference data for DPO: rows of {"instruction", "prompt", "chosen", "rejected"}
# ---------------------------------------------------------------------------

HH_RLHF = "Anthropic/hh-rlhf"
ULTRAFEEDBACK = "HuggingFaceH4/ultrafeedback_binarized"


def _hh_single_turn(example):
    """Parse a single-turn HH-RLHF pair; multi-turn transcripts are dropped."""
    chosen, rejected = example["chosen"], example["rejected"]
    if chosen.count("\n\nHuman:") != 1 or rejected.count("\n\nHuman:") != 1:
        return None
    if chosen.count("\n\nAssistant:") != 1 or rejected.count("\n\nAssistant:") != 1:
        return None
    prompt_c, chosen_resp = chosen.split("\n\nAssistant:")
    prompt_r, rejected_resp = rejected.split("\n\nAssistant:")
    if prompt_c != prompt_r:
        return None
    return {
        "instruction": prompt_c.replace("\n\nHuman:", "", 1).strip(),
        "chosen": chosen_resp.strip(),
        "rejected": rejected_resp.strip(),
    }


def _uf_single_turn(example):
    chosen, rejected = example["chosen"], example["rejected"]
    if len(chosen) != 2 or len(rejected) != 2:
        return None
    return {
        "instruction": example["prompt"].strip(),
        "chosen": chosen[1]["content"].strip(),
        "rejected": rejected[1]["content"].strip(),
    }


PREFERENCE_SOURCES = {
    # name: (hub id, train split, val split, parser)
    "hh_rlhf": (HH_RLHF, "train", "test", _hh_single_turn),
    "ultrafeedback": (ULTRAFEEDBACK, "train_prefs", "test_prefs", _uf_single_turn),
}


def load_preference_data(name, tokenizer, n_train=15_000, n_val=1_000, max_prompt_length=256, seed=42):
    """Return (train, val, stats) for a preference dataset.

    Keeps single-turn pairs with non-empty, non-identical responses whose formatted
    prompt fits in `max_prompt_length` tokens (longer prompts are dropped rather
    than cut, so no example loses its question). Responses longer than the total
    max_length are truncated later by build_sft_batch.
    """
    hub_id, train_split, val_split, parse = PREFERENCE_SOURCES[name]
    splits, stats = {}, {}
    for split_name, split, n in (("train", train_split, n_train), ("val", val_split, n_val)):
        ds = load_dataset(hub_id, split=split).shuffle(seed=seed)
        rows, skipped = [], {"multi_turn_or_malformed": 0, "empty_or_identical": 0, "prompt_too_long": 0}
        for example in ds:
            row = parse(example)
            if row is None:
                skipped["multi_turn_or_malformed"] += 1
                continue
            if not row["instruction"] or not row["chosen"] or not row["rejected"] or row["chosen"] == row["rejected"]:
                skipped["empty_or_identical"] += 1
                continue
            row["prompt"] = format_prompt(row["instruction"])
            if len(tokenizer.encode(row["prompt"], add_special_tokens=False)) > max_prompt_length:
                skipped["prompt_too_long"] += 1
                continue
            rows.append(row)
            if len(rows) == n:
                break
        splits[split_name] = rows
        stats[split_name] = {"kept": len(rows), "skipped": skipped}
    return splits["train"], splits["val"], stats

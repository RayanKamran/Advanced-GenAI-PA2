"""Batched, deterministic generation with the shared prompt template."""

from __future__ import annotations

import torch
from tqdm.auto import tqdm

from pa2.prompting import STOP_STRINGS, clean_response, format_prompt


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    instructions,
    max_new_tokens=512,
    batch_size=16,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
):
    """Greedy-decode one response per instruction, preserving input order."""
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"  # decoder-only batching needs left padding
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs = []
    for start in tqdm(range(0, len(instructions), batch_size), desc="generate"):
        prompts = [format_prompt(x) for x in instructions[start:start + batch_size]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        generated = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            stop_strings=STOP_STRINGS,
            tokenizer=tokenizer,
        )
        new_tokens = generated[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        outputs.extend(clean_response(t) for t in texts)
    return outputs

"""Single prompt template shared by SFT, DPO, and generation.

A plain-text instruction/response template is used instead of Qwen's ChatML
chat template: the base model has never been trained on ChatML turns, and the
response ends with the tokenizer's EOS (<|endoftext|>), which is exactly what
build_sft_batch appends. Keep this identical across every stage.
"""

from __future__ import annotations

TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

# Base/weak models often keep going and invent the next turn; generation stops here.
STOP_STRINGS = ["\n### Instruction:"]


def format_prompt(instruction: str) -> str:
    return TEMPLATE.format(instruction=instruction.strip())


def clean_response(text: str) -> str:
    """Cut a generated continuation at the first stop string and strip whitespace."""
    for stop in STOP_STRINGS:
        index = text.find(stop)
        if index != -1:
            text = text[:index]
    return text.strip()

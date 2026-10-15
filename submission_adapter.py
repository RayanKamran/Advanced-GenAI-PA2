"""Stable grading interface for the SFT and DPO implementation.

Keep the function signatures below unchanged. Each function should be a thin
wrapper around your own implementation; you do not need to place the training
code itself in this file.
"""

from __future__ import annotations
import torch


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""
    input_ids = []
    attention_mask = []
    labels = []

    # EOS is appended only when the tokenizer has one; pad falls back to EOS, then 0
    # (padding positions are masked out, so any valid id works).
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    for prompt, response in zip(prompts, responses):
        p = tokenizer.encode(text = prompt, add_special_tokens=False)
        r = tokenizer.encode(text = response, add_special_tokens=False) + eos

        # Truncate after appending EOS, then right-pad:
        ids = (p + r)[:max_length]
        n_pad = max_length - len(ids)

        input_ids.append(ids + [pad_id] * n_pad)

        # attention_mask:
        attention_mask.append([1] * len(ids) + [0] * n_pad)

        # labels (prompt and padding masked):
        current_label = ([-100] * len(p) + r)[:max_length]
        labels.append(current_label + [-100] * n_pad)

    return {"input_ids": torch.tensor(input_ids, dtype = torch.long), "attention_mask": torch.tensor(attention_mask, dtype = torch.long), 
            "labels": torch.tensor(labels, dtype = torch.long)}

def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)

    # Move the batch to wherever the model lives (e.g. cuda:0 policy, cuda:1 reference):
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # Model, and raw logits:
    output = model(input_ids = input_ids, attention_mask = attention_mask)
    # Upcast so an fp16 model (e.g. the frozen DPO reference) gets an fp32 log-softmax:
    logits = output.logits.float()

    # Log Softmax, with slicing:
    log_probs = torch.log_softmax(logits, dim = -1)
    log_probs_sliced = log_probs[:, :-1]
    labels_sliced = labels[:, 1:]

    # Choosing one log prob, based on label:
    ignore_index = -100
    mask = labels_sliced != ignore_index
    safe_labels = labels_sliced.clamp_min(0)

    single_log_probs = torch.gather(log_probs_sliced, -1, safe_labels.unsqueeze(-1)).squeeze(-1)
    single_log_probs = single_log_probs * mask

    # Summing all response probs:
    summed_log_probs = single_log_probs.sum(dim = 1) 

    return summed_log_probs

def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss using either documented reduction."""
    log_prob = compute_response_logprobs(model = model, tokenizer = tokenizer, prompts = prompts, responses = responses, max_length = max_length)

    # Token-level mean: total response NLL / number of scored response tokens.
    # Shifted labels match the positions summed in compute_response_logprobs.
    labels = build_sft_batch(tokenizer, prompts, responses, max_length)["labels"]
    sum_log_probs = torch.sum(log_prob)
    count_log_probs = max(int((labels[:, 1:] != -100).sum()), 1)

    return (-1 *sum_log_probs) / count_log_probs

def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""
    chosen = build_sft_batch(tokenizer = tokenizer, prompts = prompts, responses = chosen_responses, max_length = max_length)
    rejected = build_sft_batch(tokenizer = tokenizer, prompts = prompts, responses = rejected_responses, max_length = max_length)

    return {"chosen": chosen, "rejected": rejected}

def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""
    z = beta * ((policy_chosen_logps - policy_rejected_logps) - (reference_chosen_logps - reference_rejected_logps))

    loss = torch.nn.functional.logsigmoid(z)

    return -loss.mean()


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""
    policy_chosen_logps = compute_response_logprobs(model = policy_model, tokenizer = tokenizer, prompts = prompts, responses = chosen_responses, max_length = max_length)
    policy_rejected_logps = compute_response_logprobs(model = policy_model, tokenizer = tokenizer, prompts = prompts, responses = rejected_responses, max_length = max_length)

    with torch.no_grad():
        reference_chosen_logps = compute_response_logprobs(model = reference_model, tokenizer = tokenizer, prompts = prompts, responses = chosen_responses, max_length = max_length)
        reference_rejected_logps = compute_response_logprobs(model = reference_model, tokenizer = tokenizer, prompts = prompts, responses = rejected_responses, max_length = max_length)

    dpo_loss = compute_dpo_loss_from_logps(policy_chosen_logps = policy_chosen_logps, policy_rejected_logps = policy_rejected_logps, 
                                           reference_chosen_logps = reference_chosen_logps, reference_rejected_logps = reference_rejected_logps, beta = beta)

    return dpo_loss
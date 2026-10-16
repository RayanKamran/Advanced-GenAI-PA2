"""Task 5: TRL DPOTrainer + LoRA persona tuning on the Ramsay-style preference data.

Trains a LoRA adapter on HuggingFaceTB/SmolLM2-360M-Instruct, then generates
responses from the base model and the LoRA model on the fixed style-eval prompts
(eval_sets/persona_style_eval.json) and builds a blind pairs file for judging.

Example (Kaggle):
    pip install -q "trl==0.24.0" "peft>=0.13" --no-deps
    python scripts/train_lora_persona.py --output_dir outputs/lora_persona
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

# One GPU is plenty for a 360M model + LoRA. With two visible GPUs the HF Trainer
# wraps the model in DataParallel, which breaks with PEFT (device mismatch).
# Must be set before torch initialises CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def to_conversational(row):
    """TRL conversational preference format: role-message lists."""
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


@torch.no_grad()
def chat_generate(model, tokenizer, instructions, max_new_tokens, batch_size=8):
    """Greedy decoding with the model's own chat template (SmolLM2-Instruct is a chat model)."""
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    outputs = []
    for i in range(0, len(instructions), batch_size):
        texts = [
            tokenizer.apply_chat_template([{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True)
            for x in instructions[i:i + batch_size]
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        outputs += [t.strip() for t in tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--data_path", default="data/persona_lora/ramsay_style_preferences.json")
    p.add_argument("--eval_path", default="eval_sets/persona_style_eval.json")
    p.add_argument("--output_dir", default="outputs/lora_persona")
    p.add_argument("--n_val", type=int, default=20)
    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    # DPO / optimisation
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per_device_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2))
    torch.manual_seed(args.seed)
    cuda = torch.cuda.is_available()

    rows = json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    random.Random(args.seed).shuffle(rows)
    val_rows, train_rows = rows[: args.n_val], rows[args.n_val:]
    train_ds = Dataset.from_list([to_conversational(r) for r in train_rows])
    val_ds = Dataset.from_list([to_conversational(r) for r in val_rows])
    print(f"train {len(train_ds)} | val {len(val_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # fp32 weights; fp16 autocast via the trainer on GPU. With peft_config and no
    # ref_model, TRL uses the base model (adapter disabled) as the frozen reference.
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(","),
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_args = DPOConfig(
        output_dir=str(out / "trainer"),
        beta=args.beta,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        fp16=cuda,
        bf16=False,  # T4 has no bf16; TRL 0.24 defaults it on
        gradient_checkpointing=False,  # not needed at this size; TRL 0.24 defaults it on
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()

    # Training curve + final eval metrics (rewards/accuracies, rewards/margins, ...).
    with open(out / "log_history.jsonl", "w", encoding="utf-8") as f:
        for rec in trainer.state.log_history:
            f.write(json.dumps(rec) + "\n")
    adapter_dir = out / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print("saved adapter to", adapter_dir)

    # --- Style evaluation: base vs LoRA on the same prompts, same greedy decoding ---
    device = "cuda:0" if cuda else "cpu"
    eval_rows = json.loads(Path(args.eval_path).read_text(encoding="utf-8"))
    instructions = [r["instruction"] for r in eval_rows]
    del trainer, model
    if cuda:
        torch.cuda.empty_cache()

    dtype = torch.float16 if cuda else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype).to(device).eval()
    base_out = chat_generate(base, tokenizer, instructions, args.max_new_tokens)
    lora = PeftModel.from_pretrained(base, adapter_dir).to(device).eval()
    lora_out = chat_generate(lora, tokenizer, instructions, args.max_new_tokens)

    gen_dir = out / "generations"
    gen_dir.mkdir(exist_ok=True)
    for name, outs in (("smollm2-base", base_out), ("smollm2-lora-dpo", lora_out)):
        records = [{**r, "output": o, "generator": name} for r, o in zip(eval_rows, outs)]
        (gen_dir / f"{name}_persona_eval.json").write_text(json.dumps(records, indent=2, ensure_ascii=False),
                                                           encoding="utf-8")
    for r, b, l in zip(eval_rows, base_out, lora_out):
        text = f"[{r['category']}] {r['instruction']}\n  BASE: {b[:200]}\n  LORA: {l[:200]}\n"
        print(text.encode("ascii", "replace").decode())  # console-safe preview; the JSON keeps the full text


if __name__ == "__main__":
    main()

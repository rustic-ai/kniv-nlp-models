"""LoRA fine-tuning for business domain LLM.

Continued pretraining on business corpus (SEC filings, textbooks, emails,
ERP docs, contracts, academic abstracts, Wikipedia).

Supports any causal LM base model — change model.base in config.yaml:
  - microsoft/Phi-4-mini-reasoning  (default, 3.8B)
  - Qwen/Qwen3-4B
  - google/gemma-4-E4B
  - deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

Usage:
    python models/phi4-mini-llm-en/train.py
    python models/phi4-mini-llm-en/train.py --resume outputs/phi4-mini-llm-en/checkpoint-1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


CONFIG_PATH = Path(__file__).parent / "config.yaml"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "prepared" / "phi4-mini-llm-en"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_chunked_dataset(path: Path, max_length: int) -> Dataset:
    """Load pre-tokenized chunks from JSONL."""
    examples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            ids = data["input_ids"]
            # Pad or truncate to max_length
            if len(ids) > max_length:
                ids = ids[:max_length]
            examples.append({
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": ids,  # causal LM: labels = input_ids
            })
    return Dataset.from_list(examples)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    args = parser.parse_args()

    config = load_config()
    base_model = config["model"]["base"]
    max_length = config["model"]["max_length"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    output_dir = Path(config["output"]["dir"])

    # ── Device ────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # ── Load base model ───────────────────────────────────
    print(f"\nLoading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if train_cfg["bf16"] else torch.float32,
        device_map="auto",
    )

    # ── Apply LoRA ────────────────────────────────────────
    print("Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ── Load data ─────────────────────────────────────────
    print(f"\nLoading training data from {DATA_DIR}")
    train_dataset = load_chunked_dataset(DATA_DIR / "train.jsonl", max_length)
    dev_dataset = load_chunked_dataset(DATA_DIR / "dev.jsonl", max_length)
    print(f"  Train: {len(train_dataset)} chunks")
    print(f"  Dev:   {len(dev_dataset)} chunks")

    # ── Training args ─────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_total_limit=config["output"]["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
    )

    # ── Data collator ─────────────────────────────────────
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM, not masked LM
    )

    # ── Train ─────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
    )

    print(f"\nTraining for {train_cfg['epochs']} epochs")
    print(f"  Effective batch size: {train_cfg['batch_size'] * train_cfg['gradient_accumulation_steps']}")
    print(f"  Learning rate: {train_cfg['learning_rate']}")
    print(f"  LoRA rank: {lora_cfg['r']}, alpha: {lora_cfg['lora_alpha']}")
    print()

    trainer.train(resume_from_checkpoint=args.resume)

    # ── Save ──────────────────────────────────────────────
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapters (not the full model — much smaller)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save training config for reproducibility
    with open(final_dir / "training_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # ── Eval summary ──────────────────────────────────────
    eval_results = trainer.evaluate()
    print(f"\nTraining complete.")
    print(f"  Final eval loss: {eval_results['eval_loss']:.4f}")
    print(f"  Final perplexity: {2 ** eval_results['eval_loss']:.2f}")
    print(f"  LoRA adapters saved to: {final_dir}")
    print(f"  To merge into base model:")
    print(f"    from peft import PeftModel")
    print(f"    model = PeftModel.from_pretrained(base_model, '{final_dir}')")
    print(f"    merged = model.merge_and_unload()")


if __name__ == "__main__":
    train()

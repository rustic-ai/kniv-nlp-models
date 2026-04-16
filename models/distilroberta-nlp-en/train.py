"""Train the multi-task NLP model on UD EWT + CoNLL-2003.

Four tasks: NER, POS, dependency parsing (dep2label), sentence classification.

Usage:
    python models/distilroberta-nlp-en/train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import yaml

from model import MultiTaskNLPModel


# ── Configuration ─────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "config.yaml"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "prepared" / "distilroberta-nlp-en"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Token-level dataset ──────────────────────────────────────────

class TokenClassificationDataset(Dataset):
    """Dataset for token classification with subword alignment."""

    def __init__(
        self,
        examples: list[dict],
        tokenizer,
        label_map: dict[str, int],
        label_key: str,
        max_length: int = 128,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.label_key = label_key
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        words = example["words"]
        labels = example[self.label_key]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Align labels to subword tokens
        word_ids = encoding.word_ids()
        aligned_labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                label_str = labels[word_id] if word_id < len(labels) else "O"
                aligned_labels.append(self.label_map.get(label_str, 0))
            else:
                aligned_labels.append(-100)
            prev_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


# ── Sequence-level dataset ────────────────────────────────────────

class SequenceClassificationDataset(Dataset):
    """Dataset for sentence-level classification (CLS head)."""

    def __init__(
        self,
        examples: list[dict],
        tokenizer,
        label_map: dict[str, int],
        max_length: int = 128,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example.get("text", " ".join(example["words"]))
        cls_label = example["cls_label"]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.label_map[cls_label], dtype=torch.long),
        }


# ── Training loop ─────────────────────────────────────────────────

def train():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # Load label vocabularies
    with open(DATA_DIR / "label_vocabs.json") as f:
        vocabs = json.load(f)

    ner_labels = vocabs["ner_labels"]
    pos_labels = vocabs["pos_labels"]
    dep_labels = vocabs["dep_labels"]
    cls_labels = vocabs["cls_labels"]

    ner_map = {l: i for i, l in enumerate(ner_labels)}
    pos_map = {l: i for i, l in enumerate(pos_labels)}
    dep_map = {l: i for i, l in enumerate(dep_labels)}
    cls_map = {l: i for i, l in enumerate(cls_labels)}

    # Load data
    with open(DATA_DIR / "conll_train.json") as f:
        conll_train = json.load(f)
    with open(DATA_DIR / "ud_train.json") as f:
        ud_train = json.load(f)

    encoder_name = config["model"]["encoder"]
    max_length = config["model"]["max_length"]
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    # Create datasets — 4 tasks
    ner_dataset = TokenClassificationDataset(conll_train, tokenizer, ner_map, "ner_tags", max_length)
    pos_dataset = TokenClassificationDataset(ud_train, tokenizer, pos_map, "pos_tags", max_length)
    dep_dataset = TokenClassificationDataset(ud_train, tokenizer, dep_map, "dep_labels", max_length)

    # CLS dataset: combine UD + CoNLL examples (both have cls_label)
    cls_examples = ud_train + conll_train
    cls_dataset = SequenceClassificationDataset(cls_examples, tokenizer, cls_map, max_length)

    batch_size = config["training"]["batch_size"]
    ner_loader = DataLoader(ner_dataset, batch_size=batch_size, shuffle=True)
    pos_loader = DataLoader(pos_dataset, batch_size=batch_size, shuffle=True)
    dep_loader = DataLoader(dep_dataset, batch_size=batch_size, shuffle=True)
    cls_loader = DataLoader(cls_dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = MultiTaskNLPModel(
        encoder_name=encoder_name,
        ner_labels=ner_labels,
        pos_labels=pos_labels,
        dep_labels=dep_labels,
        cls_labels=cls_labels,
        dropout=config["model"]["dropout"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    total_steps = config["training"]["epochs"] * (
        len(ner_loader) + len(pos_loader) + len(dep_loader) + len(cls_loader)
    )
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    seq_loss_fn = nn.CrossEntropyLoss()

    # Task weights
    task_weights = {
        "ner": config["training"]["ner_loss_weight"],
        "pos": config["training"]["pos_loss_weight"],
        "dep": config["training"]["dep_loss_weight"],
        "cls": config["training"]["cls_loss_weight"],
    }

    output_dir = Path(config["output"]["dir"])

    print(f"\nTraining for {config['training']['epochs']} epochs")
    print(f"  NER batches/epoch: {len(ner_loader)}")
    print(f"  POS batches/epoch: {len(pos_loader)}")
    print(f"  Dep batches/epoch: {len(dep_loader)}")
    print(f"  CLS batches/epoch: {len(cls_loader)}")
    print(f"  Total steps: {total_steps}")
    print()

    for epoch in range(config["training"]["epochs"]):
        model.train()
        total_loss = 0.0
        task_losses = {"ner": 0.0, "pos": 0.0, "dep": 0.0, "cls": 0.0}
        step_count = 0

        # Interleave task batches
        active_tasks = {
            "ner": iter(ner_loader),
            "pos": iter(pos_loader),
            "dep": iter(dep_loader),
            "cls": iter(cls_loader),
        }

        while active_tasks:
            for task_name in list(active_tasks.keys()):
                try:
                    batch = next(active_tasks[task_name])
                except StopIteration:
                    del active_tasks[task_name]
                    continue

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask)

                # Compute loss based on task type
                if task_name == "cls":
                    # Sequence classification: labels are [batch], logits are [batch, num_cls]
                    loss = seq_loss_fn(outputs["cls_logits"], labels)
                else:
                    # Token classification: labels are [batch, seq], logits are [batch, seq, num_labels]
                    logits = outputs[f"{task_name}_logits"]
                    loss = token_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

                weighted_loss = loss * task_weights[task_name]

                optimizer.zero_grad()
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
                optimizer.step()
                scheduler.step()

                total_loss += weighted_loss.item()
                task_losses[task_name] += loss.item()
                step_count += 1

        avg_loss = total_loss / max(step_count, 1)
        task_avg = {k: v / max(step_count // 4, 1) for k, v in task_losses.items()}
        print(
            f"Epoch {epoch + 1}/{config['training']['epochs']} — "
            f"Loss: {avg_loss:.4f} "
            f"[NER: {task_avg['ner']:.4f}, POS: {task_avg['pos']:.4f}, "
            f"Dep: {task_avg['dep']:.4f}, CLS: {task_avg['cls']:.4f}]"
        )

        if config["output"]["save_every_epoch"]:
            epoch_dir = output_dir / f"epoch-{epoch + 1}"
            model.save(str(epoch_dir))
            tokenizer.save_pretrained(str(epoch_dir))

    # Save final model
    model.save(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"\nTraining complete. Model saved to {output_dir / 'final'}")


if __name__ == "__main__":
    train()

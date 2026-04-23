"""Train a CLS (dialog act) silver labeler on SwDA+MRDA data.

Fine-tunes DeBERTa-v3-base on 286K SwDA+MRDA examples mapped to our 9 labels.
The trained model is used as a silver labeler to annotate conversational data.

Usage:
    python train_cls_labeler.py
    python train_cls_labeler.py --output-dir outputs/cls_labeler --epochs 3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from tqdm import tqdm

CLS_LABELS = [
    "inform", "correction", "agreement", "question",
    "plan_commit", "request", "feedback", "social", "filler",
]
LABEL2ID = {l: i for i, l in enumerate(CLS_LABELS)}
ID2LABEL = {i: l for i, l in enumerate(CLS_LABELS)}

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "prepared" / "kniv-deberta-cascade"


class CLSDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = ex["text"]
        prev_text = ex.get("prev_text")

        if prev_text:
            encoding = self.tokenizer(
                prev_text, text,
                max_length=self.max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            )
        else:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(LABEL2ID[ex["cls_label"]], dtype=torch.long),
        }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load data
    with open(DATA_DIR / "cls_swda_mrda_train.json") as f:
        all_examples = json.load(f)
    print(f"Loaded {len(all_examples):,} examples", flush=True)

    # Train/val split (90/10, stratified)
    import random
    random.seed(42)
    random.shuffle(all_examples)
    split = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split]
    val_examples = all_examples[split:]
    print(f"Train: {len(train_examples):,}, Val: {len(val_examples):,}", flush=True)

    # Label distribution
    from collections import Counter
    train_dist = Counter(ex["cls_label"] for ex in train_examples)
    print("Train distribution:", flush=True)
    for label in CLS_LABELS:
        print(f"  {label}: {train_dist[label]:,} ({100*train_dist[label]/len(train_examples):.1f}%)",
              flush=True)

    # Model and tokenizer
    model_name = args.encoder
    print(f"Loading {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(CLS_LABELS),
        id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(device)

    # Datasets
    train_ds = CLSDataset(train_examples, tokenizer, args.max_length)
    val_ds = CLSDataset(val_examples, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Class weights for imbalanced data
    total = sum(train_dist.values())
    weights = torch.tensor(
        [total / (len(CLS_LABELS) * train_dist[l]) for l in CLS_LABELS],
        dtype=torch.float, device=device,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    print(f"Class weights: {[f'{w:.2f}' for w in weights.tolist()]}", flush=True)

    # Training
    best_macro_f1 = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Eval"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1).cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        report = classification_report(
            all_labels, all_preds,
            target_names=CLS_LABELS, digits=3, zero_division=0,
        )
        print(f"\nEpoch {epoch+1}/{args.epochs} — Loss: {avg_loss:.4f}", flush=True)
        print(report, flush=True)

        # Parse macro F1
        from sklearn.metrics import f1_score
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        print(f"Macro F1: {macro_f1:.4f}", flush=True)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            model.save_pretrained(str(output_dir / "best"))
            tokenizer.save_pretrained(str(output_dir / "best"))
            print(f"  * New best — saved to {output_dir / 'best'}", flush=True)

    # Save final
    model.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"\nTraining complete. Best macro F1: {best_macro_f1:.4f}", flush=True)
    print(f"Best model: {output_dir / 'best'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train CLS dialog act silver labeler")
    parser.add_argument("--encoder", type=str, default="microsoft/deberta-v3-base",
                        help="Pretrained encoder model")
    parser.add_argument("--output-dir", type=str, default="outputs/cls_labeler",
                        help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Max token length")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

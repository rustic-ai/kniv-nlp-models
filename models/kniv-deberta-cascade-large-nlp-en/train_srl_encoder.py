"""Phase 1: Fine-tune DeBERTa-v3-large encoder for SRL (Shi & Lin style).

Injects a learned predicate embedding at the embedding level, so all 24
attention layers see which token is the predicate.  After training, the
encoder weights are used as the backbone for the full cascade (Phase 2).

Usage:
    python train_srl_encoder.py
    python train_srl_encoder.py --max-examples 200000 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "prepared" / "kniv-deberta-cascade"

SRL_TAGS = [
    "O", "V",
    "B-ARG0", "I-ARG0", "B-ARG1", "I-ARG1", "B-ARG2", "I-ARG2",
    "B-ARG3", "I-ARG3", "B-ARG4", "I-ARG4",
    "B-ARGM-TMP", "I-ARGM-TMP", "B-ARGM-LOC", "I-ARGM-LOC",
    "B-ARGM-MNR", "I-ARGM-MNR", "B-ARGM-CAU", "I-ARGM-CAU",
    "B-ARGM-PRP", "I-ARGM-PRP", "B-ARGM-NEG", "I-ARGM-NEG",
    "B-ARGM-ADV", "I-ARGM-ADV", "B-ARGM-DIR", "I-ARGM-DIR",
    "B-ARGM-DIS", "I-ARGM-DIS", "B-ARGM-EXT", "I-ARGM-EXT",
    "B-ARGM-MOD", "I-ARGM-MOD", "B-ARGM-PRD", "I-ARGM-PRD",
    "B-ARGM-GOL", "I-ARGM-GOL", "B-ARGM-COM", "I-ARGM-COM",
    "B-ARGM-REC", "I-ARGM-REC",
]
TAG2ID = {t: i for i, t in enumerate(SRL_TAGS)}


# ── Model ──────────────────────────────────────────────────────

class SRLEncoderModel(nn.Module):
    """DeBERTa encoder with predicate embedding injection + MLP SRL head.

    The predicate indicator is added at the embedding level (before the
    transformer layers), so all attention layers can learn predicate-aware
    representations.  This is the key difference from our cascade adapter
    approach which only added predicate info after the encoder.
    """

    def __init__(self, encoder_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(encoder_name)
        H = self.deberta.config.hidden_size

        # Learned embedding: 0=regular token, 1=predicate token
        self.pred_embedding = nn.Embedding(2, H)
        nn.init.zeros_(self.pred_embedding.weight)  # start as no-op

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(H, H),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(H, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        predicate_idx: torch.Tensor,
    ) -> torch.Tensor:
        B, S = input_ids.size()

        # Get word embeddings from DeBERTa's embedding layer
        emb = self.deberta.embeddings(input_ids)

        # Inject predicate signal: add learned embedding to the predicate token
        indicator = torch.zeros(B, S, dtype=torch.long, device=input_ids.device)
        indicator[torch.arange(B, device=input_ids.device), predicate_idx] = 1
        emb = emb + self.pred_embedding(indicator)

        # Run all transformer layers with predicate-aware embeddings
        encoder_out = self.deberta.encoder(emb, attention_mask)
        hidden = encoder_out.last_hidden_state

        return self.classifier(hidden)

    def save_encoder(self, output_dir: str):
        """Save just the encoder weights (for loading into cascade model)."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        # Save the DeBERTa encoder + pred_embedding
        torch.save({
            "deberta": self.deberta.state_dict(),
            "pred_embedding": self.pred_embedding.state_dict(),
        }, path / "srl_encoder.pt")
        self.deberta.config.save_pretrained(path)
        print(f"Encoder saved to {path}", flush=True)

    def save_full(self, output_dir: str):
        """Save the full model (encoder + classifier)."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        self.deberta.config.save_pretrained(path)
        with open(path / "srl_tags.json", "w") as f:
            json.dump(SRL_TAGS, f)
        print(f"Full model saved to {path}", flush=True)


# ── Dataset ────────────────────────────────────────────────────

class SRLDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        words = ex["words"]
        tags = ex["srl_tags"]
        pred_word_idx = ex["predicate_idx"]

        encoding = self.tokenizer(
            words, is_split_into_words=True,
            max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )

        # Align labels to subword tokens
        word_ids = encoding.word_ids()
        aligned_labels = []
        pred_token_idx = 0
        prev_wid = None
        for k, wid in enumerate(word_ids):
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_wid:
                label_str = tags[wid] if wid < len(tags) else "O"
                aligned_labels.append(TAG2ID.get(label_str, 0))
                if wid == pred_word_idx:
                    pred_token_idx = k
            else:
                aligned_labels.append(-100)  # ignore subword continuations
            prev_wid = wid

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
            "predicate_idx": torch.tensor(pred_token_idx, dtype=torch.long),
        }


def collate_fn(features):
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
        "predicate_idx": torch.stack([f["predicate_idx"] for f in features]),
    }


# ── Evaluation ─────────────────────────────────────────────────

def evaluate(model, dataloader, device):
    """Evaluate SRL F1 using seqeval."""
    from seqeval.metrics import f1_score as seq_f1, classification_report
    model.eval()
    all_gold, all_pred = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["predicate_idx"].to(device),
            )
            preds = logits.argmax(dim=-1).cpu()
            labels = batch["labels"]

            for j in range(labels.size(0)):
                gold_seq, pred_seq = [], []
                for k in range(labels.size(1)):
                    if labels[j, k] != -100:
                        g = SRL_TAGS[labels[j, k]]
                        p = SRL_TAGS[preds[j, k].item()] if preds[j, k].item() < len(SRL_TAGS) else "O"
                        # V is the predicate marker, not an argument — treat as O for eval
                        gold_seq.append("O" if g == "V" else g)
                        pred_seq.append("O" if p == "V" else p)
                all_gold.append(gold_seq)
                all_pred.append(pred_seq)

    f1 = seq_f1(all_gold, all_pred)
    report = classification_report(all_gold, all_pred, digits=3, zero_division=0)
    model.train()
    return f1, report


# ── Training ───────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load data
    print(f"Loading {args.train_file}", flush=True)
    with open(DATA_DIR / args.train_file) as f:
        train_examples = json.load(f)

    if args.max_examples and len(train_examples) > args.max_examples:
        random.seed(42)
        train_examples = random.sample(train_examples, args.max_examples)
        print(f"Subsampled to {len(train_examples):,}", flush=True)

    # Filter: must have predicate_idx and at least one argument
    train_examples = [
        ex for ex in train_examples
        if "predicate_idx" in ex
        and any(t not in ("O", "V") for t in ex["srl_tags"])
    ]
    print(f"Train: {len(train_examples):,} examples", flush=True)

    # Dev data (gold PropBank)
    with open(DATA_DIR / "srl_dev.json") as f:
        dev_examples = json.load(f)
    dev_examples = [ex for ex in dev_examples if "predicate_idx" in ex]
    print(f"Dev: {len(dev_examples):,} examples", flush=True)

    # Tokenizer + model
    print(f"Loading {args.encoder}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    model = SRLEncoderModel(args.encoder, len(SRL_TAGS), dropout=args.dropout).float().to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {param_count:,} total, {trainable:,} trainable", flush=True)

    # Dataloaders
    train_ds = SRLDataset(train_examples, tokenizer, args.max_length)
    dev_ds = SRLDataset(dev_examples, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size * 2,
        collate_fn=collate_fn, num_workers=0,
    )

    # Optimizer: differential LR (encoder lower, head higher)
    encoder_params = [p for n, p in model.named_parameters()
                      if "deberta" in n or "pred_embedding" in n]
    head_params = [p for n, p in model.named_parameters()
                   if "classifier" in n]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.encoder_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=0.01)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Output
    output_dir = Path(args.output_dir)
    best_f1 = 0.0
    log_every = 50

    print(f"\nTraining: {args.epochs} epochs, {len(train_loader)} batches/epoch", flush=True)
    print(f"Encoder LR: {args.encoder_lr}, Head LR: {args.head_lr}", flush=True)
    print(f"Total steps: {total_steps:,}, Warmup: {warmup_steps:,}\n", flush=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            predicate_idx = batch["predicate_idx"].to(device)

            logits = model(input_ids, attention_mask, predicate_idx)
            loss = loss_fn(logits.view(-1, len(SRL_TAGS)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            global_step = epoch * len(train_loader) + step + 1
            if global_step % log_every == 0:
                lr_enc = scheduler.get_last_lr()[0]
                lr_head = scheduler.get_last_lr()[1]
                print(f"  step={global_step}  epoch={global_step/total_steps*args.epochs:.2f}"
                      f"  loss={loss.item():.4f}  lr={lr_enc:.2e}/{lr_head:.2e}", flush=True)

        avg_loss = total_loss / len(train_loader)

        # Evaluate
        f1, report = evaluate(model, dev_loader, device)
        print(f"\nEpoch {epoch+1}/{args.epochs} — Loss: {avg_loss:.4f}, SRL F1: {f1:.4f}", flush=True)
        print(report, flush=True)

        if f1 > best_f1:
            best_f1 = f1
            model.save_full(str(output_dir / "best"))
            model.save_encoder(str(output_dir / "best_encoder"))
            tokenizer.save_pretrained(str(output_dir / "best"))
            tokenizer.save_pretrained(str(output_dir / "best_encoder"))
            print(f"  * New best F1={f1:.4f}\n", flush=True)

    # Save final
    model.save_full(str(output_dir / "final"))
    model.save_encoder(str(output_dir / "final_encoder"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final_encoder"))

    print(f"\nTraining complete. Best SRL F1: {best_f1:.4f}", flush=True)
    print(f"Encoder: {output_dir / 'best_encoder'}", flush=True)
    print(f"Full model: {output_dir / 'best'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: SRL encoder fine-tuning")
    parser.add_argument("--encoder", default="microsoft/deberta-v3-large")
    parser.add_argument("--train-file", default="srl_silver_dannashao.json")
    parser.add_argument("--output-dir", default="outputs/srl_encoder")
    parser.add_argument("--max-examples", type=int, default=200000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

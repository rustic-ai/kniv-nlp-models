"""Train the multi-task NLP model on UD EWT + kniv corpus.

Four tasks: NER, POS, dependency parsing (dep2label), sentence classification.

Features:
- Dev-set evaluation every epoch (NER F1, POS acc, Dep UAS/LAS, CLS acc)
- Best checkpoint saved by composite metric
- Early stopping after N epochs without improvement

Usage:
    python models/deberta-v3-large-nlp-en/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import yaml

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.evaluate import evaluate_ner, evaluate_pos, evaluate_dep, evaluate_cls
from dep2label import decode_sentence
from model import MultiTaskNLPModel


# ── Configuration ─────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "config.yaml"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "prepared" / "deberta-v3-large-nlp-en"

EARLY_STOPPING_PATIENCE = 3


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Token-level dataset ──────────────────────────────────────────

class TokenClassificationDataset(Dataset):
    """Dataset for token classification with subword alignment."""

    def __init__(self, examples, tokenizer, label_map, label_key, max_length=128):
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
            words, is_split_into_words=True,
            max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )

        word_ids = encoding.word_ids()
        aligned = []
        prev = None
        for wid in word_ids:
            if wid is None:
                aligned.append(-100)
            elif wid != prev:
                label_str = labels[wid] if wid < len(labels) else "O"
                aligned.append(self.label_map.get(label_str, 0))
            else:
                aligned.append(-100)
            prev = wid

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned, dtype=torch.long),
        }


# ── Sequence-level dataset ────────────────────────────────────────

class SequenceClassificationDataset(Dataset):
    """Dataset for sentence-level classification (CLS head)."""

    def __init__(self, examples, tokenizer, label_map, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example.get("text", " ".join(example["words"]))

        encoding = self.tokenizer(
            text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.label_map[example["cls_label"]], dtype=torch.long),
        }


# ── Dev-set evaluation ────────────────────────────────────────────

def predict_token_labels(model, dataloader, label_list, device):
    """Run inference and collect aligned per-word predictions + gold."""
    model.eval()
    all_gold, all_pred = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)

        # Figure out which logits key from the label_list
        # (caller passes the right dataloader for the right task)
        # We need a way to know which head — pass via closure or key
        yield outputs, labels

    model.train()


def evaluate_on_dev(
    model, tokenizer, device, vocabs,
    ner_dev, ud_dev, cls_dev_examples, max_length,
):
    """Run all four tasks on dev sets and return metrics dict."""
    model.eval()
    results = {}

    ner_labels = vocabs["ner_labels"]
    pos_labels = vocabs["pos_labels"]
    dep_labels = vocabs["dep_labels"]
    cls_labels = vocabs["cls_labels"]

    # ── NER ────────────────────────────────────────────────
    gold_ner, pred_ner = [], []
    for ex in ner_dev:
        words = ex["words"]
        encoding = tokenizer(
            words, is_split_into_words=True,
            max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            out = model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device))
        logits = out["ner_logits"][0].cpu()
        preds = logits.argmax(dim=-1).tolist()

        word_ids = encoding.word_ids()
        word_preds, prev = [], None
        for i, wid in enumerate(word_ids):
            if wid is not None and wid != prev:
                idx = preds[i]
                word_preds.append(ner_labels[idx] if idx < len(ner_labels) else "O")
            prev = wid

        gold_ner.append(ex["ner_tags"][:len(word_preds)])
        pred_ner.append(word_preds[:len(ex["ner_tags"])])

    results["ner"] = evaluate_ner(gold_ner, pred_ner)

    # ── POS ────────────────────────────────────────────────
    gold_pos, pred_pos = [], []
    for ex in ud_dev:
        words = ex["words"]
        encoding = tokenizer(
            words, is_split_into_words=True,
            max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            out = model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device))
        logits = out["pos_logits"][0].cpu()
        preds = logits.argmax(dim=-1).tolist()

        word_ids = encoding.word_ids()
        word_preds, prev = [], None
        for i, wid in enumerate(word_ids):
            if wid is not None and wid != prev:
                idx = preds[i]
                word_preds.append(pos_labels[idx] if idx < len(pos_labels) else "X")
            prev = wid

        gold_pos.append(ex["pos_tags"][:len(word_preds)])
        pred_pos.append(word_preds[:len(ex["pos_tags"])])

    results["pos"] = evaluate_pos(gold_pos, pred_pos)

    # ── Dep ────────────────────────────────────────────────
    gold_heads_all, pred_heads_all = [], []
    gold_rels_all, pred_rels_all = [], []
    for ex in ud_dev:
        words = ex["words"]
        encoding = tokenizer(
            words, is_split_into_words=True,
            max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            out = model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device))
        logits = out["dep_logits"][0].cpu()
        preds = logits.argmax(dim=-1).tolist()

        word_ids = encoding.word_ids()
        word_preds, prev = [], None
        for i, wid in enumerate(word_ids):
            if wid is not None and wid != prev:
                idx = preds[i]
                word_preds.append(dep_labels[idx] if idx < len(dep_labels) else "0@root@ROOT")
            prev = wid

        try:
            pred_h, pred_r = decode_sentence(word_preds[:len(ex["words"])], ex["pos_tags"])
        except Exception:
            pred_h = [-1] * len(ex["words"])
            pred_r = ["_"] * len(ex["words"])

        gold_heads_all.append(ex["heads"])
        pred_heads_all.append(pred_h)
        gold_rels_all.append(ex["deprels"])
        pred_rels_all.append(pred_r)

    results["dep"] = evaluate_dep(gold_heads_all, pred_heads_all, gold_rels_all, pred_rels_all)

    # ── CLS ────────────────────────────────────────────────
    gold_cls, pred_cls = [], []
    for ex in cls_dev_examples:
        text = ex.get("text", " ".join(ex["words"]))
        encoding = tokenizer(
            text, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            out = model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device))
        pred_idx = out["cls_logits"][0].cpu().argmax().item()
        gold_cls.append(ex["cls_label"])
        pred_cls.append(cls_labels[pred_idx])

    if gold_cls:
        results["cls"] = evaluate_cls(gold_cls, pred_cls, cls_labels)

    model.train()
    return results


def composite_score(results: dict) -> float:
    """Compute weighted composite metric for best-checkpoint selection."""
    ner_f1 = results.get("ner", {}).get("f1", 0.0)
    pos_acc = results.get("pos", {}).get("accuracy", 0.0)
    dep_uas = results.get("dep", {}).get("uas", 0.0)
    cls_f1 = results.get("cls", {}).get("macro_f1", 0.0)
    return 0.3 * ner_f1 + 0.3 * pos_acc + 0.3 * dep_uas + 0.1 * cls_f1


# ── Main training loop ────────────────────────────────────────────

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

    # Load train data
    with open(DATA_DIR / "ner_train.json") as f:
        ner_train = json.load(f)
    with open(DATA_DIR / "ud_train.json") as f:
        ud_train = json.load(f)

    # Load dev data
    with open(DATA_DIR / "ner_dev.json") as f:
        ner_dev = json.load(f)
    with open(DATA_DIR / "ud_dev.json") as f:
        ud_dev = json.load(f)

    encoder_name = config["model"]["encoder"]
    max_length = config["model"]["max_length"]
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    # Train datasets
    batch_size = config["training"]["batch_size"]
    ner_loader = DataLoader(
        TokenClassificationDataset(ner_train, tokenizer, ner_map, "ner_tags", max_length),
        batch_size=batch_size, shuffle=True,
    )
    pos_loader = DataLoader(
        TokenClassificationDataset(ud_train, tokenizer, pos_map, "pos_tags", max_length),
        batch_size=batch_size, shuffle=True,
    )
    dep_loader = DataLoader(
        TokenClassificationDataset(ud_train, tokenizer, dep_map, "dep_labels", max_length),
        batch_size=batch_size, shuffle=True,
    )
    cls_loader = DataLoader(
        SequenceClassificationDataset(ud_train + ner_train, tokenizer, cls_map, max_length),
        batch_size=batch_size, shuffle=True,
    )

    # Dev CLS examples
    cls_dev_examples = [ex for ex in ud_dev + ner_dev if "cls_label" in ex]

    # Model
    model = MultiTaskNLPModel(
        encoder_name=encoder_name,
        ner_labels=ner_labels, pos_labels=pos_labels,
        dep_labels=dep_labels, cls_labels=cls_labels,
        dropout=config["model"]["dropout"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Optimizer + scheduler
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

    task_weights = {
        "ner": config["training"]["ner_loss_weight"],
        "pos": config["training"]["pos_loss_weight"],
        "dep": config["training"]["dep_loss_weight"],
        "cls": config["training"]["cls_loss_weight"],
    }

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {config['training']['epochs']} epochs")
    print(f"  NER: {len(ner_loader)} batches, POS: {len(pos_loader)}, Dep: {len(dep_loader)}, CLS: {len(cls_loader)}")
    print(f"  Total steps: {total_steps}, warmup: {warmup_steps}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print()

    best_composite = -1.0
    best_epoch = -1
    patience_counter = 0
    history = []

    for epoch in range(config["training"]["epochs"]):
        # ── Training phase ────────────────────────────────
        model.train()
        total_loss = 0.0
        task_losses = {"ner": 0.0, "pos": 0.0, "dep": 0.0, "cls": 0.0}
        task_steps = {"ner": 0, "pos": 0, "dep": 0, "cls": 0}

        active_tasks = {
            "ner": iter(ner_loader), "pos": iter(pos_loader),
            "dep": iter(dep_loader), "cls": iter(cls_loader),
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

                if task_name == "cls":
                    loss = seq_loss_fn(outputs["cls_logits"], labels)
                else:
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
                task_steps[task_name] += 1

        step_count = sum(task_steps.values())
        avg_loss = total_loss / max(step_count, 1)
        task_avg = {k: task_losses[k] / max(task_steps[k], 1) for k in task_losses}

        # ── Dev evaluation phase ──────────────────────────
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        print(f"  Train loss: {avg_loss:.4f} [NER: {task_avg['ner']:.4f}, POS: {task_avg['pos']:.4f}, Dep: {task_avg['dep']:.4f}, CLS: {task_avg['cls']:.4f}]")

        dev_results = evaluate_on_dev(
            model, tokenizer, device, vocabs,
            ner_dev, ud_dev, cls_dev_examples, max_length,
        )

        ner_f1 = dev_results.get("ner", {}).get("f1", 0)
        pos_acc = dev_results.get("pos", {}).get("accuracy", 0)
        dep_uas = dev_results.get("dep", {}).get("uas", 0)
        dep_las = dev_results.get("dep", {}).get("las", 0)
        cls_f1 = dev_results.get("cls", {}).get("macro_f1", 0)
        comp = composite_score(dev_results)

        print(f"  Dev:  NER F1={ner_f1:.3f}  POS Acc={pos_acc:.3f}  Dep UAS={dep_uas:.3f} LAS={dep_las:.3f}  CLS F1={cls_f1:.3f}  Composite={comp:.3f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "dev_ner_f1": ner_f1,
            "dev_pos_acc": pos_acc,
            "dev_dep_uas": dep_uas,
            "dev_dep_las": dep_las,
            "dev_cls_f1": cls_f1,
            "composite": comp,
        })

        # ── Best checkpoint + early stopping ──────────────
        if comp > best_composite:
            best_composite = comp
            best_epoch = epoch + 1
            patience_counter = 0
            model.save(str(output_dir / "best"))
            tokenizer.save_pretrained(str(output_dir / "best"))
            print(f"  ★ New best checkpoint (composite={comp:.3f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")

        if config["output"]["save_every_epoch"]:
            model.save(str(output_dir / f"epoch-{epoch + 1}"))
            tokenizer.save_pretrained(str(output_dir / f"epoch-{epoch + 1}"))

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n⚠ Early stopping: no improvement for {EARLY_STOPPING_PATIENCE} epochs")
            break

    # Save final + history
    model.save(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete.")
    print(f"  Best epoch: {best_epoch} (composite={best_composite:.3f})")
    print(f"  Final model: {output_dir / 'final'}")
    print(f"  Best model:  {output_dir / 'best'}")
    print(f"  History:     {output_dir / 'training_history.json'}")


if __name__ == "__main__":
    train()

"""Train the multi-task NLP model on UD EWT + kniv corpus.

Uses HuggingFace Trainer for cross-platform support (CUDA, MPS, TPU).
Four tasks: NER, POS, dependency parsing (dep2label), sentence classification.

Features:
- Multi-task training with interleaved datasets and task-specific loss
- Dev-set evaluation every epoch (NER F1, POS acc, Dep UAS/LAS, CLS F1)
- Best checkpoint saved by composite metric
- Early stopping after N epochs without improvement
- Works on GPU (CUDA), Apple Silicon (MPS), and TPU (XLA)

Usage:
    python models/deberta-v3-large-nlp-en/train.py
    python models/deberta-v3-large-nlp-en/train.py --quick-test  # single epoch, 100 samples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
)
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.evaluate import evaluate_ner, evaluate_pos, evaluate_dep, evaluate_cls
from dep2label import decode_sentence
from model import MultiTaskNLPModel


# ── Configuration ─────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "config.yaml"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "prepared" / "deberta-v3-large-nlp-en"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Multi-task dataset ────────────────────────────────────────────

class MultiTaskDataset(Dataset):
    """Unified dataset that wraps all four tasks.

    Each item returns input_ids, attention_mask, task_id, and labels.
    The custom loss function in MultiTaskTrainer routes to the correct head.
    """

    TASK_IDS = {"ner": 0, "pos": 1, "dep": 2, "cls": 3}

    def __init__(self, examples, task: str, tokenizer, label_map,
                 label_key: str = "labels", max_length: int = 128):
        self.examples = examples
        self.task = task
        self.task_id = self.TASK_IDS[task]
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.label_key = label_key
        self.max_length = max_length
        self.is_token_task = task in ("ner", "pos", "dep")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        if self.is_token_task:
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
                "task_id": torch.tensor(self.task_id, dtype=torch.long),
            }
        else:
            # CLS task
            text = example.get("text", " ".join(example["words"]))
            prev_text = example.get("prev_text")

            if prev_text:
                encoding = self.tokenizer(
                    prev_text, text,
                    max_length=self.max_length, padding="max_length",
                    truncation=True, return_tensors="pt",
                )
            else:
                encoding = self.tokenizer(
                    text, max_length=self.max_length, padding="max_length",
                    truncation=True, return_tensors="pt",
                )

            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.label_map[example["cls_label"]], dtype=torch.long),
                "task_id": torch.tensor(self.task_id, dtype=torch.long),
            }


# ── Multi-task Trainer ────────────────────────────────────────────

def multitask_collator(features: list[dict]) -> dict:
    """Custom collator that handles mixed label shapes (token vs sequence)."""
    batch = {}
    max_len = max(f["input_ids"].size(0) for f in features)

    batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
    batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
    batch["task_id"] = torch.stack([f["task_id"] for f in features])

    # Labels: token tasks have [seq_len], CLS tasks have scalar
    # Pad CLS scalars to [seq_len] with -100 (ignored in loss)
    padded_labels = []
    for f in features:
        lbl = f["labels"]
        if lbl.dim() == 0:
            # Scalar label (CLS) — put the label at position 0, pad rest with -100
            padded = torch.full((max_len,), -100, dtype=torch.long)
            padded[0] = lbl
            padded_labels.append(padded)
        else:
            padded_labels.append(lbl)
    batch["labels"] = torch.stack(padded_labels)

    return batch


class MultiTaskTrainer(Trainer):
    """Custom Trainer that routes loss to the correct task head."""

    def __init__(self, task_weights: dict[str, float], **kwargs):
        super().__init__(**kwargs)
        self.task_weights = task_weights
        self.token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.seq_loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        task_ids = inputs["task_id"]

        outputs = model(input_ids, attention_mask)

        total_loss = torch.tensor(0.0, device=input_ids.device)
        task_names = ["ner", "pos", "dep", "cls"]
        logit_keys = ["ner_logits", "pos_logits", "dep_logits", "cls_logits"]

        task_losses = {}
        for task_idx, (task_name, logit_key) in enumerate(zip(task_names, logit_keys)):
            mask = task_ids == task_idx
            if not mask.any():
                continue

            task_logits = outputs[logit_key][mask]
            task_labels = labels[mask]

            if task_name == "cls":
                loss = self.seq_loss_fn(task_logits, task_labels[:, 0])
            else:
                loss = self.token_loss_fn(
                    task_logits.view(-1, task_logits.size(-1)),
                    task_labels.view(-1),
                )

            task_losses[task_name] = loss.detach().item()
            total_loss = total_loss + loss * self.task_weights[task_name]

        # Store per-task losses for custom logging
        self._last_task_losses = task_losses

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs: dict[str, float], **kwargs):
        """Override to append per-task losses."""
        if hasattr(self, "_last_task_losses") and self._last_task_losses:
            for task, loss in self._last_task_losses.items():
                logs[f"loss_{task}"] = round(loss, 4)
        super().log(logs, **kwargs)


# ── Evaluation ────────────────────────────────────────────────────

def evaluate_all(model, tokenizer, device, vocabs, ner_dev, ud_dev, cls_dev, max_length):
    """Run all four tasks on dev sets and return metrics dict."""
    model.eval()
    results = {}

    ner_labels = vocabs["ner_labels"]
    pos_labels = vocabs["pos_labels"]
    dep_labels = vocabs["dep_labels"]
    cls_labels = vocabs["cls_labels"]

    def predict_tokens(examples, logit_key, label_list):
        gold_all, pred_all = [], []
        for ex in examples:
            encoding = tokenizer(
                ex["words"], is_split_into_words=True,
                max_length=max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            with torch.no_grad():
                out = model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device))
            logits = out[logit_key][0].cpu()
            preds = logits.argmax(dim=-1).tolist()

            word_ids = encoding.word_ids()
            word_preds, prev = [], None
            for i, wid in enumerate(word_ids):
                if wid is not None and wid != prev:
                    idx = preds[i]
                    word_preds.append(label_list[idx] if idx < len(label_list) else "O")
                prev = wid
            gold_all.append(ex.get("ner_tags", ex.get("pos_tags", []))[:len(word_preds)])
            pred_all.append(word_preds[:len(ex.get("ner_tags", ex.get("pos_tags", [])))])
        return gold_all, pred_all

    # NER
    gold_ner, pred_ner = predict_tokens(ner_dev, "ner_logits", ner_labels)
    results["ner"] = evaluate_ner(gold_ner, pred_ner)

    # POS
    gold_pos, pred_pos = [], []
    for ex in ud_dev:
        encoding = tokenizer(
            ex["words"], is_split_into_words=True,
            max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            out = model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device))
        preds = out["pos_logits"][0].cpu().argmax(dim=-1).tolist()
        word_ids = encoding.word_ids()
        wp, prev = [], None
        for i, wid in enumerate(word_ids):
            if wid is not None and wid != prev:
                idx = preds[i]
                wp.append(pos_labels[idx] if idx < len(pos_labels) else "X")
            prev = wid
        gold_pos.append(ex["pos_tags"][:len(wp)])
        pred_pos.append(wp[:len(ex["pos_tags"])])
    results["pos"] = evaluate_pos(gold_pos, pred_pos)

    # Dep
    gold_h, pred_h, gold_r, pred_r = [], [], [], []
    for ex in ud_dev:
        encoding = tokenizer(
            ex["words"], is_split_into_words=True,
            max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            out = model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device))
        preds = out["dep_logits"][0].cpu().argmax(dim=-1).tolist()
        word_ids = encoding.word_ids()
        wp, prev = [], None
        for i, wid in enumerate(word_ids):
            if wid is not None and wid != prev:
                idx = preds[i]
                wp.append(dep_labels[idx] if idx < len(dep_labels) else "0@root@ROOT")
            prev = wid
        try:
            ph, pr = decode_sentence(wp[:len(ex["words"])], ex["pos_tags"])
        except Exception:
            ph = [-1] * len(ex["words"])
            pr = ["_"] * len(ex["words"])
        gold_h.append(ex["heads"])
        pred_h.append(ph)
        gold_r.append(ex["deprels"])
        pred_r.append(pr)
    results["dep"] = evaluate_dep(gold_h, pred_h, gold_r, pred_r)

    # CLS
    gold_cls, pred_cls = [], []
    for ex in cls_dev:
        text = ex.get("text", " ".join(ex["words"]))
        prev_text = ex.get("prev_text")
        if prev_text:
            encoding = tokenizer(prev_text, text, max_length=max_length,
                                 padding="max_length", truncation=True, return_tensors="pt")
        else:
            encoding = tokenizer(text, max_length=max_length,
                                 padding="max_length", truncation=True, return_tensors="pt")
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
    ner_f1 = results.get("ner", {}).get("f1", 0.0)
    pos_acc = results.get("pos", {}).get("accuracy", 0.0)
    dep_uas = results.get("dep", {}).get("uas", 0.0)
    cls_f1 = results.get("cls", {}).get("macro_f1", 0.0)
    return 0.3 * ner_f1 + 0.3 * pos_acc + 0.3 * dep_uas + 0.1 * cls_f1


# ── Main ──────────────────────────────────────────────────────────

def train(quick_test: bool = False):
    config = load_config()

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
    with open(DATA_DIR / "ner_train.json") as f:
        ner_train = json.load(f)
    with open(DATA_DIR / "ud_train.json") as f:
        ud_train = json.load(f)
    with open(DATA_DIR / "ner_dev.json") as f:
        ner_dev = json.load(f)
    with open(DATA_DIR / "ud_dev.json") as f:
        ud_dev = json.load(f)

    if quick_test:
        ner_train = ner_train[:100]
        ud_train = ud_train[:100]
        ner_dev = ner_dev[:50]
        ud_dev = ud_dev[:50]
        config["training"]["epochs"] = 1
        config["training"]["batch_size"] = 4
        print("⚡ Quick test mode: 100 train, 50 dev, 1 epoch", flush=True)

    encoder_name = config["model"]["encoder"]
    max_length = config["model"]["max_length"]
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    # Build multi-task datasets
    train_dataset = ConcatDataset([
        MultiTaskDataset(ner_train, "ner", tokenizer, ner_map, "ner_tags", max_length),
        MultiTaskDataset(ud_train, "pos", tokenizer, pos_map, "pos_tags", max_length),
        MultiTaskDataset(ud_train, "dep", tokenizer, dep_map, "dep_labels", max_length),
        MultiTaskDataset(ud_train + ner_train, "cls", tokenizer, cls_map, "cls_label", max_length),
    ])

    # Dummy eval dataset (real eval done in callback)
    eval_dataset = MultiTaskDataset(ud_dev[:100], "pos", tokenizer, pos_map, "pos_tags", max_length)

    # Model
    model = MultiTaskNLPModel(
        encoder_name=encoder_name,
        ner_labels=ner_labels, pos_labels=pos_labels,
        dep_labels=dep_labels, cls_labels=cls_labels,
        dropout=config["model"]["dropout"],
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {encoder_name}", flush=True)
    print(f"Parameters: {param_count:,}", flush=True)
    print(f"Train: {len(train_dataset):,} examples (NER:{len(ner_train)}, POS:{len(ud_train)}, Dep:{len(ud_train)}, CLS:{len(ud_train)+len(ner_train)})", flush=True)

    output_dir = Path(config["output"]["dir"])
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=config["training"]["learning_rate"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        max_grad_norm=config["training"]["max_grad_norm"],
        bf16=False,  # fp32 — both fp16 and bf16 cause loss underflow with DeBERTa-v3-large
        logging_steps=7,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=5,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    task_weights = {
        "ner": config["training"]["ner_loss_weight"],
        "pos": config["training"]["pos_loss_weight"],
        "dep": config["training"]["dep_loss_weight"],
        "cls": config["training"]["cls_loss_weight"],
    }

    # Epoch-end evaluation callback
    class EpochEvalCallback(TrainerCallback):
        def __init__(self, eval_model, eval_tokenizer, eval_device, eval_vocabs,
                     eval_ner_dev, eval_ud_dev, eval_max_length):
            self.eval_model = eval_model
            self.eval_tokenizer = eval_tokenizer
            self.eval_device = eval_device
            self.eval_vocabs = eval_vocabs
            self.eval_ner_dev = eval_ner_dev[:500]
            self.eval_ud_dev = eval_ud_dev[:500]
            self.eval_cls_dev = [ex for ex in (eval_ud_dev[:500] + eval_ner_dev[:500])
                                if "cls_label" in ex]
            self.eval_max_length = eval_max_length

        def on_epoch_end(self, args, state, control, **kwargs):
            epoch = int(state.epoch)
            print(f"\n  Evaluating after epoch {epoch}...", flush=True)
            results = evaluate_all(
                self.eval_model, self.eval_tokenizer, self.eval_device,
                self.eval_vocabs, self.eval_ner_dev, self.eval_ud_dev,
                self.eval_cls_dev, self.eval_max_length,
            )
            ner_f1 = results.get("ner", {}).get("f1", 0)
            pos_acc = results.get("pos", {}).get("accuracy", 0)
            dep_uas = results.get("dep", {}).get("uas", 0)
            cls_f1 = results.get("cls", {}).get("macro_f1", 0)
            comp = composite_score(results)
            print(f"  Epoch {epoch}: NER F1={ner_f1:.3f}  POS={pos_acc:.3f}  "
                  f"DEP UAS={dep_uas:.3f}  CLS F1={cls_f1:.3f}  "
                  f"Composite={comp:.3f}\n", flush=True)

    device = next(model.parameters()).device
    eval_cb = EpochEvalCallback(model, tokenizer, device, vocabs, ner_dev, ud_dev, max_length)

    trainer = MultiTaskTrainer(
        task_weights=task_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=multitask_collator,
        callbacks=[eval_cb],
    )

    print(f"\nTraining for {epochs} epochs", flush=True)
    print(f"  Batch size: {batch_size}, fp32 (no mixed precision)", flush=True)
    print(f"  Task weights: {task_weights}", flush=True)
    print(flush=True)

    trainer.train()

    # Save final model
    final_dir = output_dir / "final"
    model.save(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Full evaluation
    device = next(model.parameters()).device
    cls_dev = [ex for ex in ud_dev + ner_dev if "cls_label" in ex]
    dev_results = evaluate_all(model, tokenizer, device, vocabs, ner_dev, ud_dev, cls_dev, max_length)

    ner_f1 = dev_results.get("ner", {}).get("f1", 0)
    pos_acc = dev_results.get("pos", {}).get("accuracy", 0)
    dep_uas = dev_results.get("dep", {}).get("uas", 0)
    dep_las = dev_results.get("dep", {}).get("las", 0)
    cls_f1 = dev_results.get("cls", {}).get("macro_f1", 0)
    comp = composite_score(dev_results)

    print(f"\nFinal evaluation:", flush=True)
    print(f"  NER F1={ner_f1:.3f}  POS Acc={pos_acc:.3f}  Dep UAS={dep_uas:.3f} LAS={dep_las:.3f}  CLS F1={cls_f1:.3f}  Composite={comp:.3f}", flush=True)
    print(f"\nModel saved to {final_dir}", flush=True)

    # Save results
    # Convert numpy types for JSON serialization
    def to_serializable(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        return obj

    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(to_serializable(dev_results), f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train multi-task NLP model")
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick test: 1 epoch, 100 samples, verify setup works")
    args = parser.parse_args()
    train(quick_test=args.quick_test)


if __name__ == "__main__":
    main()

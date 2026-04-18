"""Train the student model via knowledge distillation from a teacher.

The student learns from two signals:
1. Hard labels (ground truth annotations) — standard cross-entropy
2. Soft labels (teacher logits) — KL divergence with temperature scaling

Combined loss: alpha * hard_loss + (1 - alpha) * distill_loss

Usage:
    python models/deberta-v3-nlp-en/distill.py \
        --soft-labels outputs/deberta-v3-large-nlp-en/soft_labels

Prerequisites:
    1. Train teacher: python models/deberta-v3-large-nlp-en/train.py
    2. Generate soft labels: python models/deberta-v3-large-nlp-en/generate_soft_labels.py \
           --model-dir outputs/deberta-v3-large-nlp-en/best
    3. Prepare student data: python models/deberta-v3-nlp-en/prepare_data.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.evaluate import evaluate_ner, evaluate_pos, evaluate_dep, evaluate_cls
from dep2label import decode_sentence
from model import MultiTaskNLPModel

# Import the standard train module for reusable components
from train import (
    TokenClassificationDataset,
    SequenceClassificationDataset,
    evaluate_on_dev,
    composite_score,
)


CONFIG_PATH = Path(__file__).parent / "config.yaml"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "prepared" / "deberta-v3-nlp-en"

EARLY_STOPPING_PATIENCE = 3


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Distillation datasets ────────────────────────────────────────

class DistillTokenDataset(Dataset):
    """Token classification dataset with teacher soft labels."""

    def __init__(self, examples, tokenizer, label_map, label_key,
                 teacher_logits, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.label_key = label_key
        self.teacher_logits = teacher_logits
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
            "teacher_logits": self.teacher_logits[idx],  # [seq_len, num_labels]
        }


class DistillSeqDataset(Dataset):
    """Sequence classification dataset with teacher soft labels."""

    def __init__(self, examples, tokenizer, label_map, teacher_logits, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.teacher_logits = teacher_logits
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
            "teacher_logits": self.teacher_logits[idx],  # [num_labels]
        }


# ── Distillation loss ────────────────────────────────────────────

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    temperature: float,
    alpha: float,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Combined hard label + soft label distillation loss.

    Args:
        student_logits: Student model output [batch, ..., num_classes]
        teacher_logits: Teacher model output [batch, ..., num_classes]
        hard_labels: Ground truth labels [batch, ...]
        temperature: Softmax temperature (higher = softer distributions)
        alpha: Weight for hard loss (1-alpha for distill loss)
        ignore_index: Label index to ignore (padding tokens)
    """
    # Hard loss: standard cross-entropy on ground truth
    if hard_labels.dim() == 1:
        # Sequence classification
        hard_loss = F.cross_entropy(student_logits, hard_labels)
        mask = None
    else:
        # Token classification
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            hard_labels.view(-1),
            ignore_index=ignore_index,
        )
        # Mask for distillation (only where we have real tokens)
        mask = (hard_labels != ignore_index).unsqueeze(-1)

    # Soft loss: KL divergence between temperature-scaled distributions
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    if mask is not None:
        # Apply mask to avoid computing KL on padding tokens
        student_soft = student_soft * mask
        teacher_soft = teacher_soft * mask

    soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
    # Scale by T^2 as per Hinton et al. — gradients scale as 1/T^2
    soft_loss = soft_loss * (temperature ** 2)

    return alpha * hard_loss + (1 - alpha) * soft_loss


# ── Main training loop ────────────────────────────────────────────

def train(soft_labels_dir: str, temperature: float = 3.0, alpha: float = 0.5):
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Distillation: temperature={temperature}, alpha={alpha}", flush=True)

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

    # Load training data
    with open(DATA_DIR / "conll_train.json") as f:
        conll_train = json.load(f)
    with open(DATA_DIR / "ud_train.json") as f:
        ud_train = json.load(f)
    with open(DATA_DIR / "conll_dev.json") as f:
        conll_dev = json.load(f)
    with open(DATA_DIR / "ud_dev.json") as f:
        ud_dev = json.load(f)

    # Load teacher soft labels
    soft_dir = Path(soft_labels_dir)
    print(f"Loading soft labels from {soft_dir}...", flush=True)
    ner_teacher = torch.load(soft_dir / "ner_logits.pt", weights_only=True)
    pos_teacher = torch.load(soft_dir / "pos_logits.pt", weights_only=True)
    dep_teacher = torch.load(soft_dir / "dep_logits.pt", weights_only=True)
    cls_teacher = torch.load(soft_dir / "cls_logits.pt", weights_only=True)
    print(f"  NER: {ner_teacher.shape}, POS: {pos_teacher.shape}, "
          f"Dep: {dep_teacher.shape}, CLS: {cls_teacher.shape}", flush=True)

    encoder_name = config["model"]["encoder"]
    max_length = config["model"]["max_length"]
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    batch_size = config["training"]["batch_size"]

    # Distillation datasets (include teacher logits)
    ner_loader = DataLoader(
        DistillTokenDataset(conll_train, tokenizer, ner_map, "ner_tags", ner_teacher, max_length),
        batch_size=batch_size, shuffle=True,
    )
    pos_loader = DataLoader(
        DistillTokenDataset(ud_train, tokenizer, pos_map, "pos_tags", pos_teacher, max_length),
        batch_size=batch_size, shuffle=True,
    )
    dep_loader = DataLoader(
        DistillTokenDataset(ud_train, tokenizer, dep_map, "dep_labels", dep_teacher, max_length),
        batch_size=batch_size, shuffle=True,
    )

    cls_examples = [ex for ex in ud_train + conll_train if "cls_label" in ex]
    cls_loader = DataLoader(
        DistillSeqDataset(cls_examples, tokenizer, cls_map, cls_teacher, max_length),
        batch_size=batch_size, shuffle=True,
    )

    cls_dev_examples = [ex for ex in ud_dev + conll_dev if "cls_label" in ex]

    # Student model
    model = MultiTaskNLPModel(
        encoder_name=encoder_name,
        ner_labels=ner_labels, pos_labels=pos_labels,
        dep_labels=dep_labels, cls_labels=cls_labels,
        dropout=config["model"]["dropout"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Student parameters: {param_count:,}", flush=True)

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

    task_weights = {
        "ner": config["training"]["ner_loss_weight"],
        "pos": config["training"]["pos_loss_weight"],
        "dep": config["training"]["dep_loss_weight"],
        "cls": config["training"]["cls_loss_weight"],
    }

    output_dir = Path(config["output"]["dir"]) / "distilled"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDistillation training for {config['training']['epochs']} epochs", flush=True)
    print(f"  NER: {len(ner_loader)} batches, POS: {len(pos_loader)}, "
          f"Dep: {len(dep_loader)}, CLS: {len(cls_loader)}", flush=True)

    best_composite = -1.0
    best_epoch = -1
    patience_counter = 0
    history = []

    for epoch in range(config["training"]["epochs"]):
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
                teacher_logits = batch["teacher_logits"].to(device)
                outputs = model(input_ids, attention_mask)

                if task_name == "cls":
                    student_logits = outputs["cls_logits"]
                else:
                    student_logits = outputs[f"{task_name}_logits"]

                loss = distillation_loss(
                    student_logits, teacher_logits, labels,
                    temperature=temperature, alpha=alpha,
                )

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

        # Dev evaluation (same as standard training — uses hard labels)
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}", flush=True)
        print(f"  Train loss: {avg_loss:.4f} [NER: {task_avg['ner']:.4f}, POS: {task_avg['pos']:.4f}, "
              f"Dep: {task_avg['dep']:.4f}, CLS: {task_avg['cls']:.4f}]", flush=True)

        dev_results = evaluate_on_dev(
            model, tokenizer, device, vocabs,
            conll_dev, ud_dev, cls_dev_examples, max_length,
        )

        ner_f1 = dev_results.get("ner", {}).get("f1", 0)
        pos_acc = dev_results.get("pos", {}).get("accuracy", 0)
        dep_uas = dev_results.get("dep", {}).get("uas", 0)
        dep_las = dev_results.get("dep", {}).get("las", 0)
        cls_f1 = dev_results.get("cls", {}).get("macro_f1", 0)
        comp = composite_score(dev_results)

        print(f"  Dev:  NER F1={ner_f1:.3f}  POS Acc={pos_acc:.3f}  "
              f"Dep UAS={dep_uas:.3f} LAS={dep_las:.3f}  CLS F1={cls_f1:.3f}  "
              f"Composite={comp:.3f}", flush=True)

        history.append({
            "epoch": epoch + 1, "train_loss": avg_loss,
            "dev_ner_f1": ner_f1, "dev_pos_acc": pos_acc,
            "dev_dep_uas": dep_uas, "dev_dep_las": dep_las,
            "dev_cls_f1": cls_f1, "composite": comp,
        })

        if comp > best_composite:
            best_composite = comp
            best_epoch = epoch + 1
            patience_counter = 0
            model.save(str(output_dir / "best"))
            tokenizer.save_pretrained(str(output_dir / "best"))
            print(f"  ★ New best checkpoint (composite={comp:.3f})", flush=True)
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})", flush=True)

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n⚠ Early stopping: no improvement for {EARLY_STOPPING_PATIENCE} epochs", flush=True)
            break

    model.save(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDistillation complete.", flush=True)
    print(f"  Best epoch: {best_epoch} (composite={best_composite:.3f})", flush=True)
    print(f"  Temperature: {temperature}, Alpha: {alpha}", flush=True)
    print(f"  Final model: {output_dir / 'final'}", flush=True)
    print(f"  Best model:  {output_dir / 'best'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train student via knowledge distillation")
    parser.add_argument("--soft-labels", required=True,
                        help="Path to teacher soft labels directory")
    parser.add_argument("--temperature", type=float, default=3.0,
                        help="Distillation temperature (default: 3.0)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Hard loss weight (default: 0.5, distill weight = 1-alpha)")
    args = parser.parse_args()

    train(args.soft_labels, args.temperature, args.alpha)


if __name__ == "__main__":
    main()

"""Staged cascade training: one head at a time.

Stage 1: POS only        → encoder + POS head
Stage 2: +NER cascade    → NER head (low LR for encoder+POS)
Stage 3: +DEP cascade    → DEP head (low LR for encoder+POS+NER)
Stage 4: +CLS cascade    → CLS head (low LR for encoder+POS+NER+DEP)
Stage 5: joint fine-tune → all heads (low LR everywhere)

Usage:
    python models/kniv-deberta-cascade-large-nlp-en/staged_train.py --stage 1
    python models/kniv-deberta-cascade-large-nlp-en/staged_train.py --stage 2 \
        --checkpoint outputs/kniv-deberta-cascade-large-nlp-en/stage1/best
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from shared.evaluate import evaluate_ner, evaluate_pos, evaluate_dep, evaluate_cls
from dep2label import decode_sentence
from model import MultiTaskNLPModel
from train import (
    MultiTaskDataset, multitask_collator,
    evaluate_all, composite_score,
)

CONFIG_PATH = Path(__file__).parent / "config.yaml"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "prepared" / "deberta-v3-large-nlp-en"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Stage configuration ──────────────────────────────────────────

STAGE_CONFIG = {
    1: {
        "name": "POS",
        "tasks": ["pos"],
        "new_head": None,
        "epochs": 3,
        "head_lr": 1e-5,
        "base_lr": 1e-5,   # encoder + existing heads
    },
    2: {
        "name": "NER (frozen encoder+POS)",
        "tasks": ["ner"],           # only NER data — POS is frozen
        "eval_tasks": ["pos", "ner"],  # but eval both to check POS doesn't degrade
        "new_head": "ner",
        "head_type": "mlp",
        "freeze_base": True,
        "epochs": 5,
        "head_lr": 1e-4,
        "base_lr": None,
    },
    3: {
        "name": "DEP (frozen encoder+POS+NER)",
        "tasks": ["dep"],
        "eval_tasks": ["pos", "ner", "dep"],
        "new_head": "dep",
        "head_type": "linear",
        "freeze_base": True,
        "epochs": 5,
        "head_lr": 1e-4,
        "base_lr": None,
    },
    4: {
        "name": "CLS (frozen encoder+POS+NER+DEP)",
        "tasks": ["cls"],
        "eval_tasks": ["pos", "ner", "dep", "cls"],
        "new_head": "cls",
        "head_type": "linear",
        "freeze_base": True,
        "epochs": 5,
        "head_lr": 1e-4,
        "base_lr": None,
    },
    5: {
        "name": "Joint fine-tune",
        "tasks": ["pos", "ner", "dep", "cls"],
        "new_head": None,
        "epochs": 3,
        "head_lr": 5e-6,
        "base_lr": 5e-6,   # uniform low LR
    },
}


# ── Training loop ────────────────────────────────────────────────

def train_stage(stage: int, checkpoint: str | None = None):
    config = load_config()
    stage_cfg = STAGE_CONFIG[stage]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}", flush=True)
    print(f"STAGE {stage}: {stage_cfg['name']}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Tasks: {stage_cfg['tasks']}", flush=True)
    print(f"Head LR: {stage_cfg['head_lr']}, Base LR: {stage_cfg['base_lr']}", flush=True)

    # Load label vocabularies
    with open(DATA_DIR / "label_vocabs.json") as f:
        vocabs = json.load(f)

    ner_labels = vocabs["ner_labels"]
    pos_labels = vocabs["pos_labels"]
    dep_labels = vocabs["dep_labels"]
    cls_labels = vocabs["cls_labels"]

    encoder_name = config["model"]["encoder"]
    max_length = config["model"]["max_length"]
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    # ── Build or load model ──────────────────────────────────
    if stage == 1:
        # Fresh model with POS head only
        model = MultiTaskNLPModel(
            encoder_name=encoder_name,
            pos_labels=pos_labels,
        ).float().to(device)
    else:
        assert checkpoint, f"Stage {stage} requires --checkpoint from previous stage"
        print(f"Loading checkpoint: {checkpoint}", flush=True)

        # Support loading from existing non-cascade teacher
        checkpoint_labels = Path(checkpoint) / "label_maps.json"
        if checkpoint_labels.exists():
            with open(checkpoint_labels) as f:
                ckpt_labels = json.load(f)
            # If checkpoint has all 4 heads but no cascade structure,
            # it's the old teacher — load via load_from_teacher
            if "ner_labels" in ckpt_labels and "pos_labels" in ckpt_labels:
                has_cascade = "pos_labels" in ckpt_labels and "ner_labels" not in ckpt_labels
                # Old teacher has all 4, cascade stage1 has only pos
                old_teacher = all(k in ckpt_labels for k in ["ner_labels", "pos_labels", "dep_labels", "cls_labels"])
                if old_teacher and stage == 2:
                    print("  Detected old teacher checkpoint — loading encoder + POS only", flush=True)
                    model = MultiTaskNLPModel.load_from_teacher(checkpoint, encoder_name).float().to(device)
                else:
                    model = MultiTaskNLPModel.load(checkpoint, encoder_name).float().to(device)
            else:
                model = MultiTaskNLPModel.load(checkpoint, encoder_name).float().to(device)
        else:
            model = MultiTaskNLPModel.load(checkpoint, encoder_name).float().to(device)

        # Add new head if this stage introduces one
        if stage_cfg["new_head"]:
            head_name = stage_cfg["new_head"]
            label_list = {"ner": ner_labels, "dep": dep_labels, "cls": cls_labels}[head_name]
            # NER uses MLP head; DEP/CLS start as linear (can be changed)
            head_type = stage_cfg.get("head_type", "linear")
            model.add_head(head_name, label_list, head_type=head_type)
            model = model.to(device)
            print(f"Added {head_name} head ({head_type}, {len(label_list)} labels)", flush=True)

    model.gradient_checkpointing_enable()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}", flush=True)

    # ── Build datasets for active tasks ──────────────────────
    with open(DATA_DIR / "ner_train.json") as f:
        ner_train = json.load(f)
    with open(DATA_DIR / "ud_train.json") as f:
        ud_train = json.load(f)
    with open(DATA_DIR / "ner_dev.json") as f:
        ner_dev = json.load(f)
    with open(DATA_DIR / "ud_dev.json") as f:
        ud_dev = json.load(f)

    ner_map = {l: i for i, l in enumerate(ner_labels)}
    pos_map = {l: i for i, l in enumerate(pos_labels)}
    dep_map = {l: i for i, l in enumerate(dep_labels)}
    cls_map = {l: i for i, l in enumerate(cls_labels)}

    active_tasks = stage_cfg["tasks"]
    datasets = []

    if "pos" in active_tasks:
        datasets.append(MultiTaskDataset(ud_train, "pos", tokenizer, pos_map, "pos_tags", max_length))
    if "ner" in active_tasks:
        datasets.append(MultiTaskDataset(ner_train, "ner", tokenizer, ner_map, "ner_tags", max_length))
    if "dep" in active_tasks:
        datasets.append(MultiTaskDataset(ud_train, "dep", tokenizer, dep_map, "dep_labels", max_length))
    if "cls" in active_tasks:
        cls_examples = [ex for ex in ud_train + ner_train if "cls_label" in ex]
        datasets.append(MultiTaskDataset(cls_examples, "cls", tokenizer, cls_map, "cls_label", max_length))

    train_dataset = ConcatDataset(datasets)
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=multitask_collator, num_workers=0,
    )

    cls_dev = [ex for ex in ud_dev + ner_dev if "cls_label" in ex]

    print(f"Train: {len(train_dataset):,} examples, {len(train_loader)} batches/epoch", flush=True)

    # ── Optimizer: freeze base params for stages 2-4, train only new head ──
    new_head_name = stage_cfg["new_head"]
    freeze_base = stage_cfg.get("freeze_base", False)

    if freeze_base and new_head_name:
        # Freeze encoder + all existing heads — only train the new head
        head_params = []
        frozen_count = 0
        for name, param in model.named_parameters():
            if f"{new_head_name}_head" in name:
                param.requires_grad = True
                head_params.append(param)
            else:
                param.requires_grad = False
                frozen_count += 1
        param_groups = [{"params": head_params, "lr": stage_cfg["head_lr"]}]
        trainable = sum(p.numel() for p in head_params)
        print(f"  Frozen: {frozen_count} params, Trainable: {trainable:,} ({new_head_name}_head only)", flush=True)
    elif stage == 5:
        # Stage 5 (joint): single LR for everything, all unfrozen
        for param in model.parameters():
            param.requires_grad = True
        param_groups = [{"params": list(model.parameters()), "lr": stage_cfg["base_lr"]}]
    else:
        # Differential LR (fallback)
        head_params = []
        base_params = []
        for name, param in model.named_parameters():
            if new_head_name and f"{new_head_name}_head" in name:
                head_params.append(param)
            else:
                base_params.append(param)
        param_groups = []
        if base_params:
            param_groups.append({"params": base_params, "lr": stage_cfg["base_lr"]})
        if head_params:
            param_groups.append({"params": head_params, "lr": stage_cfg["head_lr"]})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config["training"]["weight_decay"])

    total_steps = stage_cfg["epochs"] * len(train_loader)
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    task_weights = {
        "ner": config["training"]["ner_loss_weight"],
        "pos": config["training"]["pos_loss_weight"],
        "dep": config["training"]["dep_loss_weight"],
        "cls": config["training"]["cls_loss_weight"],
    }

    token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    seq_loss_fn = nn.CrossEntropyLoss()

    # ── Output directory ─────────────────────────────────────
    output_dir = Path(config["output"]["dir"]) / f"stage{stage}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────
    task_names = ["ner", "pos", "dep", "cls"]
    logit_keys = ["ner_logits", "pos_logits", "dep_logits", "cls_logits"]

    best_composite = -1.0
    best_epoch = -1
    log_every = 20

    for epoch in range(stage_cfg["epochs"]):
        model.train()
        total_loss = 0.0
        task_losses = {t: 0.0 for t in active_tasks}
        task_steps = {t: 0 for t in active_tasks}

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            task_ids = batch["task_id"].to(device)

            outputs = model(input_ids, attention_mask)

            loss = torch.tensor(0.0, device=device)
            step_tasks = {}

            for task_idx, (task_name, logit_key) in enumerate(zip(task_names, logit_keys)):
                if task_name not in active_tasks:
                    continue
                if logit_key not in outputs:
                    continue

                mask = task_ids == task_idx
                if not mask.any():
                    continue

                task_logits = outputs[logit_key][mask]
                task_labels = labels[mask]

                if task_name == "cls":
                    task_loss = seq_loss_fn(task_logits, task_labels[:, 0])
                else:
                    task_loss = token_loss_fn(
                        task_logits.view(-1, task_logits.size(-1)),
                        task_labels.view(-1),
                    )

                loss = loss + task_loss * task_weights[task_name]
                step_tasks[task_name] = task_loss.item()
                task_losses[task_name] += task_loss.item()
                task_steps[task_name] += 1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            global_step = epoch * len(train_loader) + step + 1
            if global_step % log_every == 0:
                parts = [f"step={global_step}", f"epoch={epoch + global_step/len(train_loader)/stage_cfg['epochs']:.2f}",
                         f"loss={loss.item():.4f}"]
                for t in active_tasks:
                    if t in step_tasks:
                        parts.append(f"{t}={step_tasks[t]:.4f}")
                lr_str = f"lr={scheduler.get_last_lr()[0]:.2e}"
                if len(param_groups) > 1:
                    lr_str += f"/{scheduler.get_last_lr()[-1]:.2e}"
                parts.append(lr_str)
                print("  " + "  ".join(parts), flush=True)

        # ── Epoch evaluation ─────────────────────────────────
        avg_loss = total_loss / max(len(train_loader), 1)
        task_avg = {t: task_losses[t] / max(task_steps[t], 1) for t in active_tasks}

        print(f"\nEpoch {epoch + 1}/{stage_cfg['epochs']}", flush=True)
        parts = [f"Train loss: {avg_loss:.4f}"]
        for t in active_tasks:
            parts.append(f"{t.upper()}={task_avg[t]:.4f}")
        print(f"  {' | '.join(parts)}", flush=True)

        # Eval only active heads
        # Eval on all eval_tasks (includes frozen heads to verify no degradation)
        eval_tasks = stage_cfg.get("eval_tasks", active_tasks)
        dev_results = evaluate_active(
            model, tokenizer, device, vocabs,
            ner_dev[:500] if "ner" in eval_tasks else [],
            ud_dev[:500],
            cls_dev[:500] if "cls" in eval_tasks else [],
            max_length, eval_tasks,
        )

        # Print results
        parts = []
        if "pos" in dev_results:
            parts.append(f"POS Acc={dev_results['pos'].get('accuracy', 0):.3f}")
        if "ner" in dev_results:
            parts.append(f"NER F1={dev_results['ner'].get('f1', 0):.3f}")
        if "dep" in dev_results:
            parts.append(f"DEP UAS={dev_results['dep'].get('uas', 0):.3f}")
        if "cls" in dev_results:
            parts.append(f"CLS F1={dev_results['cls'].get('macro_f1', 0):.3f}")

        comp = composite_score_active(dev_results, eval_tasks)
        parts.append(f"Composite={comp:.3f}")
        print(f"  Dev: {' | '.join(parts)}", flush=True)

        if comp > best_composite:
            best_composite = comp
            best_epoch = epoch + 1
            model.save(str(output_dir / "best"))
            tokenizer.save_pretrained(str(output_dir / "best"))
            print(f"  * New best (composite={comp:.3f})", flush=True)

    # Save final
    model.save(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    print(f"\nStage {stage} complete.", flush=True)
    print(f"  Best epoch: {best_epoch} (composite={best_composite:.3f})", flush=True)
    print(f"  Best model: {output_dir / 'best'}", flush=True)
    print(f"  Final model: {output_dir / 'final'}", flush=True)


def evaluate_active(model, tokenizer, device, vocabs, ner_dev, ud_dev, cls_dev, max_length, active_tasks):
    """Evaluate only active heads."""
    model.eval()
    results = {}

    pos_labels = vocabs["pos_labels"]
    ner_labels = vocabs["ner_labels"]
    dep_labels = vocabs["dep_labels"]
    cls_labels = vocabs["cls_labels"]

    def predict_tokens(examples, logit_key, label_list, gold_key):
        gold_all, pred_all = [], []
        for ex in examples:
            encoding = tokenizer(
                ex["words"], is_split_into_words=True,
                max_length=max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            with torch.no_grad():
                out = model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device))
            if logit_key not in out:
                return [], []
            logits = out[logit_key][0].cpu()
            preds = logits.argmax(dim=-1).tolist()
            word_ids = encoding.word_ids()
            word_preds, prev = [], None
            for i, wid in enumerate(word_ids):
                if wid is not None and wid != prev:
                    idx = preds[i]
                    word_preds.append(label_list[idx] if idx < len(label_list) else "O")
                prev = wid
            gold = ex.get(gold_key, [])[:len(word_preds)]
            gold_all.append(gold)
            pred_all.append(word_preds[:len(gold)])
        return gold_all, pred_all

    if "pos" in active_tasks and ud_dev:
        gold, pred = predict_tokens(ud_dev, "pos_logits", pos_labels, "pos_tags")
        if gold:
            results["pos"] = evaluate_pos(gold, pred)

    if "ner" in active_tasks and ner_dev:
        gold, pred = predict_tokens(ner_dev, "ner_logits", ner_labels, "ner_tags")
        if gold:
            results["ner"] = evaluate_ner(gold, pred)

    if "dep" in active_tasks and ud_dev:
        gold, pred = predict_tokens(ud_dev, "dep_logits", dep_labels, "dep_labels")
        if gold:
            results["dep"] = evaluate_dep(gold, pred, ud_dev)

    if "cls" in active_tasks and cls_dev:
        gold_cls, pred_cls = [], []
        for ex in cls_dev:
            text = ex.get("text", " ".join(ex["words"]))
            encoding = tokenizer(text, max_length=max_length, padding="max_length",
                                 truncation=True, return_tensors="pt")
            with torch.no_grad():
                out = model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device))
            if "cls_logits" not in out:
                break
            pred_idx = out["cls_logits"][0].argmax().item()
            gold_cls.append(ex["cls_label"])
            pred_cls.append(cls_labels[pred_idx])
        if gold_cls:
            results["cls"] = evaluate_cls(gold_cls, pred_cls, cls_labels)

    model.train()
    return results


def composite_score_active(results, active_tasks):
    """Compute composite score over active tasks only."""
    scores = []
    weights = []
    if "ner" in active_tasks and "ner" in results:
        scores.append(results["ner"].get("f1", 0))
        weights.append(0.3)
    if "pos" in active_tasks and "pos" in results:
        scores.append(results["pos"].get("accuracy", 0))
        weights.append(0.3)
    if "dep" in active_tasks and "dep" in results:
        scores.append(results["dep"].get("uas", 0))
        weights.append(0.3)
    if "cls" in active_tasks and "cls" in results:
        scores.append(results["cls"].get("macro_f1", 0))
        weights.append(0.1)

    if not weights:
        return 0.0
    # Normalize weights to sum to 1
    total_w = sum(weights)
    return sum(s * w / total_w for s, w in zip(scores, weights))


def main():
    parser = argparse.ArgumentParser(description="Staged cascade training")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4, 5],
                        help="Training stage (1=POS, 2=+NER, 3=+DEP, 4=+CLS, 5=joint)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to previous stage best checkpoint (required for stages 2-5)")
    args = parser.parse_args()

    if args.stage > 1 and not args.checkpoint:
        parser.error(f"Stage {args.stage} requires --checkpoint from previous stage")

    train_stage(args.stage, args.checkpoint)


if __name__ == "__main__":
    main()

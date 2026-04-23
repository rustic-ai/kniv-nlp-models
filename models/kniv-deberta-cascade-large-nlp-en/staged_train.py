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
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "prepared" / "kniv-deberta-cascade"


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
        "base_lr": 1e-5,
    },
    # Stage 2 splits into 2a/2b/2c for NER cascade training
    "2a": {
        "name": "NER head (frozen encoder+POS, Few-NERD 45K)",
        "tasks": ["ner"],
        "data_file": "fewnerd_train.json",   # 45K Few-NERD subsample
        "eval_tasks": ["pos", "ner"],
        "new_head": "ner",
        "head_type": "mlp",
        "freeze_base": True,
        "epochs": 5,
        "head_lr": 1e-4,
        "base_lr": None,
    },
    "2b": {
        "name": "POS+NER alignment (frozen encoder, UD EWT 12.5K)",
        "tasks": ["pos", "ner"],
        "data_file": "ud_ner_train.json",    # UD EWT with expert POS + spaCy NER
        "eval_tasks": ["pos", "ner"],
        "new_head": None,
        "freeze_base": True,
        "epochs": 1,
        "head_lr": 1e-4,                    # both heads at same LR
        "base_lr": None,
    },
    "2c": {
        "name": "Encoder fine-tune (unfrozen, POS+NER 163K) — DEPRECATED, degraded POS+NER",
        "tasks": ["pos", "ner"],
        "data_file": "posner_train.json",
        "eval_tasks": ["pos", "ner"],
        "new_head": None,
        "freeze_base": False,
        "epochs": 1,
        "head_lr": 1e-5,
        "base_lr": 1e-6,
    },
    "2d": {
        "name": "Gentle encoder fine-tune (UD EWT 12.5K + Few-NERD 20K, 3 epochs) — DEPRECATED",
        "tasks": ["pos", "ner"],
        "data_file": "posner_small_train.json",
        "eval_tasks": ["pos", "ner"],
        "new_head": None,
        "freeze_base": False,
        "epochs": 3,
        "head_lr": 1e-5,
        "pos_lr": 1e-7,
        "base_lr": 1e-7,
    },
    "2e": {
        "name": "NER on SpanMarker 195K (frozen encoder+POS)",
        "tasks": ["ner"],
        "data_file": "ner_spanmarker_train.json",
        "dev_file": "ner_spanmarker_dev.json",
        "eval_tasks": ["pos", "ner"],
        "new_head": "ner",
        "head_type": "mlp",
        "freeze_base": True,
        "epochs": 5,
        "head_lr": 1e-4,
        "base_lr": None,
    },
    "2f": {
        "name": "Encoder tune (UD EWT SpanMarker, POS+NER, 2 epochs)",
        "tasks": ["pos", "ner"],
        "data_file": "ud_spanmarker_train.json",
        "dev_file": "ner_spanmarker_dev.json",
        "eval_tasks": ["pos", "ner"],
        "new_head": None,
        "freeze_base": False,
        "epochs": 2,
        "head_lr": 1e-6,                    # NER head: small adjustment
        "pos_lr": 1e-7,                     # POS head: barely moves
        "base_lr": 1e-7,                    # encoder: barely moves
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
    "3s": {
        "name": "SRL Linear (frozen) — DEPRECATED, F1=0.263",
        "tasks": ["srl"],
        "data_file": "srl_train.json",
        "eval_tasks": ["pos", "ner", "dep", "srl"],
        "new_head": "srl",
        "head_type": "linear",
        "freeze_base": True,
        "epochs": 5,
        "head_lr": 1e-4,
        "base_lr": None,
    },
    "3m": {
        "name": "SRL MLP (frozen) — DEPRECATED, F1=0.337",
        "tasks": ["srl"],
        "data_file": "srl_full_train.json",
        "eval_tasks": ["pos", "ner", "dep", "srl"],
        "new_head": "srl",
        "head_type": "mlp",
        "freeze_base": True,
        "epochs": 5,
        "head_lr": 1e-4,
        "base_lr": None,
    },
    "3b": {
        "name": "SRL Biaffine (frozen encoder+POS+NER+DEP)",
        "tasks": ["srl"],
        "data_file": "srl_full_train.json",
        "eval_tasks": ["pos", "ner", "dep", "srl"],
        "new_head": "srl",
        "head_type": "biaffine",
        "freeze_base": True,
        "epochs": 10,
        "patience": 3,
        "batch_size": 128,
        "head_lr": 1e-4,
        "base_lr": None,
    },
    "3b-512": {
        "name": "SRL Biaffine 512d: encoder+DEP only",
        "tasks": ["srl"],
        "data_file": "srl_full_train.json",
        "eval_tasks": ["pos", "ner", "dep", "srl"],
        "new_head": "srl",
        "head_type": "biaffine",
        "srl_cascade": ["dep"],
        "biaffine_dim": 512,
        "freeze_base": True,
        "epochs": 10,
        "patience": 3,
        "batch_size": 128,
        "head_lr": 1e-4,
        "base_lr": None,
    },
    "3b-dep": {
        "name": "SRL Biaffine: encoder+DEP only, 20 epochs",
        "tasks": ["srl"],
        "data_file": "srl_full_train.json",
        "eval_tasks": ["pos", "ner", "dep", "srl"],
        "new_head": "srl",
        "head_type": "biaffine",
        "srl_cascade": ["dep"],
        "freeze_base": True,
        "epochs": 20,
        "patience": 5,
        "batch_size": 128,
        "head_lr": 1e-4,
        "base_lr": None,
    },
    "3c": {
        "name": "SRL pred-aware MLP + encoder tune (998K silver)",
        "tasks": ["srl"],
        "data_file": "srl_silver_dannashao.json",
        "eval_tasks": ["pos", "ner", "dep", "srl"],
        "new_head": "srl",
        "head_type": "pred_mlp",
        "srl_cascade": ["dep"],
        "freeze_base": False,
        "epochs": 3,
        "patience": 2,
        "batch_size": 32,
        "head_lr": 1e-4,
        "base_lr": 1e-6,
    },
    "3c-mt": {
        "name": "SRL pred-aware MLP + encoder tune + DEP/POS regularization",
        "tasks": ["srl", "dep", "pos"],
        "data_file": "srl_silver_dannashao.json",
        "max_primary_examples": 200000,
        "eval_tasks": ["pos", "ner", "dep", "srl"],
        "new_head": "srl",
        "head_type": "pred_mlp",
        "srl_cascade": ["dep"],
        "freeze_base": False,
        "epochs": 3,
        "patience": 2,
        "batch_size": 32,
        "head_lr": 1e-4,
        "dep_lr": 1e-5,
        "pos_lr": 1e-6,
        "base_lr": 1e-6,
    },
    "3c-adapt": {
        "name": "SRL adapter (predicate-conditioned attention) + MLP",
        "tasks": ["srl"],
        "data_file": "srl_silver_dannashao.json",
        "max_primary_examples": 200000,
        "eval_tasks": ["pos", "ner", "dep", "srl"],
        "new_head": "srl",
        "head_type": "adapter",
        "srl_cascade": ["dep"],
        "freeze_base": True,
        "epochs": 10,
        "patience": 3,
        "batch_size": 128,
        "head_lr": 1e-4,
        "base_lr": None,
    },
    "3c-lora": {
        "name": "SRL pred-aware MLP + LoRA encoder (200K silver)",
        "tasks": ["srl"],
        "data_file": "srl_silver_dannashao.json",
        "max_primary_examples": 200000,
        "eval_tasks": ["pos", "ner", "dep", "srl"],
        "new_head": "srl",
        "head_type": "pred_mlp",
        "srl_cascade": ["dep"],
        "freeze_base": True,
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "epochs": 5,
        "patience": 3,
        "batch_size": 128,
        "head_lr": 1e-4,
        "base_lr": None,
    },
    4: {
        "name": "CLS (frozen encoder+POS+NER+DEP+SRL)",
        "tasks": ["cls"],
        "data_file": "cls_swda_mrda_train.json",
        "eval_tasks": ["pos", "ner", "dep", "srl", "cls"],
        "new_head": "cls",
        "head_type": "linear",
        "freeze_base": True,
        "epochs": 5,
        "head_lr": 1e-4,
        "base_lr": None,
    },
    5: {
        "name": "Joint fine-tune",
        "tasks": ["pos", "ner", "dep", "srl", "cls"],
        "new_head": None,
        "epochs": 3,
        "head_lr": 5e-6,
        "base_lr": 5e-6,
    },
}


# ── Training loop ────────────────────────────────────────────────

def train_stage(stage: str | int, checkpoint: str | None = None):
    config = load_config()
    # Support both int (1, 3, 4, 5) and string ("2a", "2b", "2c") stages
    stage_key = int(stage) if str(stage).isdigit() else stage
    stage_cfg = STAGE_CONFIG[stage_key]
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
    srl_labels = vocabs.get("srl_labels", [])
    cls_labels = vocabs["cls_labels"]

    encoder_name = config["model"]["encoder"]
    max_length = config["model"]["max_length"]
    # Load tokenizer from checkpoint (not HF hub) — the teacher's saved
    # tokenizer includes BOS/EOS tokens that the HF default may omit
    if checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    else:
        tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    # ── Build or load model ──────────────────────────────────
    if stage_key == 1:
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
                if old_teacher and str(stage).startswith("2"):
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
            label_list = {"ner": ner_labels, "dep": dep_labels, "srl": srl_labels, "cls": cls_labels}[head_name]
            # NER uses MLP head; DEP/CLS start as linear (can be changed)
            head_type = stage_cfg.get("head_type", "linear")
            srl_cascade = stage_cfg.get("srl_cascade")
            biaffine_dim = stage_cfg.get("biaffine_dim")
            model.add_head(head_name, label_list, head_type=head_type,
                           srl_cascade=srl_cascade, biaffine_dim=biaffine_dim)
            model = model.to(device)
            print(f"Added {head_name} head ({head_type}, {len(label_list)} labels)", flush=True)

    # LoRA: wrap encoder with low-rank adapters (original weights frozen)
    if stage_cfg.get("use_lora"):
        from peft import get_peft_model, LoraConfig
        lora_r = stage_cfg.get("lora_r", 16)
        lora_alpha = stage_cfg.get("lora_alpha", 32)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query_proj", "key_proj", "value_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        model.encoder = get_peft_model(model.encoder, lora_config)
        lora_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        print(f"  LoRA applied: r={lora_r}, alpha={lora_alpha}, "
              f"trainable encoder params={lora_params:,}", flush=True)

    # Gradient checkpointing: only when encoder is trainable AND VRAM is limited.
    # Frozen encoder + checkpointing causes incorrect outputs (PyTorch bug).
    # H100 80GB doesn't need checkpointing even with unfrozen encoder.
    freeze_base = stage_cfg.get("freeze_base", False)
    if not freeze_base:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        if gpu_mem < 50:  # A100 40GB needs checkpointing, H100 80GB doesn't
            model.gradient_checkpointing_enable()
            print(f"  Gradient checkpointing: ON ({gpu_mem:.0f}GB VRAM)", flush=True)
        else:
            print(f"  Gradient checkpointing: OFF ({gpu_mem:.0f}GB VRAM)", flush=True)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}", flush=True)

    # ── Build datasets for active tasks ──────────────────────

    # Load data — use stage-specific data_file if specified
    data_file = stage_cfg.get("data_file")
    if data_file:
        print(f"Loading training data: {data_file}", flush=True)
        with open(DATA_DIR / data_file) as f:
            train_examples = json.load(f)
    else:
        # Default: load standard NER + UD files
        train_examples = None

    # Dev data — use stage-specific dev_file if specified
    dev_file = stage_cfg.get("dev_file", "ner_dev.json")
    with open(DATA_DIR / dev_file) as f:
        ner_dev = json.load(f)
    print(f"Dev data: {dev_file} ({len(ner_dev)} examples)", flush=True)
    with open(DATA_DIR / "ud_dev.json") as f:
        ud_dev = json.load(f)

    ner_map = {l: i for i, l in enumerate(ner_labels)}
    pos_map = {l: i for i, l in enumerate(pos_labels)}
    dep_map = {l: i for i, l in enumerate(dep_labels)}
    srl_map = {l: i for i, l in enumerate(srl_labels)}
    cls_map = {l: i for i, l in enumerate(cls_labels)}

    active_tasks = stage_cfg["tasks"]
    datasets = []

    # Optional cap on primary data file examples
    max_primary = stage_cfg.get("max_primary_examples")
    if max_primary and train_examples and len(train_examples) > max_primary:
        import random
        random.seed(42)
        train_examples = random.sample(train_examples, max_primary)
        print(f"  Subsampled primary data to {max_primary:,} examples", flush=True)

    if data_file and train_examples:
        # Stage-specific data file — build datasets from single source
        for task in active_tasks:
            if task == "ner":
                task_exs = [ex for ex in train_examples if "ner_tags" in ex]
                if task_exs:
                    datasets.append(MultiTaskDataset(task_exs, "ner", tokenizer, ner_map, "ner_tags", max_length))
            elif task == "pos":
                task_exs = [ex for ex in train_examples if "pos_tags" in ex]
                if task_exs:
                    datasets.append(MultiTaskDataset(task_exs, "pos", tokenizer, pos_map, "pos_tags", max_length))
            elif task == "dep":
                task_exs = [ex for ex in train_examples if "dep_labels" in ex]
                if task_exs:
                    datasets.append(MultiTaskDataset(task_exs, "dep", tokenizer, dep_map, "dep_labels", max_length))
            elif task == "srl":
                task_exs = [ex for ex in train_examples if "srl_tags" in ex]
                if task_exs:
                    datasets.append(MultiTaskDataset(task_exs, "srl", tokenizer, srl_map, "srl_tags", max_length))
            elif task == "cls":
                task_exs = [ex for ex in train_examples if "cls_label" in ex]
                if task_exs:
                    datasets.append(MultiTaskDataset(task_exs, "cls", tokenizer, cls_map, "cls_label", max_length))

        # Fallback: load standard files for tasks missing from data_file
        loaded_tasks = {ds.task for ds in datasets}
        missing = set(active_tasks) - loaded_tasks
        if missing:
            print(f"  Loading standard data for regularization tasks: {missing}", flush=True)
            if "dep" in missing or "pos" in missing:
                with open(DATA_DIR / "ud_train.json") as f:
                    ud_train = json.load(f)
                if "dep" in missing:
                    datasets.append(MultiTaskDataset(ud_train, "dep", tokenizer, dep_map, "dep_labels", max_length))
                if "pos" in missing:
                    datasets.append(MultiTaskDataset(ud_train, "pos", tokenizer, pos_map, "pos_tags", max_length))
            if "ner" in missing:
                with open(DATA_DIR / "ner_train.json") as f:
                    ner_train = json.load(f)
                datasets.append(MultiTaskDataset(ner_train, "ner", tokenizer, ner_map, "ner_tags", max_length))
    else:
        # Default loading from standard files
        with open(DATA_DIR / "ner_train.json") as f:
            ner_train = json.load(f)
        with open(DATA_DIR / "ud_train.json") as f:
            ud_train = json.load(f)

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
    batch_size = stage_cfg.get("batch_size", config["training"]["batch_size"])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=multitask_collator, num_workers=0,
    )

    cls_dev = [ex for ex in ud_dev + ner_dev if "cls_label" in ex]

    # SRL dev (may not exist yet)
    srl_dev_path = DATA_DIR / "srl_dev.json"
    srl_dev = []
    if srl_dev_path.exists():
        with open(srl_dev_path) as f:
            srl_dev = json.load(f)

    print(f"Train: {len(train_dataset):,} examples, {len(train_loader)} batches/epoch", flush=True)

    # ── Optimizer: configure freezing and learning rates per stage ──
    new_head_name = stage_cfg["new_head"]
    freeze_base = stage_cfg.get("freeze_base", False)

    use_lora = stage_cfg.get("use_lora", False)

    if freeze_base and new_head_name:
        # Freeze encoder + existing heads — only train the NEW head
        # For SRL: also train dep_proj (compresses DEP probs for SRL cascade)
        # For LoRA: also train LoRA adapter params in the encoder
        head_params = []
        lora_params = []
        frozen_count = 0
        for name, param in model.named_parameters():
            if (f"{new_head_name}_head" in name
                    or (new_head_name == "srl" and "dep_proj" in name)
                    or (new_head_name == "srl" and "srl_adapter" in name)):
                param.requires_grad = True
                head_params.append(param)
            elif use_lora and "lora_" in name:
                param.requires_grad = True
                lora_params.append(param)
            else:
                param.requires_grad = False
                frozen_count += 1
        param_groups = [{"params": head_params, "lr": stage_cfg["head_lr"]}]
        if lora_params:
            param_groups.append({"params": lora_params, "lr": stage_cfg["head_lr"] * 0.1})
        trainable = sum(p.numel() for p in head_params) + sum(p.numel() for p in lora_params)
        trainable_names = f"{new_head_name}_head"
        if new_head_name == "srl" and model.dep_proj is not None:
            trainable_names += " + dep_proj"
        if new_head_name == "srl" and getattr(model, "srl_adapter", None) is not None:
            trainable_names += " + srl_adapter"
        if lora_params:
            trainable_names += f" + LoRA({sum(p.numel() for p in lora_params):,})"
        print(f"  Frozen: {frozen_count} params, Trainable: {trainable:,} ({trainable_names})", flush=True)

    elif freeze_base and not new_head_name:
        # Freeze encoder — train ALL heads (e.g. 2b: POS+NER alignment)
        head_params = []
        frozen_count = 0
        for name, param in model.named_parameters():
            if "_head" in name:
                param.requires_grad = True
                head_params.append(param)
            else:
                param.requires_grad = False
                frozen_count += 1
        param_groups = [{"params": head_params, "lr": stage_cfg["head_lr"]}]
        trainable = sum(p.numel() for p in head_params)
        print(f"  Frozen encoder: {frozen_count} params, Trainable heads: {trainable:,}", flush=True)

    elif not freeze_base and stage_cfg.get("base_lr"):
        # Differential LR — encoder at low LR, heads at per-head LR (e.g. 2c, 5)
        for param in model.parameters():
            param.requires_grad = True
        base_params = [p for n, p in model.named_parameters() if "_head" not in n and "dep_proj" not in n]
        param_groups = [{"params": base_params, "lr": stage_cfg["base_lr"]}]

        # Per-head LR: check for head-specific LR keys (e.g. pos_lr, ner_lr)
        for head_name in ["pos", "ner", "dep", "srl", "cls"]:
            head_params = [p for n, p in model.named_parameters() if f"{head_name}_head" in n]
            if head_params:
                lr = stage_cfg.get(f"{head_name}_lr", stage_cfg["head_lr"])
                param_groups.append({"params": head_params, "lr": lr})

        # dep_proj gets head_lr by default
        dep_proj_params = [p for n, p in model.named_parameters() if "dep_proj" in n]
        if dep_proj_params:
            param_groups.append({"params": dep_proj_params, "lr": stage_cfg["head_lr"]})

        lr_summary = {k.replace("_lr", ""): v for k, v in stage_cfg.items() if k.endswith("_lr")}
        print(f"  Unfrozen: LRs={lr_summary}", flush=True)

    else:
        # Uniform LR for everything
        for param in model.parameters():
            param.requires_grad = True
        param_groups = [{"params": list(model.parameters()), "lr": stage_cfg["head_lr"]}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config["training"]["weight_decay"])

    total_steps = stage_cfg["epochs"] * len(train_loader)
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    task_weights = {
        "ner": config["training"]["ner_loss_weight"],
        "pos": config["training"]["pos_loss_weight"],
        "dep": config["training"]["dep_loss_weight"],
        "srl": config["training"].get("srl_loss_weight", 1.0),
        "cls": config["training"]["cls_loss_weight"],
    }

    token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    seq_loss_fn = nn.CrossEntropyLoss()

    # ── Output directory ─────────────────────────────────────
    output_dir = Path(config["output"]["dir"]) / f"stage{stage_key}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────
    task_names = ["ner", "pos", "dep", "srl", "cls"]
    logit_keys = ["ner_logits", "pos_logits", "dep_logits", "srl_logits", "cls_logits"]

    best_composite = -1.0
    best_epoch = -1
    patience = stage_cfg.get("patience", 0)  # 0 = no early stopping
    no_improve = 0
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

            predicate_idx = batch.get("predicate_idx")
            if predicate_idx is not None:
                predicate_idx = predicate_idx.to(device)
            outputs = model(input_ids, attention_mask, predicate_idx=predicate_idx)

            loss = torch.tensor(0.0, device=device, requires_grad=True)
            step_tasks = {}

            task_id_map = {"ner": 0, "pos": 1, "dep": 2, "cls": 3, "srl": 4}
            for task_name, logit_key in zip(task_names, logit_keys):
                if task_name not in active_tasks:
                    continue
                if logit_key not in outputs:
                    continue

                mask = task_ids == task_id_map[task_name]
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
            srl_dev[:500] if "srl" in eval_tasks else [],
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
        if "srl" in dev_results:
            parts.append(f"SRL F1={dev_results['srl'].get('f1', 0):.3f}")
        if "cls" in dev_results:
            parts.append(f"CLS F1={dev_results['cls'].get('macro_f1', 0):.3f}")

        comp = composite_score_active(dev_results, eval_tasks)
        parts.append(f"Composite={comp:.3f}")
        print(f"  Dev: {' | '.join(parts)}", flush=True)

        if comp > best_composite:
            best_composite = comp
            best_epoch = epoch + 1
            no_improve = 0
            model.save(str(output_dir / "best"))
            tokenizer.save_pretrained(str(output_dir / "best"))
            print(f"  * New best (composite={comp:.3f})", flush=True)
        elif patience > 0:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping: no improvement for {patience} epochs", flush=True)
                break

    # Save final
    model.save(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    print(f"\nStage {stage} complete.", flush=True)
    print(f"  Best epoch: {best_epoch} (composite={best_composite:.3f})", flush=True)
    print(f"  Best model: {output_dir / 'best'}", flush=True)
    print(f"  Final model: {output_dir / 'final'}", flush=True)


def evaluate_active(model, tokenizer, device, vocabs, ner_dev, ud_dev, srl_dev, cls_dev, max_length, active_tasks):
    """Evaluate only active heads."""
    model.eval()
    results = {}

    pos_labels = vocabs["pos_labels"]
    ner_labels = vocabs["ner_labels"]
    dep_labels = vocabs["dep_labels"]
    srl_labels = vocabs.get("srl_labels", [])
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
        # DEP eval requires decoding dep2label → heads + rels
        gold_h, pred_h, gold_r, pred_r = [], [], [], []
        for ex in ud_dev:
            encoding = tokenizer(
                ex["words"], is_split_into_words=True,
                max_length=max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            with torch.no_grad():
                out = model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device))
            if "dep_logits" not in out:
                break
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
        if gold_h:
            results["dep"] = evaluate_dep(gold_h, pred_h, gold_r, pred_r)

    if "srl" in active_tasks and srl_dev:
        gold, pred = predict_tokens(srl_dev, "srl_logits", srl_labels, "srl_tags")
        if gold:
            # SRL uses BIO encoding — evaluate like NER (span-level F1)
            results["srl"] = evaluate_ner(gold, pred)

    if "cls" in active_tasks and cls_dev:
        gold_cls, pred_cls = [], []
        for ex in cls_dev:
            text = ex.get("text", " ".join(ex["words"]))
            prev_text = ex.get("prev_text")
            if prev_text:
                encoding = tokenizer(prev_text, text, max_length=max_length,
                                     padding="max_length", truncation=True, return_tensors="pt")
            else:
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
    if "srl" in active_tasks and "srl" in results:
        scores.append(results["srl"].get("f1", 0))
        weights.append(0.2)
    if "cls" in active_tasks and "cls" in results:
        scores.append(results["cls"].get("macro_f1", 0))
        weights.append(0.1)

    if not weights:
        return 0.0
    # Normalize weights to sum to 1
    total_w = sum(weights)
    return sum(s * w / total_w for s, w in zip(scores, weights))


def main():
    valid_stages = ["1", "2a", "2b", "2c", "2d", "2e", "2f", "3", "3s", "3m", "3b", "3b-dep", "3b-512",
                     "3c", "3c-mt", "3c-adapt", "3c-lora", "4", "5"]
    parser = argparse.ArgumentParser(description="Staged cascade training")
    parser.add_argument("--stage", type=str, required=True, choices=valid_stages,
                        help="Training stage (1=POS, 2a=NER, 2b=POS+NER align, 2c=encoder, 3=DEP, 4=CLS, 5=joint)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to previous stage best checkpoint (required for stages 2+)")
    args = parser.parse_args()

    if args.stage != "1" and not args.checkpoint:
        parser.error(f"Stage {args.stage} requires --checkpoint from previous stage")

    train_stage(args.stage, args.checkpoint)


if __name__ == "__main__":
    main()

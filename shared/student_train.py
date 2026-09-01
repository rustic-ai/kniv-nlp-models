"""Local-reproducible v2 distillation training for the kniv student cascade.

Mirrors the production Colab pipeline that produced the published xsmall and
small student models. The v2 recipe is:

* **Stage 1** — joint multi-loss distillation across all five heads.
  Loss = ALPHA * (logit-KL across heads + DEP arc CE + DEP rel CE)
       + (1-ALPHA) * alpha_hard * (hard CE for POS/NER/SRL/CLS)
       + gamma_hidden * (PKD-style hidden-state MSE, layer-norm'd)
       + delta_rdrop * (symmetric R-Drop KL across two forward passes)

* **Stage 2a** — SRL-only fine-tune on (teacher silver + gold × 2) PropBank
  data. Encoder, predicate embedding, and POS/NER/DEP/SRL heads remain
  trainable; CLS is frozen. Best Viterbi-F1 checkpoint is kept.

* **Stage 2b** — CLS-only fine-tune on SGD/MultiWOZ. Encoder + every
  non-CLS module set to ``eval()`` and ``requires_grad=False`` (so dropout
  stays off and SRL/POS/NER/DEP do not drift). Only ``cls_sm``,
  ``cls_pool``, ``cls_head`` train.

The pipeline strips the train-only PKD projections from the final
checkpoint so published model.pt has no dead weights.

Two entry points:

1. ``train_stage1`` / ``train_stage2a`` / ``train_stage2b`` — single-stage
   functions for explicit orchestration (used by
   ``scripts/run_student_pipeline.py``).
2. ``train_one_trial`` — single-stage Stage-1 wrapper that matches the
   ``shared.hpo_trial.TrialRunner`` protocol used by the Optuna sweep.

Stage 2 prerequisites (not needed for sweeps that only run Stage 1):

* ``data/prepared/kniv-deberta-cascade/srl_train.json``
* ``data/prepared/kniv-deberta-cascade/srl_dev.json``
* ``data/prepared/kniv-deberta-cascade/cls_sgd_mwoz_train.json``

These are produced by the corpus-prep pipeline; see
``docs/data-preparation-plan.md``.

Distillation parquet schema (v2 — emitted by the data-gen notebook):

    words, n_words, pos_logits, ner_logits, cls_logits, dep_arc_scores,
    dep_label_top5, pos_hard, ner_hard,
    per_verb_count, per_verb_indices, per_verb,
    per_verb_srl_logits, per_verb_pred_hidden,
    hidden_l12, hidden_l18, hidden_l24

If the shards are an older simpler format that lacks ``hidden_l*``,
``--no-pkd`` must be passed (Stage 1 will refuse to start otherwise so
nobody silently reproduces a different recipe).

Usage:

    # Run all three stages end-to-end (default).
    uv run python -m shared.student_train \\
        --student microsoft/deberta-v3-xsmall \\
        --distillation-data data/distillation \\
        --stage2-data data/prepared/kniv-deberta-cascade \\
        --output models/kniv-deberta-nlp-base-en-xsmall-trial

    # Only Stage 1 (e.g., for a quick sanity run or sweep).
    uv run python -m shared.student_train --stage 1 ...
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "models"))

from student_loader import (  # noqa: E402
    StudentCascadeV5, viterbi_decode,
    POS_LABELS, NER_LABELS, SRL_TAGS, CLS_LABELS, DEPREL_LIST,
    N_POS, N_NER, N_SRL, N_DEP, N_CLS, H_TEACHER,
)
from shared.hpo_trial import TrialParams, TrialResult  # noqa: E402

pos_map = {t: i for i, t in enumerate(POS_LABELS)}
ner_map = {t: i for i, t in enumerate(NER_LABELS)}
srl_map = {t: i for i, t in enumerate(SRL_TAGS)}
cls_map = {t: i for i, t in enumerate(CLS_LABELS)}

DEFAULT_TEACHER_LAYERS = (12, 18, 24)


# ── Stage 1 dataset ─────────────────────────────────────────────────

class DistillationDataset(Dataset):
    """v2 distillation shard reader with per-verb SRL sampling.

    Each row is one sentence. SRL distillation samples *one* of the
    sentence's verbs per epoch — randomized per ``__getitem__`` call so
    over multiple epochs the student sees every predicate.

    Optional columns (auto-detected):
      * ``hidden_l12`` / ``hidden_l18`` / ``hidden_l24`` — teacher hidden
        states for PKD; if absent, hidden tensors come back zero and the
        caller should disable PKD.
      * ``per_verb_pred_hidden`` — predicate-aware teacher hidden for the
        sampled verb; same fallback.
    """

    def __init__(self, parquet_paths, tokenizer, max_length: int = 128,
                 teacher_layers: tuple[int, ...] = DEFAULT_TEACHER_LAYERS):
        try:
            import pyarrow.parquet as pq
            import pyarrow as pa
        except ImportError as e:
            raise ImportError(
                "pyarrow required to read distillation shards. "
                "Install with: uv pip install pyarrow"
            ) from e
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.teacher_layers = tuple(teacher_layers)
        tables = [pq.read_table(p) for p in parquet_paths]
        self.data = pa.concat_tables(tables).to_pandas()
        cols = set(self.data.columns)
        self.has_hidden = all(f"hidden_l{l}" in cols for l in self.teacher_layers)
        self.has_pred_hidden = "per_verb_pred_hidden" in cols
        self.has_dep_label_top5 = "dep_label_top5" in cols
        self.has_per_verb = "per_verb_count" in cols and "per_verb" in cols

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.data.iloc[idx]
        words = list(row["words"])
        n = int(row["n_words"])
        S = self.max_length

        enc = self.tokenizer(
            words, is_split_into_words=True, max_length=S,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        word_ids = enc.word_ids()

        # Word→first-subtoken-index map, capped at n
        w2t: dict[int, int] = {}
        prev = None
        for k, wid in enumerate(word_ids):
            if wid is not None and wid != prev and wid < n:
                w2t[wid] = k
            prev = wid
        valid_word_indices = sorted(w2t.keys())

        # Sentence-level teacher tensors
        t_pos = np.frombuffer(row["pos_logits"], dtype=np.float16).reshape(n, N_POS).astype(np.float32)
        t_ner = np.frombuffer(row["ner_logits"], dtype=np.float16).reshape(n, N_NER).astype(np.float32)
        t_cls = np.frombuffer(row["cls_logits"], dtype=np.float16).reshape(N_CLS).astype(np.float32)
        t_arc = np.frombuffer(row["dep_arc_scores"], dtype=np.float16).reshape(n, n).astype(np.float32)

        # DEP label top-5 (optional — older shards stored this differently)
        dep_label_top5: list[list[list[int]]]
        if self.has_dep_label_top5:
            dep_label_top5 = json.loads(row["dep_label_top5"])
        else:
            dep_label_top5 = [[] for _ in range(n)]

        # Teacher hidden states (optional)
        teacher_hidden: dict[int, np.ndarray] = {}
        if self.has_hidden:
            for tl in self.teacher_layers:
                teacher_hidden[tl] = (
                    np.frombuffer(row[f"hidden_l{tl}"], dtype=np.float16)
                    .reshape(n, H_TEACHER).astype(np.float32)
                )

        # Per-verb sampling for SRL
        srl_nonzero = False
        verb_idx = 0
        t_srl = np.zeros((n, N_SRL), dtype=np.float32)
        t_pred_hidden = np.zeros((n, H_TEACHER), dtype=np.float32)
        srl_hard_seq: list[str] = ["O"] * n

        if self.has_per_verb:
            per_verb_count = int(row["per_verb_count"])
            srl_nonzero = per_verb_count > 0
            if srl_nonzero:
                per_verb_indices = json.loads(row["per_verb_indices"])
                pick = random.randint(0, per_verb_count - 1)
                verb_idx = int(per_verb_indices[pick])

                srl_size_per_verb = n * N_SRL * 2  # fp16 bytes
                srl_offset = pick * srl_size_per_verb
                t_srl = (
                    np.frombuffer(row["per_verb_srl_logits"], dtype=np.float16,
                                  count=n * N_SRL, offset=srl_offset)
                    .reshape(n, N_SRL).astype(np.float32)
                )
                if self.has_pred_hidden:
                    hidden_size_per_verb = n * H_TEACHER * 2
                    hidden_offset = pick * hidden_size_per_verb
                    t_pred_hidden = (
                        np.frombuffer(row["per_verb_pred_hidden"], dtype=np.float16,
                                      count=n * H_TEACHER, offset=hidden_offset)
                        .reshape(n, H_TEACHER).astype(np.float32)
                    )
                per_verb_data = json.loads(row["per_verb"])
                srl_hard_seq = per_verb_data[pick]["srl_hard"]

        # Aligned tensors
        pos_t = torch.zeros(S, N_POS)
        ner_t = torch.zeros(S, N_NER)
        srl_t = torch.zeros(S, N_SRL)
        valid = torch.zeros(S, dtype=torch.bool)
        pos_h = torch.full((S,), -100, dtype=torch.long)
        ner_h = torch.full((S,), -100, dtype=torch.long)
        srl_hard = torch.full((S,), -100, dtype=torch.long)
        dep_head_hard = torch.full((S,), -100, dtype=torch.long)
        dep_rel_hard = torch.full((S,), -100, dtype=torch.long)
        hidden_targets = {tl: torch.zeros(S, H_TEACHER) for tl in self.teacher_layers}
        pred_hidden_target = torch.zeros(S, H_TEACHER)

        pos_hard_list = list(row["pos_hard"])
        ner_hard_list = list(row["ner_hard"])

        for wid, k in w2t.items():
            pos_t[k] = torch.from_numpy(t_pos[wid])
            ner_t[k] = torch.from_numpy(t_ner[wid])
            srl_t[k] = torch.from_numpy(t_srl[wid])
            valid[k] = True
            if wid < len(pos_hard_list):
                pos_h[k] = pos_map.get(pos_hard_list[wid], 0)
            if wid < len(ner_hard_list):
                ner_h[k] = ner_map.get(ner_hard_list[wid], 0)
            if self.has_hidden:
                for tl in self.teacher_layers:
                    hidden_targets[tl][k] = torch.from_numpy(teacher_hidden[tl][wid])
            if self.has_pred_hidden and srl_nonzero:
                pred_hidden_target[k] = torch.from_numpy(t_pred_hidden[wid])
            if srl_nonzero and wid < len(srl_hard_seq):
                srl_hard[k] = srl_map.get(srl_hard_seq[wid], 0)
            # DEP head — argmax over valid words
            if valid_word_indices:
                arc_at_valid = t_arc[wid, valid_word_indices]
                best_word = valid_word_indices[int(np.argmax(arc_at_valid))]
                dep_head_hard[k] = w2t[best_word]
            # DEP relation — top-1 from teacher's stored top-5 (id, score)
            if wid < len(dep_label_top5) and dep_label_top5[wid]:
                dep_rel_hard[k] = int(dep_label_top5[wid][0][0])

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "predicate_idx": torch.tensor(w2t.get(verb_idx, 0), dtype=torch.long),
            "valid": valid,
            "pos_t": pos_t, "ner_t": ner_t, "srl_t": srl_t,
            "cls_t": torch.from_numpy(t_cls),
            "pos_h": pos_h, "ner_h": ner_h, "srl_hard": srl_hard,
            "dep_head_hard": dep_head_hard, "dep_rel_hard": dep_rel_hard,
            "srl_valid": torch.tensor(srl_nonzero, dtype=torch.bool),
            **{f"teacher_hidden_l{tl}": hidden_targets[tl] for tl in self.teacher_layers},
            "teacher_pred_hidden": pred_hidden_target,
        }


def _collate(feats: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([f[k] for f in feats]) for k in feats[0]}


# ── Loss components ─────────────────────────────────────────────────

def kl_loss(s_logits: torch.Tensor, t_logits: torch.Tensor,
            valid_mask: torch.Tensor, T: float) -> torch.Tensor:
    sl = s_logits.view(-1, s_logits.size(-1))[valid_mask]
    tl = t_logits.view(-1, t_logits.size(-1))[valid_mask]
    if sl.numel() == 0:
        return torch.tensor(0.0, device=s_logits.device)
    return F.kl_div(
        F.log_softmax(sl / T, dim=-1),
        F.softmax(tl / T, dim=-1),
        reduction="batchmean",
    ) * (T ** 2)


def hard_ce(s_logits: torch.Tensor, hard_labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        s_logits.view(-1, s_logits.size(-1)),
        hard_labels.view(-1),
        ignore_index=-100,
    )


def hidden_state_mse(s_proj: torch.Tensor, t_hidden: torch.Tensor,
                     valid_mask: torch.Tensor) -> torch.Tensor:
    sh = s_proj.view(-1, H_TEACHER)[valid_mask]
    th = t_hidden.view(-1, H_TEACHER)[valid_mask]
    if sh.numel() == 0:
        return torch.tensor(0.0, device=s_proj.device)
    sh = F.layer_norm(sh, [H_TEACHER])
    th = F.layer_norm(th, [H_TEACHER])
    return F.mse_loss(sh, th)


def rdrop_kl(logits1: torch.Tensor, logits2: torch.Tensor,
             valid_mask: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """Symmetric KL between two student forward passes (different dropout)."""
    l1 = logits1.view(-1, logits1.size(-1))[valid_mask]
    l2 = logits2.view(-1, logits2.size(-1))[valid_mask]
    if l1.numel() == 0:
        return torch.tensor(0.0, device=logits1.device)
    p1 = F.log_softmax(l1 / T, dim=-1)
    p2 = F.log_softmax(l2 / T, dim=-1)
    kl_12 = F.kl_div(p1, p2, log_target=True, reduction="batchmean")
    kl_21 = F.kl_div(p2, p1, log_target=True, reduction="batchmean")
    return 0.5 * (kl_12 + kl_21) * (T ** 2)


# ── Stage 1: joint distillation ────────────────────────────────────

@dataclass
class StageConfig:
    student_encoder: str
    distillation_data: Path
    output_dir: Path
    device: str | None = None
    max_length: int = 128
    seed: int = 42
    teacher_layers: tuple[int, ...] = DEFAULT_TEACHER_LAYERS


def _build_model(student_encoder: str, device: str,
                 teacher_layers: tuple[int, ...] | None = None
                 ) -> tuple[StudentCascadeV5, Any, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(student_encoder)
    encoder = AutoModel.from_pretrained(student_encoder).float()
    H = encoder.config.hidden_size
    NL = encoder.config.num_hidden_layers + 1
    model = StudentCascadeV5(encoder, H, NL).to(device)
    if teacher_layers:
        model.enable_pkd_projections(teacher_layers=teacher_layers)
    info = {
        "encoder_id": student_encoder,
        "encoder_hidden_dim": H,
        "encoder_layers": encoder.config.num_hidden_layers,
        "params_total_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "params_encoder_millions": round(sum(p.numel() for p in encoder.parameters()) / 1e6, 1),
    }
    return model, tokenizer, info


def train_stage1(params: TrialParams, cfg: StageConfig,
                 model: StudentCascadeV5 | None = None,
                 tokenizer: Any = None) -> tuple[StudentCascadeV5, Any, dict[str, Any]]:
    """Stage 1: joint multi-loss distillation across all five heads."""
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    teacher_layers = cfg.teacher_layers if params.use_pkd else ()
    if model is None:
        model, tokenizer, info = _build_model(
            cfg.student_encoder, device,
            teacher_layers=teacher_layers,
        )
    else:
        info = {"encoder_id": cfg.student_encoder}

    shards = sorted(Path(cfg.distillation_data).glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No shard_*.parquet under {cfg.distillation_data}. "
            "Generate distillation data first (see docs/data-preparation-plan.md)."
        )
    dataset = DistillationDataset(
        shards, tokenizer, max_length=cfg.max_length,
        teacher_layers=cfg.teacher_layers,
    )
    if params.use_pkd and not dataset.has_hidden:
        raise RuntimeError(
            "PKD enabled but distillation shards lack hidden_l* columns. "
            "Re-run data generation to include teacher hidden states, "
            "or pass --no-pkd to disable PKD-style hidden-state matching."
        )
    if params.use_pkd and not dataset.has_pred_hidden:
        print("[stage1] WARNING: shards lack per_verb_pred_hidden — "
              "predicate-aware hidden distillation will be skipped.")

    loader = DataLoader(
        dataset, batch_size=params.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=2, pin_memory=True,
    )

    enc_params = list(model.encoder.parameters()) + list(model.pred_embedding.parameters())
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("encoder.") and not n.startswith("pred_embedding.")]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": params.lr_encoder},
         {"params": head_params, "lr": params.lr_heads}],
        weight_decay=params.weight_decay,
    )
    total_steps = max(1, params.epochs * len(loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * params.warmup_ratio), total_steps,
    )

    T = params.temperature
    last_avg_loss = 0.0

    for epoch in range(params.epochs):
        model.train()
        running = {"total": 0.0, "kl": 0.0, "hard": 0.0, "hidden": 0.0, "rdrop": 0.0, "n": 0}
        for batch in tqdm(loader, desc=f"S1 E{epoch + 1}/{params.epochs}"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            pred = batch["predicate_idx"].to(device)
            valid = batch["valid"].to(device)
            B, S = ids.size()
            fv = valid.view(B * S)

            out1 = model(ids, mask, pred, return_hidden_states=params.use_pkd)
            if params.use_pkd:
                s_pos, s_ner, s_arc, s_lab, s_srl, s_cls, student_hidden = out1
            else:
                s_pos, s_ner, s_arc, s_lab, s_srl, s_cls = out1

            # Soft KL
            kl_pos = kl_loss(s_pos, batch["pos_t"].to(device), fv, T)
            kl_ner = kl_loss(s_ner, batch["ner_t"].to(device), fv, T)

            sv = batch["srl_valid"].to(device)
            srl_mask = (valid & sv.unsqueeze(1)).view(B * S)
            if srl_mask.any():
                kl_srl = kl_loss(s_srl, batch["srl_t"].to(device), srl_mask, T)
            else:
                kl_srl = torch.tensor(0.0, device=device)

            # CLS soft KL — scaled by avg tokens per sentence so the
            # batchmean reduction is comparable to token-level losses
            tps = max(valid.sum().float().item() / B, 1.0)
            cls_ones = torch.ones(B, dtype=torch.bool, device=device)
            kl_cls = kl_loss(s_cls, batch["cls_t"].to(device), cls_ones, T) * tps

            # Hard CE
            h_pos = hard_ce(s_pos, batch["pos_h"].to(device))
            h_ner = hard_ce(s_ner, batch["ner_h"].to(device))
            if srl_mask.any():
                h_srl = hard_ce(s_srl, batch["srl_hard"].to(device))
            else:
                h_srl = torch.tensor(0.0, device=device)
            h_cls = F.cross_entropy(s_cls, batch["cls_t"].to(device).argmax(-1))

            # DEP arc + relation
            dh = batch["dep_head_hard"].to(device)
            dep_arc = F.cross_entropy(
                s_arc.view(B * S, S), dh.view(B * S), ignore_index=-100,
            )
            ph = dh.clamp(0).unsqueeze(-1).unsqueeze(-1).expand(B, S, 1, N_DEP)
            gathered_lab = s_lab.gather(2, ph).squeeze(2)
            dep_rel = F.cross_entropy(
                gathered_lab.view(B * S, N_DEP),
                batch["dep_rel_hard"].to(device).view(B * S),
                ignore_index=-100,
            )

            # PKD hidden-state MSE
            if params.use_pkd and model.distill_projections is not None:
                hidden_loss = torch.tensor(0.0, device=device)
                for tl in cfg.teacher_layers:
                    s_proj = model.distill_projections[f"l{tl}"](student_hidden[tl])
                    t_h = batch[f"teacher_hidden_l{tl}"].to(device)
                    hidden_loss = hidden_loss + hidden_state_mse(s_proj, t_h, fv)
                if dataset.has_pred_hidden and srl_mask.any():
                    last_tl = max(cfg.teacher_layers)
                    s_last_proj = model.distill_projections[f"l{last_tl}"](
                        student_hidden[last_tl]
                    )
                    pred_h = hidden_state_mse(
                        s_last_proj, batch["teacher_pred_hidden"].to(device), srl_mask,
                    )
                    hidden_loss = hidden_loss + params.hidden_mult_pred * pred_h
            else:
                hidden_loss = torch.tensor(0.0, device=device)

            # R-Drop
            if params.use_rdrop:
                out2 = model(ids, mask, pred)
                _, s2_ner, _, _, s2_srl, s2_cls = out2
                rdrop_total = rdrop_kl(s_ner, s2_ner, fv)
                if srl_mask.any():
                    rdrop_total = rdrop_total + rdrop_kl(s_srl, s2_srl, srl_mask)
                rdrop_total = rdrop_total + rdrop_kl(
                    s_cls.unsqueeze(1), s2_cls.unsqueeze(1),
                    torch.ones(B, dtype=torch.bool, device=device),
                )
            else:
                rdrop_total = torch.tensor(0.0, device=device)

            soft_total = (
                params.loss_w_pos * kl_pos
                + params.loss_w_ner * kl_ner
                + params.loss_w_srl * kl_srl
                + params.loss_w_cls * kl_cls
                + params.loss_w_dep * dep_arc
                + params.loss_w_dep_rel * dep_rel
            )
            hard_total = (
                params.loss_w_pos * h_pos
                + params.loss_w_ner * h_ner
                + params.loss_w_srl * h_srl
                + params.loss_w_cls * h_cls
            )

            loss = (
                params.alpha * soft_total
                + (1 - params.alpha) * params.alpha_hard * hard_total
                + params.gamma_hidden * hidden_loss
                + params.rdrop_gamma * rdrop_total
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running["total"] += float(loss.item())
            running["kl"] += float(soft_total.item())
            running["hard"] += float(hard_total.item())
            running["hidden"] += float(hidden_loss.item())
            running["rdrop"] += float(rdrop_total.item())
            running["n"] += 1

        n = max(running["n"], 1)
        last_avg_loss = running["total"] / n
        print(f"  S1 E{epoch + 1} — loss {last_avg_loss:.3f}  "
              f"(kl {running['kl'] / n:.3f}, hard {running['hard'] / n:.3f}, "
              f"hidden {running['hidden'] / n:.3f}, rdrop {running['rdrop'] / n:.3f})")

    info["stage1_avg_loss"] = last_avg_loss
    return model, tokenizer, info


# ── Stage 2 helpers ─────────────────────────────────────────────────

def _extract_teacher_silver_srl(parquet_paths: list[Path]) -> list[dict[str, Any]]:
    """Pull (sentence, verb_idx, srl_hard) triples from per-verb shard data."""
    import pyarrow.parquet as pq
    silver: list[dict[str, Any]] = []
    for path in parquet_paths:
        table = pq.read_table(
            path, columns=["words", "n_words", "per_verb_count",
                           "per_verb_indices", "per_verb"],
        )
        df = table.to_pandas()
        for _, row in df.iterrows():
            n_verbs = int(row["per_verb_count"])
            if n_verbs == 0:
                continue
            words = list(row["words"])
            per_verb = json.loads(row["per_verb"])
            for pv in per_verb:
                silver.append({
                    "words": words,
                    "predicate_idx": int(pv["verb_idx"]),
                    "srl_tags": pv["srl_hard"],
                })
    return silver


class _SRLFineTuneDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length: int = 128):
        self.e = examples
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.e)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ex = self.e[i]
        enc = self.tok(
            ex["words"], is_split_into_words=True, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        labels: list[int] = []
        pt = 0
        prev = None
        for k, wid in enumerate(enc.word_ids()):
            if wid is None:
                labels.append(-100)
            elif wid != prev:
                tag = ex["srl_tags"][wid] if wid < len(ex["srl_tags"]) else "O"
                labels.append(srl_map.get(tag, 0))
                if wid == ex["predicate_idx"]:
                    pt = k
            else:
                labels.append(-100)
            prev = wid
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
            "predicate_idx": torch.tensor(pt, dtype=torch.long),
        }


class _CLSFineTuneDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length: int = 128):
        self.e = examples
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.e)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ex = self.e[i]
        text = ex.get("text", "")
        prev_text = ex.get("prev_text")
        if prev_text:
            enc = self.tok(prev_text, text, max_length=self.max_length,
                           padding="max_length", truncation=True, return_tensors="pt")
        else:
            enc = self.tok(text, max_length=self.max_length,
                           padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(cls_map[ex["cls_label"]], dtype=torch.long),
        }


def _eval_srl_dev(model: StudentCascadeV5, tokenizer: Any, srl_dev: list[dict[str, Any]],
                  device: str, max_examples: int = 300) -> tuple[float, float]:
    """Returns (argmax_acc, viterbi_acc) on the first ``max_examples`` items."""
    model.eval()
    sc_arg = sc_vit = st = 0
    with torch.no_grad():
        for ex in srl_dev[:max_examples]:
            enc = tokenizer(
                ex["words"], is_split_into_words=True, return_tensors="pt",
                padding=True, truncation=True, max_length=128,
            )
            pt = 0
            prev = None
            for k, wid in enumerate(enc.word_ids()):
                if wid is not None and wid != prev and wid == ex["predicate_idx"]:
                    pt = k
                    break
                prev = wid
            o = model(
                enc["input_ids"].to(device), enc["attention_mask"].to(device),
                torch.tensor([pt], dtype=torch.long, device=device),
            )[4]
            valid_idx: list[tuple[int, int]] = []
            prev = None
            for k, wid in enumerate(enc.word_ids()):
                if wid is not None and wid != prev and wid < len(ex["srl_tags"]):
                    valid_idx.append((k, wid))
                prev = wid
            if not valid_idx:
                continue
            valid_logits = o[0, [k for k, _ in valid_idx]].cpu()
            vit_path = viterbi_decode(valid_logits, SRL_TAGS)
            for idx, (k, wid) in enumerate(valid_idx):
                gold = ex["srl_tags"][wid]
                if SRL_TAGS[o[0, k].argmax().item()] == gold:
                    sc_arg += 1
                if SRL_TAGS[vit_path[idx]] == gold:
                    sc_vit += 1
                st += 1
    if st == 0:
        return 0.0, 0.0
    return sc_arg / st, sc_vit / st


def _eval_cls_dev(model: StudentCascadeV5, tokenizer: Any, cls_dev: list[dict[str, Any]],
                  device: str, max_examples: int = 500) -> float:
    model.eval()
    cc = ct = 0
    with torch.no_grad():
        for ex in cls_dev[:max_examples]:
            text = ex.get("text", "")
            prev_text = ex.get("prev_text")
            if prev_text:
                enc = tokenizer(prev_text, text, return_tensors="pt",
                                padding=True, truncation=True, max_length=128)
            else:
                enc = tokenizer(text, return_tensors="pt",
                                padding=True, truncation=True, max_length=128)
            o = model(
                enc["input_ids"].to(device), enc["attention_mask"].to(device),
                torch.zeros(1, dtype=torch.long, device=device),
            )[5]
            if CLS_LABELS[o.argmax(-1).item()] == ex["cls_label"]:
                cc += 1
            ct += 1
    return cc / ct if ct else 0.0


@dataclass
class Stage2DataPaths:
    srl_train: Path
    srl_dev: Path
    cls_sgd_mwoz: Path

    @classmethod
    def from_root(cls, root: Path) -> "Stage2DataPaths":
        return cls(
            srl_train=root / "srl_train.json",
            srl_dev=root / "srl_dev.json",
            cls_sgd_mwoz=root / "cls_sgd_mwoz_train.json",
        )

    def assert_exist(self) -> None:
        for p in (self.srl_train, self.srl_dev, self.cls_sgd_mwoz):
            if not p.exists():
                raise FileNotFoundError(
                    f"Stage 2 data missing: {p}. See docs/data-preparation-plan.md."
                )


def train_stage2a(model: StudentCascadeV5, tokenizer: Any, params: TrialParams,
                  cfg: StageConfig, data: Stage2DataPaths,
                  silver_limit: int = 80_000) -> dict[str, Any]:
    """Stage 2a: SRL fine-tune (encoder + non-CLS heads trainable, CLS frozen)."""
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(cfg.seed)

    # Freeze CLS, train everything else
    for p in model.parameters():
        p.requires_grad = False
    for p in model.encoder.parameters():
        p.requires_grad = True
    for p in model.pred_embedding.parameters():
        p.requires_grad = True
    non_cls_modules = [
        model.pos_sm, model.pos_head,
        model.ner_sm, model.ner_lstm, model.ner_proj, model.ner_head,
        model.dep_sm, model.dep_proj, model.dep_lstm, model.dep_lstm_proj, model.dep_biaff,
        model.srl_sm, model.srl_interaction_proj, model.srl_lstm, model.srl_proj, model.srl_head,
    ]
    for m in non_cls_modules:
        for p in m.parameters():
            p.requires_grad = True

    # Data: gold × 2 + capped silver
    srl_gold = json.loads(data.srl_train.read_text())
    srl_dev = json.loads(data.srl_dev.read_text())
    shards = sorted(Path(cfg.distillation_data).glob("shard_*.parquet"))
    print(f"[stage2a] extracting teacher silver SRL from {len(shards)} shards...")
    silver = _extract_teacher_silver_srl(shards)
    print(f"[stage2a]   {len(silver):,} verb-level silver examples extracted")
    if silver_limit and len(silver) > silver_limit:
        silver = random.sample(silver, silver_limit)
    srl_combined = silver + srl_gold * 2
    random.shuffle(srl_combined)
    print(f"[stage2a]   training set: {len(srl_combined):,} "
          f"({len(silver):,} silver + {len(srl_gold) * 2:,} gold×2)")

    loader = DataLoader(
        _SRLFineTuneDataset(srl_combined, tokenizer, cfg.max_length),
        batch_size=params.batch_size, shuffle=True, collate_fn=_collate,
        num_workers=2, pin_memory=True,
    )

    enc_params = list(model.encoder.parameters()) + list(model.pred_embedding.parameters())
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("encoder.")
                   and not n.startswith("pred_embedding.")]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": params.lr_encoder_s2a},
         {"params": head_params, "lr": params.lr_heads}],
        weight_decay=params.weight_decay,
    )
    total_steps = max(1, params.epochs_s2a * len(loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * 0.1), total_steps,
    )

    best_srl = 0.0
    history: list[dict[str, Any]] = []
    s2a_best_path = cfg.output_dir / "model_s2a_best.pt"

    for epoch in range(params.epochs_s2a):
        model.train()
        for b in tqdm(loader, desc=f"S2a E{epoch + 1}/{params.epochs_s2a}", leave=False):
            o = model(
                b["input_ids"].to(device), b["attention_mask"].to(device),
                b["predicate_idx"].to(device),
            )[4]
            loss = F.cross_entropy(
                o.view(-1, N_SRL),
                b["labels"].to(device).view(-1),
                ignore_index=-100,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        srl_arg, srl_vit = _eval_srl_dev(model, tokenizer, srl_dev, device)
        history.append({"epoch": epoch + 1, "srl_arg": srl_arg, "srl_vit": srl_vit})
        marker = ""
        if srl_vit > best_srl:
            best_srl = srl_vit
            torch.save(model.state_dict(), s2a_best_path)
            marker = f"  ★ BEST SRL ({srl_vit:.4f})"
        print(f"  S2a E{epoch + 1} — argmax {srl_arg:.4f}, viterbi {srl_vit:.4f}{marker}")

    # Reload best for Stage 2b handoff
    if s2a_best_path.exists():
        model.load_state_dict(torch.load(s2a_best_path, map_location="cpu"))
        model.to(device)

    return {"best_srl": best_srl, "history": history}


def train_stage2b(model: StudentCascadeV5, tokenizer: Any, params: TrialParams,
                  cfg: StageConfig, data: Stage2DataPaths) -> dict[str, Any]:
    """Stage 2b: CLS-only fine-tune; encoder + non-CLS heads frozen and in eval()."""
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(cfg.seed)

    for p in model.parameters():
        p.requires_grad = False
    for m in [model.cls_sm, model.cls_pool, model.cls_head]:
        for p in m.parameters():
            p.requires_grad = True

    cls_all = json.loads(data.cls_sgd_mwoz.read_text())
    random.seed(42)
    random.shuffle(cls_all)
    cls_dev = cls_all[:4000]
    cls_train = cls_all[4000:]
    srl_dev = json.loads(data.srl_dev.read_text())
    print(f"[stage2b] cls_train={len(cls_train):,}, cls_dev={len(cls_dev):,}")

    loader = DataLoader(
        _CLSFineTuneDataset(cls_train, tokenizer, cfg.max_length),
        batch_size=params.batch_size, shuffle=True, collate_fn=_collate,
        num_workers=2, pin_memory=True,
    )

    cls_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        cls_params, lr=params.lr_cls_s2b, weight_decay=params.weight_decay,
    )
    total_steps = max(1, params.epochs_s2b * len(loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * 0.1), total_steps,
    )

    # Modules to keep in eval() so dropout stays off in the frozen path
    frozen_modules = [
        model.encoder,
        model.pos_sm, model.pos_head,
        model.ner_sm, model.ner_lstm, model.ner_proj, model.ner_head,
        model.dep_sm, model.dep_proj, model.dep_lstm, model.dep_lstm_proj, model.dep_biaff,
        model.srl_sm, model.srl_interaction_proj, model.srl_lstm, model.srl_proj, model.srl_head,
    ]

    best_cls = 0.0
    history: list[dict[str, Any]] = []
    final_path = cfg.output_dir / "model.pt"

    for epoch in range(params.epochs_s2b):
        model.train()
        for m in frozen_modules:
            m.eval()
        for b in tqdm(loader, desc=f"S2b E{epoch + 1}/{params.epochs_s2b}", leave=False):
            o = model(
                b["input_ids"].to(device), b["attention_mask"].to(device),
                torch.zeros(b["input_ids"].size(0), dtype=torch.long, device=device),
            )[5]
            loss = F.cross_entropy(o, b["labels"].to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        srl_arg, srl_vit = _eval_srl_dev(model, tokenizer, srl_dev, device)
        cls_acc = _eval_cls_dev(model, tokenizer, cls_dev, device)
        history.append({"epoch": epoch + 1, "srl_vit_frozen": srl_vit, "cls": cls_acc})
        marker = ""
        if cls_acc > best_cls:
            best_cls = cls_acc
            # Save without train-only PKD projections
            model.strip_pkd_projections()
            torch.save(model.state_dict(), final_path)
            tokenizer.save_pretrained(cfg.output_dir)
            marker = f"  ★ BEST CLS ({cls_acc:.4f})"
        print(f"  S2b E{epoch + 1} — srl(frozen) {srl_vit:.4f}, cls {cls_acc:.4f}{marker}")

    return {"best_cls": best_cls, "history": history}


# ── Full benchmark eval (after pipeline completes) ─────────────────

def evaluate_5_heads(model: StudentCascadeV5, tokenizer: Any, device: str,
                     batch_size: int = 32) -> TrialResult:
    """Run all 5-head benchmarks against an in-memory model."""
    from student_benchmark_standard import (  # noqa: E402
        PyTorchBackend, run_benchmarks,
    )
    from datasets import load_dataset

    bench_dir = REPO / "data" / "benchmarks"
    ud_test = json.loads((bench_dir / "ud_ewt_test.json").read_text())
    srl_test = json.loads((bench_dir / "propbank_srl_test.json").read_text())
    dd_examples = json.loads((bench_dir / "dailydialog_test.json").read_text())
    conll = load_dataset("eriktks/conll2003")
    conll_test = conll["test"]
    conll_tag_names = conll_test.features["ner_tags"].feature.names
    internal_cls_dev: list = []

    backend = PyTorchBackend(model, device)
    model.eval()
    results = run_benchmarks(
        backend, tokenizer,
        conll_test, conll_tag_names, ud_test, srl_test,
        dd_examples, internal_cls_dev,
        batch_size,
    )
    model.train()

    return TrialResult(
        pos=results["pos"]["score"],
        ner=results["ner"]["score"],
        dep_uas=results["dep"]["score"],
        dep_las=results["dep"].get("las", 0.0),
        srl=results["srl"]["score"],
        cls=results.get("cls_internal", {}).get("score") or
            results.get("cls", {}).get("score") or 0.0,
    )


# ── Sweep adapter (Stage 1 only — sweeps don't run Stage 2) ────────

def train_one_trial(params: TrialParams, cfg: StageConfig) -> TrialResult:
    """Single Stage-1 trial for the Optuna sweep driver.

    Implements the ``shared.hpo_trial.TrialRunner`` contract: takes a
    sampled hyperparameter point, trains Stage 1 only, evaluates on the
    full benchmark suite, and returns per-head scores.
    """
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, info = train_stage1(params, cfg)
    result = evaluate_5_heads(model, tokenizer, device)
    result.train_loss = info.get("stage1_avg_loss")
    result.epoch = params.epochs
    return result


def make_runner(*, student_encoder: str, distillation_data: str, output_dir: str,
                device: str | None = None):
    cfg = StageConfig(
        student_encoder=student_encoder,
        distillation_data=Path(distillation_data),
        output_dir=Path(output_dir),
        device=device,
    )

    def runner(params: TrialParams) -> TrialResult:
        return train_one_trial(params, cfg)

    return runner


# ── End-to-end pipeline ─────────────────────────────────────────────

def run_full_pipeline(params: TrialParams, cfg: StageConfig,
                      stage2_root: Path,
                      silver_limit: int = 80_000,
                      stages: tuple[str, ...] = ("1", "2a", "2b"),
                      ) -> dict[str, Any]:
    """Stage 1 → 2a → 2b end-to-end. Saves checkpoints under ``cfg.output_dir``."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log: dict[str, Any] = {"stages_run": list(stages)}

    model = tokenizer = None
    if "1" in stages:
        model, tokenizer, info = train_stage1(params, cfg)
        # Save Stage 1 checkpoint with PKD projections still attached so
        # later runs can resume from it; the published model.pt comes
        # from Stage 2b after stripping.
        torch.save(model.state_dict(), cfg.output_dir / "model_s1.pt")
        log["stage1"] = info
    else:
        # Resume: load model from Stage 1 checkpoint
        model, tokenizer, info = _build_model(
            cfg.student_encoder, device,
            teacher_layers=cfg.teacher_layers if params.use_pkd else (),
        )
        s1_path = cfg.output_dir / "model_s1.pt"
        if not s1_path.exists():
            raise FileNotFoundError(
                f"Stage 1 checkpoint missing: {s1_path}. Run --stages 1 first."
            )
        model.load_state_dict(torch.load(s1_path, map_location="cpu"), strict=False)
        model.to(device)
        log["stage1"] = {"loaded_from": str(s1_path)}

    if "2a" in stages or "2b" in stages:
        data = Stage2DataPaths.from_root(stage2_root)
        data.assert_exist()

        if "2a" in stages:
            log["stage2a"] = train_stage2a(
                model, tokenizer, params, cfg, data, silver_limit=silver_limit,
            )
        if "2b" in stages:
            log["stage2b"] = train_stage2b(model, tokenizer, params, cfg, data)
        else:
            # If only S2a runs (no S2b), still write final published checkpoint
            model.strip_pkd_projections()
            torch.save(model.state_dict(), cfg.output_dir / "model.pt")
            tokenizer.save_pretrained(cfg.output_dir)
    elif "1" in stages:
        # Stage 1 only — write a published-ish checkpoint for benchmarking
        model.strip_pkd_projections()
        torch.save(model.state_dict(), cfg.output_dir / "model.pt")
        tokenizer.save_pretrained(cfg.output_dir)

    # Final benchmark + metadata
    result = evaluate_5_heads(model, tokenizer, device)
    log["final_scores"] = result.to_dict()

    meta = {
        "version": "v2",
        "encoder": _encoder_friendly_name(cfg.student_encoder),
        "transformers": "5.6.2",
        "training": {"params": params.to_dict()},
        "log": log,
    }
    (cfg.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    (cfg.output_dir / "trial_result.json").write_text(json.dumps(result.to_dict(), indent=2))

    # Ship the canonical label maps alongside each model so inference
    # clients (Rust, ONNX runners, JS) don't have to hardcode them.
    label_maps_src = REPO / "models" / "label_maps.json"
    if label_maps_src.exists():
        shutil.copyfile(label_maps_src, cfg.output_dir / "label_maps.json")

    return log


def _encoder_friendly_name(hf_id: str) -> str:
    table = {
        "microsoft/deberta-v3-xsmall": "DeBERTa-v3-xsmall",
        "microsoft/deberta-v3-small": "DeBERTa-v3-small",
        "microsoft/deberta-v3-base": "DeBERTa-v3-base",
        "microsoft/deberta-v3-large": "DeBERTa-v3-large",
    }
    return table.get(hf_id, hf_id.split("/")[-1])


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--student", required=True,
                    help="HF encoder id, e.g. microsoft/deberta-v3-xsmall")
    ap.add_argument("--distillation-data", default="data/distillation",
                    help="Directory containing shard_*.parquet (Stage 1)")
    ap.add_argument("--stage2-data", default="data/prepared/kniv-deberta-cascade",
                    help="Directory with srl_train.json, srl_dev.json, "
                         "cls_sgd_mwoz_train.json (Stages 2a/2b)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--stages", default="1,2a,2b",
                    help="Comma-separated subset of {1, 2a, 2b}")

    # Hyperparams (defaults match published v2 xsmall recipe)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--epochs-s2a", type=int, default=8)
    ap.add_argument("--epochs-s2b", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--lr-encoder-s2a", type=float, default=3e-6)
    ap.add_argument("--lr-heads", type=float, default=1e-3)
    ap.add_argument("--lr-cls-s2b", type=float, default=5e-4)
    ap.add_argument("--temperature", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--alpha-hard", type=float, default=0.5)
    ap.add_argument("--gamma-hidden", type=float, default=0.3)
    ap.add_argument("--hidden-mult-pred", type=float, default=0.5)
    ap.add_argument("--rdrop-gamma", type=float, default=0.2)
    ap.add_argument("--no-pkd", action="store_true",
                    help="Disable PKD-style hidden-state distillation")
    ap.add_argument("--no-rdrop", action="store_true",
                    help="Disable R-Drop second-forward consistency loss")
    ap.add_argument("--silver-limit", type=int, default=80_000,
                    help="Max teacher-silver SRL examples in Stage 2a")

    args = ap.parse_args()

    params = TrialParams(
        lr_encoder=args.lr_encoder,
        lr_heads=args.lr_heads,
        lr_encoder_s2a=args.lr_encoder_s2a,
        lr_cls_s2b=args.lr_cls_s2b,
        temperature=args.temperature,
        alpha=args.alpha,
        alpha_hard=args.alpha_hard,
        gamma_hidden=args.gamma_hidden,
        hidden_mult_pred=args.hidden_mult_pred,
        rdrop_gamma=args.rdrop_gamma,
        use_pkd=not args.no_pkd,
        use_rdrop=not args.no_rdrop,
        epochs=args.epochs,
        epochs_s2a=args.epochs_s2a,
        epochs_s2b=args.epochs_s2b,
        batch_size=args.batch_size,
    )
    cfg = StageConfig(
        student_encoder=args.student,
        distillation_data=Path(args.distillation_data),
        output_dir=Path(args.output),
        device=args.device,
    )
    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())

    print(f"Training {args.student} → {args.output}  (stages: {stages})")
    print(f"  params: {params.to_dict()}")
    log = run_full_pipeline(
        params, cfg, Path(args.stage2_data),
        silver_limit=args.silver_limit, stages=stages,
    )
    print()
    print("=" * 60)
    print(f"Final scores: {json.dumps(log['final_scores'], indent=2)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

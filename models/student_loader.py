"""Shared loader for distilled student models in the kniv cascade family.

Loads any of the student model directories (xsmall / small / base) from a
flat PyTorch state_dict produced by the distillation training pipeline.

Usage:
    from student_loader import load_student
    model, tokenizer, info = load_student("models/kniv-deberta-nlp-base-en-xsmall")

The student architecture (StudentCascadeV5) is identical across sizes;
only the encoder dimensions differ. This loader auto-detects the encoder
from `metadata.json` in the model directory.
"""
from __future__ import annotations
import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


# ── Label sets (must match training pipeline) ───────────────────────
POS_LABELS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
              "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
NER_LABELS = [
    "O", "B-PERSON", "I-PERSON", "B-NORP", "I-NORP", "B-FAC", "I-FAC",
    "B-ORG", "I-ORG", "B-GPE", "I-GPE", "B-LOC", "I-LOC", "B-PRODUCT", "I-PRODUCT",
    "B-EVENT", "I-EVENT", "B-WORK_OF_ART", "I-WORK_OF_ART", "B-LAW", "I-LAW",
    "B-LANGUAGE", "I-LANGUAGE", "B-DATE", "I-DATE", "B-TIME", "I-TIME",
    "B-PERCENT", "I-PERCENT", "B-MONEY", "I-MONEY", "B-QUANTITY", "I-QUANTITY",
    "B-ORDINAL", "I-ORDINAL", "B-CARDINAL", "I-CARDINAL",
]
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
CLS_LABELS = ["inform", "request", "question", "confirm", "reject",
              "offer", "social", "status"]
DEPREL_LIST = [
    "root", "acl", "acl:relcl", "advcl", "advcl:relcl", "advmod", "amod", "appos",
    "aux", "aux:pass", "case", "cc", "cc:preconj", "ccomp", "compound", "compound:prt",
    "conj", "cop", "csubj", "csubj:outer", "csubj:pass", "dep", "det", "det:predet",
    "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "list", "mark",
    "nmod", "nmod:desc", "nmod:npmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubj:outer",
    "nsubj:pass", "nummod", "obj", "obl", "obl:agent", "obl:npmod", "obl:tmod",
    "orphan", "parataxis", "punct", "reparandum", "vocative", "xcomp",
]
N_POS, N_NER, N_SRL, N_DEP, N_CLS = (len(POS_LABELS), len(NER_LABELS),
                                      len(SRL_TAGS), len(DEPREL_LIST), len(CLS_LABELS))

# Teacher hidden dim — only used by PKD-style hidden-state distillation. The
# teacher (DeBERTa-v3-large) has 1024d; if this ever changes, the dataset
# generator and training loop must agree.
H_TEACHER = 1024


# ── Model components ────────────────────────────────────────────────
class ScalarMix(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, layers):
        w = torch.softmax(self.weights, dim=0)
        return self.scale * sum(wi * li for wi, li in zip(w, layers))


class AttentionPool(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.attn = nn.Linear(H, 1)

    def forward(self, hidden, mask):
        scores = self.attn(hidden).squeeze(-1).masked_fill(~mask.bool(), -1e9)
        return (hidden * torch.softmax(scores, -1).unsqueeze(-1)).sum(1)


class Biaffine(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_dim, in_dim + 1, in_dim + 1))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h_dep, h_head):
        B, S, D = h_dep.size()
        ones_dep = torch.ones(B, S, 1, device=h_dep.device, dtype=h_dep.dtype)
        ones_head = torch.ones(B, S, 1, device=h_head.device, dtype=h_head.dtype)
        h_dep = torch.cat([h_dep, ones_dep], -1)
        h_head = torch.cat([h_head, ones_head], -1)
        scores = torch.einsum("bxi,oij,byj->boxy", h_dep, self.weight, h_head)
        if scores.size(1) == 1:
            return scores.squeeze(1).contiguous()
        return scores.permute(0, 2, 3, 1).contiguous()


class BiaffineDEPHead(nn.Module):
    def __init__(self, H, arc_dim, label_dim, num_labels, dropout=0.1):
        super().__init__()
        self.arc_dep = nn.Sequential(
            nn.Linear(H, arc_dim), nn.LayerNorm(arc_dim), nn.GELU(), nn.Dropout(dropout))
        self.arc_head = nn.Sequential(
            nn.Linear(H, arc_dim), nn.LayerNorm(arc_dim), nn.GELU(), nn.Dropout(dropout))
        self.label_dep = nn.Sequential(
            nn.Linear(H, label_dim), nn.LayerNorm(label_dim), nn.GELU(), nn.Dropout(dropout))
        self.label_head = nn.Sequential(
            nn.Linear(H, label_dim), nn.LayerNorm(label_dim), nn.GELU(), nn.Dropout(dropout))
        self.biaffine_arc = Biaffine(arc_dim, 1)
        self.biaffine_label = Biaffine(label_dim, num_labels)

    def forward(self, hidden):
        return (self.biaffine_arc(self.arc_dep(hidden), self.arc_head(hidden)),
                self.biaffine_label(self.label_dep(hidden), self.label_head(hidden)))


class StudentCascadeV5(nn.Module):
    """Distilled student cascade: POS -> NER -> DEP -> SRL + CLS, sharing one encoder pass.

    Auto-scales internal head dimensions to the encoder hidden size.
    """

    def __init__(self, encoder, H: int, num_layers: int):
        super().__init__()
        self.encoder = encoder
        self.pred_embedding = nn.Embedding(2, H)
        nn.init.zeros_(self.pred_embedding.weight)
        NL = num_layers

        # POS
        self.pos_sm = ScalarMix(NL)
        self.pos_head = nn.Linear(H, N_POS)

        # NER
        self.ner_sm = ScalarMix(NL)
        lstm_h = max(H // 4, 64)
        self.ner_lstm = nn.LSTM(H, lstm_h, bidirectional=True, batch_first=True)
        self.ner_proj = nn.Linear(lstm_h * 2, H)
        self.ner_head = nn.Sequential(
            nn.LayerNorm(H + N_POS), nn.Linear(H + N_POS, H),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(H, N_NER))

        # DEP — BiLSTM before biaffine
        self.dep_sm = ScalarMix(NL)
        in_dim = H + N_POS + N_NER
        self.dep_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, H), nn.GELU())
        dep_lstm_h = max(H // 4, 64)
        self.dep_lstm = nn.LSTM(H, dep_lstm_h, bidirectional=True, batch_first=True)
        self.dep_lstm_proj = nn.Linear(dep_lstm_h * 2, H)
        arc_dim, label_dim = max(H // 2, 64), max(H // 8, 32)
        self.dep_biaff = BiaffineDEPHead(H, arc_dim, label_dim, N_DEP)

        # SRL — predicate interaction + DEP cascade + POS cascade + wider BiLSTM
        srl_input_dim = H * 4 + N_POS + N_DEP
        self.srl_sm = ScalarMix(NL)
        self.srl_interaction_proj = nn.Sequential(
            nn.LayerNorm(srl_input_dim), nn.Linear(srl_input_dim, H),
            nn.GELU(), nn.Dropout(0.1))
        srl_lstm_h = max(H // 2, 96)
        self.srl_lstm = nn.LSTM(H, srl_lstm_h, bidirectional=True, batch_first=True)
        self.srl_proj = nn.Linear(srl_lstm_h * 2, H)
        self.srl_head = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(H, H), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(H, N_SRL))

        # CLS
        self.cls_sm = ScalarMix(NL)
        self.cls_pool = AttentionPool(H)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(H), nn.Linear(H, H * 2), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(H * 2, H), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(H, N_CLS))

        # PKD-style hidden-state projection heads — train-only, opt-in via
        # ``enable_pkd_projections``. Stripped from the published checkpoint
        # by ``strip_pkd_projections`` before save.
        self._H = H
        self._teacher_layers: tuple[int, ...] = ()
        self.distill_projections: nn.ModuleDict | None = None

    # ── PKD projections (train-only) ──────────────────────────────────
    def enable_pkd_projections(self, teacher_layers=(12, 18, 24),
                               teacher_hidden_dim: int = H_TEACHER) -> None:
        """Add Linear projections from student hidden dim → teacher dim.

        One per teacher layer in ``teacher_layers``. These are needed only
        during PKD-style hidden-state distillation; call
        ``strip_pkd_projections`` before saving the inference checkpoint.
        """
        self._teacher_layers = tuple(teacher_layers)
        self.distill_projections = nn.ModuleDict({
            f"l{l}": nn.Linear(self._H, teacher_hidden_dim, bias=False)
            for l in self._teacher_layers
        })
        for proj in self.distill_projections.values():
            nn.init.xavier_uniform_(proj.weight, gain=0.5)
        # Move to same device as the rest of the model
        device = next(self.parameters()).device
        self.distill_projections = self.distill_projections.to(device)

    def strip_pkd_projections(self) -> None:
        """Drop projection heads so the inference checkpoint has no dead weights."""
        self.distill_projections = None
        self._teacher_layers = ()

    def map_teacher_to_student_layer(self, teacher_layer: int,
                                     n_student_layers: int,
                                     n_teacher_layers: int = 24) -> int:
        """Proportional mapping: teacher layer L → student layer.

        Matches the Colab v2 recipe: ``round(n_student * tl / n_teacher)``,
        clamped to ≥ 1 to skip the embedding layer (index 0).
        """
        return max(1, round(n_student_layers * teacher_layer / n_teacher_layers))

    def forward(self, input_ids, attention_mask, predicate_idx,
                return_hidden_states: bool = False):
        B, S = input_ids.size()
        if hasattr(self.encoder, "embeddings"):
            emb = self.encoder.embeddings(input_ids)
            indicator = torch.zeros(B, S, dtype=torch.long, device=input_ids.device)
            indicator.scatter_(1, predicate_idx.unsqueeze(1), 1)
            emb = emb + self.pred_embedding(indicator)
            enc_out = self.encoder.encoder(emb, attention_mask, output_hidden_states=True)
        else:
            enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                   output_hidden_states=True)
        layers = list(enc_out.hidden_states)

        # POS
        pos_logits = self.pos_head(self.pos_sm(layers))
        pos_p = torch.softmax(pos_logits, -1)

        # NER
        ner_h = self.ner_sm(layers)
        lo, _ = self.ner_lstm(ner_h)
        ner_logits = self.ner_head(torch.cat([self.ner_proj(lo) + ner_h, pos_p.detach()], -1))
        ner_p = torch.softmax(ner_logits, -1)

        # DEP
        dep_h = self.dep_sm(layers)
        dep_proj = self.dep_proj(torch.cat([dep_h, pos_p.detach(), ner_p.detach()], -1))
        dep_lo, _ = self.dep_lstm(dep_proj)
        dep_adapted = self.dep_lstm_proj(dep_lo) + dep_proj
        arc_scores, label_scores = self.dep_biaff(dep_adapted)

        # DEP cascade features for SRL
        pred_heads = arc_scores.argmax(dim=-1)
        ph = pred_heads.clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(B, S, 1, N_DEP)
        dep_label_at_head = label_scores.gather(2, ph).squeeze(2)
        dep_rel_probs = torch.softmax(dep_label_at_head, dim=-1).detach()

        # SRL
        srl_h = self.srl_sm(layers)
        pred_idx_expanded = predicate_idx.view(B, 1, 1).expand(B, 1, srl_h.size(-1))
        h_pred = srl_h.gather(1, pred_idx_expanded).expand_as(srl_h)
        h_i = srl_h
        srl_features = torch.cat([
            h_i, h_pred, h_i * h_pred, (h_i - h_pred).abs(),
            pos_p.detach(), dep_rel_probs,
        ], dim=-1)
        srl_projected = self.srl_interaction_proj(srl_features)
        srl_lo, _ = self.srl_lstm(srl_projected)
        srl_adapted = self.srl_proj(srl_lo) + srl_projected
        srl_logits = self.srl_head(srl_adapted)

        # CLS
        cls_h = self.cls_sm(layers)
        cls_logits = self.cls_head(self.cls_pool(cls_h, attention_mask))

        if return_hidden_states:
            n_student = len(layers) - 1  # exclude embedding layer
            mapped: dict[int, torch.Tensor] = {}
            for tl in self._teacher_layers:
                mapped[tl] = layers[self.map_teacher_to_student_layer(tl, n_student)]
            return (pos_logits, ner_logits, arc_scores, label_scores,
                    srl_logits, cls_logits, mapped)
        return pos_logits, ner_logits, arc_scores, label_scores, srl_logits, cls_logits


# ── Viterbi BIO decoder (used for NER and SRL at inference) ─────────
def viterbi_decode(logits: torch.Tensor, tag_list: list[str]) -> list[int]:
    """Constrained BIO Viterbi: I-X may only follow B-X or I-X of same role."""
    num_tags, seq_len = logits.size(-1), logits.size(0)
    log_probs = torch.log_softmax(logits, dim=-1)
    tag2id = {t: i for i, t in enumerate(tag_list)}
    allowed = torch.ones(num_tags, num_tags, dtype=torch.bool)
    for j in range(num_tags):
        if tag_list[j].startswith("I-"):
            role = tag_list[j][2:]
            b_tag = tag2id.get(f"B-{role}", -1)
            for i in range(num_tags):
                if i != b_tag and i != j:
                    allowed[i][j] = False
    NEG_INF = -1e9
    viterbi = torch.full((seq_len, num_tags), NEG_INF)
    backptr = torch.zeros(seq_len, num_tags, dtype=torch.long)
    viterbi[0] = log_probs[0]
    for t in range(1, seq_len):
        for j in range(num_tags):
            scores = viterbi[t - 1].clone()
            scores[~allowed[:, j]] = NEG_INF
            best = scores.argmax()
            viterbi[t, j] = scores[best] + log_probs[t, j]
            backptr[t, j] = best
    path = [0] * seq_len
    path[-1] = viterbi[-1].argmax().item()
    for t in range(seq_len - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]].item()
    return path


# ── Loader ──────────────────────────────────────────────────────────
def load_student(model_dir: str | Path,
                 checkpoint: str = "model.pt",
                 device: str | torch.device | None = None,
                 strict: bool = True):
    """Load a student model from a directory.

    Args:
        model_dir: Path to a kniv-deberta-nlp-base-en-{xsmall,small,base} dir.
        checkpoint: Filename of the .pt to load (default: "model.pt").
        device: Target device (default: cuda if available, else cpu).
        strict: Pass-through to load_state_dict.

    Returns:
        (model, tokenizer, info) where:
            model — StudentCascadeV5 in eval mode on device
            tokenizer — Hugging Face tokenizer
            info — dict with metadata fields (encoder, hidden_dim, layers, params, ...)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model_dir = Path(model_dir)
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No metadata.json in {model_dir} — can't determine encoder.")
    with open(meta_path) as f:
        meta = json.load(f)

    encoder_name_map = {
        "DeBERTa-v3-xsmall": "microsoft/deberta-v3-xsmall",
        "DeBERTa-v3-small": "microsoft/deberta-v3-small",
        "DeBERTa-v3-base": "microsoft/deberta-v3-base",
        "DeBERTa-v3-large": "microsoft/deberta-v3-large",
    }
    encoder_id = encoder_name_map.get(meta["encoder"])
    if encoder_id is None:
        raise ValueError(f"Unknown encoder in metadata.json: {meta['encoder']!r}")

    # Load tokenizer (prefer model_dir which has the trained tokenizer files)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Build model with a fresh pretrained encoder (weights will be overwritten by state_dict)
    encoder = AutoModel.from_pretrained(encoder_id).float()
    H = encoder.config.hidden_size
    NL = encoder.config.num_hidden_layers + 1
    model = StudentCascadeV5(encoder, H, NL)

    # Load state dict
    ckpt_path = model_dir / checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        print(f"[student_loader] missing keys ({len(missing)}): {missing[:3]}{'...' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"[student_loader] unexpected keys ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")

    model = model.to(device).eval()

    info = {
        "encoder": meta["encoder"],
        "hidden_dim": meta.get("encoder_hidden_dim", H),
        "layers": meta.get("encoder_layers", NL - 1),
        "params_total_millions": meta.get("params_total_millions"),
        "checkpoint": str(ckpt_path),
        "device": str(device),
    }
    return model, tokenizer, info


__all__ = [
    "StudentCascadeV5",
    "ScalarMix", "AttentionPool", "Biaffine", "BiaffineDEPHead",
    "viterbi_decode", "load_student",
    "POS_LABELS", "NER_LABELS", "SRL_TAGS", "CLS_LABELS", "DEPREL_LIST",
    "N_POS", "N_NER", "N_SRL", "N_DEP", "N_CLS",
    "H_TEACHER",
]

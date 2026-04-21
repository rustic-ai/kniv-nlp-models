"""Multi-task NLP model with cascaded heads, trained stage-by-stage.

Architecture (cascaded):
    DeBERTa-v3 encoder (shared)
      1. POS head:  Linear(H, 17)                      -- always present
      2. NER head:  Linear(H+17, 37)                   -- POS → NER cascade
      3. DEP head:  Linear(H+17+37, dep_labels)        -- POS+NER → DEP cascade
      4. CLS head:  Linear(H+17+37, cls_labels)        -- pooled POS+NER → CLS

Heads are added progressively during staged training:
  Stage 1: POS only → train encoder + POS
  Stage 2: +NER    → freeze POS, train NER with POS cascade
  Stage 3: +DEP    → freeze POS+NER, train DEP
  Stage 4: +CLS    → freeze POS+NER+DEP, train CLS
  Stage 5: joint   → fine-tune everything end-to-end

Soft probabilities flow between heads. forward() only computes
active heads (those that are not None).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class MultiTaskNLPModel(nn.Module):
    """Multi-task model with progressive cascade heads."""

    def __init__(
        self,
        encoder_name: str,
        pos_labels: list[str],
        ner_labels: list[str] | None = None,
        dep_labels: list[str] | None = None,
        cls_labels: list[str] | None = None,
        dropout: float = 0.1,
        _empty_encoder: bool = False,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(encoder_name)
        if _empty_encoder:
            # Create encoder architecture without downloading pretrained weights
            # (used when loading all weights from a checkpoint)
            self.encoder = AutoModel.from_config(self.config)
        else:
            self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        # POS: always present (anchor of cascade)
        self.pos_head = nn.Linear(hidden_size, len(pos_labels))
        self.pos_labels = pos_labels

        # NER: optional, cascaded from POS
        self.ner_head = None
        self.ner_labels = ner_labels
        if ner_labels is not None:
            self.ner_head = nn.Linear(hidden_size + len(pos_labels), len(ner_labels))

        # DEP: optional, cascaded from POS + NER
        self.dep_head = None
        self.dep_labels = dep_labels
        if dep_labels is not None and ner_labels is not None:
            self.dep_head = nn.Linear(
                hidden_size + len(pos_labels) + len(ner_labels), len(dep_labels),
            )

        # CLS: optional, cascaded from pooled POS + NER
        self.cls_head = None
        self.cls_labels = cls_labels
        if cls_labels is not None and ner_labels is not None:
            self.cls_head = nn.Linear(
                hidden_size + len(pos_labels) + len(ner_labels), len(cls_labels),
            )

    def _make_mlp_head(self, in_dim: int, out_dim: int) -> nn.Sequential:
        """MLP head: LayerNorm → Linear → GELU → Dropout → Linear."""
        hidden_size = self.config.hidden_size
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout.p),
            nn.Linear(hidden_size, out_dim),
        )

    def add_head(self, head_name: str, labels: list[str], head_type: str = "linear"):
        """Add a new cascade head after loading a previous-stage checkpoint.

        Args:
            head_name: "ner", "dep", or "cls"
            labels: list of label strings
            head_type: "linear" or "mlp"
        """
        hidden_size = self.config.hidden_size
        num_pos = len(self.pos_labels)
        make = self._make_mlp_head if head_type == "mlp" else nn.Linear

        if head_name == "ner":
            self.ner_labels = labels
            in_dim = hidden_size + num_pos
            self.ner_head = make(in_dim, len(labels))
        elif head_name == "dep":
            assert self.ner_labels is not None, "DEP requires NER head in cascade"
            self.dep_labels = labels
            num_ner = len(self.ner_labels)
            in_dim = hidden_size + num_pos + num_ner
            self.dep_head = make(in_dim, len(labels))
        elif head_name == "cls":
            assert self.ner_labels is not None, "CLS requires NER head in cascade"
            self.cls_labels = labels
            num_ner = len(self.ner_labels)
            in_dim = hidden_size + num_pos + num_ner
            self.cls_head = make(in_dim, len(labels))
        else:
            raise ValueError(f"Unknown head: {head_name}")

    @classmethod
    def load_from_teacher(cls, teacher_dir: str, encoder_name: str) -> "MultiTaskNLPModel":
        """Load encoder + POS head from an existing (non-cascade) teacher.

        Drops NER/DEP/CLS heads — only keeps encoder weights and POS head.
        """
        path = Path(teacher_dir)
        with open(path / "label_maps.json") as f:
            label_maps = json.load(f)

        # Create model shell without downloading pretrained weights
        model = cls(
            encoder_name=encoder_name,
            pos_labels=label_maps["pos_labels"],
            _empty_encoder=True,
        )

        # Load teacher weights — filter to only keys the model expects
        teacher_state = torch.load(path / "model.pt", weights_only=True)
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in teacher_state.items() if k in model_keys}
        dropped = [k for k in teacher_state if k not in model_keys]
        model.load_state_dict(filtered, strict=True)

        # Verify a sample weight actually loaded
        sample_key = "encoder.encoder.layer.0.attention.self.query_proj.weight"
        if sample_key in filtered:
            match = torch.equal(model.state_dict()[sample_key].cpu(), filtered[sample_key].cpu())
            print(f"  Encoder weight verification: {'OK' if match else 'FAILED'}", flush=True)

        print(f"  Loaded {len(filtered)} params, dropped {len(dropped)} "
              f"({set(k.split('.')[0] for k in dropped)})", flush=True)
        return model

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Cascaded forward pass — only computes active heads.

        Returns dict with keys for each active head:
            pos_logits: [batch, seq_len, num_pos]       (always)
            ner_logits: [batch, seq_len, num_ner]        (if ner_head exists)
            dep_logits: [batch, seq_len, num_dep]        (if dep_head exists)
            cls_logits: [batch, num_cls]                 (if cls_head exists)
        """
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.dropout(hidden)

        result = {}

        # 1. POS (always)
        pos_logits = self.pos_head(hidden)
        result["pos_logits"] = pos_logits

        # 2. NER (cascade from POS)
        if self.ner_head is not None:
            pos_probs = F.softmax(pos_logits, dim=-1)
            ner_logits = self.ner_head(torch.cat([hidden, pos_probs], dim=-1))
            result["ner_logits"] = ner_logits

            # 3. DEP (cascade from POS + NER)
            if self.dep_head is not None:
                ner_probs = F.softmax(ner_logits, dim=-1)
                dep_logits = self.dep_head(torch.cat([hidden, pos_probs, ner_probs], dim=-1))
                result["dep_logits"] = dep_logits

            # 4. CLS (cascade from pooled POS + NER)
            if self.cls_head is not None:
                ner_probs = F.softmax(ner_logits, dim=-1)
                mask = attention_mask.unsqueeze(-1).float()
                token_count = mask.sum(dim=1).clamp(min=1)
                pos_pooled = (pos_probs * mask).sum(dim=1) / token_count
                ner_pooled = (ner_probs * mask).sum(dim=1) / token_count
                cls_hidden = hidden[:, 0, :]
                cls_logits = self.cls_head(torch.cat([cls_hidden, pos_pooled, ner_pooled], dim=-1))
                result["cls_logits"] = cls_logits

        return result

    def save(self, output_dir: str):
        """Save model weights and label maps."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        state = {k: v.float() if v.is_floating_point() else v for k, v in self.state_dict().items()}
        torch.save(state, path / "model.pt")
        self.encoder.config.save_pretrained(path)

        label_maps = {"pos_labels": self.pos_labels}
        if self.ner_labels is not None:
            label_maps["ner_labels"] = self.ner_labels
        if self.dep_labels is not None:
            label_maps["dep_labels"] = self.dep_labels
        if self.cls_labels is not None:
            label_maps["cls_labels"] = self.cls_labels

        with open(path / "label_maps.json", "w") as f:
            json.dump(label_maps, f, indent=2)

    @classmethod
    def load(cls, model_dir: str, encoder_name: str) -> "MultiTaskNLPModel":
        """Load a saved model (with whatever heads were active when saved)."""
        path = Path(model_dir)
        with open(path / "label_maps.json") as f:
            label_maps = json.load(f)

        # Create model with POS only first
        model = cls(
            encoder_name=encoder_name,
            pos_labels=label_maps["pos_labels"],
            _empty_encoder=True,
        )

        # Detect head types from state_dict keys and add them
        state_dict = torch.load(path / "model.pt", weights_only=True)
        for head_name in ["ner", "dep", "cls"]:
            labels = label_maps.get(f"{head_name}_labels")
            if labels is None:
                continue
            # MLP heads have keys like "ner_head.0.weight", Linear has "ner_head.weight"
            is_mlp = f"{head_name}_head.0.weight" in state_dict
            model.add_head(head_name, labels, head_type="mlp" if is_mlp else "linear")

        model.load_state_dict(state_dict)
        return model

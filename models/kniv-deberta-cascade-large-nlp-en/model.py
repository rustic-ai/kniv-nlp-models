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
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(encoder_name)
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

    def add_head(self, head_name: str, labels: list[str]):
        """Add a new cascade head after loading a previous-stage checkpoint."""
        hidden_size = self.config.hidden_size
        num_pos = len(self.pos_labels)

        if head_name == "ner":
            self.ner_labels = labels
            self.ner_head = nn.Linear(hidden_size + num_pos, len(labels))
        elif head_name == "dep":
            assert self.ner_labels is not None, "DEP requires NER head in cascade"
            self.dep_labels = labels
            num_ner = len(self.ner_labels)
            self.dep_head = nn.Linear(hidden_size + num_pos + num_ner, len(labels))
        elif head_name == "cls":
            assert self.ner_labels is not None, "CLS requires NER head in cascade"
            self.cls_labels = labels
            num_ner = len(self.ner_labels)
            self.cls_head = nn.Linear(hidden_size + num_pos + num_ner, len(labels))
        else:
            raise ValueError(f"Unknown head: {head_name}")

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

        model = cls(
            encoder_name=encoder_name,
            pos_labels=label_maps["pos_labels"],
            ner_labels=label_maps.get("ner_labels"),
            dep_labels=label_maps.get("dep_labels"),
            cls_labels=label_maps.get("cls_labels"),
        )
        model.load_state_dict(torch.load(path / "model.pt", weights_only=True))
        return model

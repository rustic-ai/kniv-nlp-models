"""Multi-task NLP model: shared encoder + cascaded MLP heads.

Architecture (cascaded, MLP + LayerNorm):
    DeBERTa-v3-small encoder (shared)
      1. POS head:  LN(H) → Linear(H, H/4) → GELU → Drop → Linear(H/4, 17)
      2. NER head:  LN(H+17) → Linear → GELU → Drop → Linear(→37)     POS → NER
      3. DEP head:  LN(H+54) → Linear → GELU → Drop → Linear(→dep)    POS+NER → DEP
      4. CLS head:  LN(H+54) → Linear → GELU → Drop → Linear(→cls)    pooled POS+NER → CLS

Cascade order: POS → NER → DEP/CLS.
Soft probabilities (not hard labels) flow between heads so gradients
propagate back through the full cascade during training.
Each head uses LayerNorm + bottleneck MLP instead of a single Linear.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class MultiTaskNLPModel(nn.Module):
    """Multi-task model with cascaded head connections."""

    def __init__(
        self,
        encoder_name: str,
        ner_labels: list[str],
        pos_labels: list[str],
        dep_labels: list[str],
        cls_labels: list[str],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        num_pos = len(pos_labels)
        num_ner = len(ner_labels)

        # Cascade: POS → NER → DEP/CLS (MLP heads with LayerNorm)
        self.pos_head = self._make_head(hidden_size, num_pos)
        self.ner_head = self._make_head(hidden_size + num_pos, num_ner)
        self.dep_head = self._make_head(hidden_size + num_pos + num_ner, len(dep_labels))
        self.cls_head = self._make_head(hidden_size + num_pos + num_ner, len(cls_labels))

        self.ner_labels = ner_labels
        self.pos_labels = pos_labels
        self.dep_labels = dep_labels
        self.cls_labels = cls_labels

    def _make_head(self, in_dim: int, out_dim: int) -> nn.Sequential:
        bottleneck = self.config.hidden_size // 4
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(self.dropout.p),
            nn.Linear(bottleneck, out_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Cascaded forward pass: POS → NER → DEP/CLS.

        Returns:
            ner_logits: [batch, seq_len, num_ner_labels]
            pos_logits: [batch, seq_len, num_pos_labels]
            dep_logits: [batch, seq_len, num_dep_labels]
            cls_logits: [batch, num_cls_labels]
        """
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.dropout(hidden)

        # 1. POS (no dependencies — anchor of the cascade)
        pos_logits = self.pos_head(hidden)
        pos_probs = F.softmax(pos_logits, dim=-1)

        # 2. NER (receives POS soft probs)
        ner_logits = self.ner_head(torch.cat([hidden, pos_probs], dim=-1))
        ner_probs = F.softmax(ner_logits, dim=-1)

        # 3. DEP (receives POS + NER soft probs)
        dep_logits = self.dep_head(torch.cat([hidden, pos_probs, ner_probs], dim=-1))

        # 4. CLS (receives masked-mean-pooled POS + NER probs)
        mask = attention_mask.unsqueeze(-1).float()
        token_count = mask.sum(dim=1).clamp(min=1)
        pos_pooled = (pos_probs * mask).sum(dim=1) / token_count
        ner_pooled = (ner_probs * mask).sum(dim=1) / token_count
        cls_hidden = hidden[:, 0, :]
        cls_logits = self.cls_head(torch.cat([cls_hidden, pos_pooled, ner_pooled], dim=-1))

        return {
            "ner_logits": ner_logits,
            "pos_logits": pos_logits,
            "dep_logits": dep_logits,
            "cls_logits": cls_logits,
        }

    def save(self, output_dir: str):
        """Save model weights and label maps."""
        import json
        from pathlib import Path

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), path / "model.pt")
        self.encoder.config.save_pretrained(path)

        with open(path / "label_maps.json", "w") as f:
            json.dump({
                "ner_labels": self.ner_labels,
                "pos_labels": self.pos_labels,
                "dep_labels": self.dep_labels,
                "cls_labels": self.cls_labels,
            }, f, indent=2)

    @classmethod
    def load(cls, model_dir: str, encoder_name: str) -> "MultiTaskNLPModel":
        """Load a saved multi-task model."""
        import json
        from pathlib import Path

        path = Path(model_dir)
        with open(path / "label_maps.json") as f:
            label_maps = json.load(f)

        model = cls(
            encoder_name=encoder_name,
            ner_labels=label_maps["ner_labels"],
            pos_labels=label_maps["pos_labels"],
            dep_labels=label_maps["dep_labels"],
            cls_labels=label_maps["cls_labels"],
        )
        model.load_state_dict(torch.load(path / "model.pt", weights_only=True))
        return model

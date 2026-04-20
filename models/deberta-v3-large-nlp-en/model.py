"""Multi-task NLP model: shared encoder + NER/POS/Dep/CLS heads.

Architecture:
    DeBERTa-v3-small encoder (shared)
      +-- NER head: Linear(hidden, ner_labels)   -- per-token BIO tags
      +-- POS head: Linear(hidden, pos_labels)   -- per-token UPOS tags
      +-- Dep head: Linear(hidden, dep_labels)   -- per-token dep2label tags
      +-- CLS head: Linear(hidden, cls_labels)   -- per-sequence intent class

Token-level heads operate on every hidden state.
Sequence-level CLS head operates on the [CLS] / first-token hidden state.
One encoder forward pass, four outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class MultiTaskNLPModel(nn.Module):
    """Multi-task model with shared transformer encoder."""

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

        # Token-level heads (one output per token)
        self.ner_head = nn.Linear(hidden_size, len(ner_labels))
        self.pos_head = nn.Linear(hidden_size, len(pos_labels))
        self.dep_head = nn.Linear(hidden_size, len(dep_labels))

        # Sequence-level head (one output per sentence, from [CLS] token)
        self.cls_head = nn.Linear(hidden_size, len(cls_labels))

        self.ner_labels = ner_labels
        self.pos_labels = pos_labels
        self.dep_labels = dep_labels
        self.cls_labels = cls_labels

    def gradient_checkpointing_enable(self, **kwargs):
        """Delegate to the encoder for HF Trainer compatibility."""
        self.encoder.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning logits for all four tasks.

        Returns:
            ner_logits: [batch, seq_len, num_ner_labels]
            pos_logits: [batch, seq_len, num_pos_labels]
            dep_logits: [batch, seq_len, num_dep_labels]
            cls_logits: [batch, num_cls_labels]
        """
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.dropout(hidden)

        # Token-level: use all hidden states
        ner_logits = self.ner_head(hidden)
        pos_logits = self.pos_head(hidden)
        dep_logits = self.dep_head(hidden)

        # Sequence-level: use first token ([CLS]) hidden state
        cls_hidden = hidden[:, 0, :]
        cls_logits = self.cls_head(cls_hidden)

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

        # Save in fp32 to avoid dtype mismatches when loading
        state = {k: v.float() if v.is_floating_point() else v for k, v in self.state_dict().items()}
        torch.save(state, path / "model.pt")
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

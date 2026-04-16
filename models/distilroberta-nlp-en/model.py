"""Multi-task NLP model: shared encoder + NER/POS/Dep classification heads.

Architecture:
    DistilRoBERTa encoder (shared)
      ├── NER head: Linear(hidden_size, num_ner_labels)
      ├── POS head: Linear(hidden_size, num_pos_labels)
      └── Dep head: Linear(hidden_size, num_dep_labels)

One forward pass through the encoder, three classification outputs.
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
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.ner_head = nn.Linear(hidden_size, len(ner_labels))
        self.pos_head = nn.Linear(hidden_size, len(pos_labels))
        self.dep_head = nn.Linear(hidden_size, len(dep_labels))

        self.ner_labels = ner_labels
        self.pos_labels = pos_labels
        self.dep_labels = dep_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning logits for all three tasks."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.dropout(hidden)
        return {
            "ner_logits": self.ner_head(hidden),
            "pos_logits": self.pos_head(hidden),
            "dep_logits": self.dep_head(hidden),
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
        )
        model.load_state_dict(torch.load(path / "model.pt", weights_only=True))
        return model

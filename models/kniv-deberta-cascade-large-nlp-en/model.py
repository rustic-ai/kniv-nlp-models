"""Multi-task NLP model with cascaded heads, trained stage-by-stage.

Architecture (cascaded):
    DeBERTa-v3 encoder (shared)
      1. POS head:  Linear(H, 17)                      -- always present
      2. NER head:  Linear(H+17, 37)                   -- POS → NER cascade
      3. DEP head:  Linear(H+17+37, dep_labels)        -- POS+NER → DEP cascade
      4. SRL head:  Biaffine(H+17+37+64, srl_labels)   -- POS+NER+DEP_proj → SRL cascade
      5. CLS head:  Linear(H+17+37, cls_labels)        -- pooled POS+NER → CLS

Heads are added progressively during staged training:
  Stage 1: POS only → train encoder + POS
  Stage 2: +NER    → freeze POS, train NER with POS cascade
  Stage 3: +DEP    → freeze POS+NER, train DEP
  Stage 3b:+SRL    → freeze POS+NER+DEP, train SRL (biaffine)
  Stage 4: +CLS    → freeze POS+NER+DEP+SRL, train CLS
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


class BiaffineSRLHead(nn.Module):
    """Biaffine SRL head — scores each token's role relative to the predicate.

    Uses DEP ROOT probabilities to soft-select the predicate token, then
    applies biaffine attention between each argument token and the predicate.

    Scoring: score_l(arg_i) = arg_i^T W_l pred + U_l arg_i + V_l pred + b_l
    where pred is the soft-selected predicate vector (weighted by P(ROOT)).
    """

    BIAFFINE_DIM = 256

    def __init__(
        self,
        in_dim: int,
        num_labels: int,
        dropout_p: float,
        root_indices: list[int],
    ):
        super().__init__()
        d = self.BIAFFINE_DIM

        # Project tokens into argument and predicate subspaces
        self.arg_mlp = nn.Sequential(
            nn.Linear(in_dim, d), nn.GELU(), nn.Dropout(dropout_p),
        )
        self.pred_mlp = nn.Sequential(
            nn.Linear(in_dim, d), nn.GELU(), nn.Dropout(dropout_p),
        )

        # Biaffine weight: W[l, d, d] for each label l
        self.biaffine = nn.Parameter(torch.zeros(num_labels, d, d))
        self.arg_linear = nn.Linear(d, num_labels)
        self.pred_linear = nn.Linear(d, num_labels)

        # Which DEP label indices correspond to ROOT (for predicate selection)
        self.register_buffer(
            "root_indices", torch.tensor(root_indices, dtype=torch.long),
        )
        nn.init.xavier_uniform_(self.biaffine)

    def forward(
        self, token_features: torch.Tensor, dep_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            token_features: [B, S, in_dim] — cascaded features for each token
            dep_probs:      [B, S, num_dep] — DEP softmax probabilities
        Returns:
            srl_logits:     [B, S, num_labels]
        """
        # Project all tokens into argument and predicate subspaces
        arg = self.arg_mlp(token_features)    # [B, S, d]
        pred = self.pred_mlp(token_features)  # [B, S, d]

        # Soft-select predicate via ROOT probabilities from DEP head
        # Sum P(label) for all dep labels that correspond to ROOT
        root_prob = dep_probs.index_select(-1, self.root_indices).sum(-1)  # [B, S]
        root_attn = root_prob / root_prob.sum(-1, keepdim=True).clamp(min=1e-8)
        pred_vec = torch.einsum("bs,bsd->bd", root_attn, pred)  # [B, d]

        # Biaffine scoring: arg^T W pred + U*arg + V*pred
        scores = torch.einsum("bsd,lde,be->bsl", arg, self.biaffine, pred_vec)
        scores = scores + self.arg_linear(arg) + self.pred_linear(pred_vec).unsqueeze(1)

        return scores


class MultiTaskNLPModel(nn.Module):
    """Multi-task model with progressive cascade heads."""

    DEP_PROJ_DIM = 64  # compress DEP probs for SRL cascade

    def __init__(
        self,
        encoder_name: str,
        pos_labels: list[str],
        ner_labels: list[str] | None = None,
        dep_labels: list[str] | None = None,
        srl_labels: list[str] | None = None,
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

        # SRL: optional, cascaded from configurable upstream heads
        self.dep_proj = None
        self.srl_head = None
        self.srl_labels = srl_labels
        self.srl_cascade = {"pos", "ner", "dep"}  # which upstream heads feed SRL input
        if srl_labels is not None and dep_labels is not None and ner_labels is not None:
            self.dep_proj = nn.Linear(len(dep_labels), self.DEP_PROJ_DIM)
            srl_in = hidden_size + len(pos_labels) + len(ner_labels) + self.DEP_PROJ_DIM
            self.srl_head = nn.Linear(srl_in, len(srl_labels))

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

    def add_head(self, head_name: str, labels: list[str], head_type: str = "linear",
                 srl_cascade: set[str] | list[str] | None = None):
        """Add a new cascade head after loading a previous-stage checkpoint.

        Args:
            head_name: "ner", "dep", "srl", or "cls"
            labels: list of label strings
            head_type: "linear", "mlp", or "biaffine" (SRL only)
            srl_cascade: which upstream heads feed SRL (subset of {"pos","ner","dep"})
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
        elif head_name == "srl":
            assert self.dep_labels is not None, "SRL requires DEP head in cascade"
            self.srl_labels = labels
            if srl_cascade is not None:
                self.srl_cascade = set(srl_cascade)
            # Compute input dim based on which upstream heads feed SRL
            in_dim = hidden_size
            if "pos" in self.srl_cascade:
                in_dim += num_pos
            if "ner" in self.srl_cascade:
                in_dim += len(self.ner_labels)
            if "dep" in self.srl_cascade:
                if self.dep_proj is None:
                    self.dep_proj = nn.Linear(len(self.dep_labels), self.DEP_PROJ_DIM)
                in_dim += self.DEP_PROJ_DIM
            if head_type == "biaffine":
                # Find dep label indices that correspond to ROOT
                root_indices = [
                    i for i, l in enumerate(self.dep_labels)
                    if l.split("@")[1] == "root"
                ]
                assert root_indices, "No ROOT labels found in dep_labels"
                self.srl_head = BiaffineSRLHead(
                    in_dim, len(labels), self.dropout.p, root_indices,
                )
            else:
                self.srl_head = make(in_dim, len(labels))
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
            srl_logits: [batch, seq_len, num_srl]        (if srl_head exists)
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

                # 4. SRL (cascade from configurable upstream heads)
                if self.srl_head is not None:
                    # When upstream is frozen, detach and create fresh tensors
                    # so dep_proj and srl_head can still receive gradients
                    dep_probs = F.softmax(dep_logits.detach(), dim=-1).requires_grad_(True)
                    srl_parts = [hidden.detach()]
                    if "pos" in self.srl_cascade:
                        srl_parts.append(pos_probs.detach())
                    if "ner" in self.srl_cascade:
                        srl_parts.append(ner_probs.detach())
                    if "dep" in self.srl_cascade:
                        srl_parts.append(self.dep_proj(dep_probs))
                    srl_input = torch.cat(srl_parts, dim=-1)
                    if isinstance(self.srl_head, BiaffineSRLHead):
                        srl_logits = self.srl_head(srl_input, dep_probs)
                    else:
                        srl_logits = self.srl_head(srl_input)
                    result["srl_logits"] = srl_logits

            # 5. CLS (cascade from pooled POS + NER)
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
        if self.srl_labels is not None:
            label_maps["srl_labels"] = self.srl_labels
            label_maps["srl_cascade"] = sorted(self.srl_cascade)
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
        # Order matters: ner → dep → srl → cls (cascade dependencies)
        state_dict = torch.load(path / "model.pt", weights_only=True)
        for head_name in ["ner", "dep", "srl", "cls"]:
            labels = label_maps.get(f"{head_name}_labels")
            if labels is None:
                continue
            # Detect head type from state_dict key patterns:
            #   Biaffine: "srl_head.biaffine"
            #   MLP:      "ner_head.0.weight" (nn.Sequential)
            #   Linear:   "ner_head.weight"
            is_biaffine = f"{head_name}_head.biaffine" in state_dict
            is_mlp = f"{head_name}_head.0.weight" in state_dict
            if is_biaffine:
                head_type = "biaffine"
            elif is_mlp:
                head_type = "mlp"
            else:
                head_type = "linear"
            kwargs = {}
            if head_name == "srl" and "srl_cascade" in label_maps:
                kwargs["srl_cascade"] = label_maps["srl_cascade"]
            model.add_head(head_name, labels, head_type=head_type, **kwargs)

        model.load_state_dict(state_dict)
        return model

"""Export kniv-deberta-nlp-base-en-large to a single ONNX model.

One model, one encoder pass, all 5 heads:
  Input:  input_ids, attention_mask, predicate_idx
  Output: pos_logits, ner_logits, arc_scores, label_scores, cls_logits, srl_logits

Usage:
    .venv/bin/python models/kniv-deberta-nlp-base-en-large/export_onnx.py
    .venv/bin/python models/kniv-deberta-nlp-base-en-large/export_onnx.py --output onnx/cascade.onnx
"""
from __future__ import annotations
import os
os.environ["TORCH_ONNX_USE_OLD_EXPORTER"] = "1"

import argparse, json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
from transformers import AutoModel, AutoTokenizer

# ── Labels ────────────────────────────────────────────────────
POS_LABELS = 17
NER_LABELS = 37
DEPREL_LABELS = 53
SRL_LABELS = 42
CLS_LABELS = 8

# ── Components ────────────────────────────────────────────────

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
    def forward(self, h_dep, h_head):
        B, S, D = h_dep.size()
        h_dep = torch.cat([h_dep, torch.ones(B, S, 1, device=h_dep.device)], -1)
        h_head = torch.cat([h_head, torch.ones(B, S, 1, device=h_head.device)], -1)
        scores = torch.einsum("bxi,oij,byj->boxy", h_dep, self.weight, h_head)
        return scores.squeeze(1).contiguous() if scores.size(1) == 1 else scores.permute(0, 2, 3, 1).contiguous()

class BiaffineDEPHead(nn.Module):
    def __init__(self, H, arc_dim=512, label_dim=128, num_labels=53):
        super().__init__()
        self.arc_dep = nn.Sequential(nn.Linear(H, arc_dim), nn.LayerNorm(arc_dim), nn.GELU(), nn.Dropout(0.1))
        self.arc_head = nn.Sequential(nn.Linear(H, arc_dim), nn.LayerNorm(arc_dim), nn.GELU(), nn.Dropout(0.1))
        self.label_dep = nn.Sequential(nn.Linear(H, label_dim), nn.LayerNorm(label_dim), nn.GELU(), nn.Dropout(0.1))
        self.label_head = nn.Sequential(nn.Linear(H, label_dim), nn.LayerNorm(label_dim), nn.GELU(), nn.Dropout(0.1))
        self.biaffine_arc = Biaffine(arc_dim, 1)
        self.biaffine_label = Biaffine(label_dim, num_labels)
    def forward(self, hidden):
        return (self.biaffine_arc(self.arc_dep(hidden), self.arc_head(hidden)),
                self.biaffine_label(self.label_dep(hidden), self.label_head(hidden)))


# ── Combined cascade model ───────────────────────────────────

class CascadeModel(nn.Module):
    """Single model: one encoder pass, all 5 heads.

    The encoder runs with predicate embedding injected at the embedding
    level (for SRL). For non-SRL tokens, pred_embedding adds zero (since
    predicate_idx marks only one token). All heads read from the same
    encoder output — no separate forward pass needed.
    """

    def __init__(self, encoder, H, num_layers):
        super().__init__()
        self.encoder = encoder
        self.pred_embedding = nn.Embedding(2, H)

        # POS
        self.pos_scalar_mix = ScalarMix(num_layers)
        self.pos_head = nn.Linear(H, POS_LABELS)

        # NER
        self.ner_scalar_mix = ScalarMix(num_layers)
        self.ner_lstm = nn.LSTM(H, 256, bidirectional=True, batch_first=True)
        self.ner_proj = nn.Linear(512, H)
        self.ner_head = nn.Sequential(
            nn.LayerNorm(H + POS_LABELS), nn.Linear(H + POS_LABELS, H),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(H, NER_LABELS))

        # DEP
        self.dep_scalar_mix = ScalarMix(num_layers)
        in_dim = H + POS_LABELS + NER_LABELS
        self.dep_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, H), nn.GELU())
        self.dep_biaffine = BiaffineDEPHead(H, num_labels=DEPREL_LABELS)

        # SRL (uses same encoder output, just the last hidden state)
        self.srl_classifier = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(H, H), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(H, SRL_LABELS))

        # CLS
        self.cls_scalar_mix = ScalarMix(num_layers)
        self.cls_pool = AttentionPool(H)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(H), nn.Linear(H, H // 2), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(H // 2, CLS_LABELS))

    def forward(self, input_ids, attention_mask, predicate_idx):
        B, S = input_ids.size()

        # Single encoder pass with predicate embedding
        emb = self.encoder.embeddings(input_ids)
        indicator = torch.zeros(B, S, dtype=torch.long, device=input_ids.device)
        indicator.scatter_(1, predicate_idx.unsqueeze(1), 1)
        emb = emb + self.pred_embedding(indicator)

        # Run encoder (with output_hidden_states for ScalarMix heads)
        enc_out = self.encoder.encoder(emb, attention_mask, output_hidden_states=True)
        hidden_states = enc_out.hidden_states  # tuple of 25 tensors
        last_hidden = enc_out.last_hidden_state

        layers = list(hidden_states)

        # POS
        pos_h = self.pos_scalar_mix(layers)
        pos_logits = self.pos_head(pos_h)
        pos_probs = torch.softmax(pos_logits, dim=-1)

        # NER (BiLSTM + POS cascade)
        ner_h = self.ner_scalar_mix(layers)
        lstm_out, _ = self.ner_lstm(ner_h)
        adapted = self.ner_proj(lstm_out) + ner_h
        ner_logits = self.ner_head(torch.cat([adapted, pos_probs], dim=-1))
        ner_probs = torch.softmax(ner_logits, dim=-1)

        # DEP (Biaffine + POS/NER cascade)
        dep_h = self.dep_scalar_mix(layers)
        dep_input = torch.cat([dep_h, pos_probs, ner_probs], dim=-1)
        proj_h = self.dep_proj(dep_input)
        arc_scores, label_scores = self.dep_biaffine(proj_h)

        # SRL (from last hidden state — encoder already saw predicate embedding)
        srl_logits = self.srl_classifier(last_hidden)

        # CLS (AttentionPool)
        cls_h = self.cls_scalar_mix(layers)
        cls_pooled = self.cls_pool(cls_h, attention_mask)
        cls_logits = self.cls_head(cls_pooled)

        return pos_logits, ner_logits, arc_scores, label_scores, srl_logits, cls_logits


# ── Export ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/kniv-deberta-nlp-base-en-large")
    parser.add_argument("--output", default="models/kniv-deberta-nlp-base-en-large/onnx/cascade.onnx")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    encoder = AutoModel.from_pretrained("microsoft/deberta-v3-large").float()
    H = encoder.config.hidden_size
    num_layers = encoder.config.num_hidden_layers + 1

    state = torch.load(f"{args.model_dir}/model.pt", map_location="cpu", weights_only=True)
    encoder.load_state_dict(state["deberta"])

    # Build combined model
    model = CascadeModel(encoder, H, num_layers)
    model.pred_embedding.load_state_dict(state["pred_embedding"])
    model.pos_scalar_mix.load_state_dict(state["pos_scalar_mix"])
    model.pos_head.load_state_dict(state["pos_head"])
    model.ner_scalar_mix.load_state_dict(state["ner_scalar_mix"])
    model.ner_lstm.load_state_dict(state["ner_lstm"])
    model.ner_proj.load_state_dict(state["ner_proj"])
    model.ner_head.load_state_dict(state["ner_head"])
    model.dep_scalar_mix.load_state_dict(state["dep_scalar_mix"])
    model.dep_proj.load_state_dict(state["dep_proj"])
    model.dep_biaffine.load_state_dict(state["dep_biaffine"])
    model.srl_classifier.load_state_dict(state["classifier"])
    model.cls_scalar_mix.load_state_dict(state["cls_scalar_mix"])
    model.cls_pool.load_state_dict(state["cls_pool"])
    model.cls_head.load_state_dict(state["cls_head"])
    model.eval()

    print("Exporting single cascade ONNX model...")

    # Dummy inputs
    dummy = tokenizer("Hello world", return_tensors="pt", padding="max_length",
                       max_length=args.max_length, truncation=True)
    dummy_pred = torch.tensor([1], dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"], dummy_pred),
        args.output,
        dynamo=False,
        opset_version=14,
        input_names=["input_ids", "attention_mask", "predicate_idx"],
        output_names=["pos_logits", "ner_logits", "arc_scores", "label_scores", "srl_logits", "cls_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "predicate_idx": {0: "batch"},
            "pos_logits": {0: "batch", 1: "seq"},
            "ner_logits": {0: "batch", 1: "seq"},
            "arc_scores": {0: "batch", 1: "seq"},
            "label_scores": {0: "batch", 1: "seq"},
            "srl_logits": {0: "batch", 1: "seq"},
            "cls_logits": {0: "batch"},
        },
    )

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved cascade.onnx ({size_mb:.0f} MB)")

    # Validate
    if not args.skip_validation:
        print("\nValidating ONNX vs PyTorch...")
        text = "Steve Jobs founded Apple in 1976."
        enc = tokenizer(text, return_tensors="pt", padding="max_length",
                         max_length=args.max_length, truncation=True)
        pred_idx = torch.tensor([3], dtype=torch.long)

        with torch.no_grad():
            pt_out = model(enc["input_ids"], enc["attention_mask"], pred_idx)

        sess = ort.InferenceSession(args.output)
        onnx_out = sess.run(None, {
            "input_ids": enc["input_ids"].numpy(),
            "attention_mask": enc["attention_mask"].numpy(),
            "predicate_idx": pred_idx.numpy(),
        })

        names = ["pos_logits", "ner_logits", "arc_scores", "label_scores", "srl_logits", "cls_logits"]
        all_ok = True
        for name, pt, ox in zip(names, pt_out, onnx_out):
            diff = np.abs(pt.numpy() - ox).max()
            status = "OK" if diff < 0.01 else "MISMATCH"
            if status != "OK":
                all_ok = False
            print(f"  {name:20s} max diff: {diff:.6f}  {status}")

        if all_ok:
            print("\nAll validations passed.")
        else:
            print("\nWARNING: Some outputs differ.")

    # Save metadata
    meta = {
        "file": "cascade.onnx",
        "size_mb": round(size_mb, 1),
        "max_length": args.max_length,
        "opset_version": 14,
        "inputs": {
            "input_ids": "int64 [batch, seq]",
            "attention_mask": "int64 [batch, seq]",
            "predicate_idx": "int64 [batch] — token index of SRL predicate (0 if no SRL needed)",
        },
        "outputs": {
            "pos_logits": "float32 [batch, seq, 17]",
            "ner_logits": "float32 [batch, seq, 37]",
            "arc_scores": "float32 [batch, seq, seq] — DEP head selection",
            "label_scores": "float32 [batch, seq, seq, 53] — DEP relation labels",
            "srl_logits": "float32 [batch, seq, 42]",
            "cls_logits": "float32 [batch, 8]",
        },
    }
    meta_path = os.path.join(os.path.dirname(args.output), "onnx_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"ONNX EXPORT COMPLETE")
    print(f"{'=' * 60}")
    print(f"  cascade.onnx    {size_mb:8.0f} MB")
    print(f"  Inputs:  input_ids, attention_mask, predicate_idx")
    print(f"  Outputs: pos, ner, arc, label, srl, cls logits")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

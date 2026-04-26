"""Export kniv-deberta-nlp-base-en-large to ONNX.

Exports three ONNX models:
  1. encoder.onnx     — shared DeBERTa encoder (input_ids → 25 hidden states)
  2. heads.onnx       — all non-SRL heads (hidden_states → POS, NER, DEP, CLS)
  3. srl.onnx         — SRL with predicate embedding (input_ids + pred_idx → SRL logits)

This split allows running the encoder once and dispatching to multiple heads,
while SRL gets its own model due to the different forward path (predicate embedding).

Usage:
    .venv/bin/python models/kniv-deberta-nlp-base-en-large/export_onnx.py
    .venv/bin/python models/kniv-deberta-nlp-base-en-large/export_onnx.py --output-dir onnx-export
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import os
os.environ["TORCH_ONNX_USE_OLD_EXPORTER"] = "1"  # Force legacy tracer, not dynamo

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from transformers import AutoModel, AutoTokenizer

# ── Model components (must match checkpoint exactly) ──────────

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
        self.weight = nn.Parameter(torch.zeros(out_dim, in_dim+1, in_dim+1))
    def forward(self, h_dep, h_head):
        B, S, D = h_dep.size()
        h_dep = torch.cat([h_dep, torch.ones(B,S,1,device=h_dep.device)], -1)
        h_head = torch.cat([h_head, torch.ones(B,S,1,device=h_head.device)], -1)
        scores = torch.einsum("bxi,oij,byj->boxy", h_dep, self.weight, h_head)
        return scores.squeeze(1).contiguous() if scores.size(1)==1 else scores.permute(0,2,3,1).contiguous()

class BiaffineDEPHead(nn.Module):
    def __init__(self, H, arc_dim=512, label_dim=128, num_labels=53, dropout=0.0):
        super().__init__()
        # No dropout for export (eval mode)
        self.arc_dep = nn.Sequential(nn.Linear(H, arc_dim), nn.LayerNorm(arc_dim), nn.GELU())
        self.arc_head = nn.Sequential(nn.Linear(H, arc_dim), nn.LayerNorm(arc_dim), nn.GELU())
        self.label_dep = nn.Sequential(nn.Linear(H, label_dim), nn.LayerNorm(label_dim), nn.GELU())
        self.label_head = nn.Sequential(nn.Linear(H, label_dim), nn.LayerNorm(label_dim), nn.GELU())
        self.biaffine_arc = Biaffine(arc_dim, 1)
        self.biaffine_label = Biaffine(label_dim, num_labels)
    def forward(self, hidden):
        return (self.biaffine_arc(self.arc_dep(hidden), self.arc_head(hidden)),
                self.biaffine_label(self.label_dep(hidden), self.label_head(hidden)))


# ── Wrapper models for ONNX export ───────────────────────────

NER_LABELS = 37
POS_LABELS = 17
DEPREL_LABELS = 53
SRL_LABELS = 42
CLS_LABELS = 8


class CascadeHeads(nn.Module):
    """All non-SRL heads. Takes encoder hidden_states tuple as input."""

    def __init__(self, H, num_layers):
        super().__init__()
        self.pos_scalar_mix = ScalarMix(num_layers)
        self.pos_head = nn.Linear(H, POS_LABELS)

        self.ner_scalar_mix = ScalarMix(num_layers)
        self.ner_lstm = nn.LSTM(H, 256, bidirectional=True, batch_first=True)
        self.ner_proj = nn.Linear(512, H)
        # Match training structure exactly (dropout becomes no-op in eval)
        self.ner_head = nn.Sequential(
            nn.LayerNorm(H + POS_LABELS), nn.Linear(H + POS_LABELS, H),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(H, NER_LABELS))

        self.dep_scalar_mix = ScalarMix(num_layers)
        self.dep_proj = nn.Sequential(
            nn.LayerNorm(H + POS_LABELS + NER_LABELS),
            nn.Linear(H + POS_LABELS + NER_LABELS, H), nn.GELU())
        self.dep_biaffine = BiaffineDEPHead(H, num_labels=DEPREL_LABELS, dropout=0.0)

        self.cls_scalar_mix = ScalarMix(num_layers)
        self.cls_pool = AttentionPool(H)
        # Match training structure exactly
        self.cls_head = nn.Sequential(
            nn.LayerNorm(H), nn.Linear(H, H // 2), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(H // 2, CLS_LABELS))

    def forward(self, attention_mask, stacked_hidden):
        # stacked_hidden: [num_layers, batch, seq, hidden]
        layers = [stacked_hidden[i] for i in range(stacked_hidden.size(0))]

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

        # CLS (AttentionPool)
        cls_h = self.cls_scalar_mix(layers)
        cls_pooled = self.cls_pool(cls_h, attention_mask)
        cls_logits = self.cls_head(cls_pooled)

        return pos_logits, ner_logits, arc_scores, label_scores, cls_logits


class SRLModel(nn.Module):
    """SRL with predicate embedding — separate ONNX model."""

    def __init__(self, encoder, pred_embedding, classifier):
        super().__init__()
        self.encoder = encoder
        self.pred_embedding = pred_embedding
        self.classifier = classifier

    def forward(self, input_ids, attention_mask, predicate_idx):
        B, S = input_ids.size()
        emb = self.encoder.embeddings(input_ids)
        indicator = torch.zeros(B, S, dtype=torch.long, device=input_ids.device)
        indicator.scatter_(1, predicate_idx.unsqueeze(1), 1)
        emb = emb + self.pred_embedding(indicator)
        hidden = self.encoder.encoder(emb, attention_mask).last_hidden_state
        return self.classifier(hidden)


# ── Export functions ──────────────────────────────────────────

def export_encoder(encoder, tokenizer, output_dir, max_length=128):
    """Export shared encoder to ONNX."""
    print("\n[1/3] Exporting encoder...")
    encoder.eval()

    dummy = tokenizer("Hello world", return_tensors="pt", padding="max_length",
                       max_length=max_length, truncation=True)
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]

    class EncoderWrapper(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc
        def forward(self, input_ids, attention_mask):
            out = self.enc(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            # Stack all hidden states: [25, batch, seq, hidden]
            return torch.stack(out.hidden_states, dim=0)

    wrapper = EncoderWrapper(encoder)
    wrapper.eval()

    path = os.path.join(output_dir, "encoder.onnx")

    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        path,
        dynamo=False,
        opset_version=14,
        input_names=["input_ids", "attention_mask"],
        output_names=["stacked_hidden"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "stacked_hidden": {1: "batch", 2: "seq"},
        },
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved encoder.onnx ({size_mb:.0f} MB)")
    return path


def export_heads(heads_model, output_dir, max_length=128):
    """Export cascade heads (POS, NER, DEP, CLS) to ONNX."""
    print("\n[2/3] Exporting cascade heads...")
    heads_model.eval()

    B, S, H = 1, max_length, 1024
    dummy_mask = torch.ones(B, S, dtype=torch.long)
    dummy_stacked = torch.randn(25, B, S, H)

    path = os.path.join(output_dir, "heads.onnx")

    torch.onnx.export(
        heads_model,
        (dummy_mask, dummy_stacked),
        path,
        dynamo=False,
        opset_version=14,
        input_names=["attention_mask", "stacked_hidden"],
        output_names=["pos_logits", "ner_logits", "arc_scores", "label_scores", "cls_logits"],
        dynamic_axes={
            "attention_mask": {0: "batch", 1: "seq"},
            "stacked_hidden": {1: "batch", 2: "seq"},
            "pos_logits": {0: "batch", 1: "seq"},
            "ner_logits": {0: "batch", 1: "seq"},
            "arc_scores": {0: "batch", 1: "seq"},
            "label_scores": {0: "batch", 1: "seq"},
            "cls_logits": {0: "batch"},
        },
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved heads.onnx ({size_mb:.0f} MB)")
    return path


def export_srl(srl_model, tokenizer, output_dir, max_length=128):
    """Export SRL model (encoder + predicate embedding + classifier) to ONNX."""
    print("\n[3/3] Exporting SRL model...")
    srl_model.eval()

    dummy = tokenizer("Hello world", return_tensors="pt", padding="max_length",
                       max_length=max_length, truncation=True)
    dummy_pred_idx = torch.tensor([1], dtype=torch.long)

    path = os.path.join(output_dir, "srl.onnx")

    torch.onnx.export(
        srl_model,
        (dummy["input_ids"], dummy["attention_mask"], dummy_pred_idx),
        path,
        dynamo=False,
        opset_version=14,
        input_names=["input_ids", "attention_mask", "predicate_idx"],
        output_names=["srl_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "predicate_idx": {0: "batch"},
            "srl_logits": {0: "batch", 1: "seq"},
        },
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved srl.onnx ({size_mb:.0f} MB)")
    return path


def validate(encoder_path, heads_path, srl_path, encoder, heads_model, srl_model, tokenizer):
    """Validate ONNX outputs match PyTorch."""
    print("\nValidating ONNX outputs...")

    text = "Steve Jobs founded Apple in 1976."
    enc = tokenizer(text, return_tensors="pt", padding="max_length",
                     max_length=128, truncation=True)

    # PyTorch encoder
    with torch.no_grad():
        pt_out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                         output_hidden_states=True)
        pt_stacked = torch.stack(pt_out.hidden_states, dim=0)

        # PyTorch heads
        pt_pos, pt_ner, pt_arc, pt_lab, pt_cls = heads_model(
            enc["attention_mask"], pt_stacked)
        pt_pos_np, pt_ner_np, pt_cls_np = pt_pos.numpy(), pt_ner.numpy(), pt_cls.numpy()

        # PyTorch SRL
        pt_srl = srl_model(enc["input_ids"], enc["attention_mask"], torch.tensor([3])).numpy()

    # ONNX encoder
    enc_sess = ort.InferenceSession(encoder_path)
    onnx_stacked = enc_sess.run(None, {
        "input_ids": enc["input_ids"].numpy(),
        "attention_mask": enc["attention_mask"].numpy(),
    })[0]

    # ONNX heads
    heads_sess = ort.InferenceSession(heads_path)
    onnx_pos, onnx_ner, onnx_arc, onnx_lab, onnx_cls = heads_sess.run(None, {
        "attention_mask": enc["attention_mask"].numpy(),
        "stacked_hidden": onnx_stacked,
    })

    # ONNX SRL
    srl_sess = ort.InferenceSession(srl_path)
    onnx_srl = srl_sess.run(None, {
        "input_ids": enc["input_ids"].numpy(),
        "attention_mask": enc["attention_mask"].numpy(),
        "predicate_idx": np.array([3], dtype=np.int64),
    })[0]

    # Compare
    checks = [
        ("Encoder stacked[0]", pt_stacked[0].numpy(), onnx_stacked[0]),
        ("Encoder stacked[24]", pt_stacked[24].numpy(), onnx_stacked[24]),
        ("POS logits", pt_pos_np, onnx_pos),
        ("NER logits", pt_ner_np, onnx_ner),
        ("CLS logits", pt_cls_np, onnx_cls),
        ("SRL logits", pt_srl, onnx_srl),
    ]

    all_ok = True
    for name, pt, ox in checks:
        diff = np.abs(pt - ox).max()
        status = "OK" if diff < 0.01 else "MISMATCH"
        if status != "OK":
            all_ok = False
        print(f"  {name:25s} max diff: {diff:.6f}  {status}")

    return all_ok


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/kniv-deberta-nlp-base-en-large")
    parser.add_argument("--output-dir", default="models/kniv-deberta-nlp-base-en-large/onnx")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading PyTorch model...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    encoder = AutoModel.from_pretrained("microsoft/deberta-v3-large").float()
    H = encoder.config.hidden_size
    num_layers = encoder.config.num_hidden_layers + 1

    state = torch.load(f"{args.model_dir}/model.pt", map_location="cpu", weights_only=True)
    encoder.load_state_dict(state["deberta"])
    encoder.eval()

    # Build heads model — use exact same Sequential structure as training
    heads = CascadeHeads(H, num_layers)
    heads.pos_scalar_mix.load_state_dict(state["pos_scalar_mix"])
    heads.pos_head.load_state_dict(state["pos_head"])
    heads.ner_scalar_mix.load_state_dict(state["ner_scalar_mix"])
    heads.ner_lstm.load_state_dict(state["ner_lstm"])
    heads.ner_proj.load_state_dict(state["ner_proj"])
    heads.ner_head.load_state_dict(state["ner_head"])
    heads.dep_scalar_mix.load_state_dict(state["dep_scalar_mix"])
    heads.dep_proj.load_state_dict(state["dep_proj"])
    # DEP biaffine: remap keys from training structure (with dropout) to export (without)
    dep_state = state["dep_biaffine"]
    dep_remap = {}
    for k, v in dep_state.items():
        # Skip dropout layer keys (e.g., "arc_dep.3.weight" where .3 is dropout)
        parts = k.split(".")
        # The training structure had: Linear, LN, GELU, Dropout (indices 0,1,2,3)
        # Our export structure has: Linear, LN, GELU (indices 0,1,2)
        # Weight keys from training: arc_dep.0.weight, arc_dep.1.weight, etc.
        # Since dropout has no weights, all keys should load directly
        dep_remap[k] = v
    heads.dep_biaffine.load_state_dict(dep_remap, strict=False)
    heads.cls_scalar_mix.load_state_dict(state["cls_scalar_mix"])
    heads.cls_pool.load_state_dict(state["cls_pool"])
    heads.cls_head.load_state_dict(state["cls_head"])
    heads.eval()

    # Build SRL model — match training classifier structure exactly
    pred_embedding = nn.Embedding(2, H)
    pred_embedding.load_state_dict(state["pred_embedding"])
    srl_classifier = nn.Sequential(
        nn.Dropout(0.1), nn.Linear(H, H), nn.GELU(),
        nn.Dropout(0.1), nn.Linear(H, SRL_LABELS))
    srl_classifier.load_state_dict(state["classifier"])

    srl_model = SRLModel(encoder, pred_embedding, srl_classifier)
    srl_model.eval()

    print(f"Model loaded. Exporting to {args.output_dir}/")

    # Export
    enc_path = export_encoder(encoder, tokenizer, args.output_dir, args.max_length)
    heads_path = export_heads(heads, args.output_dir, args.max_length)
    srl_path = export_srl(srl_model, tokenizer, args.output_dir, args.max_length)

    # Save metadata
    meta = {
        "files": ["encoder.onnx", "heads.onnx", "srl.onnx"],
        "max_length": args.max_length,
        "opset_version": 14,
        "encoder_outputs": 25,
        "heads_outputs": ["pos_logits", "ner_logits", "arc_scores", "label_scores", "cls_logits"],
        "srl_outputs": ["srl_logits"],
    }
    with open(os.path.join(args.output_dir, "onnx_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Validate
    if not args.skip_validation:
        ok = validate(enc_path, heads_path, srl_path, encoder, heads, srl_model, tokenizer)
        if ok:
            print("\nAll validations passed.")
        else:
            print("\nWARNING: Some outputs differ significantly.")

    # Summary
    total_mb = sum(os.path.getsize(os.path.join(args.output_dir, f)) / 1e6
                   for f in ["encoder.onnx", "heads.onnx", "srl.onnx"])
    print(f"\n{'=' * 60}")
    print(f"ONNX EXPORT COMPLETE")
    print(f"{'=' * 60}")
    for f in ["encoder.onnx", "heads.onnx", "srl.onnx"]:
        sz = os.path.getsize(os.path.join(args.output_dir, f)) / 1e6
        print(f"  {f:20s} {sz:8.1f} MB")
    print(f"  {'Total':20s} {total_mb:8.1f} MB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

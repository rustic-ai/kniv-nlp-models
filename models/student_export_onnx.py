"""Export a kniv distilled student to ONNX (FP32 + INT8 dynamic quantization).

Produces:
    <model-dir>/onnx/cascade.onnx          — FP32 ONNX
    <model-dir>/onnx/cascade-int8.onnx     — INT8 dynamic-quantized ONNX
    <model-dir>/onnx/onnx_meta.json        — sizes, validation status, inputs/outputs

Usage:
    python models/student_export_onnx.py --model-dir models/kniv-deberta-nlp-base-en-xsmall
    python models/student_export_onnx.py --model-dir models/kniv-deberta-nlp-base-en-small
"""
from __future__ import annotations
import os
os.environ["TORCH_ONNX_USE_OLD_EXPORTER"] = "1"

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

sys.path.insert(0, str(Path(__file__).parent))
from student_loader import load_student


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True,
                        help="Path to kniv-deberta-nlp-base-en-{xsmall,small,base}")
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--skip-int8", action="store_true",
                        help="Skip INT8 dynamic quantization step")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip ONNX vs PyTorch validation")
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    onnx_dir = model_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    fp32_path = onnx_dir / "cascade.onnx"
    int8_path = onnx_dir / "cascade-int8.onnx"

    # Load student via shared loader (uses CPU for export)
    print("=" * 60)
    print(f"ONNX EXPORT — {model_dir.name}")
    print("=" * 60)
    model, tokenizer, info = load_student(model_dir, checkpoint=args.checkpoint, device="cpu")
    print(f"  Encoder: {info['encoder']} ({info['hidden_dim']}d, {info['layers']} layers)")
    print(f"  Params:  {info.get('params_total_millions', '?')}M")

    # Dummy inputs
    dummy = tokenizer("Hello world", return_tensors="pt", padding="max_length",
                      max_length=args.max_length, truncation=True)
    dummy_pred = torch.tensor([1], dtype=torch.long)

    # ── FP32 export ──
    print(f"\nExporting FP32 ONNX → {fp32_path}")
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"], dummy_pred),
        str(fp32_path),
        dynamo=False,
        opset_version=args.opset,
        input_names=["input_ids", "attention_mask", "predicate_idx"],
        output_names=["pos_logits", "ner_logits", "arc_scores", "label_scores",
                      "srl_logits", "cls_logits"],
        dynamic_axes={
            "input_ids":      {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "predicate_idx":  {0: "batch"},
            "pos_logits":     {0: "batch", 1: "seq"},
            "ner_logits":     {0: "batch", 1: "seq"},
            "arc_scores":     {0: "batch", 1: "seq"},
            "label_scores":   {0: "batch", 1: "seq"},
            "srl_logits":     {0: "batch", 1: "seq"},
            "cls_logits":     {0: "batch"},
        },
    )
    fp32_size = os.path.getsize(fp32_path) / 1e6
    print(f"  Saved {fp32_size:.0f} MB")

    # ── INT8 dynamic quantization ──
    int8_size = None
    if not args.skip_int8:
        print(f"\nQuantizing to INT8 → {int8_path}")
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True,
        )
        int8_size = os.path.getsize(int8_path) / 1e6
        print(f"  Saved {int8_size:.0f} MB ({fp32_size/int8_size:.1f}x smaller)")

    # ── Validation: PT vs FP32 vs INT8 ──
    validation = {"fp32": None, "int8": None}
    if not args.skip_validation:
        print(f"\nValidating outputs (PT vs ONNX FP32{' vs INT8' if int8_size else ''})...")
        text = "Steve Jobs founded Apple Inc. in 1976 ."
        enc = tokenizer(text.split(), is_split_into_words=True, return_tensors="pt",
                        padding="max_length", max_length=args.max_length, truncation=True)
        pred_idx = torch.tensor([3], dtype=torch.long)

        with torch.no_grad():
            pt_out = model(enc["input_ids"], enc["attention_mask"], pred_idx)

        names = ["pos_logits", "ner_logits", "arc_scores",
                 "label_scores", "srl_logits", "cls_logits"]

        # FP32 validation
        sess_fp32 = ort.InferenceSession(str(fp32_path),
                                          providers=["CPUExecutionProvider"])
        fp32_out = sess_fp32.run(None, {
            "input_ids": enc["input_ids"].numpy(),
            "attention_mask": enc["attention_mask"].numpy(),
            "predicate_idx": pred_idx.numpy(),
        })
        max_diffs_fp32 = []
        for name, pt, ox in zip(names, pt_out, fp32_out):
            diff = float(np.abs(pt.numpy() - ox).max())
            max_diffs_fp32.append((name, diff))
        validation["fp32"] = {
            "max_diff_overall": max(d for _, d in max_diffs_fp32),
            "per_output": dict(max_diffs_fp32),
        }
        print(f"  FP32 max diff vs PT: {validation['fp32']['max_diff_overall']:.6f}")

        # INT8 validation
        if int8_size is not None:
            sess_int8 = ort.InferenceSession(str(int8_path),
                                              providers=["CPUExecutionProvider"])
            int8_out = sess_int8.run(None, {
                "input_ids": enc["input_ids"].numpy(),
                "attention_mask": enc["attention_mask"].numpy(),
                "predicate_idx": pred_idx.numpy(),
            })
            max_diffs_int8 = []
            for name, pt, ox in zip(names, pt_out, int8_out):
                diff = float(np.abs(pt.numpy() - ox).max())
                max_diffs_int8.append((name, diff))
            validation["int8"] = {
                "max_diff_overall": max(d for _, d in max_diffs_int8),
                "per_output": dict(max_diffs_int8),
            }
            print(f"  INT8 max diff vs PT: {validation['int8']['max_diff_overall']:.6f}")
            print(f"    (INT8 has expected quantization error; "
                  f"verify quality via benchmark, not max-diff)")

    # ── Metadata ──
    meta = {
        "encoder": info["encoder"],
        "hidden_dim": info["hidden_dim"],
        "layers": info["layers"],
        "max_length": args.max_length,
        "opset_version": args.opset,
        "files": {
            "cascade.onnx": {"size_mb": round(fp32_size, 1)},
        },
        "validation": validation,
        "inputs": {
            "input_ids":      "int64 [batch, seq]",
            "attention_mask": "int64 [batch, seq]",
            "predicate_idx":  "int64 [batch] — SRL predicate token index (0 if unused)",
        },
        "outputs": {
            "pos_logits":   "float32 [batch, seq, 17]",
            "ner_logits":   "float32 [batch, seq, 37]",
            "arc_scores":   "float32 [batch, seq, seq] — DEP head selection",
            "label_scores": "float32 [batch, seq, seq, 53] — DEP relation labels",
            "srl_logits":   "float32 [batch, seq, 42]",
            "cls_logits":   "float32 [batch, 8]",
        },
    }
    if int8_size is not None:
        meta["files"]["cascade-int8.onnx"] = {"size_mb": round(int8_size, 1)}
    meta_path = onnx_dir / "onnx_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"ONNX EXPORT COMPLETE — {model_dir.name}")
    print(f"{'=' * 60}")
    print(f"  cascade.onnx        {fp32_size:8.0f} MB  (FP32)")
    if int8_size is not None:
        print(f"  cascade-int8.onnx   {int8_size:8.0f} MB  (INT8 dynamic)")
    print(f"  Metadata: {meta_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

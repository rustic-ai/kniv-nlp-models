"""Export the trained DistilRoBERTa multi-task model to ONNX + INT8.

Loads the PyTorch checkpoint, exports to ONNX, quantizes, validates,
and copies tokenizer + label_maps to the output directory.

Usage:
    uv run python models/distilroberta-nlp-en/export.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Add project root and model dir to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from model import MultiTaskNLPModel
from shared.export_onnx import export_to_onnx, quantize_int8, validate_onnx


MODEL_DIR = PROJECT_ROOT / "outputs" / "distilroberta-nlp-en" / "best"
OUTPUT_DIR = PROJECT_ROOT / "onnx-output" / "distilroberta-nlp-en"
ENCODER_NAME = "distilroberta-base"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load trained model
    print(f"Loading model from {MODEL_DIR}...")
    model = MultiTaskNLPModel.load(str(MODEL_DIR), ENCODER_NAME)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load tokenizer (with add_prefix_space for RoBERTa)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), add_prefix_space=True)

    # Export to ONNX
    print("\nExporting to ONNX...")
    export_to_onnx(model, tokenizer, OUTPUT_DIR, max_length=128)

    # Validate FP32 ONNX parity
    print("\nValidating FP32 ONNX parity...")
    validate_onnx(
        OUTPUT_DIR / "model.onnx",
        model,
        tokenizer,
        test_text="Caroline went to the hospital yesterday.",
    )

    # Quantize to INT8
    print("\nQuantizing to INT8...")
    quantize_int8(OUTPUT_DIR, OUTPUT_DIR)

    # Validate INT8 parity
    print("\nValidating INT8 parity...")
    validate_onnx(
        OUTPUT_DIR / "model-int8.onnx",
        model,
        tokenizer,
        test_text="Caroline went to the hospital yesterday.",
    )

    # Copy tokenizer files
    print("\nCopying tokenizer files...")
    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                   "merges.txt", "special_tokens_map.json"]:
        src = MODEL_DIR / fname
        if src.exists():
            shutil.copy2(src, OUTPUT_DIR / fname)
            print(f"  {fname}")

    # Copy label maps
    src = MODEL_DIR / "label_maps.json"
    if src.exists():
        shutil.copy2(src, OUTPUT_DIR / "label_maps.json")
        print("  label_maps.json")

    # Copy config
    src = MODEL_DIR / "config.json"
    if src.exists():
        shutil.copy2(src, OUTPUT_DIR / "config.json")
        print("  config.json")

    # Summary — account for external data files
    fp32_size = (OUTPUT_DIR / "model.onnx").stat().st_size / 1024 / 1024
    fp32_data = OUTPUT_DIR / "model.onnx.data"
    if fp32_data.exists():
        fp32_size += fp32_data.stat().st_size / 1024 / 1024
    int8_size = (OUTPUT_DIR / "model-int8.onnx").stat().st_size / 1024 / 1024
    int8_data = OUTPUT_DIR / "model-int8.onnx.data"
    if int8_data.exists():
        int8_size += int8_data.stat().st_size / 1024 / 1024

    with open(MODEL_DIR / "label_maps.json") as f:
        labels = json.load(f)

    print(f"\n{'='*60}")
    print(f"Export complete: {OUTPUT_DIR}")
    print(f"  FP32:  {fp32_size:.1f} MB")
    print(f"  INT8:  {int8_size:.1f} MB")
    print(f"  NER:   {len(labels['ner_labels'])} labels")
    print(f"  POS:   {len(labels['pos_labels'])} labels")
    print(f"  Dep:   {len(labels['dep_labels'])} labels")
    print(f"  CLS:   {len(labels['cls_labels'])} labels")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

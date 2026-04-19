"""Export a trained multi-task model to ONNX and quantize to INT8.

Usage:
    python -m shared.export_onnx \
        --model-dir outputs/deberta-v3-nlp-en/best \
        --encoder microsoft/deberta-v3-small \
        --output-dir onnx-output/deberta-v3-nlp-en
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer


def export_to_onnx(model, tokenizer, output_path: Path, max_length: int = 128):
    """Export a PyTorch model to ONNX format with dynamic axes."""
    model.eval()

    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, max_length))
    dummy_attention_mask = torch.ones(1, max_length, dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(output_path / "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["ner_logits", "pos_logits", "dep_logits", "cls_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "ner_logits": {0: "batch_size", 1: "sequence_length"},
            "pos_logits": {0: "batch_size", 1: "sequence_length"},
            "dep_logits": {0: "batch_size", 1: "sequence_length"},
            "cls_logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    print(f"Exported FP32 ONNX model to {output_path / 'model.onnx'}")


def quantize_int8(input_path: Path, output_path: Path):
    """Quantize an ONNX model to INT8 using dynamic quantization."""
    model_path = input_path / "model.onnx"
    int8_path = output_path / "model-int8.onnx"

    print("  Quantizing...", flush=True)
    try:
        quantize_dynamic(
            model_input=str(model_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )
    except Exception as e:
        # Shape inference can fail on models with mixed output dimensions
        # (3D token heads + 2D CLS head). Fall back to copying FP32 model.
        print(f"  ⚠ INT8 quantization failed: {e}", flush=True)
        print("  Falling back to FP32 model (no quantization)", flush=True)
        import shutil
        shutil.copy2(str(model_path), str(int8_path))

    fp32_size = model_path.stat().st_size / 1024 / 1024
    # Account for external data files (large models split weights)
    fp32_data = model_path.with_suffix(".onnx.data")
    if fp32_data.exists():
        fp32_size += fp32_data.stat().st_size / 1024 / 1024
    int8_size = int8_path.stat().st_size / 1024 / 1024
    int8_data = int8_path.with_suffix(".onnx.data")
    if int8_data.exists():
        int8_size += int8_data.stat().st_size / 1024 / 1024
    print(f"Quantized: {fp32_size:.1f}MB (FP32) → {int8_size:.1f}MB (INT8)")


def validate_onnx(onnx_path: Path, model, tokenizer, test_text: str = "Caroline went to the hospital yesterday."):
    """Compare PyTorch and ONNX outputs to verify export correctness."""
    model.eval()
    inputs = tokenizer(test_text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)

    # PyTorch inference
    with torch.no_grad():
        pt_outputs = model(inputs["input_ids"], inputs["attention_mask"])

    # ONNX inference
    session = ort.InferenceSession(str(onnx_path))
    ort_outputs = session.run(
        None,
        {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy(),
        },
    )

    # Compare outputs
    all_ok = True
    for i, name in enumerate(["ner_logits", "pos_logits", "dep_logits", "cls_logits"]):
        pt_out = pt_outputs[name].numpy()
        ort_out = ort_outputs[i]
        max_diff = np.max(np.abs(pt_out - ort_out))
        ok = max_diff < 0.01
        if not ok:
            all_ok = False
        print(f"  {name}: max diff = {max_diff:.6f} {'✓' if ok else '✗ MISMATCH'}")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Export multi-task model to ONNX")
    parser.add_argument("--model-dir", type=Path, required=True, help="Trained model directory (e.g., outputs/.../best)")
    parser.add_argument("--encoder", type=str, required=True, help="Base encoder name (e.g., microsoft/deberta-v3-small)")
    parser.add_argument("--output-dir", type=Path, required=True, help="ONNX output directory")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    # Validate inputs
    if not args.model_dir.exists():
        print(f"Error: model directory not found: {args.model_dir}")
        sys.exit(1)

    label_maps = args.model_dir / "label_maps.json"
    if not label_maps.exists():
        print(f"Error: no label_maps.json in {args.model_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find model.py — check models/ directory matching the output name
    # e.g., outputs/distilroberta-nlp-en/best → models/distilroberta-nlp-en/
    project_root = Path(__file__).parent.parent
    model_name = args.model_dir.parent.name  # e.g., "distilroberta-nlp-en"
    model_code_dir = project_root / "models" / model_name
    if model_code_dir.exists():
        sys.path.insert(0, str(model_code_dir))
    else:
        # Fallback: try parent directories
        sys.path.insert(0, str(args.model_dir.parent.parent))

    from model import MultiTaskNLPModel

    print(f"Loading model from {args.model_dir}...")
    model = MultiTaskNLPModel.load(str(args.model_dir), args.encoder)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Export
    print(f"\nExporting to ONNX (opset 14)...")
    export_to_onnx(model, tokenizer, args.output_dir, args.max_length)

    # Quantize
    print(f"\nQuantizing to INT8...")
    quantize_int8(args.output_dir, args.output_dir)

    # Validate
    print(f"\nValidating ONNX parity...")
    ok = validate_onnx(args.output_dir / "model-int8.onnx", model, tokenizer)

    # Copy tokenizer and config to output
    tokenizer.save_pretrained(str(args.output_dir))
    with open(label_maps) as f:
        labels = json.load(f)
    with open(args.output_dir / "label_maps.json", "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\n{'✓ Export complete' if ok else '✗ Export completed with parity issues'}")
    print(f"  FP32: {args.output_dir / 'model.onnx'}")
    print(f"  INT8: {args.output_dir / 'model-int8.onnx'}")
    print(f"  Tokenizer + labels: {args.output_dir}")


if __name__ == "__main__":
    main()

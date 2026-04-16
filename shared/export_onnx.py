"""Export a trained multi-task model to ONNX and quantize to INT8.

Usage:
    python -m shared.export_onnx --model-dir outputs/distilroberta-nlp-en --output-dir onnx-output/
"""

import argparse
from pathlib import Path

import torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


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
        output_names=["ner_logits", "pos_logits", "dep_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "ner_logits": {0: "batch_size", 1: "sequence_length"},
            "pos_logits": {0: "batch_size", 1: "sequence_length"},
            "dep_logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
    )
    print(f"Exported FP32 ONNX model to {output_path / 'model.onnx'}")


def quantize_int8(input_path: Path, output_path: Path):
    """Quantize an ONNX model to INT8 using dynamic quantization."""
    quantize_dynamic(
        model_input=str(input_path / "model.onnx"),
        model_output=str(output_path / "model-int8.onnx"),
        weight_type=QuantType.QInt8,
    )
    fp32_size = (input_path / "model.onnx").stat().st_size / 1024 / 1024
    int8_size = (output_path / "model-int8.onnx").stat().st_size / 1024 / 1024
    print(f"Quantized: {fp32_size:.1f}MB (FP32) → {int8_size:.1f}MB (INT8)")


def validate_onnx(onnx_path: Path, model, tokenizer, test_text: str = "Caroline went to the hospital yesterday."):
    """Compare PyTorch and ONNX outputs to verify export correctness."""
    import numpy as np

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
    for i, name in enumerate(["ner_logits", "pos_logits", "dep_logits"]):
        pt_out = pt_outputs[name].numpy()
        ort_out = ort_outputs[i]
        max_diff = np.max(np.abs(pt_out - ort_out))
        print(f"  {name}: max diff = {max_diff:.6f} {'✓' if max_diff < 0.01 else '✗ MISMATCH'}")


def main():
    parser = argparse.ArgumentParser(description="Export multi-task model to ONNX")
    parser.add_argument("--model-dir", type=Path, required=True, help="Trained model directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="ONNX output directory")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    # (Implemented by each model's train.py — this is the generic export logic)
    print(f"Export from {args.model_dir} to {args.output_dir}")
    print("NOTE: Load your trained model and call export_to_onnx() + quantize_int8()")


if __name__ == "__main__":
    main()

"""INT4 weight-only quantization sweep for the student model.

Tests:
    - INT4 RTN  — round-to-nearest, no calibration (baseline)
    - INT4 HQQ  — half-quadratic, calibration-free iterative optimization
    - INT4 GPTQ — layer-wise calibrated quantization (uses our wild corpus)
    - bnb-fp4   — fixed-point 4-bit, for comparison with the bnb-nf4 we tested

All ONNX variants use block_size=128 weight quantization (only MatMul ops).
"""
from __future__ import annotations
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO / "models/kniv-deberta-nlp-base-en-base"
ONNX_DIR = MODEL_DIR / "onnx"
sys.path.insert(0, str(REPO / "models"))
from student_loader import load_student, CLS_LABELS  # noqa: E402

device = torch.device("cuda")


# ──────────────────────────────────────────────────────────────────────────────
# Calibration data reader for GPTQ
# ──────────────────────────────────────────────────────────────────────────────
from onnxruntime.quantization import CalibrationDataReader  # noqa: E402


class WildCalibReader(CalibrationDataReader):
    """Feeds the wild-corpus sentences through the ONNX model for GPTQ calibration.

    Pre-tokenizes all samples into a list (no generators on self — those can't
    be pickled, and ORT's GPTQ path serializes the reader internally).
    """
    def __init__(self, tokenizer, sentences: list[str]):
        self.samples = []
        for s in sentences:
            enc = tokenizer(s, return_tensors="pt", padding="max_length",
                            max_length=128, truncation=True)
            self.samples.append({
                "input_ids": enc["input_ids"].numpy().astype(np.int64),
                "attention_mask": enc["attention_mask"].numpy().astype(np.int64),
                "predicate_idx": np.zeros((1,), dtype=np.int64),
            })
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.samples):
            return None
        s = self.samples[self.idx]
        self.idx += 1
        return s

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def rewind(self):
        self.idx = 0


# ──────────────────────────────────────────────────────────────────────────────
# Build INT4 variants
# ──────────────────────────────────────────────────────────────────────────────
def build_int4_default(block_size: int = 128) -> Path:
    """Native ORT INT4 weight-only quant (DefaultWeightOnlyQuantConfig path).

    This uses the ORT-native quantizer (not neural_compressor's), which handles
    arbitrary weight shapes correctly. Equivalent to RTN.
    """
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig,
    )
    out = ONNX_DIR / f"cascade-int4-default-b{block_size}.onnx"
    print(f"  Quantizing → {out.name}  (block_size={block_size})")
    cfg = DefaultWeightOnlyQuantConfig(block_size=block_size, is_symmetric=False)
    q = MatMulNBitsQuantizer(
        model=str(ONNX_DIR / "cascade.onnx"),
        bits=4, block_size=block_size, is_symmetric=False,
        algo_config=cfg,
    )
    q.process()
    q.model.save_model_to_file(str(out), use_external_data_format=False)
    return out


def build_int4_hqq() -> Path:
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        MatMulNBitsQuantizer, HQQWeightOnlyQuantConfig,
    )
    out = ONNX_DIR / "cascade-int4-hqq.onnx"
    print(f"  Quantizing → {out.name}")
    cfg = HQQWeightOnlyQuantConfig(block_size=128, bits=4)
    q = MatMulNBitsQuantizer(
        model=str(ONNX_DIR / "cascade.onnx"),
        bits=4, block_size=128, is_symmetric=False,
        algo_config=cfg,
    )
    q.process()
    q.model.save_model_to_file(str(out), use_external_data_format=False)
    return out


def build_int4_gptq(tokenizer, calib_sentences) -> Path:
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        MatMulNBitsQuantizer, GPTQWeightOnlyQuantConfig,
    )
    out = ONNX_DIR / "cascade-int4-gptq.onnx"
    print(f"  Quantizing → {out.name}  (calibrating on {len(calib_sentences)} samples)")
    reader = WildCalibReader(tokenizer, calib_sentences)
    cfg = GPTQWeightOnlyQuantConfig(
        calibration_data_reader=reader,
        block_size=128, perchannel=True,
    )
    q = MatMulNBitsQuantizer(
        model=str(ONNX_DIR / "cascade.onnx"),
        bits=4, block_size=128, is_symmetric=False,
        algo_config=cfg,
    )
    q.process()
    q.model.save_model_to_file(str(out), use_external_data_format=False)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Evaluators (reuse ONNX eval pattern)
# ──────────────────────────────────────────────────────────────────────────────
def load_wild():
    return [(r["sentence"], r["cls"])
            for r in csv.DictReader(open(REPO / "data/cls_wild_samples.csv"))]


def load_locomo():
    msgs = {r["dia_id"]: r["sentence"]
            for r in csv.DictReader(open(REPO / "data/locomo50_messages.csv"))}
    gold = {r["dia_id"]: r["gold_cls"]
            for r in csv.DictReader(open(REPO / "data/locomo50_gold_labels.csv"))}
    return [(msgs[d], gold[d]) for d in gold]


def eval_onnx(onnx_path: Path, tokenizer, pairs):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    correct, times = 0, []
    for text, g in pairs:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        t0 = time.perf_counter()
        out = sess.run(None, {
            "input_ids": enc["input_ids"].numpy(),
            "attention_mask": enc["attention_mask"].numpy(),
            "predicate_idx": np.zeros((1,), dtype=np.int64),
        })
        times.append((time.perf_counter() - t0) * 1000)
        cls_logits = out[5]
        correct += int(CLS_LABELS[int(cls_logits[0].argmax())] == g)
    return correct / len(pairs), float(np.mean(times[5:]))


def to_bnb(model, qtype: str):
    import bitsandbytes as bnb
    def _swap(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear):
                new = bnb.nn.Linear4bit(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    quant_type=qtype, compute_dtype=torch.float16,
                )
                new.weight = bnb.nn.Params4bit(
                    child.weight.data.contiguous(),
                    requires_grad=False, quant_type=qtype,
                )
                if child.bias is not None:
                    new.bias = torch.nn.Parameter(child.bias.data.clone())
                setattr(parent, name, new)
            else:
                _swap(child)
    _swap(model.encoder if hasattr(model, "encoder") else model)
    return model.to(device)


def eval_pt(model, tokenizer, pairs):
    model.eval()
    correct, times = 0, []
    pidx = torch.zeros(1, dtype=torch.long, device=device)
    for text, g in pairs:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            *_, logits = model(enc["input_ids"], enc["attention_mask"], pidx)
        torch.cuda.synchronize(); times.append((time.perf_counter() - t0) * 1000)
        correct += int(CLS_LABELS[int(logits[0].argmax())] == g)
    return correct / len(pairs), float(np.mean(times[5:]))


# ──────────────────────────────────────────────────────────────────────────────
def main():
    wild = load_wild()
    locomo = load_locomo()
    _, tokenizer, _ = load_student(str(MODEL_DIR), device="cpu")

    results = []
    def record(name, sz, wa, wt, la, lt):
        results.append((name, sz, wa, wt, la, lt))
        print(f"  → size={sz:.1f} MB  wild={wa:.1%} ({wt:.2f} ms)  locomo={la:.1%} ({lt:.2f} ms)")
    def fail(name, e):
        import traceback
        print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
        traceback.print_exc()
        nan = float('nan')
        results.append((name, nan, nan, nan, nan, nan))

    # Baselines for comparison
    print("[ONNX FP32 — baseline]")
    p = ONNX_DIR / "cascade.onnx"
    sz = p.stat().st_size / 1e6
    wa, wt = eval_onnx(p, tokenizer, wild)
    la, lt = eval_onnx(p, tokenizer, locomo)
    record("ONNX FP32", sz, wa, wt, la, lt)

    print("[ONNX INT8 — baseline]")
    p = ONNX_DIR / "cascade-int8.onnx"
    sz = p.stat().st_size / 1e6
    wa, wt = eval_onnx(p, tokenizer, wild)
    la, lt = eval_onnx(p, tokenizer, locomo)
    record("ONNX INT8", sz, wa, wt, la, lt)

    # INT4 variants
    print("[ONNX INT4-Default (ORT native RTN)]")
    try:
        p = build_int4_default()
        sz = p.stat().st_size / 1e6
        wa, wt = eval_onnx(p, tokenizer, wild)
        la, lt = eval_onnx(p, tokenizer, locomo)
        record("ONNX INT4 Default", sz, wa, wt, la, lt)
    except Exception as e: fail("ONNX INT4 Default", e)

    print("[ONNX INT4-HQQ]")
    try:
        p = build_int4_hqq()
        sz = p.stat().st_size / 1e6
        wa, wt = eval_onnx(p, tokenizer, wild)
        la, lt = eval_onnx(p, tokenizer, locomo)
        record("ONNX INT4 HQQ", sz, wa, wt, la, lt)
    except Exception as e: fail("ONNX INT4 HQQ", e)

    print("[ONNX INT4-GPTQ]")
    try:
        calib = [s for s, _ in wild]  # 160 sentences
        p = build_int4_gptq(tokenizer, calib)
        sz = p.stat().st_size / 1e6
        wa, wt = eval_onnx(p, tokenizer, wild)
        la, lt = eval_onnx(p, tokenizer, locomo)
        record("ONNX INT4 GPTQ", sz, wa, wt, la, lt)
    except Exception as e: fail("ONNX INT4 GPTQ", e)

    # bnb-fp4 for comparison with bnb-nf4 from previous sweep
    print("[PT bnb-4bit fp4]")
    try:
        model, tok, _ = load_student(str(MODEL_DIR), device="cpu")
        model = to_bnb(model, "fp4")
        pt_sz = sum(
            (p.numel() * 0.5 if "weight" in n and p.dtype == torch.uint8
             else p.numel() * p.element_size())
            for n, p in model.named_parameters()
        ) / 1e6
        wa, wt = eval_pt(model, tok, wild)
        la, lt = eval_pt(model, tok, locomo)
        record("PT bnb-4bit fp4", pt_sz, wa, wt, la, lt)
        del model; torch.cuda.empty_cache()
    except Exception as e: fail("PT bnb-4bit fp4", e)

    # Report
    print()
    print("=" * 92)
    print(f"{'variant':22s}{'size MB':>10s}{'wild acc':>12s}{'wild ms':>11s}"
          f"{'locomo acc':>14s}{'locomo ms':>12s}")
    print("-" * 92)
    for name, sz, wa, wt, la, lt in results:
        try:
            print(f"{name:22s}{sz:>10.1f}{wa:>11.1%}{wt:>10.2f}{la:>13.1%}{lt:>11.2f}")
        except (ValueError, TypeError):
            print(f"{name:22s}     —          —          —            —          —")
    print("=" * 92)


if __name__ == "__main__":
    main()

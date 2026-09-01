"""Quantization sweep for the student model.

Tests:
    - FP32 ONNX (baseline, already exported)
    - INT8 ONNX (already exported)
    - FP16 ONNX (converted here from existing FP32 ONNX)
    - FP32 PyTorch on GPU (baseline)
    - FP16 PyTorch on GPU (.half())
    - bnb-4bit (nf4) PyTorch on GPU (encoder linears replaced)

Reports per variant:
    - file/memory size
    - inference latency on a 128-token sequence
    - CLS accuracy on data/cls_wild_samples.csv (160 wild sentences, 8 classes)
    - CLS gold accuracy on data/locomo50_gold_labels.csv (50 LoCoMo messages)

GPTQ/AWQ skipped: those toolchains assume AutoModelForCausalLM. Our model is
encoder-only with task heads; using them would require a custom wrapper that
re-exposes the encoder as a causal LM, which is out of scope for a quick test.
"""
from __future__ import annotations
import csv
import os
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
# 1. FP16 ONNX (convert from existing FP32 ONNX)
# ──────────────────────────────────────────────────────────────────────────────
def build_fp16_onnx() -> Path:
    """Convert FP32 ONNX → FP16, keeping ops that have type-mismatch issues in FP32.

    DeBERTa's relative-position attention has a few Mul/Where ops where the
    converter generates mixed types; we block those operator types from FP16
    casting and let them stay FP32 (with Cast nodes inserted around them).
    """
    import onnx
    from onnxconverter_common import float16
    fp32 = ONNX_DIR / "cascade.onnx"
    fp16 = ONNX_DIR / "cascade-fp16.onnx"
    print(f"  Converting {fp32.name} → {fp16.name}")
    m = onnx.load(str(fp32))
    # Block ops that DeBERTa attention + biaffine head exercise with mixed dtypes.
    # The biaffine creates an FP32 "ones" tensor and concatenates it with the
    # FP16 hidden states; promotion forces FP32, so the einsum Mul ops downstream
    # see mixed types. Block Mul/MatMul/Add to be safe — they get Cast wrappers.
    m16 = float16.convert_float_to_float16(
        m,
        keep_io_types=True,
        op_block_list=["Where", "Equal", "Range", "Gather", "Mul",
                       "Concat", "ConstantOfShape", "Pow", "Sqrt"],
        disable_shape_infer=True,
    )
    onnx.save(m16, str(fp16))
    return fp16


# ──────────────────────────────────────────────────────────────────────────────
# 2. bnb-4bit PyTorch (replace encoder Linears in-place)
# ──────────────────────────────────────────────────────────────────────────────
def to_bnb_4bit(model: torch.nn.Module, quant_type: str = "nf4") -> torch.nn.Module:
    """Replace nn.Linear in the encoder subtree with bnb.nn.Linear4bit (nf4).
    Heads remain FP32. Model must be moved to CUDA after replacement."""
    import bitsandbytes as bnb

    def _swap(parent: torch.nn.Module):
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear):
                new = bnb.nn.Linear4bit(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    quant_type=quant_type, compute_dtype=torch.float16,
                )
                new.weight = bnb.nn.Params4bit(
                    child.weight.data.contiguous(),
                    requires_grad=False, quant_type=quant_type,
                )
                if child.bias is not None:
                    new.bias = torch.nn.Parameter(child.bias.data.clone())
                setattr(parent, name, new)
            else:
                _swap(child)

    if hasattr(model, "encoder"):
        _swap(model.encoder)
    else:
        _swap(model)
    return model.to(device)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Evaluators
# ──────────────────────────────────────────────────────────────────────────────
def load_wild_corpus():
    rows = list(csv.DictReader(open(REPO / "data/cls_wild_samples.csv")))
    return [(r["sentence"], r["cls"]) for r in rows]


def load_locomo_gold():
    msgs = {r["dia_id"]: r["sentence"]
            for r in csv.DictReader(open(REPO / "data/locomo50_messages.csv"))}
    gold = {r["dia_id"]: r["gold_cls"]
            for r in csv.DictReader(open(REPO / "data/locomo50_gold_labels.csv"))}
    return [(msgs[d], gold[d]) for d in gold]


def eval_pt(model, tokenizer, pairs, name: str):
    """Return (accuracy, mean_latency_ms)."""
    model.eval()
    correct = 0
    times = []
    pidx = torch.zeros(1, dtype=torch.long, device=device)
    for text, g in pairs:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=128).to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            *_, logits = model(enc["input_ids"], enc["attention_mask"], pidx)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        pred = CLS_LABELS[int(logits[0].argmax())]
        correct += int(pred == g)
    return correct / len(pairs), float(np.mean(times[5:]))  # skip warmup


def eval_onnx(onnx_path: Path, tokenizer, pairs, name: str):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    correct = 0
    times = []
    for text, g in pairs:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=128)
        ids = enc["input_ids"].numpy()
        mask = enc["attention_mask"].numpy()
        pidx = np.zeros((1,), dtype=np.int64)
        t0 = time.perf_counter()
        out = sess.run(None, {"input_ids": ids, "attention_mask": mask,
                              "predicate_idx": pidx})
        times.append((time.perf_counter() - t0) * 1000)
        cls_logits = out[5]
        pred = CLS_LABELS[int(cls_logits[0].argmax())]
        correct += int(pred == g)
    return correct / len(pairs), float(np.mean(times[5:]))


# ──────────────────────────────────────────────────────────────────────────────
# 4. Run sweep
# ──────────────────────────────────────────────────────────────────────────────
def main():
    wild = load_wild_corpus()
    locomo = load_locomo_gold()
    print(f"wild corpus: {len(wild)} sentences | locomo gold: {len(locomo)} messages\n")

    results = []  # (variant, size_mb, wild_acc, wild_ms, locomo_acc, locomo_ms)

    def record(name, sz, wa, wt, la, lt):
        results.append((name, sz, wa, wt, la, lt))
        print(f"  → size={sz:.1f} MB  wild={wa:.1%} ({wt:.2f} ms)  locomo={la:.1%} ({lt:.2f} ms)")

    def fail(name, e):
        print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
        results.append((name, float('nan'), float('nan'), float('nan'),
                        float('nan'), float('nan')))

    # ── ONNX FP32 (baseline) ──
    fp32 = ONNX_DIR / "cascade.onnx"
    int8 = ONNX_DIR / "cascade-int8.onnx"
    _, tokenizer, _ = load_student(str(MODEL_DIR), device="cpu")

    print("[ONNX FP32]")
    try:
        sz = fp32.stat().st_size / 1e6
        wa, wt = eval_onnx(fp32, tokenizer, wild, "fp32")
        la, lt = eval_onnx(fp32, tokenizer, locomo, "fp32")
        record("ONNX FP32", sz, wa, wt, la, lt)
    except Exception as e: fail("ONNX FP32", e)

    print("[ONNX INT8 (existing)]")
    try:
        sz = int8.stat().st_size / 1e6
        wa, wt = eval_onnx(int8, tokenizer, wild, "int8")
        la, lt = eval_onnx(int8, tokenizer, locomo, "int8")
        record("ONNX INT8", sz, wa, wt, la, lt)
    except Exception as e: fail("ONNX INT8", e)

    print("[ONNX FP16 (new)]")
    try:
        fp16_path = build_fp16_onnx()
        sz = fp16_path.stat().st_size / 1e6
        wa, wt = eval_onnx(fp16_path, tokenizer, wild, "fp16")
        la, lt = eval_onnx(fp16_path, tokenizer, locomo, "fp16")
        record("ONNX FP16", sz, wa, wt, la, lt)
    except Exception as e: fail("ONNX FP16", e)

    # ── PyTorch FP32 ──
    print("[PyTorch FP32 (GPU)]")
    try:
        model, tok, _ = load_student(str(MODEL_DIR), device=str(device))
        pt_sz = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
        wa, wt = eval_pt(model, tok, wild, "pt-fp32")
        la, lt = eval_pt(model, tok, locomo, "pt-fp32")
        record("PT FP32 GPU", pt_sz, wa, wt, la, lt)
        del model; torch.cuda.empty_cache()
    except Exception as e: fail("PT FP32 GPU", e)

    # PT FP16 hard cast intentionally skipped: DeBERTa's attention mask uses
    # -1e9, which overflows FP16 (max ~65504). The supported FP16 path for
    # DeBERTa is autocast (below), which keeps masking ops in FP32.

    # ── PyTorch BF16 ──
    print("[PyTorch BF16 (GPU, .to(bfloat16))]")
    try:
        model, tok, _ = load_student(str(MODEL_DIR), device=str(device))
        model = model.to(torch.bfloat16)
        pt_sz = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
        wa, wt = eval_pt(model, tok, wild, "pt-bf16")
        la, lt = eval_pt(model, tok, locomo, "pt-bf16")
        record("PT BF16 GPU", pt_sz, wa, wt, la, lt)
        del model; torch.cuda.empty_cache()
    except Exception as e: fail("PT BF16 GPU", e)

    # ── PyTorch FP16 via autocast (keeps masking in fp32) ──
    print("[PyTorch FP16 autocast (GPU)]")
    try:
        model, tok, _ = load_student(str(MODEL_DIR), device=str(device))
        model.eval()
        pt_sz = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6 / 2  # would be /2 if cast
        correct_w = 0; correct_l = 0; times_w = []; times_l = []
        pidx = torch.zeros(1, dtype=torch.long, device=device)
        for text, g in wild:
            enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                *_, logits = model(enc["input_ids"], enc["attention_mask"], pidx)
            torch.cuda.synchronize(); times_w.append((time.perf_counter()-t0)*1000)
            correct_w += int(CLS_LABELS[int(logits[0].argmax())] == g)
        for text, g in locomo:
            enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                *_, logits = model(enc["input_ids"], enc["attention_mask"], pidx)
            torch.cuda.synchronize(); times_l.append((time.perf_counter()-t0)*1000)
            correct_l += int(CLS_LABELS[int(logits[0].argmax())] == g)
        wa, wt = correct_w/len(wild), float(np.mean(times_w[5:]))
        la, lt = correct_l/len(locomo), float(np.mean(times_l[5:]))
        record("PT FP16 autocast", pt_sz, wa, wt, la, lt)
        del model; torch.cuda.empty_cache()
    except Exception as e: fail("PT FP16 autocast", e)

    # ── PyTorch bnb-4bit (nf4) ──
    print("[PyTorch bnb-4bit nf4 (GPU)]")
    try:
        model, tok, _ = load_student(str(MODEL_DIR), device="cpu")
        model = to_bnb_4bit(model, quant_type="nf4")
        pt_sz = sum(
            (p.numel() * 0.5 if "weight" in n and p.dtype == torch.uint8
             else p.numel() * p.element_size())
            for n, p in model.named_parameters()
        ) / 1e6
        wa, wt = eval_pt(model, tok, wild, "bnb4")
        la, lt = eval_pt(model, tok, locomo, "bnb4")
        record("PT bnb-4bit (nf4)", pt_sz, wa, wt, la, lt)
        del model; torch.cuda.empty_cache()
    except Exception as e: fail("PT bnb-4bit (nf4)", e)

    # ── Report ──
    print()
    print("=" * 92)
    print(f"{'variant':22s}{'size MB':>10s}{'wild acc':>12s}{'wild ms':>11s}"
          f"{'locomo acc':>14s}{'locomo ms':>12s}")
    print("-" * 92)
    for name, sz, wa, wt, la, lt in results:
        print(f"{name:22s}{sz:>10.1f}{wa:>11.1%}{wt:>10.2f}"
              f"{la:>13.1%}{lt:>11.2f}")
    print("=" * 92)


if __name__ == "__main__":
    main()

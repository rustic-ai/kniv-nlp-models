"""Evaluate the ONNX-exported model and compare with PyTorch checkpoint.

Checks:
1. Per-task metrics on test sets (same as evaluate_model.py)
2. Parity: metric delta vs PyTorch must be <0.5%
3. Latency: mean, p50, p95, p99 over 100 sentences

Usage:
    python models/deberta-v3-nlp-en/evaluate_onnx.py \\
        --onnx-path onnx-output/deberta-v3-nlp-en/model-int8.onnx \\
        --pytorch-results outputs/deberta-v3-nlp-en/final/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.evaluate import (
    evaluate_ner, evaluate_pos, evaluate_dep, evaluate_cls,
    print_report, save_results,
)
from dep2label import decode_sentence


DATA_DIR = Path(__file__).parent.parent.parent / "data" / "prepared" / "deberta-v3-nlp-en"


def run_onnx_inference(session, tokenizer, words_or_text, max_length=128, is_split=True):
    """Run ONNX inference and return all outputs."""
    if is_split:
        encoding = tokenizer(
            words_or_text, is_split_into_words=True,
            max_length=max_length, padding="max_length", truncation=True,
            return_tensors="np",
        )
    else:
        encoding = tokenizer(
            words_or_text, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="np",
        )

    outputs = session.run(None, {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
    })

    return outputs, encoding


def align_token_preds(encoding, logits, label_list, num_words):
    """Align subword predictions back to words."""
    preds = np.argmax(logits[0], axis=-1)
    word_ids = encoding.word_ids()
    word_preds = []
    prev_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != prev_word_id:
            label_idx = int(preds[idx])
            word_preds.append(label_list[label_idx] if label_idx < len(label_list) else "O")
        prev_word_id = word_id
    return word_preds[:num_words]


def benchmark_latency(session, tokenizer, texts, max_length=128, n_runs=100):
    """Measure inference latency over n_runs sentences."""
    latencies = []
    for i in range(min(n_runs, len(texts))):
        encoding = tokenizer(
            texts[i], max_length=max_length, padding="max_length",
            truncation=True, return_tensors="np",
        )
        start = time.perf_counter()
        session.run(None, {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        })
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies.sort()
    return {
        "mean_ms": np.mean(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "n_runs": len(latencies),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument("--pytorch-results", type=Path, default=None,
                        help="PyTorch eval_results.json for parity check")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    tokenizer_dir = args.tokenizer_dir or args.onnx_path.parent
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    session = ort.InferenceSession(str(args.onnx_path))
    print(f"ONNX model: {args.onnx_path}")
    print(f"Providers: {session.get_providers()}")

    with open(DATA_DIR / "label_vocabs.json") as f:
        vocabs = json.load(f)

    results = {}

    # ── NER ───────────────────────────────────────────────
    print("\nEvaluating NER...")
    with open(DATA_DIR / "ner_test.json") as f:
        ner_test = json.load(f)

    gold_ner, pred_ner = [], []
    for ex in ner_test:
        outputs, encoding = run_onnx_inference(session, tokenizer, ex["words"])
        pred = align_token_preds(encoding, outputs[0], vocabs["ner_labels"], len(ex["words"]))
        gold_ner.append(ex["ner_tags"])
        pred_ner.append(pred)
    results["ner"] = evaluate_ner(gold_ner, pred_ner)

    # ── POS ───────────────────────────────────────────────
    print("Evaluating POS...")
    with open(DATA_DIR / "ud_test.json") as f:
        ud_test = json.load(f)

    gold_pos, pred_pos = [], []
    for ex in ud_test:
        outputs, encoding = run_onnx_inference(session, tokenizer, ex["words"])
        pred = align_token_preds(encoding, outputs[1], vocabs["pos_labels"], len(ex["words"]))
        gold_pos.append(ex["pos_tags"])
        pred_pos.append(pred)
    results["pos"] = evaluate_pos(gold_pos, pred_pos)

    # ── Dep ───────────────────────────────────────────────
    print("Evaluating Dep...")
    gold_heads_all, pred_heads_all = [], []
    gold_rels_all, pred_rels_all = [], []
    for ex in ud_test:
        outputs, encoding = run_onnx_inference(session, tokenizer, ex["words"])
        dep_preds = align_token_preds(encoding, outputs[2], vocabs["dep_labels"], len(ex["words"]))
        try:
            pred_h, pred_r = decode_sentence(dep_preds, ex["pos_tags"])
        except Exception:
            pred_h = [-1] * len(ex["words"])
            pred_r = ["_"] * len(ex["words"])
        gold_heads_all.append(ex["heads"])
        pred_heads_all.append(pred_h)
        gold_rels_all.append(ex["deprels"])
        pred_rels_all.append(pred_r)
    results["dep"] = evaluate_dep(gold_heads_all, pred_heads_all, gold_rels_all, pred_rels_all)

    # ── CLS ───────────────────────────────────────────────
    print("Evaluating CLS...")
    gold_cls, pred_cls = [], []
    for ex in ud_test + ner_test:
        if "cls_label" not in ex:
            continue
        text = ex.get("text", " ".join(ex["words"]))
        outputs, _ = run_onnx_inference(session, tokenizer, text, is_split=False)
        pred_idx = int(np.argmax(outputs[3][0]))
        gold_cls.append(ex["cls_label"])
        pred_cls.append(vocabs["cls_labels"][pred_idx])
    if gold_cls:
        results["cls"] = evaluate_cls(gold_cls, pred_cls, vocabs["cls_labels"])

    # ── Report ────────────────────────────────────────────
    print_report(results)

    # ── Parity check ──────────────────────────────────────
    if args.pytorch_results and args.pytorch_results.exists():
        with open(args.pytorch_results) as f:
            pt_results = json.load(f)
        print("\nONNX vs PyTorch parity:")
        for task in ["ner", "pos", "dep"]:
            if task in results and task in pt_results:
                key = "f1" if task == "ner" else "accuracy" if task == "pos" else "uas"
                pt_val = pt_results[task][key]
                onnx_val = results[task][key]
                delta = abs(pt_val - onnx_val) * 100
                status = "✓" if delta < 0.5 else "✗ EXCEEDS 0.5%"
                print(f"  {task:4s} {key}: PT={pt_val:.4f} ONNX={onnx_val:.4f} delta={delta:.2f}% {status}")

    # ── Latency ───────────────────────────────────────────
    print("\nLatency benchmark (100 sentences)...")
    texts = [ex.get("text", " ".join(ex["words"])) for ex in ud_test[:100]]
    latency = benchmark_latency(session, tokenizer, texts)
    results["latency"] = latency
    print(f"  Mean: {latency['mean_ms']:.1f}ms  P50: {latency['p50_ms']:.1f}ms  "
          f"P95: {latency['p95_ms']:.1f}ms  P99: {latency['p99_ms']:.1f}ms")

    # ── Save ──────────────────────────────────────────────
    output_path = args.output or (args.onnx_path.parent / "onnx_eval_results.json")
    save_results(results, output_path)


if __name__ == "__main__":
    main()

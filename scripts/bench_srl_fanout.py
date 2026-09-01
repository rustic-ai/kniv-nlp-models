"""Benchmark Option 1 (per-sentence fan-out) vs Option 2 (cross-sentence packing) for SRL.

Both produce identical SRL outputs — they differ only in batch shape:
  - Option 1: for each sentence, run one batch of k_verbs copies of that sentence.
  - Option 2: flatten all (sentence, verb) pairs corpus-wide, run fixed-size batches.

Phase A: pre-pass — run POS once per unique sentence to discover verb positions.
Phase B: time SRL under each strategy.

We measure ONLY Phase B (the SRL fan-out). Phase A is shared cost.

Usage:
    uv run python scripts/bench_srl_fanout.py --n-sentences 2000 --batch-size 32
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "models"))
sys.path.insert(0, str(REPO / "scripts"))
from student_loader import load_student, POS_LABELS  # noqa: E402
from predict_cls_longmemeval import collect_sentences  # noqa: E402

MAX_LEN = 128
VERB_ID = POS_LABELS.index("VERB")
AUX_ID = POS_LABELS.index("AUX")


def find_verb_positions(model, tokenizer, sentences, device, batch_size=32):
    """Return list of (input_ids[:L], attention_mask[:L], [verb_token_indices])."""
    out = []
    with torch.inference_mode():
        for i in range(0, len(sentences), batch_size):
            chunk = sentences[i:i + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_LEN).to(device)
            B, S = enc["input_ids"].shape
            pidx = torch.zeros(B, dtype=torch.long, device=device)
            pos_logits, *_ = model(enc["input_ids"], enc["attention_mask"], pidx)
            pos_pred = pos_logits.argmax(-1)
            mask = enc["attention_mask"].bool()
            verb_mask = ((pos_pred == VERB_ID) | (pos_pred == AUX_ID)) & mask
            for b in range(B):
                positions = verb_mask[b].nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
                if not positions:
                    continue
                L = int(mask[b].sum().item())
                out.append((
                    enc["input_ids"][b, :L].clone(),
                    enc["attention_mask"][b, :L].clone(),
                    positions,
                ))
    return out


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench_option1(model, items, device):
    """Per-sentence fan-out: one model() call per sentence, batch=k_verbs."""
    total_pairs = sum(len(p) for _, _, p in items)
    torch.manual_seed(0)
    _sync(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for ids, mask, verbs in items:
            k = len(verbs)
            b_ids = ids.unsqueeze(0).expand(k, -1)
            b_mask = mask.unsqueeze(0).expand(k, -1)
            pidx = torch.tensor(verbs, dtype=torch.long, device=device)
            _ = model(b_ids, b_mask, pidx)
    _sync(device)
    elapsed = time.perf_counter() - t0
    return elapsed, total_pairs


def bench_option2(model, items, device, batch_size, tokenizer_pad_id):
    """Cross-sentence packing: flatten (sentence,verb) pairs, fixed-size batches.

    Sentences in a batch have different lengths, so we pad to the longest in the batch.
    """
    # Flatten to (ids, mask, verb_idx) tuples.
    pairs: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    for ids, mask, verbs in items:
        for v in verbs:
            pairs.append((ids, mask, v))

    # Sort by length to reduce padding (same trick as `--bucket`).
    pairs.sort(key=lambda p: p[0].size(0))
    total_pairs = len(pairs)

    _sync(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, total_pairs, batch_size):
            chunk = pairs[i:i + batch_size]
            Lmax = max(p[0].size(0) for p in chunk)
            B = len(chunk)
            ids_b = torch.full((B, Lmax), tokenizer_pad_id, dtype=torch.long, device=device)
            mask_b = torch.zeros((B, Lmax), dtype=torch.long, device=device)
            pidx_b = torch.zeros(B, dtype=torch.long, device=device)
            for j, (ids, mask, v) in enumerate(chunk):
                L = ids.size(0)
                ids_b[j, :L] = ids
                mask_b[j, :L] = mask
                pidx_b[j] = v
            _ = model(ids_b, mask_b, pidx_b)
    _sync(device)
    elapsed = time.perf_counter() - t0
    return elapsed, total_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/home/rohit/work/dragonscale/uniko2/data/longmemeval_s_cleaned.json")
    ap.add_argument("--model", default=str(REPO / "models/kniv-deberta-nlp-base-en-base"))
    ap.add_argument("--n-sentences", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32, help="Option 2 batch size.")
    ap.add_argument("--out", default=str(REPO / "data/bench_srl_fanout.json"))
    args = ap.parse_args()

    model, tokenizer, info = load_student(args.model)
    device = next(model.parameters()).device
    print(f"[bench] loaded {info['encoder']} on {device}")

    # Collect a slice of sentences.
    raw = []
    for sid, t_idx, s_idx, role, sent in collect_sentences(Path(args.input)):
        if not sent.strip():
            continue
        raw.append(sent)
        if len(raw) >= args.n_sentences:
            break
    print(f"[bench] using {len(raw)} sentences")

    # Phase A: discover verbs (shared cost, not timed for comparison).
    tA = time.perf_counter()
    items = find_verb_positions(model, tokenizer, raw, device, batch_size=32)
    phaseA = time.perf_counter() - tA
    n_pairs = sum(len(p) for _, _, p in items)
    n_sent = len(items)
    avg_k = n_pairs / max(n_sent, 1)
    print(f"[bench] Phase A (POS pre-pass): {phaseA:.2f}s — {n_sent} sentences with verbs, "
          f"{n_pairs} (sent,verb) pairs, avg k = {avg_k:.2f}")

    # Warmup — fairness for whichever runs first.
    with torch.inference_mode():
        ids, mask, verbs = items[0]
        k = min(len(verbs), args.batch_size)
        _ = model(ids.unsqueeze(0).expand(k, -1),
                  mask.unsqueeze(0).expand(k, -1),
                  torch.tensor(verbs[:k], dtype=torch.long, device=device))

    # Phase B — strategies.
    pad_id = tokenizer.pad_token_id or 0
    t1, n1 = bench_option1(model, items, device)
    t2, n2 = bench_option2(model, items, device, args.batch_size, pad_id)
    assert n1 == n2 == n_pairs

    results = {
        "model": info["encoder"],
        "device": str(device),
        "n_sentences_with_verbs": n_sent,
        "n_pairs": n_pairs,
        "avg_verbs_per_sentence": round(avg_k, 2),
        "phase_a_pos_prepass_sec": round(phaseA, 3),
        "option1_per_sentence_fanout": {
            "elapsed_sec": round(t1, 3),
            "pairs_per_sec": round(n_pairs / t1, 1),
        },
        "option2_cross_sentence_packing": {
            "elapsed_sec": round(t2, 3),
            "pairs_per_sec": round(n_pairs / t2, 1),
            "batch_size": args.batch_size,
        },
        "speedup_opt2_over_opt1": round(t1 / t2, 2),
    }
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

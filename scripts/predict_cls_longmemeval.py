"""Run the cascade student over LongMemEval at sentence granularity.

Modeled on predict_cls_locomo.py + predict_cls_sentence_split.py:
  - reads LongMemEval JSON (list of questions, each with haystack_sessions)
  - dedupes sessions by haystack_session_id (sessions are reused across questions)
  - splits each turn into sentences using the repo's regex splitter
  - batches sentences (length-bucketed) for real GPU/CPU throughput
  - emits one row per sentence with CLS + verb/entity counts + truncation flag
  - writes a summary JSON with throughput and label distributions

Usage:
    uv run python scripts/predict_cls_longmemeval.py \
        --input /home/rohit/work/dragonscale/uniko2/data/longmemeval_s_cleaned.json \
        --model models/kniv-deberta-nlp-base-en-base \
        --limit 5000        # cap on *sentences*; omit for the full corpus
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "models"))
from student_loader import (  # noqa: E402
    load_student, CLS_LABELS, POS_LABELS, NER_LABELS,
)

MAX_LEN = 128  # matches training distribution

# Same splitter as predict_cls_sentence_split.py — kept identical for cross-eval consistency.
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_sentences(text: str) -> list[str]:
    parts = _SPLIT.split(text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def collect_sentences(input_path: Path):
    """Yield (session_id, turn_idx, sent_idx, role, sentence) for every unique sentence."""
    data = json.load(open(input_path))
    seen = set()
    for q in data:
        for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"]):
            if sid in seen:
                continue
            seen.add(sid)
            for t_idx, t in enumerate(sess):
                role = t.get("role", "")
                content = t.get("content", "")
                for s_idx, sent in enumerate(split_sentences(content)):
                    yield sid, t_idx, s_idx, role, sent


def batched(rows, batch_size):
    buf = []
    for r in rows:
        buf.append(r)
        if len(buf) == batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/home/rohit/work/dragonscale/uniko2/data/longmemeval_s_cleaned.json")
    ap.add_argument("--model", default=str(REPO / "models/kniv-deberta-nlp-base-en-base"))
    ap.add_argument("--out-csv", default=str(REPO / "data/longmemeval_predictions.csv"))
    ap.add_argument("--out-summary", default=str(REPO / "data/longmemeval_summary.json"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="Cap on sentences processed (0 = all).")
    ap.add_argument("--bucket", action="store_true",
                    help="Length-bucket sentences to reduce pad waste (changes CSV order).")
    args = ap.parse_args()

    model, tokenizer, info = load_student(args.model)
    device = next(model.parameters()).device
    print(f"[lme] loaded {info['encoder']} on {device}")

    # Pre-collect sentences so we can bucket / cap.
    sents = list(collect_sentences(Path(args.input)))
    print(f"[lme] unique sentences: {len(sents)}")
    if args.limit:
        sents = sents[: args.limit]
        print(f"[lme] limited to first {len(sents)}")

    if args.bucket:
        # Sort by content length; rows are still keyed by (sid, turn_idx, sent_idx).
        sents.sort(key=lambda r: len(r[4]))

    pos_verb_id = POS_LABELS.index("VERB")
    pos_aux_id = POS_LABELS.index("AUX")
    ner_o_id = NER_LABELS.index("O")

    cls_counter: Counter = Counter()
    verb_total = ent_total = trunc_total = 0
    tok_total = 0

    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(csv_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "session_id", "turn_idx", "sent_idx", "role",
        "predicted_cls", "cls_confidence",
        "n_tokens", "n_verbs", "n_entities", "truncated",
    ])

    t0 = time.perf_counter()
    pred_idx_buf = torch.zeros(args.batch_size, dtype=torch.long, device=device)

    with torch.inference_mode():
        for batch in batched(sents, args.batch_size):
            contents = [r[4] if r[4] else " " for r in batch]
            enc = tokenizer(
                contents, return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_LEN,
            ).to(device)
            B = enc["input_ids"].size(0)
            pidx = pred_idx_buf[:B] if B == args.batch_size else torch.zeros(B, dtype=torch.long, device=device)

            pos_logits, ner_logits, _arc, _lab, _srl, cls_logits = model(
                enc["input_ids"], enc["attention_mask"], pidx,
            )
            cls_probs = F.softmax(cls_logits, dim=-1)
            cls_top = cls_probs.argmax(-1)
            cls_conf = cls_probs.gather(-1, cls_top.unsqueeze(-1)).squeeze(-1)

            pos_pred = pos_logits.argmax(-1)
            ner_pred = ner_logits.argmax(-1)
            mask = enc["attention_mask"].bool()

            verb_per_row = ((pos_pred == pos_verb_id) | (pos_pred == pos_aux_id)) & mask
            verb_counts = verb_per_row.sum(-1).cpu().tolist()
            ent_counts = ((ner_pred != ner_o_id) & mask).sum(-1).cpu().tolist()
            tok_counts = mask.sum(-1).cpu().tolist()
            # Truncation: input was longer than what fits in MAX_LEN (last non-pad == MAX_LEN-1)
            trunc_flags = [int(c >= MAX_LEN) for c in tok_counts]

            cls_top_cpu = cls_top.cpu().tolist()
            cls_conf_cpu = cls_conf.cpu().tolist()

            for i, (sid, tidx, s_idx, role, _) in enumerate(batch):
                lbl = CLS_LABELS[cls_top_cpu[i]]
                writer.writerow([
                    sid, tidx, s_idx, role,
                    lbl, f"{cls_conf_cpu[i]:.4f}",
                    tok_counts[i], verb_counts[i], ent_counts[i], trunc_flags[i],
                ])
                cls_counter[lbl] += 1
                verb_total += verb_counts[i]
                ent_total += ent_counts[i]
                tok_total += tok_counts[i]
                trunc_total += trunc_flags[i]

    f.close()
    elapsed = time.perf_counter() - t0
    n = len(sents)

    summary = {
        "model": info["encoder"],
        "device": str(device),
        "input": str(args.input),
        "n_sentences": n,
        "elapsed_sec": round(elapsed, 2),
        "sentences_per_sec": round(n / elapsed, 1) if elapsed > 0 else None,
        "tokens_per_sec": round(tok_total / elapsed, 0) if elapsed > 0 else None,
        "avg_tokens_per_sentence": round(tok_total / n, 1) if n else None,
        "truncation_rate": round(trunc_total / n, 4) if n else None,
        "cls_distribution": dict(cls_counter.most_common()),
        "verbs_per_sentence_mean": round(verb_total / n, 2) if n else None,
        "entities_per_sentence_mean": round(ent_total / n, 2) if n else None,
        "batch_size": args.batch_size,
        "max_length": MAX_LEN,
    }
    Path(args.out_summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[lme] wrote {csv_path} and {args.out_summary}")


if __name__ == "__main__":
    main()

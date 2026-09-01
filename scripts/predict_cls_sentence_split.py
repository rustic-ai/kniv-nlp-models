"""Split each LoCoMo message into sentences, predict per-sentence, aggregate."""
from __future__ import annotations
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "models"))
sys.path.insert(0, str(REPO / "scripts"))
from student_loader import load_student, CLS_LABELS  # noqa: E402
from generate_distillation_shards import TeacherCascade  # noqa: E402

INPUT = REPO / "data/locomo50_messages.csv"
OUT = REPO / "data/locomo50_predictions_sentence_split.csv"

# Simple regex sentence splitter — splits on . ! ? followed by whitespace + capital
# Falls back gracefully on edge cases (URLs, abbreviations are rare in this dataset).
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_sentences(text: str) -> list[str]:
    parts = _SPLIT.split(text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts or [text]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student, s_tok, _ = load_student(str(REPO / "models/kniv-deberta-nlp-base-en-base"))
teacher = TeacherCascade(REPO / "models/kniv-deberta-nlp-base-en-large", device)


def student_pred(text: str) -> tuple[str, float]:
    enc = s_tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    pidx = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        *_, logits = student(enc["input_ids"], enc["attention_mask"], pidx)
    p = F.softmax(logits[0], dim=-1)
    i = int(p.argmax())
    return CLS_LABELS[i], float(p[i])


def teacher_pred(text: str) -> tuple[str, float]:
    enc = teacher.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        emb = teacher.encoder.embeddings(ids)
        out = teacher.encoder.encoder(emb, mask, output_hidden_states=True)
        layers = list(out.hidden_states)
        logits = teacher.cls_head(teacher.cls_pool(teacher.cls_sm(layers), mask))
    p = F.softmax(logits[0], dim=-1)
    i = int(p.argmax())
    return CLS_LABELS[i], float(p[i])


def aggregate(preds: list[tuple[str, float]]) -> dict[str, str]:
    """preds: list of (label, conf) per sentence. Returns dict of strategy → label."""
    labels = [p[0] for p in preds]
    confs = [p[1] for p in preds]
    out = {}
    out["last"] = labels[-1]
    out["highest_conf"] = labels[max(range(len(preds)), key=lambda i: confs[i])]
    # first_substantive: skip leading 'social'; if all social, fall back to last
    sub = [(l, c) for l, c in preds if l != "social"]
    out["first_substantive"] = sub[0][0] if sub else labels[-1]
    # majority: most common class; ties broken by highest avg confidence
    counts = Counter(labels)
    top_count = counts.most_common(1)[0][1]
    tied = [l for l, c in counts.items() if c == top_count]
    if len(tied) == 1:
        out["majority"] = tied[0]
    else:
        # tie-break by sum of confidences for that class
        sums = {l: sum(c for ll, c in preds if ll == l) for l in tied}
        out["majority"] = max(sums, key=sums.get)
    return out


rows = list(csv.DictReader(open(INPUT)))
out_rows = []
for r in rows:
    sentences = split_sentences(r["sentence"])
    s_preds = [student_pred(s) for s in sentences]
    t_preds = [teacher_pred(s) for s in sentences]
    s_agg = aggregate(s_preds)
    t_agg = aggregate(t_preds)
    out_rows.append({
        **r,
        "n_sentences": len(sentences),
        "student_per_sent": " | ".join(f"{p[0]}({p[1]:.2f})" for p in s_preds),
        "student_last": s_agg["last"],
        "student_highest_conf": s_agg["highest_conf"],
        "student_first_substantive": s_agg["first_substantive"],
        "student_majority": s_agg["majority"],
        "teacher_per_sent": " | ".join(f"{p[0]}({p[1]:.2f})" for p in t_preds),
        "teacher_last": t_agg["last"],
        "teacher_highest_conf": t_agg["highest_conf"],
        "teacher_first_substantive": t_agg["first_substantive"],
        "teacher_majority": t_agg["majority"],
    })

fields = list(rows[0].keys()) + [
    "n_sentences",
    "student_per_sent", "student_last", "student_highest_conf",
    "student_first_substantive", "student_majority",
    "teacher_per_sent", "teacher_last", "teacher_highest_conf",
    "teacher_first_substantive", "teacher_majority",
]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)
print(f"[split] wrote {OUT}")

# Score against gold
gold = {r["dia_id"]: r["gold_cls"]
        for r in csv.DictReader(open(REPO / "data/locomo50_gold_labels.csv"))}
strategies = ["last", "highest_conf", "first_substantive", "majority"]
print()
print("Per-strategy accuracy on LoCoMo (n=50):")
print(f"  {'strategy':22s}{'student':>10s}{'teacher':>10s}")
for strat in strategies:
    s = sum(1 for r in out_rows if r[f"student_{strat}"] == gold[r["dia_id"]])
    t = sum(1 for r in out_rows if r[f"teacher_{strat}"] == gold[r["dia_id"]])
    print(f"  {strat:22s}{s:>5d}/{len(out_rows):<4d}{t:>5d}/{len(out_rows):<4d}")
# Baselines for comparison
import json
prev_student = list(csv.DictReader(open(REPO / "data/locomo50_predictions.csv")))
prev_teacher = list(csv.DictReader(open(REPO / "data/locomo50_predictions_real_teacher.csv")))
ps = sum(1 for r in prev_student if r["predicted"] == gold[r["dia_id"]])
pt = sum(1 for r in prev_teacher if r["predicted"] == gold[r["dia_id"]])
print(f"  {'(baseline: whole turn)':22s}{ps:>5d}/{len(prev_student):<4d}{pt:>5d}/{len(prev_teacher):<4d}")

"""Predict CLS on LoCoMo with the prior turn prepended as context.

Format fed to tokenizer: (prev_text, curr_text) → [CLS] prev [SEP] curr [SEP]
Compares teacher (kniv-large) and student (kniv-base) with vs. without context.
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "models"))
sys.path.insert(0, str(REPO / "scripts"))
from student_loader import load_student, CLS_LABELS  # noqa: E402
from generate_distillation_shards import TeacherCascade  # noqa: E402

INPUT = REPO / "data/locomo50_messages.csv"
OUT = REPO / "data/locomo50_predictions_with_context.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ctx] loading models on {device}")
student, s_tok, _ = load_student(str(REPO / "models/kniv-deberta-nlp-base-en-base"))
teacher = TeacherCascade(REPO / "models/kniv-deberta-nlp-base-en-large", device)
print("[ctx] models ready")


def student_predict(prev: str | None, curr: str) -> tuple[str, float]:
    if prev:
        enc = s_tok(prev, curr, return_tensors="pt", truncation="only_first", max_length=128).to(device)
    else:
        enc = s_tok(curr, return_tensors="pt", truncation=True, max_length=128).to(device)
    predicate_idx = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        *_, cls_logits = student(enc["input_ids"], enc["attention_mask"], predicate_idx)
    p = F.softmax(cls_logits[0], dim=-1)
    i = int(p.argmax())
    return CLS_LABELS[i], float(p[i])


def teacher_predict(prev: str | None, curr: str) -> tuple[str, float]:
    if prev:
        enc = teacher.tokenizer(prev, curr, return_tensors="pt",
                                truncation="only_first", max_length=128)
    else:
        enc = teacher.tokenizer(curr, return_tensors="pt", truncation=True, max_length=128)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        emb = teacher.encoder.embeddings(ids)
        out = teacher.encoder.encoder(emb, mask, output_hidden_states=True)
        layers = list(out.hidden_states)
        cls_logits = teacher.cls_head(teacher.cls_pool(teacher.cls_sm(layers), mask))
    p = F.softmax(cls_logits[0], dim=-1)
    i = int(p.argmax())
    return CLS_LABELS[i], float(p[i])


rows = list(csv.DictReader(open(INPUT)))
out_rows = []
prev_text: str | None = None
prev_session: str | None = None
for r in rows:
    # Reset context across session boundaries
    ctx = prev_text if r["session"] == prev_session else None
    s_no, s_no_c = student_predict(None, r["sentence"])
    s_ctx, s_ctx_c = student_predict(ctx, r["sentence"])
    t_no, t_no_c = teacher_predict(None, r["sentence"])
    t_ctx, t_ctx_c = teacher_predict(ctx, r["sentence"])
    out_rows.append({
        **r,
        "prev_used": ctx if ctx else "",
        "student_no_ctx": s_no, "student_no_ctx_conf": f"{s_no_c:.4f}",
        "student_ctx":    s_ctx, "student_ctx_conf":    f"{s_ctx_c:.4f}",
        "teacher_no_ctx": t_no, "teacher_no_ctx_conf": f"{t_no_c:.4f}",
        "teacher_ctx":    t_ctx, "teacher_ctx_conf":    f"{t_ctx_c:.4f}",
    })
    prev_text = r["sentence"]
    prev_session = r["session"]

fields = list(rows[0].keys()) + [
    "prev_used",
    "student_no_ctx", "student_no_ctx_conf", "student_ctx", "student_ctx_conf",
    "teacher_no_ctx", "teacher_no_ctx_conf", "teacher_ctx", "teacher_ctx_conf",
]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)
print(f"[ctx] wrote {OUT}")

# Quick deltas
from collections import Counter
def dist(key): return Counter(r[key] for r in out_rows)
print()
print("Class distribution shift (no_ctx → ctx):")
print(f"  {'class':10s}{'S no':>6s}{'S ctx':>7s}{'T no':>7s}{'T ctx':>7s}")
all_classes = sorted(set().union(*(dist(k) for k in
    ['student_no_ctx','student_ctx','teacher_no_ctx','teacher_ctx'])))
for c in all_classes:
    print(f"  {c:10s}{dist('student_no_ctx')[c]:>6d}{dist('student_ctx')[c]:>7d}"
          f"{dist('teacher_no_ctx')[c]:>7d}{dist('teacher_ctx')[c]:>7d}")
flips_s = sum(1 for r in out_rows if r["student_no_ctx"] != r["student_ctx"])
flips_t = sum(1 for r in out_rows if r["teacher_no_ctx"] != r["teacher_ctx"])
print(f"\nLabel flips: student {flips_s}/{len(out_rows)}, teacher {flips_t}/{len(out_rows)}")

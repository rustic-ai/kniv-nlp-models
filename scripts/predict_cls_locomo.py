"""Predict CLS labels for locomo50_messages.csv (no gold labels)."""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "models"))
from student_loader import load_student, CLS_LABELS  # noqa: E402

INPUT = REPO / "data/locomo50_messages.csv"
OUTPUT = REPO / "data/locomo50_predictions.csv"
MODEL_DIR = REPO / "models/kniv-deberta-nlp-base-en-base"

model, tokenizer, info = load_student(str(MODEL_DIR))
device = next(model.parameters()).device
print(f"[predict] loaded {info['encoder']} on {device}")

rows = list(csv.DictReader(open(INPUT)))
print(f"[predict] {len(rows)} messages")

out_rows = []
for r in rows:
    sent = r["sentence"]
    enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128).to(device)
    predicate_idx = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        *_, cls_logits = model(enc["input_ids"], enc["attention_mask"], predicate_idx)
    probs = F.softmax(cls_logits[0], dim=-1)
    pred_id = int(probs.argmax())
    out_rows.append({**r,
                     "predicted": CLS_LABELS[pred_id],
                     "confidence": f"{float(probs[pred_id]):.4f}"})

with open(OUTPUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["predicted", "confidence"])
    w.writeheader()
    w.writerows(out_rows)
print(f"[predict] wrote {OUTPUT}")

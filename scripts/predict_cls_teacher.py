"""Predict CLS labels from the DeBERTa-v3-large teacher on the wild + locomo CSVs."""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "models/deberta-v3-large-nlp-en"))
from model import MultiTaskNLPModel  # noqa: E402

TEACHER_DIR = REPO / "outputs/deberta-v3-large-nlp-en/final"
ENCODER_ID = "microsoft/deberta-v3-large"

print(f"[teacher] loading from {TEACHER_DIR}")
model = MultiTaskNLPModel.load(str(TEACHER_DIR), encoder_name=ENCODER_ID)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.float().to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(TEACHER_DIR)
CLS_LABELS = model.cls_labels
print(f"[teacher] cls_labels: {CLS_LABELS}")


def predict_csv(input_path: Path, output_path: Path, gold_col: str | None = "cls"):
    rows = list(csv.DictReader(open(input_path)))
    print(f"[teacher] {len(rows)} rows from {input_path.name}")
    out_rows = []
    for r in rows:
        sent = r["sentence"]
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"])["cls_logits"]
        probs = F.softmax(logits[0], dim=-1)
        pred_id = int(probs.argmax())
        pred = CLS_LABELS[pred_id]
        new = {**r, "predicted": pred, "confidence": f"{float(probs[pred_id]):.4f}"}
        if gold_col and gold_col in r:
            gold = r[gold_col]
            if gold in CLS_LABELS:
                new["match"] = "1" if pred == gold else "0"
            else:
                new["match"] = "n/a"
        out_rows.append(new)
    fields = list(rows[0].keys()) + ["predicted", "confidence"]
    if gold_col and gold_col in rows[0]:
        fields.append("match")
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"[teacher] wrote {output_path}")


predict_csv(REPO / "data/cls_wild_samples.csv",
            REPO / "data/cls_wild_predictions_teacher.csv", gold_col="cls")
predict_csv(REPO / "data/locomo50_messages.csv",
            REPO / "data/locomo50_predictions_teacher.csv", gold_col=None)

"""Predict CLS labels using the real distillation teacher (kniv-deberta-nlp-base-en-large)."""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "models"))
sys.path.insert(0, str(REPO / "scripts"))
from student_loader import CLS_LABELS  # same 8 classes as student
from generate_distillation_shards import TeacherCascade  # real teacher loader

TEACHER_DIR = REPO / "models/kniv-deberta-nlp-base-en-large"


def predict_cls(teacher: TeacherCascade, sentence: str) -> tuple[str, float]:
    enc = teacher.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    ids = enc["input_ids"].to(teacher.device)
    mask = enc["attention_mask"].to(teacher.device)
    with torch.no_grad():
        emb = teacher.encoder.embeddings(ids)
        out = teacher.encoder.encoder(emb, mask, output_hidden_states=True)
        layers = list(out.hidden_states)
        cls_logits = teacher.cls_head(teacher.cls_pool(teacher.cls_sm(layers), mask))
    probs = F.softmax(cls_logits[0], dim=-1)
    pred_id = int(probs.argmax())
    return CLS_LABELS[pred_id], float(probs[pred_id])


def predict_csv(teacher: TeacherCascade, input_path: Path, output_path: Path,
                gold_col: str | None = "cls"):
    rows = list(csv.DictReader(open(input_path)))
    print(f"[teacher] {len(rows)} rows from {input_path.name}")
    out_rows = []
    for r in rows:
        pred, conf = predict_cls(teacher, r["sentence"])
        new = {**r, "predicted": pred, "confidence": f"{conf:.4f}"}
        if gold_col and gold_col in r:
            new["match"] = "1" if pred == r[gold_col] else "0"
        out_rows.append(new)
    fields = list(rows[0].keys()) + ["predicted", "confidence"]
    if gold_col and gold_col in rows[0]:
        fields.append("match")
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"[teacher] wrote {output_path}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[teacher] loading from {TEACHER_DIR} on {device}")
teacher = TeacherCascade(TEACHER_DIR, device)
print(f"[teacher] cls_labels: {CLS_LABELS}")

predict_csv(teacher, REPO / "data/cls_wild_samples.csv",
            REPO / "data/cls_wild_predictions_real_teacher.csv", gold_col="cls")
predict_csv(teacher, REPO / "data/locomo50_messages.csv",
            REPO / "data/locomo50_predictions_real_teacher.csv", gold_col=None)

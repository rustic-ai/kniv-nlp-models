"""Run the kniv student CLS head over data/cls_wild_samples.csv and write predictions."""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "models"))
from student_loader import load_student, CLS_LABELS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "models/kniv-deberta-nlp-base-en-base"))
    ap.add_argument("--input", default=str(REPO / "data/cls_wild_samples.csv"))
    ap.add_argument("--output", default=str(REPO / "data/cls_wild_predictions.csv"))
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args()

    model, tokenizer, info = load_student(args.model_dir)
    device = next(model.parameters()).device
    print(f"[eval] loaded {info['encoder']} on {device}")

    with open(args.input) as f:
        rows = list(csv.DictReader(f))
    print(f"[eval] {len(rows)} sentences")

    out_rows = []
    correct = 0
    for r in rows:
        gold, sent = r["cls"], r["sentence"]
        enc = tokenizer(sent, return_tensors="pt", truncation=True,
                        max_length=args.max_length).to(device)
        predicate_idx = torch.zeros(1, dtype=torch.long, device=device)
        with torch.no_grad():
            *_, cls_logits = model(enc["input_ids"], enc["attention_mask"], predicate_idx)
        probs = F.softmax(cls_logits[0], dim=-1)
        pred_id = int(probs.argmax())
        pred = CLS_LABELS[pred_id]
        conf = float(probs[pred_id])
        out_rows.append({"cls": gold, "sentence": sent, "predicted": pred,
                         "confidence": f"{conf:.4f}", "match": "1" if pred == gold else "0"})
        correct += int(pred == gold)

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cls", "sentence", "predicted", "confidence", "match"])
        w.writeheader()
        w.writerows(out_rows)
    acc = correct / len(rows)
    print(f"[eval] accuracy: {correct}/{len(rows)} = {acc:.3f}")
    print(f"[eval] wrote {args.output}")


if __name__ == "__main__":
    main()

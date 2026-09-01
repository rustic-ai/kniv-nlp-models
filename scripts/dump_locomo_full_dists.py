"""Dump full per-sentence CLS probabilities for all 50 LoCoMo messages, with and without punctuation."""
from __future__ import annotations
import csv
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "models"))
sys.path.insert(0, str(REPO / "scripts"))
from student_loader import CLS_LABELS  # noqa: E402
from generate_distillation_shards import TeacherCascade  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher = TeacherCascade(REPO / "models/kniv-deberta-nlp-base-en-large", device)

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_sents(text: str) -> list[str]:
    parts = [p.strip() for p in _SPLIT.split(text.strip()) if p.strip()]
    return parts or [text]


def strip_punct(s: str) -> str:
    s = s.translate(str.maketrans("", "", ".!?,;:"))
    return re.sub(r"\s+", " ", s).strip()


def predict(text: str) -> list[float]:
    enc = teacher.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        emb = teacher.encoder.embeddings(ids)
        out = teacher.encoder.encoder(emb, mask, output_hidden_states=True)
        layers = list(out.hidden_states)
        logits = teacher.cls_head(teacher.cls_pool(teacher.cls_sm(layers), mask))
    return F.softmax(logits[0], dim=-1).cpu().tolist()


gold = {r["dia_id"]: r["gold_cls"]
        for r in csv.DictReader(open(REPO / "data/locomo50_gold_labels.csv"))}
msgs = list(csv.DictReader(open(REPO / "data/locomo50_messages.csv")))

# Long-form text dump
txt_path = REPO / "data/locomo50_full_distributions.txt"
csv_path = REPO / "data/locomo50_full_distributions.csv"

txt_lines: list[str] = []
csv_rows: list[dict] = []

for m in msgs:
    g = gold.get(m["dia_id"], "")
    text = m["sentence"]
    sents = split_sents(text)
    txt_lines.append("")
    txt_lines.append(f"{m['dia_id']} [{m['speaker']}] gold={g}")
    txt_lines.append(f'  message: "{text}"')

    # Whole turn — with punctuation
    probs = predict(text)
    pairs = sorted(zip(CLS_LABELS, probs), key=lambda x: -x[1])
    txt_lines.append(f"  WHOLE TURN (with punct):")
    txt_lines.append("    " + " ".join(f"{l}={p:.3f}" for l, p in pairs))
    csv_rows.append({"dia_id": m["dia_id"], "speaker": m["speaker"], "gold": g,
                     "mode": "whole_punct", "sent_idx": 0, "text": text,
                     **{f"p_{l}": f"{probs[i]:.4f}" for i, l in enumerate(CLS_LABELS)}})

    # Whole turn — without punctuation
    text_np = strip_punct(text)
    probs = predict(text_np)
    pairs = sorted(zip(CLS_LABELS, probs), key=lambda x: -x[1])
    txt_lines.append(f"  WHOLE TURN (no punct):")
    txt_lines.append("    " + " ".join(f"{l}={p:.3f}" for l, p in pairs))
    csv_rows.append({"dia_id": m["dia_id"], "speaker": m["speaker"], "gold": g,
                     "mode": "whole_nopunct", "sent_idx": 0, "text": text_np,
                     **{f"p_{l}": f"{probs[i]:.4f}" for i, l in enumerate(CLS_LABELS)}})

    # Per-sentence — with punctuation
    txt_lines.append(f"  PER-SENTENCE (with punct):")
    for i, s in enumerate(sents, 1):
        probs = predict(s)
        pairs = sorted(zip(CLS_LABELS, probs), key=lambda x: -x[1])
        txt_lines.append(f'    {i}. "{s}"')
        txt_lines.append("       " + " ".join(f"{l}={p:.3f}" for l, p in pairs))
        csv_rows.append({"dia_id": m["dia_id"], "speaker": m["speaker"], "gold": g,
                         "mode": "sent_punct", "sent_idx": i, "text": s,
                         **{f"p_{l}": f"{probs[i2]:.4f}" for i2, l in enumerate(CLS_LABELS)}})

    # Per-sentence — without punctuation
    txt_lines.append(f"  PER-SENTENCE (no punct):")
    for i, s in enumerate(sents, 1):
        s_np = strip_punct(s)
        if not s_np:
            continue
        probs = predict(s_np)
        pairs = sorted(zip(CLS_LABELS, probs), key=lambda x: -x[1])
        txt_lines.append(f'    {i}. "{s_np}"')
        txt_lines.append("       " + " ".join(f"{l}={p:.3f}" for l, p in pairs))
        csv_rows.append({"dia_id": m["dia_id"], "speaker": m["speaker"], "gold": g,
                         "mode": "sent_nopunct", "sent_idx": i, "text": s_np,
                         **{f"p_{l}": f"{probs[i2]:.4f}" for i2, l in enumerate(CLS_LABELS)}})

with open(txt_path, "w") as f:
    f.write("\n".join(txt_lines))

with open(csv_path, "w", newline="") as f:
    fields = ["dia_id", "speaker", "gold", "mode", "sent_idx", "text"] + [f"p_{l}" for l in CLS_LABELS]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(csv_rows)

print(f"wrote {txt_path}  ({len(txt_lines)} lines)")
print(f"wrote {csv_path}  ({len(csv_rows)} rows)")

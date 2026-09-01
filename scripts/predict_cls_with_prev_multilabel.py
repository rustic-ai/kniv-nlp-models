"""Predict CLS with prev+current turn pair, full distributions, multi-label at threshold."""
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

THRESHOLD = 0.20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher = TeacherCascade(REPO / "models/kniv-deberta-nlp-base-en-large", device)


def strip_punct(s: str) -> str:
    s = s.translate(str.maketrans("", "", ".!?,;:"))
    return re.sub(r"\s+", " ", s).strip()


def predict(text_a: str | None, text_b: str) -> list[float]:
    if text_a:
        enc = teacher.tokenizer(text_a, text_b, return_tensors="pt",
                                truncation="only_first", max_length=128)
    else:
        enc = teacher.tokenizer(text_b, return_tensors="pt", truncation=True, max_length=128)
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

rows = []
prev_text = None
prev_session = None
for m in msgs:
    ctx = prev_text if m["session"] == prev_session else None
    text = m["sentence"]
    text_np = strip_punct(text)
    ctx_np = strip_punct(ctx) if ctx else None

    for mode_name, a, b in [
        ("whole_no_prev", None, text),
        ("whole_with_prev", ctx, text),
        ("whole_no_prev_nopunct", None, text_np),
        ("whole_with_prev_nopunct", ctx_np, text_np),
    ]:
        probs = predict(a, b)
        pairs = sorted(zip(CLS_LABELS, probs), key=lambda x: -x[1])
        above = [(l, p) for l, p in pairs if p >= THRESHOLD] or [pairs[0]]
        pred_top1 = pairs[0][0]
        g = gold[m["dia_id"]]
        rows.append({
            "dia_id": m["dia_id"], "speaker": m["speaker"], "gold": g,
            "mode": mode_name,
            "prev_used": a or "",
            "current": b,
            **{f"p_{l}": f"{p:.4f}" for l, p in zip(CLS_LABELS, probs)},
            "predicted": pred_top1,
            "correct": "1" if pred_top1 == g else "0",
            "multi_label": "+".join(l for l, _ in above),
            "multi_label_with_probs": " ".join(f"{l}({p:.2f})" for l, p in above),
            "gold_in_multi": "1" if g in [l for l, _ in above] else "0",
        })

    prev_text = m["sentence"]
    prev_session = m["session"]

out_path = REPO / "data/locomo50_with_prev_multilabel.csv"
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out_path}")

from collections import defaultdict
top1 = defaultdict(lambda: [0, 0])
multi = defaultdict(lambda: [0, 0])
sizes = defaultdict(list)
for r in rows:
    top1[r["mode"]][1] += 1
    top1[r["mode"]][0] += int(r["correct"])
    multi[r["mode"]][1] += 1
    multi[r["mode"]][0] += int(r["gold_in_multi"])
    sizes[r["mode"]].append(len(r["multi_label"].split("+")))

print()
print(f"threshold = {THRESHOLD}")
print(f"{'mode':28s}{'top-1':>12s}{'gold-in-multi':>16s}{'avg_set_size':>14s}")
for mode in ["whole_no_prev", "whole_with_prev", "whole_no_prev_nopunct", "whole_with_prev_nopunct"]:
    c1, n = top1[mode]; c2, _ = multi[mode]
    avg = sum(sizes[mode]) / len(sizes[mode])
    print(f"  {mode:26s}{c1:>3d}/{n:<3d} ({c1/n:.0%}) {c2:>3d}/{n:<3d} ({c2/n:.0%}){avg:>11.2f}")

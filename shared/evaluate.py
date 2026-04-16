"""Evaluation framework for multi-task NLP models.

Per-task metrics:
- NER: entity-level precision, recall, F1 (seqeval)
- POS: token-level accuracy + confusion matrix
- Dep: UAS, LAS, per-relation accuracy, by sentence length
- CLS: macro F1, per-class P/R/F1, confusion matrix
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from seqeval.metrics import classification_report as ner_report
from seqeval.metrics import f1_score as ner_f1
from seqeval.metrics import precision_score as ner_precision
from seqeval.metrics import recall_score as ner_recall


# ── NER ───────────────────────────────────────────────────────────

def evaluate_ner(gold_labels: list[list[str]], pred_labels: list[list[str]]) -> dict:
    """Evaluate NER using entity-level metrics (seqeval)."""
    return {
        "precision": ner_precision(gold_labels, pred_labels),
        "recall": ner_recall(gold_labels, pred_labels),
        "f1": ner_f1(gold_labels, pred_labels),
        "report": ner_report(gold_labels, pred_labels, output_dict=True),
    }


# ── POS ───────────────────────────────────────────────────────────

def evaluate_pos(gold_tags: list[list[str]], pred_tags: list[list[str]]) -> dict:
    """Evaluate POS tagging: accuracy + top confusion pairs."""
    correct = 0
    total = 0
    confusion: Counter = Counter()

    for gold_seq, pred_seq in zip(gold_tags, pred_tags):
        for g, p in zip(gold_seq, pred_seq):
            total += 1
            if g == p:
                correct += 1
            else:
                confusion[(g, p)] += 1

    accuracy = correct / total if total > 0 else 0.0
    top_confusions = confusion.most_common(10)

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "top_confusions": [
            {"gold": g, "pred": p, "count": c} for (g, p), c in top_confusions
        ],
    }


# ── Dependency Parsing ────────────────────────────────────────────

def evaluate_dep(
    gold_heads: list[list[int]],
    pred_heads: list[list[int]],
    gold_rels: list[list[str]],
    pred_rels: list[list[str]],
) -> dict:
    """Evaluate dependency parsing: UAS, LAS, per-relation, by length."""
    correct_unlabeled = 0
    correct_labeled = 0
    total = 0

    rel_correct: Counter = Counter()
    rel_total: Counter = Counter()

    length_buckets: dict[str, list[int]] = {
        "<10": [0, 0], "10-20": [0, 0], "20-40": [0, 0], "40+": [0, 0],
    }

    for g_heads, p_heads, g_rels, p_rels in zip(gold_heads, pred_heads, gold_rels, pred_rels):
        sent_len = len(g_heads)
        sent_correct = 0

        for gh, ph, gr, pr in zip(g_heads, p_heads, g_rels, p_rels):
            total += 1
            rel_total[gr] += 1
            if gh == ph:
                correct_unlabeled += 1
                sent_correct += 1
                if gr == pr:
                    correct_labeled += 1
                    rel_correct[gr] += 1

        bucket = "<10" if sent_len < 10 else "10-20" if sent_len < 20 else "20-40" if sent_len < 40 else "40+"
        length_buckets[bucket][0] += sent_correct
        length_buckets[bucket][1] += sent_len

    uas = correct_unlabeled / total if total > 0 else 0.0
    las = correct_labeled / total if total > 0 else 0.0

    per_relation = {}
    for rel in sorted(rel_total, key=rel_total.get, reverse=True)[:15]:
        acc = rel_correct[rel] / rel_total[rel] if rel_total[rel] > 0 else 0.0
        per_relation[rel] = {"accuracy": acc, "total": rel_total[rel]}

    by_length = {b: c / t if t > 0 else 0.0 for b, (c, t) in length_buckets.items()}

    return {
        "uas": uas, "las": las, "total": total,
        "per_relation": per_relation, "by_length": by_length,
    }


# ── Classification ────────────────────────────────────────────────

def evaluate_cls(gold_labels: list[str], pred_labels: list[str], label_names: list[str]) -> dict:
    """Evaluate sentence classification: macro F1, per-class, confusion."""
    confusion: dict[str, Counter] = defaultdict(Counter)
    for g, p in zip(gold_labels, pred_labels):
        confusion[g][p] += 1

    all_f1s = []
    per_class: dict[str, dict] = {}
    for label in label_names:
        tp = confusion[label].get(label, 0)
        fp = sum(confusion[o].get(label, 0) for o in label_names if o != label)
        fn = sum(confusion[label].get(o, 0) for o in label_names if o != label)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[label] = {"precision": p, "recall": r, "f1": f1, "support": tp + fn}
        all_f1s.append(f1)

    macro_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0
    accuracy = sum(1 for g, p in zip(gold_labels, pred_labels) if g == p) / max(len(gold_labels), 1)

    return {
        "macro_f1": macro_f1, "accuracy": accuracy,
        "per_class": per_class,
        "confusion": {g: dict(c) for g, c in confusion.items()},
    }


# ── Reporting ─────────────────────────────────────────────────────

def print_report(results: dict):
    """Print a human-readable evaluation report."""
    print("=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    if "ner" in results:
        n = results["ner"]
        print(f"\nNER:  P={n['precision']:.3f}  R={n['recall']:.3f}  F1={n['f1']:.3f}")

    if "pos" in results:
        p = results["pos"]
        print(f"\nPOS:  Accuracy={p['accuracy']:.3f}  ({p['correct']}/{p['total']})")
        if p.get("top_confusions"):
            print("  Top confusions:")
            for c in p["top_confusions"][:5]:
                print(f"    {c['gold']:8s} → {c['pred']:8s}  ({c['count']})")

    if "dep" in results:
        d = results["dep"]
        print(f"\nDep:  UAS={d['uas']:.3f}  LAS={d['las']:.3f}  ({d['total']} tokens)")
        if d.get("by_length"):
            print("  By sentence length:")
            for bucket, uas in d["by_length"].items():
                print(f"    {bucket:6s}: UAS={uas:.3f}")

    if "cls" in results:
        c = results["cls"]
        print(f"\nCLS:  Macro F1={c['macro_f1']:.3f}  Accuracy={c['accuracy']:.3f}")
        if c.get("per_class"):
            print("  Per class:")
            for label, m in c["per_class"].items():
                print(f"    {label:20s}: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  (n={m['support']})")

    print("=" * 70)


def save_results(results: dict, output_path: Path):
    """Save evaluation results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")

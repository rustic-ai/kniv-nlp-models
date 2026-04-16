"""Generic evaluation framework for multi-task NLP models.

Computes per-task metrics:
- NER: entity-level precision, recall, F1 (seqeval)
- POS: token-level accuracy
- Dep: UAS (unlabeled attachment score), LAS (labeled attachment score)
"""

import json
from pathlib import Path

from seqeval.metrics import classification_report as ner_report
from seqeval.metrics import f1_score as ner_f1


def evaluate_ner(gold_labels: list[list[str]], pred_labels: list[list[str]]) -> dict:
    """Evaluate NER using entity-level metrics (seqeval)."""
    f1 = ner_f1(gold_labels, pred_labels)
    report = ner_report(gold_labels, pred_labels, output_dict=True)
    return {"f1": f1, "report": report}


def evaluate_pos(gold_tags: list[list[str]], pred_tags: list[list[str]]) -> dict:
    """Evaluate POS tagging using token-level accuracy."""
    correct = 0
    total = 0
    for gold_seq, pred_seq in zip(gold_tags, pred_tags):
        for g, p in zip(gold_seq, pred_seq):
            total += 1
            if g == p:
                correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def evaluate_dep(gold_heads: list[list[int]], pred_heads: list[list[int]],
                 gold_rels: list[list[str]], pred_rels: list[list[str]]) -> dict:
    """Evaluate dependency parsing using UAS and LAS."""
    correct_unlabeled = 0
    correct_labeled = 0
    total = 0
    for g_heads, p_heads, g_rels, p_rels in zip(gold_heads, pred_heads, gold_rels, pred_rels):
        for gh, ph, gr, pr in zip(g_heads, p_heads, g_rels, p_rels):
            total += 1
            if gh == ph:
                correct_unlabeled += 1
                if gr == pr:
                    correct_labeled += 1
    uas = correct_unlabeled / total if total > 0 else 0.0
    las = correct_labeled / total if total > 0 else 0.0
    return {"uas": uas, "las": las, "total": total}


def save_results(results: dict, output_path: Path):
    """Save evaluation results as JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

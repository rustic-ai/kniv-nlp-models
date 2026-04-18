"""Evaluate baselines on the same test sets for comparison.

Runs spaCy models and reports NER F1, POS accuracy, and Dep UAS/LAS.

Usage:
    python models/deberta-v3-nlp-en/evaluate_baseline.py

Requires: pip install spacy && python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.evaluate import evaluate_ner, evaluate_pos, evaluate_dep, print_report, save_results


DATA_DIR = Path(__file__).parent.parent.parent / "data" / "prepared" / "deberta-v3-nlp-en"

# spaCy NER label mapping: spaCy uses PERSON/ORG/GPE/etc, CoNLL uses PER/ORG/LOC/MISC
SPACY_TO_CONLL_NER = {
    "PERSON": "PER", "ORG": "ORG", "GPE": "LOC", "LOC": "LOC",
    "FAC": "LOC", "NORP": "MISC", "EVENT": "MISC", "PRODUCT": "MISC",
    "WORK_OF_ART": "MISC", "LAW": "MISC", "LANGUAGE": "MISC",
    "DATE": "MISC", "TIME": "MISC", "PERCENT": "MISC", "MONEY": "MISC",
    "QUANTITY": "MISC", "ORDINAL": "MISC", "CARDINAL": "MISC",
}

# spaCy POS to UPOS (spaCy already uses UPOS for .pos_)
# dep_ labels differ: spaCy uses clear semantics, UD uses different conventions
SPACY_TO_UD_DEP = {
    "nsubj": "nsubj", "dobj": "obj", "iobj": "iobj", "ROOT": "root",
    "det": "det", "amod": "amod", "advmod": "advmod", "prep": "case",
    "pobj": "nmod", "aux": "aux", "neg": "advmod", "cc": "cc",
    "conj": "conj", "punct": "punct", "mark": "mark", "compound": "compound",
    "poss": "nmod:poss", "case": "case", "nsubjpass": "nsubj:pass",
    "auxpass": "aux:pass", "xcomp": "xcomp", "ccomp": "ccomp",
    "relcl": "acl:relcl", "advcl": "advcl", "nummod": "nummod",
    "appos": "appos", "acl": "acl", "dep": "dep",
}


def evaluate_spacy(model_name: str):
    """Run spaCy model on test sets and compute metrics."""
    try:
        import spacy
    except ImportError:
        print(f"spaCy not installed. Run: pip install spacy && python -m spacy download {model_name}")
        return None

    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"spaCy model '{model_name}' not found. Run: python -m spacy download {model_name}")
        return None

    print(f"\nEvaluating spaCy '{model_name}'...")

    # Load test data
    with open(DATA_DIR / "conll_test.json") as f:
        conll_test = json.load(f)
    with open(DATA_DIR / "ud_test.json") as f:
        ud_test = json.load(f)

    results = {}

    # ── NER ────────────────────────────────────────────────
    gold_ner, pred_ner = [], []
    for ex in conll_test:
        text = " ".join(ex["words"])
        doc = nlp(text)

        # Build BIO tags aligned to original words
        pred_tags = ["O"] * len(ex["words"])
        for ent in doc.ents:
            ent_type = SPACY_TO_CONLL_NER.get(ent.label_, "MISC")
            # Approximate word alignment
            for i, word in enumerate(ex["words"]):
                if word in ent.text:
                    prefix = "B" if i == 0 or pred_tags[i - 1] == "O" else "I"
                    pred_tags[i] = f"{prefix}-{ent_type}"

        gold_ner.append(ex["ner_tags"])
        pred_ner.append(pred_tags)

    results["ner"] = evaluate_ner(gold_ner, pred_ner)

    # ── POS ────────────────────────────────────────────────
    gold_pos, pred_pos = [], []
    for ex in ud_test:
        text = " ".join(ex["words"])
        doc = nlp(text)

        pred_tags = [token.pos_ for token in doc][:len(ex["words"])]
        while len(pred_tags) < len(ex["words"]):
            pred_tags.append("X")

        gold_pos.append(ex["pos_tags"])
        pred_pos.append(pred_tags)

    results["pos"] = evaluate_pos(gold_pos, pred_pos)

    # ── Dep ────────────────────────────────────────────────
    gold_heads, pred_heads = [], []
    gold_rels, pred_rels = [], []
    for ex in ud_test:
        text = " ".join(ex["words"])
        doc = nlp(text)

        p_heads = [token.head.i if token.head != token else -1 for token in doc][:len(ex["words"])]
        p_rels = [SPACY_TO_UD_DEP.get(token.dep_, token.dep_) for token in doc][:len(ex["words"])]

        while len(p_heads) < len(ex["words"]):
            p_heads.append(-1)
            p_rels.append("dep")

        gold_heads.append(ex["heads"])
        pred_heads.append(p_heads)
        gold_rels.append(ex["deprels"])
        pred_rels.append(p_rels)

    results["dep"] = evaluate_dep(gold_heads, pred_heads, gold_rels, pred_rels)

    return results


def main():
    all_results = {}

    for model_name in ["en_core_web_sm", "en_core_web_trf"]:
        results = evaluate_spacy(model_name)
        if results:
            all_results[model_name] = results
            print(f"\n--- {model_name} ---")
            print_report(results)

    if all_results:
        save_results(all_results, Path("outputs/baseline_results.json"))

    # Print comparison table
    if all_results:
        print("\n" + "=" * 70)
        print("BASELINE COMPARISON")
        print("=" * 70)
        header = f"{'Model':<25} {'NER F1':>8} {'POS Acc':>8} {'Dep UAS':>8} {'Dep LAS':>8}"
        print(header)
        print("-" * len(header))
        for name, r in all_results.items():
            ner_f1 = r.get("ner", {}).get("f1", 0)
            pos_acc = r.get("pos", {}).get("accuracy", 0)
            dep_uas = r.get("dep", {}).get("uas", 0)
            dep_las = r.get("dep", {}).get("las", 0)
            print(f"{name:<25} {ner_f1:>8.3f} {pos_acc:>8.3f} {dep_uas:>8.3f} {dep_las:>8.3f}")


if __name__ == "__main__":
    main()

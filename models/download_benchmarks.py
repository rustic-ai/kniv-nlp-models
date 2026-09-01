"""Download standard benchmark datasets for evaluating kniv-deberta-nlp-base-en-large.

Downloads datasets that cannot be redistributed due to licensing.
All datasets are downloaded from HuggingFace Hub to a local cache.

Usage:
    uv run python models/kniv-deberta-nlp-base-en-large/download_benchmarks.py

Datasets downloaded:
    - UD English EWT test (CC-BY-SA-4.0) — POS, DEP
    - OntoNotes 5.0 NER test via tner/ontonotes5 (LDC, eval use) — NER
    - CoNLL-2003 NER test via eriktks/conll2003 (Reuters, eval use) — NER
    - PropBank EWT test (CC-BY-SA-4.0) — SRL
    - DailyDialog test (CC-BY-NC-SA-4.0, eval use) — CLS
"""
from __future__ import annotations
import json, os
from pathlib import Path
from datasets import load_dataset

BENCHMARK_DIR = Path("data/benchmarks")


def download_ud_ewt():
    """UD English EWT test — already in our prepared data."""
    src = Path("data/prepared/kniv-deberta-cascade/ud_test.json")
    if src.exists():
        dst = BENCHMARK_DIR / "ud_ewt_test.json"
        if not dst.exists():
            import shutil
            shutil.copy(src, dst)
        with open(dst) as f:
            data = json.load(f)
        print(f"  UD EWT test: {len(data):,} sentences (POS + DEP)")
        return True
    print("  UD EWT test: NOT FOUND — run data preparation first")
    return False


def download_ontonotes_ner():
    """OntoNotes 5.0 NER test via tner/ontonotes5."""
    print("  Downloading OntoNotes 5.0 NER (tner/ontonotes5)...")
    try:
        ds = load_dataset("tner/ontonotes5")
        # Tag mapping (empirically verified)
        TNER_TAGS = {
            0: "O", 1: "B-CARDINAL", 2: "B-DATE", 3: "I-DATE",
            4: "B-PERSON", 5: "I-PERSON", 6: "B-NORP", 7: "B-GPE", 8: "I-GPE",
            9: "B-LAW", 10: "I-LAW", 11: "B-ORG", 12: "I-ORG",
            13: "B-PERCENT", 14: "I-PERCENT", 15: "B-ORDINAL",
            16: "B-MONEY", 17: "I-MONEY", 18: "B-WORK_OF_ART", 19: "I-WORK_OF_ART",
            20: "B-FAC", 21: "B-TIME", 22: "I-CARDINAL",
            23: "B-LOC", 24: "B-QUANTITY", 25: "I-QUANTITY", 26: "I-NORP",
            27: "I-LOC", 28: "B-PRODUCT", 29: "I-TIME",
            30: "B-EVENT", 31: "I-EVENT", 32: "I-FAC",
            33: "B-LANGUAGE", 34: "I-PRODUCT", 35: "I-LANGUAGE",
        }
        examples = []
        for ex in ds["test"]:
            examples.append({
                "words": ex["tokens"],
                "ner_tags": [TNER_TAGS.get(t, "O") for t in ex["tags"]],
            })
        dst = BENCHMARK_DIR / "ontonotes_ner_test.json"
        with open(dst, "w") as f:
            json.dump(examples, f)
        print(f"  OntoNotes NER test: {len(examples):,} sentences, 18 entity types")
        return True
    except Exception as e:
        print(f"  OntoNotes NER: FAILED — {e}")
        return False


def download_conll2003():
    """CoNLL-2003 NER test via eriktks/conll2003."""
    print("  Downloading CoNLL-2003 NER (eriktks/conll2003)...")
    try:
        ds = load_dataset("eriktks/conll2003")
        tag_names = ds["test"].features["ner_tags"].feature.names
        examples = []
        for ex in ds["test"]:
            examples.append({
                "words": ex["tokens"],
                "ner_tags": [tag_names[t] for t in ex["ner_tags"]],
            })
        dst = BENCHMARK_DIR / "conll2003_ner_test.json"
        with open(dst, "w") as f:
            json.dump(examples, f)
        print(f"  CoNLL-2003 NER test: {len(examples):,} sentences, 4 entity types (PER, ORG, LOC, MISC)")
        return True
    except Exception as e:
        print(f"  CoNLL-2003 NER: FAILED — {e}")
        return False


def download_propbank_srl():
    """PropBank EWT SRL test — already in our prepared data."""
    src = Path("data/prepared/kniv-deberta-cascade/srl_test.json")
    if src.exists():
        dst = BENCHMARK_DIR / "propbank_srl_test.json"
        if not dst.exists():
            import shutil
            shutil.copy(src, dst)
        with open(dst) as f:
            data = json.load(f)
        print(f"  PropBank EWT SRL test: {len(data):,} predicate-argument examples")
        return True
    print("  PropBank SRL test: NOT FOUND — run data preparation first")
    return False


def download_dailydialog():
    """DailyDialog test for dialog act classification."""
    print("  Downloading DailyDialog (daily_dialog)...")
    try:
        ds = load_dataset("daily_dialog")
        dd_label_names = {0: "dummy", 1: "inform", 2: "question", 3: "directive", 4: "commissive"}
        examples = []
        for conv in ds["test"]:
            utterances = conv["dialog"]
            acts = conv["act"]
            for i, (utt, act) in enumerate(zip(utterances, acts)):
                if act == 0:
                    continue
                examples.append({
                    "text": utt,
                    "prev_text": utterances[i - 1] if i > 0 else None,
                    "gold_act": act,
                    "gold_act_name": dd_label_names[act],
                })
        dst = BENCHMARK_DIR / "dailydialog_test.json"
        with open(dst, "w") as f:
            json.dump(examples, f)
        print(f"  DailyDialog test: {len(examples):,} utterances, 4 act types")
        return True
    except Exception as e:
        print(f"  DailyDialog: FAILED — {e}")
        return False


def main():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    print("Downloading standard benchmark datasets...\n")

    results = {
        "ud_ewt": download_ud_ewt(),
        "ontonotes_ner": download_ontonotes_ner(),
        "conll2003": download_conll2003(),
        "propbank_srl": download_propbank_srl(),
        "dailydialog": download_dailydialog(),
    }

    print(f"\n{'=' * 60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'=' * 60}")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name:20s} {status}")
    print(f"\nBenchmark data saved to: {BENCHMARK_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

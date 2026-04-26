"""Download SNLI + MNLI and convert to our format.

Usage:
    uv run python models/kniv-deberta-cascade-large-nlp-en/prepare_nli.py
"""
import json, os
from datasets import load_dataset

OUTPUT_DIR = "data/prepared/kniv-deberta-cascade"
LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # SNLI
    print("Loading SNLI...")
    snli = load_dataset("stanfordnlp/snli")
    snli_train, snli_dev = [], []
    for ex in snli["train"]:
        if ex["label"] == -1: continue
        snli_train.append({"premise": ex["premise"], "hypothesis": ex["hypothesis"],
                           "label": LABEL_MAP[ex["label"]], "source": "snli"})
    for ex in snli["validation"]:
        if ex["label"] == -1: continue
        snli_dev.append({"premise": ex["premise"], "hypothesis": ex["hypothesis"],
                         "label": LABEL_MAP[ex["label"]], "source": "snli"})
    print(f"  SNLI train: {len(snli_train):,}, dev: {len(snli_dev):,}")

    # MNLI
    print("Loading MNLI...")
    mnli = load_dataset("nyu-mll/multi_nli")
    mnli_train = []
    for ex in mnli["train"]:
        if ex["label"] == -1: continue
        mnli_train.append({"premise": ex["premise"], "hypothesis": ex["hypothesis"],
                           "label": LABEL_MAP[ex["label"]], "source": "mnli"})
    print(f"  MNLI train: {len(mnli_train):,}")

    # Merge
    nli_train = snli_train + mnli_train
    nli_dev = snli_dev
    print(f"\nMerged: train={len(nli_train):,}, dev={len(nli_dev):,}")

    # Label distribution
    from collections import Counter
    train_dist = Counter(e["label"] for e in nli_train)
    print(f"Train labels: {dict(train_dist)}")

    for name, data in [("nli_train", nli_train), ("nli_dev", nli_dev)]:
        path = f"{OUTPUT_DIR}/{name}.json"
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Saved {name}: {len(data):,} examples, {os.path.getsize(path)/1e6:.1f} MB")

if __name__ == "__main__":
    main()

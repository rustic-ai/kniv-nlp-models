"""Prepare training data for the multi-task model.

Loads UD English EWT (POS + dep) and CoNLL-2003 (NER), converts
dependency annotations to dep2label format, and builds unified
label vocabularies.

Output: JSON files in data/ with tokenized + labeled examples.
"""

from __future__ import annotations

import json
from pathlib import Path

import conllu
from datasets import load_dataset

from dep2label import encode_sentence, collect_label_vocabulary


DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "prepared" / "distilroberta-nlp-en"

# UD English EWT paths
UD_DIR = DATA_DIR / "ud-english-ewt"
UD_TRAIN = UD_DIR / "en_ewt-ud-train.conllu"
UD_DEV = UD_DIR / "en_ewt-ud-dev.conllu"
UD_TEST = UD_DIR / "en_ewt-ud-test.conllu"

# UPOS tagset (17 tags from Universal Dependencies)
UPOS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
    "SCONJ", "SYM", "VERB", "X",
]

# CoNLL-2003 NER tags (BIO scheme)
NER_TAGS = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
]


def load_ud_data(conllu_path: Path) -> list[dict]:
    """Load a CoNLL-U file and extract words, POS, heads, deprels."""
    with open(conllu_path) as f:
        sentences = conllu.parse(f.read())

    examples = []
    for sent in sentences:
        # Skip multi-word tokens and empty nodes
        tokens = [t for t in sent if isinstance(t["id"], int)]
        words = [t["form"] for t in tokens]
        upos = [t["upos"] for t in tokens]
        heads = [t["head"] - 1 if t["head"] > 0 else -1 for t in tokens]
        deprels = [t["deprel"] for t in tokens]

        dep_labels = encode_sentence(words, heads, deprels, upos)

        examples.append({
            "words": words,
            "pos_tags": upos,
            "dep_labels": dep_labels,
            "heads": heads,
            "deprels": deprels,
        })

    return examples


def load_conll_ner() -> dict[str, list[dict]]:
    """Load CoNLL-2003 NER data via HuggingFace Datasets."""
    dataset = load_dataset("conll2003", trust_remote_code=True)

    tag_names = dataset["train"].features["ner_tags"].feature.names

    splits = {}
    for split in ["train", "validation", "test"]:
        examples = []
        for item in dataset[split]:
            words = item["tokens"]
            ner_tags = [tag_names[t] for t in item["ner_tags"]]
            examples.append({
                "words": words,
                "ner_tags": ner_tags,
            })
        splits[split] = examples

    return splits


def build_dep_label_vocab() -> list[str]:
    """Collect all dep2label tags from training data."""
    labels = set()
    for split_path in [UD_TRAIN, UD_DEV, UD_TEST]:
        if split_path.exists():
            labels |= collect_label_vocabulary(str(split_path))
    return sorted(labels)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading UD English EWT...")
    ud_train = load_ud_data(UD_TRAIN)
    ud_dev = load_ud_data(UD_DEV)
    ud_test = load_ud_data(UD_TEST)
    print(f"  Train: {len(ud_train)}, Dev: {len(ud_dev)}, Test: {len(ud_test)}")

    print("Loading CoNLL-2003 NER...")
    conll = load_conll_ner()
    print(f"  Train: {len(conll['train'])}, Dev: {len(conll['validation'])}, Test: {len(conll['test'])}")

    print("Building dep2label vocabulary...")
    dep_labels = build_dep_label_vocab()
    print(f"  {len(dep_labels)} unique dep2label tags")

    # Save processed data
    with open(OUTPUT_DIR / "ud_train.json", "w") as f:
        json.dump(ud_train, f)
    with open(OUTPUT_DIR / "ud_dev.json", "w") as f:
        json.dump(ud_dev, f)
    with open(OUTPUT_DIR / "ud_test.json", "w") as f:
        json.dump(ud_test, f)

    with open(OUTPUT_DIR / "conll_train.json", "w") as f:
        json.dump(conll["train"], f)
    with open(OUTPUT_DIR / "conll_dev.json", "w") as f:
        json.dump(conll["validation"], f)
    with open(OUTPUT_DIR / "conll_test.json", "w") as f:
        json.dump(conll["test"], f)

    # Save label vocabularies
    with open(OUTPUT_DIR / "label_vocabs.json", "w") as f:
        json.dump({
            "ner_labels": NER_TAGS,
            "pos_labels": UPOS_TAGS,
            "dep_labels": dep_labels,
        }, f, indent=2)

    print(f"\nData prepared in {OUTPUT_DIR}")
    print(f"  NER labels: {len(NER_TAGS)}")
    print(f"  POS labels: {len(UPOS_TAGS)}")
    print(f"  Dep labels: {len(dep_labels)}")


if __name__ == "__main__":
    main()

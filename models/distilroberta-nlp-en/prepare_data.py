"""Prepare training data for the multi-task model.

Loads UD English EWT (POS + dep), CoNLL-2003 (NER), and generates
bootstrapped CLS (intent/type) labels.  Converts dependency annotations
to dep2label format and builds unified label vocabularies.

Output: JSON files in data/prepared/ with labeled examples.
"""

from __future__ import annotations

import json
import re
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

# Sentence-level classification labels
CLS_LABELS = [
    "statement",       # declarative factual content
    "question",        # genuine question
    "question_fact",   # question embedding factual content
    "command",         # imperative / request
    "greeting",        # social opener
    "filler",          # reaction / backchannel
    "acknowledgment",  # agreement / confirmation
]

# ── Bootstrap CLS labeling rules ──────────────────────────────────

GREETINGS = {
    "hey", "hi", "hello", "howdy", "hi there", "hey there",
    "good morning", "good afternoon", "good evening",
}

FILLERS = {
    "wow", "oh wow", "whoa", "omg", "lol", "haha", "hehe", "lmao",
    "thanks", "thank you", "thx", "ty",
    "no worries", "no problem", "np",
    "you're welcome", "yw",
    "sounds good", "sounds great", "cool", "nice", "awesome",
    "bye", "goodbye", "see you", "later", "ttyl",
}

ACKNOWLEDGMENTS = {
    "ok", "okay", "k", "sure", "sure thing",
    "yes", "yeah", "yep", "yup", "nope", "no",
    "right", "exactly", "agreed", "absolutely",
    "got it", "understood", "roger", "copy that",
}

INTERROGATIVE_WORDS = {
    "who", "what", "where", "when", "why", "how",
    "which", "whom", "whose",
}

AUX_QUESTION_STARTERS = {
    "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "will", "would", "shall", "should",
    "may", "might", "has", "have", "had",
}

EMBEDDED_FACT_PATTERNS = [
    "did you know", "have you heard", "is it true that",
    "do you remember", "did you see", "did you notice",
]

COMMAND_STARTERS = {
    "show", "find", "create", "delete", "remove", "add", "update",
    "run", "start", "stop", "open", "close", "check", "verify",
    "list", "get", "set", "put", "move", "copy", "send", "tell",
    "give", "make", "let", "help", "explain", "describe",
    "please",  # "Please do X" is a command
}


def classify_sentence(text: str) -> str:
    """Bootstrap CLS label using rule-based heuristics.

    This produces noisy labels — the model will learn to generalize
    beyond these rules.  Manual review of a sample is recommended
    before final training.
    """
    stripped = text.strip()
    lower = stripped.lower()
    normalized = re.sub(r"[.!?,;]+$", "", lower).strip()

    # Greetings
    if normalized in GREETINGS:
        return "greeting"

    # Fillers / reactions
    if normalized in FILLERS:
        return "filler"

    # Acknowledgments
    if normalized in ACKNOWLEDGMENTS:
        return "acknowledgment"

    # Questions
    if stripped.endswith("?"):
        # Check for embedded facts
        for pattern in EMBEDDED_FACT_PATTERNS:
            if lower.startswith(pattern):
                return "question_fact"
        return "question"

    # Commands (imperative mood)
    first_word = lower.split()[0] if lower.split() else ""
    if first_word in COMMAND_STARTERS:
        return "command"

    # Default: statement
    return "statement"


# ── Data loading ──────────────────────────────────────────────────

def load_ud_data(conllu_path: Path) -> list[dict]:
    """Load a CoNLL-U file and extract words, POS, heads, deprels, + CLS."""
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

        # Reconstruct sentence text for CLS bootstrap
        text = sent.metadata.get("text", " ".join(words))
        cls_label = classify_sentence(text)

        examples.append({
            "words": words,
            "text": text,
            "pos_tags": upos,
            "dep_labels": dep_labels,
            "cls_label": cls_label,
            "heads": heads,
            "deprels": deprels,
        })

    return examples


def load_conll_ner() -> dict[str, list[dict]]:
    """Load CoNLL-2003 NER data via HuggingFace Datasets."""
    # Use the Parquet conversion branch — the original loading script
    # is no longer supported by the datasets library.
    dataset = load_dataset("conll2003", revision="refs/convert/parquet")

    tag_names = dataset["train"].features["ner_tags"].feature.names

    splits = {}
    for split in ["train", "validation", "test"]:
        examples = []
        for item in dataset[split]:
            words = item["tokens"]
            ner_tags = [tag_names[t] for t in item["ner_tags"]]

            # Bootstrap CLS label from reconstructed text
            text = " ".join(words)
            cls_label = classify_sentence(text)

            examples.append({
                "words": words,
                "text": text,
                "ner_tags": ner_tags,
                "cls_label": cls_label,
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


def print_cls_distribution(examples: list[dict], name: str):
    """Print the distribution of CLS labels for review."""
    from collections import Counter
    counts = Counter(ex["cls_label"] for ex in examples)
    total = sum(counts.values())
    print(f"  {name} CLS distribution:")
    for label in CLS_LABELS:
        count = counts.get(label, 0)
        pct = 100 * count / total if total > 0 else 0
        print(f"    {label:20s}: {count:6d} ({pct:5.1f}%)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading UD English EWT...")
    ud_train = load_ud_data(UD_TRAIN)
    ud_dev = load_ud_data(UD_DEV)
    ud_test = load_ud_data(UD_TEST)
    print(f"  Train: {len(ud_train)}, Dev: {len(ud_dev)}, Test: {len(ud_test)}")
    print_cls_distribution(ud_train, "UD Train")

    print("Loading CoNLL-2003 NER...")
    conll = load_conll_ner()
    print(f"  Train: {len(conll['train'])}, Dev: {len(conll['validation'])}, Test: {len(conll['test'])}")
    print_cls_distribution(conll["train"], "CoNLL Train")

    print("Building dep2label vocabulary...")
    dep_labels = build_dep_label_vocab()
    print(f"  {len(dep_labels)} unique dep2label tags")

    # Save processed data
    for name, data in [("ud_train", ud_train), ("ud_dev", ud_dev), ("ud_test", ud_test)]:
        with open(OUTPUT_DIR / f"{name}.json", "w") as f:
            json.dump(data, f)

    for name, key in [("conll_train", "train"), ("conll_dev", "validation"), ("conll_test", "test")]:
        with open(OUTPUT_DIR / f"{name}.json", "w") as f:
            json.dump(conll[key], f)

    # Save label vocabularies
    with open(OUTPUT_DIR / "label_vocabs.json", "w") as f:
        json.dump({
            "ner_labels": NER_TAGS,
            "pos_labels": UPOS_TAGS,
            "dep_labels": dep_labels,
            "cls_labels": CLS_LABELS,
        }, f, indent=2)

    print(f"\nData prepared in {OUTPUT_DIR}")
    print(f"  NER labels: {len(NER_TAGS)}")
    print(f"  POS labels: {len(UPOS_TAGS)}")
    print(f"  Dep labels: {len(dep_labels)}")
    print(f"  CLS labels: {len(CLS_LABELS)}")


if __name__ == "__main__":
    main()

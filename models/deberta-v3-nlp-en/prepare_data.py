"""Prepare training data for the multi-task model.

Loads UD English EWT (POS + dep) and the kniv validated corpus (NER),
generates bootstrapped CLS (intent/type) labels, converts dependency
annotations to dep2label format, and builds unified label vocabularies.

Output: JSON files in data/prepared/ with labeled examples.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import conllu

from dep2label import encode_sentence, collect_label_vocabulary


DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "prepared" / "deberta-v3-nlp-en"
CORPUS_DIR = Path(__file__).parent.parent.parent / "corpus" / "output" / "annotated"

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

# spaCy NER entity types (BIO scheme)
# Richer than CoNLL-2003's 4 types — 18 entity types from spaCy en_core_web_trf
SPACY_ENTITY_TYPES = [
    "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT",
    "MONEY", "QUANTITY", "ORDINAL", "CARDINAL",
]

NER_TAGS = ["O"]
for etype in SPACY_ENTITY_TYPES:
    NER_TAGS.extend([f"B-{etype}", f"I-{etype}"])

# Sentence-level classification labels (uniko dialog act types)
# Each label maps to a specific action in uniko's cognitive memory pipeline.
CLS_LABELS = [
    "inform",       # extract observation (primary knowledge source)
    "correction",   # flag existing knowledge for update
    "agreement",    # reinforce existing observation
    "question",     # record knowledge gap
    "plan_commit",  # link to Goal/Task (promise, offer, suggest)
    "request",      # create action node (command, instruction)
    "feedback",     # skip extraction (acknowledgment, backchannel)
    "social",       # skip extraction (greeting, goodbye, thanks, apology)
    "filler",       # skip entirely (turn/time management, stalling)
]

# ── Bootstrap CLS labeling rules ──────────────────────────────────
# These are fallback heuristics used when GPT-classified labels
# (from corpus/pipeline/classify.py) are not yet available.

SOCIAL_PHRASES = {
    "hey", "hi", "hello", "howdy", "hi there", "hey there",
    "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "thx", "ty",
    "you're welcome", "yw", "no worries",
    "bye", "goodbye", "see you", "later", "ttyl",
    "sorry", "my apologies", "pardon",
}

FEEDBACK_PHRASES = {
    "ok", "okay", "k", "sure", "sure thing",
    "right", "got it", "understood", "roger", "copy that",
    "i see", "hmm", "hm", "mhm", "uh huh",
}

AGREEMENT_PHRASES = {
    "yes", "yeah", "yep", "yup", "absolutely",
    "exactly", "agreed", "correct", "that's right",
    "i agree", "definitely", "for sure",
}

FILLER_PHRASES = {
    "um", "uh", "well", "so", "anyway", "like",
    "you know", "i mean", "let me think",
    "wow", "oh wow", "whoa", "omg", "lol", "haha",
}

REQUEST_STARTERS = {
    "show", "find", "create", "delete", "remove", "add", "update",
    "run", "start", "stop", "open", "close", "check", "verify",
    "list", "get", "set", "put", "move", "copy", "send", "tell",
    "give", "make", "help", "explain", "describe",
    "please",
}

PLAN_STARTERS = {
    "i'll", "i will", "let's", "let us", "we could",
    "we should", "how about", "what if", "i can",
    "i could", "we'll", "i suggest", "i propose",
    "i offer", "i promise",
}

CORRECTION_STARTERS = [
    "no,", "no ", "actually,", "actually ", "that's wrong",
    "that's not", "it's not", "it wasn't", "they're not",
    "correction:", "to correct",
]


def classify_sentence(text: str) -> str:
    """Bootstrap CLS label using rule-based heuristics.

    These are fallback labels — the real labels come from
    corpus/pipeline/classify.py (GPT-nano classification).
    """
    stripped = text.strip()
    lower = stripped.lower()
    normalized = re.sub(r"[.!?,;]+$", "", lower).strip()

    # Social (greetings, thanks, apologies, goodbyes)
    if normalized in SOCIAL_PHRASES:
        return "social"

    # Filler (turn management, stalling)
    if normalized in FILLER_PHRASES:
        return "filler"

    # Feedback (acknowledgment without content)
    if normalized in FEEDBACK_PHRASES:
        return "feedback"

    # Agreement (explicit agreement)
    if normalized in AGREEMENT_PHRASES:
        return "agreement"

    # Correction (flagging wrong info)
    for pattern in CORRECTION_STARTERS:
        if lower.startswith(pattern):
            return "correction"

    # Question
    if stripped.endswith("?"):
        return "question"

    # Request (imperative commands)
    first_word = lower.split()[0] if lower.split() else ""
    if first_word in REQUEST_STARTERS:
        return "request"

    # Plan/Commit (offers, suggestions, commitments)
    for starter in PLAN_STARTERS:
        if lower.startswith(starter):
            return "plan_commit"

    # Default: inform
    return "inform"


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


def load_corpus_ner(
    domains: list[str] | None = None,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Load NER data from the kniv validated corpus.

    Extracts NER annotations from spaCy-annotated JSONL files.
    Splits into train/validation/test by document (not by sentence).
    """
    if domains is None:
        # Use all available annotated domains
        domains = [d.name for d in CORPUS_DIR.iterdir() if d.is_dir() and (d / "annotated.jsonl").exists()]

    VALIDATED_DIR = CORPUS_DIR.parent / "validated"

    all_examples = []
    for domain in sorted(domains):
        jsonl_file = CORPUS_DIR / domain / "annotated.jsonl"
        if not jsonl_file.exists():
            print(f"  ⚠ No annotations for domain '{domain}', skipping")
            continue

        # Load GPT CLS labels if available
        cls_labels_file = VALIDATED_DIR / domain / "cls_labels.jsonl"
        gpt_cls: dict[str, str] = {}
        if cls_labels_file.exists():
            with open(cls_labels_file) as f:
                for line in f:
                    r = json.loads(line)
                    if r["status"] == "ok":
                        gpt_cls[r["sent_id"]] = r["cls_label"]
            print(f"  {domain}: loaded {len(gpt_cls)} GPT CLS labels")

        count = 0
        gpt_used = 0
        with open(jsonl_file) as f:
            for line in f:
                data = json.loads(line)
                tokens = data.get("tokens", [])
                ner_spans = data.get("ner_spans", [])
                if not tokens:
                    continue

                words = [t["form"] for t in tokens]

                # Convert NER spans to BIO tags
                bio_tags = ["O"] * len(tokens)
                for span in ner_spans:
                    label = span["label"]
                    for i in range(span["start"], min(span["end"], len(tokens))):
                        prefix = "B" if i == span["start"] else "I"
                        bio_tags[i] = f"{prefix}-{label}"

                text = data.get("text", " ".join(words))
                sent_id = data.get("sent_id", "")

                # Prefer GPT CLS label, fall back to bootstrap
                if sent_id in gpt_cls:
                    cls_label = gpt_cls[sent_id]
                    gpt_used += 1
                else:
                    cls_label = classify_sentence(text)

                example = {
                    "words": words,
                    "text": text,
                    "ner_tags": bio_tags,
                    "cls_label": cls_label,
                }
                if data.get("prev_text"):
                    example["prev_text"] = data["prev_text"]
                all_examples.append(example)
                count += 1

        src = f"{gpt_used} GPT + {count - gpt_used} bootstrap" if gpt_used else "all bootstrap"
        print(f"  {domain}: {count} NER examples ({src})")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_examples)

    n = len(all_examples)
    test_end = int(n * test_ratio)
    dev_end = test_end + int(n * dev_ratio)

    return {
        "test": all_examples[:test_end],
        "validation": all_examples[test_end:dev_end],
        "train": all_examples[dev_end:],
    }


# GMB entity type mapping → our 18-type scheme
GMB_TO_SPACY = {
    "per": "PERSON",
    "org": "ORG",
    "gpe": "GPE",
    "geo": "LOC",
    "tim": "DATE",
    "art": "PRODUCT",
    "eve": "EVENT",
    "nat": "EVENT",
}


def load_gmb_ner(
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Load GMB (Groningen Meaning Bank) NER dataset from HuggingFace.

    CC BY 4.0 licensed, human-corrected annotations.
    Maps GMB's 8 entity types to our 18-type spaCy scheme.
    """
    from datasets import load_dataset

    # GMB integer tag → BIO string mapping
    GMB_INT_TAGS = [
        "O",       "B-geo", "I-geo", "B-org", "I-org",
        "B-gpe",   "I-gpe", "B-per", "I-per", "B-tim",
        "I-tim",   "B-art", "I-art", "B-eve", "I-eve",
        "B-nat",   "I-nat",
    ]

    print("  Loading GMB NER from HuggingFace...")
    dataset = load_dataset(
        "rjac/kaggle-entity-annotated-corpus-ner-dataset",
        split="train",
    )

    all_examples = []
    for row in dataset:
        words = row["tokens"]
        int_tags = row["ner_tags"]
        if not words or len(words) < 3:
            continue

        # Map integer tags → GMB strings → our spaCy scheme
        bio_tags = []
        for tag_id in int_tags:
            gmb_tag = GMB_INT_TAGS[tag_id] if tag_id < len(GMB_INT_TAGS) else "O"
            if gmb_tag == "O":
                bio_tags.append("O")
            else:
                prefix, gmb_type = gmb_tag.split("-", 1)
                mapped = GMB_TO_SPACY.get(gmb_type)
                bio_tags.append(f"{prefix}-{mapped}" if mapped else "O")

        text = " ".join(words)
        all_examples.append({
            "words": words,
            "text": text,
            "ner_tags": bio_tags,
            "cls_label": classify_sentence(text),
        })

    print(f"  GMB: {len(all_examples)} sentences loaded")

    # Shuffle and split
    random.seed(seed + 1)  # different seed to avoid correlation with corpus split
    random.shuffle(all_examples)

    n = len(all_examples)
    test_end = int(n * test_ratio)
    dev_end = test_end + int(n * dev_ratio)

    return {
        "test": all_examples[:test_end],
        "validation": all_examples[test_end:dev_end],
        "train": all_examples[dev_end:],
    }


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

    print("Loading corpus NER (kniv validated)...")
    corpus_ner = load_corpus_ner()
    print(f"  Corpus — Train: {len(corpus_ner['train'])}, Dev: {len(corpus_ner['validation'])}, Test: {len(corpus_ner['test'])}")

    print("Loading GMB NER (CC BY 4.0, human-corrected)...")
    gmb_ner = load_gmb_ner()
    print(f"  GMB — Train: {len(gmb_ner['train'])}, Dev: {len(gmb_ner['validation'])}, Test: {len(gmb_ner['test'])}")

    # Merge: corpus + GMB
    ner_data = {
        split: corpus_ner[split] + gmb_ner[split]
        for split in ["train", "validation", "test"]
    }
    print(f"  Combined — Train: {len(ner_data['train'])}, Dev: {len(ner_data['validation'])}, Test: {len(ner_data['test'])}")
    print_cls_distribution(ner_data["train"], "NER Train")

    print("Building dep2label vocabulary...")
    dep_labels = build_dep_label_vocab()
    print(f"  {len(dep_labels)} unique dep2label tags")

    # Save processed data
    for name, data in [("ud_train", ud_train), ("ud_dev", ud_dev), ("ud_test", ud_test)]:
        with open(OUTPUT_DIR / f"{name}.json", "w") as f:
            json.dump(data, f)

    for name, key in [("ner_train", "train"), ("ner_dev", "validation"), ("ner_test", "test")]:
        with open(OUTPUT_DIR / f"{name}.json", "w") as f:
            json.dump(ner_data[key], f)

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

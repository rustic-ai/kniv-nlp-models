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
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "prepared" / "deberta-v3-large-nlp-en"
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
    corpus_dir: Path | None = None,
    validated_dir: Path | None = None,
) -> dict[str, list[dict]]:
    """Load NER data from the kniv validated corpus.

    Extracts NER annotations from spaCy-annotated JSONL files.
    Splits into train/validation/test by document (not by sentence).
    """
    corpus_dir = corpus_dir or CORPUS_DIR
    if domains is None:
        domains = [d.name for d in corpus_dir.iterdir() if d.is_dir() and (d / "annotated.jsonl").exists()]

    VALIDATED_DIR = validated_dir or (CORPUS_DIR.parent / "validated")

    all_examples = []
    for domain in sorted(domains):
        jsonl_file = corpus_dir / domain / "annotated.jsonl"
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
                    "_sent_id": data.get("sent_id", f"{domain}-{count}"),
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


# ── Few-NERD ─────────────────────────────────────────────────────

# Few-NERD coarse types → our 18-type spaCy scheme
FEWNERD_TO_SPACY = {
    "person": "PERSON",
    "organization": "ORG",
    "location": "GPE",          # Few-NERD "location" includes both GPE and LOC
    "building": "FAC",
    "product": "PRODUCT",
    "event": "EVENT",
    "art": "WORK_OF_ART",
    # "other" subtypes mapped individually below
}

# Fine-grained "other-*" types that map to our scheme
FEWNERD_OTHER_TO_SPACY = {
    "other-language": "LANGUAGE",
    "other-law": "LAW",
    "other-currency": "MONEY",
}

# Numeric/temporal types that spaCy extracts (Few-NERD doesn't annotate these)
SPACY_NUMERIC_TYPES = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}


def load_few_nerd(
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    enrich_numeric: bool = True,
) -> dict[str, list[dict]]:
    """Load Few-NERD dataset, map to our 18-type scheme, enrich with spaCy numerics.

    Few-NERD has excellent human annotations for contextual entities but
    completely ignores numeric/temporal types (DATE, TIME, MONEY, etc.).
    ~49% of sentences contain such entities tagged as O.

    When enrich_numeric=True, runs spaCy on each sentence to extract
    numeric types, merging them with Few-NERD's annotations (Few-NERD
    takes priority on conflicts).
    """
    from datasets import load_dataset

    print("  Loading Few-NERD from HuggingFace...")
    dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")

    # Get label name lists from the dataset features
    coarse_names = dataset["train"].features["ner_tags"].feature.names
    fine_names = dataset["train"].features["fine_ner_tags"].feature.names

    # Load spaCy for numeric enrichment
    nlp = None
    if enrich_numeric:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        print("  spaCy loaded for numeric enrichment")

    all_examples = []
    enriched_count = 0

    for split_name in ["train", "validation", "test"]:
        split_data = dataset[split_name]

        for row in split_data:
            tokens = row["tokens"]
            coarse_ids = row["ner_tags"]       # integer IDs
            fine_ids = row["fine_ner_tags"]     # integer IDs

            if not tokens or len(tokens) < 3:
                continue

            # Convert integer IDs to string names
            coarse_tags = [coarse_names[i] for i in coarse_ids]
            fine_tags = [fine_names[i] for i in fine_ids]

            # Map Few-NERD tags to our BIO scheme
            bio_tags = []
            for i, (coarse, fine) in enumerate(zip(coarse_tags, fine_tags)):
                if coarse == "O":
                    bio_tags.append("O")
                    continue

                # Determine prefix (B or I) — Few-NERD uses flat tags,
                # so we add B- for the first token of each entity span
                is_start = (i == 0 or coarse_tags[i - 1] != coarse
                            or fine_tags[i - 1] != fine)
                prefix = "B" if is_start else "I"

                # Map coarse type
                mapped = FEWNERD_TO_SPACY.get(coarse)

                # Try fine-grained "other-*" mapping
                if not mapped and fine:
                    mapped = FEWNERD_OTHER_TO_SPACY.get(fine)

                if mapped:
                    bio_tags.append(f"{prefix}-{mapped}")
                else:
                    bio_tags.append("O")  # unmapped "other" subtypes

            # Enrich with spaCy numeric entities
            if nlp:
                text = " ".join(tokens)
                doc = nlp(text)

                # Build character-to-token index
                char_pos = 0
                char_to_tok = {}
                for ti, tok in enumerate(tokens):
                    start = text.find(tok, char_pos)
                    if start == -1:
                        start = char_pos
                    for c in range(start, start + len(tok)):
                        char_to_tok[c] = ti
                    char_pos = start + len(tok)

                added_numeric = False
                for ent in doc.ents:
                    if ent.label_ not in SPACY_NUMERIC_TYPES:
                        continue

                    # Find token range for this entity
                    tok_start = char_to_tok.get(ent.start_char)
                    tok_end = char_to_tok.get(ent.end_char - 1)
                    if tok_start is None or tok_end is None:
                        continue

                    # Only add if ALL tokens in range are currently O
                    if all(bio_tags[t] == "O" for t in range(tok_start, tok_end + 1)):
                        bio_tags[tok_start] = f"B-{ent.label_}"
                        for t in range(tok_start + 1, tok_end + 1):
                            bio_tags[t] = f"I-{ent.label_}"
                        added_numeric = True

                if added_numeric:
                    enriched_count += 1

            text = " ".join(tokens)
            all_examples.append({
                "words": tokens,
                "text": text,
                "ner_tags": bio_tags,
                "cls_label": classify_sentence(text),
            })

    print(f"  Few-NERD: {len(all_examples)} sentences loaded")
    if enrich_numeric:
        print(f"  Numeric enrichment: {enriched_count} sentences got spaCy entities added")

    # Shuffle and split
    rng = random.Random(seed + 2)
    rng.shuffle(all_examples)

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


def load_from_gold_parquet(parquet_path: Path) -> dict[str, list[dict]]:
    """Load corpus data from gold-filtered Parquet files.

    Returns train/validation/test splits with the same format as load_corpus_ner().
    """
    import pandas as pd

    splits = {}
    for split_name, file_name in [("train", "train.parquet"),
                                   ("validation", "dev.parquet"),
                                   ("test", "test.parquet")]:
        path = parquet_path / file_name
        if not path.exists():
            # Try alternate naming
            for p in parquet_path.glob(f"{file_name.split('.')[0]}*.parquet"):
                path = p
                break
        if not path.exists():
            print(f"  Warning: {path} not found, skipping {split_name}")
            splits[split_name] = []
            continue

        df = pd.read_parquet(path)
        examples = []
        for _, row in df.iterrows():
            ex = {
                "words": list(row["tokens"]),
                "text": row["text"],
                "ner_tags": list(row["ner_tags"]),
                "cls_label": row.get("cls", "inform"),
                "_sent_id": row.get("sent_id", ""),
            }
            prev_text = row.get("prev_text")
            if prev_text and isinstance(prev_text, str) and prev_text.strip():
                ex["prev_text"] = prev_text
            examples.append(ex)

        splits[split_name] = examples
        print(f"  {split_name}: {len(examples)} examples from Parquet")

    return splits


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--gold", type=Path, default=None,
                        help="Path to gold-filtered Parquet dir (e.g. corpus/output/final-gold)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # UD data — always from CoNLL-U (expert-annotated, not filtered)
    ud_train_path = UD_TRAIN
    ud_dev_path = UD_DEV
    ud_test_path = UD_TEST

    print("Loading UD English EWT...")
    ud_train = load_ud_data(ud_train_path)
    ud_dev = load_ud_data(ud_dev_path)
    ud_test = load_ud_data(ud_test_path)
    print(f"  Train: {len(ud_train)}, Dev: {len(ud_dev)}, Test: {len(ud_test)}")
    print_cls_distribution(ud_train, "UD Train")

    if args.gold:
        print(f"Loading gold-filtered corpus from Parquet: {args.gold}...")
        corpus_ner = load_from_gold_parquet(args.gold)
    else:
        print("Loading corpus NER (kniv validated, spaCy annotations)...")
        corpus_ner = load_corpus_ner()
    print(f"  Corpus — Train: {len(corpus_ner['train'])}, Dev: {len(corpus_ner['validation'])}, Test: {len(corpus_ner['test'])}")

    # Subsample for manageable training size
    # Conversation gets highest weight (richest CLS, has prev_text)
    SUBSAMPLE_TARGETS = {
        "conversation": 15000,
        "business": 10000,
        "technical": 7000,
        "narrative": 5000,
        "news": 5000,
        "encyclopedic": 3000,
    }

    def subsample_by_domain(examples: list[dict], targets: dict, seed: int = 42) -> list[dict]:
        """Subsample examples to target counts per domain."""
        from collections import defaultdict
        by_domain = defaultdict(list)
        for ex in examples:
            sent_id = ex.get("_sent_id", "")
            domain = sent_id.split("-")[0] if "-" in sent_id else "unknown"
            by_domain[domain].append(ex)

        rng = random.Random(seed)
        sampled = []
        for domain, target in targets.items():
            pool = by_domain.get(domain, [])
            if not pool:
                continue
            n = min(target, len(pool))
            sampled.extend(rng.sample(pool, n))
            print(f"    {domain}: {n} / {len(pool)}", flush=True)

        # Include any domains not in targets (pass through)
        for domain, pool in by_domain.items():
            if domain not in targets and domain != "unknown":
                sampled.extend(pool)
                print(f"    {domain}: {len(pool)} (all)", flush=True)

        rng.shuffle(sampled)
        return sampled

    print(f"  Corpus subsampling:")
    corpus_train_sampled = subsample_by_domain(corpus_ner["train"], SUBSAMPLE_TARGETS)
    print(f"  Corpus subsampled: {len(corpus_train_sampled)} / {len(corpus_ner['train'])}")

    # Few-NERD: 131K human-annotated sentences + spaCy numeric enrichment
    print("\nLoading Few-NERD (human-annotated + spaCy numeric enrichment)...")
    few_nerd = load_few_nerd(enrich_numeric=True)
    print(f"  Few-NERD — Train: {len(few_nerd['train'])}, Dev: {len(few_nerd['validation'])}, Test: {len(few_nerd['test'])}")

    # Combine: kniv corpus (domain-balanced) + Few-NERD
    ner_data = {
        "train": corpus_train_sampled + few_nerd["train"],
        "validation": corpus_ner["validation"] + few_nerd["validation"],
        "test": corpus_ner["test"] + few_nerd["test"],
    }
    random.shuffle(ner_data["train"])
    print(f"  Combined — Train: {len(ner_data['train'])}, Dev: {len(ner_data['validation'])}, Test: {len(ner_data['test'])}")
    print_cls_distribution(ner_data["train"], "NER Train")

    print("Building dep2label vocabulary...")
    dep_labels = build_dep_label_vocab()
    print(f"  {len(dep_labels)} unique dep2label tags")

    # Save processed data
    for name, data in [("ud_train", ud_train), ("ud_dev", ud_dev), ("ud_test", ud_test)]:
        with open(OUTPUT_DIR / f"{name}.json", "w") as f:
            json.dump(data, f)

    # Strip internal _sent_id before saving
    def clean_examples(examples):
        return [{k: v for k, v in ex.items() if not k.startswith("_")} for ex in examples]

    for name, key in [("ner_train", "train"), ("ner_dev", "validation"), ("ner_test", "test")]:
        with open(OUTPUT_DIR / f"{name}.json", "w") as f:
            json.dump(clean_examples(ner_data[key]), f)

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

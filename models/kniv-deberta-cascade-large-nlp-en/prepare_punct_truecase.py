"""Generate self-supervised punctuation restoration and truecasing data.

Usage:
    uv run python models/kniv-deberta-cascade-large-nlp-en/prepare_punct_truecase.py
"""
import json, os, random
from pathlib import Path

CORPUS_DIR = Path("corpus/output/annotated")
OUTPUT_DIR = Path("data/prepared/kniv-deberta-cascade")
MAX_SENTENCES = 200000
SEED = 42

PUNCT_MAP = {
    ",": "COMMA", ".": "PERIOD", "?": "QUESTION", "!": "EXCLAIM",
    ":": "COLON", ";": "SEMICOLON",
}

def classify_case(token):
    if token.isupper() and len(token) > 1: return "ALL_CAPS"
    if token[0].isupper(): return "UPPER"
    if token.islower(): return "LOWER"
    return "MIXED"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)

    sentences = []
    for domain_dir in sorted(CORPUS_DIR.iterdir()):
        jsonl = domain_dir / "annotated.jsonl"
        if not jsonl.exists(): continue
        count = 0
        with open(jsonl) as f:
            for line in f:
                if len(sentences) >= MAX_SENTENCES: break
                obj = json.loads(line)
                tokens = obj.get("tokens", [])
                words = [t["form"] for t in tokens]
                if len(words) < 4: continue
                sentences.append({"words": words, "text": obj.get("text", " ".join(words)),
                                  "domain": domain_dir.name})
                count += 1
        print(f"  {domain_dir.name}: {count:,}")
        if len(sentences) >= MAX_SENTENCES: break

    print(f"Total sentences: {len(sentences):,}")
    random.shuffle(sentences)

    # Build punct and truecase examples
    punct_examples = []
    truecase_examples = []

    for sent in sentences:
        words = sent["words"]

        # Punctuation: label is what punctuation follows each word
        punct_labels = []
        clean_words = []
        for w in words:
            # Check if word itself is punctuation
            if w in PUNCT_MAP:
                # Attach to previous word's label
                if punct_labels:
                    punct_labels[-1] = PUNCT_MAP[w]
                continue
            # Check trailing punctuation
            label = "NONE"
            clean = w
            if len(w) > 1 and w[-1] in PUNCT_MAP:
                label = PUNCT_MAP[w[-1]]
                clean = w[:-1]
            clean_words.append(clean)
            punct_labels.append(label)

        if len(clean_words) >= 4:
            punct_examples.append({"words": clean_words, "punct_labels": punct_labels})

        # Truecasing
        if len(words) >= 4:
            truecase_labels = [classify_case(w) for w in words]
            words_lower = [w.lower() for w in words]
            truecase_examples.append({
                "words_lower": words_lower,
                "words_original": words,
                "truecase_labels": truecase_labels,
            })

    # Split train/dev (5%)
    def split_data(data, dev_ratio=0.05):
        n_dev = int(len(data) * dev_ratio)
        return data[n_dev:], data[:n_dev]

    random.shuffle(punct_examples)
    random.shuffle(truecase_examples)
    punct_train, punct_dev = split_data(punct_examples)
    truecase_train, truecase_dev = split_data(truecase_examples)

    # Save
    for name, data in [("punct_train", punct_train), ("punct_dev", punct_dev),
                        ("truecase_train", truecase_train), ("truecase_dev", truecase_dev)]:
        path = OUTPUT_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Saved {name}: {len(data):,} examples, {os.path.getsize(path)/1e6:.1f} MB")

    # Label distributions
    from collections import Counter
    punct_dist = Counter(l for ex in punct_train for l in ex["punct_labels"])
    truecase_dist = Counter(l for ex in truecase_train for l in ex["truecase_labels"])
    print(f"\nPunct label distribution: {dict(punct_dist.most_common())}")
    print(f"Truecase label distribution: {dict(truecase_dist.most_common())}")

if __name__ == "__main__":
    main()

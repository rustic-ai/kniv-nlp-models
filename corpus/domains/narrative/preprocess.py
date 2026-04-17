"""Preprocess Gutenberg books into individual sentences."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "raw" / "narrative"
DOMAIN_DIR = Path(__file__).parent

MIN_WORDS = 5
MAX_WORDS = 80
MIN_CHARS = 15


def load_skip_patterns() -> list[re.Pattern]:
    """Load skip patterns from config."""
    with open(DOMAIN_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)
    patterns = config.get("filtering", {}).get("skip_patterns", [])
    return [re.compile(p) for p in patterns]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with basic abbreviation awareness."""
    # Common abbreviations that shouldn't trigger sentence splits
    abbrevs = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs",
               "etc", "inc", "ltd", "vol", "no", "fig", "approx"}

    sentences = []
    current = []
    words = text.split()

    for i, word in enumerate(words):
        current.append(word)

        # Check if this word ends a sentence
        if word and word[-1] in ".!?":
            # Check for abbreviation
            base = word.rstrip(".!?").lower()
            if base in abbrevs:
                continue  # Don't split on abbreviations

            # Check for ellipsis
            if word.endswith("..."):
                continue

            # Check for initials (single letter + period)
            if len(base) == 1 and word.endswith("."):
                continue

            sentence = " ".join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []

    # Don't forget trailing text
    if current:
        sentence = " ".join(current).strip()
        if sentence:
            sentences.append(sentence)

    return sentences


def is_valid_sentence(text: str, skip_patterns: list[re.Pattern]) -> bool:
    """Check quality thresholds."""
    words = text.split()
    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return False
    if len(text) < MIN_CHARS:
        return False

    # Skip patterns (chapter headers, etc.)
    for pattern in skip_patterns:
        if pattern.search(text):
            return False

    # Skip all-caps lines
    if text == text.upper() and len(text) > 20:
        return False

    # Must have reasonable alpha ratio
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.7:
        return False

    return True


def main():
    text_file = OUTPUT_DIR / "all_books.txt"
    if not text_file.exists():
        raise FileNotFoundError(f"No text file at {text_file}. Run collect.py first.")

    skip_patterns = load_skip_patterns()
    text = text_file.read_text(encoding="utf-8", errors="replace")

    # Split into paragraphs first, then sentences
    paragraphs = text.split("\n\n")
    sentences = []
    seen = set()

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Join wrapped lines within paragraph
        para = re.sub(r"\n(?!\n)", " ", para)
        para = re.sub(r"\s+", " ", para)

        for sent in split_sentences(para):
            if not is_valid_sentence(sent, skip_patterns):
                continue

            key = sent.lower()
            if key in seen:
                continue
            seen.add(key)

            sentences.append({
                "text": sent,
                "source": "gutenberg",
                "domain": "narrative",
            })

    # Write output
    output_file = OUTPUT_DIR / "sentences.jsonl"
    with open(output_file, "w") as f:
        for sent in sentences:
            f.write(json.dumps(sent) + "\n")

    print(f"Preprocessed {len(sentences)} sentences from Gutenberg → {output_file}")


if __name__ == "__main__":
    main()

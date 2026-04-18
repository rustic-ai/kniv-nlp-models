"""Preprocess conversation utterances into individual sentences.

Reads utterances from all collected sources, filters, deduplicates,
and outputs sentences.jsonl.

Usage:
    python -m corpus.domains.conversation.preprocess
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

DOMAIN_DIR = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "raw" / "conversation"


def load_config() -> dict:
    with open(DOMAIN_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    if text.count('"') == 1:
        text = text.replace('"', "")
    return text


def split_sentences(text: str) -> list[str]:
    sentences, current = [], []
    for word in text.split():
        current.append(word)
        if word and word[-1] in ".!?":
            base = word.rstrip(".!?").lower()
            if base in {"mr", "mrs", "ms", "dr", "etc", "vs", "e.g", "i.e"} or word.endswith("..."):
                continue
            sentence = " ".join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
    if current:
        sentence = " ".join(current).strip()
        if sentence:
            sentences.append(sentence)
    return sentences


def is_valid(text: str, min_words: int, max_words: int, min_chars: int,
             skip_patterns: list) -> bool:
    words = text.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    if len(text) < min_chars:
        return False
    for pattern in skip_patterns:
        if pattern.search(text):
            return False
    if text == text.upper() and len(text) > 30:
        return False
    alpha = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
    if alpha < 0.7:
        return False
    return True


def main():
    config = load_config()
    filt = config.get("filtering", {})
    min_words = filt.get("min_words", 5)
    max_words = filt.get("max_words", 100)
    min_chars = filt.get("min_chars", 15)
    skip_patterns = [re.compile(p) for p in filt.get("skip_patterns", [])]

    all_sentences = []
    for source_dir in sorted(OUTPUT_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        for jsonl_file in sorted(source_dir.glob("*.jsonl")):
            count = 0
            with open(jsonl_file) as f:
                for line in f:
                    data = json.loads(line)
                    text = clean_text(data.get("text", ""))
                    if not text:
                        continue
                    for sent in split_sentences(text):
                        all_sentences.append({
                            "text": sent,
                            "source": data.get("source", source_dir.name),
                            "domain": "conversation",
                        })
                        count += 1
            print(f"  {source_dir.name}: {count} raw sentences", flush=True)

    print(f"\nTotal raw: {len(all_sentences)}", flush=True)

    final, seen = [], set()
    for sent in all_sentences:
        text = sent["text"].strip()
        if not text or not is_valid(text, min_words, max_words, min_chars, skip_patterns):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        final.append(sent)

    output_file = OUTPUT_DIR / "sentences.jsonl"
    with open(output_file, "w") as f:
        for sent in final:
            f.write(json.dumps(sent) + "\n")

    print(f"\nPreprocessed {len(final)} sentences → {output_file}", flush=True)


if __name__ == "__main__":
    main()

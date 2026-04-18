"""Preprocess DailyDialog into individual sentences.

Reads raw_dialogues.jsonl (from collect.py), splits utterances into
sentences, filters, deduplicates, and outputs sentences.jsonl.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "raw" / "conversation"

MIN_WORDS = 5
MAX_WORDS = 100
MIN_CHARS = 15


def clean_text(text: str) -> str:
    """Clean a single utterance."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    if text.count('"') == 1:
        text = text.replace('"', "")
    return text


def is_valid_sentence(text: str) -> bool:
    """Check if a sentence meets quality thresholds."""
    words = text.split()
    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return False
    if len(text) < MIN_CHARS:
        return False
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.7:
        return False
    return True


def main():
    raw_file = OUTPUT_DIR / "raw_dialogues.jsonl"
    if not raw_file.exists():
        raise FileNotFoundError(f"No raw data at {raw_file}. Run collect.py first.")

    sentences = []
    seen = set()

    with open(raw_file) as f:
        for line in f:
            data = json.loads(line.strip())
            for utterance in data["utterances"]:
                cleaned = clean_text(utterance)
                if not cleaned:
                    continue

                # Split utterance into sentences
                parts = re.split(r"(?<=[.!?])\s+", cleaned)
                for part in parts:
                    part = part.strip()
                    if not part or not is_valid_sentence(part):
                        continue

                    key = part.lower()
                    if key in seen:
                        continue
                    seen.add(key)

                    sentences.append({
                        "text": part,
                        "source": "dailydialog",
                        "domain": "conversation",
                    })

    output_file = OUTPUT_DIR / "sentences.jsonl"
    with open(output_file, "w") as f:
        for sent in sentences:
            f.write(json.dumps(sent) + "\n")

    print(f"Preprocessed {len(sentences)} sentences from DailyDialog → {output_file}")


if __name__ == "__main__":
    main()

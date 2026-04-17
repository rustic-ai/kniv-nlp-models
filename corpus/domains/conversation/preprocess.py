"""Preprocess DailyDialog into individual sentences.

Each dialogue is a sequence of utterances separated by __eou__.
We split each utterance into sentences and output as JSONL.
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
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing quotes if unbalanced
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
    # Skip if too many special characters
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.7:
        return False
    return True


def process_dailydialog():
    """Process DailyDialog dialogues_text.txt into sentences."""
    text_file = OUTPUT_DIR / "dialogues_text.txt"
    if not text_file.exists():
        # Try nested directory
        for p in OUTPUT_DIR.rglob("dialogues_text.txt"):
            text_file = p
            break

    if not text_file.exists():
        raise FileNotFoundError(f"dialogues_text.txt not found in {OUTPUT_DIR}")

    sentences = []
    seen = set()  # deduplication

    with open(text_file, encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # DailyDialog format: utterances separated by __eou__
            utterances = line.split("__eou__")

            for utterance in utterances:
                cleaned = clean_text(utterance)
                if not cleaned:
                    continue

                # Simple sentence splitting at . ? !
                # (each utterance may contain multiple sentences)
                parts = re.split(r"(?<=[.!?])\s+", cleaned)

                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    if not is_valid_sentence(part):
                        continue

                    # Deduplicate
                    key = part.lower()
                    if key in seen:
                        continue
                    seen.add(key)

                    sentences.append({
                        "text": part,
                        "source": "dailydialog",
                        "domain": "conversation",
                    })

    # Write output
    output_file = OUTPUT_DIR / "sentences.jsonl"
    with open(output_file, "w") as f:
        for sent in sentences:
            f.write(json.dumps(sent) + "\n")

    print(f"Preprocessed {len(sentences)} sentences from DailyDialog → {output_file}")


def main():
    process_dailydialog()


if __name__ == "__main__":
    main()

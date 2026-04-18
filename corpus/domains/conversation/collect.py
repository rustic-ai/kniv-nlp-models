"""Collect conversational dialogue text from DailyDialog dataset.

DailyDialog: 13,118 daily conversations covering 10 topics.
License: CC BY-NC-SA 4.0
Loaded via HuggingFace Datasets (no manual download needed).
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "raw" / "conversation"


def collect_dailydialog():
    """Load DailyDialog via HuggingFace and save raw dialogues."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "raw_dialogues.jsonl"
    if output_file.exists():
        print("DailyDialog already collected.")
        return

    print("Loading DailyDialog from HuggingFace...")
    dataset = load_dataset("daily_dialog", revision="refs/convert/parquet")

    count = 0
    with open(output_file, "w") as f:
        for split in ["train", "validation", "test"]:
            for item in dataset[split]:
                dialogue = item["dialog"]
                f.write(json.dumps({
                    "utterances": dialogue,
                    "split": split,
                }) + "\n")
                count += 1

    print(f"Collected {count} dialogues → {output_file}")


def main():
    collect_dailydialog()


if __name__ == "__main__":
    main()

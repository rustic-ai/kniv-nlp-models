"""Collect conversational dialogue text from DailyDialog dataset.

DailyDialog: 13,118 daily conversations covering 10 topics.
License: CC BY-NC-SA 4.0
Source: http://yanran.li/dailydialog.html
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import requests

DOMAIN_DIR = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "raw" / "conversation"
DAILYDIALOG_URL = "http://yanran.li/files/ijcnlp_dailydialog.zip"


def download_dailydialog():
    """Download and extract DailyDialog dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUTPUT_DIR / "dailydialog.zip"

    if (OUTPUT_DIR / "dialogues_text.txt").exists():
        print("DailyDialog already downloaded.")
        return

    print(f"Downloading DailyDialog from {DAILYDIALOG_URL}...")
    response = requests.get(DAILYDIALOG_URL, stream=True)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(OUTPUT_DIR)

    # Find the text file (may be nested)
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for fname in files:
            if fname == "dialogues_text.txt":
                src = Path(root) / fname
                if src.parent != OUTPUT_DIR:
                    src.rename(OUTPUT_DIR / fname)
                    print(f"Moved {fname} to {OUTPUT_DIR}")

    zip_path.unlink()
    print("DailyDialog downloaded and extracted.")


def main():
    download_dailydialog()


if __name__ == "__main__":
    main()

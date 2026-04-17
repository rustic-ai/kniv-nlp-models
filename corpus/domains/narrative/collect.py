"""Collect narrative text from Project Gutenberg.

Downloads selected public domain novels and extracts text.
"""

from __future__ import annotations

from pathlib import Path

import requests
import yaml

DOMAIN_DIR = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "raw" / "narrative"
GUTENBERG_MIRROR = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"


def download_book(book_id: int) -> str | None:
    """Download a single book from Project Gutenberg."""
    cache_file = OUTPUT_DIR / f"pg{book_id}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8", errors="replace")

    url = GUTENBERG_MIRROR.format(book_id=book_id)
    print(f"  Downloading book {book_id} from {url}...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text = response.text
        cache_file.write_text(text, encoding="utf-8")
        return text
    except Exception as e:
        print(f"  ⚠ Failed to download book {book_id}: {e}")
        return None


def strip_gutenberg_header_footer(text: str) -> str:
    """Remove Project Gutenberg header and footer boilerplate."""
    lines = text.split("\n")

    # Find start of actual text (after "*** START OF")
    start = 0
    for i, line in enumerate(lines):
        if "*** START OF" in line or "***START OF" in line:
            start = i + 1
            break

    # Find end of actual text (before "*** END OF")
    end = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if "*** END OF" in lines[i] or "***END OF" in lines[i]:
            end = i
            break

    return "\n".join(lines[start:end])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DOMAIN_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)

    book_ids = config["sources"]["gutenberg"]["book_ids"]
    print(f"Collecting {len(book_ids)} books from Project Gutenberg...")

    all_text_file = OUTPUT_DIR / "all_books.txt"
    with open(all_text_file, "w", encoding="utf-8") as out:
        for book_id in book_ids:
            text = download_book(book_id)
            if text:
                cleaned = strip_gutenberg_header_footer(text)
                out.write(cleaned + "\n\n")
                print(f"  ✓ Book {book_id}: {len(cleaned)} chars")

    print(f"All books saved to {all_text_file}")


if __name__ == "__main__":
    main()

"""Preprocess collected business text into individual sentences.

Reads raw data from all sources under output/raw/business/,
extracts sentences, filters by quality, deduplicates, and outputs
a single sentences.jsonl file.

Usage:
    python -m corpus.domains.business.preprocess
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

DOMAIN_DIR = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "raw" / "business"


def load_config() -> dict:
    with open(DOMAIN_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


# ── Sentence splitting ────────────────────────────────────────────

# Abbreviations that shouldn't trigger sentence splits
ABBREVS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs",
    "etc", "inc", "ltd", "vol", "no", "fig", "approx", "dept",
    "corp", "assn", "bros", "co", "est", "govt", "natl",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "u.s", "e.g", "i.e", "cf",
}


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with abbreviation awareness."""
    sentences = []
    current: list[str] = []
    words = text.split()

    for word in words:
        current.append(word)

        if not word or word[-1] not in ".!?":
            continue

        base = word.rstrip(".!?").lower()

        # Don't split on abbreviations
        if base in ABBREVS:
            continue
        # Don't split on ellipsis
        if word.endswith("..."):
            continue
        # Don't split on single-letter initials (A., B., etc.)
        if len(base) == 1 and word.endswith("."):
            continue
        # Don't split on decimal numbers (3.5, $12.50)
        if base.replace(",", "").replace(".", "").replace("$", "").replace("%", "").isdigit():
            continue

        sentence = " ".join(current).strip()
        if sentence:
            sentences.append(sentence)
        current = []

    # Trailing text
    if current:
        sentence = " ".join(current).strip()
        if sentence:
            sentences.append(sentence)

    return sentences


# ── Text cleaning ────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """General text cleaning."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove unmatched quotes
    if text.count('"') == 1:
        text = text.replace('"', "")
    return text.strip()


def clean_email_body(body: str) -> str:
    """Clean email body: strip forwarded content, signatures, etc."""
    lines = body.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Stop at forwarded messages or reply chains
        if stripped.startswith(("----- Original Message", "-----Forwarded",
                               "From:", "> ", ">>>", "Forwarded by")):
            break
        # Skip empty lines and separator lines
        if not stripped or re.match(r"^[-=_*]{3,}$", stripped):
            continue
        cleaned.append(stripped)
    return " ".join(cleaned)


# ── Quality filtering ────────────────────────────────────────────

def is_valid_sentence(text: str, min_words: int, max_words: int,
                      min_chars: int, skip_patterns: list[re.Pattern]) -> bool:
    """Check if a sentence meets quality thresholds."""
    words = text.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    if len(text) < min_chars:
        return False

    # Skip matching patterns
    for pattern in skip_patterns:
        if pattern.search(text):
            return False

    # Skip all-caps lines
    if text == text.upper() and len(text) > 30:
        return False

    # Must have reasonable alpha ratio (skip tables, numbers-only)
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.6:  # slightly lower than narrative — business text has more numbers
        return False

    # Skip lines that are mostly punctuation or special chars
    if text.count("|") > 3 or text.count("\t") > 2:
        return False

    return True


# ── Source processors ────────────────────────────────────────────

def process_sec_edgar(raw_dir: Path) -> list[dict]:
    """Extract sentences from SEC filings."""
    filings_file = raw_dir / "sec_edgar" / "filings.jsonl"
    if not filings_file.exists():
        return []

    sentences = []
    with open(filings_file) as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]
            company = data.get("company", "unknown")

            # Split into paragraphs, then sentences
            for para in text.split("\n"):
                para = clean_text(para)
                if not para or len(para) < 20:
                    continue
                for sent in split_sentences(para):
                    sentences.append({
                        "text": sent,
                        "source": f"sec_edgar/{company}",
                        "domain": "business",
                    })

    return sentences


def process_enron(raw_dir: Path) -> list[dict]:
    """Extract sentences from Enron emails."""
    emails_file = raw_dir / "enron" / "emails.jsonl"
    if not emails_file.exists():
        return []

    sentences = []
    with open(emails_file) as f:
        for line in f:
            data = json.loads(line)
            body = clean_email_body(data["body"])
            if not body:
                continue

            for sent in split_sentences(body):
                sentences.append({
                    "text": sent,
                    "source": "enron",
                    "domain": "business",
                })

    return sentences


def process_openstax(raw_dir: Path) -> list[dict]:
    """Extract sentences from OpenStax textbooks."""
    books_file = raw_dir / "openstax" / "books.jsonl"
    if not books_file.exists():
        return []

    sentences = []
    with open(books_file) as f:
        for line in f:
            data = json.loads(line)
            title = data["title"]
            text = data["text"]

            for para in text.split("\n"):
                para = clean_text(para)
                if not para or len(para) < 20:
                    continue
                for sent in split_sentences(para):
                    sentences.append({
                        "text": sent,
                        "source": f"openstax/{title}",
                        "domain": "business",
                    })

    return sentences


def process_odoo(raw_dir: Path) -> list[dict]:
    """Extract sentences from Odoo documentation."""
    docs_file = raw_dir / "odoo" / "docs.jsonl"
    if not docs_file.exists():
        return []

    sentences = []
    with open(docs_file) as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]

            for para in text.split("\n"):
                para = clean_text(para)
                if not para or len(para) < 20:
                    continue
                for sent in split_sentences(para):
                    sentences.append({
                        "text": sent,
                        "source": "odoo",
                        "domain": "business",
                    })

    return sentences


def process_wikipedia(raw_dir: Path) -> list[dict]:
    """Extract sentences from Wikipedia articles."""
    articles_file = raw_dir / "wikipedia" / "articles.jsonl"
    if not articles_file.exists():
        return []

    sentences = []
    with open(articles_file) as f:
        for line in f:
            data = json.loads(line)
            title = data.get("title", "")
            text = data["text"]

            for para in text.split("\n"):
                para = clean_text(para)
                if not para or len(para) < 20:
                    continue
                for sent in split_sentences(para):
                    sentences.append({
                        "text": sent,
                        "source": f"wikipedia/{title}",
                        "domain": "business",
                    })

    return sentences


def process_cuad(raw_dir: Path) -> list[dict]:
    """Extract sentences from CUAD contracts."""
    contracts_file = raw_dir / "cuad" / "contracts.jsonl"
    if not contracts_file.exists():
        return []

    sentences = []
    with open(contracts_file) as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]

            for para in text.split("\n"):
                para = clean_text(para)
                if not para or len(para) < 20:
                    continue
                for sent in split_sentences(para):
                    sentences.append({
                        "text": sent,
                        "source": "cuad",
                        "domain": "business",
                    })

    return sentences


def process_s2orc(raw_dir: Path) -> list[dict]:
    """Extract sentences from Semantic Scholar abstracts."""
    abstracts_file = raw_dir / "s2orc" / "abstracts.jsonl"
    if not abstracts_file.exists():
        return []

    sentences = []
    with open(abstracts_file) as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]

            for sent in split_sentences(clean_text(text)):
                sentences.append({
                    "text": sent,
                    "source": "s2orc",
                    "domain": "business",
                })

    return sentences


# ── Main ─────────────────────────────────────────────────────────

PROCESSORS = [
    ("SEC EDGAR", process_sec_edgar),
    ("Enron", process_enron),
    ("OpenStax", process_openstax),
    ("Odoo", process_odoo),
    ("Wikipedia", process_wikipedia),
    ("CUAD", process_cuad),
    ("S2ORC", process_s2orc),
]


def main():
    config = load_config()
    filtering = config.get("filtering", {})
    min_words = filtering.get("min_words", 5)
    max_words = filtering.get("max_words", 120)
    min_chars = filtering.get("min_chars", 15)

    skip_patterns = [re.compile(p) for p in filtering.get("skip_patterns", [])]

    # Collect sentences from all sources
    all_sentences = []
    for name, processor in PROCESSORS:
        raw = processor(OUTPUT_DIR)
        print(f"  {name}: {len(raw)} raw sentences")
        all_sentences.extend(raw)

    print(f"\nTotal raw sentences: {len(all_sentences)}")

    # Filter and deduplicate
    final = []
    seen = set()

    for sent in all_sentences:
        text = sent["text"].strip()
        if not text:
            continue

        if not is_valid_sentence(text, min_words, max_words, min_chars, skip_patterns):
            continue

        key = text.lower()
        if key in seen:
            continue
        seen.add(key)

        final.append(sent)

    # Write output
    output_file = OUTPUT_DIR / "sentences.jsonl"
    with open(output_file, "w") as f:
        for sent in final:
            f.write(json.dumps(sent) + "\n")

    # Source distribution
    source_counts: dict[str, int] = {}
    for sent in final:
        base_source = sent["source"].split("/")[0]
        source_counts[base_source] = source_counts.get(base_source, 0) + 1

    print(f"\nPreprocessed {len(final)} sentences → {output_file}")
    print(f"  (filtered {len(all_sentences) - len(final)} sentences)")
    print(f"\n  Source distribution:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {source:20s}: {count:7d}")


if __name__ == "__main__":
    main()

"""Prepare business corpus for LLM continued pretraining.

Reads raw collected documents (full text, not sentence-split),
tokenizes into fixed-length chunks, and creates train/dev splits.

The key difference from the NLP pipeline: we preserve document structure
and context flow. No sentence splitting — the LLM needs to learn
discourse, coreference, and reasoning patterns across paragraphs.

Usage:
    python models/phi4-mini-llm-en/prepare_data.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import yaml
from transformers import AutoTokenizer


CONFIG_PATH = Path(__file__).parent / "config.yaml"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "prepared" / "phi4-mini-llm-en"
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Source-specific text extractors ──────────────────────────────

def extract_sec_edgar(path: Path) -> list[str]:
    """Extract full filing text. Each filing is one document."""
    docs = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            text = data.get("text", "").strip()
            if len(text) > 500:
                # Add a header for context
                company = data.get("company", "Unknown")
                date = data.get("date", "")
                header = f"SEC Filing — {company} — 10-K — {date}\n\n"
                docs.append(header + text)
    return docs


def extract_enron(path: Path) -> list[str]:
    """Extract email bodies. Group into threads where possible."""
    docs = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            body = data.get("body", "").strip()
            if len(body) < 50:
                continue
            subject = data.get("subject", "")
            header = f"Subject: {subject}\n\n" if subject else ""
            docs.append(header + body)
    return docs


def extract_books(path: Path) -> list[str]:
    """Extract full book text. Each book is one long document."""
    docs = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            text = data.get("text", "").strip()
            if len(text) > 500:
                title = data.get("title", "")
                header = f"{title}\n\n" if title else ""
                docs.append(header + text)
    return docs


def extract_generic(path: Path) -> list[str]:
    """Generic extractor — works for articles, abstracts, contracts, docs."""
    docs = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            text = data.get("text", "").strip()
            if len(text) > 100:
                title = data.get("title", "")
                header = f"{title}\n\n" if title else ""
                docs.append(header + text)
    return docs


# Map source filenames to extractors
EXTRACTORS = {
    "filings.jsonl": extract_sec_edgar,
    "emails.jsonl": extract_enron,
    "books.jsonl": extract_books,
}


def load_all_documents(config: dict) -> list[str]:
    """Load all source documents as full text strings."""
    all_docs = []

    for source_path_str in config["data"]["sources"]:
        source_path = PROJECT_ROOT / source_path_str
        if not source_path.exists():
            print(f"  ⚠ Missing: {source_path}")
            continue

        filename = source_path.name
        extractor = EXTRACTORS.get(filename, extract_generic)
        docs = extractor(source_path)
        source_name = source_path.parent.name
        print(f"  {source_name}: {len(docs)} documents, {sum(len(d) for d in docs) / 1e6:.1f}M chars")
        all_docs.extend(docs)

    return all_docs


# ── Chunking ─────────────────────────────────────────────────────

def chunk_documents(
    documents: list[str],
    tokenizer,
    max_length: int,
    overlap: int = 128,
) -> list[dict]:
    """Tokenize documents and chunk into fixed-length training examples.

    Uses a sliding window with overlap to preserve context across chunks.
    Each chunk is a contiguous sequence of tokens from a single document.
    """
    examples = []

    for doc in documents:
        token_ids = tokenizer.encode(doc, add_special_tokens=False)

        if len(token_ids) <= max_length:
            examples.append({"input_ids": token_ids})
            continue

        # Sliding window
        start = 0
        while start < len(token_ids):
            end = start + max_length
            chunk = token_ids[start:end]
            if len(chunk) >= max_length // 4:  # skip tiny trailing chunks
                examples.append({"input_ids": chunk})
            start += max_length - overlap

    return examples


# ── Main ─────────────────────────────────────────────────────────

def main():
    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_model = config["model"]["base"]
    max_length = config["model"]["max_length"]
    dev_ratio = config["data"]["dev_ratio"]
    seed = config["data"]["seed"]

    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print("Loading documents from raw corpus...")
    documents = load_all_documents(config)
    print(f"Total: {len(documents)} documents")

    # Shuffle documents
    random.seed(seed)
    random.shuffle(documents)

    # Split documents into train/dev BEFORE chunking
    # (so dev evaluates on unseen documents, not unseen chunks of seen docs)
    dev_count = max(1, int(len(documents) * dev_ratio))
    dev_docs = documents[:dev_count]
    train_docs = documents[dev_count:]

    print(f"\nChunking into {max_length}-token sequences...")
    train_examples = chunk_documents(train_docs, tokenizer, max_length)
    dev_examples = chunk_documents(dev_docs, tokenizer, max_length)

    print(f"  Train: {len(train_examples)} chunks from {len(train_docs)} docs")
    print(f"  Dev:   {len(dev_examples)} chunks from {len(dev_docs)} docs")

    # Save as JSONL (token IDs)
    for name, examples in [("train", train_examples), ("dev", dev_examples)]:
        output_path = OUTPUT_DIR / f"{name}.jsonl"
        with open(output_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"  Saved {output_path}")

    # Save tokenizer info
    meta = {
        "base_model": base_model,
        "max_length": max_length,
        "vocab_size": tokenizer.vocab_size,
        "train_chunks": len(train_examples),
        "dev_chunks": len(dev_examples),
        "train_docs": len(train_docs),
        "dev_docs": len(dev_docs),
        "total_train_tokens": sum(len(ex["input_ids"]) for ex in train_examples),
        "total_dev_tokens": sum(len(ex["input_ids"]) for ex in dev_examples),
    }
    with open(OUTPUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    total_tokens = meta["total_train_tokens"] + meta["total_dev_tokens"]
    print(f"\nData prepared in {OUTPUT_DIR}")
    print(f"  Total tokens: {total_tokens:,} ({total_tokens / 1e6:.1f}M)")
    print(f"  Train tokens: {meta['total_train_tokens']:,}")
    print(f"  Dev tokens:   {meta['total_dev_tokens']:,}")


if __name__ == "__main__":
    main()

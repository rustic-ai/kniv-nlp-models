"""Export annotated + validated corpus to final format.

Merges all domain outputs, applies corrections from validation,
splits into train/dev/test, and exports as CoNLL-U + HF Datasets.

Usage:
    python -m corpus.pipeline.export
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .config import ANNOTATED_DIR, VALIDATED_DIR, FINAL_DIR


def load_domain_annotations(domain: str) -> list[str]:
    """Load CoNLL-U annotations for a domain."""
    conllu_file = ANNOTATED_DIR / domain / "annotated.conllu"
    if not conllu_file.exists():
        print(f"  ⚠ No annotations for domain '{domain}', skipping")
        return []

    with open(conllu_file) as f:
        content = f.read()

    # Split into sentences (separated by double newlines)
    sentences = []
    current = []
    for line in content.split("\n"):
        if line.strip() == "" and current:
            sentences.append("\n".join(current))
            current = []
        else:
            current.append(line)
    if current:
        sentences.append("\n".join(current))

    return [s for s in sentences if s.strip()]


def merge_and_split(
    domains: list[str],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    seed: int = 42,
):
    """Merge all domains and split into train/dev/test."""
    all_sentences = []

    for domain in domains:
        sentences = load_domain_annotations(domain)
        print(f"  {domain}: {len(sentences)} sentences")
        all_sentences.extend(sentences)

    print(f"\nTotal: {len(all_sentences)} sentences")

    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(all_sentences)

    # Split
    n = len(all_sentences)
    train_end = int(n * train_ratio)
    dev_end = int(n * (train_ratio + dev_ratio))

    splits = {
        "train": all_sentences[:train_end],
        "dev": all_sentences[train_end:dev_end],
        "test": all_sentences[dev_end:],
    }

    for name, sents in splits.items():
        print(f"  {name}: {len(sents)} sentences")

    return splits


def write_conllu(sentences: list[str], output_path: Path):
    """Write sentences to a CoNLL-U file."""
    with open(output_path, "w") as f:
        for sent in sentences:
            f.write(sent + "\n\n")
    print(f"  Written: {output_path} ({len(sentences)} sentences)")


def export_corpus(domains: list[str]):
    """Export the full corpus."""
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Merging domains and splitting...")
    splits = merge_and_split(domains)

    for name, sentences in splits.items():
        write_conllu(sentences, FINAL_DIR / f"{name}.conllu")

    # Write metadata
    metadata = {
        "domains": domains,
        "splits": {name: len(sents) for name, sents in splits.items()},
        "total": sum(len(s) for s in splits.values()),
    }
    with open(FINAL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nCorpus exported to {FINAL_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Export corpus to final format")
    parser.add_argument("--domains", nargs="+", required=True,
                        help="List of domain names to include")
    args = parser.parse_args()

    export_corpus(args.domains)


if __name__ == "__main__":
    main()

"""Corpus statistics and quality report.

Usage:
    python -m corpus.pipeline.stats --domain conversation
    python -m corpus.pipeline.stats --final
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .config import ANNOTATED_DIR, FINAL_DIR


def stats_from_conllu(conllu_path: Path) -> dict:
    """Compute statistics from a CoNLL-U file."""
    sentence_count = 0
    token_count = 0
    pos_counts: Counter = Counter()
    dep_counts: Counter = Counter()
    ner_counts: Counter = Counter()
    sentence_lengths: list[int] = []

    current_tokens = 0
    with open(conllu_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            if line == "":
                if current_tokens > 0:
                    sentence_count += 1
                    sentence_lengths.append(current_tokens)
                    current_tokens = 0
                continue

            parts = line.split("\t")
            if len(parts) >= 10:
                token_count += 1
                current_tokens += 1
                pos_counts[parts[3]] += 1  # UPOS
                dep_counts[parts[7]] += 1  # DEPREL
                misc = parts[9]
                if misc.startswith("NER=") and misc != "NER=O":
                    ner_tag = misc.split("=")[1]
                    ner_counts[ner_tag] += 1

    if current_tokens > 0:
        sentence_count += 1
        sentence_lengths.append(current_tokens)

    avg_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

    return {
        "sentences": sentence_count,
        "tokens": token_count,
        "avg_sentence_length": round(avg_length, 1),
        "pos_distribution": dict(pos_counts.most_common(20)),
        "dep_distribution": dict(dep_counts.most_common(20)),
        "ner_distribution": dict(ner_counts.most_common(20)),
        "ner_total_spans": sum(1 for t in ner_counts.values()),
    }


def print_stats(stats: dict, name: str):
    """Print a human-readable stats report."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Sentences:    {stats['sentences']:,}")
    print(f"  Tokens:       {stats['tokens']:,}")
    print(f"  Avg length:   {stats['avg_sentence_length']} tokens/sentence")

    print(f"\n  POS distribution (top 10):")
    for pos, count in list(stats["pos_distribution"].items())[:10]:
        pct = 100 * count / stats["tokens"]
        print(f"    {pos:8s}: {count:8,} ({pct:.1f}%)")

    print(f"\n  Dep relations (top 10):")
    for dep, count in list(stats["dep_distribution"].items())[:10]:
        pct = 100 * count / stats["tokens"]
        print(f"    {dep:12s}: {count:8,} ({pct:.1f}%)")

    if stats["ner_distribution"]:
        print(f"\n  NER spans (top 10):")
        for ner, count in list(stats["ner_distribution"].items())[:10]:
            print(f"    {ner:12s}: {count:8,}")


def main():
    parser = argparse.ArgumentParser(description="Corpus statistics")
    parser.add_argument("--domain", help="Domain name")
    parser.add_argument("--final", action="store_true", help="Stats for final exported corpus")
    args = parser.parse_args()

    if args.final:
        for split in ["train", "dev", "test"]:
            path = FINAL_DIR / f"{split}.conllu"
            if path.exists():
                stats = stats_from_conllu(path)
                print_stats(stats, f"Final corpus — {split}")
    elif args.domain:
        path = ANNOTATED_DIR / args.domain / "annotated.conllu"
        if path.exists():
            stats = stats_from_conllu(path)
            print_stats(stats, f"Domain: {args.domain}")
        else:
            print(f"No annotations found for domain '{args.domain}'")
    else:
        print("Specify --domain or --final")


if __name__ == "__main__":
    main()

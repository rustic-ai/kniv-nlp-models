"""Convert CoNLL-2012 OntoNotes SRL data to our evaluation format.

Downloads from HuggingFace (ontonotes/conll2012_ontonotesv5), converts
multi-predicate sentences to one-example-per-predicate with BIO tags,
and maps labels to our 42-tag set.

Usage:
    python prepare_conll2012_eval.py
    python prepare_conll2012_eval.py --split test --output conll2012_test.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

SRL_TAGS = [
    "O", "V",
    "B-ARG0", "I-ARG0", "B-ARG1", "I-ARG1", "B-ARG2", "I-ARG2",
    "B-ARG3", "I-ARG3", "B-ARG4", "I-ARG4",
    "B-ARGM-TMP", "I-ARGM-TMP", "B-ARGM-LOC", "I-ARGM-LOC",
    "B-ARGM-MNR", "I-ARGM-MNR", "B-ARGM-CAU", "I-ARGM-CAU",
    "B-ARGM-PRP", "I-ARGM-PRP", "B-ARGM-NEG", "I-ARGM-NEG",
    "B-ARGM-ADV", "I-ARGM-ADV", "B-ARGM-DIR", "I-ARGM-DIR",
    "B-ARGM-DIS", "I-ARGM-DIS", "B-ARGM-EXT", "I-ARGM-EXT",
    "B-ARGM-MOD", "I-ARGM-MOD", "B-ARGM-PRD", "I-ARGM-PRD",
    "B-ARGM-GOL", "I-ARGM-GOL", "B-ARGM-COM", "I-ARGM-COM",
    "B-ARGM-REC", "I-ARGM-REC",
]
SRL_TAG_SET = set(SRL_TAGS)


def map_label(label: str) -> str:
    """Map OntoNotes BIO label to our 42-tag set."""
    if label == "O":
        return "O"
    if label in ("B-V", "I-V"):
        return "V"
    if label in SRL_TAG_SET:
        return label
    # Strip continuation/reference: B-C-ARG0 → B-ARG0, I-R-ARGM-LOC → I-ARGM-LOC
    for prefix in ("B-C-", "I-C-", "B-R-", "I-R-"):
        if label.startswith(prefix):
            rest = label[len(prefix):]
            mapped = f"{label[0]}-{rest}"
            if mapped in SRL_TAG_SET:
                return mapped
    # ARG5 → ARG4, ARG1-DSP → ARG1
    normalized = label
    for src, dst in [("ARG1-DSP", "ARG1"), ("ARG5", "ARG4")]:
        normalized = normalized.replace(src, dst)
    if normalized in SRL_TAG_SET:
        return normalized
    # Unknown → O
    return "O"


def convert_dataset(split: str = "test", config: str = "english_v12"):
    """Load CoNLL-2012 from HuggingFace and convert to our format."""
    from datasets import load_dataset

    print(f"Loading CoNLL-2012 OntoNotes ({config}, {split})...", flush=True)
    ds = load_dataset("ontonotes/conll2012_ontonotesv5", config,
                       split=split, trust_remote_code=True)
    print(f"Loaded {len(ds)} documents", flush=True)

    examples = []
    skipped_no_args = 0
    skipped_short = 0
    label_counts = Counter()

    for doc in ds:
        for sentence in doc["sentences"]:
            words = sentence["words"]
            if len(words) < 3:
                skipped_short += 1
                continue

            srl_frames = sentence.get("srl_frames", [])
            if not srl_frames:
                continue

            for frame in srl_frames:
                verb = frame["verb"]
                tags = frame["frames"]

                if len(tags) != len(words):
                    continue

                # Map to our label set
                mapped = [map_label(t) for t in tags]

                # Find predicate index (V tag position)
                pred_idx = -1
                for i, t in enumerate(mapped):
                    if t == "V":
                        pred_idx = i
                        break

                if pred_idx == -1:
                    continue

                # Must have at least one argument
                has_args = any(t not in ("O", "V") for t in mapped)
                if not has_args:
                    skipped_no_args += 1
                    continue

                for t in mapped:
                    label_counts[t] += 1

                examples.append({
                    "words": words,
                    "text": " ".join(words),
                    "srl_tags": mapped,
                    "predicate_idx": pred_idx,
                })

    print(f"\nConverted: {len(examples):,} predicate instances", flush=True)
    print(f"Skipped (no args): {skipped_no_args:,}", flush=True)
    print(f"Skipped (short): {skipped_short:,}", flush=True)

    print(f"\nLabel distribution (top 15):", flush=True)
    total_tokens = sum(label_counts.values())
    for tag, count in label_counts.most_common(15):
        print(f"  {tag}: {count:,} ({100*count/total_tokens:.1f}%)", flush=True)

    # Span length stats
    span_lens = []
    for ex in examples:
        cur_len = 0
        for t in ex["srl_tags"]:
            if t.startswith("B-"):
                if cur_len > 0:
                    span_lens.append(cur_len)
                cur_len = 1
            elif t.startswith("I-"):
                cur_len += 1
            else:
                if cur_len > 0:
                    span_lens.append(cur_len)
                cur_len = 0
        if cur_len > 0:
            span_lens.append(cur_len)

    if span_lens:
        multi = sum(1 for l in span_lens if l > 1)
        print(f"\nSpan stats: {len(span_lens):,} total spans", flush=True)
        print(f"  Mean length: {sum(span_lens)/len(span_lens):.2f}", flush=True)
        print(f"  Multi-word: {multi:,} ({100*multi/len(span_lens):.1f}%)", flush=True)

    return examples


def main():
    parser = argparse.ArgumentParser(description="Convert CoNLL-2012 for SRL evaluation")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--config", default="english_v12")
    parser.add_argument("--output", default=None,
                        help="Output path (default: data/prepared/kniv-deberta-cascade/conll2012_{split}.json)")
    args = parser.parse_args()

    output = args.output or str(
        Path(__file__).parent.parent.parent
        / "data" / "prepared" / "kniv-deberta-cascade"
        / f"conll2012_{args.split}.json"
    )

    examples = convert_dataset(args.split, args.config)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(examples, f)
    print(f"\nSaved {len(examples):,} examples to {output}", flush=True)


if __name__ == "__main__":
    main()

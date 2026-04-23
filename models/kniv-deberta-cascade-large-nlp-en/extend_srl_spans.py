"""Extend single-token silver SRL spans to full NP constituents using dep trees.

The dannashao silver labeler produces head-word-only spans (100% single-token).
This script extends each B-ROLE token to cover its full dependency subtree,
producing proper multi-word argument spans.

Example:
  Before: ["O", "O", "B-ARG1", "O", "O"]  words: ["the", "big", "window", "was", "broken"]
  After:  ["B-ARG1", "I-ARG1", "I-ARG1", "O", "O"]  (extended "window" → "the big window")

Requires the annotated corpus (with dep trees) to be joined with the silver SRL data.

Usage:
    python extend_srl_spans.py
    python extend_srl_spans.py --silver srl_silver_dannashao.json --confidence 0.85
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "prepared" / "kniv-deberta-cascade"
CORPUS_DIR = Path(__file__).parent.parent.parent / "corpus" / "output" / "annotated"

# Deprels that belong inside an NP/argument span
NP_DEPRELS = {
    "det", "amod", "compound", "nummod", "nmod", "poss",
    "quantmod", "nn", "advmod", "neg",
}

# Deprels to EXCLUDE from span extension (clausal, PP attachment, etc.)
EXCLUDE_DEPRELS = {
    "relcl", "acl", "advcl", "prep", "punct", "cc", "conj",
    "appos", "parataxis", "mark", "case",
}


def build_children_map(tokens: list[dict]) -> dict[int, list[int]]:
    """Build parent→children mapping. Token IDs are 1-based."""
    children = defaultdict(list)
    for tok in tokens:
        head = tok["head"]
        if head > 0:  # skip root
            children[head].append(tok["id"])
    return children


def get_np_subtree(token_id: int, tokens: list[dict], children: dict[int, list[int]]) -> set[int]:
    """Get contiguous NP tokens in the subtree rooted at token_id.

    Only follows NP-internal deprels (det, amod, compound, etc.).
    Stops at clausal boundaries, PPs, and punctuation.
    """
    tok_by_id = {t["id"]: t for t in tokens}
    result = {token_id}
    stack = [token_id]

    while stack:
        parent = stack.pop()
        for child_id in children.get(parent, []):
            child = tok_by_id.get(child_id)
            if child is None:
                continue
            deprel = child["deprel"].lower().split(":")[0]  # strip subtypes
            if deprel in NP_DEPRELS:
                result.add(child_id)
                stack.append(child_id)  # recurse into this child's subtree

    return result


def extend_spans(
    words: list[str],
    srl_tags: list[str],
    tokens: list[dict],
) -> list[str]:
    """Extend single-token B-ROLE tags to cover the full NP subtree."""
    if len(words) != len(tokens):
        return srl_tags  # length mismatch, skip

    children = build_children_map(tokens)
    extended = list(srl_tags)

    for i, tag in enumerate(srl_tags):
        if not tag.startswith("B-"):
            continue
        # Only extend if this is a single-token span (no following I- tag)
        if i + 1 < len(srl_tags) and srl_tags[i + 1].startswith("I-"):
            continue  # already multi-word, don't touch

        role = tag[2:]  # e.g., "ARG0", "ARGM-TMP"
        token_id = i + 1  # 1-based

        # Get NP subtree
        subtree_ids = get_np_subtree(token_id, tokens, children)

        if len(subtree_ids) <= 1:
            continue  # no extension possible

        # Convert to 0-based indices and sort
        span_indices = sorted(tid - 1 for tid in subtree_ids)

        # Make contiguous: take min to max
        span_start = span_indices[0]
        span_end = span_indices[-1]

        # Don't extend if span would overlap with another role
        can_extend = True
        for j in range(span_start, span_end + 1):
            if j == i:
                continue  # this is the head token
            if extended[j] != "O":
                can_extend = False
                break

        if not can_extend:
            continue

        # Apply extension: head gets B-, others get I-
        for j in range(span_start, span_end + 1):
            if j == span_start:
                extended[j] = f"B-{role}"
            else:
                extended[j] = f"I-{role}"

    return extended


def load_corpus_index(corpus_dir: Path) -> dict[str, list[dict]]:
    """Load corpus sentences indexed by text for joining with silver data."""
    print("Loading corpus dep trees...", flush=True)
    index = {}
    total = 0

    for domain_dir in sorted(corpus_dir.iterdir()):
        jsonl = domain_dir / "annotated.jsonl"
        if not jsonl.exists():
            continue
        count = 0
        with open(jsonl) as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text", "")
                if text:
                    index[text] = obj["tokens"]
                    count += 1
        print(f"  {domain_dir.name}: {count:,}", flush=True)
        total += count

    print(f"Total indexed: {total:,} sentences", flush=True)
    return index


def main():
    parser = argparse.ArgumentParser(description="Extend silver SRL spans via dep trees")
    parser.add_argument("--silver", default="srl_silver_dannashao.json",
                        help="Silver SRL data filename in data dir")
    parser.add_argument("--corpus-dir", default=str(CORPUS_DIR),
                        help="Path to annotated corpus")
    parser.add_argument("--output", default=None,
                        help="Output filename (default: srl_silver_extended.json)")
    args = parser.parse_args()

    silver_path = DATA_DIR / args.silver
    output_path = DATA_DIR / (args.output or "srl_silver_extended.json")

    # Load silver data
    print(f"Loading silver data: {silver_path}", flush=True)
    with open(silver_path) as f:
        silver = json.load(f)
    print(f"Silver examples: {len(silver):,}", flush=True)

    # Load corpus dep trees
    corpus_index = load_corpus_index(Path(args.corpus_dir))

    # Extend spans
    extended_data = []
    matched = 0
    extended_count = 0
    total_spans_before = 0
    total_spans_after = 0
    multi_before = 0
    multi_after = 0

    for ex in tqdm(silver, desc="Extending spans"):
        text = ex["text"]
        tokens = corpus_index.get(text)

        if tokens is None or len(tokens) != len(ex["words"]):
            # No dep tree available, keep original
            extended_data.append(ex)
            continue

        matched += 1
        old_tags = ex["srl_tags"]
        new_tags = extend_spans(ex["words"], old_tags, tokens)

        # Count stats
        for t in old_tags:
            if t.startswith("B-"):
                total_spans_before += 1
        for t in new_tags:
            if t.startswith("B-"):
                total_spans_after += 1
        old_multi = sum(1 for t in old_tags if t.startswith("I-"))
        new_multi = sum(1 for t in new_tags if t.startswith("I-"))
        multi_before += old_multi
        multi_after += new_multi
        if new_tags != old_tags:
            extended_count += 1

        extended_data.append({
            "words": ex["words"],
            "text": ex["text"],
            "srl_tags": new_tags,
            "predicate_idx": ex["predicate_idx"],
        })

    print(f"\nResults:", flush=True)
    print(f"  Matched with dep trees: {matched:,} / {len(silver):,}", flush=True)
    print(f"  Extended: {extended_count:,} examples modified", flush=True)
    print(f"  I-tags before: {multi_before:,}", flush=True)
    print(f"  I-tags after:  {multi_after:,}", flush=True)

    # Span length stats
    span_lens = []
    for ex in extended_data:
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
        print(f"\nSpan stats:", flush=True)
        print(f"  Total spans: {len(span_lens):,}", flush=True)
        print(f"  Mean length: {sum(span_lens)/len(span_lens):.2f}", flush=True)
        print(f"  Multi-word: {multi:,} ({100*multi/len(span_lens):.1f}%)", flush=True)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(extended_data, f)
    print(f"\nSaved {len(extended_data):,} examples to {output_path}", flush=True)


if __name__ == "__main__":
    main()

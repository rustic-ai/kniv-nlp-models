"""Preprocess conversation utterances with prev_text linking.

Unlike other domains, conversation utterances are NOT sentence-split.
Each utterance is kept as one unit (the natural dialog act boundary).
prev_text is linked by sorting utterances within each conv_id by turn_idx.

Usage:
    python -m corpus.domains.conversation.preprocess
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

DOMAIN_DIR = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "raw" / "conversation"


def load_config() -> dict:
    with open(DOMAIN_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # Remove stray ChatML artifacts
    text = re.sub(r"<\|im_(?:start|end)\|>", "", text)
    text = re.sub(r"<\|end_of_text\|>", "", text)
    return text.strip()


def is_valid(text: str, min_words: int, max_words: int, min_chars: int,
             skip_patterns: list) -> bool:
    words = text.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    if len(text) < min_chars:
        return False
    for pattern in skip_patterns:
        if pattern.search(text):
            return False
    alpha = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
    if alpha < 0.6:
        return False
    return True


def main():
    config = load_config()
    filt = config.get("filtering", {})
    min_words = filt.get("min_words", 5)
    max_words = filt.get("max_words", 100)
    min_chars = filt.get("min_chars", 15)
    skip_patterns = [re.compile(p) for p in filt.get("skip_patterns", [])]

    # Load all utterances from all sources
    all_utterances = []
    for source_dir in sorted(OUTPUT_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        for jsonl_file in sorted(source_dir.glob("*.jsonl")):
            count = 0
            with open(jsonl_file) as f:
                for line in f:
                    data = json.loads(line)
                    text = clean_text(data.get("text", ""))
                    if not text or not is_valid(text, min_words, max_words, min_chars, skip_patterns):
                        continue
                    data["text"] = text
                    all_utterances.append(data)
                    count += 1
            print(f"  {source_dir.name}: {count} valid utterances", flush=True)

    print(f"\nTotal valid utterances: {len(all_utterances)}", flush=True)

    # Dedup by (conv_id, turn_idx) for structured data, by text for unstructured
    seen_conv_turns = set()
    seen_texts = set()
    deduped = []
    for utt in all_utterances:
        conv_id = utt.get("conv_id")
        if conv_id:
            key = (conv_id, utt.get("turn_idx", 0))
            if key in seen_conv_turns:
                continue
            seen_conv_turns.add(key)
        else:
            key = utt["text"].lower()
            if key in seen_texts:
                continue
            seen_texts.add(key)
        deduped.append(utt)

    print(f"After dedup: {len(deduped)}", flush=True)

    # Group by conv_id and link prev_text
    by_conv = defaultdict(list)
    no_conv = []
    for utt in deduped:
        conv_id = utt.get("conv_id")
        if conv_id:
            by_conv[conv_id].append(utt)
        else:
            no_conv.append(utt)

    # Sort each conversation by turn_idx and link prev_text
    linked = 0
    final = []
    for conv_id, turns in by_conv.items():
        turns.sort(key=lambda x: x.get("turn_idx", 0))
        for i, turn in enumerate(turns):
            if i > 0:
                turn["prev_text"] = turns[i - 1]["text"]
                linked += 1
            final.append(turn)

    # Add non-conversation utterances (no prev_text)
    final.extend(no_conv)

    print(f"Conversations: {len(by_conv)}", flush=True)
    print(f"Utterances with prev_text: {linked}", flush=True)
    print(f"Utterances without prev_text: {len(final) - linked}", flush=True)

    # Write output
    output_file = OUTPUT_DIR / "sentences.jsonl"
    with open(output_file, "w") as f:
        for sent in final:
            f.write(json.dumps(sent) + "\n")

    print(f"\nPreprocessed {len(final)} utterances → {output_file}", flush=True)


if __name__ == "__main__":
    main()

"""Export annotated + validated + classified corpus to final format.

Applies GPT validation corrections to spaCy annotations,
merges CLS labels, splits into train/dev/test, and exports as CoNLL-U.

Pipeline: annotate → validate → classify → export (this step)

Usage:
    python -m corpus.pipeline.export \
        --domains conversation narrative business technical news encyclopedic
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .config import ANNOTATED_DIR, VALIDATED_DIR, FINAL_DIR


def load_corrections(domain: str) -> dict[str, list[dict]]:
    """Load GPT validation corrections for a domain.

    Returns {sent_id: [corrections]} where each correction has
    token_index, field (pos/dep/ner), old_value, new_value.
    """
    # Deduplicate: for each sent_id, keep the last (most recent) result
    results_file = VALIDATED_DIR / domain / "validation_results.jsonl"
    if not results_file.exists():
        return {}

    by_id: dict[str, list[dict]] = {}
    with open(results_file) as f:
        for line in f:
            r = json.loads(line)
            if r["status"] == "ok" and r["result"].get("corrections"):
                corrections = r["result"]["corrections"]
                if corrections:
                    by_id[r["sent_id"]] = corrections

    return by_id


def load_cls_labels(domain: str) -> dict[str, str]:
    """Load GPT CLS labels for a domain."""
    cls_file = VALIDATED_DIR / domain / "cls_labels.jsonl"
    if not cls_file.exists():
        return {}

    labels = {}
    with open(cls_file) as f:
        for line in f:
            r = json.loads(line)
            if r["status"] == "ok":
                labels[r["sent_id"]] = r["cls_label"]
    return labels


def apply_corrections_to_conllu(sentence_block: str, corrections: list[dict]) -> str:
    """Apply GPT corrections to a CoNLL-U sentence block.

    Modifies POS (column 3), deprel (column 7), and NER (MISC column 9).
    """
    lines = sentence_block.split("\n")
    result = []

    for line in lines:
        # Comment or metadata lines — pass through
        if line.startswith("#") or not line.strip():
            result.append(line)
            continue

        cols = line.split("\t")
        if len(cols) < 10:
            result.append(line)
            continue

        token_idx = int(cols[0]) - 1  # CoNLL-U is 1-indexed, corrections are 0-indexed

        for corr in corrections:
            if not isinstance(corr, dict):
                continue
            if corr.get("token_index") != token_idx:
                continue

            field = corr.get("field", "")
            new_value = corr.get("new_value", "")
            if not new_value:
                continue

            # Handle nested dict values (GPT sometimes returns {head: X, dep: Y})
            if isinstance(new_value, dict):
                if field == "dep" and "dep" in new_value:
                    new_value = str(new_value["dep"])
                elif field == "pos" and "pos" in new_value:
                    new_value = str(new_value["pos"])
                else:
                    continue  # skip malformed corrections

            if field == "pos":
                # Only accept valid UPOS tags — reject GPT garbage
                valid_upos = {"ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ",
                              "NOUN","NUM","PART","PRON","PROPN","PUNCT",
                              "SCONJ","SYM","VERB","X"}
                val = str(new_value).strip()
                if val in valid_upos:
                    cols[3] = val
            elif field == "dep":
                cols[7] = str(new_value)  # DEPREL column
            elif field == "ner":
                # NER is in MISC column (index 9) as NER=B-ORG etc.
                if new_value == "O":
                    cols[9] = "_"
                else:
                    cols[9] = f"NER={new_value}"

        result.append("\t".join(cols))

    return "\n".join(result)


def fix_bio_tags(sentence_block: str) -> str:
    """Fix orphan I- tags (I- without preceding B- of same type)."""
    lines = sentence_block.split("\n")
    result = []
    prev_ner_type = None

    for line in lines:
        if line.startswith("#") or not line.strip():
            result.append(line)
            continue

        cols = line.split("\t")
        if len(cols) < 10:
            result.append(line)
            continue

        misc = cols[9]
        if misc.startswith("NER=I-"):
            ner_type = misc[6:]
            if prev_ner_type != ner_type:
                # Orphan I- tag — convert to B-
                cols[9] = f"NER=B-{ner_type}"
            prev_ner_type = ner_type
        elif misc.startswith("NER=B-"):
            prev_ner_type = misc[6:]
        else:
            prev_ner_type = None

        result.append("\t".join(cols))

    return "\n".join(result)


def add_cls_to_conllu(sentence_block: str, cls_label: str) -> str:
    """Add CLS label as a comment line in the CoNLL-U block."""
    lines = sentence_block.split("\n")
    # Insert after sent_id and text comments
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("#"):
            insert_at = i + 1
        else:
            break
    lines.insert(insert_at, f"# cls = {cls_label}")
    return "\n".join(lines)


def extract_sent_id(sentence_block: str) -> str | None:
    """Extract sent_id from a CoNLL-U sentence block."""
    for line in sentence_block.split("\n"):
        if line.startswith("# sent_id"):
            return line.split("=", 1)[1].strip()
    return None


def load_domain_annotations(domain: str) -> list[str]:
    """Load CoNLL-U annotations, apply corrections and CLS labels."""
    conllu_file = ANNOTATED_DIR / domain / "annotated.conllu"
    if not conllu_file.exists():
        print(f"  ⚠ No annotations for domain '{domain}', skipping", flush=True)
        return []

    # Load corrections and CLS labels
    corrections = load_corrections(domain)
    cls_labels = load_cls_labels(domain)
    print(f"  {domain}: {len(corrections)} corrections, {len(cls_labels)} CLS labels", flush=True)

    with open(conllu_file) as f:
        content = f.read()

    # Split into sentences
    sentences = []
    current: list[str] = []
    for line in content.split("\n"):
        if line.strip() == "" and current:
            sentences.append("\n".join(current))
            current = []
        else:
            current.append(line)
    if current:
        sentences.append("\n".join(current))

    # Apply corrections and CLS labels
    corrected_count = 0
    cls_count = 0
    result = []
    for sent in sentences:
        if not sent.strip():
            continue

        sent_id = extract_sent_id(sent)

        # Apply validation corrections
        if sent_id and sent_id in corrections:
            sent = apply_corrections_to_conllu(sent, corrections[sent_id])
            corrected_count += 1

        # Fix orphan I- NER tags (I- without preceding B-)
        sent = fix_bio_tags(sent)

        # Add CLS label
        if sent_id and sent_id in cls_labels:
            sent = add_cls_to_conllu(sent, cls_labels[sent_id])
            cls_count += 1

        result.append(sent)

    print(f"  {domain}: {len(result)} sentences ({corrected_count} corrected, {cls_count} with CLS)", flush=True)
    return result


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
        all_sentences.extend(sentences)

    print(f"\nTotal: {len(all_sentences)} sentences", flush=True)

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
        print(f"  {name}: {len(sents)} sentences", flush=True)

    return splits


def write_conllu(sentences: list[str], output_path: Path):
    """Write sentences to a CoNLL-U file."""
    with open(output_path, "w") as f:
        for sent in sentences:
            f.write(sent + "\n\n")
    print(f"  Written: {output_path} ({len(sentences)} sentences)", flush=True)


def export_corpus(domains: list[str]):
    """Export the full corpus with corrections applied."""
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading annotations, applying corrections and CLS labels...", flush=True)
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

    print(f"\nCorpus exported to {FINAL_DIR}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Export corpus to final format")
    parser.add_argument("--domains", nargs="+", required=True,
                        help="List of domain names to include")
    args = parser.parse_args()

    export_corpus(args.domains)


if __name__ == "__main__":
    main()

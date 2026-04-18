"""Annotate raw sentences with spaCy en_core_web_trf.

Produces CoNLL-U formatted output with NER, POS, and dependency annotations.

Usage:
    python -m corpus.pipeline.annotate --domain conversation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import spacy
from spacy.tokens import Doc

from .config import SPACY_MODEL, RAW_DIR, ANNOTATED_DIR


def load_sentences(domain: str) -> list[dict]:
    """Load preprocessed sentences for a domain."""
    sentences_file = RAW_DIR / domain / "sentences.jsonl"
    if not sentences_file.exists():
        raise FileNotFoundError(f"No sentences file at {sentences_file}. Run collect.py first.")

    sentences = []
    with open(sentences_file) as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(json.loads(line))
    return sentences


def annotate_sentence(doc: Doc) -> dict:
    """Extract all annotations from a spaCy Doc."""
    tokens = []
    for token in doc:
        tokens.append({
            "id": token.i + 1,  # CoNLL-U is 1-indexed
            "form": token.text,
            "lemma": token.lemma_,
            "upos": token.pos_,
            "xpos": token.tag_,
            "head": token.head.i + 1 if token.head != token else 0,  # 0 = root
            "deprel": token.dep_,
        })

    # NER spans
    ner_spans = []
    for ent in doc.ents:
        ner_spans.append({
            "start": ent.start,
            "end": ent.end,
            "label": ent.label_,
            "text": ent.text,
        })

    return {
        "text": doc.text,
        "tokens": tokens,
        "ner_spans": ner_spans,
    }


def to_conllu(annotated: dict, sent_id: str) -> str:
    """Convert annotated sentence to CoNLL-U format."""
    lines = [
        f"# sent_id = {sent_id}",
        f"# text = {annotated['text']}",
    ]

    # Encode NER as MISC field (BIO tags)
    bio_tags = ["O"] * len(annotated["tokens"])
    for span in annotated["ner_spans"]:
        for i in range(span["start"], span["end"]):
            prefix = "B" if i == span["start"] else "I"
            bio_tags[i] = f"{prefix}-{span['label']}"

    for token, bio in zip(annotated["tokens"], bio_tags):
        # CoNLL-U columns: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
        line = "\t".join([
            str(token["id"]),
            token["form"],
            token["lemma"],
            token["upos"],
            token["xpos"],
            "_",  # FEATS
            str(token["head"]),
            token["deprel"],
            "_",  # DEPS
            f"NER={bio}" if bio != "O" else "_",  # MISC
        ])
        lines.append(line)

    lines.append("")  # blank line between sentences
    return "\n".join(lines)


def annotate_domain(domain: str, batch_size: int = 100):
    """Annotate all sentences for a domain using spaCy."""
    print(f"Loading spaCy model: {SPACY_MODEL}...")
    nlp = spacy.load(SPACY_MODEL)

    sentences = load_sentences(domain)
    print(f"Loaded {len(sentences)} sentences from domain '{domain}'")

    output_dir = ANNOTATED_DIR / domain
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "annotated.conllu"

    texts = [s["text"] for s in sentences]
    metadata = sentences  # preserve source info

    json_file = output_dir / "annotated.jsonl"
    annotated_count = 0

    # Single pass: write both CoNLL-U and JSONL from one spaCy run
    with open(output_file, "w") as conllu_f, open(json_file, "w") as json_f:
        for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size)):
            sent_id = f"{domain}-{i:06d}"
            annotated = annotate_sentence(doc)

            # Write CoNLL-U
            conllu = to_conllu(annotated, sent_id)
            conllu_f.write(conllu + "\n")

            # Write JSONL
            annotated["sent_id"] = sent_id
            annotated["domain"] = domain
            if i < len(metadata):
                annotated["source"] = metadata[i].get("source", "")
            json_f.write(json.dumps(annotated) + "\n")

            annotated_count += 1
            if annotated_count % 1000 == 0:
                print(f"  Annotated {annotated_count}/{len(sentences)}...")

    print(f"Annotated {annotated_count} sentences")
    print(f"  CoNLL-U → {output_file}")
    print(f"  JSONL   → {json_file}")


def main():
    parser = argparse.ArgumentParser(description="Annotate domain sentences with spaCy")
    parser.add_argument("--domain", required=True, help="Domain name (e.g., conversation)")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    annotate_domain(args.domain, args.batch_size)


if __name__ == "__main__":
    main()

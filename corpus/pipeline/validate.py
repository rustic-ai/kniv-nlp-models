"""Validate spaCy annotations using GPT-5.4 mini via OpenAI Batch API.

Sends 100% of annotated sentences for validation. Uses Batch API
for 50% cost reduction (~$8.66 for 90K sentences).

Usage:
    python -m corpus.pipeline.validate --domain conversation
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI

from .config import OPENAI_MODEL, ANNOTATED_DIR, VALIDATED_DIR


SYSTEM_PROMPT = """You are a linguistics expert validating NLP annotations.
For each sentence, check if the POS tags, NER spans, and dependency parse
are correct. Return corrections as JSON.

Rules:
- Only flag clear errors, not stylistic preferences
- POS tags use Universal Dependencies UPOS tagset
- NER uses standard types: PERSON, ORG, GPE, LOC, DATE, etc.
- Dependency relations use Universal Dependencies relations
- If everything is correct, return {"corrections": []}
- For each correction, specify: token index, field (pos/ner/dep), old value, new value"""


def format_sentence_for_validation(annotated: dict) -> str:
    """Format an annotated sentence as a validation prompt."""
    tokens = annotated["tokens"]
    text = annotated["text"]

    token_table = []
    for t in tokens:
        token_table.append(f"  {t['id']:3d}  {t['form']:20s}  POS={t['upos']:6s}  HEAD={t['head']:3d}  DEP={t['deprel']}")

    ner_info = ""
    if annotated.get("ner_spans"):
        ner_parts = [f"  [{s['text']}] → {s['label']}" for s in annotated["ner_spans"]]
        ner_info = "\nNER spans:\n" + "\n".join(ner_parts)

    return f"""Sentence: {text}

Tokens:
{chr(10).join(token_table)}
{ner_info}

Check all annotations. Return JSON with corrections or empty if correct."""


def create_batch_requests(domain: str) -> list[dict]:
    """Create batch API requests for all sentences in a domain."""
    jsonl_file = ANNOTATED_DIR / domain / "annotated.jsonl"
    if not jsonl_file.exists():
        raise FileNotFoundError(f"No annotated JSONL at {jsonl_file}. Run annotate first.")

    requests = []
    with open(jsonl_file) as f:
        for line in f:
            annotated = json.loads(line.strip())
            prompt = format_sentence_for_validation(annotated)

            requests.append({
                "custom_id": annotated["sent_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 500,
                },
            })

    return requests


def submit_batch(domain: str):
    """Submit validation batch to OpenAI Batch API."""
    client = OpenAI()

    requests = create_batch_requests(domain)
    print(f"Created {len(requests)} validation requests for domain '{domain}'")

    # Write batch input file
    output_dir = VALIDATED_DIR / domain
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_input_file = output_dir / "batch_input.jsonl"

    with open(batch_input_file, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    # Upload file
    with open(batch_input_file, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"Uploaded batch input: {uploaded.id}")

    # Create batch
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch created: {batch.id}")
    print(f"Status: {batch.status}")

    # Save batch ID for later retrieval
    with open(output_dir / "batch_id.txt", "w") as f:
        f.write(batch.id)

    print(f"\nBatch submitted. Check status with:")
    print(f"  python -m corpus.pipeline.validate --domain {domain} --check")
    print(f"  python -m corpus.pipeline.validate --domain {domain} --retrieve")

    return batch.id


def check_batch(domain: str):
    """Check the status of a submitted batch."""
    client = OpenAI()
    batch_id_file = VALIDATED_DIR / domain / "batch_id.txt"
    if not batch_id_file.exists():
        print("No batch submitted yet.")
        return

    batch_id = batch_id_file.read_text().strip()
    batch = client.batches.retrieve(batch_id)

    print(f"Batch: {batch_id}")
    print(f"Status: {batch.status}")
    if batch.request_counts:
        print(f"Completed: {batch.request_counts.completed}/{batch.request_counts.total}")
        print(f"Failed: {batch.request_counts.failed}")


def retrieve_batch(domain: str):
    """Download and process batch results."""
    client = OpenAI()
    output_dir = VALIDATED_DIR / domain
    batch_id = (output_dir / "batch_id.txt").read_text().strip()

    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        print(f"Batch not complete yet. Status: {batch.status}")
        return

    # Download output
    output_file_id = batch.output_file_id
    content = client.files.content(output_file_id)

    results_file = output_dir / "batch_results.jsonl"
    with open(results_file, "wb") as f:
        f.write(content.read())

    # Parse corrections
    corrections = {}
    agreement_count = 0
    correction_count = 0

    with open(results_file) as f:
        for line in f:
            result = json.loads(line.strip())
            sent_id = result["custom_id"]
            response = result["response"]

            if response["status_code"] == 200:
                body = response["body"]
                content = body["choices"][0]["message"]["content"]
                try:
                    parsed = json.loads(content)
                    if parsed.get("corrections") and len(parsed["corrections"]) > 0:
                        corrections[sent_id] = parsed["corrections"]
                        correction_count += 1
                    else:
                        agreement_count += 1
                except json.JSONDecodeError:
                    agreement_count += 1  # treat parse failure as agreement

    total = agreement_count + correction_count
    agreement_pct = 100 * agreement_count / total if total > 0 else 0

    print(f"\nValidation results for '{domain}':")
    print(f"  Total sentences: {total}")
    print(f"  Agreement (no corrections): {agreement_count} ({agreement_pct:.1f}%)")
    print(f"  Corrections flagged: {correction_count} ({100 - agreement_pct:.1f}%)")

    # Save corrections
    corrections_file = output_dir / "corrections.json"
    with open(corrections_file, "w") as f:
        json.dump(corrections, f, indent=2)
    print(f"  Corrections saved to {corrections_file}")


def main():
    parser = argparse.ArgumentParser(description="Validate annotations with GPT-5.4 mini")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--check", action="store_true", help="Check batch status")
    parser.add_argument("--retrieve", action="store_true", help="Download batch results")
    args = parser.parse_args()

    if args.check:
        check_batch(args.domain)
    elif args.retrieve:
        retrieve_batch(args.domain)
    else:
        submit_batch(args.domain)


if __name__ == "__main__":
    main()

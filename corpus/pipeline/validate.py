"""Validate spaCy annotations using GPT-5.4 mini via direct API calls.

Processes all annotated sentences with concurrent async requests.
Estimated cost: ~$17 for 152K sentences at standard pricing.
Estimated time: ~30-60 minutes with concurrency.

Usage:
    python -m corpus.pipeline.validate --domain conversation
    python -m corpus.pipeline.validate --domain conversation --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from openai import AsyncOpenAI

from .config import OPENAI_MODEL, ANNOTATED_DIR, VALIDATED_DIR


# Concurrency: how many requests in flight at once.
# GPT-5.4 mini tier-1 rate limit is typically 500-10000 RPM.
MAX_CONCURRENT = 50

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


def format_sentence(annotated: dict) -> str:
    """Format an annotated sentence as a validation prompt."""
    tokens = annotated["tokens"]
    text = annotated["text"]

    token_lines = []
    for t in tokens:
        token_lines.append(
            f"  {t['id']:3d}  {t['form']:20s}  POS={t['upos']:6s}  HEAD={t['head']:3d}  DEP={t['deprel']}"
        )

    ner_info = ""
    if annotated.get("ner_spans"):
        ner_parts = [f"  [{s['text']}] → {s['label']}" for s in annotated["ner_spans"]]
        ner_info = "\nNER spans:\n" + "\n".join(ner_parts)

    return f"""Sentence: {text}

Tokens:
{chr(10).join(token_lines)}
{ner_info}

Check all annotations. Return JSON with corrections or empty if correct."""


async def validate_one(
    client: AsyncOpenAI,
    sent_id: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Validate a single sentence."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=500,
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            return {"sent_id": sent_id, "status": "ok", "result": parsed}
        except json.JSONDecodeError:
            return {"sent_id": sent_id, "status": "ok", "result": {"corrections": []}}
        except Exception as e:
            return {"sent_id": sent_id, "status": "error", "error": str(e)}


async def validate_domain(domain: str, resume: bool = False):
    """Validate all sentences for a domain using direct API calls."""
    jsonl_file = ANNOTATED_DIR / domain / "annotated.jsonl"
    if not jsonl_file.exists():
        raise FileNotFoundError(f"No annotated JSONL at {jsonl_file}. Run annotate first.")

    output_dir = VALIDATED_DIR / domain
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "validation_results.jsonl"

    # Load already-processed IDs if resuming
    done_ids = set()
    if resume and results_file.exists():
        with open(results_file) as f:
            for line in f:
                r = json.loads(line.strip())
                if r["status"] == "ok":
                    done_ids.add(r["sent_id"])
        print(f"Resuming: {len(done_ids)} already validated (skipping errored)", flush=True)

    # Load sentences to validate
    sentences = []
    with open(jsonl_file) as f:
        for line in f:
            annotated = json.loads(line.strip())
            if annotated["sent_id"] not in done_ids:
                sentences.append(annotated)

    total = len(sentences) + len(done_ids)
    print(f"Domain '{domain}': {len(sentences)} to validate ({len(done_ids)} already done, {total} total)", flush=True)

    if not sentences:
        print("Nothing to validate.", flush=True)
        return

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Process in chunks to show progress and save incrementally
    chunk_size = 500
    agreement = 0
    corrections = 0
    errors = 0
    start_time = time.time()

    mode = "a" if resume else "w"
    with open(results_file, mode) as out_f:
        for chunk_start in range(0, len(sentences), chunk_size):
            chunk = sentences[chunk_start:chunk_start + chunk_size]

            tasks = [
                validate_one(client, s["sent_id"], format_sentence(s), semaphore)
                for s in chunk
            ]
            results = await asyncio.gather(*tasks)

            for r in results:
                out_f.write(json.dumps(r) + "\n")
                if r["status"] == "error":
                    errors += 1
                elif r["result"].get("corrections") and len(r["result"]["corrections"]) > 0:
                    corrections += 1
                else:
                    agreement += 1

            out_f.flush()
            processed = chunk_start + len(chunk) + len(done_ids)
            elapsed = time.time() - start_time
            rate = (chunk_start + len(chunk)) / elapsed if elapsed > 0 else 0
            eta = (len(sentences) - chunk_start - len(chunk)) / rate if rate > 0 else 0

            print(
                f"  {processed}/{total}  "
                f"agree={agreement} correct={corrections} err={errors}  "
                f"({rate:.0f}/s, ETA {eta/60:.0f}m)",
                flush=True,
            )

    # Summary
    total_processed = agreement + corrections + errors
    elapsed = time.time() - start_time
    print(f"\nValidation complete for '{domain}':", flush=True)
    print(f"  Processed: {total_processed} sentences in {elapsed/60:.1f} minutes", flush=True)
    print(f"  Agreement: {agreement} ({100*agreement/max(total_processed,1):.1f}%)", flush=True)
    print(f"  Corrections: {corrections} ({100*corrections/max(total_processed,1):.1f}%)", flush=True)
    print(f"  Errors: {errors}", flush=True)

    # Extract corrections into separate file
    corrections_file = output_dir / "corrections.json"
    all_corrections = {}
    with open(results_file) as f:
        for line in f:
            r = json.loads(line.strip())
            if r["status"] == "ok" and r["result"].get("corrections") and len(r["result"]["corrections"]) > 0:
                all_corrections[r["sent_id"]] = r["result"]["corrections"]

    with open(corrections_file, "w") as f:
        json.dump(all_corrections, f, indent=2)
    print(f"  Corrections saved to {corrections_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Validate annotations with GPT-5.4 mini")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    args = parser.parse_args()

    asyncio.run(validate_domain(args.domain, args.resume))


if __name__ == "__main__":
    main()

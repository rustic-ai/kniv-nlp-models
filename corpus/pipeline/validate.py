"""Validate spaCy annotations using GPT-5.4 nano via batched API calls.

Sends batches of sentences per API call to amortize system prompt cost.
Each batch gets corrections as a JSON array.

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


MAX_CONCURRENT = 50
VALIDATE_BATCH_SIZE = 5  # sentences per API call

SYSTEM_PROMPT = """You are a linguistics expert validating NLP annotations.
You will receive multiple numbered sentences with their POS tags, NER spans,
and dependency parse annotations. Check each for correctness.

Rules:
- Only flag clear errors, not stylistic preferences
- POS tags use Universal Dependencies UPOS tagset
- NER uses standard types: PERSON, ORG, GPE, LOC, DATE, etc.
- Dependency relations use Universal Dependencies relations

Return a JSON object with one key per sentence number. Each value is a list
of corrections (empty list if correct).

Example output:
{
  "1": [],
  "2": [{"token_index": 3, "field": "pos", "old_value": "NOUN", "new_value": "PROPN"}],
  "3": []
}"""


def format_sentence(annotated: dict) -> str:
    """Format an annotated sentence for validation."""
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
{ner_info}"""


async def validate_batch(
    client: AsyncOpenAI,
    batch: list[tuple[str, dict]],  # [(sent_id, annotated), ...]
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Validate a batch of sentences in one API call."""
    async with semaphore:
        # Format numbered batch
        parts = []
        for i, (_, annotated) in enumerate(batch):
            parts.append(f"--- Sentence {i+1} ---\n{format_sentence(annotated)}")
        prompt = "\n\n".join(parts) + "\n\nReturn corrections as JSON."

        try:
            response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=300 * len(batch),
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)

            results = []
            for i, (sent_id, _) in enumerate(batch):
                key = str(i + 1)
                corrections = parsed.get(key, parsed.get(f"sentence_{i+1}", []))
                if not isinstance(corrections, list):
                    corrections = []
                results.append({
                    "sent_id": sent_id,
                    "status": "ok",
                    "result": {"corrections": corrections},
                })
            return results

        except json.JSONDecodeError:
            return [{"sent_id": sid, "status": "ok", "result": {"corrections": []}}
                    for sid, _ in batch]
        except Exception as e:
            return [{"sent_id": sid, "status": "error", "error": str(e)}
                    for sid, _ in batch]


async def validate_domain(domain: str, resume: bool = False):
    """Validate all sentences for a domain using batched API calls."""
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

    # Build API batches
    api_batches = []
    for i in range(0, len(sentences), VALIDATE_BATCH_SIZE):
        batch = [(s["sent_id"], s) for s in sentences[i:i + VALIDATE_BATCH_SIZE]]
        api_batches.append(batch)

    chunk_size = 100  # concurrent API batches per progress update
    agreement = 0
    corrections = 0
    errors = 0
    start_time = time.time()

    mode = "a" if resume else "w"
    with open(results_file, mode) as out_f:
        for chunk_start in range(0, len(api_batches), chunk_size):
            chunk = api_batches[chunk_start:chunk_start + chunk_size]

            tasks = [validate_batch(client, batch, semaphore) for batch in chunk]
            batch_results = await asyncio.gather(*tasks)

            for results in batch_results:
                for r in results:
                    out_f.write(json.dumps(r) + "\n")
                    if r["status"] == "error":
                        errors += 1
                    elif r["result"].get("corrections") and len(r["result"]["corrections"]) > 0:
                        corrections += 1
                    else:
                        agreement += 1

            out_f.flush()
            sentences_done = min((chunk_start + len(chunk)) * VALIDATE_BATCH_SIZE, len(sentences))
            processed = sentences_done + len(done_ids)
            elapsed = time.time() - start_time
            rate = sentences_done / elapsed if elapsed > 0 else 0
            eta = (len(sentences) - sentences_done) / rate if rate > 0 else 0

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
    parser = argparse.ArgumentParser(description="Validate annotations with GPT-5.4 nano")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    args = parser.parse_args()

    asyncio.run(validate_domain(args.domain, args.resume))


if __name__ == "__main__":
    main()

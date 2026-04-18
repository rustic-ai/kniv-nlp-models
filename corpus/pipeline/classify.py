"""Classify sentences by dialog act type using GPT-5.4 nano.

Assigns one of 9 uniko dialog act labels to each annotated sentence.
Runs independently of validation — can be applied to any annotated domain.

Labels (driven by uniko's cognitive memory processing):
  inform      → extract observation (primary knowledge source)
  correction  → flag existing knowledge for update
  agreement   → reinforce existing observation
  question    → record knowledge gap
  plan_commit → link to Goal/Task (promise, offer, suggest)
  request     → create action node (command, instruction)
  feedback    → skip extraction (acknowledgment, backchannel)
  social      → skip extraction (greeting, goodbye, thanks, apology)
  filler      → skip entirely (turn/time management, stalling)

Usage:
    python -m corpus.pipeline.classify --domain conversation
    python -m corpus.pipeline.classify --domain business --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from openai import AsyncOpenAI

from .config import OPENAI_MODEL, ANNOTATED_DIR, VALIDATED_DIR


CLS_LABELS = [
    "inform",
    "correction",
    "agreement",
    "question",
    "plan_commit",
    "request",
    "feedback",
    "social",
    "filler",
]

MAX_CONCURRENT = 50

SYSTEM_PROMPT = """You are classifying sentences by their communicative function.

Assign exactly ONE label from this list:
- inform: states a fact, opinion, or observation ("Revenue grew 15%", "The meeting is at 3pm")
- correction: corrects prior information ("No, it was actually Tuesday", "That's wrong, the budget is 50K")
- agreement: agrees with or confirms something ("Yes, exactly", "That's correct", "I agree")
- question: asks for information ("Where is the report?", "Did you finish?", "How much?")
- plan_commit: commits to action, offers, suggests, or plans ("I'll send it tomorrow", "Let's try option B", "We could use a different approach")
- request: asks someone to do something ("Send me the report", "Please update the spreadsheet", "Show me the data")
- feedback: acknowledges without adding information ("OK", "Got it", "I see", "Hmm", "Right")
- social: greeting, goodbye, thanks, apology ("Hi", "Thanks", "Sorry about that", "See you later")
- filler: turn management, stalling, non-content ("Um", "Well", "So", "Anyway", "Let me think")

You will receive numbered sentences. Return one label per line, in the same order.
Example input:
1. How are you?
2. Revenue increased 15%.
3. Send me the report.

Example output:
social
inform
request"""

# Sentences per API call — amortizes system prompt cost across batch
CLASSIFY_BATCH_SIZE = 20


async def classify_batch(
    client: AsyncOpenAI,
    batch: list[tuple[str, str]],  # [(sent_id, text), ...]
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Classify a batch of sentences in one API call."""
    async with semaphore:
        # Format numbered list
        numbered = "\n".join(f"{i+1}. {text}" for i, (_, text) in enumerate(batch))

        try:
            response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": numbered},
                ],
                max_completion_tokens=5 * len(batch),
            )
            content = response.choices[0].message.content.strip()
            labels = [l.strip().lower() for l in content.split("\n") if l.strip()]

            results = []
            for i, (sent_id, _) in enumerate(batch):
                if i < len(labels):
                    label = labels[i]
                    # Strip numbering if model includes it (e.g., "1. inform")
                    label = label.lstrip("0123456789.-) ").strip()
                    if label not in CLS_LABELS:
                        for valid in CLS_LABELS:
                            if valid in label:
                                label = valid
                                break
                        else:
                            label = "inform"
                    results.append({"sent_id": sent_id, "status": "ok", "cls_label": label})
                else:
                    results.append({"sent_id": sent_id, "status": "ok", "cls_label": "inform"})

            return results
        except Exception as e:
            return [{"sent_id": sid, "status": "error", "error": str(e)} for sid, _ in batch]


async def classify_domain(domain: str, resume: bool = False):
    """Classify all sentences for a domain."""
    jsonl_file = ANNOTATED_DIR / domain / "annotated.jsonl"
    if not jsonl_file.exists():
        raise FileNotFoundError(f"No annotated JSONL at {jsonl_file}. Run annotate first.")

    output_dir = VALIDATED_DIR / domain
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "cls_labels.jsonl"

    # Load already-classified IDs if resuming
    done_ids = set()
    if resume and results_file.exists():
        with open(results_file) as f:
            for line in f:
                r = json.loads(line.strip())
                if r["status"] == "ok":
                    done_ids.add(r["sent_id"])
        print(f"Resuming: {len(done_ids)} already classified (skipping errored)", flush=True)

    # Load sentences to classify
    sentences = []
    with open(jsonl_file) as f:
        for line in f:
            data = json.loads(line.strip())
            if data["sent_id"] not in done_ids:
                sentences.append(data)

    total = len(sentences) + len(done_ids)
    print(f"Domain '{domain}': {len(sentences)} to classify ({len(done_ids)} already done, {total} total)", flush=True)

    if not sentences:
        print("Nothing to classify.", flush=True)
        return

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Build batches of CLASSIFY_BATCH_SIZE sentences for API efficiency
    api_batches = []
    for i in range(0, len(sentences), CLASSIFY_BATCH_SIZE):
        batch = [(s["sent_id"], s["text"]) for s in sentences[i:i + CLASSIFY_BATCH_SIZE]]
        api_batches.append(batch)

    chunk_size = 25  # concurrent API batches per progress chunk (25 × 20 = 500 sentences)
    classified = 0
    errors = 0
    label_counts: dict[str, int] = {}
    start_time = time.time()

    mode = "a" if resume else "w"
    with open(results_file, mode) as out_f:
        for chunk_start in range(0, len(api_batches), chunk_size):
            chunk = api_batches[chunk_start:chunk_start + chunk_size]

            tasks = [classify_batch(client, batch, semaphore) for batch in chunk]
            batch_results = await asyncio.gather(*tasks)

            for results in batch_results:
                for r in results:
                    out_f.write(json.dumps(r) + "\n")
                    if r["status"] == "error":
                        errors += 1
                    else:
                        label = r["cls_label"]
                        label_counts[label] = label_counts.get(label, 0) + 1
                        classified += 1

            out_f.flush()
            sentences_done = min((chunk_start + len(chunk)) * CLASSIFY_BATCH_SIZE, len(sentences))
            processed = sentences_done + len(done_ids)
            elapsed = time.time() - start_time
            rate = sentences_done / elapsed if elapsed > 0 else 0
            eta = (len(sentences) - sentences_done) / rate if rate > 0 else 0

            print(
                f"  {processed}/{total}  "
                f"classified={classified} err={errors}  "
                f"({rate:.0f}/s, ETA {eta/60:.0f}m)",
                flush=True,
            )

    # Summary
    elapsed = time.time() - start_time
    print(f"\nClassification complete for '{domain}':", flush=True)
    print(f"  Processed: {classified + errors} sentences in {elapsed/60:.1f} minutes", flush=True)
    print(f"  Classified: {classified}", flush=True)
    print(f"  Errors: {errors}", flush=True)
    print(f"\n  Label distribution:", flush=True)
    for label in CLS_LABELS:
        count = label_counts.get(label, 0)
        pct = 100 * count / max(classified, 1)
        print(f"    {label:15s}: {count:7d} ({pct:5.1f}%)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Classify sentences by dialog act type")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    args = parser.parse_args()

    asyncio.run(classify_domain(args.domain, args.resume))


if __name__ == "__main__":
    main()

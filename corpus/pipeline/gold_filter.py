"""Gold-filter training data using an LLM.

Validates NER + CLS on corpus data, POS on UD data. Keeps only clean examples.
Reads from Parquet (already on HuggingFace) — no extra data upload needed.
Uses parallel requests with API backend for full GPU utilization.

Usage:
    # Filter using local Parquet files
    python -m corpus.pipeline.gold_filter --data-dir corpus/output/final

    # With vLLM API on Colab (64 concurrent requests)
    python -m corpus.pipeline.gold_filter \
        --data-dir corpus/output/final \
        --api-url http://localhost:8000/v1 --api-model Qwen/Qwen3-8B \
        --concurrency 64

    # Resume after interruption
    python -m corpus.pipeline.gold_filter --data-dir corpus/output/final --resume
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


# ── LLM Backends ────────────────────────────────────────────────

class LocalLLM:
    """llama-cpp-python backend for local GGUF models. Sequential only."""

    def __init__(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 2048):
        from llama_cpp import Llama
        print(f"Loading model: {model_path}", flush=True)
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )
        print("Model loaded.", flush=True)
        self.supports_parallel = False

    def ask(self, prompt: str) -> str:
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": f"/no_think\n{prompt}"}],
            max_tokens=30,
            temperature=0.0,
        )
        return response["choices"][0]["message"]["content"].strip()


class APILLM:
    """OpenAI-compatible API backend (for vLLM). Thread-safe for parallel use."""

    def __init__(self, base_url: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="unused")
        self.model = model
        print(f"Using API: {base_url} model={model}", flush=True)
        self.supports_parallel = True

    def ask(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": f"/no_think\n{prompt}"}],
            max_tokens=30,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()


# ── Prompt builders ─────────────────────────────────────────────

CLS_LABELS = ["inform", "correction", "agreement", "question",
              "plan_commit", "request", "feedback", "social", "filler"]


def is_correct(answer: str) -> bool:
    low = answer.lower()
    return "incorrect" not in low and "correct" in low


def build_ner_prompt(text: str, tokens: list[str], ner_tags: list[str]) -> str | None:
    """Build NER validation prompt from BIO tags. Returns None if no entities."""
    # Extract entity spans from BIO tags
    entities = []
    i = 0
    while i < len(ner_tags):
        if ner_tags[i].startswith("B-"):
            etype = ner_tags[i][2:]
            ew = [tokens[i]]
            j = i + 1
            while j < len(ner_tags) and ner_tags[j] == f"I-{etype}":
                ew.append(tokens[j])
                j += 1
            entities.append((" ".join(ew), etype))
            i = j
        else:
            i += 1

    if not entities:
        return None

    text = text[:500]
    entities = entities[:10]

    if len(entities) == 1:
        name, etype = entities[0]
        return (
            f'Sentence: "{text}"\n'
            f'The phrase "{name}" is tagged as entity type {etype}.\n'
            f"Is this entity type correct? Answer only: correct or incorrect."
        )
    ent_list = ", ".join(f'"{n}" -> {t}' for n, t in entities)
    return (
        f'Sentence: "{text}"\n'
        f"Entities: {ent_list}\n"
        f"Are all entity labels correct? Answer only: correct or incorrect."
    )


def build_cls_prompt(text: str, cls_label: str, prev_text: str = "") -> str | None:
    """Build CLS validation prompt."""
    if not cls_label:
        return None

    text = text[:500]
    labels_str = ", ".join(CLS_LABELS)

    if prev_text:
        return (
            f'Previous utterance: "{prev_text[:150]}"\n'
            f'Current utterance: "{text}"\n'
            f"This utterance is classified as: {cls_label}\n"
            f"Available labels: {labels_str}\n"
            f"Is this classification correct? Answer only: correct or incorrect."
        )
    return (
        f'Sentence: "{text}"\n'
        f"This sentence is classified as: {cls_label}\n"
        f"Available labels: {labels_str}\n"
        f"Is this classification correct? Answer only: correct or incorrect."
    )


def build_pos_prompt(text: str, tokens: list[str], pos_tags: list[str]) -> str | None:
    """Build POS validation prompt."""
    pairs = [(w, p) for w, p in zip(tokens, pos_tags) if p != "PUNCT"]
    if not pairs:
        return None

    text = text[:500]
    if len(pairs) > 8:
        import random
        pairs = random.sample(pairs, 8)

    tags_str = ", ".join(f'"{w}" -> {p}' for w, p in pairs)
    return (
        f'Sentence: "{text}"\n'
        f"POS tags: {tags_str}\n"
        f"Are all POS tags correct (Universal POS tagset)? Answer only: correct or incorrect."
    )


# ── Parallel validation engine ──────────────────────────────────

BATCH_SIZE = 256


def validate_batch(llm, prompts: list[tuple[int, str]], concurrency: int) -> dict[int, bool]:
    """Send prompts in parallel, return {index: is_clean} map."""
    results = {}

    if not llm.supports_parallel or concurrency <= 1:
        for idx, prompt in prompts:
            try:
                results[idx] = is_correct(llm.ask(prompt))
            except Exception:
                results[idx] = True
        return results

    def _check(item):
        idx, prompt = item
        try:
            return idx, is_correct(llm.ask(prompt))
        except Exception:
            return idx, True

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_check, item) for item in prompts]
        for future in as_completed(futures):
            idx, clean = future.result()
            results[idx] = clean

    return results


# ── Main filter logic ───────────────────────────────────────────

def filter_parquet(llm, data_dir: Path, output_dir: Path, concurrency: int,
                   resume: bool = False):
    """Gold-filter the corpus Parquet: NER + CLS on corpus, POS on UD-sourced rows."""

    train_path = data_dir / "train.parquet"
    if not train_path.exists():
        # Try alternate naming
        for pattern in data_dir.glob("train*.parquet"):
            train_path = pattern
            break

    print(f"Loading {train_path}...", flush=True)
    df = pd.read_parquet(train_path)
    print(f"  {len(df)} total sentences", flush=True)

    # Resume: load already-validated sent_ids
    progress_file = output_dir / "progress.json"
    clean_ids: set[str] = set()
    dirty_ids: set[str] = set()
    if resume and progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
        clean_ids = set(progress.get("clean", []))
        dirty_ids = set(progress.get("dirty", []))
        print(f"  Resuming: {len(clean_ids)} clean, {len(dirty_ids)} dirty already done", flush=True)

    stats = {"dirty_ner": 0, "dirty_cls": 0, "dirty_pos": 0, "clean": 0}
    start = time.time()

    # Process in batches
    rows = df.to_dict("records")
    total = len(rows)

    for batch_start in range(0, total, BATCH_SIZE):
        batch = rows[batch_start:batch_start + BATCH_SIZE]

        # Skip already-processed
        batch_filtered = []
        for row in batch:
            sid = row.get("sent_id", "")
            if sid in clean_ids or sid in dirty_ids:
                if sid in clean_ids:
                    stats["clean"] += 1
                continue
            batch_filtered.append(row)

        if not batch_filtered:
            continue

        # ── Pass 1: NER validation ──
        ner_prompts = []
        for i, row in enumerate(batch_filtered):
            tokens = row.get("tokens", [])
            ner_tags = row.get("ner_tags", [])
            text = row.get("text", "")
            prompt = build_ner_prompt(text, tokens, ner_tags)
            if prompt:
                ner_prompts.append((i, prompt))

        ner_results = validate_batch(llm, ner_prompts, concurrency) if ner_prompts else {}

        ner_clean = set()
        for i in range(len(batch_filtered)):
            if i in ner_results:
                if ner_results[i]:
                    ner_clean.add(i)
                else:
                    stats["dirty_ner"] += 1
                    dirty_ids.add(batch_filtered[i].get("sent_id", ""))
            else:
                ner_clean.add(i)  # No entities — auto-pass

        # ── Pass 2: CLS validation on NER-clean ──
        cls_prompts = []
        for i in ner_clean:
            row = batch_filtered[i]
            text = row.get("text", "")
            cls_label = row.get("cls", "")
            prev_text = row.get("prev_text", "") or ""
            prompt = build_cls_prompt(text, cls_label, prev_text)
            if prompt:
                cls_prompts.append((i, prompt))

        cls_results = validate_batch(llm, cls_prompts, concurrency) if cls_prompts else {}

        cls_clean = set()
        for i in ner_clean:
            if i in cls_results and not cls_results[i]:
                stats["dirty_cls"] += 1
                dirty_ids.add(batch_filtered[i].get("sent_id", ""))
            else:
                cls_clean.add(i)

        # ── Pass 3: POS validation on CLS-clean ──
        pos_prompts = []
        for i in cls_clean:
            row = batch_filtered[i]
            tokens = row.get("tokens", [])
            pos_tags = row.get("pos_tags", [])
            text = row.get("text", "")
            prompt = build_pos_prompt(text, tokens, pos_tags)
            if prompt:
                pos_prompts.append((i, prompt))

        pos_results = validate_batch(llm, pos_prompts, concurrency) if pos_prompts else {}

        for i in cls_clean:
            if i in pos_results and not pos_results[i]:
                stats["dirty_pos"] += 1
                dirty_ids.add(batch_filtered[i].get("sent_id", ""))
            else:
                stats["clean"] += 1
                clean_ids.add(batch_filtered[i].get("sent_id", ""))

        # Progress
        done = stats["clean"] + stats["dirty_ner"] + stats["dirty_cls"] + stats["dirty_pos"]
        elapsed = time.time() - start
        rate = done / max(elapsed, 0.1)
        dirty_total = stats["dirty_ner"] + stats["dirty_cls"] + stats["dirty_pos"]
        print(f"  {done}/{total} checked, "
              f"{stats['clean']} clean, {dirty_total} dirty "
              f"(NER:{stats['dirty_ner']} CLS:{stats['dirty_cls']} POS:{stats['dirty_pos']}), "
              f"{rate:.1f}/s", flush=True)

        # Save progress periodically
        if done % (BATCH_SIZE * 4) == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(progress_file, "w") as f:
                json.dump({"clean": list(clean_ids), "dirty": list(dirty_ids)}, f)

    # ── Save final results ──────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save progress
    with open(progress_file, "w") as f:
        json.dump({"clean": list(clean_ids), "dirty": list(dirty_ids)}, f)

    # Filter DataFrame and save gold Parquet
    gold_df = df[df["sent_id"].isin(clean_ids)]
    gold_path = output_dir / "train.parquet"
    gold_df.to_parquet(gold_path, index=False)

    # Copy dev/test unchanged
    import shutil
    for split in ["dev", "test"]:
        for pattern in data_dir.glob(f"{split}*.parquet"):
            shutil.copy2(pattern, output_dir / pattern.name)

    # Summary
    elapsed = time.time() - start
    dirty_total = stats["dirty_ner"] + stats["dirty_cls"] + stats["dirty_pos"]
    processed = stats["clean"] + dirty_total
    pct = 100 * stats["clean"] / max(processed, 1)

    print(f"\n{'='*50}", flush=True)
    print("GOLD FILTER SUMMARY", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  Clean: {stats['clean']}/{processed} ({pct:.1f}%)", flush=True)
    print(f"  Dirty NER: {stats['dirty_ner']}", flush=True)
    print(f"  Dirty CLS: {stats['dirty_cls']}", flush=True)
    print(f"  Dirty POS: {stats['dirty_pos']}", flush=True)
    print(f"  Time: {elapsed:.0f}s ({processed / max(elapsed, 0.1):.1f}/s)", flush=True)
    print(f"  Output: {gold_path} ({len(gold_df)} rows)", flush=True)

    return stats


# ── Main ────────────────────────────────────────────────────────

DEFAULT_MODEL = str(Path.home() / ".cache/llmfit/models/Qwen3-8B-Q6_K.gguf")


def main():
    parser = argparse.ArgumentParser(description="Gold-filter training data with LLM")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory with train/dev/test Parquet files")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: {data-dir}-gold)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to GGUF model (local mode)")
    parser.add_argument("--api-url", type=str, default=None,
                        help="OpenAI-compatible API URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--api-model", type=str, default="default",
                        help="Model name for API mode")
    parser.add_argument("--concurrency", type=int, default=64,
                        help="Parallel requests for API mode (default: 64)")
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from where we left off")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    output_dir = args.output_dir or Path(str(args.data_dir) + "-gold")

    # Load LLM
    if args.api_url:
        llm = APILLM(args.api_url, args.api_model)
    else:
        llm = LocalLLM(args.model, n_gpu_layers=args.n_gpu_layers)
        if args.concurrency > 1:
            print("Note: local mode is sequential (concurrency ignored)", flush=True)
            args.concurrency = 1

    filter_parquet(llm, args.data_dir, output_dir, args.concurrency, args.resume)


if __name__ == "__main__":
    main()

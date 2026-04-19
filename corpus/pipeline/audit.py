"""Audit training data quality using a local LLM.

Validates NER, POS, dependency parse, and CLS annotations independently.
Uses llama-cpp-python with any GGUF model for pluggable LLM backend.

Usage:
    # Audit all heads on a sample
    python -m corpus.pipeline.audit --head ner --sample 500
    python -m corpus.pipeline.audit --head pos --sample 500
    python -m corpus.pipeline.audit --head dep --sample 500
    python -m corpus.pipeline.audit --head cls --sample 500

    # Audit all heads
    python -m corpus.pipeline.audit --head all --sample 200

    # Use a different model
    python -m corpus.pipeline.audit --head ner --model /path/to/model.gguf

    # Full sweep (no sampling)
    python -m corpus.pipeline.audit --head ner --full
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


# ── LLM Backend ───────────────────────────────────────────────────

class LLMAuditor:
    """Pluggable LLM backend for annotation validation."""

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

    def ask(self, prompt: str, max_tokens: int = 100) -> str:
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": f"/no_think\n{prompt}"}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return response["choices"][0]["message"]["content"].strip()


# ── Head Auditors ─────────────────────────────────────────────────

def audit_ner(auditor: LLMAuditor, examples: list[dict], label_list: list[str]) -> dict:
    """Audit NER annotations — check each entity span."""
    results = {"total": 0, "correct": 0, "incorrect": 0, "errors": [], "skipped": 0}

    for ex in examples:
        words = ex["words"]
        ner_tags = ex["ner_tags"]
        text = ex.get("text", " ".join(words))

        # Extract entity spans
        i = 0
        while i < len(ner_tags):
            if ner_tags[i].startswith("B-"):
                etype = ner_tags[i][2:]
                entity_words = [words[i]]
                j = i + 1
                while j < len(ner_tags) and ner_tags[j] == f"I-{etype}":
                    entity_words.append(words[j])
                    j += 1

                entity_text = " ".join(entity_words)
                results["total"] += 1

                prompt = (
                    f'Sentence: "{text}"\n'
                    f'The phrase "{entity_text}" is tagged as entity type {etype}.\n'
                    f"Is this entity type correct? Answer only: correct or incorrect."
                )

                try:
                    answer = auditor.ask(prompt, max_tokens=30)
                    if "incorrect" in answer.lower():
                        results["incorrect"] += 1
                        results["errors"].append({
                            "text": text[:100],
                            "entity": entity_text,
                            "tagged_as": etype,
                            "llm_says": "incorrect",
                        })
                    elif "correct" in answer.lower():
                        results["correct"] += 1
                    else:
                        results["skipped"] += 1
                except Exception as e:
                    results["skipped"] += 1

                i = j
            else:
                i += 1

        if results["total"] % 50 == 0 and results["total"] > 0:
            print(f"  NER: {results['total']} entities checked, "
                  f"{results['incorrect']} incorrect so far", flush=True)

    return results


def audit_pos(auditor: LLMAuditor, examples: list[dict], label_list: list[str]) -> dict:
    """Audit POS annotations — check a sample of tokens per sentence."""
    results = {"total": 0, "correct": 0, "incorrect": 0, "errors": [], "skipped": 0}

    for ex in examples:
        words = ex["words"]
        pos_tags = ex["pos_tags"]
        text = ex.get("text", " ".join(words))

        # Check 2-3 random tokens per sentence (not all — too slow)
        indices = random.sample(range(len(words)), min(3, len(words)))

        for idx in indices:
            word = words[idx]
            pos = pos_tags[idx]
            if pos == "PUNCT":
                continue  # Skip punctuation — trivially correct

            results["total"] += 1

            prompt = (
                f'Sentence: "{text}"\n'
                f'The word "{word}" is tagged as {pos} (Universal POS tag).\n'
                f"Is this POS tag correct? Answer only: correct or incorrect."
            )

            try:
                answer = auditor.ask(prompt, max_tokens=30)
                if "incorrect" in answer.lower():
                    results["incorrect"] += 1
                    results["errors"].append({
                        "text": text[:100],
                        "word": word,
                        "tagged_as": pos,
                        "llm_says": "incorrect",
                    })
                elif "correct" in answer.lower():
                    results["correct"] += 1
                else:
                    results["skipped"] += 1
            except Exception:
                results["skipped"] += 1

        if results["total"] % 100 == 0 and results["total"] > 0:
            print(f"  POS: {results['total']} tokens checked, "
                  f"{results['incorrect']} incorrect so far", flush=True)

    return results


def audit_dep(auditor: LLMAuditor, examples: list[dict], label_list: list[str]) -> dict:
    """Audit dependency parse — check head and relation for sample tokens."""
    results = {"total": 0, "correct": 0, "incorrect": 0, "errors": [], "skipped": 0}

    for ex in examples:
        words = ex["words"]
        heads = ex["heads"]
        deprels = ex["deprels"]
        text = ex.get("text", " ".join(words))

        # Check 2 random non-root tokens per sentence
        non_root = [i for i in range(len(words)) if deprels[i] != "root" and deprels[i] != "punct"]
        if not non_root:
            continue
        indices = random.sample(non_root, min(2, len(non_root)))

        for idx in indices:
            word = words[idx]
            head_idx = heads[idx]
            deprel = deprels[idx]
            head_word = words[head_idx] if 0 <= head_idx < len(words) else "ROOT"

            results["total"] += 1

            prompt = (
                f'Sentence: "{text}"\n'
                f'The word "{word}" has dependency relation "{deprel}" with head word "{head_word}".\n'
                f"Is this dependency annotation correct? Answer only: correct or incorrect."
            )

            try:
                answer = auditor.ask(prompt, max_tokens=30)
                if "incorrect" in answer.lower():
                    results["incorrect"] += 1
                    results["errors"].append({
                        "text": text[:100],
                        "word": word,
                        "head": head_word,
                        "relation": deprel,
                        "llm_says": "incorrect",
                    })
                elif "correct" in answer.lower():
                    results["correct"] += 1
                else:
                    results["skipped"] += 1
            except Exception:
                results["skipped"] += 1

        if results["total"] % 100 == 0 and results["total"] > 0:
            print(f"  Dep: {results['total']} relations checked, "
                  f"{results['incorrect']} incorrect so far", flush=True)

    return results


def audit_cls(auditor: LLMAuditor, examples: list[dict], label_list: list[str]) -> dict:
    """Audit CLS (dialog act) annotations."""
    results = {"total": 0, "correct": 0, "incorrect": 0, "errors": [], "skipped": 0}

    labels_str = ", ".join(label_list)

    for ex in examples:
        text = ex.get("text", " ".join(ex.get("words", [])))
        cls_label = ex.get("cls_label", "")
        prev_text = ex.get("prev_text", "")
        if not cls_label:
            continue

        results["total"] += 1

        if prev_text:
            prompt = (
                f'Previous utterance: "{prev_text[:150]}"\n'
                f'Current utterance: "{text}"\n'
                f'This utterance is classified as: {cls_label}\n'
                f"Available labels: {labels_str}\n"
                f"Is this classification correct? Answer only: correct or incorrect."
            )
        else:
            prompt = (
                f'Sentence: "{text}"\n'
                f'This sentence is classified as: {cls_label}\n'
                f"Available labels: {labels_str}\n"
                f"Is this classification correct? Answer only: correct or incorrect."
            )

        try:
            answer = auditor.ask(prompt, max_tokens=30)
            if "incorrect" in answer.lower():
                results["incorrect"] += 1
                results["errors"].append({
                    "text": text[:100],
                    "prev_text": prev_text[:80] if prev_text else "",
                    "tagged_as": cls_label,
                    "llm_says": "incorrect",
                })
            elif "correct" in answer.lower():
                results["correct"] += 1
            else:
                results["skipped"] += 1
        except Exception:
            results["skipped"] += 1

        if results["total"] % 50 == 0 and results["total"] > 0:
            print(f"  CLS: {results['total']} checked, "
                  f"{results['incorrect']} incorrect so far", flush=True)

    return results


AUDITORS = {
    "ner": audit_ner,
    "pos": audit_pos,
    "dep": audit_dep,
    "cls": audit_cls,
}


# ── Main ──────────────────────────────────────────────────────────

DEFAULT_MODEL = str(Path.home() / ".cache/llmfit/models/Qwen3-8B-Q6_K.gguf")


def main():
    parser = argparse.ArgumentParser(description="Audit training data quality with local LLM")
    parser.add_argument("--head", required=True, choices=["ner", "pos", "dep", "cls", "all"],
                        help="Which annotation head to audit")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/prepared/deberta-v3-large-nlp-en"),
                        help="Path to prepared training data")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to GGUF model file")
    parser.add_argument("--sample", type=int, default=200,
                        help="Number of examples to sample (default: 200)")
    parser.add_argument("--full", action="store_true",
                        help="Audit all examples (no sampling)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--n-gpu-layers", type=int, default=-1,
                        help="GPU layers (-1 = all)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load data
    with open(args.data_dir / "label_vocabs.json") as f:
        vocabs = json.load(f)

    heads = ["ner", "pos", "dep", "cls"] if args.head == "all" else [args.head]
    all_results = {}

    # Load model once
    auditor = LLMAuditor(args.model, n_gpu_layers=args.n_gpu_layers)

    for head in heads:
        print(f"\n{'=' * 50}", flush=True)
        print(f"Auditing: {head.upper()}", flush=True)
        print(f"{'=' * 50}", flush=True)

        # Load appropriate data
        if head in ("ner", "cls"):
            with open(args.data_dir / "ner_train.json") as f:
                data = json.load(f)
        else:
            with open(args.data_dir / "ud_train.json") as f:
                data = json.load(f)

        # Sample
        if not args.full:
            n = min(args.sample, len(data))
            data = random.sample(data, n)
            print(f"Sampled {n} examples from {head} data", flush=True)

        # Run audit
        label_key = f"{head}_labels" if head != "cls" else "cls_labels"
        label_list = vocabs.get(label_key, [])

        start = time.time()
        results = AUDITORS[head](auditor, data, label_list)
        elapsed = time.time() - start

        # Report
        total = results["total"]
        correct = results["correct"]
        incorrect = results["incorrect"]
        skipped = results["skipped"]

        print(f"\n{head.upper()} Audit Results:", flush=True)
        print(f"  Total checked:  {total}", flush=True)
        print(f"  Correct:        {correct} ({100 * correct / max(total, 1):.1f}%)", flush=True)
        print(f"  Incorrect:      {incorrect} ({100 * incorrect / max(total, 1):.1f}%)", flush=True)
        print(f"  Skipped:        {skipped}", flush=True)
        print(f"  Time:           {elapsed:.0f}s ({total / max(elapsed, 1):.1f} checks/s)", flush=True)

        if results["errors"]:
            print(f"\n  Sample errors:", flush=True)
            for err in results["errors"][:10]:
                if head == "ner":
                    print(f"    '{err['entity']}' tagged {err['tagged_as']} in: {err['text']}", flush=True)
                elif head == "pos":
                    print(f"    '{err['word']}' tagged {err['tagged_as']} in: {err['text']}", flush=True)
                elif head == "dep":
                    print(f"    '{err['word']}' →{err['relation']}→ '{err['head']}' in: {err['text']}", flush=True)
                elif head == "cls":
                    prev = f" (prev: {err['prev_text'][:40]})" if err.get("prev_text") else ""
                    print(f"    tagged {err['tagged_as']}{prev} in: {err['text']}", flush=True)

        all_results[head] = results

    # Save results
    if args.output:
        # Strip non-serializable fields
        serializable = {}
        for head, res in all_results.items():
            serializable[head] = {
                "total": res["total"],
                "correct": res["correct"],
                "incorrect": res["incorrect"],
                "skipped": res["skipped"],
                "error_rate": res["incorrect"] / max(res["total"], 1),
                "errors": res["errors"][:50],
            }
        with open(args.output, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {args.output}", flush=True)

    # Summary
    if len(heads) > 1:
        print(f"\n{'=' * 50}", flush=True)
        print("SUMMARY", flush=True)
        print(f"{'=' * 50}", flush=True)
        for head, res in all_results.items():
            total = res["total"]
            err_rate = 100 * res["incorrect"] / max(total, 1)
            print(f"  {head.upper():4s}: {err_rate:.1f}% error rate ({res['incorrect']}/{total})", flush=True)


if __name__ == "__main__":
    main()

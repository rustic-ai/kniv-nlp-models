"""Silver SRL labeling using dannashao BERT SRL model (F1=86.7 on UP v1.0).

Runs a high-quality SRL model over the kniv corpus to generate silver
BIO labels with per-token confidence scores, then filters to keep only
high-confidence examples.

Follows the same pattern as SpanMarker NER silver labeling.

Usage:
    python prepare_srl_silver.py --corpus-dir corpus/output/annotated
    python prepare_srl_silver.py --corpus-dir corpus/output/annotated \
        --confidence 0.90 --max-sentences 300000
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ── Our 42-label SRL tag set ────────────────────────────────────

SRL_TAGS = [
    "O", "V",
    "B-ARG0", "I-ARG0", "B-ARG1", "I-ARG1", "B-ARG2", "I-ARG2",
    "B-ARG3", "I-ARG3", "B-ARG4", "I-ARG4",
    "B-ARGM-TMP", "I-ARGM-TMP", "B-ARGM-LOC", "I-ARGM-LOC",
    "B-ARGM-MNR", "I-ARGM-MNR", "B-ARGM-CAU", "I-ARGM-CAU",
    "B-ARGM-PRP", "I-ARGM-PRP", "B-ARGM-NEG", "I-ARGM-NEG",
    "B-ARGM-ADV", "I-ARGM-ADV", "B-ARGM-DIR", "I-ARGM-DIR",
    "B-ARGM-DIS", "I-ARGM-DIS", "B-ARGM-EXT", "I-ARGM-EXT",
    "B-ARGM-MOD", "I-ARGM-MOD", "B-ARGM-PRD", "I-ARGM-PRD",
    "B-ARGM-GOL", "I-ARGM-GOL", "B-ARGM-COM", "I-ARGM-COM",
    "B-ARGM-REC", "I-ARGM-REC",
]
SRL_TAG_SET = set(SRL_TAGS)


def map_label(label: str) -> str:
    """Map model output label to our 42-tag set."""
    if label == "O":
        return "O"
    if label in ("B-V", "I-V", "V"):
        return "V"
    if label in SRL_TAG_SET:
        return label
    # Strip continuation/reference prefixes: B-C-ARG0 → B-ARG0
    for prefix in ("B-C-", "I-C-", "B-R-", "I-R-"):
        if label.startswith(prefix):
            rest = label[len(prefix):]
            mapped = f"{label[0]}-{rest}"
            if mapped in SRL_TAG_SET:
                return mapped
    # ARG5 → ARG4
    label = label.replace("ARG5", "ARG4")
    if label in SRL_TAG_SET:
        return label
    return "O"


# ── Corpus loading ──────────────────────────────────────────────

def load_corpus_sentences(corpus_dir: str, max_sentences: int | None = None) -> list[dict]:
    """Load sentences from kniv corpus annotated.jsonl files."""
    corpus_path = Path(corpus_dir)
    jsonl_files = sorted(corpus_path.glob("*/annotated.jsonl"))
    if not jsonl_files:
        jsonl_files = sorted(corpus_path.glob("**/annotated.jsonl"))

    print(f"Found {len(jsonl_files)} annotated files", flush=True)
    sentences = []

    for jsonl_file in jsonl_files:
        domain = jsonl_file.parent.name
        count = 0
        with open(jsonl_file) as f:
            for line in f:
                if max_sentences and len(sentences) >= max_sentences:
                    break
                obj = json.loads(line)
                tokens = obj["tokens"]
                words = [t["form"] for t in tokens]
                pos_tags = [t["upos"] for t in tokens]

                # Skip sentences too short or too long for useful SRL
                if len(words) < 4 or len(words) > 100:
                    continue

                sentences.append({
                    "words": words,
                    "text": obj.get("text", " ".join(words)),
                    "pos_tags": pos_tags,
                    "domain": domain,
                })
                count += 1

        print(f"  {domain}: {count:,} sentences", flush=True)
        if max_sentences and len(sentences) >= max_sentences:
            break

    print(f"Total: {len(sentences):,} sentences loaded", flush=True)
    return sentences


def find_predicates(sentences: list[dict]) -> list[dict]:
    """Create one (sentence, predicate_idx) pair for each verb."""
    pairs = []
    for sent in sentences:
        for i, pos in enumerate(sent["pos_tags"]):
            if pos == "VERB":
                pairs.append({
                    "words": sent["words"],
                    "text": sent["text"],
                    "predicate_idx": i,
                    "domain": sent.get("domain", ""),
                })
    print(f"Found {len(pairs):,} predicate instances "
          f"({len(pairs)/len(sentences):.1f} verbs/sentence avg)", flush=True)
    return pairs


# ── Model inference ─────────────────────────────────────────────

def run_silver_labeling(
    pairs: list[dict],
    model_name: str,
    confidence_threshold: float,
    batch_size: int,
    max_length: int,
    device: torch.device,
    save_every: int = 50000,
    output_path: str | None = None,
) -> list[dict]:
    """Run SRL model and filter by confidence."""

    print(f"Loading model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.eval().to(device)

    id2label = model.config.id2label
    print(f"Model labels: {len(id2label)} ({', '.join(list(id2label.values())[:10])}...)",
          flush=True)

    results = []
    skipped_no_args = 0
    skipped_low_conf = 0
    skipped_align = 0

    num_batches = (len(pairs) + batch_size - 1) // batch_size
    for batch_start in tqdm(range(0, len(pairs), batch_size),
                            total=num_batches, desc="Silver SRL"):
        batch = pairs[batch_start : batch_start + batch_size]

        # Insert [V] marker before each predicate
        batch_marked_words = []
        for item in batch:
            w = list(item["words"])
            idx = item["predicate_idx"]
            batch_marked_words.append(w[:idx] + ["[V]"] + w[idx:])

        # Tokenize (is_split_into_words aligns subwords to word indices)
        encodings = tokenizer(
            batch_marked_words,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(
                input_ids=encodings["input_ids"].to(device),
                attention_mask=encodings["attention_mask"].to(device),
            ).logits

        probs = torch.softmax(logits, dim=-1).cpu()
        pred_ids = logits.argmax(dim=-1).cpu()

        for j in range(len(batch)):
            orig_words = batch[j]["words"]
            pred_idx = batch[j]["predicate_idx"]
            word_ids = encodings.word_ids(j)

            # Align subword predictions → one label per marked word
            marked_labels = []
            marked_confs = []
            prev_wid = None
            for k, wid in enumerate(word_ids):
                if wid is not None and wid != prev_wid:
                    lid = pred_ids[j, k].item()
                    marked_labels.append(id2label.get(lid, "O"))
                    marked_confs.append(probs[j, k, lid].item())
                prev_wid = wid

            # Remove the [V] marker position (it's at index pred_idx in marked list)
            if len(marked_labels) <= pred_idx:
                skipped_align += 1
                continue

            labels = marked_labels[:pred_idx] + marked_labels[pred_idx + 1:]
            confs = marked_confs[:pred_idx] + marked_confs[pred_idx + 1:]

            # Pad if truncation cut words short
            if len(labels) < len(orig_words):
                pad_n = len(orig_words) - len(labels)
                labels.extend(["O"] * pad_n)
                confs.extend([1.0] * pad_n)
            labels = labels[: len(orig_words)]
            confs = confs[: len(orig_words)]

            # Map to our 42-tag set and force predicate token to V
            mapped = [map_label(l) for l in labels]
            mapped[pred_idx] = "V"

            # Must have at least one argument span
            arg_confs = [c for t, c in zip(mapped, confs) if t not in ("O", "V")]
            if not arg_confs:
                skipped_no_args += 1
                continue

            # Confidence filter on argument tokens
            mean_conf = sum(arg_confs) / len(arg_confs)
            if mean_conf < confidence_threshold:
                skipped_low_conf += 1
                continue

            results.append({
                "words": orig_words,
                "text": batch[j]["text"],
                "srl_tags": mapped,
                "predicate_idx": pred_idx,
                "confidence": round(mean_conf, 4),
                "domain": batch[j].get("domain", ""),
            })

        # Periodic save
        if save_every and output_path and len(results) % save_every < batch_size:
            _save_checkpoint(results, output_path)

    print(f"\nResults: {len(results):,} high-confidence examples", flush=True)
    print(f"  Skipped (no args): {skipped_no_args:,}", flush=True)
    print(f"  Skipped (low conf < {confidence_threshold}): {skipped_low_conf:,}", flush=True)
    print(f"  Skipped (alignment): {skipped_align:,}", flush=True)
    return results


def _save_checkpoint(results: list[dict], output_path: str):
    """Save intermediate checkpoint."""
    ckpt_path = output_path.replace(".json", "_checkpoint.json")
    train_data = [{k: v for k, v in r.items() if k != "confidence" and k != "domain"}
                  for r in results]
    with open(ckpt_path, "w") as f:
        json.dump(train_data, f)
    print(f"  [checkpoint] {len(results):,} examples saved", flush=True)


# ── Main ─────────────────────────────────────────────��──────────

def main():
    parser = argparse.ArgumentParser(
        description="Silver SRL labeling with confidence filtering")
    parser.add_argument(
        "--corpus-dir", type=str, default="corpus/output/annotated",
        help="Path to kniv corpus annotated directory")
    parser.add_argument(
        "--output", type=str,
        default="data/prepared/kniv-deberta-cascade/srl_silver_dannashao.json",
        help="Output JSON path")
    parser.add_argument(
        "--model", type=str,
        default="dannashao/bert-base-uncased-finetuned-advanced-srl_arg",
        help="HuggingFace SRL model name")
    parser.add_argument(
        "--confidence", type=float, default=0.85,
        help="Min mean argument confidence (0.85 = keep top ~60-70%%)")
    parser.add_argument(
        "--max-sentences", type=int, default=None,
        help="Max corpus sentences to process (None = all)")
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Inference batch size")
    parser.add_argument(
        "--max-length", type=int, default=128,
        help="Max token length for model input")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # 1. Load corpus sentences
    sentences = load_corpus_sentences(args.corpus_dir, args.max_sentences)

    # 2. Find verb predicates
    pairs = find_predicates(sentences)

    # 3. Run silver labeling with confidence filtering
    results = run_silver_labeling(
        pairs, args.model, args.confidence,
        args.batch_size, args.max_length, device,
        output_path=args.output,
    )

    # 4. Stats
    domain_counts = Counter(r["domain"] for r in results)
    print(f"\nPer-domain:", flush=True)
    for d, c in domain_counts.most_common():
        print(f"  {d}: {c:,}", flush=True)

    confs = [r["confidence"] for r in results]
    if confs:
        print(f"\nConfidence: mean={sum(confs)/len(confs):.3f}, "
              f"min={min(confs):.3f}, max={max(confs):.3f}", flush=True)

    tag_counts = Counter()
    for r in results:
        for t in r["srl_tags"]:
            tag_counts[t] += 1
    print(f"\nLabel distribution (top 15):", flush=True)
    for tag, count in tag_counts.most_common(15):
        pct = 100.0 * count / sum(tag_counts.values())
        print(f"  {tag}: {count:,} ({pct:.1f}%)", flush=True)

    # 5. Save training data (strip metadata)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    train_data = [
        {"words": r["words"], "text": r["text"],
         "srl_tags": r["srl_tags"], "predicate_idx": r["predicate_idx"]}
        for r in results
    ]
    with open(args.output, "w") as f:
        json.dump(train_data, f)
    print(f"\nSaved {len(train_data):,} examples to {args.output}", flush=True)

    # Save full metadata for analysis
    meta_path = args.output.replace(".json", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(results, f)
    print(f"Saved metadata to {meta_path}", flush=True)


if __name__ == "__main__":
    main()

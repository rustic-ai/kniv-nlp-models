"""Build a balanced CLS dataset using SwDA labeler + GPT labels + conversation data.

1. Runs the trained SwDA labeler on the conversation corpus (with prev_text context)
2. Cross-references with existing GPT labels for agreement filtering
3. Combines SwDA gold + high-confidence conversation silver
4. Balances across all 9 classes

Usage:
    python prepare_cls_balanced.py --labeler-dir outputs/cls_labeler/best
    python prepare_cls_balanced.py --labeler-dir outputs/cls_labeler/best \
        --confidence 0.90 --target-per-class 5000
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CLS_LABELS = [
    "inform", "correction", "agreement", "question",
    "plan_commit", "request", "feedback", "social", "filler",
]
LABEL2ID = {l: i for i, l in enumerate(CLS_LABELS)}

CORPUS_DIR = Path(__file__).parent.parent.parent / "corpus" / "output"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "prepared" / "kniv-deberta-cascade"


def load_gpt_labels(domain: str = "conversation") -> dict[str, str]:
    """Load GPT-classified CLS labels by sent_id."""
    path = CORPUS_DIR / "validated" / domain / "cls_labels.jsonl"
    labels = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("status") == "ok":
                labels[obj["sent_id"]] = obj["cls_label"]
    print(f"Loaded {len(labels):,} GPT labels for {domain}", flush=True)
    return labels


def load_conversation_sentences(domain: str = "conversation") -> list[dict]:
    """Load annotated conversation sentences with prev_text context."""
    path = CORPUS_DIR / "annotated" / domain / "annotated.jsonl"
    sentences = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            sent = {
                "text": obj["text"],
                "sent_id": obj["sent_id"],
                "source": obj.get("source", ""),
            }
            if "prev_text" in obj:
                sent["prev_text"] = obj["prev_text"]
            sentences.append(sent)
    print(f"Loaded {len(sentences):,} conversation sentences", flush=True)
    return sentences


def run_labeler(
    sentences: list[dict],
    model_dir: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> list[dict]:
    """Run the SwDA labeler on sentences, return predictions with confidence."""

    print(f"Loading labeler: {model_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval().to(device)

    results = []
    for batch_start in tqdm(range(0, len(sentences), batch_size), desc="Labeling"):
        batch = sentences[batch_start : batch_start + batch_size]

        # Encode with context where available
        texts = []
        text_pairs = []
        for sent in batch:
            if "prev_text" in sent:
                texts.append(sent["prev_text"])
                text_pairs.append(sent["text"])
            else:
                texts.append(sent["text"])
                text_pairs.append(None)

        # Tokenize — handle mixed single/pair inputs
        encodings = tokenizer(
            texts,
            text_pair=[p for p in text_pairs],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ) if all(p is not None for p in text_pairs) else None

        # If mixed (some with context, some without), process individually
        if encodings is None:
            batch_input_ids = []
            batch_attention = []
            for t, tp in zip(texts, text_pairs):
                if tp is not None:
                    enc = tokenizer(t, tp, max_length=max_length, padding="max_length",
                                    truncation=True, return_tensors="pt")
                else:
                    enc = tokenizer(t, max_length=max_length, padding="max_length",
                                    truncation=True, return_tensors="pt")
                batch_input_ids.append(enc["input_ids"].squeeze(0))
                batch_attention.append(enc["attention_mask"].squeeze(0))
            encodings = {
                "input_ids": torch.stack(batch_input_ids),
                "attention_mask": torch.stack(batch_attention),
            }

        with torch.no_grad():
            logits = model(
                input_ids=encodings["input_ids"].to(device),
                attention_mask=encodings["attention_mask"].to(device),
            ).logits

        probs = torch.softmax(logits, dim=-1).cpu()
        pred_ids = logits.argmax(dim=-1).cpu()

        for j in range(len(batch)):
            pred_label = CLS_LABELS[pred_ids[j].item()]
            conf = probs[j, pred_ids[j].item()].item()
            results.append({
                **batch[j],
                "labeler_label": pred_label,
                "labeler_confidence": round(conf, 4),
            })

    return results


def build_balanced_dataset(
    swda_data: list[dict],
    conv_labeled: list[dict],
    gpt_labels: dict[str, str],
    confidence_threshold: float,
    target_per_class: int,
) -> list[dict]:
    """Build a balanced CLS dataset from SwDA gold + high-confidence conversation silver."""

    # Bucket SwDA data by label
    swda_by_label = defaultdict(list)
    for ex in swda_data:
        swda_by_label[ex["cls_label"]].append(ex)

    # Filter conversation data: high confidence AND/OR GPT agreement
    conv_by_label = defaultdict(list)
    agree_count = 0
    for ex in conv_labeled:
        gpt_label = gpt_labels.get(ex["sent_id"])
        labeler_label = ex["labeler_label"]
        conf = ex["labeler_confidence"]

        # High confidence from labeler
        if conf >= confidence_threshold:
            conv_by_label[labeler_label].append({
                "text": ex["text"],
                "cls_label": labeler_label,
                "prev_text": ex.get("prev_text"),
                "source": f"conv_{ex.get('source', 'unknown')}",
                "confidence": conf,
                "gpt_agrees": gpt_label == labeler_label,
            })
            if gpt_label == labeler_label:
                agree_count += 1
        # Lower confidence but GPT agrees — still keep
        elif gpt_label == labeler_label and conf >= confidence_threshold * 0.8:
            conv_by_label[labeler_label].append({
                "text": ex["text"],
                "cls_label": labeler_label,
                "prev_text": ex.get("prev_text"),
                "source": f"conv_{ex.get('source', 'unknown')}",
                "confidence": conf,
                "gpt_agrees": True,
            })
            agree_count += 1

    print(f"\nConversation silver labels by class:", flush=True)
    for label in CLS_LABELS:
        n = len(conv_by_label[label])
        print(f"  {label}: {n:,}", flush=True)
    print(f"GPT agreement rate: {agree_count}/{sum(len(v) for v in conv_by_label.values())}", flush=True)

    # Build balanced dataset: target_per_class from each source
    random.seed(42)
    balanced = []
    for label in CLS_LABELS:
        swda_pool = swda_by_label[label]
        conv_pool = conv_by_label[label]
        # Prefer GPT-agreeing conversation examples
        conv_agree = [ex for ex in conv_pool if ex.get("gpt_agrees")]
        conv_other = [ex for ex in conv_pool if not ex.get("gpt_agrees")]

        # Take up to target from: conv_agree first, then SwDA, then conv_other
        selected = []
        for pool in [conv_agree, swda_pool, conv_other]:
            remaining = target_per_class - len(selected)
            if remaining <= 0:
                break
            random.shuffle(pool)
            selected.extend(pool[:remaining])

        balanced.extend(selected)
        print(f"  {label}: {len(selected):,} "
              f"(conv_agree={min(len(conv_agree), target_per_class)}, "
              f"swda={min(len(swda_pool), max(0, target_per_class - len(conv_agree)))}, "
              f"conv_other={max(0, len(selected) - len(conv_agree) - len(swda_pool))})",
              flush=True)

    random.shuffle(balanced)
    return balanced


def main():
    parser = argparse.ArgumentParser(description="Build balanced CLS dataset")
    parser.add_argument("--labeler-dir", type=str, required=True,
                        help="Path to trained CLS labeler model")
    parser.add_argument("--output", type=str,
                        default=str(DATA_DIR / "cls_balanced_train.json"),
                        help="Output JSON path")
    parser.add_argument("--confidence", type=float, default=0.85,
                        help="Min labeler confidence for silver labels")
    parser.add_argument("--target-per-class", type=int, default=5000,
                        help="Target examples per class in balanced dataset")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Labeler inference batch size")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Max token length")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # 1. Load GPT labels
    gpt_labels = load_gpt_labels("conversation")

    # 2. Load conversation sentences
    conv_sentences = load_conversation_sentences("conversation")

    # 3. Run SwDA labeler on conversation data
    conv_labeled = run_labeler(
        conv_sentences, args.labeler_dir,
        args.batch_size, args.max_length, device,
    )

    # 4. Load SwDA gold data
    with open(DATA_DIR / "cls_swda_mrda_train.json") as f:
        swda_data = json.load(f)
    print(f"Loaded {len(swda_data):,} SwDA gold examples", flush=True)

    # 5. Build balanced dataset
    balanced = build_balanced_dataset(
        swda_data, conv_labeled, gpt_labels,
        args.confidence, args.target_per_class,
    )

    # 6. Stats
    dist = Counter(ex["cls_label"] for ex in balanced)
    print(f"\nFinal balanced dataset: {len(balanced):,} examples", flush=True)
    for label in CLS_LABELS:
        print(f"  {label}: {dist[label]:,}", flush=True)

    # Count examples with prev_text
    has_context = sum(1 for ex in balanced if ex.get("prev_text"))
    print(f"With conversation context: {has_context:,} ({100*has_context/len(balanced):.0f}%)",
          flush=True)

    # 7. Save (strip metadata for training)
    train_data = []
    for ex in balanced:
        item = {"text": ex["text"], "cls_label": ex["cls_label"]}
        if ex.get("prev_text"):
            item["prev_text"] = ex["prev_text"]
        train_data.append(item)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(train_data, f)
    print(f"\nSaved to {args.output}", flush=True)


if __name__ == "__main__":
    main()

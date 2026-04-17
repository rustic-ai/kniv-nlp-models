"""Full benchmark of a trained multi-task model on held-out test sets.

Runs inference on CoNLL-2003 test (NER), UD EWT test (POS + dep),
and bootstrap CLS labels.  Reports all metrics and saves eval_results.json.

Usage:
    python models/distilroberta-nlp-en/evaluate_model.py --model-dir outputs/distilroberta-nlp-en/final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Add project root to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.evaluate import (
    evaluate_ner, evaluate_pos, evaluate_dep, evaluate_cls,
    print_report, save_results,
)
from dep2label import decode_sentence
from model import MultiTaskNLPModel


DATA_DIR = Path(__file__).parent.parent.parent / "data" / "prepared" / "distilroberta-nlp-en"


def predict_token_labels(model, tokenizer, words, label_list, logits_key, device, max_length=128):
    """Run inference and extract aligned per-word predictions."""
    encoding = tokenizer(
        words, is_split_into_words=True,
        max_length=max_length, padding="max_length", truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    logits = outputs[logits_key][0].cpu()  # [seq_len, num_labels]
    preds = logits.argmax(dim=-1).tolist()

    # Align subword predictions back to words
    word_ids = encoding.word_ids()
    word_preds = []
    prev_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != prev_word_id:
            label_idx = preds[idx]
            word_preds.append(label_list[label_idx] if label_idx < len(label_list) else "O")
        prev_word_id = word_id

    return word_preds[:len(words)]


def predict_cls(model, tokenizer, text, cls_labels, device, max_length=128):
    """Run inference for sentence classification."""
    encoding = tokenizer(
        text, max_length=max_length, padding="max_length", truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    logits = outputs["cls_logits"][0].cpu()
    pred_idx = logits.argmax().item()
    return cls_labels[pred_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--encoder", default="distilroberta-base")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = MultiTaskNLPModel.load(str(args.model_dir), args.encoder).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), add_prefix_space=True)

    with open(DATA_DIR / "label_vocabs.json") as f:
        vocabs = json.load(f)

    results = {}

    # ── NER evaluation ────────────────────────────────────────
    print("\nEvaluating NER on CoNLL-2003 test...")
    with open(DATA_DIR / "conll_test.json") as f:
        conll_test = json.load(f)

    gold_ner, pred_ner = [], []
    for ex in conll_test:
        pred = predict_token_labels(
            model, tokenizer, ex["words"], vocabs["ner_labels"], "ner_logits", device,
        )
        gold = ex["ner_tags"]
        # Align lengths: truncation may shorten pred
        min_len = min(len(gold), len(pred))
        gold_ner.append(gold[:min_len])
        pred_ner.append(pred[:min_len])

    results["ner"] = evaluate_ner(gold_ner, pred_ner)

    # ── POS evaluation ────────────────────────────────────────
    print("Evaluating POS on UD EWT test...")
    with open(DATA_DIR / "ud_test.json") as f:
        ud_test = json.load(f)

    gold_pos, pred_pos = [], []
    for ex in ud_test:
        pred = predict_token_labels(
            model, tokenizer, ex["words"], vocabs["pos_labels"], "pos_logits", device,
        )
        gold = ex["pos_tags"]
        min_len = min(len(gold), len(pred))
        gold_pos.append(gold[:min_len])
        pred_pos.append(pred[:min_len])

    results["pos"] = evaluate_pos(gold_pos, pred_pos)

    # ── Dep evaluation ────────────────────────────────────────
    print("Evaluating Dep on UD EWT test...")
    gold_heads_all, pred_heads_all = [], []
    gold_rels_all, pred_rels_all = [], []

    for ex in ud_test:
        dep_preds = predict_token_labels(
            model, tokenizer, ex["words"], vocabs["dep_labels"], "dep_logits", device,
        )
        # Truncate dep preds to match word count (truncation may shorten)
        n = min(len(dep_preds), len(ex["words"]))
        dep_preds = dep_preds[:n]
        pos_tags = ex["pos_tags"][:n]
        try:
            pred_heads, pred_rels = decode_sentence(dep_preds, pos_tags)
        except Exception:
            pred_heads = [-1] * n
            pred_rels = ["_"] * n

        gold_h = ex["heads"][:n]
        gold_r = ex["deprels"][:n]
        pred_heads = pred_heads[:n]
        pred_rels = pred_rels[:n]
        gold_heads_all.append(gold_h)
        pred_heads_all.append(pred_heads)
        gold_rels_all.append(gold_r)
        pred_rels_all.append(pred_rels)

    results["dep"] = evaluate_dep(gold_heads_all, pred_heads_all, gold_rels_all, pred_rels_all)

    # ── CLS evaluation ────────────────────────────────────────
    print("Evaluating CLS on bootstrap labels...")
    gold_cls, pred_cls_list = [], []
    all_examples = ud_test + conll_test
    for ex in all_examples:
        if "cls_label" not in ex:
            continue
        text = ex.get("text", " ".join(ex["words"]))
        gold_cls.append(ex["cls_label"])
        pred_cls_list.append(predict_cls(model, tokenizer, text, vocabs["cls_labels"], device))

    if gold_cls:
        results["cls"] = evaluate_cls(gold_cls, pred_cls_list, vocabs["cls_labels"])

    # ── Report ────────────────────────────────────────────────
    print_report(results)

    output_path = args.output or (args.model_dir / "eval_results.json")
    save_results(results, output_path)


if __name__ == "__main__":
    main()

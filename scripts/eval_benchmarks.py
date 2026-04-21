"""Evaluate the kniv teacher model against standard NER benchmarks.

Downloads the model from HuggingFace and evaluates on:
- CoNLL-2003 (4 NER types: PER, ORG, LOC, MISC)
- UD EWT test set (POS accuracy)

Usage:
    python scripts/eval_benchmarks.py
"""

import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from seqeval.metrics import classification_report, f1_score
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent.parent / "models" / "deberta-v3-large-nlp-en"))
from model import MultiTaskNLPModel


# Our 18 types → CoNLL-2003 4 types
KNIV_TO_CONLL = {
    "PERSON": "PER", "ORG": "ORG", "GPE": "LOC", "LOC": "LOC",
    "NORP": "MISC", "FAC": "MISC", "PRODUCT": "MISC", "EVENT": "MISC",
    "WORK_OF_ART": "MISC", "LAW": "MISC", "LANGUAGE": "MISC",
    "DATE": "MISC", "TIME": "MISC", "PERCENT": "MISC", "MONEY": "MISC",
    "QUANTITY": "MISC", "ORDINAL": "MISC", "CARDINAL": "MISC",
}

CONLL_LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']


def load_model(model_dir: str, encoder: str):
    """Load the kniv teacher model."""
    model = MultiTaskNLPModel.load(model_dir, encoder).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(encoder)
    return model, tokenizer, device


def map_prediction_to_conll(tag: str) -> str:
    """Map our 18-type BIO tag to CoNLL 4-type."""
    if tag == "O":
        return "O"
    prefix, etype = tag[:2], tag[2:]
    mapped = KNIV_TO_CONLL.get(etype)
    if mapped:
        return f"{prefix}{mapped}"
    return "O"


def eval_conll2003(model, tokenizer, device, max_length=128):
    """Evaluate NER on CoNLL-2003 test set."""
    print("Loading CoNLL-2003 test set...", flush=True)
    ds = load_dataset("conll2003", split="test")
    print(f"  {len(ds)} examples", flush=True)

    with open(Path(model_dir) / "label_maps.json") as f:
        label_maps = json.load(f)
    ner_labels = label_maps["ner_labels"]

    gold_all = []
    pred_all = []

    for i, example in enumerate(ds):
        tokens = example["tokens"]
        gold_tags_idx = example["ner_tags"]
        gold_tags = [CONLL_LABELS[idx] for idx in gold_tags_idx]

        # Run model
        encoding = tokenizer(
            tokens, is_split_into_words=True,
            max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(
                encoding["input_ids"].to(device),
                encoding["attention_mask"].to(device),
            )

        # Align predictions to words
        logits = outputs["ner_logits"][0].cpu()
        word_ids = encoding.word_ids()

        pred_tags = []
        prev_word = None
        for j, wid in enumerate(word_ids):
            if wid is None or wid == prev_word:
                continue
            if wid >= len(tokens):
                break
            pred_idx = logits[j].argmax().item()
            pred_tag = ner_labels[pred_idx]
            pred_tags.append(map_prediction_to_conll(pred_tag))
            prev_word = wid

        # Ensure same length
        min_len = min(len(gold_tags), len(pred_tags))
        gold_all.append(gold_tags[:min_len])
        pred_all.append(pred_tags[:min_len])

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(ds)}", flush=True)

    f1 = f1_score(gold_all, pred_all)
    print(f"\nCoNLL-2003 Results:", flush=True)
    print(f"  Overall F1: {f1:.4f}", flush=True)
    print(classification_report(gold_all, pred_all), flush=True)
    return f1


def eval_pos(model, tokenizer, device, max_length=128):
    """Evaluate POS on UD EWT test set."""
    import conllu

    ud_test = Path(__file__).parent.parent / "data" / "ud-english-ewt" / "en_ewt-ud-test.conllu"
    if not ud_test.exists():
        print("UD EWT test not found, skipping POS eval", flush=True)
        return

    with open(Path(model_dir) / "label_maps.json") as f:
        label_maps = json.load(f)
    pos_labels = label_maps["pos_labels"]

    with open(ud_test) as f:
        sentences = conllu.parse(f.read())

    correct = 0
    total = 0

    for i, sent in enumerate(sentences):
        words = [t["form"] for t in sent if isinstance(t["id"], int)]
        gold_pos = [t["upos"] for t in sent if isinstance(t["id"], int)]

        encoding = tokenizer(
            words, is_split_into_words=True,
            max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(
                encoding["input_ids"].to(device),
                encoding["attention_mask"].to(device),
            )

        logits = outputs["pos_logits"][0].cpu()
        word_ids = encoding.word_ids()

        pred_pos = []
        prev_word = None
        for j, wid in enumerate(word_ids):
            if wid is None or wid == prev_word:
                continue
            if wid >= len(words):
                break
            pred_idx = logits[j].argmax().item()
            pred_pos.append(pos_labels[pred_idx])
            prev_word = wid

        min_len = min(len(gold_pos), len(pred_pos))
        for g, p in zip(gold_pos[:min_len], pred_pos[:min_len]):
            if g == p:
                correct += 1
            total += 1

    acc = correct / max(total, 1)
    print(f"\nUD EWT POS Results:", flush=True)
    print(f"  Accuracy: {acc:.4f} ({correct}/{total})", flush=True)
    return acc


if __name__ == "__main__":
    model_dir = "/tmp/kniv-teacher-eval"
    encoder = "microsoft/deberta-v3-large"

    print("Loading model...", flush=True)
    model, tokenizer, device = load_model(model_dir, encoder)
    print(f"  Device: {device}", flush=True)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("BENCHMARK EVALUATION", flush=True)
    print("=" * 60, flush=True)

    conll_f1 = eval_conll2003(model, tokenizer, device)

    print("\n" + "-" * 60, flush=True)
    pos_acc = eval_pos(model, tokenizer, device)

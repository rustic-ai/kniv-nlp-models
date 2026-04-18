"""Generate soft labels from trained teacher model for knowledge distillation.

Runs the trained teacher on all training data and saves the raw logits
(pre-softmax) for each task head. These become the distillation targets
for the student model.

Usage:
    python models/deberta-v3-large-nlp-en/generate_soft_labels.py \
        --model-dir outputs/deberta-v3-large-nlp-en/best

Output:
    outputs/deberta-v3-large-nlp-en/soft_labels/
        ner_logits.pt    — [N, seq_len, num_ner_labels]
        pos_logits.pt    — [N, seq_len, num_pos_labels]
        dep_logits.pt    — [N, seq_len, num_dep_labels]
        cls_logits.pt    — [N, num_cls_labels]
        metadata.json    — label maps, counts, model info
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model import MultiTaskNLPModel


CONFIG_PATH = Path(__file__).parent / "config.yaml"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "prepared" / "deberta-v3-large-nlp-en"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class SimpleTokenDataset(torch.utils.data.Dataset):
    """Tokenize examples for inference only (no labels needed)."""

    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        words = example["words"]
        encoding = self.tokenizer(
            words, is_split_into_words=True,
            max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


class SimpleSeqDataset(torch.utils.data.Dataset):
    """Tokenize text for sequence classification inference."""

    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx].get("text", " ".join(self.examples[idx]["words"]))
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def generate(model_dir: str):
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_name = config["model"]["encoder"]
    max_length = config["model"]["max_length"]

    print(f"Device: {device}", flush=True)
    print(f"Loading teacher from {model_dir}...", flush=True)

    model = MultiTaskNLPModel.load(model_dir, encoder_name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load label vocabs
    with open(DATA_DIR / "label_vocabs.json") as f:
        vocabs = json.load(f)

    # Load training data (we generate soft labels on the TRAINING set)
    with open(DATA_DIR / "conll_train.json") as f:
        conll_train = json.load(f)
    with open(DATA_DIR / "ud_train.json") as f:
        ud_train = json.load(f)

    output_dir = Path(model_dir).parent / "soft_labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 32

    # ── Token-level soft labels (NER from CoNLL, POS+Dep from UD) ──

    # NER logits from CoNLL training data
    print(f"Generating NER soft labels ({len(conll_train)} examples)...", flush=True)
    ner_loader = DataLoader(SimpleTokenDataset(conll_train, tokenizer, max_length), batch_size=batch_size)
    ner_logits_all = []
    for i, batch in enumerate(ner_loader):
        with torch.no_grad():
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        ner_logits_all.append(out["ner_logits"].cpu())
        if (i + 1) % 100 == 0:
            print(f"  NER: {(i+1)*batch_size}/{len(conll_train)}", flush=True)
    torch.save(torch.cat(ner_logits_all), output_dir / "ner_logits.pt")
    print(f"  Saved ner_logits.pt: {torch.cat(ner_logits_all).shape}", flush=True)

    # POS and Dep logits from UD training data
    print(f"Generating POS+Dep soft labels ({len(ud_train)} examples)...", flush=True)
    ud_loader = DataLoader(SimpleTokenDataset(ud_train, tokenizer, max_length), batch_size=batch_size)
    pos_logits_all = []
    dep_logits_all = []
    for i, batch in enumerate(ud_loader):
        with torch.no_grad():
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        pos_logits_all.append(out["pos_logits"].cpu())
        dep_logits_all.append(out["dep_logits"].cpu())
        if (i + 1) % 100 == 0:
            print(f"  POS+Dep: {(i+1)*batch_size}/{len(ud_train)}", flush=True)
    torch.save(torch.cat(pos_logits_all), output_dir / "pos_logits.pt")
    torch.save(torch.cat(dep_logits_all), output_dir / "dep_logits.pt")
    print(f"  Saved pos_logits.pt: {torch.cat(pos_logits_all).shape}", flush=True)
    print(f"  Saved dep_logits.pt: {torch.cat(dep_logits_all).shape}", flush=True)

    # CLS logits from combined data
    cls_examples = [ex for ex in ud_train + conll_train if "cls_label" in ex]
    print(f"Generating CLS soft labels ({len(cls_examples)} examples)...", flush=True)
    cls_loader = DataLoader(SimpleSeqDataset(cls_examples, tokenizer, max_length), batch_size=batch_size)
    cls_logits_all = []
    for i, batch in enumerate(cls_loader):
        with torch.no_grad():
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        cls_logits_all.append(out["cls_logits"].cpu())
    torch.save(torch.cat(cls_logits_all), output_dir / "cls_logits.pt")
    print(f"  Saved cls_logits.pt: {torch.cat(cls_logits_all).shape}", flush=True)

    # Save metadata
    metadata = {
        "teacher_model": encoder_name,
        "teacher_dir": model_dir,
        "max_length": max_length,
        "vocabs": vocabs,
        "counts": {
            "ner_examples": len(conll_train),
            "ud_examples": len(ud_train),
            "cls_examples": len(cls_examples),
        },
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSoft labels saved to {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Generate soft labels from teacher model")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to trained teacher model (e.g., outputs/deberta-v3-large-nlp-en/best)")
    args = parser.parse_args()
    generate(args.model_dir)


if __name__ == "__main__":
    main()

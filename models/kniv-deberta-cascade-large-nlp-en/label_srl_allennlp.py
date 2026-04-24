"""Silver SRL labeling using AllenNLP BERT SRL model (86 F1, proper multi-word spans).

Runs via uv with Python 3.9 (AllenNLP dependency). Loads model weights
manually without importing AllenNLP directly.

Usage (must run with uv):
    CUDA_VISIBLE_DEVICES="" uv run --python 3.9 --no-project \
        --with "transformer-srl" --with "numpy<1.24" \
        python label_srl_allennlp.py \
        --corpus-dir corpus/output/annotated \
        --output data/prepared/kniv-deberta-cascade/srl_allennlp_silver.json \
        --max-sentences 200000
"""

from __future__ import annotations
import argparse, json, os, urllib.request, tarfile
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
from tqdm import tqdm

MODEL_URL = "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
ARCHIVE_PATH = "/tmp/srl_bert.tar.gz"
EXTRACT_DIR = "/tmp/srl_bert_extracted"

# Our 42-label tag set
SRL_TAGS_42 = {
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
}


def map_label(label: str) -> str:
    if label == "O":
        return "O"
    if label in ("B-V", "I-V"):
        return "V"
    if label in SRL_TAGS_42:
        return label
    # C-/R- prefix: B-C-ARG0 → B-ARG0, I-R-ARGM-LOC → I-ARGM-LOC
    for prefix in ("B-C-", "I-C-", "B-R-", "I-R-"):
        if label.startswith(prefix):
            mapped = f"{label[0]}-{label[len(prefix):]}"
            if mapped in SRL_TAGS_42:
                return mapped
    normalized = label.replace("ARG5", "ARG4").replace("ARG1-DSP", "ARG1")
    if normalized in SRL_TAGS_42:
        return normalized
    return "O"


class SRLBert(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.tag_projection_layer = nn.Linear(768, num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask):
        h = self.bert_model(input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask).last_hidden_state
        return self.tag_projection_layer(h)


def load_model():
    if not os.path.exists(ARCHIVE_PATH):
        print("Downloading AllenNLP SRL model...", flush=True)
        urllib.request.urlretrieve(MODEL_URL, ARCHIVE_PATH)
    if not os.path.exists(EXTRACT_DIR):
        with tarfile.open(ARCHIVE_PATH) as tar:
            tar.extractall(EXTRACT_DIR)

    with open(f"{EXTRACT_DIR}/vocabulary/labels.txt") as f:
        labels = [l.strip() for l in f if l.strip()]

    model = SRLBert(len(labels))
    state = torch.load(f"{EXTRACT_DIR}/weights.th", map_location="cpu")
    model.load_state_dict(state, strict=False)  # ignore position_ids buffer mismatch
    model.eval()
    return model, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default="corpus/output/annotated")
    parser.add_argument("--output", default="data/prepared/kniv-deberta-cascade/srl_allennlp_silver.json")
    parser.add_argument("--max-sentences", type=int, default=200000)
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    model, labels = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    print(f"Model loaded on {device}. Labels: {len(labels)}", flush=True)

    # Load corpus sentences
    sentences = []
    corpus_path = Path(args.corpus_dir)
    for domain_dir in sorted(corpus_path.iterdir()):
        jsonl = domain_dir / "annotated.jsonl"
        if not jsonl.exists():
            continue
        count = 0
        with open(jsonl) as f:
            for line in f:
                if len(sentences) >= args.max_sentences:
                    break
                obj = json.loads(line)
                tokens = obj["tokens"]
                words = [t["form"] for t in tokens]
                pos_tags = [t["upos"] for t in tokens]
                if len(words) < 4 or len(words) > 80:
                    continue
                verb_indices = [i for i, p in enumerate(pos_tags) if p == "VERB"]
                if not verb_indices:
                    continue
                sentences.append({"words": words, "pos_tags": pos_tags,
                                  "text": obj.get("text", " ".join(words)),
                                  "domain": domain_dir.name, "verb_indices": verb_indices})
                count += 1
        print(f"  {domain_dir.name}: {count:,}", flush=True)
        if len(sentences) >= args.max_sentences:
            break
    print(f"Total sentences: {len(sentences):,}", flush=True)

    # Build all (sentence, verb) pairs for batching
    pairs = []
    for sent in sentences:
        for vi in sent["verb_indices"][:3]:
            pairs.append({"words": sent["words"], "text": sent["text"],
                          "domain": sent["domain"], "verb_idx": vi})
    print(f"Total pairs: {len(pairs):,} ({len(pairs)/len(sentences):.1f} verbs/sent)", flush=True)

    # Batch inference
    results = []
    skipped_no_args = 0
    skipped_low_conf = 0
    batch_size = 64

    for batch_start in tqdm(range(0, len(pairs), batch_size), desc="Labeling"):
        batch = pairs[batch_start:batch_start + batch_size]

        # Tokenize all sentences in batch
        batch_words = [p["words"] for p in batch]
        encs = tokenizer(batch_words, is_split_into_words=True, return_tensors="pt",
                         padding=True, truncation=True, max_length=args.max_length)

        # Build token_type_ids for each item (mark the verb)
        batch_tti = []
        batch_word_ids = []
        for i, p in enumerate(batch):
            wids = encs.word_ids(i)
            batch_word_ids.append(wids)
            tti = [1 if (wid is not None and wid == p["verb_idx"]) else 0 for wid in wids]
            batch_tti.append(tti)
        encs["token_type_ids"] = torch.tensor(batch_tti)

        # Forward pass — full batch on GPU
        with torch.no_grad():
            logits = model(encs["input_ids"].to(device),
                           encs["token_type_ids"].to(device),
                           encs["attention_mask"].to(device))
            probs = torch.softmax(logits, dim=-1).cpu()
            pred_ids = logits.argmax(dim=-1).cpu()

        # Process each item
        for i, p in enumerate(batch):
            words = p["words"]
            vi = p["verb_idx"]
            word_ids = batch_word_ids[i]

            srl_tags, tag_confs = [], []
            prev_wid = None
            for k, wid in enumerate(word_ids):
                if wid is not None and wid != prev_wid:
                    srl_tags.append(labels[pred_ids[i, k]])
                    tag_confs.append(probs[i, k].max().item())
                prev_wid = wid

            srl_tags = srl_tags[:len(words)]
            tag_confs = tag_confs[:len(words)]
            while len(srl_tags) < len(words):
                srl_tags.append("O")
                tag_confs.append(1.0)

            mapped = [map_label(t) for t in srl_tags]
            if vi < len(mapped):
                mapped[vi] = "V"

            has_args = any(t not in ("O", "V") for t in mapped)
            if not has_args:
                skipped_no_args += 1
                continue

            arg_confs = [c for t, c in zip(mapped, tag_confs) if t not in ("O", "V")]
            mean_conf = sum(arg_confs) / len(arg_confs) if arg_confs else 0
            if mean_conf < args.confidence:
                skipped_low_conf += 1
                continue

            results.append({
                "words": words,
                "text": p["text"],
                "srl_tags": mapped,
                "predicate_idx": vi,
            })

    print(f"\nResults: {len(results):,} high-confidence examples", flush=True)
    print(f"  Skipped (no args): {skipped_no_args:,}", flush=True)
    print(f"  Skipped (low conf): {skipped_low_conf:,}", flush=True)

    # Stats
    span_lens = []
    for r in results:
        cur = 0
        for t in r["srl_tags"]:
            if t.startswith("B-"):
                if cur > 0: span_lens.append(cur)
                cur = 1
            elif t.startswith("I-"):
                cur += 1
            else:
                if cur > 0: span_lens.append(cur)
                cur = 0
        if cur > 0: span_lens.append(cur)

    multi = sum(1 for l in span_lens if l > 1)
    print(f"\nSpan stats: {len(span_lens):,} total spans", flush=True)
    print(f"  Mean length: {sum(span_lens)/len(span_lens):.2f}", flush=True)
    print(f"  Multi-word: {multi:,} ({100*multi/len(span_lens):.1f}%)", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"\nSaved to {args.output}", flush=True)


if __name__ == "__main__":
    main()

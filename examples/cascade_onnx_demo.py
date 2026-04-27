# /// script
# requires-python = ">=3.11"
# dependencies = ["onnxruntime>=1.17", "transformers==5.6.2", "numpy"]
# ///
"""kniv-deberta-nlp-base-en-large — ONNX Inference Demo

Single ONNX model, one call, all 5 heads.

Usage:
    uv run python examples/cascade_onnx_demo.py
    uv run python examples/cascade_onnx_demo.py --onnx models/kniv-deberta-nlp-base-en-large/onnx/cascade.onnx
    uv run python examples/cascade_onnx_demo.py --text "Steve Jobs founded Apple in 1976."
"""
from __future__ import annotations
import argparse, re

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# ── Labels ────────────────────────────────────────────────────

POS_LABELS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]
NER_LABELS = [
    "O", "B-PERSON", "I-PERSON", "B-NORP", "I-NORP", "B-FAC", "I-FAC",
    "B-ORG", "I-ORG", "B-GPE", "I-GPE", "B-LOC", "I-LOC",
    "B-PRODUCT", "I-PRODUCT", "B-EVENT", "I-EVENT",
    "B-WORK_OF_ART", "I-WORK_OF_ART", "B-LAW", "I-LAW",
    "B-LANGUAGE", "I-LANGUAGE", "B-DATE", "I-DATE",
    "B-TIME", "I-TIME", "B-PERCENT", "I-PERCENT",
    "B-MONEY", "I-MONEY", "B-QUANTITY", "I-QUANTITY",
    "B-ORDINAL", "I-ORDINAL", "B-CARDINAL", "I-CARDINAL",
]
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
CLS_LABELS = ["inform", "request", "question", "confirm", "reject", "offer", "social", "status"]
DEPREL_LIST = [
    "root", "acl", "acl:relcl", "advcl", "advcl:relcl", "advmod", "amod", "appos",
    "aux", "aux:pass", "case", "cc", "cc:preconj", "ccomp", "compound", "compound:prt",
    "conj", "cop", "csubj", "csubj:outer", "csubj:pass", "dep", "det", "det:predet",
    "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "list", "mark",
    "nmod", "nmod:desc", "nmod:npmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubj:outer",
    "nsubj:pass", "nummod", "obj", "obl", "obl:agent", "obl:npmod", "obl:tmod",
    "orphan", "parataxis", "punct", "reparandum", "vocative", "xcomp",
]


def tokenize_words(text):
    return re.findall(r"\w+(?:'\w+)*|[^\w\s]", text)


def extract_bio_spans(tags, words):
    spans = []
    current_type, current_tokens = None, []
    for word, tag in zip(words, tags):
        if tag.startswith("B-"):
            if current_type:
                spans.append((current_type, " ".join(current_tokens)))
            current_type = tag[2:]
            current_tokens = [word]
        elif tag.startswith("I-") and current_type:
            current_tokens.append(word)
        else:
            if current_type:
                spans.append((current_type, " ".join(current_tokens)))
            current_type, current_tokens = None, []
    if current_type:
        spans.append((current_type, " ".join(current_tokens)))
    return spans


def predict(text: str, session: ort.InferenceSession, tokenizer, predicate_word_idx: int = 0):
    """Run all 5 heads via ONNX. Single call."""
    words = tokenize_words(text)

    enc = tokenizer(words, is_split_into_words=True, return_tensors="np",
                    padding=True, truncation=True, max_length=128)
    word_ids = tokenizer(words, is_split_into_words=True, return_tensors="pt",
                         padding=True, truncation=True, max_length=128).word_ids()

    # Word ↔ token maps
    word_to_token, token_to_word = {}, {}
    prev = None
    for k, wid in enumerate(word_ids):
        if wid is not None and wid != prev:
            word_to_token[wid] = k
            token_to_word[k] = wid
        prev = wid
    valid_indices = sorted(word_to_token.values())

    # Find predicate token index
    pred_tidx = word_to_token.get(predicate_word_idx, 0)

    # Single ONNX call — all heads
    pos_logits, ner_logits, arc_scores, label_scores, srl_logits, cls_logits = session.run(None, {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "predicate_idx": np.array([pred_tidx], dtype=np.int64),
    })

    # Decode POS
    pos_tags = [POS_LABELS[pos_logits[0, tidx].argmax()]
                for _, tidx in sorted(word_to_token.items())]

    # Decode NER (argmax — Viterbi not implemented in numpy for simplicity)
    ner_tags = [NER_LABELS[ner_logits[0, tidx].argmax()]
                for tidx in valid_indices]

    # Decode SRL
    srl_tags = [SRL_TAGS[srl_logits[0, tidx].argmax()]
                for tidx in valid_indices]

    # Decode DEP
    dep_results = []
    for wid, tidx in sorted(word_to_token.items()):
        head_tidx = arc_scores[0, tidx].argmax()
        head_wid = token_to_word.get(int(head_tidx), wid)
        label_idx = label_scores[0, tidx, int(head_tidx)].argmax()
        rel = DEPREL_LIST[label_idx] if label_idx < len(DEPREL_LIST) else "dep"
        dep_results.append({"word_idx": wid, "head_idx": head_wid, "relation": rel})

    # Decode CLS
    cls_label = CLS_LABELS[cls_logits[0].argmax()]

    return {"words": words, "pos": pos_tags, "ner": ner_tags, "srl": srl_tags,
            "dep": dep_results, "cls": cls_label}


def display(r: dict):
    words = r["words"]
    print(f"\n{'=' * 70}")
    print(f"  {' '.join(words)}")
    print(f"{'=' * 70}")

    print(f"\n  POS:")
    for w, p in zip(words, r["pos"]):
        print(f"    {w:15s} {p}")

    entities = extract_bio_spans(r["ner"], words)
    print(f"\n  NER:")
    for t, text in entities:
        print(f"    {text:30s} [{t}]")
    if not entities:
        print(f"    (none)")

    print(f"\n  DEP:")
    for d in r["dep"]:
        w = words[d["word_idx"]]
        h = words[d["head_idx"]] if d["word_idx"] != d["head_idx"] else "ROOT"
        print(f"    {w:15s} ──{d['relation']}──> {h}")

    srl_spans = extract_bio_spans(r["srl"], words)
    print(f"\n  SRL:")
    for role, text in srl_spans:
        if role != "V":
            print(f"    {role:12s}  {text}")
    if not srl_spans:
        print(f"    (no predicate set)")

    print(f"\n  CLS: {r['cls']}")
    print()


SAMPLES = [
    "Barack Obama visited Paris last Friday to meet with French officials.",
    "Can you book me a flight from San Francisco to Tokyo next Tuesday?",
    "Apple was founded by Steve Jobs and Steve Wozniak in a garage in 1976.",
]

def main():
    parser = argparse.ArgumentParser(description="kniv-cascade ONNX Demo")
    parser.add_argument("--onnx", default="models/kniv-deberta-nlp-base-en-large/onnx/cascade.onnx")
    parser.add_argument("--text", default=None)
    parser.add_argument("--predicate", type=int, default=None,
                        help="Word index of SRL predicate (auto-detects first verb if omitted)")
    args = parser.parse_args()

    print(f"Loading ONNX model from {args.onnx}...")
    session = ort.InferenceSession(args.onnx)
    tokenizer = AutoTokenizer.from_pretrained("dragonscale-ai/kniv-deberta-nlp-base-en-large")
    print(f"Loaded. Providers: {session.get_providers()}")

    texts = [args.text] if args.text else SAMPLES

    for text in texts:
        # Auto-detect predicate: find first verb-like word
        words = tokenize_words(text)
        pred_idx = args.predicate
        if pred_idx is None:
            # Quick heuristic: common verbs. For proper detection, run POS first.
            pred_idx = 0
            for i, w in enumerate(words):
                if w.lower() in ("is", "was", "were", "are", "has", "had", "have",
                                  "do", "did", "does", "can", "could", "will", "would",
                                  "shall", "should", "may", "might", "must"):
                    continue  # skip auxiliaries
                if len(w) > 2 and w[0].islower():
                    # Heuristic: first lowercase multi-char word after a noun is likely a verb
                    if i > 0:
                        pred_idx = i
                        break

        r = predict(text, session, tokenizer, predicate_word_idx=pred_idx)
        display(r)

if __name__ == "__main__":
    main()

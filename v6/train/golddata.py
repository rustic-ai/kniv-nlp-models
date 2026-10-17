"""Human-annotated (and v5 gold-filtered) linguistic data for joint training.

    uv run python -m v6.train.golddata --encoder deberta-v3-large   # featurise + report

The corpus supervises the linguistic heads with our own annotators' labels,
and the encoder fits those better than it transfers to human gold (EVALUATION.md,
"Against v5"). v5 trained on gold and led by 1-3 points. These sources put
gold into the shared training mix (stage B3, ``loop --gold-data``) and give
``best.pt`` a gold dev set (``loop --gold-dev``):

``ud``
    UD English EWT train (CC BY-SA 4.0): POS, morphology, lemma, dependencies;
    each sentence also carries its PropBank EWT frames (CC BY-SA 4.0, every
    predicate, ``gold.propbank_full``) where PropBank annotates it. Sentences
    PropBank does not annotate leave SRL unsupervised (IGNORE), never
    "no predicate". The root-predicate SRL test sentences are excluded.
``v5ner``
    ``corpus/gold/train.parquet`` of ``dragonscale-ai/kniv-corpus-en``
    (CC BY-SA 4.0): v5's corpus sentences whose every entity Qwen3-8B
    confirmed (237,646). NER only -- its POS and parses are spaCy silver.

LitBank is not used: the LitBank benchmark scores all 100 documents, so there
is no held-out part to train on. OntoNotes (LDC) is never trained on.

Items are corpus-format features (``data.encode_row``), one sentence each, so
the token-budget batcher packs many per batch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from collections import Counter

import numpy as np

from ..config import DATA_DIR, PREPARED_DIR
from .data import FEATURES, USED_CACHES, encode_row

V5_DATA = "dragonscale-ai/kniv-corpus-en"
SOURCES = ("ud", "v5ner")
JUNK_NER = ("None", "NONE", "null", "_MISSING")


def _row(wid: str, tokens: list[str], **layers) -> dict:
    n = len(tokens)
    row = {"window_id": wid, "domain": "gold", "tokens": tokens, "loss_mask": None,
           "sentence_spans": [[0, n]], "pos": None, "morph": None, "lemma": None,
           "ner": None, "dep": None, "dep_heads": None, "srl_frames": None,
           "cls": None, "sentiment": None, "coref": None, "relations": None}
    row.update(layers)
    return row


def _srl_test() -> set:
    return {tuple(e["words"]) for e in json.loads((PREPARED_DIR / "srl_test.json").read_text())
            if e.get("words")}


def ud_rows(split: str) -> list[dict]:
    """UD EWT sentences with their PropBank frames where annotated."""
    from ..gold.ud_ewt import load_ud_items
    pb = DATA_DIR / "prepared" / "propbank-full" / f"{split}.json"
    frames = {}
    if pb.exists():
        for r in json.loads(pb.read_text()):
            if r["frames"]:
                frames[tuple(r["words"])] = [{"predicate_idx": f["predicate_idx"],
                                               "tags": f["srl_tags"]} for f in r["frames"]]
    test = _srl_test() if split != "test" else set()
    rows = []
    for it in load_ud_items(split, max_tokens=250):
        if tuple(it.tokens) in test:
            continue
        g = it.layers
        rows.append(_row(f"ud-{split}-{it.id}", it.tokens, pos=g["pos"], morph=g["morph"],
                         lemma=g["lemma"], dep=g["dep"]["rels"],
                         dep_heads=[h - 1 for h in g["dep"]["heads"]],
                         srl_frames=frames.get(tuple(it.tokens))))
    return rows


def _bio(tags: list[str]) -> list[str]:
    """v5's gold corpus has ~0.4% tags without B-/I- (validation fixes) and a
    few null types: repair the first to BIO, the second to O."""
    out = []
    for t in tags:
        if t.split("-", 1)[-1] in JUNK_NER:
            t = "O"
        elif t != "O" and t[:2] not in ("B-", "I-"):
            t = ("I-" if out and out[-1] != "O" and out[-1][2:] == t else "B-") + t
        out.append(t)
    return out


def v5ner_rows(split: str) -> list[dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(V5_DATA, f"corpus/gold/{split}.parquet", repo_type="dataset")
    df = pd.read_parquet(p, columns=["sent_id", "tokens", "ner_tags"])
    return [_row(f"v5ner-{split}-{sid}", list(tok), ner=_bio(list(tags)))
            for sid, tok, tags in zip(df["sent_id"], df["tokens"], df["ner_tags"])
            if len(tok)]


def _source_digest(src: str, split: str) -> str:
    if src == "ud":
        # Fixed releases, keyed by name: a VM given the cache has no UD files
        # to hash (bump on a new UD or PropBank release).
        return "ud-ewt-2.14+propbank-full"
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(V5_DATA, files_metadata=False)
    return info.sha


def gold_features(src: str, split: str, tok, spec, vocab, phase2: bool = False,
                  limit: int | None = None, seed: int = 0) -> list[dict]:
    """Encoded items of one source, cached in the corpus feature directory
    (and listed in ``USED_CACHES``, so ``--hub-data`` ships them)."""
    if src not in SOURCES:
        raise SystemExit(f"unknown gold source {src!r}; one of {', '.join(SOURCES)}")
    h = hashlib.sha256(json.dumps([src, split, spec.hf_id, spec.max_subwords, limit, seed,
                                   vocab.to_json()["labels"], phase2, "gold-v1"]).encode())
    h.update(_source_digest(src, split).encode())
    path = FEATURES / f"{spec.name}-gold-{src}-{split}-{h.hexdigest()[:16]}.pkl"
    USED_CACHES.append(path)
    if path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    rows = ud_rows(split) if src == "ud" else v5ner_rows(split)
    if limit and limit < len(rows):
        rng = np.random.default_rng(seed)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), limit, replace=False))]
    p2 = {"entities": {}} if phase2 else None
    stats = Counter()
    out = [encode_row(r, tok, spec.max_subwords, vocab, stats, phase2=p2) for r in rows]
    print(f"gold {src} {split}: {len(out):,} items in {time.time() - t0:.0f}s; "
          + ", ".join(f"{k} {v:,}" for k, v in sorted(stats.items())), flush=True)
    FEATURES.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return out


def gold_dev(tok, spec, vocab, phase2: bool = False, ner_limit: int = 3000) -> list[dict]:
    """The gold dev set for best.pt selection: UD EWT dev (with PropBank dev
    frames) and a fixed sample of v5's gold-filtered dev sentences."""
    return (gold_features("ud", "dev", tok, spec, vocab, phase2)
            + gold_features("v5ner", "dev", tok, spec, vocab, phase2, limit=ner_limit))


def main() -> int:
    from transformers import AutoTokenizer

    from . import encoders
    from .vocab import Vocab
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--phase2", action="store_true")
    args = ap.parse_args()
    spec = encoders.get(args.encoder)
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    vocab = Vocab.load()
    for src in SOURCES:
        f = gold_features(src, "train", tok, spec, vocab, args.phase2)
        n = Counter()
        for x in f:
            for k in ("pos", "lemma", "ner", "dep_head"):
                n[k] += int((np.asarray(x[k]) != -100).sum())
            n["srl_frames"] += len(x["frames"])
        print(f"{src} train: {len(f):,} items; labelled words " + ", ".join(f"{k} {v:,}" for k, v in n.items()))
    d = gold_dev(tok, spec, vocab, args.phase2)
    print(f"gold dev: {len(d):,} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

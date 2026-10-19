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
import subprocess
import time
from collections import Counter

import numpy as np

from ..config import DATA_DIR, PREPARED_DIR
from .data import FEATURES, USED_CACHES, encode_row

V5_DATA = "dragonscale-ai/kniv-corpus-en"
SOURCES = ("ud", "v5ner", "gum")
# GUM genres whose text licence allows commercial training (CC BY / BY-SA,
# data/gum/repo/LICENSE.md); fiction, essays, letters, podcasts, wikiHow are
# non-commercial and reddit is not distributed; genres the licence file does
# not list (conversation, speech, textbook, vlog) are left out unverified.
GUM_REV = "22fdf87f9c71c96bcc771461d06e689b1f90020d"         # data/gum/repo HEAD the windows were built from
GUM_GENRES = ("academic", "bio", "court", "interview", "news", "voyage")
# LitBank-like entity mentions: people, places (incl. GPE, facility),
# organisations. GUM also marks abstract, event, time, object ... mentions,
# the generic noun phrases the corpus-trained coreference over-links.
GUM_TYPES = ("person", "place", "organization")
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


def _gum_entities(value: str):
    """CorefUD ``Entity=`` value -> (opened [(eid, type)], single-token
    [(eid, type)], closed [eid])."""
    opens, singles, closes = [], [], []
    i = 0
    while i < len(value):
        if value[i] == "(":
            j = i + 1
            while j < len(value) and value[j] not in "()":
                j += 1
            parts = value[i + 1:j].split("-")
            ent = (parts[0], parts[1] if len(parts) > 1 else "")
            if j < len(value) and value[j] == ")":
                singles.append(ent); i = j + 1
            else:
                opens.append(ent); i = j
        elif value[i].isdigit():
            j = i
            while j < len(value) and value[j].isdigit():
                j += 1
            if j < len(value) and value[j] == ")":
                closes.append(value[i:j])
            i = j + 1
        else:
            i += 1
    return opens, singles, closes


def gum_docs(split: str) -> list[dict]:
    """GUM documents of the licence-cleared genres: tokens, sentence spans,
    0-based document heads (root -1), and person/place/organization chains
    of 2+ mentions (inclusive spans). ``split``: train, or dev (2 documents
    per genre, by name order)."""
    root = DATA_DIR / "gum" / "repo" / "dep"
    if not root.exists():
        raise SystemExit(f"{root} missing: sparse-clone github.com/amir-zeldes/gum (dep/)")
    rev = subprocess.run(["git", "-C", str(root.parent), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    if (root.parent / ".git").exists() and rev != GUM_REV:     # a VM gets the files, no checkout
        raise SystemExit(f"GUM checkout {rev[:10]} is not the pinned {GUM_REV[:10]}")
    files = sorted(f for f in root.glob("GUM_*.conllu") if f.name.split("_")[1] in GUM_GENRES)
    dev = {f.name for g in GUM_GENRES for f in sorted(x for x in files if x.name.split("_")[1] == g)[:2]}
    files = [f for f in files if (f.name in dev) == (split == "dev")]
    docs = []
    for f in files:
        tokens, heads, spans, ents = [], [], [], {}
        stack: dict[str, list[int]] = {}
        sent_start = 0
        base = 0
        for line in f.read_text(encoding="utf-8").splitlines() + [""]:
            if not line.strip():
                if len(tokens) > sent_start:
                    spans.append([sent_start, len(tokens)])
                    sent_start = len(tokens)
                    base = len(tokens)
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if "-" in cols[0] or "." in cols[0]:
                continue
            i = len(tokens)
            tokens.append(cols[1])
            h = int(cols[6])
            heads.append(-1 if h == 0 else base + h - 1)
            misc = dict(kv.split("=", 1) for kv in cols[9].split("|") if "=" in kv)
            if "Entity" in misc:
                opens, singles, closes = _gum_entities(misc["Entity"])
                for eid, typ in opens:
                    stack.setdefault(eid, []).append(i)
                    ents.setdefault(eid, [typ, []])
                for eid, typ in singles:
                    ents.setdefault(eid, [typ, []])[1].append((i, i))
                for eid in closes:
                    if stack.get(eid):
                        ents[eid][1].append((stack[eid].pop(), i))
        chains = [sorted(m) for t, m in ents.values() if t in GUM_TYPES and len(m) >= 2]
        docs.append({"name": f.stem, "tokens": tokens, "heads": heads, "spans": spans,
                     "chains": chains})
    return docs


def gum_rows(split: str, max_words: int = 350) -> list[dict]:
    """GUM documents as windows of whole sentences (at most ``max_words``),
    each with the chains that keep 2+ mentions inside it."""
    rows = []
    for d in gum_docs(split):
        win = []
        for span in d["spans"] + [None]:
            s, e = span if span is not None else (None, None)
            if s is not None and (not win or e - win[0][0] <= max_words):
                win.append((s, e))
                continue
            if win:
                a, b = win[0][0], win[-1][1]
                chains = [[[ms - a, me - a] for ms, me in c if a <= ms and me < b] for c in d["chains"]]
                chains = [c for c in chains if len(c) >= 2]
                rows.append(_row(f"gum-{split}-{d['name']}-{a}", d["tokens"][a:b],
                                 dep_heads=[h - a if a <= h < b else -1 for h in d["heads"][a:b]],
                                 coref=chains))
                rows[-1]["sentence_spans"] = [[x - a, y - a] for x, y in win]
            win = [(s, e)] if s is not None else []
    return rows


def _source_digest(src: str, split: str) -> str:
    if src == "gum":
        # Pinned, so a VM given the cache (and no GUM checkout) finds it.
        return "gum-" + GUM_REV
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
    rows = (ud_rows(split) if src == "ud" else gum_rows(split) if src == "gum"
            else v5ner_rows(split))
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

"""Relation distillation from the two v7 models together (``ensemble_probe``:
their averaged relation margins beat either on Re-DocRED dev by +2.7, and
two heads on one encoder do not).

    uv run python -m v6.train.kd_teacher prep                      # unseen DocRED-distant docs
    uv run python -m v6.train.kd_teacher dump --run R [--upload]   # one teacher's margins (A100: colab post --jobs kd)
    uv run python -m v6.train.kd_teacher merge --runs A,B          # the averaged teacher, for the bundle

Teacher text: Re-DocRED train (both teachers were trained on it, so their
margins there mostly echo the gold) and 8,000 DocRED-distant documents past
the 40,000 the models trained on (hash order of ``gold.docred_distant``;
entities given, labels unused) -- text neither teacher was fitted to.

Per document and teacher: which entities it reached (an entity past its
subword limit is dropped) and every (head, tail, relation) margin above
``FLOOR`` over Re-DocRED's 96 relations; the rest is "at most ``FLOOR``".
``merge`` averages the two teachers on the pairs both reached, and the
training loop (``--distil``) pulls the student's margins toward it
(``model.distil_loss``).
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from ..config import DATA_DIR

CORPUS = DATA_DIR / "v6-corpus"
DISTANT = CORPUS / "kd_distant.json"
TEACHER = CORPUS / "kd_teacher.pkl"
OUT = DATA_DIR / "v6-eval" / "kd"
FLOOR = -6.0           # sigmoid(-6) = 0.0025: below it a margin says "no"
N_DISTANT = 8000
SKIP_DISTANT = 40000   # the subset kdata --prep trained on


def prep(args) -> int:
    from ..gold import docred_distant
    docs = docred_distant.load(None)[SKIP_DISTANT:SKIP_DISTANT + args.n]
    DISTANT.write_text(json.dumps({"train": [
        {"id": d["id"], "tokens": d["tokens"], "sentence_spans": d["sentence_spans"],
         "entities": d["entities"]} for d in docs]}))
    print(f"wrote {DISTANT}: {len(docs)} documents", flush=True)
    return 0


def sources() -> dict[str, list[dict]]:
    """Each teacher source as ``{"tokens", "sentence_spans", "mentions": per
    entity [(start, end_exclusive)]}``, in file order."""
    from .data import REDOCRED_REL
    rd = [{"tokens": d["tokens"], "sentence_spans": d["sentence_spans"],
           "mentions": [[(a, b + 1) for a, b in e["mentions"]] for e in d["entities"]]}
          for d in json.loads(REDOCRED_REL.read_text())["train"]]
    dd = [{"tokens": d["tokens"], "sentence_spans": d["sentence_spans"],
           "mentions": [[tuple(m) for m in e["mentions"]] for e in d["entities"]]}
          for d in json.loads(DISTANT.read_text())["train"]]
    return {"redocred": rd, "distant": dd}


@torch.no_grad()
def dump(args) -> int:
    from .gold_eval import HF_REPO, _pairs, load_run, phase2_window, run_phase2
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    model, tok, spec, vocab, _ = load_run(args.run, HF_REPO, device, args.local)
    res = {"run": args.run, "floor": FLOOR}
    for name, docs in sources().items():
        docs = docs[:args.limit] if args.limit else docs
        t0 = time.time()
        out = []
        for k in range(0, len(docs), 64):           # bounded memory: run_phase2 keeps every output
            chunk = docs[k:k + 64]
            feats = [phase2_window(d["tokens"], [tuple(s) for s in d["sentence_spans"]], tok, spec,
                                   vocab, d["mentions"], markers=model.rel_markers) for d in chunk]
            for f, o in zip(feats, run_phase2(model, feats, tok, device, batch=1)):
                keep = [int(x) for x in f["_ent_keep"]]
                m = o["relation"][:, 1:] - o["relation"][:, :1]                 # [P, 96]
                p, r = np.nonzero(m > FLOOR)
                pairs = np.asarray(_pairs(len(keep)), dtype=np.int32).reshape(-1, 2)
                ht = np.asarray(keep, dtype=np.int32)[pairs[p]] if len(p) else np.zeros((0, 2), np.int32)
                out.append({"keep": keep, "h": ht[:, 0].astype(np.int16), "t": ht[:, 1].astype(np.int16),
                            "r": r.astype(np.int8), "m": m[p, r].astype(np.float16)})
            if (k // 64) % 10 == 0:
                print(f"  {name}: {k + len(chunk):,}/{len(docs):,} [{time.time() - t0:.0f}s]", flush=True)
        res[name] = out
        print(f"{args.run} {name}: {len(out):,} documents, "
              f"{sum(len(x['m']) for x in out):,} margins above {FLOOR} [{time.time() - t0:.0f}s]",
              flush=True)
    path = OUT / f"{args.run}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(res))
    print(f"wrote {path} ({path.stat().st_size >> 20} MB)", flush=True)
    if args.upload:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=f"{args.run}/kd-margins.pkl",
                            repo_id=HF_REPO, commit_message=f"{args.run}: teacher relation margins")
        print(f"uploaded {args.run}/kd-margins.pkl", flush=True)
    return 0


def _dense(x: dict) -> dict:
    return {(int(h), int(t), int(r)): float(m) for h, t, r, m in zip(x["h"], x["t"], x["r"], x["m"])}


def merge(args) -> int:
    """The averaged teacher. Per document: the entities both teachers
    reached, and the averaged margin of every (head, tail, relation) above
    ``FLOOR`` -- a margin one teacher had at or below the floor counts as
    the floor."""
    tea = []
    for run in args.runs.split(","):
        path = OUT / f"{run}.pkl"
        if not path.exists():
            from huggingface_hub import hf_hub_download
            from .gold_eval import HF_REPO
            path = Path(hf_hub_download(HF_REPO, f"{run}/kd-margins.pkl"))
        tea.append(pickle.loads(path.read_bytes()))
    out = {"runs": args.runs.split(","), "floor": FLOOR}
    for name in ("redocred", "distant"):
        docs = []
        for xs in zip(*(t[name] for t in tea)):
            both = sorted(set.intersection(*(set(x["keep"]) for x in xs)))
            ok = set(both)
            ms = [_dense(x) for x in xs]
            keys = sorted({k for m in ms for k in m if k[0] in ok and k[1] in ok})
            avg = np.array([np.mean([m.get(k, FLOOR) for m in ms]) for k in keys], dtype=np.float32)
            sel = avg > FLOOR
            k = np.asarray(keys, dtype=np.int32).reshape(-1, 3)[sel]
            docs.append({"ents": both, "h": k[:, 0].astype(np.int16), "t": k[:, 1].astype(np.int16),
                         "r": k[:, 2].astype(np.int8), "m": avg[sel].astype(np.float16)})
        out[name] = docs
        print(f"{name}: {len(docs):,} documents, {sum(len(d['m']) for d in docs):,} margins; "
              f"entities reached by both {sum(len(d['ents']) for d in docs):,}", flush=True)
    TEACHER.write_bytes(pickle.dumps(out))
    print(f"wrote {TEACHER} ({TEACHER.stat().st_size >> 20} MB)", flush=True)
    return 0


# ── training side ────────────────────────────────────────────────────
def attach(feats: list[dict], docs: list[dict]) -> int:
    """Give each feature (``rel["doc"]`` its teacher document, ``rel["ids"]``
    the original entity index of each kept entity) its teacher margins in the
    collate's pair order: ``rel["kd"]`` = (pair indices, relation, margin)
    above the floor and ``rel["kd_pairs"]`` = the pairs the teacher scored
    (both teachers reached both entities). Returns how many features got one."""
    n = 0
    for f in feats:
        r = f.get("rel")
        if r is None or "ids" not in r or r.get("doc") is None or r["doc"] >= len(docs):
            continue
        d = docs[r["doc"]]
        ids = list(r["ids"])
        E = len(ids)
        pos = {}
        for h in range(E):
            for t in range(E):
                if h != t:
                    pos[(ids[h], ids[t])] = len(pos)
        ok = set(d["ents"])
        scored = np.array([ids[h] in ok and ids[t] in ok for h in range(E) for t in range(E) if h != t],
                          dtype=bool)
        p = [pos.get((int(h), int(t)), -1) for h, t in zip(d["h"], d["t"])]
        sel = np.asarray([x >= 0 for x in p], dtype=bool)
        r["kd"] = (np.asarray(p, dtype=np.int32)[sel] if len(p) else np.zeros(0, np.int32),
                   d["r"][sel].astype(np.int64), d["m"][sel])
        r["kd_pairs"] = scored
        n += 1
    return n


def distant_features(tok, max_subwords: int, n_cls: int, markers: bool = True) -> list[dict]:
    """The teacher's DocRED-distant documents as training items: entities and
    the marker pass, no hard labels (every pair masked: partial, nothing
    kept), so only the distillation term reads them. A document with fewer
    than two entities in the window is left out."""
    from .data import blank_window, marker_features
    out = []
    for di, d in enumerate(json.loads(DISTANT.read_text())["train"]):
        spans = [tuple(s) for s in d["sentence_spans"]]
        f = blank_window(d["tokens"], spans, tok, max_subwords, n_cls)
        W = f["_W"]
        ids, ents = [], []
        for i, e in enumerate(d["entities"]):
            ms = [(a, b) for a, b in e["mentions"] if b <= W and f["word_ok"][a]]
            if ms:
                ids.append(i)
                ents.append(np.asarray(ms, dtype=np.int32))
        if len(ents) < 2:
            continue
        f.update(coref=None, _n_rel=96,
                 rel={"spans": ents, "types": np.zeros(len(ents), dtype=np.int32), "triples": [],
                      "evidence": [], "schema": list(range(96)), "partial": True,
                      "unlabelled_keep": 0.0, "doc": di, "ids": ids})
        if markers:
            f["rm"] = marker_features(d["tokens"], spans, ents, tok)
        out.append(f)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--n", type=int, default=N_DISTANT)
    d = sub.add_parser("dump")
    d.add_argument("--run", required=True)
    d.add_argument("--local", type=Path)
    d.add_argument("--limit", type=int)
    d.add_argument("--upload", action="store_true")
    d.add_argument("--device")
    m = sub.add_parser("merge")
    m.add_argument("--runs", required=True)
    args = ap.parse_args()
    return {"prep": prep, "dump": dump, "merge": merge}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

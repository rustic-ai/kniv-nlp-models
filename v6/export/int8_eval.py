"""What INT8 costs: the runtime with the fp32 and with the INT8 encoder
graphs, on the same dev items, scored against gold.

    uv run --extra export python -m v6.export.int8_eval --release release/ultrabert-380m

- UD English EWT dev (300 sentences): POS accuracy, LAS.
- Re-DocRED dev (100 documents, gold entities): relation micro-F1 over the
  96 relations, the release's calibrated offsets.
- CrossNER politics test (300 sentences; the scorecard calibrates on it):
  exact-match span F1 from the span head.

Gate (docs/v7-release-plan.md): INT8 within 1 point of fp32 on each.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def ud(k, items):
    pos = las = n = 0
    for it in items:
        w = k.window({"tokens": it.tokens, "sentence_spans": [[0, len(it.tokens)]]}, [], None, True, 4)
        g = it.layers
        for x, p, h, r in zip(w["words"], g["pos"], g["dep"]["heads"], g["dep"]["rels"]):
            pos += x["pos"] == p
            las += x["head"] == h and x["deprel"] == r
            n += 1
    return {"pos": pos / n, "las": las / n}


def redocred(k, items, rels):
    tp = npred = ngold = 0
    for it in items:
        sp = [tuple(x) for x in it.layers["sentence_spans"]]
        ents = [{"type": e["type"], "mentions": [(a, b + 1) for a, b in e["mentions"]]}
                for e in it.entities]
        nodes, facts = k.relations({"sentence_speakers": []}, it.tokens, sp, ents, rels, k.top)
        # Nodes are the entities that reached the window, in order: map back.
        ment = {tuple(m): i for i, e in enumerate(ents) for m in e["mentions"]}
        gid = [ment[tuple(n["spans"][0])] for n in nodes]
        pred = {(gid[f["head"]], gid[f["tail"]], f["relation"]) for f in facts}
        gold = {tuple(x) for x in it.layers["rel"]}
        tp += len(pred & gold); npred += len(pred); ngold += len(gold)
    return {"rel_f1": 2 * tp / max(npred + ngold, 1)}


def crossner(k, items, types):
    from kniv_v7.runtime import align, decode_spans, span_candidates, type_text
    tp = npred = ngold = 0
    temb = k.label_emb([type_text({"name": t, "description": ""}) for t in types])
    for it in items:
        toks = it["tokens"]
        ids, first, W = align(toks, k.tok, k.meta["max_subwords"])
        first = np.asarray(first)
        sent = np.zeros(W, dtype=np.int64)
        x = {"input_ids": np.asarray([ids]), "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
             "first_subword": np.maximum(first, 0)[None], "word_end": __import__(
                 "kniv_v7.runtime", fromlist=["word_ends"]).word_ends(first, len(ids))[None],
             "word_mask": (first >= 0)[None], "sent_of_word": sent[None], "sent_ids": np.arange(1)}
        o = dict(zip(k.meta["graphs"]["plain"]["outputs"], k.run("plain", **x)))
        cands = span_candidates(sent, first >= 0)
        pred = set()
        if cands:
            lg = k.run("span", span_words=o["span_words"], start=np.asarray([s for s, _ in cands]),
                       end=np.asarray([e for _, e in cands]), type_emb=temb)[0]
            pred = {(s, e, types[t]) for s, e, t in decode_spans(lg, cands, k.span_offset)}
        gold = {(a, b, t) for t, a, b in it["spans"]}
        tp += len(pred & gold); npred += len(pred); ngold += len(gold)
    return {"span_f1": 2 * tp / max(npred + ngold, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--docs", type=int, default=100)
    args = ap.parse_args()
    sys.path.insert(0, str(args.release))
    from kniv_v7 import Kniv
    from ..gold.redocred import load_rel_items
    from ..gold.ud_ewt import load_ud_items
    from ..label_bank import load as load_bank
    from ..train.span_eval import load_crossner
    ud_items = load_ud_items("dev", limit=args.n)
    rd_items = load_rel_items("dev", limit=args.docs)
    rels = [r for r in load_bank()["relations"] if r["source"] == "redocred"]
    cn = load_crossner("politics")[:args.n]
    types = sorted({t for it in cn for t, _, _ in it["spans"]})
    res = {}
    for int8 in (False, True):
        k = Kniv(args.release, int8=int8)
        r = {**ud(k, ud_items), **redocred(k, rd_items, rels), **crossner(k, cn, types)}
        res["int8" if int8 else "fp32"] = {m: round(v, 4) for m, v in r.items()}
        print(("int8" if int8 else "fp32"), res["int8" if int8 else "fp32"], flush=True)
    drop = {m: round(res["fp32"][m] - res["int8"][m], 4) for m in res["fp32"]}
    res["drop"] = drop
    res["pass"] = all(d <= 0.01 for d in drop.values())
    print(json.dumps(res, indent=1))
    (args.release / "onnx" / "int8_eval.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

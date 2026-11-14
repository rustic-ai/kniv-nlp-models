"""Schema-free facts (v8 plan r4, review step 5): the verifier as a filter
on the rule decoder's open facts (K4).

    uv run python -m v6.eval.o_filter [--ver RUN-ver1] [--device mps]

Each judged K4 fact (500 M1 texts, c3) goes to the verifier as a candidate
``subject | predicate | object`` (the predicate as the relation name, no
description), its arguments every occurrence of their mention texts. Prints
the verifier's AUC on judged expressed / not, and judged precision with true
facts per text at a few thresholds, against the decoder's own 0.704 / 2.14.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

K4 = Path("v6/runs/k-checks/k4-labels-m1p3a-n500.json")
ITEMS = Path("v6/runs/m1-precision/preds-m1p3a-n500.pkl")
POS = ("supported", "negated", "hedged")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ver", default="ultrabert-380m-v8p3-d123-ver1")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    from ..train.verifier import _mention_entities, collate, encode, load_verifier
    from ..label_bank import load as load_bank
    dev = torch.device(a.device)
    m, tok = load_verifier(a.ver, dev)[:2]
    bank = {r["name"]: r for r in load_bank()["relations"]}
    items = {it["id"]: it for it in pickle.loads(ITEMS.read_bytes())["items"]}
    L = [t for t in json.loads(K4.read_text()) if t["judged"]]
    P, Y, T = [], [], []
    with torch.no_grad():
        for n, t in enumerate(L):
            it = items[t["id"]]
            ent, ents, names = _mention_entities(it["tokens"])
            cands = []
            for f in t["facts"]:
                if f.get("verdict") is None:
                    continue
                h, tl = ent(f["names"][0] or [f["s"]]), ent(f["names"][1] or [f["o"]])
                y = f["verdict"] in POS
                cands.append((h, tl, f["p"], y))
            rec = {"tokens": it["tokens"], "sentence_spans": it["sentence_spans"], "speakers": it.get("speakers"),
                   "participants": [], "entities": [[tuple(m_) for m_ in ms] for ms in ents], "names": names}
            cache, exs, ys = {}, [], []
            for h, tl, p, y in cands:
                e = None
                if h is not None and tl is not None and h != tl:
                    e = encode(rec, {"h": h, "t": tl, "rel": p, "y": 0}, tok, bank, None, None, cache=cache)
                if e is None:                 # not readable by the verifier: counted as dropped
                    P.append(-1.0); Y.append(y); T.append(n)
                    continue
                exs.append(e); ys.append(y)
            for i in range(0, len(exs), 16):
                lg = m(collate(exs[i:i + 16], tok.pad_token_id, dev))
                pr = torch.softmax(lg.float(), -1)
                P += (1 - pr[:, 1]).cpu().tolist()
            Y += ys; T += [n] * len(ys)
            if dev.type == "mps":
                torch.mps.empty_cache()
    P, Y = np.array(P), np.array(Y, dtype=bool)
    ok = P >= 0
    o = P[ok].argsort().argsort() + 1
    n1, n0 = Y[ok].sum(), (~Y[ok]).sum()
    auc = (o[Y[ok]].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)
    texts = len(L)
    print(f"{len(P)} judged open facts on {texts} texts ({(~ok).sum()} not readable); "
          f"decoder alone: precision {Y.mean():.3f}, {Y.sum() / texts:.2f} true facts per text")
    print(f"verifier AUC on expressed / not (readable facts): {auc:.3f}")
    print("| threshold | kept | precision | true facts per text |\n|---|---|---|---|")
    for th in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        k = P >= th
        print(f"| {th:.2f} | {k.sum()} | {Y[k].mean():.3f} | {Y[k].sum() / texts:.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

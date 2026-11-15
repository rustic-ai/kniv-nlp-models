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
    ap.add_argument("--judge", default="astra", choices=["astra", "deepseek", "both"],
                    help="labels: GPT-6, DeepSeek (k4-labels-NAME-deepseek.json), or both agreeing")
    ap.add_argument("--split", action="store_true",
                    help="held-out reading: threshold fitted on half the texts (most true minus false facts), "
                         "scored on the other half against the decoder alone, paired bootstrap over texts")
    a = ap.parse_args()
    from ..train.verifier import _mention_entities, collate, encode, load_verifier
    from ..label_bank import load as load_bank
    dev = torch.device(a.device)
    m, tok = load_verifier(a.ver, dev)[:2]
    bank = {r["name"]: r for r in load_bank()["relations"]}
    items = {it["id"]: it for it in pickle.loads(ITEMS.read_bytes())["items"]}
    LA = json.loads(K4.read_text())
    if a.judge == "astra":
        L = [t for t in LA if t["judged"]]
    else:
        LD = {t["id"]: t for t in json.loads(K4.with_name(K4.stem + "-deepseek.json").read_text())}
        L = []
        for t in LA:
            d = LD.get(t["id"])
            if not (t["judged"] and d and d["judged"]):
                continue
            fs = []
            for f, g in zip(t["facts"], d["facts"]):
                v = g.get("verdict") if a.judge == "deepseek" else (
                    None if f.get("verdict") is None or g.get("verdict") is None else
                    f["verdict"] if f["verdict"] in POS and g["verdict"] in POS else "not_supported")
                fs.append({**f, "verdict": v})
            L.append({**t, "facts": fs})
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
    if a.split:
        return _split(P, Y, np.array(T), L)
    print("| threshold | kept | precision | true facts per text |\n|---|---|---|---|")
    for th in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        k = P >= th
        print(f"| {th:.2f} | {k.sum()} | {Y[k].mean():.3f} | {Y[k].sum() / texts:.2f} |")
    return 0


def _split(P, Y, T, L) -> int:
    import hashlib
    half = np.array([int(hashlib.md5(t["id"].encode()).hexdigest(), 16) % 2 for t in L])
    fit, held = half[T] == 0, half[T] == 1
    grid = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    net = {th: (Y[fit & (P >= th)].sum() - (~Y[fit & (P >= th)]).sum()) for th in grid}
    th = max(grid, key=lambda x: net[x])
    print(f"fit half: {(half == 0).sum()} texts; threshold {th} (most true minus false facts); held half: {(half == 1).sum()} texts")
    hs = np.where(half == 1)[0]
    rng = np.random.default_rng(0)

    def stats(ix, t):
        m = np.isin(T, ix)
        k = m & (P >= t) if t is not None else m
        return Y[k].sum(), k.sum(), len(ix)
    print("| on the held half | precision | true facts per text | precision against the decoder alone |\n|---|---|---|---|")
    for name, t in [("decoder alone", None), (f"filter at {th} (fitted)", th), ("filter at 0.01", 0.01),
                    ("filter at 0.05", 0.05), ("filter at 0.2", 0.2)]:
        tp, n, nt = stats(hs, t)
        d = []
        for _ in range(1000):
            b = rng.choice(hs, len(hs))
            w = np.bincount(b, minlength=len(L))[T]
            k = (P >= t) if t is not None else np.ones_like(Y)
            pf = (w * Y * k).sum() / max((w * k).sum(), 1)
            p0 = (w * Y).sum() / max(w.sum(), 1)
            d.append(pf - p0)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"| {name} | {tp / max(n, 1):.3f} | {tp / nt:.2f} | {100 * (tp / max(n, 1) - stats(hs, None)[0] / stats(hs, None)[1]):+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Schema-free facts: the library decoder (``kniv_v7.open_facts``) against
K4's judged facts on the 500 M1 texts (plan r4, decision 11).

    uv run python -m v6.eval.o_decode check      # decoder on the saved layers against K4's facts
    uv run python -m v6.eval.o_decode filter --run RUN --ver RUN-ver3f --device mps

K4's decoder was a scratch script; its facts and judged labels are kept
(``k-checks/k4-open-*.json``, ``k4-labels-*.json``). ``check`` rebuilds the
analysed windows from the saved layers of the released xlarge-u
(``k-checks/layers-m1p3a-n500.pkl``) and reports how many of K4's facts the
library decoder returns (subject, predicate, object as text, case ignored)
and how many it adds, by source, so the judged labels carry over to the
facts it shares with K4.
"""
from __future__ import annotations

import argparse
import ast
import json
import pickle
from collections import Counter
from pathlib import Path

from kniv_v7.open_facts import decode

K = Path("v6/runs/k-checks")
LAYERS = K / "layers-m1p3a-n500.pkl"
K4 = K / "k4-open-m1p3a-n500.json"
ITEMS = Path("v6/runs/m1-precision/preds-m1p3a-n500.pkl")


def _lit(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def analysed() -> list[dict]:
    """The 500 M1 windows as ``Kniv.analyse`` returns them (linguistic part)."""
    items = {it["id"]: it for it in pickle.loads(ITEMS.read_bytes())["items"]}
    out = []
    for x in pickle.loads(LAYERS.read_bytes()):
        toks = _lit(items[x["id"]]["tokens"])
        dep, pos, lem = _lit(x["words"]), _lit(x["pos"]), _lit(x["lemma"])
        W = x["W"]
        words = [{"pos": pos[i], "lemma": lem[i], "head": dep[i][0], "deprel": dep[i][1]} for i in range(W)]
        out.append({"id": x["id"], "tokens": toks, "words": words, "srl": _lit(x["srl"]),
                    "coref": _lit(x["coref"])})
    return out


def key(s, p, o) -> tuple:
    return (s.lower(), p.lower(), o.lower())


def check(a) -> int:
    k4 = json.loads(K4.read_text())
    win = analysed()
    shared, missed, added = Counter(), Counter(), Counter()
    show_m, show_a = [], []
    for w, ref in zip(win, k4):
        got = {key(f["subject"]["text"], f["predicate"]["text"], f["object"]["text"]): f for f in decode(w)}
        want = {key(f["s"], f["p"], f["o"]): f for f in ref}
        for k_, f in want.items():
            (shared if k_ in got else missed)[f["src"]] += 1
            if k_ not in got and len(show_m) < a.show:
                show_m.append((w["id"], f["src"], k_))
        for k_, f in got.items():
            if k_ not in want:
                added[f["source"]] += 1
                if len(show_a) < a.show:
                    show_a.append((w["id"], f["source"], k_))
    n = sum(shared.values()) + sum(missed.values())
    print(f"K4 facts {n}: returned {sum(shared.values())} ({sum(shared.values()) / max(n, 1):.1%}), "
          f"missed {sum(missed.values())}; added {sum(added.values())}")
    print("| source | K4 | returned | added |\n|---|---|---|---|")
    for s in sorted(set(shared) | set(missed) | set(added)):
        print(f"| {s} | {shared[s] + missed[s]} | {shared[s]} | {added[s]} |")
    for title, rows in (("missed", show_m), ("added", show_a)):
        print(f"\n{title}:")
        for r in rows:
            print("  ", r)
    return 0


POS = ("supported", "negated", "hedged")


def labels(judge: str) -> dict:
    """``{text id: {(s, p, o): expressed}}`` for K4's judged facts: GPT-6,
    DeepSeek, or both agreeing (a fact only one judge read is left out)."""
    A = {t["id"]: t for t in json.loads((K / "k4-labels-m1p3a-n500.json").read_text()) if t["judged"]}
    D = {t["id"]: t for t in json.loads((K / "k4-labels-m1p3a-n500-deepseek.json").read_text()) if t["judged"]}
    out = {}
    for i, t in A.items():
        if judge != "astra" and i not in D:
            continue
        m = {}
        for j, f in enumerate(t["facts"]):
            va = f.get("verdict")
            vd = D[i]["facts"][j].get("verdict") if i in D else None
            if judge == "astra":
                y = None if va is None else va in POS
            elif judge == "deepseek":
                y = None if vd is None else vd in POS
            else:
                y = None if va is None or vd is None else (va in POS and vd in POS)
            if y is not None:
                m[key(f["s"], f["p"], f["o"])] = y
        out[i] = m
    return out


def filt(a) -> int:
    """The library decoder's facts on the 500 M1 texts through a verifier,
    read on K4's judged facts (shared with K4 by text); facts K4 never had are
    counted and bounded (all false / all true), never judged here."""
    import hashlib
    import pickle as pk
    import numpy as np
    from ..infer import Extractor
    cache = K / f"o-decode-{a.ver}.pkl"
    win = analysed()
    items = {it["id"]: it for it in pickle.loads(ITEMS.read_bytes())["items"]}
    if cache.exists():
        scored = pk.loads(cache.read_bytes())
    else:
        ex = Extractor(a.run, device=a.device)
        ex.use_verifier(a.ver)
        scored = {}
        for n, w in enumerate(win):
            it = items[w["id"]]
            rec = {"tokens": _lit(it["tokens"]), "sentence_spans": _lit(it["sentence_spans"]),
                   "sentence_speakers": _lit(it.get("speakers")) if it.get("speakers") else None}
            fs = ex.verify_open(rec, decode(w))
            scored[w["id"]] = [(key(f["subject"]["text"], f["predicate"]["text"], f["object"]["text"]),
                                f["verifier"], f["source"]) for f in fs]
            if n % 100 == 0:
                print(f"verified {n}/{len(win)}", flush=True)
        cache.write_bytes(pk.dumps(scored))
    print(f"verifier {a.ver}: {sum(len(v) for v in scored.values())} decoded facts on {len(scored)} texts; "
          f"{sum(p is None for v in scored.values() for _, p, _ in v)} not readable")
    for judge in ("astra", "deepseek", "both"):
        L = labels(judge)
        ids = sorted(L)
        P, Y, T, U = [], [], [], []           # U: unjudged facts per text above each threshold (filled below)
        un = {i: [] for i in ids}
        for n, i in enumerate(ids):
            for k_, p, _ in scored[i]:
                if k_ in L[i]:
                    P.append(-1.0 if p is None else p); Y.append(L[i][k_]); T.append(n)
                else:
                    un[i].append(-1.0 if p is None else p)
        P, Y, T = np.array(P), np.array(Y, dtype=bool), np.array(T)
        ok = P >= 0
        o = P[ok].argsort().argsort() + 1
        n1, n0 = Y[ok].sum(), (~Y[ok]).sum()
        auc = (o[Y[ok]].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)
        half = np.array([int(hashlib.md5(i.encode()).hexdigest(), 16) % 2 for i in ids])
        fit, held = half[T] == 0, half[T] == 1
        grid = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        th = max(grid, key=lambda x: Y[fit & (P >= x)].sum() - (~Y[fit & (P >= x)]).sum())
        hs = np.where(half == 1)[0]
        nt = len(hs)
        name = {"astra": "GPT-6", "deepseek": "DeepSeek", "both": "agreement"}[judge]
        print(f"\n{name}: {len(ids)} texts, {len(P)} judged facts shared with K4 (decoder alone {Y.mean():.3f}), "
              f"AUC {auc:.3f}; threshold fitted on half: {th}")
        print("| held half | judged kept | precision | true facts per text | unjudged kept per text (bounds on precision) |\n|---|---|---|---|---|")
        for lab, t in (("decoder alone", None), (f"filter at {th} (fitted)", th), ("filter at 0.05", 0.05),
                       ("filter at 0.2", 0.2), ("filter at 0.6", 0.6)):
            k_ = held & ((P >= t) if t is not None else np.ones_like(Y))
            tp, n = Y[k_].sum(), k_.sum()
            u = sum(sum(1 for p in un[ids[j]] if t is None or p >= t) for j in hs)
            lo, hi = tp / max(n + u, 1), (tp + u) / max(n + u, 1)
            print(f"| {lab} | {n} | {tp / max(n, 1):.3f} | {tp / nt:.2f} | {u / nt:.2f} ({lo:.3f}–{hi:.3f}) |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["check", "filter"])
    ap.add_argument("--show", type=int, default=25)
    ap.add_argument("--run", default="ultrabert-380m-v8p3-d123")
    ap.add_argument("--ver", default="ultrabert-380m-v8p3-d123-ver3f")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    return check(a) if a.cmd == "check" else filt(a)


if __name__ == "__main__":
    raise SystemExit(main())

"""M3 hand set, run against run (v8 plan r4): each statement's expected facts
returned with the right polarity and modality, paired over statement groups.

    uv run python -m v6.eval.m3_runs --base RUN[:HEAD] --other RUN[:HEAD] ...

``RUN`` names a ``m3_negation predict --run`` file set
(``preds-run-RUN-hand-*.pkl``; ``RUN-ver`` for one predicted with the
verifier); ``HEAD`` the readout inside it (default the run itself; with the
verifier ``RUN+ver``: the verdict on facts it decides, ``RUN+verall``: on
every fact it read).
"""
from __future__ import annotations

import argparse
import pickle
from collections import defaultdict

import numpy as np

from . import m3_negation as m3

KINDS = [("negated", lambda k: k == "negated"),
         ("hedged", lambda k: k not in ("negated", "asserted")),
         ("asserted", lambda k: k == "asserted")]


def rows_of(file_run: str, head: str) -> dict:
    items = {it["id"]: it for it in m3.load_hand()}
    out = {}
    for mode in ("gold", "e2e"):
        R = {}
        for p in sorted(m3.OUT.glob(f"preds-run-{file_run}-hand-*.pkl")):
            for r in pickle.loads(p.read_bytes())["rows"].get(mode, []):
                R[r["id"]] = r
        its = [items[i] for i in R if i in items]
        saved = m3.HEADS
        m3.HEADS = {head: head}
        try:
            out[mode] = m3.fact_rows(its, [R[it["id"]] for it in its], mode)
        finally:
            m3.HEADS = saved
        for r in out[mode]:
            r["_h"] = r[head]
    return out


def right(rs):
    return float(np.mean([r["returned"] and r["_h"]["pol"] and r["_h"]["mod"] for r in rs])) if rs else float("nan")


def paired(a, b, n=2000, seed=13):
    key = lambda r: (r["id"], r["relation"], r["kind"])
    A = {key(r): r for r in a}
    B = {key(r): r for r in b}
    ks = [k for k in A if k in B]
    g = defaultdict(list)
    for k in ks:
        g[A[k]["group"]].append(k)
    groups = list(g)
    d0 = right([B[k] for k in ks]) - right([A[k] for k in ks])
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n):
        s = [k for j in rng.choice(len(groups), len(groups)) for k in g[groups[j]]]
        ds.append(right([B[k] for k in s]) - right([A[k] for k in s]))
    return d0, float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--other", nargs="+", required=True)
    ap.add_argument("--loose-det", action="store_true", help="end to end, ignore a leading the / a / an")
    a = ap.parse_args()
    m3.LOOSE_DET = a.loose_det
    spec = lambda s: (s.split(":")[0], s.split(":")[1] if ":" in s else s.split(":")[0].removesuffix("-ver"))
    cfgs = [spec(a.base)] + [spec(x) for x in a.other]
    R = {c: rows_of(*c) for c in cfgs}
    names = [c[1] for c in cfgs]
    print("| subset | " + " | ".join(names) + " | " + " | ".join(f"{n} − {names[0]}" for n in names[1:]) + " |")
    print("|---|" + "---|" * (2 * len(names) - 1))
    for mode in ("gold", "e2e"):
        for kn, sel in KINDS:
            sub = {c: [r for r in R[c][mode] if sel(r["kind"])] for c in cfgs}
            n = len(sub[cfgs[0]])
            vals = [f"{right(sub[c]):.3f}" for c in cfgs]
            diffs = [paired(sub[cfgs[0]], sub[c]) for c in cfgs[1:]]
            print(f"| {kn} ({n}), {'gold entities' if mode == 'gold' else 'end to end'} | " + " | ".join(vals) + " | "
                  + " | ".join(f"{100 * d[0]:+.1f} [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}]" for d in diffs) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

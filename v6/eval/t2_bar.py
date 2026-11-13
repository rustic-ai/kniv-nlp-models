"""T2 (v8 plan r4): refit the relation bar per run off the sets it is read on.

    uv run python -m v6.eval.t2_bar rows --run RUN --offset 500      # one 500-text slice per process
    uv run python -m v6.eval.t2_bar fit --run RUN

Two shifts of the deployed bar, fitted apart: ``in`` for relations with
their own offset in the run's calibration (the bank's 63), ``new`` for the
rest (the new-relation offset). Fitted on short-dev texts 500-2,399 (outside
M1's 500) with gold entities, F1 against the teacher's facts. The teacher
lists about half of the true facts (P0), which pushes the best bar up, so
the fit is also read where the labels are complete or judged:

- M4 (business texts, complete expected facts, user-named relations: all on
  the new path), gold and end to end, from ``m4_business`` rows;
- M1 (judged, short dev 0-499), the run's own end-to-end predictions at the
  shifted bar: judged precision and pooled recall (raising only: M1 holds
  the candidates above the deployed bar).

Pass (plan r4 T2): the M4 gain holds when the bar is fitted off M4.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

OUT = Path("v6/runs/k-checks/t2")
GRID = np.arange(-3.0, 4.01, 0.5)
RUNS = {"e1b": "ultrabert-380m-v8e-w1half", "d123": "ultrabert-380m-v8p3-d123", "v7.1": "ultrabert-380m-v7b8ev"}


def rows(run: str, offset: int, n: int, device: str | None) -> None:
    from . import heldout_zs as hz
    from ..infer import Extractor
    items = hz.load_short_dev(offset + n)[offset:offset + n]
    ex = Extractor(run, device=device)
    dep = ex.unseen_offset
    ex.rel_offsets = {k: v + hz.SHIFT for k, v in ex.rel_offsets.items()}
    ex.unseen_offset = dep - hz.SHIFT
    ex.top = max(ex.top, 4)
    R = hz.predict(ex, items, "gold")
    slim = [{"cand": r["cand"], "gold": [(f["head"], f["relation"], f["tail"]) for f in r["gold"]]} for r in R]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"rows-{run}-{offset}.pkl").write_bytes(pickle.dumps(slim))
    print(f"wrote {len(slim)} rows", flush=True)


def _counts(R, d_in, d_new):
    c = np.zeros((len(R), 3))
    for i, r in enumerate(R):
        p = {x["key"] for x in r["cand"] if x["m"] > (d_new if x["unseen"] else d_in)}
        g = set(r["gold"])
        c[i] = len(p & g), len(p), len(g)
    return c


def _f1(c):
    tp, pr, g = c.sum(0)
    p, r = tp / max(pr, 1), tp / max(g, 1)
    return 2 * p * r / max(p + r, 1e-9)


def m1_read(label: str, run: str, d_in: float, d_new: float):
    """Judged precision and pooled recall of the run's M1 predictions with the
    bar raised by the shifts (margins in the M1 file are relative to the bar)."""
    from huggingface_hub import hf_hub_download
    from .m1_precision import OUT as M1, POSITIVE, _same, labels_path
    cal = json.loads(Path(hf_hub_download("dragonscale-ai/kniv-v6-runs", f"{run}/calibration.json")).read_text())
    known = set(cal.get("rel_offsets") or {})
    P = pickle.loads((M1 / "preds-m1p3b-n500.pkl").read_bytes())
    L = json.loads(labels_path("m1p3b-n500", "astra", "c3").read_text())
    tp = pr = total = 0
    for i, t in enumerate(L["texts"]):
        if not t["judged"]:
            continue
        total += sum(f["verdict"] in POSITIVE for f in t["facts"])
        for k, m in P["margins"][label][i].items():
            if m <= (d_in if k[1] in known else d_new):
                continue
            pr += 1
            f = next((f for f in t["facts"] if label in f["src"]
                      and _same((set(f["key"][0]), f["key"][1], set(f["key"][2])), k)), None)
            tp += f is not None and f["verdict"] in POSITIVE
    p, r = tp / max(pr, 1), tp / max(total, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9), pr


def fit(label: str) -> dict:
    from . import m4_business as m4
    run = RUNS[label]
    R = []
    for f in sorted(OUT.glob(f"rows-{run}-*.pkl")):
        R += pickle.loads(f.read_bytes())
    F = np.array([[_f1(_counts(R, a, b)) for b in GRID] for a in GRID])
    ia, ib = np.unravel_index(F.argmax(), F.shape)
    d_in, d_new = float(GRID[ia]), float(GRID[ib])
    f0 = _f1(_counts(R, 0.0, 0.0))
    print(f"\n## {label} ({run}): {len(R)} short-dev texts, gold entities, teacher facts")
    print(f"  deployed bar F1 {f0:.3f}; fitted in {d_in:+.1f}, new {d_new:+.1f}: F1 {F[ia, ib]:.3f}")
    print("  new-path sweep at in = fitted: " + ", ".join(f"{b:+.1f} {F[ia, j]:.3f}" for j, b in enumerate(GRID)))
    res = {"run": run, "texts": len(R), "in": d_in, "new": d_new, "short_f1": [f0, float(F[ia, ib])]}
    items = m4.load()
    for mode in ("gold", "e2e"):
        p = m4.OUT / f"rows-{run}-user-{mode}.json"
        if not p.exists():
            print(f"  M4 {mode}: no rows (run m4_business score --run {run})")
            continue
        rws = json.loads(p.read_text())
        for r in rws:
            for c in r["cand"]:
                h, rel, t = c["key"]
                c["key"] = (h, rel, t) if mode == "gold" else (frozenset(h), rel, frozenset(t))
        a, b = m4.score_rows(items, rws, mode), m4.score_rows(items, rws, mode, delta=d_new)
        d = m4.bootstrap(b["per_item"], a["per_item"])
        best = max((m4.score_rows(items, rws, mode, delta=x)["prf"][2], x) for x in GRID)
        print(f"  M4 {mode}: deployed F1 {a['prf'][2]:.3f}; at the fitted new bar {b['prf'][2]:.3f} "
              f"({100 * d[0]:+.1f} [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}]); best on M4 {best[0]:.3f} at {best[1]:+.1f}")
        res[f"m4_{mode}"] = [a["prf"][2], b["prf"][2], 100 * d[0], 100 * d[1], 100 * d[2], best[0], best[1]]
    if label in ("e1b", "d123", "v7.1"):
        p0, r0, f0m, n0 = m1_read(label, run, 0.0, 0.0)
        p1, r1, f1m, n1 = m1_read(label, run, max(d_in, 0.0), max(d_new, 0.0))
        print(f"  M1 judged (raising only): deployed P {p0:.3f} R {r0:.3f} F1 {f0m:.3f} ({n0}); "
              f"fitted P {p1:.3f} R {r1:.3f} F1 {f1m:.3f} ({n1})")
        res["m1"] = [[p0, r0, f0m, n0], [p1, r1, f1m, n1]]
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["rows", "fit"])
    ap.add_argument("--run", default="d123", help="rows: a run name; fit: a label (e1b, d123, v7.1) or all")
    ap.add_argument("--offset", type=int, default=500)
    ap.add_argument("-n", type=int, default=500)
    ap.add_argument("--device")
    a = ap.parse_args()
    if a.cmd == "rows":
        rows(RUNS.get(a.run, a.run), a.offset, a.n, a.device)
        return 0
    out = {lb: fit(lb) for lb in (RUNS if a.run == "all" else [a.run])}
    (OUT / "fit.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage 3 guard rail (v8 plan r4): Re-DocRED through the extractor, with and
without the fact verifier deciding the relations it was trained on.

    uv run python -m v6.eval.s3_redocred --run RUN --verifier RUN-ver1 [--limit 100] [--offset 0]; --combine [--thresholds ...]

Gold entities, the 96 Re-DocRED relations with the bank's descriptions,
micro triple F1 against the gold triples; per document the same candidates
(the relation step's margins, run once), decided by the margin and by the
verifier (candidates down to 3 under the bar, two per pair; where a window
and a candidate do not fit the verifier's 1,024 subwords, the margin
decides). Paired bootstrap over documents. Slices (``--offset``) run in
fresh processes and merge with ``--combine``.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

OUT = Path("v6/runs/k-checks/s3-redocred")


def items(limit: int, offset: int) -> list[dict]:
    from ..gold.redocred import load_rel_items
    from ..label_bank import load as load_bank
    rels = [{"name": r["name"], "description": r.get("description", "")}
            for r in load_bank()["relations"] if r["source"] == "redocred"]
    out = []
    for it in load_rel_items("test", limit=offset + limit)[offset:]:
        out.append({"id": it.id, "tokens": it.tokens, "participants": [],
                    "sentence_spans": it.layers["sentence_spans"],
                    "entities": [{"type": e["type"], "spans": [[a, b + 1] for a, b in e["mentions"]]}
                                 for e in it.entities],
                    "facts": [{"head": h, "tail": t, "relation": r} for h, t, r in it.layers["rel"]],
                    "relations": rels, "types": None})
    return out


def run_slice(a) -> None:
    from . import heldout_zs as hz
    from ..infer import Extractor
    its = items(a.limit, a.offset)
    ex = Extractor(a.run, device=a.device)
    dep = ex.unseen_offset
    ex.rel_offsets = {k: v + hz.SHIFT for k, v in ex.rel_offsets.items()}
    ex.unseen_offset = dep - hz.SHIFT
    ex.top = max(ex.top, 4)
    slim = lambda R: [{"cand": r["cand"], "gold": r["gold"]} for r in R]
    res = {}
    ex.ver_on = False
    res["margin"] = slim(hz.predict(ex, its, "gold"))
    ex.use_verifier(a.verifier, threshold=0.5)      # m of a verified candidate: its log-odds
    ex.ver_veto_frozen = False      # every near-bar candidate scored; the veto rule is read offline
    ex.margin_shift = hz.SHIFT
    ex.ver_on = True
    res["verifier"] = slim(hz.predict(ex, its, "gold"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.verifier}-{a.offset}.pkl").write_bytes(pickle.dumps(res))
    print(f"wrote {len(its)} documents", flush=True)


def combine(a) -> None:
    from .heldout_zs import bootstrap, f1
    from .heldout_zs import score_item
    R = {"margin": [], "verifier": []}
    for p in sorted(OUT.glob(f"{a.verifier}-[0-9]*.pkl")):
        r = pickle.loads(p.read_bytes())
        for k in R:
            R[k] += r[k]
    A = np.array([score_item(r, "gold") for r in R["margin"]], float)
    pa = f1(A.sum(0))
    n_ver = sum(c["p"] is not None for r in R["verifier"] for c in r["cand"])
    n_fb = sum(c["p"] is None and c["m"] > 0 for r in R["verifier"] for c in r["cand"])
    print(f"Re-DocRED test, gold entities, {len(A)} documents ({a.verifier}); "
          f"{n_ver} candidates verified, {n_fb} kept by the margin (input too long)")
    print(f"  margin: P {pa[0]:.3f} R {pa[1]:.3f} F1 {pa[2]:.3f}")
    for t in [float(x) for x in a.thresholds.split(",")]:
        lt = float(np.log(t) - np.log1p(-t))
        rows = [{**r, "cand": [{**c, "m": c["m"] - lt} if c["p"] is not None else c for c in r["cand"]]}
                for r in R["verifier"]]
        B = np.array([score_item(r, "gold") for r in rows], float)
        pb, d = f1(B.sum(0)), bootstrap(B, A)
        # the deployed rule on these relations: the margin decides, the verifier vetoes below t
        rows_v = []
        for rm, rv in zip(R["margin"], R["verifier"]):
            pv = {c["key"]: c["m"] for c in rv["cand"] if c["p"] is not None}     # log-odds
            rows_v.append({**rm, "cand": [{**c, "m": c["m"] if pv.get(c["key"], 99) > lt else -1.0}
                                          for c in rm["cand"]]})
        C = np.array([score_item(r, "gold") for r in rows_v], float)
        pc, e = f1(C.sum(0)), bootstrap(C, A)
        print(f"  at {t:.2f}: verifier decides P {pb[0]:.3f} R {pb[1]:.3f} F1 {pb[2]:.3f} "
              f"({100 * d[0]:+.1f} [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}]); margin with veto "
              f"P {pc[0]:.3f} R {pc[1]:.3f} F1 {pc[2]:.3f} ({100 * e[0]:+.1f} [{100 * e[1]:+.1f}, {100 * e[2]:+.1f}])")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="ultrabert-380m-v8p3-d123")
    ap.add_argument("--verifier", default="ultrabert-380m-v8p3-d123-ver1")
    ap.add_argument("--thresholds", default="0.02,0.05,0.1,0.2,0.3,0.5", help="combine: thresholds to read")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--device")
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    combine(a) if a.combine else run_slice(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

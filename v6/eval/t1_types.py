"""T1 (v8 plan r4): argument-type fit without the caller.

    uv run python -m v6.eval.t1_types [--min 10] [--rate 0.02]

Learned from D1's judged candidates on short training texts outside the
held-out fifth (gold entities, the text's own type names; a speaker counts
as "speaker"): for each relation and argument side, how often a candidate
whose argument has type T was expressed. A candidate is dropped when, on
either side, its type was seen at least ``--min`` times with the relation
and expressed at most ``--rate`` of them. Nothing is per caller: types are
the text's entity types as the entity step returns them.

Applied on M1 (short dev 0-499, judged) to d123's end-to-end predictions,
before and after the verifier (V1b), types read from the text's gold
entities by mention (unmatched mentions are kept). Pass (plan r4 T1):
precision up at equal recall, at most 1% of the true facts dropped.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict

import numpy as np


def learn() -> dict:
    from ..config import RUNS_DIR
    from ..train.verifier import _corpus
    from .negation_pilot import heldout_base
    short = json.loads((_corpus() / "kg_short.json").read_text())
    labels = {r["id"]: r for r in json.loads(
        (RUNS_DIR / "kg-judge" / "labels-d1-full-ultrabert-380m-v7b8ev.json").read_text())}
    tally = defaultdict(lambda: np.zeros(2))       # (rel, side, type) -> [expressed, seen]
    for row in short["train"]:
        lab = labels.get(row["id"])
        if not lab or not lab.get("judged") or heldout_base(row["id"]):
            continue
        p = row["payload"]
        parts = list(p.get("participants") or row.get("participants") or [])
        types = ["speaker"] * len(parts) + [e["type"] for e in p["entities"]]
        for f in lab["facts"]:
            v = f.get("verdict")
            if v is None or f["head"] == f["tail"] or max(f["head"], f["tail"]) >= len(types):
                continue
            ok = v in ("supported", "negated", "hedged")
            tally[(f["relation"], 0, types[f["head"]])] += [ok, 1]
            tally[(f["relation"], 1, types[f["tail"]])] += [ok, 1]
    return dict(tally)


def type_map(it: dict) -> dict:
    """Lower-cased mention text -> entity type, from the item's gold entities."""
    m = {w.lower(): "speaker" for w in (it.get("participants") or [])}
    for e in it["entities"]:
        for a, b in e["spans"]:
            m.setdefault(" ".join(it["tokens"][a:b]).lower(), e["type"])
    return m


def arg_type(ms, tm):
    ts = {tm[x] for x in ms if x in tm}
    return next(iter(ts)) if len(ts) == 1 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--min", type=int, default=10)
    ap.add_argument("--rate", type=float, default=0.02)
    ap.add_argument("--name", default="m1p3b-n500-v-ver1b-d123")
    a = ap.parse_args()
    from .m1_precision import OUT, POSITIVE, _same, labels_path
    T = learn()
    bad = {k for k, (e, n) in T.items() if n >= a.min and e / n <= a.rate}
    print(f"learned {len(T):,} (relation, side, type) cells from D1; {len(bad):,} rejected "
          f"(seen {a.min}+ times, expressed {100 * a.rate:.0f}% or less)")
    P = pickle.loads((OUT / f"preds-{a.name}.pkl").read_bytes())
    L = json.loads(labels_path(a.name, "astra", "c3").read_text())
    stats = {}
    for cfg in ("d123", "d123+v"):
        new = []
        drop = np.zeros(3)      # dropped true, dropped false, dropped unjudged
        keep_true = 0
        typed = n = 0
        for i, (it, t) in enumerate(zip(P["items"], L["texts"])):
            tm = type_map(it)
            out = {}
            for k, m in P["margins"][cfg][i].items():
                n += 1
                th, tt = arg_type(k[0], tm), arg_type(k[2], tm)
                typed += th is not None and tt is not None
                rej = (k[1], 0, th) in bad or (k[1], 1, tt) in bad
                f = next((f for f in t["facts"] if cfg in f["src"]
                          and _same((set(f["key"][0]), f["key"][1], set(f["key"][2])), k)), None) \
                    if t["judged"] else None
                ok = f is not None and f["verdict"] in POSITIVE
                if rej:
                    drop += [ok, f is not None and not ok, f is None]
                else:
                    out[k] = m
                    keep_true += ok
            new.append(out)
        P["margins"][cfg + "+t1"] = new
        P["preds"][cfg + "+t1"] = P["preds"][cfg]
        for t in L["texts"]:
            for f in t["facts"]:
                if cfg in f["src"]:
                    f["src"].append(cfg + "+t1")
        tot_true = keep_true + drop[0]
        stats[cfg] = {"predictions": n, "both_typed": typed, "dropped_true": int(drop[0]),
                      "dropped_false": int(drop[1]), "dropped_unjudged": int(drop[2]),
                      "true_dropped_share": float(drop[0] / max(tot_true, 1))}
        print(f"{cfg}: {n} predictions, both arguments typed {typed / max(n, 1):.2f}; dropped "
              f"{int(drop[0])} true ({100 * drop[0] / max(tot_true, 1):.1f}% of its true facts), "
              f"{int(drop[1])} false, {int(drop[2])} unjudged")
    L["configs"] = ["v7.1", "d123", "d123+t1", "d123+v", "d123+v+t1"]
    name = "t1-n500"
    (OUT / f"preds-{name}.pkl").write_bytes(pickle.dumps(P))
    labels_path(name, "astra", "c3").write_text(json.dumps(L))
    print(f"read with: m1_precision equal --run {name} --version c3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""P4 (v8 plan r2): how much relation F1 a user's own relation names cost.

    uv run python -m v6.eval.label_robustness --run ultrabert-380m-v7b8ev

Gold entities, relation step only. Each set is scored three times with the
same texts and entities, changing only the label texts of the paraphrased
relations (``data/v6-eval/p4_paraphrases.json``, hand-written):

- ``orig``: the label bank's names and descriptions;
- ``a``: another name with the definition reworded;
- ``b``: a terse user-style name with no definition.

Sets: the judged short texts (``kg_judge`` labels: every text judged, gold =
facts the judge marks expressed; the 50 most frequent judged relations are
paraphrased, the rest of each schema unchanged) and CoNLL04 test (gold, its 5
relations paraphrased). v7.1 never trained on either.

A renamed relation goes through the new-relation path with the one shared
offset, as a user's name would (a bank name of Re-DocRED keeps its own
offset and the frozen path). The new-relation offset is chosen per variant
on half the short texts (``heldout_zs.half``) and applied to the other half
and to CoNLL04; best-offset F1 is reported too. Scores count only the
paraphrased relations (other columns are unchanged). Paired bootstrap over
texts, ``orig`` against each variant.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..config import DATA_DIR, RUNS_DIR
from . import heldout_zs as hz

OUT = RUNS_DIR / "label-robustness"
PARA = DATA_DIR / "v6-eval" / "p4_paraphrases.json"
GRID = np.round(np.arange(-4.0, 4.01, 0.25), 2)
VARIANTS = ("orig", "a", "b")


def judged_short() -> list[dict]:
    from ..kg_judge import OUT as JOUT, POSITIVE, load_texts
    labels = json.loads((JOUT / "labels-astra-c1.json").read_text())
    texts = {s: {t["id"]: t for t in load_texts(s)} for s in ("train", "dev")}
    items = []
    for r in labels:
        if not r["judged"] and r["facts"]:
            continue
        it = dict(texts[r["split"]][r["id"]])
        it["facts"] = [{"head": f["head"], "relation": f["relation"], "tail": f["tail"]}
                       for f in r["facts"] if f["verdict"] in POSITIVE]
        items.append(it)
    return items


def renamed(items: list[dict], para: dict, v: str) -> tuple[list[dict], dict]:
    if v == "orig":
        return items, {}
    back = {para[n][v]["name"]: n for n in para}
    out = []
    for it in items:
        rels = [dict(para[r["name"]][v]) if r["name"] in para else r for r in it["relations"]]
        out.append({**it, "relations": rels})
    return out, back


def predict(ex, items, para, v) -> list[dict]:
    its, back = renamed(items, para, v)
    rows = hz.predict(ex, its, "gold")
    for row, it in zip(rows, items):
        row["item"] = it
        for c in row["cand"]:
            h, r, t = c["key"]
            c["key"] = (h, back.get(r, r), t)
    return rows


def counts(rows, rels: set, delta: float, part=None) -> np.ndarray:
    out = []
    for row in rows:
        if part is not None and hz.half(row["item"]) != part:
            continue
        g = {(f["head"], f["relation"], f["tail"]) for f in row["gold"] if f["relation"] in rels}
        p = {c["key"] for c in row["cand"] if c["key"][1] in rels
             and c["m"] > (delta if c["unseen"] else 0.0)}
        out.append([len(p & g), len(p), len(g)])
    return np.array(out).reshape(-1, 3)


def per_relation(rows, rels, delta) -> dict:
    c = defaultdict(lambda: np.zeros(3))
    for row in rows:
        g = {(f["head"], f["relation"], f["tail"]) for f in row["gold"] if f["relation"] in rels}
        p = {c_["key"] for c_ in row["cand"] if c_["key"][1] in rels
             and c_["m"] > (delta if c_["unseen"] else 0.0)}
        for k in p | g:
            c[k[1]] += [k in p and k in g, k in p, k in g]
    return {r: (hz.f1(v)[2], int(v[2])) for r, v in c.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="ultrabert-380m-v7b8ev")
    ap.add_argument("--device")
    ap.add_argument("--para", type=Path, default=PARA,
                    help="other paraphrases in the same format (D2: d2_labels p4file); 'orig' rows are shared")
    a = ap.parse_args()
    P = json.loads(a.para.read_text())
    data = {"short": judged_short(), "conll04": hz.load_conll04()}
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if a.para.resolve() == PARA.resolve() else f"-{a.para.stem}"
    cache = OUT / f"rows-{a.run}{tag}.pkl"
    rows = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    base = OUT / f"rows-{a.run}.pkl"
    if tag and base.exists():
        rows.update({k: v for k, v in pickle.loads(base.read_bytes()).items() if k[1] == "orig"})
    todo = [(s, v) for s in data for v in VARIANTS if (s, v) not in rows]
    if todo:
        from ..infer import Extractor
        ex = Extractor(a.run, device=a.device)
        dep = ex.unseen_offset
        ex.rel_offsets = {k: v + hz.SHIFT for k, v in ex.rel_offsets.items()}
        ex.unseen_offset = dep - hz.SHIFT
        ex.top = max(ex.top, 4)
        for s, v in todo:
            t0 = time.time()
            rows[(s, v)] = predict(ex, data[s], P[s], v)
            print(f"{s} {v}: {len(data[s])} texts [{time.time() - t0:.0f}s]", flush=True)
            cache.write_bytes(pickle.dumps(rows))
        del ex
    R = {s: set(P[s]) for s in data}
    rep = {"run": a.run, "texts": {s: len(x) for s, x in data.items()}, "variants": {}}
    print(f"\n{a.run}: F1 over the paraphrased relations "
          f"(short: {len(R['short'])}, CoNLL04: {len(R['conll04'])})\n")
    print("| variant | offset (short half 0) | short half 1 | CoNLL04 | short best | CoNLL04 best |")
    print("|---|---|---|---|---|---|")
    chosen = {}
    for v in VARIANTS:
        d = float(max(GRID, key=lambda x: hz.f1(counts(rows[("short", v)], R["short"], x, 0))[2]))
        chosen[v] = d
        f_s = hz.f1(counts(rows[("short", v)], R["short"], d, 1))[2]
        f_c = hz.f1(counts(rows[("conll04", v)], R["conll04"], d))[2]
        b_s = max(hz.f1(counts(rows[("short", v)], R["short"], x))[2] for x in GRID)
        b_c = max(hz.f1(counts(rows[("conll04", v)], R["conll04"], x))[2] for x in GRID)
        rep["variants"][v] = {"offset": d, "short_half1": f_s, "conll04": f_c,
                              "short_best": b_s, "conll04_best": b_c}
        print(f"| {v} | {d:+.2f} | {f_s:.3f} | {f_c:.3f} | {b_s:.3f} | {b_c:.3f} |")
    print("\nPaired bootstrap, variant minus orig (95% interval over texts):")
    for v in ("a", "b"):
        for s, part in (("short", 1), ("conll04", None)):
            d, lo, hi = hz.bootstrap(counts(rows[(s, v)], R[s], chosen[v], part),
                                     counts(rows[(s, "orig")], R[s], chosen["orig"], part))
            rep["variants"][v][f"diff_{s}"] = [d, lo, hi]
            print(f"  {v} {s:8s} {d * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}] points")
    print("\nPer relation, all short texts at each variant's chosen offset (>= 10 gold facts), "
          "largest drops:")
    pr = {v: per_relation(rows[("short", v)], R["short"], chosen[v]) for v in VARIANTS}
    drops = []
    for r, (fo, n) in pr["orig"].items():
        if n >= 10:
            drops.append((min(pr["a"].get(r, (0, 0))[0], pr["b"].get(r, (0, 0))[0]) - fo, r, fo,
                          pr["a"].get(r, (0, 0))[0], pr["b"].get(r, (0, 0))[0], n))
    drops.sort()
    print("| relation | gold | orig | a | b |\n|---|---|---|---|---|")
    for _, r, fo, fa, fb, n in drops[:10]:
        print(f"| {r} ({P['short'][r]['a']['name']} / {P['short'][r]['b']['name']}) | {n} | "
              f"{fo:.3f} | {fa:.3f} | {fb:.3f} |")
    for s in ("conll04",):
        print(f"\n{s} per relation (offset chosen on short half 0):")
        prc = {v: per_relation(rows[(s, v)], R[s], chosen[v]) for v in VARIANTS}
        for r in P[s]:
            print(f"  {r:12s} " + "  ".join(f"{v} {prc[v].get(r, (0, 0))[0]:.3f}" for v in VARIANTS)
                  + f"  (gold {prc['orig'].get(r, (0, 0))[1]})")
    rep["per_relation_short"] = {v: {r: x[0] for r, x in pr[v].items()} for v in VARIANTS}
    out = OUT / f"{a.run}{tag}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(rep, indent=1))
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

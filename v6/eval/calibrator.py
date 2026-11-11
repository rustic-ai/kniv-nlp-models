"""P3 pilot (v8 plan r2): a calibrator from relation candidates to probabilities.

    uv run python -m v6.eval.calibrator --run ultrabert-380m-v7b8ev

The relation step decides "margin > offset", and the right offset moves with
the text type (E5): -1.5 on teacher windows, about +1.75 on short text, lower
on CoNLL04. A calibrator maps each candidate's margin and its context to a
probability, so that one probability threshold can serve every text type.

Data (gold entities throughout):

- **short text**: the judged candidates of ``kg_judge`` (700 train / 300 dev
  texts). A candidate's label is the judge's verdict (supported, negated,
  hedged = positive). Candidates the judge never saw (below its bar, or
  ranked under the top 2 of their pair) are negatives; judged positives the
  run did not score count as misses in F1.
- **CoNLL04 test**: candidates from one run of the model, gold facts as labels.

Features per candidate (nothing per relation, so they carry to unseen
relations): the margin; whether the relation takes the new-relation offset;
its rank, gap to the pair's best and to the next candidate, and the number of
candidates on the pair; the same relation's rank and gap across the text's
pairs (does this relation fire everywhere?); sentence distance and argument
length, whether an argument is a speaker or contains a digit; the text's
length, entity count and schema size.

Compared: the deployed decision (margin above one global offset chosen on
short train), Platt scaling of the margin, and a small gradient-boosted
calibrator (``HistGradientBoostingClassifier``). Thresholds are chosen on
short-train out-of-fold predictions (5 folds by text) and applied unchanged to
every set; each set's own best threshold is the reference. Held-out
relations: relations hashed into a test fold are removed from training and
scored on dev alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..config import RUNS_DIR
from ..kg_judge import OUT as JUDGE_OUT, POSITIVE, load_texts

OUT = RUNS_DIR / "calibrator"
FLOOR = -8.0          # candidates below this margin are scored 0 by every method and left out
FEATS = ["m", "unseen", "rank", "gap_best", "gap_next", "n_pair", "col_rank", "col_gap",
         "col_n_top", "sent_dist", "len_h", "len_t", "speaker", "digit", "n_words", "n_ents",
         "n_rel"]


def _sent_of(it):
    out = {}
    for j, (s, e) in enumerate(it["sentence_spans"]):
        for i in range(s, e):
            out[i] = j
    return out


def featurise(it: dict, cands: list[dict]) -> tuple[np.ndarray, list]:
    """One row per candidate above ``FLOOR``: the features and its key."""
    k = len(it.get("participants") or [])
    so = _sent_of(it)

    def ent(i):
        if i < k:
            return None
        return it["entities"][i - k]["spans"]

    cs = [c for c in cands if c["m"] > FLOOR]
    bypair, bycol = defaultdict(list), defaultdict(list)
    for c in cs:
        bypair[(c["key"][0], c["key"][2])].append(c["m"])
        bycol[c["key"][1]].append(c["m"])
    for v in list(bypair.values()) + list(bycol.values()):
        v.sort(reverse=True)
    n_words = len(it["tokens"])
    rows, keys = [], []
    for c in cs:
        h, r, t = c["key"]
        pm, cm = bypair[(h, t)], bycol[r]
        rank = pm.index(c["m"])
        nxt = pm[rank + 1] if rank + 1 < len(pm) else FLOOR
        crank = cm.index(c["m"])
        eh, et = ent(h), ent(t)
        if eh and et:
            sd = min(abs(so.get(a, 0) - so.get(b, 0)) for a, _ in eh for b, _ in et)
            lh, lt = min(b - a for a, b in eh), min(b - a for a, b in et)
            dg = any(any(ch.isdigit() for ch in " ".join(it["tokens"][a:b])) for a, b in eh + et)
        else:
            sd, lh, lt, dg = 0, (1 if eh is None else min(b - a for a, b in eh)), \
                (1 if et is None else min(b - a for a, b in et)), False
        rows.append([c["m"], float(c["unseen"]), rank, pm[0] - c["m"], c["m"] - nxt, len(pm),
                     crank, cm[0] - c["m"], sum(1 for x in bypair.values() if False),
                     sd, lh, lt, float(eh is None or et is None), float(dg), n_words,
                     len(it["entities"]) + k, len(it["relations"])])
        keys.append(tuple(c["key"]))
    # col_n_top: pairs on which this relation is the pair's top candidate
    top = defaultdict(int)
    for c in cs:
        if c["m"] == bypair[(c["key"][0], c["key"][2])][0]:
            top[c["key"][1]] += 1
    for row, key in zip(rows, keys):
        row[8] = top[key[1]]
    return np.asarray(rows, dtype=np.float64).reshape(-1, len(FEATS)), keys


def short_set(run: str, split: str) -> list[dict]:
    """Per judged text: features, labels and the judged positives."""
    _, rows = pickle.loads((JUDGE_OUT / f"cands-{run}.pkl").read_bytes())
    labels = {(r["split"], r["id"]): r for r in json.loads((JUDGE_OUT / "labels-astra-c1.json").read_text())}
    texts = load_texts(split)
    out = []
    for i, it in enumerate(texts):
        lab = labels.get((split, it["id"]))
        if lab is None or (lab["facts"] and not lab["judged"]):
            continue
        assert rows[split][i]["item"]["id"] == it["id"]
        pos = {(f["head"], f["relation"], f["tail"]) for f in lab["facts"] if f["verdict"] in POSITIVE}
        X, keys = featurise(it, rows[split][i]["cand"])
        out.append({"id": it["id"], "X": X, "keys": keys,
                    "y": np.array([k in pos for k in keys], dtype=bool), "n_pos": len(pos)})
    return out


def conll_set(run: str, device=None) -> list[dict]:
    from . import heldout_zs as hz
    path = OUT / f"conll04-cands-{run}.pkl"
    if path.exists():
        rows = pickle.loads(path.read_bytes())
    else:
        from ..infer import Extractor
        ex = Extractor(run, device=device)
        dep = ex.unseen_offset
        ex.rel_offsets = {k: v + hz.SHIFT for k, v in ex.rel_offsets.items()}
        ex.unseen_offset = dep - hz.SHIFT
        ex.top = 50
        rows = hz.predict(ex, hz.load_conll04(), "gold")
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(rows))
        del ex
    out = []
    for r in rows:
        it = r["item"]
        pos = {(f["head"], f["relation"], f["tail"]) for f in it["facts"]}
        X, keys = featurise(it, r["cand"])
        out.append({"id": it["id"], "X": X, "keys": keys,
                    "y": np.array([k in pos for k in keys], dtype=bool), "n_pos": len(pos)})
    return out


def stack(texts):
    X = np.concatenate([t["X"] for t in texts])
    y = np.concatenate([t["y"] for t in texts])
    owner = np.concatenate([np.full(len(t["y"]), i) for i, t in enumerate(texts)])
    return X, y, owner


def counts(texts, scores, thr) -> np.ndarray:
    """Per text: (true positives, predicted, gold) at ``scores > thr``."""
    out, k = [], 0
    for t in texts:
        s = scores[k:k + len(t["y"])]
        k += len(t["y"])
        pr = s > thr
        out.append([int((pr & t["y"]).sum()), int(pr.sum()), t["n_pos"]])
    return np.asarray(out)


def f1(c):
    tp, p, g = c.sum(0)
    pr, rc = tp / max(p, 1), tp / max(g, 1)
    return 2 * pr * rc / max(pr + rc, 1e-12)


def best_thr(texts, scores, grid):
    fs = [f1(counts(texts, scores, g)) for g in grid]
    i = int(np.argmax(fs))
    return float(grid[i]), fs[i]


def ece(p, y, bins: int = 10) -> float:
    e, n = 0.0, len(p)
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b) if b < 1 else (p >= a) & (p <= b)
        if m.any():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def boot(ca, cb, n=2000, seed=13):
    rng = np.random.default_rng(seed)
    d = [f1(ca[i]) - f1(cb[i]) for i in (rng.integers(0, len(ca), len(ca)) for _ in range(n))]
    return float(f1(ca) - f1(cb)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def rel_fold(name: str) -> bool:
    return int(hashlib.sha1(f"cal:{name}".encode()).hexdigest(), 16) % 5 == 0


def main() -> int:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="ultrabert-380m-v7b8ev")
    ap.add_argument("--device")
    a = ap.parse_args()
    t0 = time.time()
    sets = {"short_train": short_set(a.run, "train"), "short_dev": short_set(a.run, "dev"),
            "conll04": conll_set(a.run, a.device)}
    D = {k: stack(v) for k, v in sets.items()}
    for k, (X, y, _) in D.items():
        print(f"{k}: {len(sets[k])} texts, {len(y)} candidates above {FLOOR}, "
              f"{int(y.sum())} positive among them, {sum(t['n_pos'] for t in sets[k])} gold", flush=True)
    Xtr, ytr, otr = D["short_train"]

    def gbm():
        return HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05,
                                              l2_regularization=1.0, random_state=0)

    def platt():
        return LogisticRegression(C=1.0, max_iter=1000)

    def fit_pred(make, cols, Xa, ya, Xb):
        mdl = make().fit(Xa[:, cols], ya)
        return mdl.predict_proba(Xb[:, cols])[:, 1]

    allc, mcol = list(range(len(FEATS))), [0]
    # out-of-fold scores on short train (threshold choice), 5 folds by text
    fold = otr % 5
    oof = {"platt": np.zeros(len(ytr)), "gbm": np.zeros(len(ytr))}
    for f in range(5):
        tr, te = fold != f, fold == f
        oof["platt"][te] = fit_pred(platt, mcol, Xtr[tr], ytr[tr], Xtr[te])
        oof["gbm"][te] = fit_pred(gbm, allc, Xtr[tr], ytr[tr], Xtr[te])
    preds = {"margin": {k: D[k][0][:, 0] for k in D}}
    preds["platt"] = {k: fit_pred(platt, mcol, Xtr, ytr, D[k][0]) for k in D}
    preds["gbm"] = {k: fit_pred(gbm, allc, Xtr, ytr, D[k][0]) for k in D}
    preds["platt"]["short_train"], preds["gbm"]["short_train"] = oof["platt"], oof["gbm"]
    grids = {"margin": np.arange(-6, 6.01, 0.25), "platt": np.linspace(0.02, 0.98, 49),
             "gbm": np.linspace(0.02, 0.98, 49)}

    res = {"run": a.run, "floor": FLOOR, "features": FEATS, "sets": {}, "paired": {}}
    print("\n| method | set | ECE | F1 at short-train threshold | F1 at own best | gap |")
    print("|---|---|---|---|---|---|")
    C = {}
    for meth in ("margin", "platt", "gbm"):
        thr, _ = best_thr(sets["short_train"], preds[meth]["short_train"], grids[meth])
        for s in D:
            sc = preds[meth][s]
            c = counts(sets[s], sc, thr)
            C[(meth, s)] = c
            own_t, own_f = best_thr(sets[s], sc, grids[meth])
            e = ece(sc, D[s][1]) if meth != "margin" else None
            res["sets"][f"{meth}|{s}"] = {"ece": e, "thr": thr, "f1": f1(c), "own_thr": own_t,
                                          "own_f1": own_f, "gap": own_f - f1(c)}
            print(f"| {meth} | {s} | {'' if e is None else f'{e:.4f}'} | {f1(c):.3f} (thr {thr:.2f}) | "
                  f"{own_f:.3f} (thr {own_t:.2f}) | {own_f - f1(c):+.3f} |", flush=True)
    print("\nPaired bootstrap (texts resampled), F1 at the short-train threshold, gbm minus margin:")
    for s in ("short_dev", "conll04"):
        d = boot(C[("gbm", s)], C[("margin", s)])
        res["paired"][f"gbm-vs-margin|{s}"] = d
        print(f"  {s}: {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")

    # held-out relations: train without them, score dev on them alone
    names = {k for t in sets["short_train"] + sets["short_dev"] for k in t["keys"]}
    test_rel = {r for _, r, _ in names if rel_fold(r)}
    keep_tr = np.array([k[1] not in test_rel for t in sets["short_train"] for k in t["keys"]])

    def only(texts, rels):
        out = []
        for t in texts:
            m = np.array([k[1] in rels for k in t["keys"]], dtype=bool).reshape(-1)
            out.append({"X": t["X"][m], "y": t["y"][m], "keys": [k for k, z in zip(t["keys"], m) if z],
                        "n_pos": None, "id": t["id"]})
        return out
    lab = {(r["split"], r["id"]): r for r in json.loads((JUDGE_OUT / "labels-astra-c1.json").read_text())}
    dev_ho = only(sets["short_dev"], test_rel)
    for t in dev_ho:
        t["n_pos"] = sum(1 for f in lab[("dev", t["id"])]["facts"]
                         if f["verdict"] in POSITIVE and f["relation"] in test_rel)
    Xh, yh, _ = stack(dev_ho)
    ho = {}
    for meth, make, cols in (("gbm_seen", gbm, allc), ("gbm_heldout", gbm, allc)):
        Xa, ya = (Xtr, ytr) if meth == "gbm_seen" else (Xtr[keep_tr], ytr[keep_tr])
        pa = fit_pred(make, cols, Xa, ya, Xh)
        # threshold from the same training data, out of fold
        o = otr if meth == "gbm_seen" else otr[keep_tr]
        Xo, yo = Xa, ya
        oo = np.zeros(len(yo))
        for f in range(5):
            tr, te = (o % 5) != f, (o % 5) == f
            oo[te] = fit_pred(make, cols, Xo[tr], yo[tr], Xo[te])
        tr_texts = sets["short_train"] if meth == "gbm_seen" else only(sets["short_train"], set(
            r for _, r, _ in names) - test_rel)
        if meth != "gbm_seen":
            for t in tr_texts:
                t["n_pos"] = sum(1 for f in lab[("train", t["id"])]["facts"]
                                 if f["verdict"] in POSITIVE and f["relation"] not in test_rel)
        thr, _ = best_thr(tr_texts, oo, grids["gbm"])
        c = counts(dev_ho, pa, thr)
        own_t, own_f = best_thr(dev_ho, pa, grids["gbm"])
        ho[meth] = {"ece": ece(pa, yh), "f1": f1(c), "own_f1": own_f, "thr": thr}
    ho["relations"] = sorted(test_rel)
    res["heldout_relations"] = ho
    print(f"\nHeld-out relations ({len(test_rel)} relations, {int(yh.sum())} positive dev candidates):")
    for k in ("gbm_seen", "gbm_heldout"):
        v = ho[k]
        print(f"  {k:12s} ECE {v['ece']:.4f} | F1 at train threshold {v['f1']:.3f} | own best {v['own_f1']:.3f}")
    mdl = gbm().fit(Xtr, ytr)
    from sklearn.inspection import permutation_importance
    Xd, yd, _ = D["short_dev"]
    sub = np.random.default_rng(0).choice(len(yd), min(20000, len(yd)), replace=False)
    imp = permutation_importance(mdl, Xd[sub], yd[sub], scoring="average_precision", n_repeats=3,
                                 random_state=0)
    res["importance"] = {FEATS[i]: float(imp.importances_mean[i]) for i in range(len(FEATS))}
    print("\nPermutation importance on dev (average precision):",
          ", ".join(f"{k} {v:.3f}" for k, v in sorted(res["importance"].items(), key=lambda x: -x[1])[:8]))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"p3-{a.run}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"written {out} [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

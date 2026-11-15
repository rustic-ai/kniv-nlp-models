"""Plan r4 readiness B: judge only the facts of an M1-style run that no judge
has read, and fold the verdicts back into the run's label files.

    uv run python -m v6.eval.b_batch trim --run m1v3-n500 --out m1det-n500    # leading articles trimmed in every config
    uv run python -m v6.eval.b_batch delta --run m1v3-n500                     # preds-m1v3-n500-new.pkl: unread facts only
    uv run python -m v6.eval.m1_precision judge --run m1v3-n500-new --version c3 --judge astra   # and deepseek
    uv run python -m v6.eval.b_batch fold --run m1v3-n500                      # verdicts into the run's astra / deepseek labels
    uv run python -m v6.eval.s5_judges merge --run m1v3-n500                   # agreement

``delta``: one config ``new`` holding, per text, each config's facts that
match no fact in the run's GPT-6 or DeepSeek labels (``_same``: shared
mention text and relation); the items carry no teacher facts, so a call
lists only those facts. ``fold``: each judged new fact is appended to the
run's labels of that judge, its sources the configs that predicted it, in
the same order for both judges (``s5_judges merge`` pairs facts by
position). ``trim``: every config's keys with a leading article dropped from
each mention (the deployed ``KNIV_TRIM_DET``), and the labels copied with
each fact's sources recomputed against the trimmed keys.
"""
from __future__ import annotations

import argparse
import json
import pickle

from .m1_precision import OUT, _same, labels_path

JUDGES = ("astra", "deepseek")
ART = ("the", "a", "an")


def _k(f):
    return (set(f["key"][0]), f["key"][1], set(f["key"][2]))


def _new(P, labs):
    """Per text: the distinct unread keys and, for each, the configs predicting it."""
    out = []
    for i in range(len(P["items"])):
        known = [[_k(f) for f in L["texts"][i]["facts"]] for L in labs]
        new = []
        for cfg, per in P["preds"].items():
            for k in per[i]:
                kk = (set(k[0]), k[1], set(k[2]))
                if all(any(_same(kk, x) for x in kn) for kn in known):
                    continue
                for n in new:
                    if _same(n[0], kk):
                        if cfg not in n[1]:
                            n[1].append(cfg)
                        break
                else:
                    new.append((kk, [cfg]))
        out.append(new)
    return out


def delta(run: str) -> int:
    P = pickle.loads((OUT / f"preds-{run}.pkl").read_bytes())
    labs = [json.loads(labels_path(run, j, "c3").read_text()) for j in JUDGES]
    new = _new(P, labs)
    items = [{**it, "facts": []} for it in P["items"]]
    keys = [[(frozenset(k[0]), k[1], frozenset(k[2])) for k, _ in n] for n in new]
    D = {"run": f"{run}-new", "items": items, "preds": {"new": keys},
         "margins": {"new": [{k: 0.0 for k in ks} for ks in keys]}}
    (OUT / f"preds-{run}-new.pkl").write_bytes(pickle.dumps(D))
    print(f"{run}: {sum(map(len, keys))} unread facts in {sum(bool(k) for k in keys)} texts "
          f"({2 * sum(bool(k) for k in keys)} calls for two judges)")
    return 0


def fold(run: str) -> int:
    P = pickle.loads((OUT / f"preds-{run}.pkl").read_bytes())
    labs = [json.loads(labels_path(run, j, "c3").read_text()) for j in JUDGES]
    new = _new(P, labs)
    for j, L in zip(JUDGES, labs):
        D = json.loads(labels_path(f"{run}-new", j, "c3").read_text())
        added = unread = 0
        for i, (t, d) in enumerate(zip(L["texts"], D["texts"])):
            assert t["id"] == d["id"]
            by = [(set(f["key"][0]), f["key"][1], set(f["key"][2])) for f in d["facts"]]
            for kk, src in new[i]:
                n = next((n for n, x in enumerate(by) if _same(x, kk)), None)
                v = None if n is None or not d["judged"] else d["facts"][n]["verdict"]
                unread += v is None
                t["facts"].append({"key": [sorted(kk[0]), kk[1], sorted(kk[2])], "src": src, "verdict": v})
                added += 1
        labels_path(run, j, "c3").write_text(json.dumps(L))
        print(f"{j}: {added} facts added to {run}'s labels, {unread} without a verdict")
    return 0


def _trim(ms):
    out = set()
    for m in ms:
        w = m.split()
        out.add(" ".join(w[1:]) if len(w) > 1 and w[0].lower() in ART else m)
    return frozenset(out)


def trim(run: str, out: str) -> int:
    P = pickle.loads((OUT / f"preds-{run}.pkl").read_bytes())
    T = {**P, "run": out, "preds": {}, "margins": {}}
    for cfg in P["preds"]:
        T["preds"][cfg] = [list(dict.fromkeys((_trim(k[0]), k[1], _trim(k[2])) for k in ks)) for ks in P["preds"][cfg]]
        T["margins"][cfg] = []
        for ms in P["margins"][cfg]:
            d = {}
            for k, m in ms.items():
                k2 = (_trim(k[0]), k[1], _trim(k[2]))
                d[k2] = max(m, d.get(k2, -1e9))
            T["margins"][cfg].append(d)
    (OUT / f"preds-{out}.pkl").write_bytes(pickle.dumps(T))
    for j in JUDGES + ("both",):
        L = json.loads(labels_path(run, j, "c3").read_text())
        for i, t in enumerate(L["texts"]):
            for f in t["facts"]:
                kk = _k(f)
                f["src"] = [s for s in f["src"] if s == "teacher"] + \
                    [c for c in T["preds"] if any(_same(kk, (set(k[0]), k[1], set(k[2]))) for k in T["preds"][c][i])]
        labels_path(out, j, "c3").write_text(json.dumps(L))
    print(f"written preds-{out}.pkl and its labels (sources recomputed on the trimmed keys)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["delta", "fold", "trim"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    return delta(a.run) if a.cmd == "delta" else fold(a.run) if a.cmd == "fold" else trim(a.run, a.out)


if __name__ == "__main__":
    raise SystemExit(main())

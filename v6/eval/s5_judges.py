"""Stage 5 (v8 plan r4): a second judge on the stage-3 M1 file.

    uv run python -m v6.eval.s5_judges merge  [--run m1s3b-n500] [--a astra] [--b deepseek]
    uv run python -m v6.eval.s5_judges sample [--n 200]
    uv run python -m v6.eval.s5_judges dev

``dev``: the threshold set (d123 nested with V1b, short-dev texts 500-999,
cut at p 0.05) under each judge and under agreement: precision, recall and
F1 by threshold, and the F1-optimal threshold.

``merge`` reads both judges' labels on the same facts, prints agreement
(raw and Cohen's kappa on expressed / not) overall and per configuration's
facts, and writes ``labels-RUN-both-c3.json``: a fact is expressed only when
both judges say so (the first judge's verdict kept). ``equal --judge both``
and ``equal --judge B`` then give the M1 gates under agreement and under the
second judge alone.

``sample`` draws facts where the judges disagree (half each direction,
seed 7) for reading by hand, and writes them with their passages to
``s5-disagree.json``; ``hand_expressed`` is filled in by the reader.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from .m1_precision import OUT, POSITIVE, build, labels_path


def _load(run, a, b, version):
    A = json.loads(labels_path(run, a, version).read_text())
    B = json.loads(labels_path(run, b, version).read_text())
    by = {t["id"]: t for t in B["texts"]}
    return A, B, by


def _kappa(c):
    n = sum(c.values())
    po = (c[(True, True)] + c[(False, False)]) / n
    pa = (c[(True, True)] + c[(True, False)]) / n
    pb = (c[(True, True)] + c[(False, True)]) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return n, po, (po - pe) / (1 - pe), pa, pb


def merge(run, a, b, version):
    A, B, by = _load(run, a, b, version)
    out, per = [], {k: Counter() for k in ["all", "teacher"] + A["configs"]}
    for t in A["texts"]:
        u = by[t["id"]]
        facts = []
        for fa, fb in zip(t["facts"], u["facts"]):
            assert fa["key"] == fb["key"]
            ea, eb = fa["verdict"] in POSITIVE, fb["verdict"] in POSITIVE
            if t["judged"] and u["judged"]:
                for k in ["all"] + fa["src"]:
                    per[k][(ea, eb)] += 1
            facts.append({**fa, "verdict": fa["verdict"] if ea and eb else
                          (None if fa["verdict"] is None or fb["verdict"] is None else "not_supported")})
        out.append({"id": t["id"], "judged": t["judged"] and u["judged"], "facts": facts})
    print(f"| facts of | n | agreement | kappa | expressed ({a}) | expressed ({b}) |\n|---|---|---|---|---|---|")
    for k, c in per.items():
        if c:
            n, po, kap, pa, pb = _kappa(c)
            print(f"| {k} | {n} | {po:.3f} | {kap:.3f} | {pa:.3f} | {pb:.3f} |")
    p = labels_path(run, "both", version)
    p.write_text(json.dumps({"configs": A["configs"], "texts": out}))
    print(f"written {p.name}")
    return 0


def sample(run, a, b, version, n):
    A, B, by = _load(run, a, b, version)
    _, rows = build(run)
    items = {it["id"]: it for it, _ in rows}
    pools = {(True, False): [], (False, True): []}
    for t in A["texts"]:
        for j, (fa, fb) in enumerate(zip(t["facts"], by[t["id"]]["facts"])):
            if fa["verdict"] is None or fb["verdict"] is None:
                continue
            k = (fa["verdict"] in POSITIVE, fb["verdict"] in POSITIVE)
            if k in pools:
                pools[k].append((t["id"], j, fa, fb))
    rng = random.Random(7)
    picked = []
    for k, pool in pools.items():
        picked += rng.sample(pool, min(n // 2, len(pool)))
    facts = []
    for i, (tid, j, fa, fb) in enumerate(picked):
        it = items[tid]
        facts.append({"i": i, "id": tid, "fact": j, "key": fa["key"], "src": fa["src"],
                      a: fa["verdict"], b: fb["verdict"], "hand_expressed": None,
                      "passage": [" ".join(it["tokens"][s:e]) for s, e in it["sentence_spans"]],
                      "relation": next((r["description"] for r in it["relations"]
                                        if r["name"] == fa["key"][1]), "")})
    (OUT / "s5-disagree.json").write_text(json.dumps({
        "note": f"{len(facts)} facts where {a} and {b} disagree (half each way, seed 7); "
                "hand_expressed = the fact is stated with arguments that fit the relation",
        "facts": facts}, indent=1))
    print(f"{len(facts)} facts; pools {a}-only {len(pools[(True, False)])}, {b}-only {len(pools[(False, True)])}")
    return 0


DEV = ("ultrabert-380m-v8p3-d123-dev-500", "ner+np+nest+merge+ver:2", 0.05)


def dev(a, b, version):
    import numpy as np
    from .p3_verifier import GRID, _logit, e2e_curve
    name, cfg, t0 = DEV
    if not labels_path(name, "both", version).exists():
        merge(name, a, b, version)
    print("| judge | 0.4 P / R | 0.5 P / R | 0.6 P / R | 0.7 P / R | 0.8 P / R | best F1 (threshold) |\n|---|---|---|---|---|---|---|")
    for j in (a, b, "both"):
        m, ok, total = e2e_curve(name, cfg, j)
        m = m + _logit(t0)
        cells, fs = [], []
        for t in GRID:
            k = m >= _logit(t)
            p, r = ok[k].sum() / max(k.sum(), 1), ok[k].sum() / max(total, 1)
            fs.append(2 * p * r / max(p + r, 1e-9))
            if any(abs(t - x) < 1e-6 for x in (0.4, 0.5, 0.6, 0.7, 0.8)):
                cells.append(f"{p:.3f} / {r:.3f}")
        bi = int(np.argmax(fs))
        print(f"| {j} | " + " | ".join(cells) + f" | {fs[bi]:.3f} ({GRID[bi]:.2f}) |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["merge", "sample", "dev"])
    ap.add_argument("--run", default="m1s3b-n500")
    ap.add_argument("--a", default="astra")
    ap.add_argument("--b", default="deepseek")
    ap.add_argument("--version", default="c3")
    ap.add_argument("--n", type=int, default=200)
    x = ap.parse_args()
    if x.cmd == "merge":
        return merge(x.run, x.a, x.b, x.version)
    if x.cmd == "dev":
        return dev(x.a, x.b, x.version)
    return sample(x.run, x.a, x.b, x.version, x.n)


if __name__ == "__main__":
    raise SystemExit(main())

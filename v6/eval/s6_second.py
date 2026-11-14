"""Second judge on the verifier's training labels (v8 plan r4, review step 3).

    uv run python -m v6.eval.s6_second judge --set SET [--n N] [--first F]
    uv run python -m v6.eval.s6_second agree [--set SET ...]

DeepSeek-V4-Pro with prompt c3k (c3's verdicts, and the kind of each hedged
fact) on facts GPT-6 already judged with c3, in the same order and the same
passage rendering:

- ``dev``: the threshold set (d123 nested + V1b, short dev 500-999), the
  prompt check: c3k against DeepSeek's own c3 labels on the same facts;
- ``s4``: the stage-4 pilot's judged long-window disagreements;
- ``d1e``: D1e's judged end-to-end candidates (short training texts, four files);
- ``d1``: a random ``n`` of D1's judged texts (gold-entity candidates), seed 7.

Writes ``s6-labels-<set>-deepseek-c3k.json``: per text the GPT-6 file's id,
whether DeepSeek's call parsed, and per fact its verdict and hedge kind.
``agree`` prints agreement and kappa on expressed / not, per set, the share of
facts both call expressed, and the hedge kinds.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import random
from collections import Counter
from pathlib import Path

from ..config import RUNS_DIR, load_annotators
from .m1_precision import OUT as M1, POSITIVE, labels_path

OUT = RUNS_DIR / "s6-second"
D1E = [f"ultrabert-380m-v8e-w1half-train-{o}" for o in (1200, 1700, 2200, 2700)]
DEV = "ultrabert-380m-v8p3-d123-dev-500"
D1 = RUNS_DIR / "kg-judge" / "labels-d1-full-ultrabert-380m-v7b8ev.json"
JUDGE, VERSION = "deepseek", "c3k"


def _m1_rows(run):
    """(id, prompt, GPT-6 facts) for an M1-format run (preds + c3 labels)."""
    from .m1_precision import build, judge_prompt
    _, rows = build(run)
    L = {t["id"]: t for t in json.loads(labels_path(run, "astra", "c3").read_text())["texts"]}
    out = []
    for it, facts in rows:
        t = L.get(it["id"])
        if not facts or t is None or not t["judged"]:
            continue
        assert [list(map(sorted, (f["key"][0], f["key"][2]))) for f in facts] == \
               [[f["key"][0], f["key"][2]] for f in t["facts"]]
        out.append((it, judge_prompt(it, facts), [f["verdict"] for f in t["facts"]]))
    return out


def rows_of(name, n, seed=7):
    if name == "dev":
        return _m1_rows(DEV)
    if name == "d1e":
        return [r for run in D1E for r in _m1_rows(run)]
    if name == "s4":
        from .m1_precision import judge_prompt
        S = RUNS_DIR / "s4-pilot"
        items = {it["id"]: it for it in json.loads((S / "items.json").read_text())}
        out = []
        for t in json.loads((S / "labels-astra-c3.json").read_text()):
            if t["judged"]:
                it = items[t["id"]]
                facts = [{"key": (c["head_m"], c["relation"], c["tail_m"])} for c in t["cands"]]
                out.append((it, judge_prompt(it, facts), [c["judge"] for c in t["cands"]]))
        return out
    if name == "d1":
        from ..kg_judge import judge_prompt, load_texts
        L = [r for r in json.loads(D1.read_text()) if r["judged"] and r["facts"] and r["split"] == "train"]
        pick = random.Random(seed).sample(L, min(n, len(L)))
        texts = {t["id"]: t for t in load_texts("train", n=10 ** 9)}
        return [(texts[r["id"]], judge_prompt(texts[r["id"]], r["facts"]), [f["verdict"] for f in r["facts"]])
                for r in pick if r["id"] in texts]
    raise SystemExit(f"unknown set {name}")


def _path(name):
    return OUT / f"s6-labels-{name}-{JUDGE}-{VERSION}.json"


def judge(name, n, first):
    from ..annotate.base import CacheStore
    from ..annotate.kg import KGItem
    from ..kg_judge import CandJudge, _run
    from ..kg_teacher import CACHE
    rows = rows_of(name, n)
    if first:
        rows = rows[:first]
    items = []
    for it, prompt, _ in rows:
        q = KGItem(id=f"{it['id']}:s6:{name}", tokens=it["tokens"], sentence_spans=it["sentence_spans"],
                   speakers=it.get("speakers"), participants=it.get("participants") or [],
                   types=it.get("types") or [], relations=it["relations"], domain=it.get("domain") or "")
        q.layers = {"judge_prompt": prompt}
        items.append(q)
    print(f"{name}: {len(items)} calls, {sum(len(g) for *_, g in rows)} facts", flush=True)
    ann = CandJudge(load_annotators([JUDGE])[JUDGE], CacheStore(CACHE, VERSION))
    ann.prompt, ann.layer = "kgcand3k", "kgcandk"
    res = asyncio.run(_run(ann, items, f"s6-{name}-{JUDGE}-{VERSION}"))
    out = []
    for it, _, g in rows:
        r = res.get(f"{it['id']}:s6:{name}")
        v = r.payload if r is not None and r.ok else {}
        fs = []
        for j in range(len(g)):
            x = v.get(j)
            verdict, _, kind = (x or "").partition(":")
            fs.append({"gpt6": g[j], "verdict": verdict or None, "kind": kind or None})
        out.append({"id": it["id"], "judged": bool(v), "facts": fs})
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"-first{first}" if first else ""
    p = _path(name).with_name(_path(name).stem + tag + ".json")
    p.write_text(json.dumps(out))
    print(Counter(f["verdict"] for t in out for f in t["facts"]))
    print(Counter(f["kind"] for t in out for f in t["facts"] if f["verdict"] == "hedged"))
    print("written", p)
    return 0


def _kappa(c):
    n = sum(c.values())
    po = (c[(True, True)] + c[(False, False)]) / n
    pa = (c[(True, True)] + c[(True, False)]) / n
    pb = (c[(True, True)] + c[(False, True)]) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return n, po, (po - pe) / (1 - pe), pa, pb


def agree(names):
    print("| set | facts | agreement | kappa | expressed (GPT-6) | expressed (DeepSeek c3k) | both expressed | hedge kinds |")
    print("|---|---|---|---|---|---|---|---|")
    for name in names:
        for p in sorted(OUT.glob(f"s6-labels-{name}-{JUDGE}-{VERSION}*.json")):
            T = json.loads(p.read_text())
            c, k = Counter(), Counter()
            for t in T:
                if not t["judged"]:
                    continue
                for f in t["facts"]:
                    if f["gpt6"] is None or f["verdict"] is None:
                        continue
                    c[(f["gpt6"] in POSITIVE, f["verdict"] in POSITIVE)] += 1
                    if f["verdict"] == "hedged":
                        k[f["kind"]] += 1
            n, po, kap, pa, pb = _kappa(c)
            print(f"| {p.stem.split('-labels-')[1]} | {n} | {po:.3f} | {kap:.3f} | {pa:.3f} | {pb:.3f} | "
                  f"{c[(True, True)] / n:.3f} | {dict(k)} |")
    if "dev" in names and _path("dev").exists():          # the prompt check: c3k against DeepSeek's c3
        T = json.loads(_path("dev").read_text())
        D = {t["id"]: t for t in json.loads(labels_path(DEV, "deepseek", "c3").read_text())["texts"]}
        c = Counter()
        for t in T:
            u = D.get(t["id"])
            if not t["judged"] or u is None or not u["judged"]:
                continue
            for f, g in zip(t["facts"], u["facts"]):
                if f["verdict"] is not None and g["verdict"] is not None:
                    c[(g["verdict"] in POSITIVE, f["verdict"] in POSITIVE)] += 1
        n, po, kap, pa, pb = _kappa(c)
        print(f"\ndev, DeepSeek c3 against c3k on the same facts: {n} facts, agreement {po:.3f}, kappa {kap:.3f}, "
              f"expressed {pa:.3f} (c3) / {pb:.3f} (c3k)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["judge", "agree"])
    ap.add_argument("--set", nargs="+", default=["dev", "s4", "d1e", "d1"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--first", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "judge":
        for s in a.set:
            judge(s, a.n, a.first)
        return 0
    return agree(a.set)


if __name__ == "__main__":
    raise SystemExit(main())

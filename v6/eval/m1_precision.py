"""M1 (v8 plan r2): judged precision of end-to-end extraction, flat against
nested mention decoding.

    uv run python -m v6.eval.m1_precision predict --run ultrabert-380m-v7b8ev --short 300
    uv run python -m v6.eval.m1_precision judge
    uv run python -m v6.eval.m1_precision score [--merge]
    uv run python -m v6.eval.m1_precision rescore --tag=-merge2   # merged decode, no new calls

Pilot P1 found that nested proposals from the parse find both arguments of
far more gold facts, yet end-to-end F1 against the teacher's facts fell
because precision halved. The teacher lists about half of the facts a text
expresses (P0), so a "false" prediction may be a fact it missed. Here every
predicted fact of every configuration, on the same short-dev texts, is judged
on its own (the ``kgcand`` verdicts: supported, not_supported, negated,
hedged; the first, third and fourth mean the fact is expressed), and each
configuration gets

- judged precision: expressed / predicted;
- pooled recall: expressed facts it found / expressed facts any configuration
  found plus the teacher's facts (a lower bound on the misses of each);
- judged F1 from the two, with a paired bootstrap over texts against the
  first configuration.

``--merge`` counts span variants of one fact once (``_groups``): nested
decoding returns "Northwind" and "Northwind 's plant" as separate heads.

Facts are matched across configurations and to the teacher's by mention text
(case-insensitive, any shared mention), as ``heldout_zs.score_item`` does.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pickle
from collections import Counter

import numpy as np

from ..config import RUNS_DIR, load_annotators
from . import heldout_zs as hz

OUT = RUNS_DIR / "m1-precision"
POSITIVE = ("supported", "negated", "hedged")


def _configure(ex, cfg: str, boost0):
    name, _, b = cfg.partition(":")
    parts = name.split("+")
    ex.nested = "nest" in parts
    ex.merge_variants = "merge" in parts
    ex.proposers = tuple(p for p in parts if p not in ("base", "nest", "merge"))
    ex.span_boost = float(b) if b else (boost0 if boost0 is not None else 0.0)


def predict(run: str, n: int, configs: list[str], tag: str = "") -> None:
    from ..infer import Extractor
    items = hz.load_short_dev(n)
    ex = Extractor(run)
    boost0, dep = ex.span_boost, ex.unseen_offset
    ex.rel_offsets = {k: v + hz.SHIFT for k, v in ex.rel_offsets.items()}
    ex.unseen_offset = dep - hz.SHIFT
    ex.top = max(ex.top, 4)
    preds, margins = {}, {}
    for cfg in configs:
        _configure(ex, cfg, boost0)
        rows = hz.predict(ex, items, "e2e")
        preds[cfg] = [sorted({c["key"] for c in r["cand"] if c["m"] > 0}, key=str) for r in rows]
        margins[cfg] = [{c["key"]: max(c["m"], -1e9) for c in r["cand"] if c["m"] > 0} for r in rows]
        print(f"{cfg}: {sum(map(len, preds[cfg]))} predicted facts on {len(items)} texts", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"preds-{run}{tag}.pkl").write_bytes(pickle.dumps({"run": run, "items": items, "preds": preds,
                                                                      "margins": margins}))


def _same(a, b) -> bool:
    return a[1] == b[1] and bool(a[0] & b[0]) and bool(a[2] & b[2])


def _teacher_keys(it: dict) -> list[tuple]:
    return [(hz._mentions(it, f["head"]), f["relation"], hz._mentions(it, f["tail"]))
            for f in it["facts"] if f["head"] != f["tail"]]


def facts_to_judge(it: dict, per_cfg: dict) -> list[dict]:
    """Teacher facts first, then each configuration's predictions not already
    listed (matched by shared mention text and relation)."""
    out = [{"key": k, "src": ["teacher"]} for k in _teacher_keys(it)]
    for cfg, keys in per_cfg.items():
        for k in keys:
            for f in out:
                if _same(f["key"], k):
                    if cfg not in f["src"]:
                        f["src"].append(cfg)
                    break
            else:
                out.append({"key": k, "src": [cfg]})
    return out


def _name(ms) -> str:
    return " / ".join(sorted(ms, key=len, reverse=True)[:3])


def judge_prompt(it: dict, facts: list[dict]) -> str:
    defs = {r["name"]: r["description"] for r in it["relations"]}
    lines = ["Passage:"]
    for j, (s, e) in enumerate(it["sentence_spans"]):
        who = f"[{it['speakers'][j]}] " if it.get("speakers") and it["speakers"][j] else ""
        lines.append(f"{j + 1}\t{who}{' '.join(it['tokens'][s:e])}")
    lines += ["", "Facts:"]
    for n, f in enumerate(facts):
        h, r, t = f["key"]
        lines.append(f"{n}\t[{_name(h)}] {r} [{_name(t)}]  ({r}: {defs.get(r, '')})")
    return "\n".join(lines)


def build(run: str):
    d = pickle.loads((OUT / f"preds-{run}.pkl").read_bytes())
    rows = []
    for i, it in enumerate(d["items"]):
        rows.append((it, facts_to_judge(it, {c: p[i] for c, p in d["preds"].items()})))
    return d, rows


def judge(run: str, judge_name: str) -> int:
    from ..annotate.base import CacheStore
    from ..annotate.kg import KGItem
    from ..kg_judge import VERSION, CandJudge, _run
    from ..kg_teacher import CACHE
    d, rows = build(run)
    spec = load_annotators([judge_name])[judge_name]
    items = []
    for it, facts in rows:
        if not facts:
            continue
        q = KGItem(id=f"{it['id']}:m1:{run}", tokens=it["tokens"], sentence_spans=it["sentence_spans"],
                   speakers=it.get("speakers"), participants=it["participants"], types=it["types"],
                   relations=it["relations"], domain=it.get("domain") or "")
        q.layers = {"judge_prompt": judge_prompt(it, facts)}
        items.append(q)
    print(f"{len(rows)} texts, {sum(len(f) for _, f in rows)} facts, {len(items)} calls", flush=True)
    res = asyncio.run(_run(CandJudge(spec, CacheStore(CACHE, VERSION)), items, f"m1-{judge_name}"))
    out = []
    for it, facts in rows:
        r = res.get(f"{it['id']}:m1:{run}")
        v = r.payload if r is not None and r.ok else {}
        out.append({"id": it["id"], "judged": bool(v) or not facts,
                    "facts": [{"key": [sorted(f["key"][0]), f["key"][1], sorted(f["key"][2])],
                               "src": f["src"], "verdict": v.get(n)} for n, f in enumerate(facts)]})
    (OUT / f"labels-{run}-{judge_name}.json").write_text(json.dumps({"configs": list(d["preds"]),
                                                                     "texts": out}))
    print(Counter(f["verdict"] for t in out for f in t["facts"]))
    return 0


def _overlap(a, b) -> bool:
    return a[1] == b[1] and all(any(x in y or y in x for x in p for y in q)
                                for p, q in ((a[0], b[0]), (a[2], b[2])))


def _groups(facts: list[dict]) -> list[list[dict]]:
    """Facts that differ only in how much of a mention their arguments span
    ("Northwind" / "Northwind 's new plant"): same relation, each argument's
    text contained in the other's. Nested decoding returns such variants."""
    out = []
    for f in facts:
        for g in out:
            if _overlap(g[0]["key"], f["key"]):
                g.append(f)
                break
        else:
            out.append([f])
    return out


def score(run: str, judge_name: str, boot: int = 2000, merge: bool = False) -> int:
    L = json.loads((OUT / f"labels-{run}-{judge_name}.json").read_text())
    cfgs, texts = L["configs"], [t for t in L["texts"] if t["judged"]]
    if merge:      # count a fact once however many span variants of it were predicted
        for t in texts:
            ok = [f for f in t["facts"] if f["verdict"] in POSITIVE]
            pool = len(_groups(ok))
            t["_merged"] = {c: (sum(any(f["verdict"] in POSITIVE for f in g) for g in gs), len(gs))
                            for c in cfgs
                            for gs in [_groups([f for f in t["facts"] if c in f["src"]])]}
            t["_pool"] = pool
    # per text, per config: [expressed predicted, predicted, expressed in the pool, teacher-tp, gold]
    C = {c: np.zeros((len(texts), 5)) for c in cfgs}
    for i, t in enumerate(texts):
        pool = sum(f["verdict"] in POSITIVE for f in t["facts"])
        gold = sum("teacher" in f["src"] for f in t["facts"])
        for c in cfgs:
            mine = [f for f in t["facts"] if c in f["src"]]
            good = sum(f["verdict"] in POSITIVE for f in mine)
            tt = sum("teacher" in f["src"] for f in mine)
            C[c][i] = [good, len(mine), pool, tt, gold]
            if merge:
                C[c][i][:3] = [*t["_merged"][c], t["_pool"]]

    def stats(a):
        s = a.sum(0)
        p, r = s[0] / max(s[1], 1), s[0] / max(s[2], 1)
        return p, r, 2 * p * r / max(p + r, 1e-12), s[3] / max(s[1], 1), s[3] / max(s[4], 1)

    res = {}
    print(f"{len(texts)} judged texts; pool of expressed facts {int(C[cfgs[0]][:, 2].sum())}\n")
    print("| config | predicted | judged precision | pooled recall | judged F1 | teacher P | teacher R |")
    print("|---|---|---|---|---|---|---|")
    for c in cfgs:
        p, r, f, tp_, tr_ = stats(C[c])
        res[c] = {"predicted": int(C[c][:, 1].sum()), "judged_p": p, "pooled_r": r, "judged_f1": f,
                  "teacher_p": tp_, "teacher_r": tr_}
        print(f"| {c} | {int(C[c][:, 1].sum())} | {p:.3f} | {r:.3f} | {f:.3f} | {tp_:.3f} | {tr_:.3f} |")
    rng = np.random.default_rng(13)
    base = cfgs[0]
    print(f"\npaired bootstrap against {base} (texts resampled): judged precision, judged F1")
    for c in cfgs[1:]:
        dp, df = [], []
        for _ in range(boot):
            ix = rng.integers(0, len(texts), len(texts))
            a, b = stats(C[c][ix]), stats(C[base][ix])
            dp.append(a[0] - b[0]); df.append(a[2] - b[2])
        a, b = stats(C[c]), stats(C[base])
        res[f"{c}-vs-{base}"] = {"precision": [a[0] - b[0], *np.percentile(dp, [2.5, 97.5])],
                                 "f1": [a[2] - b[2], *np.percentile(df, [2.5, 97.5])]}
        print(f"  {c}: precision {a[0] - b[0]:+.3f} [{np.percentile(dp, 2.5):+.3f}, "
              f"{np.percentile(dp, 97.5):+.3f}]; F1 {a[2] - b[2]:+.3f} [{np.percentile(df, 2.5):+.3f}, "
              f"{np.percentile(df, 97.5):+.3f}]")
    (OUT / f"score-{run}-{judge_name}{'-merged' if merge else ''}.json").write_text(json.dumps(res, indent=1, default=float))
    return 0


def rescore(run: str, judge_name: str, tag: str, base_tag: str, boot: int = 2000) -> int:
    """Score another predictions file (``predict --tag``) against the judged
    facts without new judge calls: valid when its predictions are a subset of
    the judged ones (e.g. the variant merge keeps one of each judged group).
    Each prediction takes the verdict of the judged fact with the same
    mentions, else of one sharing a mention (as the judge pass matched);
    recall counts the variant groups found against the pooled ones."""
    L = json.loads((OUT / f"labels-{run}-{judge_name}.json").read_text())
    texts = L["texts"]
    files = {base_tag: None, tag: None}
    res = {}
    for t_ in files:
        P = pickle.loads((OUT / f"preds-{run}{t_}.pkl").read_bytes())["preds"]
        for c, pr in P.items():
            A, un = [], 0
            for t, keys in zip(texts, pr):
                pool = len(_groups([f for f in t["facts"] if f["verdict"] in POSITIVE]))
                fs = [(set(f["key"][0]), f["key"][1], set(f["key"][2]), f["verdict"]) for f in t["facts"]]
                good, found = 0, []
                for h, r, tt in keys:
                    k = (set(h), r, set(tt))
                    v = next((x[3] for x in fs if x[:3] == k), None)
                    if v is None:
                        v = next((x[3] for x in fs if _same(k, x[:3])), None)
                    un += v is None
                    if v in POSITIVE:
                        good += 1
                        found.append({"key": [sorted(h), r, sorted(tt)]})
                A.append([good, len(keys), pool, len(_groups(found))])
            res[f"{t_ or 'judged'}|{c}"] = (np.array(A, float), un)

    def st(x):
        s = x.sum(0)
        p, r = s[0] / max(s[1], 1), s[3] / max(s[2], 1)
        return p, r, 2 * p * r / max(p + r, 1e-12)
    print("| predictions | config | facts | unjudged | judged precision | groups found | judged F1 |")
    print("|---|---|---|---|---|---|---|")
    for k, (a, un) in res.items():
        p, r, f = st(a)
        print(f"| {k.split('|')[0]} | {k.split('|')[1]} | {int(a[:, 1].sum())} | {un} | {p:.3f} | {r:.3f} | {f:.3f} |")
    keys = list(res)
    b = res[keys[0]][0]
    rng = np.random.default_rng(13)
    print(f"\npaired bootstrap against {keys[0]}")
    out = {}
    for k in keys[1:]:
        a = res[k][0]
        d = []
        for _ in range(boot):
            ix = rng.integers(0, len(a), len(a))
            x, y = st(a[ix]), st(b[ix])
            d.append((x[0] - y[0], x[2] - y[2]))
        d = np.array(d)
        x, y = st(a), st(b)
        out[k] = {"precision": [x[0] - y[0], *np.percentile(d[:, 0], [2.5, 97.5])],
                  "f1": [x[2] - y[2], *np.percentile(d[:, 1], [2.5, 97.5])], "p": x[0], "r": x[1], "f": x[2]}
        print(f"  {k}: precision {x[0] - y[0]:+.3f} [{np.percentile(d[:, 0], 2.5):+.3f}, "
              f"{np.percentile(d[:, 0], 97.5):+.3f}]; F1 {x[2] - y[2]:+.3f} "
              f"[{np.percentile(d[:, 1], 2.5):+.3f}, {np.percentile(d[:, 1], 97.5):+.3f}]")
    (OUT / f"rescore-{run}{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["predict", "judge", "score", "rescore"])
    ap.add_argument("--run", default="ultrabert-380m-v7b8ev")
    ap.add_argument("--short", type=int, default=300)
    ap.add_argument("--configs", default="base,ner+np:1,ner+np+nest:2")
    ap.add_argument("--judge", default="astra")
    ap.add_argument("--tag", default="", help="predict / rescore: suffix of the predictions file")
    ap.add_argument("--base-tag", default="-merge", help="rescore: the file whose first config is the baseline")
    ap.add_argument("--merge", action="store_true", help="span variants of one fact count once")
    a = ap.parse_args()
    if a.cmd == "predict":
        predict(a.run, a.short, a.configs.split(","), a.tag)
        return 0
    if a.cmd == "judge":
        return judge(a.run, a.judge)
    if a.cmd == "rescore":
        return rescore(a.run, a.judge, a.tag, a.base_tag)
    return score(a.run, a.judge, merge=a.merge)


if __name__ == "__main__":
    raise SystemExit(main())

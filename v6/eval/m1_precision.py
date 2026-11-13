"""M1 (v8 plan r2): judged precision of end-to-end extraction, flat against
nested mention decoding.

    uv run python -m v6.eval.m1_precision predict --run ultrabert-380m-v7b8ev --short 300
    uv run python -m v6.eval.m1_precision judge
    uv run python -m v6.eval.m1_precision score [--merge]
    uv run python -m v6.eval.m1_precision rescore --tag=-merge2   # merged decode, no new calls
    uv run python -m v6.eval.m1_precision combine --run m1p3 --runs v7.1=ultrabert-380m-v7b8ev,...
    uv run python -m v6.eval.m1_precision equal --run m1p3 --version c3   # equal recall / precision

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


def load_set(name: str, n: int) -> list[dict]:
    """Texts of one M1 slice, as ``heldout_zs`` items."""
    if name == "short":
        return hz.load_short_dev(n)
    if name.startswith("train"):    # D1e (plan r4 5.2): "train:OFFSET", short training texts
        from .negation_pilot import heldout_base  # outside M3's / V1's held-out fifth
        off = int(name.partition(":")[2] or 0)
        items = hz.load_short_dev(off + n, split="train", keep=lambda i: not heldout_base(i))
        return items[off:off + n]
    if name == "conll04":
        return hz.load_conll04(n)
    if name == "m4":            # complete expected facts: m4_business scores it without a judge
        from .m4_business import load
        items = load()
        return items[:n] if n else items
    raise ValueError(name)


def set_key(run: str, name: str) -> str:
    """File key: the short-dev slice keeps the original names."""
    return run if name == "short" else f"{run}-{name.replace(':', '-')}"


def predict(run: str, n: int, configs: list[str], tag: str = "", set_name: str = "short",
            floor: float = 0.0) -> None:
    """Each config's end-to-end facts above the deployed bar; with ``floor``,
    candidates down to ``floor`` under it, the best two relations per pair
    (D1e's candidates, as D1 took within 3 of the bar)."""
    from ..infer import Extractor
    items = load_set(set_name, n)
    ex = Extractor(run)
    boost0, dep = ex.span_boost, ex.unseen_offset
    ex.rel_offsets = {k: v + hz.SHIFT for k, v in ex.rel_offsets.items()}
    ex.unseen_offset = dep - hz.SHIFT
    ex.top = max(ex.top, 4)
    preds, margins = {}, {}
    for cfg in configs:
        _configure(ex, cfg, boost0)
        rows = hz.predict(ex, items, "e2e")
        if floor:
            rows = [{**r, "cand": _top_per_pair([c for c in r["cand"] if c["m"] > -floor], 2)} for r in rows]
        preds[cfg] = [sorted({c["key"] for c in r["cand"] if c["m"] > -floor}, key=str) for r in rows]
        margins[cfg] = [{c["key"]: max(c["m"], -1e9) for c in r["cand"] if c["m"] > -floor} for r in rows]
        print(f"{cfg}: {sum(map(len, preds[cfg]))} predicted facts on {len(items)} texts", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"preds-{set_key(run, set_name)}{tag}.pkl").write_bytes(pickle.dumps({"run": run, "items": items, "preds": preds,
                                                                      "margins": margins}))


def _top_per_pair(cands: list[dict], k: int) -> list[dict]:
    by = {}
    for c in sorted(cands, key=lambda c: -c["m"]):
        pair = (c["key"][0], c["key"][2])
        if len(by.setdefault(pair, [])) < k:
            by[pair].append(c)
    return [c for cs in by.values() for c in cs]


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


PROMPTS = {"c1": "kgcand", "c2": "kgcand2", "c3": "kgcand3"}


def judge(run: str, judge_name: str, version: str = "c1", only: set | None = None) -> int:
    from ..annotate.base import CacheStore
    from ..annotate.kg import KGItem
    from ..kg_judge import CandJudge, _run
    from ..kg_teacher import CACHE
    d, rows = build(run)
    spec = load_annotators([judge_name])[judge_name]
    items = []
    if only is not None:
        rows = [(it, f) for it, f in rows if it["id"] in only]
    for it, facts in rows:
        if not facts:
            continue
        q = KGItem(id=f"{it['id']}:m1:{run}", tokens=it["tokens"], sentence_spans=it["sentence_spans"],
                   speakers=it.get("speakers"), participants=it["participants"], types=it["types"],
                   relations=it["relations"], domain=it.get("domain") or "")
        q.layers = {"judge_prompt": judge_prompt(it, facts)}
        items.append(q)
    print(f"{len(rows)} texts, {sum(len(f) for _, f in rows)} facts, {len(items)} calls", flush=True)
    ann = CandJudge(spec, CacheStore(CACHE, version))
    ann.prompt = PROMPTS[version]
    res = asyncio.run(_run(ann, items, f"m1-{judge_name}-{version}"))
    out = []
    for it, facts in rows:
        r = res.get(f"{it['id']}:m1:{run}")
        v = r.payload if r is not None and r.ok else {}
        out.append({"id": it["id"], "judged": bool(v) or not facts,
                    "facts": [{"key": [sorted(f["key"][0]), f["key"][1], sorted(f["key"][2])],
                               "src": f["src"], "verdict": v.get(n)} for n, f in enumerate(facts)]})
    tag = "" if version == "c1" else f"-{version}"
    tag += "-handcheck" if only is not None else ""
    (OUT / f"labels-{run}-{judge_name}{tag}.json").write_text(json.dumps({"configs": list(d["preds"]),
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


def labels_path(run: str, judge_name: str, version: str = "c1"):
    return OUT / f"labels-{run}-{judge_name}{'' if version == 'c1' else '-' + version}.json"


def score(run: str, judge_name: str, boot: int = 2000, merge: bool = False, version: str = "c1") -> int:
    L = json.loads(labels_path(run, judge_name, version).read_text())
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
    (OUT / f"score-{run}-{judge_name}{'' if version == 'c1' else '-' + version}"
           f"{'-merged' if merge else ''}.json").write_text(json.dumps(res, indent=1, default=float))
    return 0


def rescore(run: str, judge_name: str, tag: str, base_tag: str, boot: int = 2000,
            version: str = "c1") -> int:
    """Score another predictions file (``predict --tag``) against the judged
    facts without new judge calls: valid when its predictions are a subset of
    the judged ones (e.g. the variant merge keeps one of each judged group).
    Each prediction takes the verdict of the judged fact with the same
    mentions, else of one sharing a mention (as the judge pass matched);
    recall counts the variant groups found against the pooled ones."""
    L = json.loads(labels_path(run, judge_name, version).read_text())
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
    (OUT / f"rescore-{run}{tag}{'' if version == 'c1' else '-' + version}.json").write_text(json.dumps(out, indent=1, default=float))
    return 0


def agree(run: str, judge_name: str, version: str) -> int:
    """A judge's verdicts on the hand-checked sample against the hand labels
    (``hand-check.json``: 200 facts, half judged expressed by c1)."""
    H = json.loads((OUT / "hand-check.json").read_text())["facts"]
    tag = "" if version == "c1" else f"-{version}-handcheck"
    L = json.loads((OUT / f"labels-{run}-{judge_name}{tag}.json").read_text())
    by = {t["id"]: t for t in L["texts"]}
    c = Counter()
    for h in H:
        t = by.get(h["id"])
        f = next((f for f in t["facts"] if f["key"] == h["key"]), None) if t else None
        if f is None or f["verdict"] is None:
            c["missing"] += 1
            continue
        j = f["verdict"] in POSITIVE
        c[(h["hand_expressed"], j)] += 1
        c[("half", h["i"] < 100, h["hand_expressed"], j)] += 1
    for name, sel in (("all", None), ("first half", True), ("second half", False)):
        g = (lambda a, b: c[(a, b)]) if sel is None else (lambda a, b: c[("half", sel, a, b)])
        tp, fp, fn, tn = g(True, True), g(False, True), g(True, False), g(False, False)
        n = tp + fp + fn + tn
        print(f"{version} {name:11s}: {n} facts | agreement {(tp + tn) / max(n, 1):.3f} | "
              f"judge 'expressed' right {tp / max(tp + fp, 1):.3f} ({tp}/{tp + fp}) | "
              f"hand-expressed found {tp / max(tp + fn, 1):.3f} ({tp}/{tp + fn})")
    return 0


def combine(name: str, specs: list[str], tag: str = "-n500") -> int:
    """One predictions file with each run's base config as its own config
    (``label=run``, baseline first), so ``judge`` and ``score`` pool and
    pair the runs on the same texts."""
    items, preds, margins = None, {}, {}
    for spec in specs:
        label, run = spec.split("=")
        d = pickle.loads((OUT / f"preds-{run}{tag}.pkl").read_bytes())
        if items is None:
            items = d["items"]
        assert [it["id"] for it in d["items"]] == [it["id"] for it in items], f"{run}: other texts"
        preds[label], margins[label] = d["preds"]["base"], d["margins"]["base"]
        print(f"{label}: {sum(map(len, preds[label]))} facts")
    (OUT / f"preds-{name}.pkl").write_bytes(pickle.dumps({"run": name, "items": items,
                                                         "preds": preds, "margins": margins}))
    print("wrote", OUT / f"preds-{name}.pkl")
    return 0


def equal(run: str, judge_name: str, version: str, boot: int = 2000) -> int:
    """Each configuration at the first one's operating point: judged precision
    at its pooled recall and recall at its precision (each run's margin cut
    swept over its own judged facts), a paired bootstrap at equal recall with
    the cuts fixed on the full set, and false facts per 100 judged texts
    ("not stated" as a rate, plan r4 section 10). Predictions are counted one
    by one, so a fact two predictions match counts twice."""
    P = pickle.loads((OUT / f"preds-{run}.pkl").read_bytes())
    L = json.loads(labels_path(run, judge_name, version).read_text())
    cfgs, texts = L["configs"], L["texts"]
    keep = np.array([i for i, t in enumerate(texts) if t["judged"]])
    rows = {c: [] for c in cfgs}
    total = 0
    for i in keep:
        t = texts[i]
        total += sum(f["verdict"] in POSITIVE for f in t["facts"])
        for c in cfgs:
            for k, m in P["margins"][c][i].items():
                f = next((f for f in t["facts"] if c in f["src"]
                          and _same((set(f["key"][0]), f["key"][1], set(f["key"][2])), k)), None)
                rows[c].append((i, m, f is not None and f["verdict"] in POSITIVE))

    def curve(c):
        r = sorted(rows[c], key=lambda x: -x[1])
        ok = np.cumsum([x[2] for x in r])
        return np.array([x[1] for x in r]), ok / np.arange(1, len(r) + 1), ok / total

    base = cfgs[0]
    _, p0, r0 = curve(base)
    P0, R0 = p0[-1], r0[-1]
    print(f"{len(keep)} judged texts, pool {total}; baseline {base}: P {P0:.3f} R {R0:.3f}\n")
    print("| run | predicted | P | R | P at base R | R at base P |\n|---|---|---|---|---|---|")
    cuts = {}
    for c in cfgs:
        m, p, r = curve(c)
        j = np.searchsorted(r, R0)
        cuts[c] = m[min(j, len(m) - 1)]
        p_at = f"{p[j]:.3f}" if j < len(r) else f"(max R {r[-1]:.3f})"
        ok = np.where(p >= P0)[0]
        print(f"| {c} | {len(m)} | {p[-1]:.3f} | {r[-1]:.3f} | {p_at} | "
              f"{f'{r[ok].max():.3f}' if len(ok) else '-'} |")

    def per_text(c, cut):
        a = np.zeros((len(texts), 2))
        for i, m, ok in rows[c]:
            if m >= cut:
                a[i] += [ok, 1]
        return a[keep]

    A = {c: per_text(c, cuts[c]) for c in cfgs}
    rng = np.random.default_rng(13)
    pairs = [(c, base) for c in cfgs[1:]] + [(cfgs[j], cfgs[1]) for j in range(2, len(cfgs))]
    draws = [rng.integers(0, len(keep), len(keep)) for _ in range(boot)]
    print("\nprecision at the baseline's recall, paired bootstrap over texts")
    for a, b in pairs:
        d = [A[a][ix].sum(0) for ix in draws], [A[b][ix].sum(0) for ix in draws]
        d = [x[0] / x[1] - y[0] / y[1] for x, y in zip(*d)]
        x, y = A[a].sum(0), A[b].sum(0)
        print(f"  {a} - {b}: {100 * (x[0] / x[1] - y[0] / y[1]):+.1f} "
              f"[{100 * np.percentile(d, 2.5):+.1f}, {100 * np.percentile(d, 97.5):+.1f}]")
    print("\nfalse facts per 100 judged texts (deployed bar / at the baseline's recall)")
    for c in cfgs:
        dep = per_text(c, 0.0)
        print(f"  {c}: {100 * (dep[:, 1] - dep[:, 0]).mean():.1f} / "
              f"{100 * (A[c][:, 1] - A[c][:, 0]).mean():.1f}")
    for a, b in pairs:
        fa, fb = A[a][:, 1] - A[a][:, 0], A[b][:, 1] - A[b][:, 0]
        d = [100 * (fa[ix] - fb[ix]).mean() for ix in draws]
        print(f"  {a} - {b} at equal recall: {100 * (fa - fb).mean():+.1f} "
              f"[{np.percentile(d, 2.5):+.1f}, {np.percentile(d, 97.5):+.1f}]")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["predict", "judge", "score", "rescore", "agree", "combine", "equal"])
    ap.add_argument("--run", default="ultrabert-380m-v7b8ev")
    ap.add_argument("--short", type=int, default=300, help="texts in the slice (0: all)")
    ap.add_argument("--set", default="short",
                    help="M1 slice: short (dev), conll04 (test), m4 (business texts), or train:OFFSET "
                         "(short training texts outside the held-out fifth, D1e)")
    ap.add_argument("--floor", type=float, default=0.0,
                    help="predict: keep candidates down to this far under the bar, two per pair (D1e)")
    ap.add_argument("--configs", default="base,ner+np:1,ner+np+nest:2")
    ap.add_argument("--runs", help="combine: label=run,... (baseline first), each run's --tag file")
    ap.add_argument("--judge", default="astra")
    ap.add_argument("--tag", default="", help="predict / rescore: suffix of the predictions file")
    ap.add_argument("--base-tag", default="-merge", help="rescore: the file whose first config is the baseline")
    ap.add_argument("--merge", action="store_true", help="span variants of one fact count once")
    ap.add_argument("--version", default="c1", choices=list(PROMPTS), help="judge prompt")
    ap.add_argument("--handcheck", action="store_true",
                    help="judge: only the texts of the hand-checked sample (hand-check.json)")
    a = ap.parse_args()
    if a.cmd == "predict":
        predict(a.run, a.short, a.configs.split(","), a.tag, a.set, a.floor)
        return 0
    if a.cmd == "combine":
        return combine(a.run, a.runs.split(","), a.tag or "-n500")
    if a.cmd == "equal":
        return equal(a.run, a.judge, a.version)
    a.run = set_key(a.run, a.set)      # every later step reads files by slice
    if a.tag and a.cmd in ("judge", "score"):
        a.run += a.tag                 # a tagged predictions file judged as its own set
    if a.cmd == "judge":
        only = ({x["id"] for x in json.loads((OUT / "hand-check.json").read_text())["facts"]}
                if a.handcheck else None)
        return judge(a.run, a.judge, a.version, only)
    if a.cmd == "agree":
        return agree(a.run, a.judge, a.version)
    if a.cmd == "rescore":
        return rescore(a.run, a.judge, a.tag, a.base_tag, version=a.version)
    return score(a.run, a.judge, merge=a.merge, version=a.version)


if __name__ == "__main__":
    raise SystemExit(main())

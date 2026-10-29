"""GLiREL and GLiNER2 on kniv's relation benchmarks, same items and label
names as kniv (internal comparison: GLiREL's weights are not licence-cleared
for anything else).

    uv run python -m v6.train.rel_compare --parts fewrel,redocred --models glirel,gliner2

**FewRel zero-shot** (``zs_protocol``'s items and splits: the label bank's 15
unseen relations, 700 instances each, m = 5 / 10 over 5 seeds and m = 15):
each instance classified among the split's m relation names, macro F1.
GLiREL gets the head and tail occurrences; its answer is the label with the
best head -> tail score. GLiNER2 gets only the sentence: its answer is the
best-scoring label whose extracted pair overlaps the gold head and tail (head
-> tail); no such pair is a wrong answer.

**Re-DocRED** (test, documents up to 512 words; dev for thresholds), the 96
relation names (GLiREL all in one pass, GLiNER2 16 per pass):
- GLiREL, gold entities (every mention; an entity pair's score is its best
  mention pair's): triple F1 at a global threshold chosen on dev and at
  per-relation thresholds chosen on dev (kniv's ``rel_f1_cal`` /
  ``rel_f1_relcal`` procedure), and at the best global threshold on test;
- GLiNER2, end to end (its own spans): a predicted span stands for the gold
  entity with that exact mention span (``exact``) or whose mention texts
  include it, case-folded (``text``); unmatched spans make false facts.

Writes ``runs/rel-compare/<time>/results.json``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..config import RUNS_DIR
from .gold_eval import MIN_DEV_FACTS
from .zs_protocol import macro

ROOT = Path(__file__).resolve().parents[1]
ENVS = {
    "glirel": ["--with", "glirel", "--with", "loguru", "--with", "torch", "--with", "huggingface_hub==0.24.7",
               "--with", "transformers<4.46", "--with", "protobuf", "--with", "sentencepiece"],
    "gliner2": ["--with", "gliner2", "--with", "torch", "--with", "transformers", "--with", "sentencepiece"],
}
MODELS = {"glirel": "jackboyla/glirel-large-v0", "gliner2": "fastino/gliner2-large-v1"}
DEVICE = {"glirel": "mps", "gliner2": "mps"}
LABEL_GROUP = {"glirel": 96, "gliner2": 16}   # GLiREL ranks best with every label in one pass
THRESHOLDS = np.round(np.arange(0.02, 1.0, 0.02), 2)


def run_baseline(kind: str, jobs: list[dict], out: Path, tag: str, **extra) -> list:
    spec = {"kind": kind, "model": MODELS[kind], "device": DEVICE[kind], "batch": 16 if kind == "glirel" else 32, "jobs": jobs, **extra}
    src, dst = out / f"{tag}-{kind}-in.json", out / f"{tag}-{kind}-out.json"
    if not dst.exists():
        src.write_text(json.dumps(spec))
        subprocess.run(["uv", "run", "--no-project", "--python", "3.11", *ENVS[kind], "python", "-I",
                        str(ROOT / "rel_baselines.py"), str(src), str(dst)], check=True)
    return json.loads(dst.read_text())["preds"]


def _overlaps(a: int, b: int, spans) -> bool:
    return any(x < b and a < y for x, y in spans)


# ── FewRel ──────────────────────────────────────────────────────────
def fewrel(kinds: list[str], out: Path, seeds: int, per_relation: int) -> dict:
    from ..gold import fewrel as FR
    from ..label_bank import load as load_bank
    bank = load_bank()
    unseen = [r for r in bank["relations"] if not r.get("train", True)]
    code = {r["code"]: r["id"] for r in unseen}
    name = {r["id"]: r["name"] for r in unseen}
    per, items = {}, []
    for x in FR.load("train") + FR.load("val"):
        rid = code.get(x["code"])
        if rid is None or per.get(rid, 0) >= per_relation:
            continue
        per[rid] = per.get(rid, 0) + 1
        items.append({"tokens": x["tokens"], "head": x["head"], "tail": x["tail"], "rel": rid})
    ids = sorted(code.values())
    rng = np.random.default_rng(13)                       # zs_protocol's splits
    splits = {f"m={m}": [sorted(rng.choice(ids, size=m, replace=False).tolist()) for _ in range(seeds)]
              for m in (5, 10) if m < len(ids)}
    splits[f"m={len(ids)}"] = [ids]
    jobs, where = [], []
    for sname, ss in splits.items():
        for si, cols in enumerate(ss):
            for i, x in enumerate(items):
                if x["rel"] in cols:
                    jobs.append({"tokens": x["tokens"], "labels": [name[c] for c in cols],
                                 "mentions": [[a, b, 0] for a, b in x["head"]] + [[a, b, 1] for a, b in x["tail"]]})
                    where.append((sname, si, i))
    print(f"fewrel: {len(items):,} instances, {len(jobs):,} classifications", flush=True)
    by_name = {v: k for k, v in name.items()}
    res = {}
    for kind in kinds:
        preds = run_baseline(kind, jobs, out, "fewrel")
        runs = defaultdict(lambda: ([], []))
        no_answer = 0
        for (sname, si, i), p in zip(where, preds):
            x = items[i]
            if kind == "glirel":
                sc = {lab: s for h, t, lab, s in p if (h, t) == (0, 1)}
            else:
                sc = {}
                for lab, hs, he, ts, te, s in p:
                    if _overlaps(hs, he, x["head"]) and _overlaps(ts, te, x["tail"]):
                        sc[lab] = max(sc.get(lab, 0.0), s)
            ans = by_name[max(sc, key=sc.get)] if sc else -1
            no_answer += not sc
            g, q = runs[(sname, si)]
            g.append(x["rel"]); q.append(ans)
        res[kind] = {}
        for sname, ss in splits.items():
            r = [macro(*runs[(sname, si)], cols) for si, cols in enumerate(ss)]
            res[kind][sname] = {k: (float(np.mean([z[k] for z in r])), float(np.std([z[k] for z in r])))
                                for k in ("p", "r", "f1", "acc")}
            f = res[kind][sname]["f1"]
            print(f"FEWREL {kind} {sname} ({len(ss)} seeds): macro F1 {f[0]:.4f} +- {f[1]:.4f}; "
                  f"accuracy {res[kind][sname]['acc'][0]:.4f}", flush=True)
        res[kind]["no_answer"] = no_answer / len(jobs)
        print(f"FEWREL {kind}: no answer for {no_answer / len(jobs):.3f} of classifications", flush=True)
    return res


# ── Re-DocRED ───────────────────────────────────────────────────────
def _f1(tp, p, g):
    return 2 * tp / max(p + g, 1)


def _fit(cands: list[dict], gold: list[set], names: list[str]) -> tuple[float, dict]:
    """Global threshold, then per relation by coordinate ascent (relations with
    fewer than MIN_DEV_FACTS dev facts keep the global one) -- kniv's dev
    calibration on scores in [0, 1]."""
    G = sum(len(g) for g in gold)
    gold_n = defaultdict(int)
    for g in gold:
        for _, _, r in g:
            gold_n[r] += 1
    rows = defaultdict(list)                              # relation -> [(score, is_gold)]
    for c, g in zip(cands, gold):
        for (h, t, r), s in c.items():
            rows[r].append((s, (h, t, r) in g))
    rows = {r: np.array(v, dtype=np.float64).reshape(-1, 2) for r, v in rows.items()}

    def counts(r, th):
        m = rows.get(r)
        if m is None:
            return 0.0, 0.0
        hit = m[:, 0] >= th
        return float((hit & (m[:, 1] > 0)).sum()), float(hit.sum())

    glob = max(THRESHOLDS, key=lambda th: _f1(*map(sum, zip(*(counts(r, th) for r in names))), G))
    per = {r: glob for r in names}
    cur = {r: counts(r, glob) for r in names}
    for _ in range(3):
        for r in names:
            if gold_n[r] < MIN_DEV_FACTS:
                continue
            TP = sum(c[0] for c in cur.values()) - cur[r][0]
            P = sum(c[1] for c in cur.values()) - cur[r][1]
            best = max(THRESHOLDS, key=lambda th: _f1(TP + counts(r, th)[0], P + counts(r, th)[1], G))
            per[r], cur[r] = float(best), counts(r, best)
    return float(glob), per


def _score(cands, gold, th) -> float:
    tp = p = 0
    for c, g in zip(cands, gold):
        pred = {k for k, s in c.items() if s >= (th[k[2]] if isinstance(th, dict) else th)}
        tp += len(pred & g); p += len(pred)
    return _f1(tp, p, sum(len(g) for g in gold))


def redocred(kinds: list[str], out: Path, limit: int | None = None) -> dict:
    from ..gold.redocred import load_rel_items, relation_inventory
    names, _ = relation_inventory()
    data = {s: load_rel_items(s, limit=limit) for s in ("dev", "test")}
    print(f"redocred: {len(data['dev'])} dev / {len(data['test'])} test documents, {len(names)} relations",
          flush=True)
    res = {}
    for kind in kinds:
        n = LABEL_GROUP[kind]
        groups = [names[k:k + n] for k in range(0, len(names), n)]
        cands = {}
        for split, items in data.items():
            jobs = [{"tokens": it.tokens, "labels": g,
                     "mentions": [[a, e + 1, en["id"]] for en in it.entities for a, e in en["mentions"]]}
                    for it in items for g in groups]
            preds = run_baseline(kind, jobs, out, f"redocred-{split}", min_score=0.02)
            per_doc = [preds[k * len(groups):(k + 1) * len(groups)] for k in range(len(items))]
            cands[split] = {}
            for mode in (("gold",) if kind == "glirel" else ("exact", "text")):
                cs = []
                for it, ps in zip(items, per_doc):
                    c: dict[tuple, float] = {}
                    for p in (x for grp in ps for x in grp):
                        if kind == "glirel":
                            h, t, lab, s = p
                        else:
                            lab, hs, he, ts, te, s = p
                            h, t = (_match(it, hs, he, mode), _match(it, ts, te, mode))
                            if h is None or t is None:
                                h, t = ("x", hs, he), ("x", ts, te)   # a false fact
                            if h == t:
                                continue
                        c[(h, t, lab)] = max(c.get((h, t, lab), 0.0), s)
                    cs.append(c)
                cands[split][mode] = cs
        gold = {s: [{tuple(x) for x in it.layers["rel"]} for it in items] for s, items in data.items()}
        res[kind] = {}
        for mode in cands["test"]:
            glob, per = _fit(cands["dev"][mode], gold["dev"], names)
            oracle = max(THRESHOLDS, key=lambda th: _score(cands["test"][mode], gold["test"], th))
            r = {"dev_global": glob, "f1_cal": _score(cands["test"][mode], gold["test"], glob),
                 "f1_relcal": _score(cands["test"][mode], gold["test"], per),
                 "f1_oracle_global": _score(cands["test"][mode], gold["test"], float(oracle)),
                 "oracle_threshold": float(oracle)}
            res[kind][mode] = r
            print(f"REDOCRED {kind} {mode}: global (dev {glob:.2f}) {r['f1_cal']:.4f}, per relation "
                  f"{r['f1_relcal']:.4f}, oracle global ({oracle:.2f}) {r['f1_oracle_global']:.4f}", flush=True)
    return res


def _match(it, a: int, b: int, mode: str):
    if mode == "exact":
        for en in it.entities:
            if any(x == a and e + 1 == b for x, e in en["mentions"]):
                return en["id"]
        return None
    text = " ".join(it.tokens[a:b]).casefold()
    for en in it.entities:
        if text in {s.casefold() for s in en["aliases"]} or \
                text in {" ".join(it.tokens[x:e + 1]).casefold() for x, e in en["mentions"]}:
            return en["id"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parts", default="fewrel,redocred")
    ap.add_argument("--models", default="glirel,gliner2")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--per-relation", type=int, default=700)
    ap.add_argument("--limit", type=int, help="Re-DocRED documents per split (smoke tests)")
    ap.add_argument("--out", type=Path, help="reuse a run directory (baseline outputs already there are kept)")
    args = ap.parse_args()
    out = args.out or RUNS_DIR / "rel-compare" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    kinds = args.models.split(",")
    res = {}
    for part in args.parts.split(","):
        res[part] = (fewrel(kinds, out, args.seeds, args.per_relation) if part == "fewrel"
                     else redocred(kinds, out, args.limit))
        (out / "results.json").write_text(json.dumps(res, indent=1))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

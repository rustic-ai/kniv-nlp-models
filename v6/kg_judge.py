"""Judged candidate facts on short texts (v8 plan r2: P0 fit test, the base
of M1 and D1).

    uv run python -m v6.kg_judge cands --runs ultrabert-380m-v7b8ev,...   # candidates, gold entities
    uv run python -m v6.kg_judge judge --train 300 --dev 100 [--limit 20] # one judge call per text
    uv run python -m v6.kg_judge show --n 40                              # verdicts to read by hand

"Verify, don't enumerate": the teacher lists one relation per pair and misses
facts, so its labels cannot say whether a candidate is wrong. Here every
candidate is judged on its own. Per text, the judge sees the union of

- the teacher's facts, and
- each run's candidates on the gold entities above a low bar (margin within
  ``BAR`` of the bar the fit diagnostic found best, top ``PER_PAIR`` per pair),

and gives each one verdict: supported, not_supported, negated, hedged. The
texts are the fit diagnostic's samples (``kg_short.json``, shuffled with seed
3): the first ``--train`` training texts and the first ``--dev`` dev texts.
Responses are cached by judge, layer and ``VERSION``; ``labels.json`` holds
each text's judged facts and their sources.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

from .annotate.base import CacheStore, RunLogger
from .annotate.kg import KGAnnotator, KGItem
from .config import DATA_DIR, RUNS_DIR, load_annotators
from .kg_teacher import CACHE
from .prompts import SYSTEM

VERSION = "c1"
OUT = RUNS_DIR / "kg-judge"
SEED, SAMPLE = 3, 700
BAR, PER_PAIR = -3.0, 2
UNSEEN_BAR = 1.75     # the fit diagnostic's best new-relation offset on short text
POSITIVE = ("supported", "negated", "hedged")   # the fact is expressed: the relation step should fire


def load_texts(split: str, n: int = SAMPLE) -> list[dict]:
    from .eval import heldout_zs as hz
    from .label_bank import load as load_bank
    bank = load_bank()
    rel = {r["name"]: r for r in bank["relations"]}
    typ = {t["name"]: t for t in bank.get("entity_types", [])}
    data = json.loads((DATA_DIR / "v6-corpus" / "kg_short.json").read_text())[split]
    random.Random(SEED).shuffle(data)
    items = []
    for x in data[:n]:
        p = hz._lit(x["payload"])
        items.append({
            "id": x["id"], "tokens": hz._lit(x["tokens"]), "sentence_spans": hz._lit(x["sentence_spans"]),
            "entities": p["entities"], "participants": list(p.get("participants") or []),
            "speakers": hz._lit(x.get("speakers")) or None,
            "facts": [{"head": f["head"], "tail": f["tail"], "relation": f["relation"],
                       "polarity": f.get("polarity", "positive")} for f in p["facts"]],
            "relations": [{"name": m, "description": rel.get(m, {}).get("description", "")}
                          for m in hz._lit(x["relations"])],
            "types": [{"name": m, "description": typ.get(m, {}).get("description", "")}
                      for m in hz._lit(x["types"])],
            "domain": x.get("domain")})
    return items


def cands_path(run: str) -> Path:
    return OUT / f"cands-{run}.pkl"


def make_cands(runs: list[str]) -> None:
    """Every candidate on the gold entities, per run (``heldout_zs.predict``
    with the bar moved out of the way), for the train and dev samples."""
    from .eval import heldout_zs as hz
    from .infer import Extractor
    data = {"dev": load_texts("dev"), "train": load_texts("train")}
    OUT.mkdir(parents=True, exist_ok=True)
    for run in runs:
        if cands_path(run).exists():
            print(run, "cached")
            continue
        ex = Extractor(run)
        dep = ex.unseen_offset
        ex.rel_offsets = {k: v + hz.SHIFT for k, v in ex.rel_offsets.items()}
        ex.unseen_offset = dep - hz.SHIFT
        ex.top = 50
        rows = {s: hz.predict(ex, items, "gold") for s, items in data.items()}
        cands_path(run).write_bytes(pickle.dumps((dep, rows)))
        print(run, "done", flush=True)
        del ex


def facts_to_judge(row: dict, runs_rows: dict, i: int) -> list[dict]:
    """The teacher's facts, then each run's candidates near the bar."""
    out, seen = [], {}
    for f in row["facts"]:
        k = (f["head"], f["relation"], f["tail"])
        if f["head"] == f["tail"] or k in seen:
            continue
        seen[k] = len(out)
        out.append({"head": f["head"], "relation": f["relation"], "tail": f["tail"], "src": ["teacher"]})
    for run, rows in runs_rows.items():
        bypair = defaultdict(list)
        for c in rows[i]["cand"]:
            th = UNSEEN_BAR if c["unseen"] else 0.0
            if c["m"] - th > BAR and c["key"][0] != c["key"][2]:
                bypair[(c["key"][0], c["key"][2])].append(c)
        for cs in bypair.values():
            for c in sorted(cs, key=lambda c: -c["m"])[:PER_PAIR]:
                k = tuple(c["key"])
                if k not in seen:
                    seen[k] = len(out)
                    out.append({"head": k[0], "relation": k[1], "tail": k[2], "src": []})
                out[seen[k]]["src"].append(run)
    return out


def name(it: dict, i: int) -> str:
    k = len(it["participants"])
    if i < k:
        return it["participants"][i]
    ms = []
    for a, b in it["entities"][i - k]["spans"]:
        m = " ".join(it["tokens"][a:b])
        if m not in ms:
            ms.append(m)
    return " / ".join(ms[:3])


def judge_prompt(it: dict, facts: list[dict]) -> str:
    defs = {r["name"]: r["description"] for r in it["relations"]}
    lines = ["Passage:"]
    for j, (s, e) in enumerate(it["sentence_spans"]):
        who = f"[{it['speakers'][j]}] " if it.get("speakers") and it["speakers"][j] else ""
        lines.append(f"{j + 1}\t{who}{' '.join(it['tokens'][s:e])}")
    lines += ["", "Facts:"]
    for n, f in enumerate(facts):
        lines.append(f"{n}\t[{name(it, f['head'])}] {f['relation']} [{name(it, f['tail'])}]"
                     f"  ({f['relation']}: {defs.get(f['relation'], '')})")
    return "\n".join(lines)


class CandJudge(KGAnnotator):
    def build_messages(self, layer: str, item) -> list[dict]:
        return [{"role": "system", "content": SYSTEM["kgcand"]},
                {"role": "user", "content": item.layers["judge_prompt"] + self._json_hint(layer)}]


async def _run(ann, items: list, tag: str) -> dict:
    logger = RunLogger(OUT / tag, every=25)
    out = {}
    try:
        for coro in asyncio.as_completed([ann.annotate_and_cache("kgcand", it) for it in items]):
            r = await coro
            logger.record(r, (ann.spec.price_in, ann.spec.price_out), len(items))
            out[r.item_id] = r
    finally:
        logger.close()
    return out


def build(runs: list[str], n_train: int, n_dev: int) -> list[tuple[str, dict, list[dict]]]:
    loaded = {r: pickle.loads(cands_path(r).read_bytes())[1] for r in runs}
    out = []
    for split, n in (("train", n_train), ("dev", n_dev)):
        texts = load_texts(split)
        for i, it in enumerate(texts[:n]):
            rr = {r: loaded[r][split] for r in runs}
            for r in runs:
                assert rr[r][i]["item"]["id"] == it["id"], "candidate file is for another sample"
            out.append((split, it, facts_to_judge(it, rr, i)))
    return out


def judge(runs: list[str], judge_name: str, n_train: int, n_dev: int, limit: int | None) -> int:
    rows = build(runs, n_train, n_dev)
    if limit:
        rows = rows[:limit // 2] + [r for r in rows if r[0] == "dev"][:limit - limit // 2]
    spec = load_annotators([judge_name])[judge_name]
    cache = CacheStore(CACHE, VERSION)
    items = []
    for split, it, facts in rows:
        if not facts:
            continue
        q = KGItem(id=f"{it['id']}:cand", tokens=it["tokens"], sentence_spans=it["sentence_spans"],
                   speakers=it.get("speakers"), participants=it["participants"], types=it["types"],
                   relations=it["relations"], domain=it.get("domain") or "")
        q.layers = {"judge_prompt": judge_prompt(it, facts)}
        items.append(q)
    print(f"{len(rows)} texts, {sum(len(f) for _, _, f in rows)} facts to judge, "
          f"{len(items)} calls (cached ones are free)", flush=True)
    res = asyncio.run(_run(CandJudge(spec, cache), items, f"judge-{judge_name}-{VERSION}"))
    labels, c = [], Counter()
    for split, it, facts in rows:
        r = res.get(f"{it['id']}:cand")
        v = r.payload if r is not None and r.ok else {}
        for n, f in enumerate(facts):
            f["verdict"] = v.get(n)
            c[f"{split}:{f['verdict']}"] += 1
            if "teacher" in f["src"]:
                c[f"{split}:teacher:{f['verdict']}"] += 1
        labels.append({"split": split, "id": it["id"], "facts": facts, "judged": bool(v)})
    tag = f"labels-{judge_name}-{VERSION}" + (f"-limit{limit}" if limit else "")
    (OUT / f"{tag}.json").write_text(json.dumps(labels))
    for split in ("train", "dev"):
        tot = sum(v for k, v in c.items() if k.startswith(f"{split}:") and k.count(":") == 1)
        th = sum(v for k, v in c.items() if k.startswith(f"{split}:teacher:"))
        print(f"{split}: {tot} facts:", {k.split(":")[1]: v for k, v in c.items()
                                          if k.startswith(f"{split}:") and k.count(":") == 1},
              f"| teacher facts {th}: supported {c[f'{split}:teacher:supported'] / max(th, 1):.1%}")
    print(f"written {OUT / tag}.json")
    return 0


def show(path: Path, n: int, seed: int = 0) -> int:
    labels = json.loads(path.read_text())
    texts = {s: {t["id"]: t for t in load_texts(s)} for s in ("train", "dev")}
    rng = random.Random(seed)
    rows = rng.sample(labels, min(n, len(labels)))
    for r in rows:
        it = texts[r["split"]][r["id"]]
        print("=" * 80)
        print(" ".join(" ".join(it["tokens"][s:e]) for s, e in it["sentence_spans"]))
        for f in r["facts"]:
            src = "T" if "teacher" in f["src"] else " "
            print(f"  {src} {str(f['verdict']):13s} [{name(it, f['head'])}] {f['relation']} "
                  f"[{name(it, f['tail'])}]")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["cands", "judge", "show"])
    ap.add_argument("--runs", default="ultrabert-380m-v7b8ev,ultrabert-380m-v8e-w1half,ultrabert-380m-v8e-w1b")
    ap.add_argument("--judge", default="astra")
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--dev", type=int, default=100)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    runs = a.runs.split(",")
    if a.cmd == "cands":
        make_cands(runs)
        return 0
    if a.cmd == "judge":
        return judge(runs, a.judge, a.train, a.dev, a.limit)
    return show(a.labels, a.n, a.seed)


if __name__ == "__main__":
    raise SystemExit(main())

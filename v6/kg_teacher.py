"""The knowledge-layer teacher: sample a schema per window, annotate, measure.

    uv run python -m v6.kg_teacher pilot --windows 500          # corpus pilot
    uv run python -m v6.kg_teacher bakeoff --docs 100           # quality on Re-DocRED dev
    uv run python -m v6.kg_teacher run --second 0.3             # full corpus (after the gates)

Each window gets its own schema, sampled deterministically from the label
bank so the head learns to READ label descriptions rather than memorise a
fixed inventory:

- relations, 8-40: half drawn by a domain prior (general relations proposed
  in that domain; Re-DocRED/FewRel relations for encyclopedic, news,
  business and narrative text), a third at random, the rest hard negatives
  from the same category as the prior's picks;
- entity types: either DocRED's coarse six or a sample of Few-NERD's fine
  66, never both (an entity would then have two right answers), always with
  "time" and "number" so facts can be anchored.

A second, disjoint relation schema is drawn for ``--second`` of the windows.

The general relations are the reviewed candidates in
``general_relations.json`` (``LABEL_BANK_REVIEW.md``); the bake-off and the
pilot measure the teacher, they teach nothing.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .annotate.base import CacheStore, RunLogger
from .annotate.kg import KGAnnotator, KGItem
from .config import DATA_DIR, RUNS_DIR, load_annotators
from .label_bank import load as load_bank
from .prompts import PROMPT_VERSION

GENERAL = DATA_DIR / "v6-corpus" / "general_relations.json"
SPEAKERS = DATA_DIR / "v6-corpus" / "speakers.json"
import os
CACHE = Path(os.environ["KG_CACHE"]) if os.environ.get("KG_CACHE") else RUNS_DIR / "_cache"
WIKI_DOMAINS = {"encyclopedic", "news", "business", "narrative"}


def _rng(key: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha1(key.encode()).hexdigest()[:12], 16))


class SchemaSampler:
    def __init__(self, bank: dict, general: list[dict]):
        self.types = bank["entity_types"]
        self.coarse = [t for t in self.types if t["source"] == "docred"]
        self.fine = [t for t in self.types if t["source"] == "fewnerd"]
        self.always = [t for t in self.coarse if t["code"] in ("TIME", "NUM")]
        wiki = [dict(r, category="wikidata") for r in bank["relations"] if r.get("train", True)]
        gen = [{"name": r["name"], "description": r["description"], "aliases": r["aliases"],
                "category": r["category"] or "other", "domains": r.get("domains", {}),
                "support": r.get("support", 1), "source": "general"} for r in general]
        self.relations = wiki + gen
        names = [r["name"] for r in self.relations]
        if len(set(names)) != len(names):            # a general name colliding with Wikidata's
            seen, keep = set(), []
            for r in self.relations:
                if r["name"] not in seen:
                    seen.add(r["name"]); keep.append(r)
            self.relations = keep
        self.by_cat = defaultdict(list)
        for i, r in enumerate(self.relations):
            self.by_cat[r["category"]].append(i)

    def prior(self, domain: str) -> np.ndarray:
        w = np.array([float(r.get("domains", {}).get(domain, 0)) if r["source"] == "general"
                      else (1.0 if domain in WIKI_DOMAINS else 0.0) for r in self.relations])
        return w + 1e-3

    def relations_for(self, key: str, domain: str, exclude: set[int] = frozenset()) -> list[int]:
        rng = _rng(f"rel:{key}")
        n = int(rng.integers(8, 41))
        pick: list[int] = []
        avail = np.array([i not in exclude for i in range(len(self.relations))])

        def draw(weights, k):
            w = weights * avail
            for i in pick:
                w[i] = 0
            k = min(k, int((w > 0).sum()))
            if k <= 0:
                return
            pick.extend(rng.choice(len(w), size=k, replace=False, p=w / w.sum()).tolist())

        draw(self.prior(domain), round(n * 0.5))
        neigh = np.zeros(len(self.relations))
        for i in list(pick):
            for j in self.by_cat[self.relations[i]["category"]]:
                neigh[j] = 1.0
        draw(neigh, round(n * 0.2))
        draw(np.ones(len(self.relations)), n - len(pick))
        return sorted(pick)

    def types_for(self, key: str) -> list[dict]:
        rng = _rng(f"type:{key}")
        if rng.random() < 0.5:
            chosen = [t for t in self.coarse if t["code"] not in ("TIME", "NUM")]
        else:
            k = int(rng.integers(5, 15))
            chosen = [self.fine[i] for i in sorted(rng.choice(len(self.fine), k, replace=False))]
        return chosen + self.always


def participants_of(speakers: list[str | None] | None) -> list[str]:
    out = []
    for s in speakers or []:
        if s and s not in out:
            out.append(s)
    return out


def make_item(window_id: str, tokens, spans, domain: str, speakers, sampler: SchemaSampler,
              schema_no: int = 0, exclude: set[int] = frozenset(),
              rel_ids: list[int] | None = None, types: list[dict] | None = None) -> KGItem:
    key = f"{window_id}:{schema_no}"
    rel_ids = rel_ids if rel_ids is not None else sampler.relations_for(key, domain, exclude)
    types = types if types is not None else sampler.types_for(key)
    rels = [sampler.relations[i] for i in rel_ids]
    digest = hashlib.sha1(json.dumps([[r["name"] for r in rels], [t["name"] for t in types]]).encode()
                          ).hexdigest()[:10]
    return KGItem(id=f"{window_id}:s{schema_no}:{digest}", tokens=tokens,
                  sentence_spans=spans, speakers=speakers,
                  participants=participants_of(speakers), types=types,
                  relations=[{"name": r["name"], "description": r["description"]} for r in rels],
                  domain=domain)


async def annotate(items: list[KGItem], annotator: str, tag: str) -> list:
    spec = load_annotators([annotator])[annotator]
    ann = KGAnnotator(spec, CacheStore(CACHE, PROMPT_VERSION))
    logger = RunLogger(RUNS_DIR / f"kg-{tag}-{annotator}", every=50)
    out = []
    try:
        first = await ann.annotate_and_cache("kg", items[0])
        logger.record(first, (0, 0), len(items)); out.append(first)
        for coro in asyncio.as_completed([ann.annotate_and_cache("kg", it) for it in items[1:]]):
            res = await coro
            logger.record(res, (0, 0), len(items)); out.append(res)
    finally:
        logger.close()
    return out


def report(results, items: list[KGItem]) -> dict:
    by_id = {it.id: it for it in items}
    agg = defaultdict(Counter)
    for res in results:
        it = by_id[res.item_id]
        c = agg[it.domain]
        c["windows"] += 1
        if not res.ok:
            c["failed"] += 1
            continue
        p = res.payload
        c["mentions"] += p["stats"]["mentions"]; c["aligned"] += p["stats"]["aligned"]
        c["entities"] += len(p["entities"]); c["facts"] += len(p["facts"])
        c["schema_relations"] += len(it.relations)
        for f in p["facts"]:
            c[f"pol:{f['polarity']}"] += 1; c[f"mod:{f['modality']}"] += 1
            c["holder_set"] += f["holder"] != -1; c["time_set"] += f["time"] != -1
            c["speaker_arg"] += f["head"] < len(it.participants) or f["tail"] < len(it.participants)
    tot = Counter()
    for c in agg.values():
        tot.update(c)
    agg["ALL"] = tot
    return {d: dict(c) for d, c in agg.items()}


def print_report(rep: dict) -> None:
    print(f"{'domain':13s} {'win':>4s} {'fail':>4s} {'align':>6s} {'ent/w':>6s} {'fact/w':>6s} "
          f"{'neg%':>5s} {'non-assert%':>11s} {'holder%':>7s} {'time%':>6s} {'spk-arg%':>8s}")
    for d, c in sorted(rep.items(), key=lambda x: (x[0] == "ALL", x[0])):
        ok = max(c.get("windows", 0) - c.get("failed", 0), 1)
        f = max(c.get("facts", 0), 1)
        print(f"{d:13s} {c.get('windows', 0):4d} {c.get('failed', 0):4d} "
              f"{c.get('aligned', 0) / max(c.get('mentions', 0), 1):6.1%} "
              f"{c.get('entities', 0) / ok:6.1f} {c.get('facts', 0) / ok:6.1f} "
              f"{c.get('pol:negative', 0) / f:5.1%} {1 - c.get('mod:asserted', 0) / f:11.1%} "
              f"{c.get('holder_set', 0) / f:7.1%} {c.get('time_set', 0) / f:6.1%} "
              f"{c.get('speaker_arg', 0) / f:8.1%}")


def corpus_items(n_windows: int | None, second: float, sampler: SchemaSampler,
                 split: str = "train") -> list[KGItem]:
    from .train.data import load_rows
    rows = load_rows(split, columns=["window_id", "domain", "tokens", "sentence_spans"])
    speakers = json.loads(SPEAKERS.read_text()) if SPEAKERS.exists() else {}
    if n_windows:
        by = defaultdict(list)
        for r in rows:
            by[r["domain"]].append(r)
        per = max(1, n_windows // len(by))
        rows = [r for d in sorted(by) for r in sorted(
            by[d], key=lambda r: hashlib.sha1(f"kgpilot:{r['window_id']}".encode()).hexdigest())[:per]]
    items = []
    for r in rows:
        sp = speakers.get(r["window_id"])
        it = make_item(r["window_id"], r["tokens"], r["sentence_spans"], r["domain"], sp, sampler)
        items.append(it)
        if _rng(f"second:{r['window_id']}").random() < second:
            used = {i for i, x in enumerate(sampler.relations) if x["name"] in {y["name"] for y in it.relations}}
            items.append(make_item(r["window_id"], r["tokens"], r["sentence_spans"], r["domain"], sp,
                                   sampler, schema_no=1, exclude=used))
    return items


def bakeoff(n_docs: int, annotator: str, sampler: SchemaSampler,
            gold_entities: bool = False) -> int:
    """Teacher quality where gold exists: Re-DocRED dev with sampled schemas of
    Re-DocRED relations and DocRED types. Entities match gold by an exact
    mention span; triple F1 is over the relations each document's schema
    listed."""
    from .gold.redocred import load_rel_items
    from .schemas import relation_names
    golds = load_rel_items("dev", max_tokens=10 ** 6)[:n_docs]
    wiki_idx = [i for i, r in enumerate(sampler.relations) if r["source"] == "redocred"]
    coarse = [t for t in sampler.coarse]
    items = []
    for g in golds:
        rng = _rng(f"bake:{g.id}")
        present = sorted({i for i in wiki_idx
                          if any(t[2] == sampler.relations[i]["name"] for t in g.layers["rel"])})
        k = int(rng.integers(8, 41))
        rel = set(rng.choice(present, size=min(len(present), k // 2), replace=False).tolist()) if present else set()
        rest = [i for i in wiki_idx if i not in rel]
        rel |= set(rng.choice(rest, size=k - len(rel), replace=False).tolist())
        it = make_item(g.id, g.tokens, g.layers["sentence_spans"], "encyclopedic", None,
                       sampler, rel_ids=sorted(rel), types=coarse)
        if gold_entities:
            # The entities are given (as participants 0..k-1, by their names),
            # so only relation quality is measured.
            it.participants = ["; ".join(e["aliases"][:3]) for e in g.entities]
            it.id += ":gold"
        items.append(it)
    results = asyncio.run(annotate(items, annotator, "bakeoff"))
    by_id = {r.item_id: r for r in results}
    tp = np_ = ng = men_tp = men_p = men_g = 0
    for it, g in zip(items, golds):
        res = by_id[it.id]
        names = {r["name"] for r in it.relations}
        gold = {(h, t, r) for h, t, r in g.layers["rel"] if r in names}
        gm = [{(a, e + 1) for a, e in en["mentions"]} for en in g.entities]
        men_g += sum(len(x) for x in gm)
        if not res.ok:
            ng += len(gold)
            continue
        p = res.payload
        if gold_entities:
            pred = {(f["head"], f["tail"], f["relation"]) for f in p["facts"]
                    if f["head"] < len(g.entities) and f["tail"] < len(g.entities)}
            tp += len(pred & gold); np_ += len(pred); ng += len(gold)
            continue
        pred_m = [{tuple(s) for s in e["spans"]} for e in p["entities"]]
        men_p += sum(len(x) for x in pred_m)
        men_tp += len(set().union(*pred_m) & set().union(*gm)) if pred_m and gm else 0
        match = {}
        for i, pm in enumerate(pred_m):
            ov = [len(pm & x) for x in gm]
            m = max(ov) if ov else 0
            match[i] = ov.index(m) if m and ov.count(m) == 1 else None
        pred = set()
        for f in p["facts"]:
            h, t = match.get(f["head"]), match.get(f["tail"])
            if h is not None and t is not None and h != t:
                pred.add((h, t, f["relation"]))
        tp += len(pred & gold); np_ += len(p["facts"]); ng += len(gold)
    f1 = 2 * tp / max(np_ + ng, 1)
    mf1 = 2 * men_tp / max(men_p + men_g, 1)
    print(f"Re-DocRED dev, {len(items)} docs, sampled schemas: triple F1 {f1:.3f} "
          f"(P {tp / max(np_, 1):.3f} R {tp / max(ng, 1):.3f}); mention F1 {mf1:.3f}")
    return 0


KG_TEACHER = DATA_DIR / "v6-corpus" / "kg_teacher.json"


def export(items: list[KGItem], annotator: str, path: Path = KG_TEACHER) -> Counter:
    """The cached, aligned teacher payloads with what the trainer needs of
    their items, as one file a VM can read (the cache stays local). Items
    never annotated, or that failed, are left out and counted."""
    cache = CacheStore(CACHE, PROMPT_VERSION)
    out, c = [], Counter()
    for it in items:
        rec = cache.get(annotator, "kg", it.id)
        p = (rec or {}).get("payload")
        if not isinstance(p, dict) or "stats" not in p:
            c["missing"] += 1
            continue
        c["items"] += 1
        out.append({"id": it.id, "domain": it.domain, "tokens": it.tokens,
                    "sentence_spans": it.sentence_spans, "speakers": it.speakers,
                    "participants": it.participants,
                    "relations": [r["name"] for r in it.relations],
                    "types": [t["name"] for t in it.types],
                    "payload": {k: p[k] for k in ("participants", "entities", "facts")}})
    path.write_text(json.dumps({"prompt_version": PROMPT_VERSION, "annotator": annotator,
                                "items": out}))
    return c


def load_export(path: Path = KG_TEACHER) -> list[tuple[KGItem, dict]]:
    """``(item, payload)`` pairs from ``export``, for ``kdata.teacher_features``."""
    out = []
    for r in json.loads(path.read_text())["items"]:
        out.append((KGItem(id=r["id"], tokens=r["tokens"], sentence_spans=r["sentence_spans"],
                           speakers=r["speakers"], participants=r["participants"],
                           types=[{"name": t} for t in r["types"]],
                           relations=[{"name": n} for n in r["relations"]],
                           domain=r["domain"]), r["payload"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["pilot", "bakeoff", "run", "export"])
    ap.add_argument("--windows", type=int, default=500)
    ap.add_argument("--docs", type=int, default=100)
    ap.add_argument("--second", type=float, default=0.3)
    ap.add_argument("--annotator", default="astra")
    ap.add_argument("--gold-entities", action="store_true",
                    help="bakeoff: give the teacher Re-DocRED's entities; measure relations only")
    args = ap.parse_args()
    bank = load_bank()
    general = json.loads(GENERAL.read_text())["relations"] if GENERAL.exists() else []
    sampler = SchemaSampler(bank, general)
    print(f"schema pool: {len(sampler.relations)} relations, {len(sampler.types)} entity types", flush=True)
    if args.cmd == "bakeoff":
        return bakeoff(args.docs, args.annotator, sampler, args.gold_entities)
    if args.cmd == "export":
        items = corpus_items(args.windows or None, args.second, sampler)
        print(dict(export(items, args.annotator)), f"-> {KG_TEACHER}", flush=True)
        return 0
    items = corpus_items(args.windows if args.cmd == "pilot" else None, args.second, sampler)
    print(f"{len(items)} items over {len({i.id.split(':')[0] for i in items})} windows", flush=True)
    results = asyncio.run(annotate(items, args.annotator, args.cmd))
    rep = report(results, items)
    (RUNS_DIR / f"kg-{args.cmd}-report.json").write_text(json.dumps(rep, indent=1))
    print_report(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

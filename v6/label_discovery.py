"""Discover the relation types the corpus actually expresses, for the label bank.

    uv run python -m v6.label_discovery --per-domain 250      # ~1,500 teacher calls

Re-DocRED's 96 relations are Wikipedia's (citizenship, publication date). A
general-purpose extractor needs the relations conversation, technical,
business, news and fiction text expresses too. Rather than invent that list,
the teacher reads a deterministic sample of training windows from every
domain and names the relation types each states, in general terms.

The output, ``data/v6-corpus/label_discovery.json``, is raw material: every
proposed name with its frequency, domains, definitions and examples. Merging
it into a curated general bank is a separate, reviewed step -- the bank
decides what the model will be taught, so a person signs it off.

Training split only, so no dev or test text shapes the inventory.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .annotate.base import CacheStore, RunLogger
from .annotate.llm import LLMAnnotator
from .config import DATA_DIR, RUNS_DIR, load_annotators
from .prompts import PROMPT_VERSION

OUT = DATA_DIR / "v6-corpus" / "label_discovery.json"
GENERAL = DATA_DIR / "v6-corpus" / "general_relations.json"
REVIEW = DATA_DIR.parent / "v6" / "LABEL_BANK_REVIEW.md"
SPEAKERS = DATA_DIR / "v6-corpus" / "speakers.json"
CACHE = RUNS_DIR / "_cache"


@dataclass
class Item:
    id: str
    tokens: list[str]
    context: str | None = None
    target: str | None = None
    predicate_idx: int | None = None
    entities: list | None = None
    layers: dict = field(default_factory=dict)


def passage(row: dict, speakers: dict) -> str:
    """The window as text; conversation sentences carry their speaker."""
    sp = speakers.get(row["window_id"])
    out, last = [], None
    for j, (s, e) in enumerate(row["sentence_spans"]):
        text = " ".join(row["tokens"][s:e])
        who = sp[j] if sp and j < len(sp) else None
        if who and who != last:
            out.append(f"\n{who}: {text}")
            last = who
        else:
            out.append(text)
    return " ".join(out).strip()


def sample(per_domain: int) -> list[dict]:
    from .train.data import load_rows
    rows = load_rows("train", columns=["window_id", "domain", "tokens", "sentence_spans"])
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["domain"]].append(r)
    out = []
    for d in sorted(by):
        rs = sorted(by[d], key=lambda r: hashlib.sha1(f"discover:{r['window_id']}".encode()).hexdigest())
        out += rs[:per_domain]
    return out


async def run(items: list[Item], annotator: str) -> list:
    spec = load_annotators([annotator])[annotator]
    ann = LLMAnnotator(spec, CacheStore(CACHE, PROMPT_VERSION))
    logger = RunLogger(RUNS_DIR / f"discover-{annotator}", every=100)
    results = []
    try:
        first = await ann.annotate_and_cache("discover", items[0])
        logger.record(first, (0, 0), len(items)); results.append(first)
        for coro in asyncio.as_completed([ann.annotate_and_cache("discover", it) for it in items[1:]]):
            res = await coro
            logger.record(res, (0, 0), len(items)); results.append(res)
    finally:
        logger.close()
    return results


def aggregate(results, domain_of: dict) -> dict:
    count, domains = Counter(), defaultdict(Counter)
    desc, examples, types = defaultdict(list), defaultdict(list), defaultdict(Counter)
    for res in results:
        if not res.ok:
            continue
        for r in res.payload:
            n = " ".join(r["name"].split())
            count[n] += 1
            domains[n][domain_of[res.item_id]] += 1
            if len(desc[n]) < 3 and r["description"] not in desc[n]:
                desc[n].append(r["description"])
            if len(examples[n]) < 3:
                examples[n].append(r["example"])
            types[n][(r["head_type"].lower(), r["tail_type"].lower())] += 1
    return {"names": [{"name": n, "count": c, "domains": dict(domains[n]),
                       "descriptions": desc[n], "examples": examples[n],
                       "types": [f"{h} -> {t}" for (h, t), _ in types[n].most_common(3)]}
                      for n, c in count.most_common()],
            "windows": len(results), "ok": sum(r.ok for r in results)}


def merge(annotator: str, min_count: int, max_names: int) -> int:
    """Consolidate the discovered names into a candidate general inventory,
    and write it out for review."""
    agg = json.loads(OUT.read_text())
    names = [x for x in agg["names"] if x["count"] >= min_count][:max_names]
    table = "\n".join(f"{x['name']} | {x['count']} | "
                      f"{', '.join(sorted(x['domains']))} | {x['descriptions'][0] if x['descriptions'] else ''}"
                      for x in names)
    key = hashlib.sha1(table.encode()).hexdigest()[:12]
    item = Item(id=f"bankmerge-{key}", tokens=["-"], target=table)
    spec = load_annotators([annotator])[annotator]
    ann = LLMAnnotator(spec, CacheStore(CACHE, PROMPT_VERSION))
    res = asyncio.run(ann.annotate_and_cache("bankmerge", item))
    if not res.ok:
        raise SystemExit(f"merge failed: {res.error}")
    support = {x["name"]: x for x in agg["names"]}
    rels = res.payload
    for r in rels:
        doms, cnt = Counter(), 0
        for a in [r["name"]] + r["aliases"]:
            x = support.get(a.lower())
            if x:
                cnt += x["count"]
                doms.update(x["domains"])
        r["support"] = cnt
        r["domains"] = dict(doms)
    GENERAL.write_text(json.dumps({"from": f"{len(names)} discovered names (count >= {min_count})",
                                   "relations": rels}, indent=1))
    by_cat: dict[str, list] = defaultdict(list)
    for r in rels:
        by_cat[r["category"] or "other"].append(r)
    lines = ["# General relation bank: candidates for review", "",
             f"Consolidated by the teacher ({annotator}) from {len(names)} relation names it "
             f"proposed on {agg['ok']} training windows (250 per domain; names proposed at "
             f"least {min_count} times). **Support** = how often the name or its aliases were "
             "proposed; **domains** = where.", "",
             "Review: strike relations that should not be taught, fix names or definitions, "
             "add missing ones. The approved list joins the label bank (`v6/label_bank.py`).", ""]
    for cat in sorted(by_cat):
        lines += [f"## {cat} ({len(by_cat[cat])})", "",
                  "| relation | definition | head → tail | support | domains | aliases |",
                  "|---|---|---|---|---|---|"]
        for r in sorted(by_cat[cat], key=lambda r: -r["support"]):
            lines.append(f"| {r['name']} | {r['description']} | {r['head_type']} → {r['tail_type']} | "
                         f"{r['support']} | {', '.join(sorted(r['domains']))} | "
                         f"{', '.join(r['aliases'][:5])} |")
        lines.append("")
    REVIEW.write_text("\n".join(lines))
    print(f"{len(rels)} consolidated relations in {len(by_cat)} categories; "
          f"wrote {GENERAL} and {REVIEW}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-domain", type=int, default=250)
    ap.add_argument("--annotator", default="astra")
    ap.add_argument("--merge", action="store_true",
                    help="consolidate an existing discovery into a reviewable inventory")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--max-names", type=int, default=1500)
    args = ap.parse_args()
    if args.merge:
        return merge(args.annotator, args.min_count, args.max_names)
    speakers = json.loads(SPEAKERS.read_text()) if SPEAKERS.exists() else {}
    rows = sample(args.per_domain)
    items = [Item(id=r["window_id"], tokens=r["tokens"], context=r["domain"],
                  target=passage(r, speakers)) for r in rows]
    print(f"{len(items)} windows: " + ", ".join(
        f"{d} {c}" for d, c in sorted(Counter(r['domain'] for r in rows).items())), flush=True)
    results = asyncio.run(run(items, args.annotator))
    agg = aggregate(results, {r["window_id"]: r["domain"] for r in rows})
    OUT.write_text(json.dumps(agg, indent=1))
    print(f"{agg['ok']}/{agg['windows']} windows ok; {len(agg['names'])} distinct relation names")
    for x in agg["names"][:25]:
        print(f"  {x['count']:5d}  {x['name']}  {dict(x['domains'])}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

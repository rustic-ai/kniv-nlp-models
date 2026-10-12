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
BUSINESS = DATA_DIR / "v6-corpus" / "business_relations.json"
# Wikidata business properties not already in the label bank.
BUSINESS_WIKIDATA = ["P169", "P3320", "P414", "P249", "P452", "P1716", "P1128", "P2139",
                     "P2295", "P2403", "P2226", "P1454", "P1830", "P1037", "P1327", "P1951",
                     "P2541", "P2137", "P3362", "P8324", "P2770"]
# Contract relations, after CUAD's clause categories (CC BY 4.0): what a
# commercial agreement establishes between its parties, dates and terms.
CONTRACT = [
    ("governed by law of", "The agreement is governed by and construed under the laws of the jurisdiction.", "agreement", "jurisdiction"),
    ("has effective date", "The agreement takes effect on the date.", "agreement", "date"),
    ("expires on", "The agreement or a right under it ends on the date unless renewed.", "agreement", "date"),
    ("renews for", "The agreement renews, automatically or by option, for the term.", "agreement", "duration"),
    ("requires notice of", "Terminating or not renewing the agreement requires the stated notice period.", "agreement", "duration"),
    ("grants license to", "The party grants the other party a license under the agreement.", "party", "party"),
    ("licenses", "The party licenses the intellectual property, product or right to another party.", "party", "intellectual property"),
    ("has exclusivity with", "The party is bound to deal exclusively with the other party for the agreement's scope.", "party", "party"),
    ("restricted from competing with", "The party is barred from competing with the other party or in the market.", "party", "party or market"),
    ("may terminate", "The party has the right to terminate the agreement.", "party", "agreement"),
    ("caps liability at", "The party's or agreement's liability is limited to the amount.", "agreement or party", "amount"),
    ("assigns rights to", "The party transfers ownership of intellectual property or other rights to the other party.", "party", "party"),
    ("indemnifies", "The party agrees to compensate or defend the other party against stated losses or claims.", "party", "party"),
    ("has audit rights over", "The party may inspect the other party's books, records or compliance.", "party", "party"),
    ("commits to purchase", "The party commits to buy a minimum quantity or amount.", "party", "amount or product"),
]
BUSINESS_SOURCES = {"sec_edgar": 150, "cuad": 120, "odoo": 70, "wikipedia": 60}
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


def sample_business() -> list[dict]:
    """Business windows stratified by source -- filings, contracts, ERP docs,
    company articles -- rather than in corpus proportion, where textbooks and
    academic abstracts dominated and crowded out corporate relations."""
    from .train.data import load_rows
    rows = [r for r in load_rows("train", columns=["window_id", "domain", "source", "tokens",
                                                     "sentence_spans"])
            if r["domain"] == "business"]
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    out = []
    for src, n in BUSINESS_SOURCES.items():
        rs = sorted(by[src], key=lambda r: hashlib.sha1(f"discover-biz:{r['window_id']}".encode()).hexdigest())
        out += rs[:n]
    return out


def business(annotator: str) -> int:
    """The business and finance pack: Wikidata business properties, contract
    relations after CUAD, and relations discovered on business sources --
    consolidated against what the bank and the general candidates already
    hold, and appended to the review file."""
    from .gold.redocred import resolve_relation_info
    from .label_bank import load as load_bank
    speakers = json.loads(SPEAKERS.read_text()) if SPEAKERS.exists() else {}
    rows = sample_business()
    items = [Item(id=r["window_id"], tokens=r["tokens"], context="business",
                  target=passage(r, speakers)) for r in rows]
    print(f"{len(items)} business windows: " + ", ".join(
        f"{k} {v}" for k, v in sorted(Counter(r["source"] for r in rows).items())), flush=True)
    results = asyncio.run(run(items, annotator))
    agg = aggregate(results, {r["window_id"]: r["source"] for r in rows})
    have = {r["name"] for r in load_bank()["relations"]}
    have |= {r["name"] for r in json.loads(GENERAL.read_text())["relations"]}
    wd = resolve_relation_info(BUSINESS_WIKIDATA)
    seeds = [f"{wd[c]['label']} | seed (Wikidata {c}) | business | {wd[c]['description']}"
             for c in BUSINESS_WIKIDATA]
    seeds += [f"{n} | seed (contract) | business | {d}" for n, d, _, _ in CONTRACT]
    found = [x for x in agg["names"] if x["count"] >= 2 and x["name"] not in have][:600]
    table = "\n".join(
        ["FOCUS: business, finance, corporate structure, governance, markets and contracts. "
         "Produce 50 to 90 relations. Every SEED below must be kept (you may reword its "
         "definition); merge discovered names into seeds where they mean the same. Do not "
         "include these relations, which the inventory already has: " + ", ".join(sorted(have))[:6000],
         ""] + seeds +
        [f"{x['name']} | {x['count']} | {', '.join(sorted(x['domains']))} | "
         f"{x['descriptions'][0] if x['descriptions'] else ''}" for x in found])
    key = hashlib.sha1(table.encode()).hexdigest()[:12]
    spec = load_annotators([annotator])[annotator]
    ann = LLMAnnotator(spec, CacheStore(CACHE, PROMPT_VERSION))
    res = asyncio.run(ann.annotate_and_cache("bankmerge", Item(id=f"bizmerge-{key}", tokens=["-"],
                                                               target=table)))
    if not res.ok:
        raise SystemExit(f"business merge failed: {res.error}")
    support = {x["name"]: x for x in agg["names"]}
    seed_names = {wd[c]["label"] for c in BUSINESS_WIKIDATA} | {n for n, *_ in CONTRACT}
    rels = [r for r in res.payload if r["name"] not in have]
    for r in rels:
        doms, cnt = Counter(), 0
        for a in [r["name"]] + r["aliases"]:
            x = support.get(a.lower())
            if x:
                cnt += x["count"]; doms.update(x["domains"])
        r["support"], r["sources"] = cnt, dict(doms)
        r["seed"] = r["name"] in seed_names or any(a in seed_names for a in r["aliases"])
    missing = sorted(seed_names - {r["name"] for r in rels} - {a for r in rels for a in r["aliases"]})
    BUSINESS.write_text(json.dumps({"windows": len(rows), "relations": rels,
                                    "seeds_dropped": missing}, indent=1))
    lines = ["", "---", "", f"# Business and finance pack ({len(rels)})", "",
             f"Added after review feedback: business relations were thin. Sources: 21 Wikidata "
             f"business properties, {len(CONTRACT)} contract relations after CUAD's clause "
             f"categories, and relations the teacher proposed on {agg['ok']} business windows "
             f"stratified by source ({', '.join(f'{k} {v}' for k, v in BUSINESS_SOURCES.items())}). "
             "**Seed** = from Wikidata or CUAD; **support** = how often proposed on those windows.", ""]
    if missing:
        lines += [f"Seeds the consolidation did not keep: {', '.join(missing)}.", ""]
    by_cat: dict[str, list] = defaultdict(list)
    for r in rels:
        by_cat[r["category"] or "business"].append(r)
    for cat in sorted(by_cat):
        lines += [f"## business pack: {cat} ({len(by_cat[cat])})", "",
                  "| relation | definition | head → tail | seed | support | sources | aliases |",
                  "|---|---|---|---|---|---|---|"]
        for r in sorted(by_cat[cat], key=lambda r: (-r["seed"], -r["support"])):
            lines.append(f"| {r['name']} | {r['description']} | {r['head_type']} → {r['tail_type']} | "
                         f"{'yes' if r['seed'] else ''} | {r['support']} | "
                         f"{', '.join(sorted(r['sources']))} | {', '.join(r['aliases'][:5])} |")
        lines.append("")
    text = REVIEW.read_text()
    if "# Business and finance pack" in text:
        text = text[:text.index("\n---\n\n# Business and finance pack")]
    REVIEW.write_text(text.rstrip("\n") + "\n" + "\n".join(lines))
    print(f"business pack: {len(rels)} relations ({sum(r['seed'] for r in rels)} from seeds); "
          f"seeds dropped: {missing or 'none'}; appended to {REVIEW}")
    return 0


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
    ap.add_argument("--business", action="store_true",
                    help="discover on business sources and build the business and finance pack")
    ap.add_argument("--merge", action="store_true",
                    help="consolidate an existing discovery into a reviewable inventory")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--max-names", type=int, default=1500)
    args = ap.parse_args()
    if args.merge:
        return merge(args.annotator, args.min_count, args.max_names)
    if args.business:
        return business(args.annotator)
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

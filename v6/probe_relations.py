"""Relation transfer probe: build entities on OUR windows, then run ATLOP.

The relation annotator's 0.790 F1 / 0.901 precision is a **Wikipedia**
number, measured on Re-DocRED. Our corpus is 44% conversation. Relations
are the newest layer and the one with no fallback if it does not transfer,
so this is the gate before annotation.

There is no gold here, so this measures distribution shift rather than
accuracy, three ways:

1. **Prediction rate** — triples per document, and per entity pair,
   against the Re-DocRED rate. A model that has left its distribution
   usually stops predicting, or starts predicting indiscriminately.
2. **Relation type mix** — Re-DocRED is Wikidata-shaped (`country`,
   `publication date`); conversation should not look like that.
3. **Agreement with an LLM** on the same entity lists, read against the
   0.304 Jaccard the same pair reach on Re-DocRED itself.

Stage 1 (this module) builds the entity lists: NER per sentence from
kniv-v5, coref per window from LingMess, merged into clusters and written
in DocRED format so the existing ATLOP runner consumes them unchanged.

    uv run python -m v6.probe_relations --stage ner --per-domain 40
    .venv-tools/bin/python -m v6.probe_relations --stage coref --per-domain 40
    uv run python -m v6.probe_relations --stage assemble --per-domain 40
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from .annotate.base import CacheStore, validate_payload
from .config import RUNS_DIR, load_annotators
from .gold.ud_ewt import GoldItem
from .prompts import PROMPT_VERSION
from .windows import build_windows, iter_documents, tokenize

DOMAINS = ("conversation", "narrative", "technical", "news", "encyclopedic")
CACHE_VERSION = f"{PROMPT_VERSION}-relprobe"
OUT = Path(__file__).resolve().parents[1] / "data" / "re-docred"

# OntoNotes 18 -> DocRED's six coarse types, which is what ATLOP was
# trained to condition on.
TYPE_MAP = {
    "PERSON": "PER", "ORG": "ORG",
    "GPE": "LOC", "LOC": "LOC", "FAC": "LOC",
    "DATE": "TIME", "TIME": "TIME",
    "PERCENT": "NUM", "MONEY": "NUM", "QUANTITY": "NUM",
    "ORDINAL": "NUM", "CARDINAL": "NUM",
    "NORP": "MISC", "PRODUCT": "MISC", "EVENT": "MISC",
    "WORK_OF_ART": "MISC", "LAW": "MISC", "LANGUAGE": "MISC",
}


def sample_windows(per_domain: int, max_tokens: int = 384, seed: int = 11):
    import random
    out = []
    for domain in DOMAINS:
        pool = []
        for doc in iter_documents(domain):
            pool.extend(build_windows(doc, tokenize, max_tokens=max_tokens))
            if len(pool) > per_domain * 15:
                break
        if not pool:
            continue
        for w in random.Random(seed).sample(pool, min(per_domain, len(pool))):
            w["probe_id"] = f"{domain}:{w['window_id']}"
            out.append(w)
    return out


def bio_spans(tags):
    """(type, start, end_exclusive) from BIO tags."""
    spans, cur, start = [], None, 0
    for i, t in enumerate(list(tags) + ["O"]):
        if not t.startswith("I-"):
            if cur:
                spans.append((cur, start, i))
            cur = t[2:] if t.startswith("B-") else None
            start = i
        elif cur is None:
            cur, start = t[2:], i
    return spans


async def run_ner(windows, cache_dir):
    """NER per SENTENCE, mapped back to window coordinates (see spec 4.1a)."""
    from .annotate import RunLogger
    from .bakeoff import make_annotator
    spec = load_annotators(["kniv-v5"])["kniv-v5"]
    ann = make_annotator(spec, CacheStore(cache_dir, CACHE_VERSION), 1)
    logger = RunLogger(RUNS_DIR / "relprobe-ner", every=50)
    try:
        for w in windows:
            for si, (s, e) in enumerate(w["sentence_spans"]):
                it = GoldItem(id=f"{w['probe_id']}:{si}",
                              tokens=w["tokens"][s:e], layers={})
                res = await ann.annotate_and_cache("ner", it)
                logger.record(res, (0, 0), len(windows))
    finally:
        logger.close()


async def run_coref(windows, cache_dir):
    from .annotate import RunLogger
    from .bakeoff import make_annotator
    spec = load_annotators(["lingmess"])["lingmess"]
    ann = make_annotator(spec, CacheStore(cache_dir, CACHE_VERSION), 1)
    logger = RunLogger(RUNS_DIR / "relprobe-coref", every=25)
    try:
        for w in windows:
            it = GoldItem(id=w["probe_id"], tokens=w["tokens"], layers={})
            res = await ann.annotate_and_cache("coref", it)
            logger.record(res, (0, 0), len(windows))
    finally:
        logger.close()


def assemble(windows, cache_dir):
    """Merge NER spans with coref chains into DocRED-format entities."""
    cache = CacheStore(cache_dir, CACHE_VERSION)
    docs, stats = [], defaultdict(int)
    for w in windows:
        toks = w["tokens"]
        spans = []                                   # (type, start, end_excl)
        for si, (s, e) in enumerate(w["sentence_spans"]):
            rec = cache.get("kniv-v5", "ner", f"{w['probe_id']}:{si}")
            if not rec or rec.get("payload") is None:
                continue
            tags, err, _ = validate_payload("ner", rec["payload"], e - s)
            if err:
                continue
            for t, a, b in bio_spans(tags):
                spans.append((TYPE_MAP.get(t, "MISC"), s + a, s + b))
        if len(spans) < 2:
            stats["too_few_entities"] += 1
            continue

        rec = cache.get("lingmess", "coref", w["probe_id"])
        chains = []
        if rec and rec.get("payload") is not None:
            cl, err, _ = validate_payload("coref", rec["payload"], len(toks))
            if not err:
                chains = cl

        # A coref chain absorbs every NER span it overlaps.
        used, groups = set(), []
        for chain in chains:
            members = []
            for cs, ce in chain:
                for i, (ty, a, b) in enumerate(spans):
                    if i not in used and a <= ce and cs <= b - 1:
                        members.append(i); used.add(i)
            if members:
                groups.append(members)
        for i in range(len(spans)):
            if i not in used:
                groups.append([i])

        # String-match fallback. Coref leaves 12.6% of entities as unmerged
        # duplicates of the same surface form (26.8% on conversation), so
        # "George Orwell" becomes three entities and yields three identical
        # triples — three graph nodes where there should be one. LingMess
        # was measured on literary prose; short conversational text with
        # repeated proper names is further from that than LitBank is.
        #
        # Only exact normalised matches are merged, and only for types where
        # a repeated string reliably means the same referent. NUM and TIME
        # are excluded: "two" and "Monday" recur without co-referring.
        MERGEABLE = {"PER", "ORG", "LOC", "MISC"}
        by_name: dict[tuple[str, str], int] = {}
        merged: list[list[int]] = []
        for g in groups:
            ty = spans[g[0]][0]
            key = (ty, " ".join(toks[spans[g[0]][1]:spans[g[0]][2]]).lower())
            if ty in MERGEABLE and len(key[1]) > 2 and key in by_name:
                merged[by_name[key]].extend(g)
            else:
                if ty in MERGEABLE:
                    by_name[key] = len(merged)
                merged.append(list(g))
        vertex = [[spans[i] for i in g] for g in merged]

        # DocRED format: sentences of tokens, vertexSet of mention dicts
        sents, offs = [], []
        for s, e in w["sentence_spans"]:
            offs.append(s); sents.append(toks[s:e])

        def to_sent(pos):
            si = max(i for i, o in enumerate(offs) if o <= pos)
            return si, pos - offs[si]

        vs = []
        for ms in vertex:
            out = []
            for ty, a, b in ms:
                si, loc = to_sent(a)
                if b - offs[si] > len(sents[si]):     # crosses a sentence
                    continue
                out.append({"name": " ".join(toks[a:b]), "pos": [loc, b - offs[si]],
                            "sent_id": si, "type": ty})
            if out:
                vs.append(out)
        if len(vs) < 2:
            stats["too_few_entities"] += 1
            continue
        docs.append({"title": w["probe_id"], "sents": sents,
                     "vertexSet": vs, "labels": []})
        stats["ok"] += 1
        stats["entities"] += len(vs)
    return docs, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=["ner", "coref", "assemble"])
    ap.add_argument("--per-domain", type=int, default=40)
    ap.add_argument("--cache-dir", type=Path, default=RUNS_DIR / "_cache")
    args = ap.parse_args()

    windows = sample_windows(args.per_domain)
    print(f"sampled {len(windows)} windows", flush=True)
    if args.stage == "ner":
        asyncio.run(run_ner(windows, args.cache_dir))
    elif args.stage == "coref":
        asyncio.run(run_coref(windows, args.cache_dir))
    else:
        docs, stats = assemble(windows, args.cache_dir)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "probe_windows_docred.json").write_text(json.dumps(docs))
        print(f"wrote {len(docs)} documents -> {OUT/'probe_windows_docred.json'}")
        print(f"  {dict(stats)}")
        print(f"  mean entities/doc: {stats['entities']/max(stats['ok'],1):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

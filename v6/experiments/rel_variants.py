"""Relation-extraction prompt variants, measured against Re-DocRED test.

The baseline emits triples from a document plus an entity list and scores
0.503 (astra+grok union). Error decomposition says **74.7% of misses are
entity pairs the model never proposed at all**, and 81.4% of false positives
land on pairs holding no gold relation — the volume is right, the pair
selection is wrong. These two variants attack that directly:

``fewshot``
    Same task, with worked examples drawn from the Re-DocRED **train**
    split. Targets DocRED's annotation conventions — relations like
    ``applies to jurisdiction`` that the benchmark records systematically
    and a reader extracting stated facts would not volunteer. Teaching the
    convention by demonstration, never by licensing the model to invent.

``pairs``
    Candidate pairs are enumerated and handed to the model, which answers
    for each. This is the structural fix for "pair never proposed". Pairs
    are chunked across calls because no cheap prune exists — measured on
    test, type constraints keep 97.4% of ordered pairs (DocRED has six
    coarse entity types, and most combinations admit 40+ relations), and
    locality pruning costs too much gold to use (same-sentence keeps 47.6%,
    a +/-3-sentence window still leaves 301 of 397 pairs per document).

Examples and type constraints come from the train split only; test is never
read for anything but scoring.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ..annotate import CacheStore, RunLogger
from ..annotate.llm import LLMAnnotator
from ..config import RUNS_DIR, load_annotators
from ..gold import load_rel_items
from ..gold.redocred import REDOCRED_DIR, download, relation_inventory
from ..gold.ud_ewt import GoldItem
from ..prompts import PROMPT_VERSION, SYSTEM
from ..score import score_layer

PAIRS_PER_CALL = 80


# ── few-shot examples, mined from train ──────────────────────────

def build_examples(n_docs: int = 2, max_tokens: int = 220) -> str:
    """Render short train documents with their gold triples as examples.

    Short documents are chosen so the examples stay a small share of the
    prompt; the point is to show the *shape and conventions* of the answer,
    not to supply retrieval material.
    """
    _, by_name = relation_inventory()
    names = {code: name for name, code in by_name.items()}
    docs = json.loads(download("train").read_text())
    picked = []
    for d in docs:
        tokens = [t for s in d["sents"] for t in s]
        if not (60 <= len(tokens) <= max_tokens):
            continue
        if not (4 <= len(d["vertexSet"]) <= 9) or not (4 <= len(d["labels"]) <= 12):
            continue
        picked.append((d, tokens))
        if len(picked) >= n_docs:
            break

    out = []
    for d, tokens in picked:
        ents = "\n".join(
            f"{i}\t[{c[0]['type']}]\t{'; '.join(sorted({m['name'] for m in c})[:3])}"
            for i, c in enumerate(d["vertexSet"]))
        triples = sorted({(l["h"], l["t"], names[l["r"]]) for l in d["labels"]})
        body = json.dumps({"triples": [{"h": h, "t": t, "r": r}
                                       for h, t, r in triples]}, indent=None)
        out.append(f"Document:\n{' '.join(tokens)}\n\n"
                   f"Entities:\n{ents}\n\nCorrect answer:\n{body}")
    return "\n\n---\n\n".join(out)


# ── annotators ───────────────────────────────────────────────────

class FewShotRel(LLMAnnotator):
    def __init__(self, *a, examples: str = "", **kw):
        super().__init__(*a, **kw)
        self.examples = examples

    def build_messages(self, layer, item):
        from ..prompts import render_entities
        user = (
            "Worked examples of this task, with the correct answers:\n\n"
            f"{self.examples}\n\n---\n\nNow do the same for this document.\n\n"
            f"Document:\n{' '.join(item.tokens)}\n\n"
            f"Entities ({len(item.entities)} total) as `id  [type]  names`:\n"
            f"{render_entities(item.entities)}\n\n"
            f"Return every relation triple the document supports. Entity ids "
            f"are 0..{len(item.entities) - 1}."
        )
        return [{"role": "system", "content": SYSTEM[layer]},
                {"role": "user", "content": user + self._json_hint(layer)}]


class PairRel(LLMAnnotator):
    def build_messages(self, layer, item):
        from ..prompts import render_entities
        pairs = item.layers["_pairs"]
        listed = "\n".join(f"{h} -> {t}" for h, t in pairs)
        user = (
            f"Document:\n{' '.join(item.tokens)}\n\n"
            f"Entities ({len(item.entities)} total) as `id  [type]  names`:\n"
            f"{render_entities(item.entities)}\n\n"
            f"Candidate entity pairs to judge ({len(pairs)}), as `head -> tail`:\n"
            f"{listed}\n\n"
            "For EACH candidate pair above, decide which relations hold from "
            "head to tail. Emit one triple per relation that holds. Emit "
            "nothing for a pair where no relation holds — most pairs hold "
            "none. Do not emit triples for pairs outside this list."
        )
        return [{"role": "system", "content": SYSTEM[layer]},
                {"role": "user", "content": user + self._json_hint(layer)}]


# ── pair chunking ────────────────────────────────────────────────

def chunk_items(items: list[GoldItem]) -> list[GoldItem]:
    """One sub-item per (document, pair-chunk); merged again before scoring."""
    out = []
    for it in items:
        n = len(it.entities)
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        for k in range(0, len(pairs), PAIRS_PER_CALL):
            part = pairs[k:k + PAIRS_PER_CALL]
            out.append(GoldItem(
                id=f"{it.id}#p{k // PAIRS_PER_CALL:02d}",
                tokens=it.tokens,
                layers={"rel": it.layers["rel"], "_pairs": part},
                entities=it.entities,
            ))
    return out


async def run(annotator, items, logger) -> dict:
    payloads = {}
    if items:
        warm = await annotator.annotate_and_cache("rel", items[0])
        logger.record(warm, (0, 0), len(items))
        if warm.ok:
            payloads[warm.item_id] = warm.payload
    tasks = [annotator.annotate_and_cache("rel", it) for it in items[1:]]
    for coro in asyncio.as_completed(tasks):
        res = await coro
        logger.record(res, (0, 0), len(items))
        if res.ok:
            payloads[res.item_id] = res.payload
    return payloads


async def main_async(args):
    items = load_rel_items(limit=args.limit)
    specs = load_annotators([args.annotator])
    spec = specs[args.annotator]
    run_dir = RUNS_DIR / f"relvar-{args.variant}-{args.annotator}"
    logger = RunLogger(run_dir, every=args.log_every)
    cache = CacheStore(args.cache_dir, f"{PROMPT_VERSION}-{args.variant}")

    if args.variant == "fewshot":
        ex = build_examples()
        print(f"few-shot examples: {len(ex)} chars", flush=True)
        ann = FewShotRel(spec, cache, examples=ex)
        work = items
    elif args.variant == "pairs":
        ann = PairRel(spec, cache)
        work = chunk_items(items)
        print(f"{len(items)} docs -> {len(work)} calls "
              f"({PAIRS_PER_CALL} pairs each)", flush=True)
    else:
        raise SystemExit(f"unknown variant {args.variant!r}")

    print(f"=== rel variant '{args.variant}' / {args.annotator} "
          f"/ {len(items)} docs ===", flush=True)
    try:
        payloads = await run(ann, work, logger)
    finally:
        logger.close()

    if args.variant == "pairs":                 # merge chunks back per document
        merged = {}
        for k, v in payloads.items():
            merged.setdefault(k.split("#")[0], []).extend(v)
        payloads = {k: [list(t) for t in {tuple(x) for x in v}]
                    for k, v in merged.items()}

    s = score_layer("rel", f"{args.annotator}/{args.variant}", items, payloads)
    print(f"\n{s.annotator:<22} F1={s.primary:.3f} P={s.precision:.3f} "
          f"R={s.recall:.3f} cover={s.coverage:.3f} n={s.n_scored}", flush=True)
    (run_dir / "report.json").write_text(json.dumps(s.to_dict(), indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", required=True, choices=["fewshot", "pairs"])
    ap.add_argument("--annotator", default="astra")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--cache-dir", type=Path, default=RUNS_DIR / "_cache")
    ap.add_argument("--log-every", type=int, default=25)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

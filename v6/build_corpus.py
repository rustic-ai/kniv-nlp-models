"""Build the unified v6 training corpus: one corpus, every head, one record.

v5 trained on disjoint per-task datasets, so each example supervised exactly
one head and each head learned a different domain. Here every sentence
carries every layer, so one example supervises all of them and the cascade
features are computed on the text the labels describe.

Annotators are assigned per layer by measurement (see DECISIONS.md), not by
convenience — v5 for POS/NER/DEP/SRL, Stanza for lemma/morph, LingMess for
coref, the LLM ensemble for CLS/sentiment/keyword.

Because those annotators need incompatible dependency stacks, generation is
split in two and the response cache is the seam:

    # one pass per annotator, each in whichever venv it needs
    uv run python -m v6.build_corpus --annotate kniv-v5   --limit 50000
    .venv-tools/bin/python -m v6.build_corpus --annotate stanza --limit 50000
    .venv-trankit/bin/python -m v6.build_corpus --annotate lingmess --limit 50000

    # then assemble from cache — no inference, cheap to re-run
    uv run python -m v6.build_corpus --assemble --limit 50000

Assembly is where provenance is recorded and contradictions are masked
rather than taught: a morph label incompatible with the recorded POS is
dropped for that token, not silently kept.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import re
from pathlib import Path

from .annotate.base import CacheStore
from .config import DATA_DIR, RUNS_DIR, load_annotators
from .gold.ud_ewt import GoldItem, load_ud_items
from .prompts import PROMPT_VERSION

OUT_DIR = DATA_DIR / "v6-corpus"
CORPUS_REPO = "dragonscale-ai/kniv-corpus-en"
CORPUS_FILE = "corpus/gold/test.parquet"

# layer -> annotator, per DECISIONS.md. Every entry is a measured choice.
LAYER_SOURCE = {
    "pos": "kniv-v5",
    "ner": "kniv-v5",
    "dep": "kniv-v5",
    "srl": "kniv-v5",
    "lemma": "stanza",
    "morph": "stanza",
    "coref": "lingmess",
}


def load_corpus(limit: int | None = None, max_tokens: int = 128) -> list[GoldItem]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    tbl = pq.read_table(hf_hub_download(CORPUS_REPO, CORPUS_FILE,
                                        repo_type="dataset"))
    items = []
    for row in tbl.select(["sent_id", "text", "prev_text", "cls", "tokens"]).to_pylist():
        toks = row["tokens"]
        if not toks or len(toks) < 3 or len(toks) > max_tokens:
            continue
        items.append(GoldItem(
            id=f"corpus-{row['sent_id']}",
            tokens=toks,
            layers={"_text": row["text"], "_prev": row["prev_text"],
                    "_cls_v5": row["cls"]},
        ))
        if limit and len(items) >= limit:
            break
    return items


# ── morph/POS compatibility, learned from UD EWT rather than asserted ──

def attested_feats() -> dict[str, set[str]]:
    """(UPOS -> allowed FEATS strings), empirically from UD EWT train."""
    table: dict[str, set[str]] = collections.defaultdict(set)
    for it in load_ud_items("train", limit=None):
        for p, f in zip(it.layers["pos"], it.layers["morph"]):
            table[p].add(f)
    return table


# ── annotate ──────────────────────────────────────────────────

async def annotate(name: str, items: list[GoldItem], cache: CacheStore) -> None:
    from .bakeoff import make_annotator
    spec = load_annotators([name])[name]
    layers = [lyr for lyr, a in LAYER_SOURCE.items() if a == name]
    print(f"[{name}] layers={layers} over {len(items):,} sentences", flush=True)
    for layer in layers:
        ann = make_annotator(spec, cache, 1)
        ok = fail = cached = 0
        for i, it in enumerate(items, 1):
            if layer == "srl":
                # SRL is per-predicate; POS must exist first to find verbs.
                pos = _cached(cache, LAYER_SOURCE["pos"], "pos", it)
                if not pos:
                    fail += 1
                    continue
                it.predicate_idx = next(
                    (k for k, p in enumerate(pos) if p == "VERB"), None)
                if it.predicate_idx is None:
                    continue                       # no predicate: no SRL frame
            res = await ann.annotate_and_cache(layer, it)
            ok += res.ok
            fail += not res.ok
            cached += res.cached
            if i % 500 == 0:
                print(f"  [{name}/{layer}] {i:,}/{len(items):,} "
                      f"ok={ok} cached={cached} fail={fail}", flush=True)
        print(f"  [{name}/{layer}] done ok={ok} cached={cached} fail={fail}",
              flush=True)


def _cached(cache: CacheStore, ann: str, layer: str, item: GoldItem):
    f = cache.path(ann, layer, item.id)
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())["payload"]
    except (json.JSONDecodeError, KeyError):
        return None


# ── assemble ──────────────────────────────────────────────────

def assemble(items: list[GoldItem], cache: CacheStore, shard_size: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    allowed = attested_feats()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("shard_*.parquet"):
        old.unlink()

    rows, shard, stats = [], 0, collections.Counter()
    masked_tokens = total_tokens = 0

    for it in items:
        n = len(it.tokens)
        payload, mask = {}, {}
        for layer, ann in LAYER_SOURCE.items():
            p = _cached(cache, ann, layer, it)
            payload[layer] = p
            mask[layer] = p is not None
            stats[f"{layer}:{'ok' if p is not None else 'missing'}"] += 1

        # Per-token morph mask: drop labels that contradict the recorded POS.
        morph_mask = [True] * n
        pos, morph = payload.get("pos"), payload.get("morph")
        if pos and morph and len(pos) == n and len(morph) == n:
            for k, (p, f) in enumerate(zip(pos, morph)):
                total_tokens += 1
                if f not in allowed.get(p, set()):
                    morph_mask[k] = False
                    masked_tokens += 1
        elif morph:
            mask["morph"] = False               # length mismatch: unusable

        rows.append({
            "sent_id": it.id,
            "domain": re.split(r"[-_]", it.id[len("corpus-"):])[0],
            "tokens": it.tokens,
            "text": it.layers.get("_text"),
            "prev_text": it.layers.get("_prev"),
            "pos": payload.get("pos"),
            "lemma": payload.get("lemma"),
            "morph": payload.get("morph"),
            "morph_token_mask": morph_mask,
            "ner": payload.get("ner"),
            "dep_heads": (payload["dep"] or {}).get("heads") if payload.get("dep") else None,
            "dep_rels": (payload["dep"] or {}).get("rels") if payload.get("dep") else None,
            "srl_tags": payload.get("srl"),
            "srl_predicate_idx": it.predicate_idx,
            "coref": json.dumps(payload.get("coref")) if payload.get("coref") else None,
            "cls_v5": it.layers.get("_cls_v5"),
            "loss_mask": json.dumps(mask),
            "provenance": json.dumps(LAYER_SOURCE),
        })
        if len(rows) >= shard_size:
            _write(rows, shard, pa, pq)
            rows, shard = [], shard + 1
    if rows:
        _write(rows, shard, pa, pq)
        shard += 1

    print(f"\nwrote {shard} shard(s) to {OUT_DIR}")
    print("\nlayer coverage:")
    for layer in LAYER_SOURCE:
        ok = stats[f"{layer}:ok"]
        tot = ok + stats[f"{layer}:missing"]
        print(f"  {layer:<7} {ok:>7,}/{tot:,} ({ok / tot:.1%})  <- {LAYER_SOURCE[layer]}")
    if total_tokens:
        print(f"\nmorph tokens masked (POS/FEATS incompatible): "
              f"{masked_tokens:,}/{total_tokens:,} ({masked_tokens / total_tokens:.2%})")

    meta = {"layer_source": LAYER_SOURCE, "n_sentences": len(items),
            "shards": shard, "prompt_version": PROMPT_VERSION,
            "morph_masked_tokens": masked_tokens, "tokens": total_tokens}
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))


def _write(rows, shard, pa, pq):
    path = OUT_DIR / f"shard_{shard:03d}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    print(f"  wrote {path.name} ({len(rows):,} rows)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--annotate", help="Run one annotator over the corpus")
    ap.add_argument("--assemble", action="store_true",
                    help="Build unified records from cached predictions")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard-size", type=int, default=25_000)
    ap.add_argument("--cache-dir", type=Path, default=RUNS_DIR / "_cache")
    args = ap.parse_args()

    items = load_corpus(args.limit)
    print(f"corpus: {len(items):,} sentences", flush=True)
    cache = CacheStore(args.cache_dir, PROMPT_VERSION)
    if args.annotate:
        asyncio.run(annotate(args.annotate, items, cache))
    if args.assemble:
        assemble(items, cache, args.shard_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

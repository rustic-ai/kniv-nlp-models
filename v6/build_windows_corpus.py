"""Build the v6 training corpus over 512-token windows.

Replaces the sentence-level ``build_corpus.py``. The unit is the window,
but **the annotation unit is not**: per the measurement in
``DATASET_SPEC.md`` §4.1a, kniv-v5 and Stanza are sentence-level models and
feeding them a whole window costs 43 points of inter-annotator agreement.
So the structural layers are annotated per sentence and mapped back through
``sentence_spans``; only coref and relations see the whole window.

Stages, each resumable because every response is cached on disk:

    # 1. materialise windows once — canonical tokenization fixed here
    uv run python -m v6.build_windows_corpus --stage windows

    # 2. annotate, one annotator per venv, sharing the cache
    uv run python -m v6.build_windows_corpus --stage annotate --annotator kniv-v5
    .venv-tools/bin/python -m v6.build_windows_corpus --stage annotate --annotator stanza
    .venv-tools/bin/python -m v6.build_windows_corpus --stage annotate --annotator lingmess

    # 3. join into the record schema with provenance and masks
    uv run python -m v6.build_windows_corpus --stage assemble
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from .annotate.base import CacheStore, validate_payload
from .config import DATA_DIR, RUNS_DIR, load_annotators
from .gold.ud_ewt import GoldItem
from .prompts import PROMPT_VERSION
from .windows import build_windows, iter_documents, tokenize

OUT = DATA_DIR / "v6-corpus"
WINDOWS = OUT / "windows.jsonl"
CACHE_VERSION = f"{PROMPT_VERSION}-corpus"
DOMAINS = ("conversation", "narrative", "technical", "news", "encyclopedic")

# Which layers each annotator owns, and at which granularity. Per
# DECISIONS.md; every entry is a measured choice, not a preference.
PER_SENTENCE = {"kniv-v5": ["pos", "ner", "dep", "srl"],
                "stanza": ["lemma", "morph"]}
PER_WINDOW = {"lingmess": ["coref"]}


def stage_windows(limit: int | None, per_domain: int | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    counts: dict[str, int] = defaultdict(int)
    with WINDOWS.open("w") as fh:
        for domain in DOMAINS:
            for doc in iter_documents(domain):
                for w in build_windows(doc, tokenize):
                    if per_domain and counts[domain] >= per_domain:
                        break
                    fh.write(json.dumps(w) + "\n")
                    counts[domain] += 1; n += 1
                    if limit and n >= limit:
                        break
                if limit and n >= limit:
                    break
    print(f"wrote {n} windows -> {WINDOWS}")
    print(f"  {dict(counts)}")


def load_windows(limit: int | None = None) -> list[dict]:
    if not WINDOWS.exists():
        raise SystemExit(f"{WINDOWS} missing — run --stage windows first")
    out = []
    with WINDOWS.open() as fh:
        for line in fh:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def sentence_items(w: dict) -> list[GoldItem]:
    items = []
    for si, (s, e) in enumerate(w["sentence_spans"]):
        items.append(GoldItem(id=f"{w['window_id']}:{si}",
                              tokens=w["tokens"][s:e], layers={}))
    return items


async def stage_annotate(name: str, windows: list[dict], cache_dir: Path,
                         layers: list[str] | None) -> None:
    from .annotate import RunLogger
    from .bakeoff import make_annotator
    spec = load_annotators([name])[name]
    cache = CacheStore(cache_dir, CACHE_VERSION)
    ann = make_annotator(spec, cache, max_repairs=1)

    todo = layers or PER_SENTENCE.get(name) or PER_WINDOW.get(name) or []
    if not todo:
        raise SystemExit(f"no layers declared for {name!r}")
    per_sentence = name in PER_SENTENCE

    total_units = (sum(len(w["sentence_spans"]) for w in windows)
                   if per_sentence else len(windows))
    for layer in todo:
        logger = RunLogger(RUNS_DIR / f"corpus-{name}-{layer}", every=200)
        n_ok = n = 0
        try:
            for w in windows:
                units = sentence_items(w) if per_sentence else [
                    GoldItem(id=w["window_id"], tokens=w["tokens"], layers={})]
                for it in units:
                    res = await ann.annotate_and_cache(layer, it)
                    # Total is the number of ANNOTATED UNITS, not windows:
                    # a window expands to ~17.7 sentences, and reporting
                    # progress against the window count made the rate and
                    # ETA meaningless.
                    logger.record(res, (0, 0), total_units)
                    n += 1; n_ok += res.ok
        finally:
            logger.close()
        print(f"{name}/{layer}: {n_ok}/{n} ok", flush=True)


def is_tree(heads: list[int], start: int, end: int) -> bool:
    """Exactly one root and no cycle, over window-absolute heads.

    Measured on the pilot, kniv-v5 emits a non-tree for 7.8% of sentences
    on our corpus — worse than the 96.7% tree-ok it scores on UD EWT, and
    concentrated in long sentences. An arc set that is not a tree is
    unusable downstream however accurate its individual arcs are, so those
    sentences are masked rather than taught, exactly as morph is masked
    where the (UPOS, FEATS) pair is impossible.
    """
    local = heads[start:end]
    if sum(1 for h in local if h == -1) != 1:
        return False
    for i in range(len(local)):
        seen, cur, steps = set(), i, 0
        while local[cur] != -1:
            nxt = local[cur] - start
            if not (0 <= nxt < len(local)) or nxt in seen or steps > len(local):
                return False
            seen.add(nxt); cur = nxt; steps += 1
    return True


def _get(cache, ann, layer, key, n):
    rec = cache.get(ann, layer, key)
    if not rec or rec.get("payload") is None:
        return None
    payload, err, _ = validate_payload(layer, rec["payload"], n)
    return None if err else payload


def stage_assemble(windows: list[dict], cache_dir: Path,
                   shard_size: int = 2000) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cache = CacheStore(cache_dir, CACHE_VERSION)
    rows, shard, stats = [], 0, defaultdict(int)
    OUT.mkdir(parents=True, exist_ok=True)

    for w in windows:
        n = w["n_tokens"]
        row = {"window_id": w["window_id"], "doc_id": w["doc_id"],
               "domain": w["domain"], "source": w["source"],
               "tokens": w["tokens"], "sentence_spans": w["sentence_spans"],
               "n_tokens": n}
        provenance, mask = {}, {}

        for ann, layers in PER_SENTENCE.items():
            for layer in layers:
                if layer == "srl":
                    continue                      # structured; handled below
                merged, heads, ok = [], [], True
                for si, (s, e) in enumerate(w["sentence_spans"]):
                    p = _get(cache, ann, layer, f"{w['window_id']}:{si}", e - s)
                    if p is None:
                        ok = False; break
                    if layer == "dep":
                        merged.extend(p["rels"])
                        # Heads arrive 1-indexed within their own sentence
                        # (0 = root). Re-base to window-absolute indices,
                        # with -1 for a sentence root, so a consumer never
                        # has to know which sentence a token came from.
                        heads.extend([-1 if h == 0 else s + h - 1
                                      for h in p["heads"]])
                    else:
                        merged.extend(p)
                if ok and len(merged) == n:
                    row[layer] = merged
                    if layer == "dep":
                        row["dep_heads"] = heads
                        # Per-token mask: keep the well-formed sentences,
                        # drop only the sentences that are not trees.
                        dm = [False] * n
                        bad = 0
                        for s2, e2 in w["sentence_spans"]:
                            if not is_tree(heads, s2, e2):
                                for i in range(s2, e2):
                                    dm[i] = True
                                bad += 1
                        if bad:
                            mask["dep_tokens"] = dm
                            stats["dep_sentences_masked"] += bad
                        stats["dep_sentences_total"] += len(w["sentence_spans"])
                    provenance[layer] = ann
                else:
                    row[layer] = None
                    if layer == "dep":
                        row["dep_heads"] = None
                    mask[layer] = True            # absent, not wrong
                    stats[f"missing_{layer}"] += 1

        p = _get(cache, "lingmess", "coref", w["window_id"], n)
        row["coref"] = p
        if p is None:
            mask["coref"] = True; stats["missing_coref"] += 1
        else:
            provenance["coref"] = "lingmess"

        row["provenance"] = json.dumps(provenance)
        row["loss_mask"] = json.dumps(mask)
        rows.append(row)
        stats["rows"] += 1

        if len(rows) >= shard_size:
            _write(rows, shard, pa, pq); rows, shard = [], shard + 1
    if rows:
        _write(rows, shard, pa, pq)

    (OUT / "MANIFEST.json").write_text(json.dumps({
        "windows": stats["rows"], "shards": shard + (1 if rows else 0),
        "layer_source": {**{l: a for a, ls in PER_SENTENCE.items() for l in ls},
                         **{l: a for a, ls in PER_WINDOW.items() for l in ls}},
        "stats": dict(stats),
    }, indent=2))
    print(f"assembled {stats['rows']} rows -> {OUT}")
    print(f"  {dict(stats)}")


def _write(rows, shard, pa, pq):
    tbl = pa.Table.from_pylist(rows)
    pq.write_table(tbl, OUT / f"shard_{shard:03d}.parquet")
    print(f"  wrote shard_{shard:03d}.parquet ({len(rows)} rows)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=["windows", "annotate", "assemble"])
    ap.add_argument("--annotator")
    ap.add_argument("--layers", help="comma-separated subset")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--per-domain", type=int)
    ap.add_argument("--cache-dir", type=Path, default=RUNS_DIR / "_cache")
    args = ap.parse_args()

    if args.stage == "windows":
        stage_windows(args.limit, args.per_domain)
        return 0
    windows = load_windows(args.limit)
    print(f"{len(windows)} windows", flush=True)
    if args.stage == "annotate":
        if not args.annotator:
            raise SystemExit("--annotator required")
        layers = [s.strip() for s in args.layers.split(",")] if args.layers else None
        asyncio.run(stage_annotate(args.annotator, windows, args.cache_dir, layers))
    else:
        stage_assemble(windows, args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

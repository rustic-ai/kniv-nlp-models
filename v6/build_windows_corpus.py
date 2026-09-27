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
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .annotate.base import CacheStore, validate_payload
from .config import DATA_DIR, RUNS_DIR, load_annotators
from .gold.ud_ewt import GoldItem
from .prompts import PROMPT_VERSION
from .entities import build_entities
from .windows import build_windows, iter_documents, tokenize

OUT = DATA_DIR / "v6-corpus"
WINDOWS = OUT / "windows.jsonl"
CACHE_VERSION = f"{PROMPT_VERSION}-corpus"
DOMAINS = ("conversation", "narrative", "technical", "news", "encyclopedic")

# Which layers each annotator owns, and at which granularity. Per
# DECISIONS.md; every entry is a measured choice, not a preference.
# SRL is handled separately: it is predicate-conditioned, so it needs one
# item per (sentence, predicate) rather than one per sentence, and the
# predicates come from the POS layer — which means POS must be annotated
# first. See stage_srl.
PER_SENTENCE = {"kniv-v5": ["pos", "ner", "dep"],
                "stanza": ["lemma", "morph"]}
SRL_PREDICATE_TAGS = {"VERB", "AUX"}
PER_WINDOW = {"lingmess": ["coref"]}

# LLM layers. CLS and sentiment are labelled PER SENTENCE with the whole
# window supplied as context: the v5 CLS head reads 0.951 in-domain and
# 0.613 in the wild partly because it saw one utterance plus at most one
# predecessor. Keywords are window-level by nature.
#
# One annotator over the bulk, not an ensemble. Consensus lost on 7 of 8
# layers in the bake-off, and on the one where it won only a union of the
# top two helped. The adjudicated CLS gold set needs 400-600 items and a
# 100-item human overlap (CLS_TAXONOMY.md) — a sample, not the corpus.
LLM_PER_SENTENCE = ["cls", "sentiment"]
LLM_PER_WINDOW = ["keywords"]
# Production ran one annotator for these, not the five-family ensemble of
# DATASET_SPEC 4.3; assembly reads that one. Widening this to a consensus
# means reading several annotators here and adjudicating, not changing the
# schema -- the row shape is the same either way.
LLM_ANNOTATOR = "astra"

# Relations are not annotated by an LLM or by a toolkit: they come from the
# ATLOP checkpoint retrained on Re-DocRED (0.790 test F1), which needs entity
# clusters supplied rather than finding them itself. So the layer is two
# stages either side of an external process:
#
#   --stage entities   NER + coref -> DocRED-format clusters
#   (ATLOP_INPUT=... python -m v6.experiments.atlop_runner)
#   --stage assemble   reads the predictions back in
#
# Yield is sparse by inventory, not by failure: 4.4 triples/window measured,
# against 27.1 on Re-DocRED, because Wikidata properties describe
# encyclopedic facts and most of this corpus is not encyclopedic. Windows
# with no relation carry a masked layer, since absence of an extractable
# triple is not evidence that none exists.
ENTITIES_FILE = OUT / "entities_docred.json"
RELATIONS_FILE = OUT / "relations_atlop.json"


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


def item_key(window_id: str, index: int, tokens: list[str]) -> str:
    """Cache key bound to the TEXT, not just its position.

    Keying on window_id:index alone is content-blind, and window ids are
    reused: window_id is hash(doc_id, window_index), so any change to how
    documents pack into windows re-points an existing id at different text.
    Artifact cleaning did exactly that, and 29.7% of cached entries ended up
    attached to sentences with a different token count — the rest matched in
    length but not in content, and would have flowed into the corpus as
    plausible wrong labels that no gate checks for.

    The digest makes staleness impossible: different text is a different
    key, so it is a cache miss rather than a silent mismatch.
    """
    digest = hashlib.sha1(" ".join(tokens).encode("utf-8")).hexdigest()[:10]
    return f"{window_id}:{index}:{digest}"


def sentence_items(w: dict) -> list[GoldItem]:
    items = []
    for si, (s, e) in enumerate(w["sentence_spans"]):
        toks = w["tokens"][s:e]
        items.append(GoldItem(id=item_key(w["window_id"], si, toks),
                              tokens=toks, layers={}))
    return items


async def stage_llm(name: str, windows: list[dict], cache_dir: Path,
                    layers: list[str] | None) -> None:
    """Annotate the LLM layers, running items concurrently.

    The structural annotators are local and sequential; these are API calls,
    so throughput comes from concurrency. Measured on this deployment:
    7.21 sentences/s at concurrency 16, 16.23 at 64.
    """
    from .annotate import RunLogger
    from .annotate.llm import LLMAnnotator
    spec = load_annotators([name])[name]
    cache = CacheStore(cache_dir, CACHE_VERSION)
    ann = LLMAnnotator(spec, cache, max_repairs=1)
    todo = layers or (LLM_PER_SENTENCE + LLM_PER_WINDOW)

    for layer in todo:
        per_sentence = layer in LLM_PER_SENTENCE
        items = []
        for w in windows:
            if per_sentence:
                ctx = " ".join(w["tokens"])[:2000]
                for si, (s, e) in enumerate(w["sentence_spans"]):
                    toks = w["tokens"][s:e]
                    items.append(GoldItem(
                        id=item_key(w["window_id"], si, toks), tokens=toks,
                        layers={}, context=ctx, target=" ".join(toks)))
            else:
                items.append(GoldItem(id=w["window_id"], tokens=w["tokens"],
                                      layers={}))
        logger = RunLogger(RUNS_DIR / f"corpus-{name}-{layer}", every=500)
        print(f"{name}/{layer}: {len(items)} items", flush=True)
        n_ok = 0
        try:
            # Warm up once so the response_format ladder settles before the
            # fan-out, then run everything concurrently.
            first = await ann.annotate_and_cache(layer, items[0])
            logger.record(first, (0, 0), len(items)); n_ok += first.ok
            tasks = [ann.annotate_and_cache(layer, it) for it in items[1:]]
            for coro in asyncio.as_completed(tasks):
                res = await coro
                logger.record(res, (0, 0), len(items)); n_ok += res.ok
        finally:
            logger.close()
        print(f"{name}/{layer}: {n_ok}/{len(items)} ok", flush=True)


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

    # Toolkit annotators (Stanza) produce every layer from ONE pipeline call
    # and memoise it, but the memo is bounded at 4096 entries. Iterating
    # layer-major over 583,824 sentences clears it ~143 times before the
    # second layer starts, so the pipeline runs twice for output a single
    # call already produced. Iterate item-major instead: one analysis, both
    # layers cached.
    if len(todo) > 1 and hasattr(ann, "_analyse_memo"):
        await _annotate_item_major(ann, todo, windows, logger_dir=name)
        return

    total_units = (sum(len(w["sentence_spans"]) for w in windows)
                   if per_sentence else len(windows))
    for layer in todo:
        logger = RunLogger(RUNS_DIR / f"corpus-{name}-{layer}", every=200)
        n_ok = n = 0
        pend = []
        try:
            for w in windows:
                units = sentence_items(w) if per_sentence else [
                    GoldItem(id=w["window_id"], tokens=w["tokens"], layers={})]
                pend.extend(units)
                if len(pend) >= 256 or w is windows[-1]:
                    # Total is the number of ANNOTATED UNITS, not windows: a
                    # window expands to ~17.7 sentences, and reporting
                    # progress against the window count made the rate and
                    # ETA meaningless.
                    if hasattr(ann, "annotate_batch_and_cache"):
                        out = await ann.annotate_batch_and_cache(layer, pend)
                    else:
                        out = [await ann.annotate_and_cache(layer, it)
                               for it in pend]
                    for res in out:
                        logger.record(res, (0, 0), total_units)
                        n += 1; n_ok += res.ok
                    pend = []
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


async def stage_srl(windows: list[dict], cache_dir: Path) -> None:
    """One predicate-conditioned pass per verb, per sentence.

    The predicate list comes from the POS layer rather than from a caller,
    so POS must already be annotated — a sentence whose POS is missing is
    skipped rather than guessed at. Measured at 4.82 predicates per
    sentence, which is why SRL dominates the build; v6 plans a head that
    scores every predicate in one pass (MODEL_CHANGES.md §1).
    """
    from .annotate import RunLogger
    from .bakeoff import make_annotator
    spec = load_annotators(["kniv-v5"])["kniv-v5"]
    cache = CacheStore(cache_dir, CACHE_VERSION)
    ann = make_annotator(spec, cache, max_repairs=1)

    items, no_pos = [], 0
    for w in windows:
        for si, (s, e) in enumerate(w["sentence_spans"]):
            key = item_key(w["window_id"], si, w["tokens"][s:e])
            pos = _get(cache, "kniv-v5", "pos", key, e - s)
            if pos is None:
                no_pos += 1
                continue
            for pi, tag in enumerate(pos):
                if tag in SRL_PREDICATE_TAGS:
                    items.append(GoldItem(id=f"{key}:p{pi}",
                                          tokens=w["tokens"][s:e], layers={},
                                          predicate_idx=pi))
    print(f"srl: {len(items)} predicate passes over {len(windows)} windows"
          f"{f' ({no_pos} sentences skipped, POS missing)' if no_pos else ''}",
          flush=True)
    if no_pos and not items:
        raise SystemExit("POS must be annotated before SRL")

    logger = RunLogger(RUNS_DIR / "corpus-kniv-v5-srl", every=1000)
    n_ok = 0
    try:
        # Batched across sentences: 4.2x measured, output identical.
        for i in range(0, len(items), 512):
            chunk = items[i:i + 512]
            if hasattr(ann, "annotate_srl_batch_and_cache"):
                out = await ann.annotate_srl_batch_and_cache(chunk)
            else:
                out = [await ann.annotate_and_cache("srl", it) for it in chunk]
            for res in out:
                logger.record(res, (0, 0), len(items)); n_ok += res.ok
    finally:
        logger.close()
    print(f"kniv-v5/srl: {n_ok}/{len(items)} ok", flush=True)


def stage_entities(windows: list[dict], cache_dir: Path) -> None:
    cache = CacheStore(cache_dir, CACHE_VERSION)
    docs, skipped = [], 0
    for w in windows:
        ner = []
        for si, (s, e) in enumerate(w["sentence_spans"]):
            ner.append(_get(cache, "kniv-v5", "ner",
                            item_key(w["window_id"], si, w["tokens"][s:e]), e - s))
        coref = _get(cache, "lingmess", "coref", w["window_id"], w["n_tokens"])
        sents, vertex = build_entities(w["tokens"], w["sentence_spans"], ner, coref)
        if vertex is None:
            skipped += 1
            continue
        docs.append({"title": w["window_id"], "sents": sents,
                     "vertexSet": vertex, "labels": []})
    OUT.mkdir(parents=True, exist_ok=True)
    ENTITIES_FILE.write_text(json.dumps(docs))
    ents = sum(len(d["vertexSet"]) for d in docs)
    print(f"wrote {len(docs)} documents -> {ENTITIES_FILE}")
    print(f"  {skipped} windows skipped (<2 entities); "
          f"mean entities/doc {ents / max(len(docs), 1):.1f}")
    print(f"  next: ATLOP_INPUT={ENTITIES_FILE} ATLOP_OUT={RELATIONS_FILE} "
          f"uv run python -m v6.experiments.atlop_runner")


def load_relations() -> dict[str, list]:
    """ATLOP predictions keyed by window_id, as readable relation names."""
    if not RELATIONS_FILE.exists():
        return {}
    from .gold.redocred import relation_inventory
    _, by_name = relation_inventory()
    code2name = {c: n for n, c in by_name.items()}
    out = {}
    for rec in json.loads(RELATIONS_FILE.read_text()):
        out[rec["title"]] = sorted({(h, t, code2name[c])
                                    for h, t, c in rec["preds"]
                                    if c in code2name})
    return out


async def _annotate_item_major(ann, layers: list[str], windows: list[dict],
                               logger_dir: str) -> None:
    """One pass over items, filling every layer per item."""
    from .annotate import RunLogger
    total = sum(len(w["sentence_spans"]) for w in windows) * len(layers)
    logger = RunLogger(RUNS_DIR / f"corpus-{logger_dir}-all", every=1000)
    counts = {lyr: 0 for lyr in layers}
    try:
        pend = []
        for w in windows:
            pend.extend(sentence_items(w))
            if len(pend) < 128 and w is not windows[-1]:
                continue
            # One pipeline call for the whole chunk; the memo then serves
            # every layer without re-analysing. 3.4x measured.
            if hasattr(ann, "analyse_bulk"):
                ann.analyse_bulk(pend)
            for it in pend:
                for lyr in layers:
                    res = await ann.annotate_and_cache(lyr, it)
                    logger.record(res, (0, 0), total)
                    counts[lyr] += res.ok
            pend = []
    finally:
        logger.close()
    for lyr in layers:
        print(f"{logger_dir}/{lyr}: {counts[lyr]}/"
              f"{total // len(layers)} ok", flush=True)


def _get(cache, ann, layer, key, n):
    rec = cache.get(ann, layer, key)
    if not rec or rec.get("payload") is None:
        return None
    payload, err, _ = validate_payload(layer, rec["payload"], n)
    return None if err else payload


def stage_assemble(windows: list[dict], cache_dir: Path,
                   shard_size: int = 2000) -> None:
    relations = load_relations() or None
    if relations is None:
        print("  NOTE: no ATLOP predictions found; relations will be masked",
              flush=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    cache = CacheStore(cache_dir, CACHE_VERSION)
    rows, shard, stats = [], 0, defaultdict(int)
    sources: dict[str, set[str]] = {}
    tok = _encoder_tokenizer()
    # Exact-duplicate windows, deduplicated BEFORE the split so the 5%
    # targets stay exact. Measured corpus-wide: 291 of 30,364 windows (0.96%)
    # in 107 groups, and 18 of those groups spanned splits -- 64 windows of
    # identical text in both train and test. Gate 16 cannot see that, because
    # the documents differ; only the content does not. Mostly templated
    # openings in synthetic assistant dialogue ("Hi, I need a new password").
    # Gate 5 asks for a dedup threshold set from the observed distribution;
    # exact token-sequence match is that threshold.
    seen_tokens: set[tuple] = set()
    deduped = []
    for w in windows:
        k = tuple(w["tokens"])
        if k in seen_tokens:
            stats["duplicate_windows_dropped"] += 1
            continue
        seen_tokens.add(k)
        deduped.append(w)
    if stats["duplicate_windows_dropped"]:
        print(f"  dropped {stats['duplicate_windows_dropped']} exact-duplicate "
              f"windows of {len(windows)}", flush=True)
    windows = deduped
    splits = assign_splits(windows)
    # Accumulated as rows are built, because rows are flushed per shard and
    # the full list is never held in memory.
    split_windows: dict[str, int] = defaultdict(int)
    split_docs: dict[str, set[str]] = defaultdict(set)
    OUT.mkdir(parents=True, exist_ok=True)

    for w in windows:
        n = w["n_tokens"]
        row = {"window_id": w["window_id"], "doc_id": w["doc_id"],
               "split": splits[w["window_id"]],
               "domain": w["domain"], "source": w["source"],
               "tokens": w["tokens"], "sentence_spans": w["sentence_spans"],
               "n_tokens": n}
        provenance, mask = {}, {}

        # The window builder packs to 512 WORDS; the encoder's limit is 512
        # SUBWORDS, and the mean ratio on this corpus is 1.079. So a third of
        # windows carry a tail the encoder never sees, and supervising a
        # position that has no representation is not something a shape gate
        # can notice. Recording where the encoder stops makes the loss
        # explicit: 3.33% of word positions corpus-wide, and no window loses
        # all of its tokens.
        n_sub, limit = _encoder_extent(tok, w["tokens"], n)
        row["n_subword_tokens"] = n_sub
        row["encoder_word_limit"] = limit
        if limit < n:
            stats["windows_truncated_by_encoder"] += 1
            stats["word_positions_past_encoder"] += n - limit

        for ann, layers in PER_SENTENCE.items():
            for layer in layers:
                if layer == "srl":
                    continue                      # structured; handled below
                merged, heads, ok = [], [], True
                for si, (s, e) in enumerate(w["sentence_spans"]):
                    p = _get(cache, ann, layer,
                             item_key(w["window_id"], si, w["tokens"][s:e]), e - s)
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

        # srl_frames: one entry per predicate, tags in window coordinates.
        frames = []
        for si, (s2, e2) in enumerate(w["sentence_spans"]):
            key = item_key(w["window_id"], si, w["tokens"][s2:e2])
            pos = _get(cache, "kniv-v5", "pos", key, e2 - s2)
            if pos is None:
                continue
            for pi, tag in enumerate(pos):
                if tag not in SRL_PREDICATE_TAGS:
                    continue
                tags = _get(cache, "kniv-v5", "srl", f"{key}:p{pi}", e2 - s2)
                if tags is None:
                    continue
                full = ["O"] * n
                full[s2:e2] = tags
                frames.append({"predicate_idx": s2 + pi, "tags": full})
        if frames:
            row["srl_frames"] = frames
            provenance["srl"] = "kniv-v5"
            stats["srl_frames"] += len(frames)
        else:
            row["srl_frames"] = None
            mask["srl"] = True
            stats["windows_without_srl"] += 1

        if relations is not None:
            rel = relations.get(w["window_id"])
            # struct, not a mixed list: [h, t, r] is (int, int, str) and
            # Arrow cannot infer a type for that. The spec specifies a struct
            # for this reason.
            row["relations"] = ([{"head": h, "tail": t, "relation": r}
                                 for h, t, r in rel] if rel else None)
            if rel:
                provenance["relations"] = "atlop-redocred"
                stats["relation_triples"] += len(rel)
            else:
                mask["relations"] = True
                stats["windows_without_relations"] += 1

        # LLM layers. These were annotated and cached but never assembled,
        # so the corpus silently shipped without CLS -- one of the three
        # stated v6 goals. cls and sentiment are per sentence, keywords per
        # window (LLM_PER_SENTENCE / LLM_PER_WINDOW).
        #
        # An empty cls list is a VALID value ("no dialogue-act function"), so
        # it cannot also mean "not annotated". Missing sentences are recorded
        # in a per-sentence mask instead, the same way a non-tree sentence is
        # masked in dep_tokens rather than deleted.
        for layer in LLM_PER_SENTENCE:
            vals, miss = [], []
            for si, (s2, e2) in enumerate(w["sentence_spans"]):
                q = _get(cache, LLM_ANNOTATOR, layer,
                         item_key(w["window_id"], si, w["tokens"][s2:e2]),
                         e2 - s2)
                miss.append(q is None)
                vals.append(([] if layer == "cls" else None) if q is None else q)
            if all(miss):
                row[layer] = None
                mask[layer] = True
                stats[f"missing_{layer}"] += 1
            else:
                row[layer] = vals
                provenance[layer] = LLM_ANNOTATOR
                if any(miss):
                    mask[f"{layer}_sentences"] = miss
                    stats[f"{layer}_sentences_masked"] += sum(miss)
            stats[f"{layer}_sentences_total"] += len(w["sentence_spans"])

        for layer in LLM_PER_WINDOW:
            q = _get(cache, LLM_ANNOTATOR, layer, w["window_id"], n)
            row[layer] = q
            if q is None:
                mask[layer] = True
                stats[f"missing_{layer}"] += 1
            else:
                provenance[layer] = LLM_ANNOTATOR

        p = _get(cache, "lingmess", "coref", w["window_id"], n)
        row["coref"] = p
        if p is None:
            mask["coref"] = True; stats["missing_coref"] += 1
        else:
            provenance["coref"] = "lingmess"

        for lyr, a in provenance.items():
            sources.setdefault(lyr, set()).add(a)
        split_windows[row["split"]] += 1
        split_docs[row["split"]].add(w["doc_id"])
        row["provenance"] = json.dumps(provenance)
        row["loss_mask"] = json.dumps(mask)
        rows.append(row)
        stats["rows"] += 1

        if len(rows) >= shard_size:
            _write(rows, shard, pa, pq); rows, shard = [], shard + 1
    if rows:
        _write(rows, shard, pa, pq)

    # layer_source was built from PER_SENTENCE/PER_WINDOW, which describe the
    # annotate stage and not what actually landed in the rows: srl is handled
    # separately and the LLM layers are not in either table, so all four were
    # missing from the manifest. Deriving it from the provenance the rows
    # carry means a layer cannot go unrecorded -- gate 17.
    (OUT / "MANIFEST.json").write_text(json.dumps({
        "windows": stats["rows"], "shards": shard + (1 if rows else 0),
        "git_sha": _git_sha(),
        "layer_source": {l: sorted(a) for l, a in sorted(sources.items())},
        "annotator_versions": _annotator_versions(sources),
        "splits": {sp: {"windows": split_windows[sp],
                        "share": round(split_windows[sp]
                                       / max(sum(split_windows.values()), 1), 4),
                        "documents": len(split_docs[sp])}
                   for sp in sorted(split_windows)},
        "stats": dict(stats),
    }, indent=2))
    print(f"assembled {stats['rows']} rows -> {OUT}")
    print(f"  {dict(stats)}")


def assign_splits(windows: list[dict], dev_frac: float = 0.05,
                  test_frac: float = 0.05) -> dict[str, str]:
    """Map window_id -> split, splitting on doc_id and stratified by domain.

    Split on doc_id, never window_id (5.1): windows from one document share
    entities, coref chains and topic, so a window-level split leaks the test
    set into training.

    The 5% targets are measured in WINDOWS, not documents, because that is
    what training sees and document sizes vary by an order of magnitude
    across domains. Documents are ordered by a hash of their id and taken
    whole until the domain's window quota is met, so the assignment is
    deterministic, reproducible from the ids alone, and never splits a
    document.
    """
    by_domain: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for w in windows:
        by_domain[w["domain"]][w["doc_id"]] += 1

    out: dict[str, str] = {}
    doc_split: dict[str, str] = {}
    for domain, docs in by_domain.items():
        total = sum(docs.values())
        # A document's position is fixed by its id, so adding a domain or
        # re-running the build cannot reshuffle an existing split.
        order = sorted(docs, key=lambda d: hashlib.sha1(
            f"split:{d}".encode()).hexdigest())
        quota = {"test": total * test_frac, "dev": total * dev_frac}
        filled = {"test": 0, "dev": 0}
        for d in order:
            for sp in ("test", "dev"):
                if filled[sp] < quota[sp]:
                    doc_split[d] = sp
                    filled[sp] += docs[d]
                    break
            else:
                doc_split[d] = "train"
    for w in windows:
        out[w["window_id"]] = doc_split[w["doc_id"]]
    return out


def _encoder_tokenizer():
    """The DeBERTa tokenizer the model will actually use, or None.

    Gate 3 is specified against this tokenizer and not a whitespace proxy,
    which is the whole reason the overflow went unnoticed.
    """
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            "models/kniv-deberta-nlp-base-en-large")
    except Exception as exc:                                # noqa: BLE001
        print(f"  NOTE: encoder tokenizer unavailable ({type(exc).__name__}); "
              f"n_subword_tokens and encoder_word_limit will be null",
              flush=True)
        return None


# 512 positions minus [CLS] and [SEP].
ENCODER_BUDGET = 510


def _encoder_extent(tok, tokens: list[str], n: int) -> tuple[int | None, int]:
    """``(subword length, first word index the encoder cannot reach)``.

    The limit is ``n`` when the whole window fits, so a consumer can always
    mask ``[limit, n)`` without special-casing.
    """
    if tok is None:
        return None, n
    enc = tok(list(tokens), add_special_tokens=False, is_split_into_words=True)
    wid = enc.word_ids()
    if len(wid) <= ENCODER_BUDGET:
        return len(wid) + 2, n
    first = wid[ENCODER_BUDGET]
    return len(wid) + 2, (n if first is None else first)


def _git_sha() -> str:
    """The working-tree commit, or 'unknown' outside a checkout."""
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        sha = out.stdout.strip()
        if out.returncode != 0 or not sha:
            return "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10)
        return sha + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:                                       # noqa: BLE001
        return "unknown"


def _annotator_versions(sources: dict[str, set[str]]) -> dict[str, str]:
    """A resolvable version for every annotator that produced a layer.

    Gate 17 asks for versions, not names: 'kniv-v5' does not say which
    checkpoint and 'stanza' does not say which release. Where a version
    cannot be resolved this records "unresolved" rather than echoing the
    annotator's own name back, which would look like an answer.

    Best-effort by construction: assembly runs in one venv and the
    annotators ran in three, so an annotator's package may not be importable
    here. A local checkpoint is identified by its weights -- size and mtime,
    which distinguish retrained weights at the same path -- because the
    directory name alone does not change when the model is retrained.
    """
    import importlib.metadata as md
    PKG = {"stanza": "stanza", "lingmess": "fastcoref"}
    CKPT = {"kniv-v5": Path("models/kniv-deberta-nlp-base-en-large/model.pt"),
            "atlop-redocred": Path("data/re-docred/atlop-run/best.pt")}
    # The ATLOP run writes the selected dev F1 beside the weights, which
    # identifies the checkpoint far better than a path does.
    F1 = {"atlop-redocred": Path("data/re-docred/atlop-run/best.f1")}
    out = {}
    for a in sorted({x for aa in sources.values() for x in aa}):
        # A stamp written by the annotator's own venv beats anything we can
        # infer from here.
        stamp = Path(RUNS_DIR) / "_cache" / a / "VERSION"
        if stamp.exists():
            txt = stamp.read_text().strip()
            if txt:
                out[a] = txt
                continue
        try:
            spec = load_annotators([a]).get(a)
        except Exception:                                   # noqa: BLE001
            spec = None
        # A hosted model names its deployment, which is the version.
        model = getattr(spec, "model", "") or ""
        if model and a not in PKG:
            out[a] = model
            continue
        if a in PKG:
            try:
                out[a] = f"{PKG[a]}=={md.version(PKG[a])}"
                continue
            except Exception:                               # noqa: BLE001
                pass
        ck = CKPT.get(a)
        if ck is not None and ck.exists():
            st = ck.stat()
            desc = f"{ck.as_posix()} ({st.st_size} bytes, mtime {int(st.st_mtime)})"
            f1 = F1.get(a)
            if f1 is not None and f1.exists():
                desc += f" dev_f1={f1.read_text().strip()}"
            out[a] = desc
            continue
        out[a] = "unresolved"
    return out


def _write(rows, shard, pa, pq):
    """One part file per (split, domain), as section 5 lays out.

    A flat shard mixed splits and domains together, so a consumer had to read
    the whole corpus to train on one split.
    """
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for r in rows:
        groups[(r["split"], r["domain"])].append(r)
    for (sp, dom), part in sorted(groups.items()):
        d = OUT / "corpus" / f"split={sp}" / f"domain={dom}"
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(part), d / f"part-{shard:03d}.parquet")
    print(f"  wrote shard {shard:03d}: " + ", ".join(
        f"{sp}/{dom} {len(p)}" for (sp, dom), p in sorted(groups.items())),
        flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=["windows", "annotate", "llm", "entities",
                             "assemble", "srl"])
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
    if args.stage == "entities":
        stage_entities(windows, args.cache_dir)
        return 0
    if args.stage == "srl":
        asyncio.run(stage_srl(windows, args.cache_dir))
        return 0
    if args.stage == "llm":
        if not args.annotator:
            raise SystemExit("--annotator required")
        layers = [x.strip() for x in args.layers.split(",")] if args.layers else None
        asyncio.run(stage_llm(args.annotator, windows, args.cache_dir, layers))
        return 0
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

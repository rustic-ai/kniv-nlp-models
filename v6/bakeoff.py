"""Annotator bake-off: measure every candidate annotator against public gold.

Answers one question per layer — *can the LLM ensemble own this, or does it
stay with the kniv v5 teacher?* — with a number instead of a judgement call.

Public treebanks are used for **evaluation only**; they are not v6 training
data. That separation is what makes the result falsifiable.

Usage
-----
    # everything declared in annotators.yaml, 300 sentences per layer
    uv run python -m v6.bakeoff --limit 300

    # one annotator, one layer, cheap smoke test
    uv run python -m v6.bakeoff --annotators luna --layers pos --limit 25

    # see the call and cost estimate without spending anything
    uv run python -m v6.bakeoff --limit 300 --dry-run

Runs resume for free: responses are cached on disk, so re-invoking after an
interruption only issues the calls that are actually missing.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .config import LAYERS, RUNS_DIR, annotator_names, load_annotators
from .annotate import CacheStore, LLMAnnotator, RunLogger, tree_is_wellformed
from .gold import (load_ner_items, load_rel_items, load_srl_items,
                   load_ud_items)
from .gold.litbank import load_coref_items
from .prompts import PROMPT_VERSION
from .score import pairwise_agreement, score_layer

# Published v5 scores, measured on the FULL test sets (UD EWT 2,077
# sentences; PropBank EWT 1,269). Used only as a fallback: a bake-off runs on
# a sample, so comparing a 300-sentence LLM score against a 2,077-sentence
# headline mixes sampling noise into every delta. When kniv-v5 is run over the
# same items its measured score replaces these — see resolve_baselines().
V5_PUBLISHED = {
    "pos": 0.977, "dep": 0.944, "srl": 0.843, "ner": 0.889,
    "lemma": None, "morph": None,
}
# Populated per run: layer -> (score, source)
V5_BASELINE: dict[str, float | None] = dict(V5_PUBLISHED)
BASELINE_SOURCE: dict[str, str] = {k: "published" for k in V5_PUBLISHED}


def resolve_baselines(scores: list, imported: dict | None = None) -> None:
    """Prefer a same-sample v5 measurement over the published headline."""
    for layer, val in (imported or {}).items():
        V5_BASELINE[layer] = val
        BASELINE_SOURCE[layer] = "measured (imported)"
    for s in scores:
        if s.annotator == "kniv-v5":
            V5_BASELINE[s.layer] = s.primary
            BASELINE_SOURCE[s.layer] = "measured (same sample)"

UD_LAYERS = ("pos", "lemma", "morph", "dep")

GOLD_SOURCE = {
    "pos": "UD English EWT test", "lemma": "UD English EWT test",
    "morph": "UD English EWT test", "dep": "UD English EWT test",
    "ner": "OntoNotes 5.0 test (rebuilt)", "srl": "PropBank EWT test",
    "coref": "LitBank", "rel": "Re-DocRED test",
}


def make_annotator(spec, cache, max_repairs: int):
    """Dispatch on annotator kind.

    ``kniv-v5`` runs the local teacher checkpoint instead of an API. It is
    imported lazily so the LLM-only path never needs torch.
    """
    if spec.kind == "kniv-v5":
        from .annotate.kniv_v5 import KnivV5Annotator
        return KnivV5Annotator(spec, cache, **(spec.extra or {}))
    if spec.kind == "fastcoref":
        from .annotate.coref import FastCorefAnnotator
        # `model` is a top-level spec field, not part of `extra` — forward it
        # explicitly or every fastcoref annotator silently gets the default.
        return FastCorefAnnotator(spec, cache, model=spec.model,
                                  **(spec.extra or {}))
    if spec.kind == "toolkit":
        from .annotate.toolkit import TOOLKITS
        return TOOLKITS[spec.model](spec, cache, **(spec.extra or {}))
    return LLMAnnotator(spec, cache, max_repairs=max_repairs)


def load_gold(layers: tuple[str, ...], limit: int | None,
              srl_path: Path | None,
              ner_benchmark: str = "ontonotes5") -> dict[str, list]:
    gold: dict[str, list] = {}
    if any(lyr in UD_LAYERS for lyr in layers):
        items = load_ud_items("test", limit=limit)
        print(f"Gold: UD EWT test — {len(items)} sentences", flush=True)
        for lyr in layers:
            if lyr in UD_LAYERS:
                gold[lyr] = items
    if "ner" in layers:
        items = load_ner_items(ner_benchmark, limit=limit)
        print(f"Gold: {ner_benchmark} test — {len(items)} sentences", flush=True)
        gold["ner"] = items
    if "coref" in layers:
        items = load_coref_items(limit=limit)
        print(f"Gold: LitBank coref — {len(items)} windows", flush=True)
        gold["coref"] = items
    if "srl" in layers:
        items = load_srl_items(limit=limit, path=srl_path)
        print(f"Gold: PropBank EWT test — {len(items)} predicates", flush=True)
        gold["srl"] = items
    if "rel" in layers:
        items = load_rel_items(limit=limit)
        n_trip = sum(len(it.layers["rel"]) for it in items)
        n_ent = sum(len(it.entities) for it in items)
        print(f"Gold: Re-DocRED test — {len(items)} documents, "
              f"{n_ent} entities, {n_trip} triples", flush=True)
        gold["rel"] = items
    return gold


async def run_layer(annotator: LLMAnnotator, layer: str, items: list,
                    logger: RunLogger) -> tuple[dict, dict]:
    """Annotate every item; return (payloads by id, well-formed by id).

    A single warm-up item runs first so the client settles which request
    parameters the endpoint accepts (the response_format ladder, sampling
    args) before the parallel fan-out. Without it, every in-flight item
    during capability discovery spends retries on the same rejections.
    """
    price = (annotator.spec.price_in, annotator.spec.price_out)
    payloads: dict[str, object] = {}
    well_formed: dict[str, bool] = {}

    if items:
        warm = await annotator.annotate_and_cache(layer, items[0])
        logger.record(warm, price, len(items))
        if warm.ok:
            payloads[warm.item_id] = warm.payload
            if warm.well_formed is not None:
                well_formed[warm.item_id] = warm.well_formed

    tasks = [annotator.annotate_and_cache(layer, it) for it in items[1:]]
    for coro in asyncio.as_completed(tasks):
        res = await coro
        logger.record(res, price, len(items))
        if res.ok:
            payloads[res.item_id] = res.payload
            if res.well_formed is not None:
                well_formed[res.item_id] = res.well_formed
    return payloads, well_formed


def vote(layer: str, items: list, per_annotator: dict[str, dict]) -> tuple[dict, dict]:
    """Majority vote across annotators.

    Included because consensus is the proposed quality mechanism, so the
    bake-off should measure it directly rather than assume it helps. For
    ``dep`` the well-formedness of the *voted* tree is also recorded — a
    per-token majority over arcs is not guaranteed to be a tree at all.
    """
    voted: dict[str, object] = {}
    well_formed: dict[str, bool] = {}
    for it in items:
        available = [p[it.id] for p in per_annotator.values() if it.id in p]
        if len(available) < 2:
            continue
        n = len(it.tokens)
        if layer == "dep":
            heads, rels = [], []
            for i in range(n):
                h = Counter(a["heads"][i] for a in available).most_common(1)[0][0]
                r = Counter(a["rels"][i] for a in available).most_common(1)[0][0]
                heads.append(h)
                rels.append(r)
            voted[it.id] = {"heads": heads, "rels": rels}
            well_formed[it.id] = tree_is_wellformed(heads)
        elif layer == "rel":
            # Set-valued: a triple is kept when a majority of the annotators
            # that answered this item proposed it.
            counts = Counter(tuple(t) for a in available for t in set(map(tuple, a)))
            need = len(available) // 2 + 1
            voted[it.id] = [list(t) for t, c in counts.items() if c >= need]
        else:
            voted[it.id] = [
                Counter(a[i] for a in available).most_common(1)[0][0]
                for i in range(n)
            ]
    return voted, well_formed


def render_table(scores: list) -> str:
    head = (f"| {'annotator':<16} | {'layer':<6} | {'primary':>9} | "
            f"{'secondary':>9} | {'cover':>6} | {'tree-ok':>7} | {'vs v5':>7} |")
    sep = "|" + "|".join("-" * (w + 2) for w in (16, 6, 9, 9, 6, 7, 7)) + "|"
    rows = [head, sep]
    for s in scores:
        base = V5_BASELINE.get(s.layer)
        if s.annotator == "kniv-v5":
            delta = "baseline"
        else:
            delta = f"{s.primary - base:+.3f}" if base is not None else "—"
        sec = f"{s.secondary:.3f}" if s.secondary is not None else "—"
        wf = f"{s.well_formed_rate:.3f}" if s.well_formed_rate is not None else "—"
        rows.append(
            f"| {s.annotator:<16} | {s.layer:<6} | {s.primary:>9.3f} | "
            f"{sec:>9} | {s.coverage:>6.3f} | {wf:>7} | {delta:>7} |"
        )
    return "\n".join(rows)


async def main_async(args) -> int:
    layers = tuple(lyr.strip() for lyr in args.layers.split(",") if lyr.strip())
    unknown = set(layers) - set(LAYERS)
    if unknown:
        print(f"Unknown layers: {sorted(unknown)}. Known: {LAYERS}", file=sys.stderr)
        return 2

    declared = annotator_names()
    selected = ([s.strip() for s in args.annotators.split(",")]
                if args.annotators else declared)
    missing = [n for n in selected if n not in declared]
    if missing:
        print(f"Unknown annotators: {missing}. Declared: {sorted(declared)}",
              file=sys.stderr)
        return 2
    # Secrets are resolved only for the annotators actually selected.
    specs = load_annotators(selected)

    gold = load_gold(layers, args.limit, args.srl_path, args.ner_benchmark)

    families = {n: specs[n].family for n in selected}
    distinct = len(set(families.values()))
    print(f"Annotators: {selected}", flush=True)
    print(f"Families: {families} ({distinct} distinct)", flush=True)
    if distinct < len(selected):
        print("  NOTE: annotators sharing a family have correlated errors; "
              "the ensemble row is weaker than its vote count suggests.",
              flush=True)

    if args.dry_run:
        total = sum(len(gold[lyr]) for lyr in layers) * len(selected)
        print(f"\nDry run: {total} calls "
              f"({len(selected)} annotators x {sum(len(gold[lyr]) for lyr in layers)} items)")
        for n in selected:
            s = specs[n]
            print(f"  {n:<16} model={s.model} family={s.family} "
                  f"conc={s.max_concurrency} "
                  f"price=${s.price_in}/${s.price_out} per 1M")
        return 0

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_DIR / run_id
    cache = CacheStore(args.cache_dir, PROMPT_VERSION)
    logger = RunLogger(run_dir, every=args.log_every)
    print(f"\nRun {run_id} -> {run_dir}", flush=True)
    print(f"Cache: {args.cache_dir} (prompt {PROMPT_VERSION})", flush=True)

    scores, raw, agree = [], {}, {}
    try:
        for layer in layers:
            items = gold[layer]
            per_annotator: dict[str, dict] = {}
            print(f"\n=== layer: {layer} ({len(items)} items) ===", flush=True)
            for name in selected:
                if specs[name].kind == "fastcoref" and layer != "coref":
                    continue
                if specs[name].kind == "toolkit" and layer not in (
                        "pos", "lemma", "morph", "dep", "ner"):
                    print(f"  [{name}] no {layer} head — skipping", flush=True)
                    continue
                if specs[name].kind == "kniv-v5" and layer in (
                        "lemma", "morph", "coref", "rel"):
                    print(f"  [{name}] no {layer} head — skipping", flush=True)
                    continue
                ann = make_annotator(specs[name], cache, args.max_repairs)
                payloads, wf = await run_layer(ann, layer, items, logger)
                per_annotator[name] = payloads
                scores.append(score_layer(layer, name, items, payloads, wf))

            if len(per_annotator) >= 2 and layer != "coref":
                voted, wf = vote(layer, items, per_annotator)
                scores.append(score_layer(layer, "ENSEMBLE", items, voted, wf))
                agree[layer] = pairwise_agreement(layer, items, per_annotator)
                print(f"\n  pairwise agreement ({layer}):", flush=True)
                for pair, val in sorted(agree[layer].items(),
                                        key=lambda kv: -kv[1]):
                    a, b = pair.split("|")
                    tag = ("  <- same family"
                           if specs[a].family == specs[b].family else "")
                    print(f"    {a:>12} vs {b:<12} {val:.3f}{tag}", flush=True)
            raw[layer] = {n: len(p) for n, p in per_annotator.items()}
    finally:
        logger.close()

    imported = None
    if args.v5_baseline:
        prior = json.loads(Path(args.v5_baseline).read_text())
        imported = {x["layer"]: x["primary"] for x in prior["scores"]
                    if x["annotator"] == "kniv-v5"}
        print(f"\nImported v5 baseline from {args.v5_baseline}: "
              f"{ {k: round(v, 4) for k, v in imported.items()} }", flush=True)
    resolve_baselines(scores, imported)

    table = render_table(scores)
    print("\n" + table, flush=True)
    print("\nbaseline source: "
          + ", ".join(f"{k}={BASELINE_SOURCE[k]}" for k in layers
                      if V5_BASELINE.get(k) is not None), flush=True)

    report = {
        "run_id": run_id,
        "prompt_version": PROMPT_VERSION,
        "limit": args.limit,
        "annotators": {n: {"model": specs[n].model, "family": specs[n].family}
                       for n in selected},
        "v5_baseline": V5_BASELINE,
        "v5_baseline_source": BASELINE_SOURCE,
        "v5_published": V5_PUBLISHED,
        "scores": [s.to_dict() for s in scores],
        "counters": {f"{ann}/{lyr}": vars(c) for (ann, lyr), c
                     in logger.counters.items()},
        "covered": raw,
        "pairwise_agreement": agree,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    (run_dir / "report.md").write_text(
        f"# v6 annotator bake-off — {run_id}\n\n"
        f"Gold (evaluation only): "
        + ", ".join(GOLD_SOURCE[lyr] for lyr in layers) + ".\n"
        f"Prompt version: `{PROMPT_VERSION}`. Limit: {args.limit}.\n\n"
        f"{table}\n\n"
        f"`vs v5` is the delta against the v5 teacher. Baseline source: "
        + ", ".join(f"{k}={BASELINE_SOURCE[k]}" for k in layers
                    if V5_BASELINE.get(k) is not None)
        + ".\nA layer stays with v5 unless a candidate clears it.\n"
    )
    print(f"\nWrote {run_dir / 'report.json'} and report.md", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotators", default=None,
                    help="Comma-separated subset of annotators.yaml (default: all)")
    ap.add_argument("--layers", default=",".join(LAYERS),
                    help=f"Comma-separated subset of {LAYERS}")
    ap.add_argument("--limit", type=int, default=300,
                    help="Gold items per layer (default: 300)")
    ap.add_argument("--max-repairs", type=int, default=1,
                    help="Corrective turns allowed on a malformed response")
    ap.add_argument("--cache-dir", type=Path, default=RUNS_DIR / "_cache",
                    help="Shared response cache; enables free resume")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--ner-benchmark", default="ontonotes5",
                    choices=["ontonotes5", "conll2003"],
                    help="NER gold set (default: ontonotes5 — the schema v5 "
                         "was trained on)")
    ap.add_argument("--srl-path", type=Path, default=None,
                    help="Override path to srl_test.json")
    ap.add_argument("--v5-baseline", default=None,
                    help="report.json from a prior kniv-v5 run over the same "
                         "items; its scores replace the published headline "
                         "so deltas are same-sample")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show call counts and pricing without calling anything")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

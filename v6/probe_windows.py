"""Domain-transfer probe over the v6 windows we actually built.

Every annotator in `DECISIONS.md` was selected on a benchmark from someone
else's domain: UD EWT is web text, OntoNotes is news, LitBank is literary
prose, Re-DocRED is Wikipedia. Our corpus is 49% conversation. A score
measured on one does not transfer to the other by default, and the
relation annotator's 0.901 precision is a Wikipedia number.

There is no gold for our corpus, so this measures **inter-annotator
agreement**, not accuracy. It is read as a *delta*: agreement on our text
against agreement between the same two annotators on the benchmark. A
large drop means the benchmark number is not describing what the annotator
will do in production.

    uv run python -m v6.probe_windows --layer pos --annotators kniv-v5 --per-domain 60
    .venv-tools/bin/python -m v6.probe_windows --layer pos --annotators stanza --per-domain 60
    uv run python -m v6.probe_windows --layer pos --compare kniv-v5,stanza
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from .annotate.base import CacheStore
from .config import RUNS_DIR, load_annotators
from .gold.ud_ewt import GoldItem
from .prompts import PROMPT_VERSION
from .windows import build_windows, iter_documents, tokenize

DOMAINS = ("conversation", "narrative", "technical", "news", "encyclopedic",
           "business")
CACHE_VERSION = f"{PROMPT_VERSION}-probe"


def sample_windows(per_domain: int, max_tokens: int = 256,
                   seed: int = 7) -> list[GoldItem]:
    """Evenly sample windows across domains, deterministically.

    Capped below the full 512 because the probe runs annotators that are
    quadratic in length, and the question here is domain, not length.
    """
    import random
    items: list[GoldItem] = []
    for domain in DOMAINS:
        pool: list[dict] = []
        for doc in iter_documents(domain):
            pool.extend(build_windows(doc, tokenize, max_tokens=max_tokens))
            if len(pool) > per_domain * 20:
                break
        if not pool:
            continue
        rng = random.Random(seed)
        for w in rng.sample(pool, min(per_domain, len(pool))):
            items.append(GoldItem(id=f"{domain}:{w['window_id']}",
                                  tokens=w["tokens"], layers={}))
    return items


async def annotate(name: str, layer: str, items: list[GoldItem],
                   cache_dir: Path) -> dict:
    from .annotate import RunLogger
    from .bakeoff import make_annotator
    spec = load_annotators([name])[name]
    cache = CacheStore(cache_dir, CACHE_VERSION)
    ann = make_annotator(spec, cache, max_repairs=1)
    logger = RunLogger(RUNS_DIR / f"probe-{layer}-{name}", every=25)
    out = {}
    try:
        for it in items:
            res = await ann.annotate_and_cache(layer, it)
            logger.record(res, (0, 0), len(items))
            if res.ok:
                out[res.item_id] = res.payload
    finally:
        logger.close()
    return out


def load_cached(name: str, layer: str, items: list[GoldItem],
                cache_dir: Path) -> dict:
    from .annotate.base import validate_payload
    cache = CacheStore(cache_dir, CACHE_VERSION)
    out = {}
    for it in items:
        rec = cache.get(name, layer, it.id)
        if not rec or rec.get("payload") is None:
            continue
        p, err, _ = validate_payload(layer, rec["payload"], len(it.tokens))
        if err is None:
            out[it.id] = p
    return out


def compare(layer: str, items: list[GoldItem], a: dict, b: dict,
            names: tuple[str, str]) -> None:
    by_domain = defaultdict(lambda: [0, 0])
    for it in items:
        if it.id not in a or it.id not in b:
            continue
        dom = it.id.split(":", 1)[0]
        pa, pb = a[it.id], b[it.id]
        if layer == "dep":
            pairs = zip(pa["heads"], pa["rels"], pb["heads"], pb["rels"])
            for h1, r1, h2, r2 in pairs:
                by_domain[dom][1] += 1
                by_domain[dom][0] += (h1 == h2 and r1 == r2)
        else:
            for x, y in zip(pa, pb):
                by_domain[dom][1] += 1
                by_domain[dom][0] += (x == y)
    print(f"\n{names[0]} vs {names[1]} — {layer} agreement on v6 windows\n")
    print(f"{'domain':<16} {'agreement':>10} {'tokens':>10}")
    tot = [0, 0]
    for dom in DOMAINS:
        if dom not in by_domain:
            continue
        same, n = by_domain[dom]
        tot[0] += same; tot[1] += n
        print(f"{dom:<16} {same / n:>10.4f} {n:>10,}")
    if tot[1]:
        print(f"{'ALL':<16} {tot[0] / tot[1]:>10.4f} {tot[1]:>10,}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--layer", default="pos")
    ap.add_argument("--annotators", default=None, help="run these (comma-sep)")
    ap.add_argument("--compare", default=None, help="score two cached runs")
    ap.add_argument("--per-domain", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--cache-dir", type=Path, default=RUNS_DIR / "_cache")
    args = ap.parse_args()

    items = sample_windows(args.per_domain, args.max_tokens)
    got = defaultdict(int)
    for it in items:
        got[it.id.split(":", 1)[0]] += 1
    print(f"sampled {len(items)} windows: {dict(got)}", flush=True)

    if args.compare:
        n1, n2 = [s.strip() for s in args.compare.split(",")]
        a = load_cached(n1, args.layer, items, args.cache_dir)
        b = load_cached(n2, args.layer, items, args.cache_dir)
        print(f"cached: {n1}={len(a)} {n2}={len(b)}")
        compare(args.layer, items, a, b, (n1, n2))
        return 0

    for name in (args.annotators or "").split(","):
        name = name.strip()
        if not name:
            continue
        print(f"\n=== {name} / {args.layer} ===", flush=True)
        payloads = asyncio.run(annotate(name, args.layer, items, args.cache_dir))
        print(f"  {len(payloads)}/{len(items)} annotated", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

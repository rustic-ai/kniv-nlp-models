"""Domain-transfer probe: do benchmark agreement rates hold on OUR corpus?

The bake-off scores each annotator on the domain its benchmark comes from —
UD EWT is web text, OntoNotes is news, LitBank is literary prose. The v6
corpus is conversation, business and technical text. A number measured on
one does not automatically transfer to the other.

There is no gold for our corpus, so this measures **inter-annotator
agreement**, not accuracy. Read it as a transfer signal: if v5 and Stanza
agree at 97.7% on UD EWT but 88% on conversational text, the benchmark
number is not describing what the annotators will do in production.

Usage:
    # each annotator runs in whichever venv it needs; the cache is shared
    uv run python -m v6.probe_corpus --annotator kniv-v5 --per-domain 100
    .venv-tools/bin/python -m v6.probe_corpus --annotator stanza --per-domain 100
    uv run python -m v6.probe_corpus --compare kniv-v5,stanza
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import re
from pathlib import Path

from .annotate.base import CacheStore
from .config import RUNS_DIR, load_annotators
from .gold.ud_ewt import GoldItem
from .prompts import PROMPT_VERSION

CORPUS_REPO = "dragonscale-ai/kniv-corpus-en"
CORPUS_FILE = "corpus/gold/test.parquet"


def load_corpus(per_domain: int = 100, max_tokens: int = 128) -> list[GoldItem]:
    """Sample sentences evenly across corpus domains."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    tbl = pq.read_table(hf_hub_download(CORPUS_REPO, CORPUS_FILE,
                                        repo_type="dataset"))
    by_domain: dict[str, list[GoldItem]] = collections.defaultdict(list)
    for row in tbl.select(["sent_id", "tokens", "pos_tags"]).to_pylist():
        toks = row["tokens"]
        if not toks or len(toks) > max_tokens or len(toks) < 3:
            continue
        domain = re.split(r"[-_]", row["sent_id"])[0]
        if len(by_domain[domain]) >= per_domain:
            continue
        by_domain[domain].append(GoldItem(
            id=f"corpus-{row['sent_id']}",
            tokens=toks,
            # spaCy's own POS, kept for reference — not gold, just a third opinion
            layers={"pos_spacy": row["pos_tags"]},
        ))
    items = [it for d in sorted(by_domain) for it in by_domain[d]]
    print(f"sampled {len(items)} sentences across {len(by_domain)} domains: "
          f"{ {d: len(v) for d, v in sorted(by_domain.items())} }", flush=True)
    return items


def domain_of(item: GoldItem) -> str:
    return re.split(r"[-_]", item.id[len("corpus-"):])[0]


async def annotate(name: str, items: list[GoldItem], cache: CacheStore) -> None:
    from .bakeoff import make_annotator
    spec = load_annotators([name])[name]
    ann = make_annotator(spec, cache, 1)
    ok = fail = 0
    for i, it in enumerate(items, 1):
        res = await ann.annotate_and_cache("pos", it)
        ok += res.ok
        fail += not res.ok
        if i % 50 == 0:
            print(f"  [{name}] {i}/{len(items)} ok={ok} fail={fail}", flush=True)
    print(f"  [{name}] done: ok={ok} fail={fail}", flush=True)


def compare(names: list[str], items: list[GoldItem], cache: CacheStore) -> None:
    def load(ann, it):
        f = cache.path(ann, "pos", it.id)
        return json.loads(f.read_text())["payload"] if f.exists() else None

    a, b = names
    per_domain = collections.defaultdict(lambda: [0, 0])   # same, total
    vs_spacy = collections.defaultdict(lambda: [0, 0, 0])  # a==sp, b==sp, total
    for it in items:
        pa, pb = load(a, it), load(b, it)
        if not pa or not pb or len(pa) != len(pb):
            continue
        d = domain_of(it)
        sp = it.layers.get("pos_spacy") or []
        for k, (x, y) in enumerate(zip(pa, pb)):
            per_domain[d][1] += 1
            per_domain[d][0] += (x == y)
            if k < len(sp):
                vs_spacy[d][2] += 1
                vs_spacy[d][0] += (x == sp[k])
                vs_spacy[d][1] += (y == sp[k])

    print(f"\nPOS agreement, {a} vs {b}, by corpus domain")
    print(f"{'domain':<16}{'agree':>9}{'tokens':>9}   "
          f"{a + ' vs spaCy':>18}{b + ' vs spaCy':>18}")
    tot_s = tot_n = 0
    for d in sorted(per_domain):
        s, n = per_domain[d]
        xa, xb, nn = vs_spacy[d]
        tot_s += s
        tot_n += n
        print(f"{d:<16}{s / n:>9.4f}{n:>9,}   "
              f"{xa / nn if nn else 0:>18.4f}{xb / nn if nn else 0:>18.4f}")
    print(f"{'ALL':<16}{tot_s / tot_n:>9.4f}{tot_n:>9,}")
    print("\nreference: same pair on UD EWT test = 0.9772")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--annotator")
    ap.add_argument("--compare")
    ap.add_argument("--per-domain", type=int, default=100)
    ap.add_argument("--cache-dir", type=Path, default=RUNS_DIR / "_cache")
    args = ap.parse_args()

    items = load_corpus(args.per_domain)
    cache = CacheStore(args.cache_dir, PROMPT_VERSION)
    if args.annotator:
        asyncio.run(annotate(args.annotator, items, cache))
    if args.compare:
        compare([n.strip() for n in args.compare.split(",")], items, cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

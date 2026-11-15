"""Judged news text for the verifier (plan r4, approved 2026-11-14).

    uv run python -m v6.news_v2 fetch --articles 700        # Wikinews articles the corpus has not used
    uv run python -m v6.news_v2 texts                       # short texts with schemas, split by article
    uv run python -m v6.news_v2 stats

Source: English Wikinews (CC BY 2.5) through the MediaWiki API, as the
corpus's news collector (``corpus/domains/news``) reads it, with a generic
client name; only main-namespace, non-redirect articles whose title is not
one of the corpus's 477 Wikinews documents, sampled with seed 7 from the full
title list. Articles are kept in ``data/v6-corpus/news_v2/articles.jsonl``
(resumable).

Texts: each article's paragraphs (cleaned and kept as prose by the corpus's
own rules), split into sentences with the release runtime's splitter and
tokenized as the corpus is; cut into consecutive 1-3 sentence pieces as
``kg_short`` round 4 cuts a window (at most ``MAX_WORDS`` words, every
sentence ``MIN_SENT_WORDS`` or more, at least two entity or value cues), at
most ``PER_ARTICLE`` per article; a piece sharing an 8-gram with CoNLL04 is
dropped. Schemas and types from ``kg_short``'s sampler with the news
domain's prior. Split by article (a seventh held out, ``eval``: the second
news evaluation set, never trained on). Written to ``news_v2.json``
(``{"train": [...], "eval": [...]}``, ``heldout_zs`` items with no gold),
read by ``m1_precision`` as the sets ``news:train`` and ``news:eval``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from .config import DATA_DIR

DIR = DATA_DIR / "v6-corpus" / "news_v2"
ARTICLES = DIR / "articles.jsonl"
TITLES = DIR / "titles.json"
OUT = DATA_DIR / "v6-corpus" / "news_v2.json"
API = "https://en.wikinews.org/w/api.php"
HEADERS = {"User-Agent": "kniv-nlp-models/1.0 (research text collection)"}
PER_ARTICLE = 3
EVAL_FRAC = 7          # one article in EVAL_FRAC goes to the evaluation set


def _get(params: dict) -> dict:
    import requests
    for k in range(8):
        try:
            r = requests.get(API, params={**params, "format": "json", "maxlag": "5"}, headers=HEADERS, timeout=30)
            if r.status_code == 429:            # rate limited: wait as told, or back off
                time.sleep(float(r.headers.get("Retry-After") or 0) or min(5 * 2 ** k, 120))
                last = RuntimeError("429")
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001 - retried
            time.sleep(min(5 * 2 ** k, 120))
            last = e
    raise RuntimeError(f"Wikinews API failed: {last}")


def _all_titles() -> list[str]:
    if TITLES.exists():
        return json.loads(TITLES.read_text())
    titles, cont = [], None
    while True:
        p = {"action": "query", "list": "allpages", "apnamespace": "0", "aplimit": "500",
             "apfilterredir": "nonredirects"}
        if cont:
            p["apcontinue"] = cont
        d = _get(p)
        titles += [x["title"] for x in d.get("query", {}).get("allpages", [])]
        cont = d.get("continue", {}).get("apcontinue")
        if not cont:
            break
        time.sleep(1.0)
    DIR.mkdir(parents=True, exist_ok=True)
    TITLES.write_text(json.dumps(titles))
    return titles


def _used_titles() -> set[str]:
    used = set()
    for line in open(DATA_DIR / "v6-corpus" / "windows.jsonl"):
        w = json.loads(line)
        if w["source"] == "wikinews":
            used.add(w["doc_id"].split("/", 1)[1])
    return used


def fetch(n: int) -> int:
    titles = _all_titles()
    used = _used_titles()
    pool = sorted(t for t in titles if t not in used)
    random.Random(7).shuffle(pool)
    have = {}
    if ARTICLES.exists():
        for line in open(ARTICLES):
            a = json.loads(line)
            have[a["title"]] = a
    print(f"{len(titles):,} Wikinews articles listed, {len(used)} in the corpus, {len(pool):,} unused; "
          f"{len(have)} fetched already", flush=True)
    DIR.mkdir(parents=True, exist_ok=True)
    kept = sum(1 for a in have.values() if a.get("text"))
    with open(ARTICLES, "a") as f:
        for t in pool:
            if kept >= n:
                break
            if t in have:
                continue
            d = _get({"action": "query", "titles": t, "prop": "extracts|categories", "explaintext": "1",
                      "cllimit": "max"})
            page = next(iter(d.get("query", {}).get("pages", {}).values()), {})
            cats = [c["title"] for c in page.get("categories", [])]
            text = page.get("extract") or ""
            ok = len(text) > 400 and any(c.endswith(":Published") for c in cats)
            f.write(json.dumps({"title": t, "source": "wikinews", "text": text if ok else "",
                                "published": ok}) + "\n")
            f.flush()
            kept += ok
            if kept % 50 == 0 and ok:
                print(f"  {kept} articles kept", flush=True)
            time.sleep(0.5)
    print(f"{kept} published articles of more than 400 characters in {ARTICLES}")
    return 0


def _split(title: str) -> str:
    return "eval" if int(hashlib.sha1(f"news_v2:{title}".encode()).hexdigest()[:8], 16) % EVAL_FRAC == 0 else "train"


def texts() -> int:
    from .kg_short import MAX_WORDS, MIN_SENT_WORDS, ShortSampler, _grams, cues, heldout_grams
    from .kg_teacher import SchemaSampler, _rng
    from .label_bank import load as load_bank
    from .windows import clean_text, looks_like_prose, split_sentences, tokenize
    bank = load_bank()
    sampler = ShortSampler(SchemaSampler(bank))
    held, what = heldout_grams()
    print(f"overlap check: {what}", flush=True)
    out = {"train": [], "eval": []}
    n_art, n_overlap = 0, 0
    for line in open(ARTICLES):
        a = json.loads(line)
        if not a.get("text"):
            continue
        n_art += 1
        sents = []
        for para in a["text"].split("\n"):
            para = clean_text(para)
            if not para or not looks_like_prose(para):
                continue
            sents += [tokenize(s) for s in split_sentences(para)]
        rng = _rng(f"news_v2:{a['title']}")
        pieces, i = [], 0
        while i < len(sents):
            L = min(int(rng.integers(1, 4)), len(sents) - i)
            pieces.append((i, L))
            i += L
        order = sorted(range(len(pieces)), key=lambda k: _rng(f"o:{a['title']}:{k}").random())
        got = 0
        for k in order:
            if got >= PER_ARTICLE:
                break
            s0, L = pieces[k]
            group = sents[s0:s0 + L]
            if any(len(s) < MIN_SENT_WORDS for s in group) or sum(map(len, group)) > MAX_WORDS:
                continue
            tokens, spans = [], []
            for s in group:
                spans.append([len(tokens), len(tokens) + len(s)])
                tokens += s
            if cues(tokens, spans) < 2:
                continue
            if held and _grams(tokens) & held:
                n_overlap += 1
                continue
            key = f"news_v2/{a['title']}#{s0}-{s0 + L}"
            rel_ids, mode = sampler.relations_for(key, "news")
            rels = [sampler.b.relations[j] for j in rel_ids]
            types = sampler.types_for(key)
            out[_split(a["title"])].append({
                "id": key, "tokens": tokens, "sentence_spans": spans, "entities": [], "participants": [],
                "speakers": None, "facts": [],
                "relations": [{"name": r["name"], "description": r["description"]} for r in rels],
                "types": [{"name": t["name"], "description": t.get("description", "")} for t in types],
                "domain": "news", "mode": mode, "article": a["title"]})
            got += 1
    for sp in out:          # spread over articles: one text per article first, then seconds, ...
        rank: dict = {}
        for it in out[sp]:
            rank[it["id"]] = sum(1 for x in out[sp] if x["article"] == it["article"] and x["id"] < it["id"])
        out[sp].sort(key=lambda it: (rank[it["id"]], hashlib.sha1(it["id"].encode()).hexdigest()))
    OUT.write_text(json.dumps(out))
    print(f"{n_art} articles: {len(out['train'])} training texts, {len(out['eval'])} evaluation texts "
          f"({n_overlap} pieces dropped for a CoNLL04 8-gram) -> {OUT}")
    return 0


def load(split: str, n: int | None = None) -> list[dict]:
    items = json.loads(OUT.read_text())[split]
    return items[:n] if n else items


def stats() -> int:
    from collections import Counter
    D = json.loads(OUT.read_text())
    for sp, items in D.items():
        rel = Counter(r["name"] for it in items for r in it["relations"])
        print(f"{sp}: {len(items)} texts from {len({it['article'] for it in items})} articles, "
              f"{sum(len(it['tokens']) for it in items) / max(len(items), 1):.0f} words, "
              f"{sum(len(it['relations']) for it in items) / max(len(items), 1):.1f} relations per text; "
              f"modes {dict(Counter(it['mode'] for it in items))}; 'located in' in {rel.get('located in', 0)} schemas")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["fetch", "texts", "stats"])
    ap.add_argument("--articles", type=int, default=700)
    a = ap.parse_args()
    if a.cmd == "fetch":
        return fetch(a.articles)
    return texts() if a.cmd == "texts" else stats()


if __name__ == "__main__":
    raise SystemExit(main())

"""Wikidata classes for entities: what each entity is an instance of (P31),
the classes' parents (P279), and the classes' English labels and
descriptions -- the entity-type vocabulary for type pretraining (v7).

Fetched from the public SPARQL endpoint in batches (CC0 data; no dump),
cached on disk so a rerun only asks for what is new:

    data/wikidata/types_cache.jsonl   one line per entity or class fetched

``instance_of(qids)`` -> {qid: [class qids]}; ``classes(qids)`` -> {class qid:
{"label", "description", "parents"}}.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from .config import DATA_DIR

ENDPOINT = "https://query.wikidata.org/sparql"
AGENT = "kniv-nlp-models/0.7 (entity-type pretraining data; contact rohit@dragonscale.ai)"
CACHE = DATA_DIR / "wikidata" / "types_cache.jsonl"
BATCH = 400


def _load(path: Path = CACHE) -> dict:
    out = {}
    if path.exists():
        for line in path.open():
            x = json.loads(line)
            out[(x["kind"], x["id"])] = x["value"]
    return out


def _query(sparql: str, tries: int = 6) -> list[dict]:
    for k in range(tries):
        try:
            r = requests.post(ENDPOINT, data={"query": sparql, "format": "json"},
                              headers={"User-Agent": AGENT, "Accept": "application/sparql-results+json"},
                              timeout=90)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(int(r.headers.get("Retry-After", 10 * (k + 1))))
                continue
            r.raise_for_status()
            return r.json()["results"]["bindings"]
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(10 * (k + 1))
    raise RuntimeError("Wikidata SPARQL kept failing")


def _fetch(kind: str, ids: list[str], build, cache: dict, progress=None) -> dict:
    todo = [i for i in dict.fromkeys(ids) if (kind, i) not in cache]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a") as fh:
        for k in range(0, len(todo), BATCH):
            chunk = todo[k:k + BATCH]
            got = build(chunk)
            for i in chunk:
                v = got.get(i, [] if kind == "p31" else None)
                cache[(kind, i)] = v
                fh.write(json.dumps({"kind": kind, "id": i, "value": v}) + "\n")
            fh.flush()
            if progress:
                progress(f"wikidata {kind}: {min(k + BATCH, len(todo)):,}/{len(todo):,}")
            time.sleep(0.5)                     # be polite to the public endpoint
    return {i: cache[(kind, i)] for i in ids}


def instance_of(qids: list[str], progress=None) -> dict[str, list[str]]:
    cache = _load()

    def build(chunk):
        vals = " ".join(f"wd:{q}" for q in chunk)
        rows = _query(f"SELECT ?e ?c WHERE {{ VALUES ?e {{ {vals} }} ?e wdt:P31 ?c . }}")
        out: dict[str, list[str]] = {}
        for b in rows:
            e, c = b["e"]["value"].rsplit("/", 1)[1], b["c"]["value"].rsplit("/", 1)[1]
            out.setdefault(e, []).append(c)
        return out
    return _fetch("p31", qids, build, cache, progress)


def classes(qids: list[str], progress=None) -> dict[str, dict | None]:
    cache = _load()

    def build(chunk):
        vals = " ".join(f"wd:{q}" for q in chunk)
        rows = _query(f"""SELECT ?c ?label ?desc ?parent WHERE {{ VALUES ?c {{ {vals} }}
            OPTIONAL {{ ?c rdfs:label ?label . FILTER(LANG(?label) = "en") }}
            OPTIONAL {{ ?c schema:description ?desc . FILTER(LANG(?desc) = "en") }}
            OPTIONAL {{ ?c wdt:P279 ?parent . }} }}""")
        out: dict[str, dict] = {}
        for b in rows:
            c = b["c"]["value"].rsplit("/", 1)[1]
            x = out.setdefault(c, {"label": None, "description": None, "parents": []})
            if "label" in b:
                x["label"] = b["label"]["value"]
            if "desc" in b:
                x["description"] = b["desc"]["value"]
            if "parent" in b:
                p = b["parent"]["value"].rsplit("/", 1)[1]
                if p not in x["parents"]:
                    x["parents"].append(p)
        return out
    return _fetch("class", qids, build, cache, progress)

"""Re-DocRED document-level relation extraction gold.

Re-DocRED is the corrected release of DocRED. The original has systematic
false negatives — annotators marked only a fraction of the true triples — and
models score roughly 13 F1 higher on the revised set. Using the original
would understate every annotator here by about that much, in a way that looks
like model error rather than data error. Same class of defect as the
OntoNotes tag-id map rebuilt in ``build_ner_gold.py``.

Entities arrive already clustered (DocRED's ``vertexSet`` is a coref chain
per entity), which is exactly the v6 pipeline's stage-2 shape: clusters are
supplied, the annotator classifies relations over them.

Relation ids are Wikidata property codes (``P108``). They are resolved to
readable English labels via the Wikidata API — annotators are given names,
not opaque codes — and the resolved map is cached in
``data/re-docred/rel_info.json``.

Source: https://github.com/tonytan48/Re-DocRED (MIT). Evaluation only.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from ..config import DATA_DIR
from .ud_ewt import GoldItem

REDOCRED_DIR = DATA_DIR / "re-docred"
RAW_BASE = "https://raw.githubusercontent.com/tonytan48/Re-DocRED/main/data"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"


def download(split: str = "test") -> Path:
    REDOCRED_DIR.mkdir(parents=True, exist_ok=True)
    dst = REDOCRED_DIR / f"{split}_revised.json"
    if not dst.exists() or dst.stat().st_size == 0:
        with urllib.request.urlopen(f"{RAW_BASE}/{split}_revised.json",
                                    timeout=120) as r:
            dst.write_bytes(r.read())
    return dst


def resolve_relation_names(codes: list[str]) -> dict[str, str]:
    """Wikidata property code -> English label, cached on disk.

    Resolved from Wikidata itself rather than a copied lookup table: the
    mapping is then verifiable against a primary source, and a code that
    fails to resolve is loud instead of silently becoming its own id.
    """
    cache = REDOCRED_DIR / "rel_info.json"
    known: dict[str, str] = {}
    if cache.exists():
        known = json.loads(cache.read_text())
    missing = [c for c in codes if c not in known]
    for i in range(0, len(missing), 50):
        batch = missing[i:i + 50]
        url = (f"{WIKIDATA_API}?action=wbgetentities&format=json&props=labels"
               f"&languages=en&ids={'|'.join(batch)}")
        req = urllib.request.Request(url, headers={"User-Agent": "kniv-v6/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.load(r)
        for code, ent in payload.get("entities", {}).items():
            label = ent.get("labels", {}).get("en", {}).get("value")
            if label:
                known[code] = label
        time.sleep(0.4)
    still_missing = [c for c in codes if c not in known]
    if still_missing:
        raise RuntimeError(f"Wikidata did not resolve: {still_missing}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(known, indent=1, sort_keys=True))
    return known


def relation_inventory(splits: tuple[str, ...] = ("test", "dev")) -> tuple[list[str], dict[str, str]]:
    """Return (sorted relation names, name -> Wikidata code).

    Taken over the union of splits, not the evaluation split alone. Only 95
    of DocRED's 96 relations occur in test; restricting the annotator's
    label enum to the 95 would leak which relations the test set contains.
    """
    codes = sorted({lab["r"] for sp in splits
                    for d in json.loads(download(sp).read_text())
                    for lab in d["labels"]})
    names = resolve_relation_names(codes)
    by_name = {names[c]: c for c in codes}
    if len(by_name) != len(codes):
        raise RuntimeError("two Wikidata codes share an English label")
    return sorted(by_name), by_name


def load_rel_items(split: str = "test", limit: int | None = None,
                   max_tokens: int = 512,
                   min_entities: int = 2) -> list[GoldItem]:
    """Load Re-DocRED documents as relation-extraction gold items.

    Documents longer than ``max_tokens`` are **dropped, not truncated** —
    truncating would silently delete gold triples whose arguments fall off
    the end and charge the annotator for the miss.
    """
    docs = json.loads(download(split).read_text())
    _, by_name = relation_inventory()
    names = {code: name for name, code in by_name.items()}

    items: list[GoldItem] = []
    for di, doc in enumerate(docs):
        sents = doc["sents"]
        offsets, tokens = [], []
        for sent in sents:
            offsets.append(len(tokens))
            tokens.extend(sent)
        if len(tokens) > max_tokens or len(doc["vertexSet"]) < min_entities:
            continue

        entities = []
        for ei, chain in enumerate(doc["vertexSet"]):
            mentions, surface = [], []
            for m in chain:
                base = offsets[m["sent_id"]]
                mentions.append([base + m["pos"][0], base + m["pos"][1] - 1])
                surface.append(m["name"])
            entities.append({
                "id": ei,
                "name": surface[0],
                "type": chain[0]["type"],
                "aliases": sorted(set(surface)),
                "mentions": sorted(mentions),
            })

        triples = sorted({(lab["h"], lab["t"], names[lab["r"]])
                          for lab in doc["labels"]})
        items.append(GoldItem(
            id=f"redocred-{split}-{di:04d}",
            tokens=tokens,
            layers={"rel": [list(t) for t in triples]},
            entities=entities,
        ))
        if limit and len(items) >= limit:
            break
    return items

"""T-REx (Elsahar et al., 2018; CC BY 4.0 on figshare, CC BY-SA 4.0 on the
project page): Wikipedia abstracts with entity mentions linked to Wikidata
and Wikidata facts aligned to the sentences -- stage A pretraining for typed
entity spans and relations (v7).

    uv run python -m v6.gold.trex convert       # data/trex/raw/*.json -> data/trex/docs.jsonl
    uv run python -m v6.gold.trex types         # Wikidata classes for the linked entities
    uv run python -m v6.gold.trex export --docs 100000   # -> data/v6-corpus/kg_trex.json

``convert`` maps T-REx's character offsets onto the corpus word tokenizer
(``windows.tokenize``), one sentence at a time, and groups mentions into
entities by Wikidata id (dates by their text): the entity linker's mentions
are typed spans, the coreference annotator's pronouns are relation mentions
only, and the property linker's (words such as "born") are not entities.
Facts are (head, tail, property) between two entities of the document.

Leakage: a document is dropped if its title is a Re-DocRED dev or test
document, or it shares a 10-word sequence with a FewRel, Few-NERD dev/test or
CrossNER test sentence -- all are built from Wikipedia too. Facts on the
never-trained relations (``label_bank``: ``train: false``) are dropped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..config import DATA_DIR

TREX = DATA_DIR / "trex"
RAW = TREX / "raw"
DOCS = TREX / "docs.jsonl"
OUT = DATA_DIR / "v6-corpus" / "kg_trex.json"
NGRAM = 10
ENTITY = "Wikidata_Spotlight_Entity_Linker"
DATE = "Date_Linker"
COREF = "Simple_Coreference"


def _norm_words(tokens) -> list[str]:
    return [w for w in (re.sub(r"[^a-z0-9]", "", t.lower()) for t in tokens) if w]


def _grams(words: list[str]) -> set[str]:
    return {" ".join(words[i:i + NGRAM]) for i in range(len(words) - NGRAM + 1)}


def eval_fingerprints() -> tuple[set[str], set[str]]:
    """Titles and word 10-grams of every evaluation (and FewRel) sentence."""
    from . import fewnerd, fewrel
    from .redocred import download
    titles = {d["title"].strip().lower() for sp in ("dev", "test")
              for d in json.loads(download(sp).read_text())}
    sents = [x["tokens"] for sp in ("train", "val") for x in fewrel.load(sp)]
    sents += [x["tokens"] for sp in ("dev", "test") for x in fewnerd.load(sp)]
    from ..train.span_eval import DOMAINS, load_crossner
    sents += [x["tokens"] for d in DOMAINS for x in load_crossner(d)]
    grams: set[str] = set()
    for s in sents:
        grams |= _grams(_norm_words(s))
    return titles, grams


def _align(text: str, start: int, end: int):
    """Words of text[start:end] with their absolute character offsets."""
    from ..windows import tokenize
    toks, offs, cur = [], [], start
    for t in tokenize(text[start:end]):
        i = text.find(t, cur, end)
        if i < 0:
            return None
        toks.append(t)
        offs.append((i, i + len(t)))
        cur = i + len(t)
    return toks, offs


def convert_doc(x: dict, stats: Counter) -> dict | None:
    text = x["text"]
    tokens, offs, sspans = [], [], []
    for a, b in x["sentences_boundaries"]:
        r = _align(text, a, b)
        if r is None:
            stats["sentences_unaligned"] += 1
            continue
        if r[0]:
            sspans.append([len(tokens), len(tokens) + len(r[0])])
            tokens += r[0]
            offs += r[1]
    if len(sspans) < 1 or len(tokens) < 8:
        stats["docs_too_short"] += 1
        return None
    starts = {s: i for i, (s, _) in enumerate(offs)}
    ends = {e: i for i, (_, e) in enumerate(offs)}

    def words(bounds):
        if not bounds:
            return None
        a, b = bounds
        i, j = starts.get(a), ends.get(b)
        return (i, j + 1) if i is not None and j is not None and j >= i else None

    ents: dict[str, dict] = {}
    span_key: dict[tuple, str] = {}
    for e in x["entities"]:
        ann = e["annotator"]
        if ann not in (ENTITY, DATE, COREF):
            continue
        ws = words(e["boundaries"])
        if ws is None:
            stats["mentions_unaligned"] += 1
            continue
        uri = e["uri"] or ""
        if ann == DATE:
            key = "date:" + " ".join(tokens[ws[0]:ws[1]]).lower()
        elif "wikidata.org/entity/Q" in uri:
            key = uri.rsplit("/", 1)[1]
        else:
            continue
        ent = ents.setdefault(key, {"key": key, "kind": "date" if ann == DATE else "item",
                                    "mentions": set(), "typed": set()})
        ent["mentions"].add(ws)
        if ann != COREF:
            ent["typed"].add(ws)              # pronouns are not typed spans
        span_key[ws] = key
    facts = set()
    for t in x["triples"]:
        p = (t["predicate"].get("uri") or "").rsplit("/", 1)[-1]
        if not p.startswith("P"):
            continue
        sides = []
        for side in (t["subject"], t["object"]):
            uri = side.get("uri") or ""
            if "wikidata.org/entity/Q" in uri and uri.rsplit("/", 1)[1] in ents:
                sides.append(uri.rsplit("/", 1)[1])
            else:
                ws = words(side.get("boundaries"))
                sides.append(span_key.get(ws) if ws else None)
        if None in sides or sides[0] == sides[1]:
            stats["facts_unaligned"] += 1
            continue
        facts.add((sides[0], sides[1], p))
    keys = list(ents)
    index = {k: i for i, k in enumerate(keys)}
    stats["docs"] += 1
    stats["facts"] += len(facts)
    return {"id": f"trex:{x['docid']}", "title": x["title"], "tokens": tokens, "sentence_spans": sspans,
            "entities": [{"key": k, "kind": ents[k]["kind"],
                          "mentions": sorted(map(list, ents[k]["mentions"])),
                          "typed": sorted(map(list, ents[k]["typed"]))} for k in keys],
            "facts": sorted([index[h], index[t], p] for h, t, p in facts)}


def convert(files: list[Path]) -> Counter:
    titles, grams = eval_fingerprints()
    print(f"leak filter: {len(titles):,} titles, {len(grams):,} 10-grams", flush=True)
    stats: Counter = Counter()
    DOCS.parent.mkdir(parents=True, exist_ok=True)
    with DOCS.open("w") as fh:
        for f in files:
            for x in json.loads(f.read_text()):
                if x["title"].strip().lower() in titles:
                    stats["dropped_eval_title"] += 1
                    continue
                d = convert_doc(x, stats)
                if d is None:
                    continue
                if _grams(_norm_words(d["tokens"])) & grams:
                    stats["dropped_eval_overlap"] += 1
                    stats["docs"] -= 1
                    continue
                fh.write(json.dumps(d) + "\n")
            print(f"{f.name}: {dict(stats)}", flush=True)
    return stats


TYPES = DATA_DIR / "wikidata" / "trex_types.json"
MIN_ENTITIES = 20          # a class becomes an entity type with >= 20 typed entities
INTERNAL = ("wikimedia", "wikidata", "disambiguation", "category", "template", "list article")


def build_types(min_entities: int = MIN_ENTITIES) -> dict:
    """Each typed entity's classes -- its P31 classes and their direct P279
    parents -- and the type vocabulary: classes with an English label held
    by at least ``min_entities`` entities of the converted documents."""
    from .. import wikidata_types as wd
    qids = sorted({e["key"] for line in DOCS.open() for e in json.loads(line)["entities"]
                   if e["kind"] == "item" and e["typed"]})
    print(f"{len(qids):,} linked entities with typed mentions", flush=True)
    p31 = wd.instance_of(qids, progress=lambda s: print(s, flush=True))
    direct = sorted({c for v in p31.values() for c in v})
    info = wd.classes(direct, progress=lambda s: print(s, flush=True))
    parents = sorted({p for v in info.values() if v for p in v["parents"]} - set(info))
    info.update(wd.classes(parents, progress=lambda s: print(s, flush=True)))
    # A parent class counts only if Wikidata also uses it directly as an
    # entity's class often enough: "organization" yes; "omnivore" (a parent
    # of "human") no. Wikimedia-internal classes are never types.
    direct_n = Counter(c for q in qids for c in set(p31.get(q) or []))

    def internal(c):
        lab = ((info.get(c) or {}).get("label") or "").lower()
        return any(w in lab for w in INTERNAL)
    ent_classes = {}
    count: Counter = Counter()
    for q in qids:
        direct = {c for c in (p31.get(q) or []) if not internal(c)}
        cs = direct | {p for c in direct for p in ((info.get(c) or {}).get("parents") or [])
                       if direct_n[p] >= min_entities and not internal(p)}
        ent_classes[q] = sorted(cs)
        count.update(cs)
    vocab = [{"qid": c, "label": info[c]["label"], "description": info[c]["description"],
              "entities": n}
             for c, n in count.most_common()
             if n >= min_entities and info.get(c) and info[c]["label"]]
    keep = {v["qid"] for v in vocab}
    out = {"min_entities": min_entities, "vocab": vocab,
           "entity_classes": {q: [c for c in cs if c in keep] for q, cs in ent_classes.items()}}
    TYPES.write_text(json.dumps(out))
    typed = sum(1 for cs in out["entity_classes"].values() if cs)
    print(f"{len(vocab):,} types; {typed:,} of {len(qids):,} entities have one; wrote {TYPES}")
    return out


TYPE_SCHEMA = (24, 48)


def export(n_docs: int | None = None) -> dict:
    """``kg_trex.json``: documents by bank id, each with a relation schema
    (its properties, their nearest neighbours, random others --
    ``relation_pretrain.PropertySchemas``) and a type schema (its entities'
    types plus random others, at least half negatives). Dates are typed
    ``time``; facts on never-trained relations are dropped."""
    import numpy as np
    from ..label_bank import load as load_bank
    from ..relation_pretrain import PropertySchemas, _rng
    bank = load_bank()
    if bank["version"] < 6:
        raise SystemExit("label bank is older than version 6; run python -m v6.label_bank --extend")
    never = {r["id"] for r in bank["relations"] if not r["train"]}
    rel_id = {r["code"]: r["id"] for r in bank["relations"] if r.get("code") and r["id"] not in never}
    type_id = {t["code"]: t["id"] for t in bank["entity_types"] if t.get("code")}
    time_id = type_id["TIME"]
    classes = json.loads(TYPES.read_text())["entity_classes"]
    wd_types = np.asarray([t["id"] for t in bank["entity_types"] if t["source"] == "wikidata"])
    ps = PropertySchemas(bank)
    docs, stats = [], Counter()
    for line in DOCS.open():
        d = json.loads(line)
        ents, typed_ids = [], set()
        for e in d["entities"]:
            if e["kind"] == "date":
                types = [time_id]
            else:
                types = sorted({type_id[c] for c in classes.get(e["key"], []) if c in type_id})
            typed_ids.update(types if e["typed"] else [])
            ents.append({"mentions": e["mentions"], "typed": e["typed"] if types else [],
                         "types": types})
        facts = [[h, t, rel_id[p]] for h, t, p in d["facts"] if p in rel_id]
        stats["facts_dropped"] += len(d["facts"]) - len(facts)
        if not facts and not typed_ids:
            stats["docs_empty"] += 1
            continue
        rel_schema = [ps.rels[i]["id"] for i in ps.schema(d["id"], [p for *_, p in d["facts"]])]
        rel_schema = sorted(set(rel_schema) | {r for *_, r in facts})
        rng = _rng(f"types:{d['id']}")
        k = int(rng.integers(TYPE_SCHEMA[0], TYPE_SCHEMA[1] + 1))
        pos = sorted(typed_ids)
        neg = [int(t) for t in rng.choice(wd_types, size=min(len(wd_types), k + len(pos)),
                                          replace=False) if int(t) not in typed_ids]
        type_schema = sorted(set(pos) | set(neg[:max(k - len(pos), len(pos))]))
        docs.append({"id": d["id"], "tokens": d["tokens"], "sentence_spans": d["sentence_spans"],
                     "entities": ents, "facts": facts, "relations": rel_schema,
                     "types": type_schema})
        stats["docs"] += 1
        stats["facts"] += len(facts)
        stats["typed_spans"] += sum(len(e["typed"]) * len(e["types"]) for e in ents)
        if n_docs and len(docs) >= n_docs:
            break
    OUT.write_text(json.dumps({"source": "T-REx (CC BY 4.0) + Wikidata (CC0)",
                               "bank_hash": bank["hash"], "docs": docs}))
    print(f"wrote {OUT}: {dict(stats)}")
    return dict(stats)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("convert")
    t = sub.add_parser("types")
    t.add_argument("--min-entities", type=int, default=MIN_ENTITIES)
    e = sub.add_parser("export")
    e.add_argument("--docs", type=int)
    args = ap.parse_args()
    if args.cmd == "export":
        export(args.docs)
    if args.cmd == "convert":
        files = sorted(RAW.glob("*.json"))
        print(json.dumps(convert(files)))
    elif args.cmd == "types":
        build_types(args.min_entities)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

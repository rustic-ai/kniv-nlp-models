"""Relation pretraining data for the label path (v6.2, stage 1).

    uv run python -m v6.relation_pretrain pilot --docs 200
    uv run python -m v6.relation_pretrain run --docs 20000
    uv run python -m v6.relation_pretrain export

The label head reads relation labels it was not trained on well in our own
label style and poorly in Wikidata's (S3b: FewRel's unseen relations near
chance). Pretraining it on many relation types, each with its own wording,
before fine-tuning on the knowledge corpus addresses that and gives the new
head a warm start.

Text: DocRED's distantly supervised documents (Wikipedia introductions, MIT),
whose entities are given -- so the teacher labels relations only, the mode
in which it was most precise (bake-off: P 0.71 with gold entities, 0.37
without). Labels: sol, against a per-document schema of Wikidata properties
from the label bank (``source == "wikidata"``, plus Re-DocRED's and FewRel's
trained relations):

- the properties the document's distant labels suggest (likely present);
- their nearest neighbours by name and description (hard negatives -- the
  teacher must tell 'founded by' from 'director / manager');
- random others.

FewRel's never-trained relations and the 40 held-out general and business
relations are not in the bank's Wikidata block, so both zero-shot tests stay
unseen. Answers are cached like the kg teacher's (``KG_CACHE``,
``PROMPT_VERSION``), so a run resumes for free.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from collections import Counter

import numpy as np

from .annotate.kg import KGItem
from .config import DATA_DIR
from .label_bank import load as load_bank

PRETRAIN = DATA_DIR / "v6-corpus" / "kg_pretrain.json"
SCHEMA_SIZE = (16, 32)


def _rng(key: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha1(key.encode()).hexdigest()[:12], 16))


def _words(r: dict) -> set[str]:
    stop = {"the", "of", "a", "an", "or", "and", "to", "in", "is", "by", "for", "this", "that",
            "item", "which", "with", "as", "on", "its", "it", "be", "from", "at", "was"}
    text = " ".join([r["name"], r.get("description") or ""] + list(r.get("aliases") or [])[:4])
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in stop and len(w) > 2}


class PropertySchemas:
    """Schemas of Wikidata-style relations from the bank, with neighbours by
    word overlap of name, description and aliases."""

    def __init__(self, bank: dict):
        self.rels = [r for r in bank["relations"] if r.get("train", True) and
                     r["source"] in ("wikidata", "redocred", "fewrel")]
        self.by_code = {r["code"]: i for i, r in enumerate(self.rels) if r.get("code")}
        words = [_words(r) for r in self.rels]
        n = len(self.rels)
        self.neigh = []
        for i in range(n):
            sims = [(len(words[i] & words[j]) / max(len(words[i] | words[j]), 1), j)
                    for j in range(n) if j != i and words[i] & words[j]]
            self.neigh.append([j for _, j in sorted(sims, reverse=True)[:6]])

    def schema(self, key: str, codes: list[str]) -> list[int]:
        rng = _rng(f"pre:{key}")
        k = int(rng.integers(SCHEMA_SIZE[0], SCHEMA_SIZE[1] + 1))
        present = [self.by_code[c] for c in dict.fromkeys(codes) if c in self.by_code]
        rng.shuffle(present)
        pick = present[: k // 3]
        for i in list(pick):
            for j in self.neigh[i][:2]:
                if j not in pick and len(pick) < (2 * k) // 3:
                    pick.append(j)
        rest = [j for j in range(len(self.rels)) if j not in pick]
        pick += rng.choice(rest, size=k - len(pick), replace=False).tolist()
        return sorted(pick)


def items(n_docs: int, offset: int = 0) -> list[tuple[KGItem, dict]]:
    """``(KGItem, doc)``: DocRED distant documents with their entities given
    as participants, each with a property schema. Documents come in a fixed
    hash order distinct from S2's subset."""
    from .gold import docred_distant
    from .label_bank import DOCRED_TYPES
    bank = load_bank()
    ps = PropertySchemas(bank)
    docs = docred_distant.load(None, seed="pretrain")[offset:offset + n_docs]
    types = [{"name": name, "description": desc} for name, desc in DOCRED_TYPES.values()]
    out = []
    for d in docs:
        names = []
        for e in d["entities"]:
            surf = list(dict.fromkeys(" ".join(d["tokens"][a:b]) for a, b in e["mentions"]))
            names.append("; ".join(surf[:3]))
        rel = [ps.rels[i] for i in ps.schema(d["id"], [c for _, _, c in d["triples"]])]
        digest = hashlib.sha1(json.dumps([r["name"] for r in rel]).encode()).hexdigest()[:10]
        it = KGItem(id=f"pre:{hashlib.sha1(d['id'].encode()).hexdigest()[:16]}:{digest}",
                    tokens=d["tokens"], sentence_spans=d["sentence_spans"], speakers=None,
                    participants=names, types=types,
                    relations=[{"name": r["name"], "description": r["description"]} for r in rel],
                    domain="encyclopedic")
        out.append((it, d))
    return out


def report(results, pairs) -> dict:
    by = {it.id: (it, d) for it, d in pairs}
    c = Counter()
    for r in results:
        it, d = by[r.item_id]
        c["docs"] += 1
        if not r.ok:
            c["failed"] += 1
            continue
        facts = r.payload["facts"]
        c["facts"] += len(facts)
        c["schema"] += len(it.relations)
        names = {x["name"] for x in it.relations}
        c["relations_used"] += len({f["relation"] for f in facts})
        c["tokens_out"] += r.tokens_out or 0
    ok = max(c["docs"] - c["failed"], 1)
    return {"docs": c["docs"], "failed": c["failed"], "facts_per_doc": round(c["facts"] / ok, 2),
            "schema_per_doc": round(c["schema"] / ok, 1),
            "distinct_relations_used_per_doc": round(c["relations_used"] / ok, 2),
            "tokens_out_per_doc": round(c["tokens_out"] / ok)}


def export(pairs, annotator: str) -> Counter:
    """Cached answers with what training needs: tokens, sentences, the given
    entities' mentions, the schema by name, and facts between entities."""
    from .annotate.base import CacheStore
    from .kg_teacher import CACHE
    from .prompts import PROMPT_VERSION
    cache = CacheStore(CACHE, PROMPT_VERSION)
    out, c = [], Counter()
    for it, d in pairs:
        rec = cache.get(annotator, "kg", it.id)
        p = (rec or {}).get("payload")
        if not isinstance(p, dict) or "stats" not in p:
            c["missing"] += 1
            continue
        # Entities the teacher added beyond the given ones (dates, amounts,
        # unlisted names) come with aligned spans: they follow the given
        # entities, ids k.., as the payload numbers them.
        k = len(it.participants)
        ents = list(d["entities"]) + [{"type": None, "mentions": [tuple(x) for x in e["spans"]]}
                                      for e in p["entities"]]
        facts = [[f["head"], f["tail"], f["relation"]] for f in p["facts"]
                 if f["head"] < len(ents) and f["tail"] < len(ents) and f["polarity"] == "positive"]
        c["docs"] += 1
        c["facts"] += len(facts)
        c["facts_on_added_entities"] += sum(1 for h, t, _ in facts if h >= k or t >= k)
        out.append({"id": it.id, "tokens": d["tokens"], "sentence_spans": d["sentence_spans"],
                    "entities": ents, "relations": [r["name"] for r in it.relations],
                    "facts": facts})
    PRETRAIN.write_text(json.dumps({"annotator": annotator, "prompt_version": PROMPT_VERSION,
                                    "docs": out}))
    return c


def main() -> int:
    from .kg_teacher import annotate
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["pilot", "run", "export"])
    ap.add_argument("--docs", type=int, default=200)
    ap.add_argument("--annotator", default="sol61")
    args = ap.parse_args()
    pairs = items(args.docs)
    print(f"{len(pairs)} documents; schema {SCHEMA_SIZE[0]}-{SCHEMA_SIZE[1]} relations each", flush=True)
    if args.cmd == "export":
        print(dict(export(pairs, args.annotator)), f"-> {PRETRAIN}", flush=True)
        return 0
    results = asyncio.run(annotate([it for it, _ in pairs], args.annotator, f"pretrain-{args.cmd}"))
    print(json.dumps(report(results, pairs), indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

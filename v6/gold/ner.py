"""NER gold: OntoNotes 5.0 and CoNLL-2003 test splits.

Both ship in the kniv-corpus-en dataset repo under ``benchmarks/`` as
``{"tokens": [...], "tags": [...]}`` with BIO tags.

OntoNotes is the schema v5's NER head was actually trained against (18 types),
so it is the honest benchmark. CoNLL-2003 uses 4 types and requires mapping
18 -> 4 with numeric entities dropped, which is a strictly harder and lossier
protocol — kept here because v5's published card reports it.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import DATA_DIR
from .ud_ewt import GoldItem

BENCH_DIR = DATA_DIR / "benchmarks"

# OntoNotes -> CoNLL-2003. Numeric/temporal types have no CoNLL equivalent.
ONTONOTES_TO_CONLL = {
    "PERSON": "PER", "ORG": "ORG", "GPE": "LOC", "LOC": "LOC", "FAC": "LOC",
    "NORP": "MISC", "PRODUCT": "MISC", "EVENT": "MISC", "WORK_OF_ART": "MISC",
    "LAW": "MISC", "LANGUAGE": "MISC",
}


def map_to_conll(tags: list[str]) -> list[str]:
    out = []
    for t in tags:
        if t == "O" or "-" not in t:
            out.append("O")
            continue
        prefix, typ = t.split("-", 1)
        mapped = ONTONOTES_TO_CONLL.get(typ)
        out.append(f"{prefix}-{mapped}" if mapped else "O")
    return out


def load_ner_items(benchmark: str = "ontonotes5", limit: int | None = None,
                   max_tokens: int = 128, path: Path | None = None) -> list[GoldItem]:
    path = path or (BENCH_DIR / f"{benchmark}_test.json")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch benchmarks/{benchmark}_test.json from "
            f"the kniv-corpus-en dataset repo."
        )
    raw = json.loads(path.read_text())
    items: list[GoldItem] = []
    for i, ex in enumerate(raw):
        toks, tags = ex.get("tokens") or [], ex.get("tags") or []
        if not toks or len(toks) != len(tags) or len(toks) > max_tokens:
            continue
        items.append(GoldItem(id=f"{benchmark}-{i:05d}", tokens=toks,
                              layers={"ner": tags}))
        if limit and len(items) >= limit:
            break
    return items

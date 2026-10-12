"""FewRel 1.0 (MIT): sentence-level relation classification over Wikidata
relations, 700 instances per relation.

    train_wiki.json  64 relations -- relation supervision for the knowledge layer
    val_wiki.json    16 relations -- evaluation only; 8 of them occur in no
                     training source (true zero-shot), 8 are Re-DocRED's

The text is Wikipedia (CC BY-SA). The test set is hidden behind a leaderboard
and not used. Each instance marks one head and one tail entity (each possibly
with several occurrences); exactly one relation, from head to tail, is
annotated -- the reverse direction is not.
"""
from __future__ import annotations

import json
import urllib.request

from ..config import DATA_DIR

FEWREL_DIR = DATA_DIR / "fewrel"
RAW = "https://raw.githubusercontent.com/thunlp/FewRel/master/data"


def download(split: str) -> dict:
    name = {"train": "train_wiki.json", "val": "val_wiki.json"}[split]
    dst = FEWREL_DIR / name
    if not dst.exists() or dst.stat().st_size == 0:
        FEWREL_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{RAW}/{name}", timeout=120) as r:
            dst.write_bytes(r.read())
    return json.loads(dst.read_text())


def _spans(positions: list[list[int]]) -> list[tuple[int, int]]:
    """FewRel lists each occurrence as its token positions; contiguous runs
    become ``(start, end_exclusive)`` spans."""
    out = []
    for occ in positions:
        occ = sorted(occ)
        out.append((occ[0], occ[-1] + 1))
    return sorted(set(out))


def load(split: str) -> list[dict]:
    """``{"tokens", "head": [(s, e)], "tail": [(s, e)], "code"}`` per instance,
    in file order (relation, then instance)."""
    data = download(split)
    out = []
    for code, insts in data.items():
        for x in insts:
            out.append({"tokens": x["tokens"], "head": _spans(x["h"][2]),
                        "tail": _spans(x["t"][2]), "code": code})
    return out


def codes(split: str) -> list[str]:
    return sorted(download(split))

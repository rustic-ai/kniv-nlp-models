"""PropBank EWT test split -> gold items for the SRL layer.

Reads the prepared JSON the v5 pipeline already produces
(``data/prepared/kniv-deberta-cascade/srl_test.json``): a list of
``{"words": [...], "srl_tags": [...], "predicate_idx": int}``. Each record is
one predicate, which matches how the layer is actually annotated.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import PREPARED_DIR
from .ud_ewt import GoldItem


def load_srl_items(limit: int | None = None, max_tokens: int = 128,
                   path: Path | None = None) -> list[GoldItem]:
    path = path or (PREPARED_DIR / "srl_test.json")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch the prepared SRL test split from the "
            f"kniv-corpus-en dataset repo, or point --srl-path at it."
        )
    raw = json.loads(path.read_text())

    items: list[GoldItem] = []
    for i, ex in enumerate(raw):
        words = ex.get("words") or []
        tags = ex.get("srl_tags") or []
        pred = ex.get("predicate_idx")
        if not words or len(words) != len(tags) or pred is None:
            continue
        if len(words) > max_tokens or not (0 <= pred < len(words)):
            continue
        items.append(GoldItem(
            id=ex.get("id", f"pb-test-{i:05d}"),
            tokens=words,
            layers={"srl": tags},
            predicate_idx=pred,
        ))
        if limit and len(items) >= limit:
            break
    return items

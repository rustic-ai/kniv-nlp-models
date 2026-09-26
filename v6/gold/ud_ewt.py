"""UD English EWT test split -> gold items for pos / lemma / morph / dep.

One CoNLL-U file yields four of the five layers under question, which is why
this is the cheapest possible bake-off.

Download with ``./data/download_ud.sh`` (CC BY-SA 4.0, evaluation use).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import conllu

from ..config import UD_DIR


@dataclass
class GoldItem:
    """One sentence with whatever gold layers are available for it."""

    id: str
    tokens: list[str]
    layers: dict[str, object] = field(default_factory=dict)
    predicate_idx: int | None = None
    # Relation extraction supplies pre-clustered entities; the annotator
    # classifies relations over them rather than finding mentions itself.
    entities: list[dict] | None = None
    # Sentence-level layers (cls, sentiment) are labelled one sentence at a
    # time with the surrounding window supplied as evidence.
    context: str | None = None
    target: str | None = None


def _feats_to_string(feats: dict | None) -> str:
    if not feats:
        return "_"
    return "|".join(f"{k}={v}" for k, v in sorted(feats.items()))


def load_ud_items(split: str = "test", limit: int | None = None,
                  max_tokens: int = 128,
                  path: Path | None = None) -> list[GoldItem]:
    """Load UD EWT sentences with POS / lemma / morph / dep gold.

    Multi-word tokens and empty nodes are dropped so that token indices are
    dense and 1:1 with the CoNLL-U ``id`` column — the same filtering the v5
    data prep used, so numbers stay comparable.
    """
    path = path or (UD_DIR / f"en_ewt-ud-{split}.conllu")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run ./data/download_ud.sh first."
        )

    items: list[GoldItem] = []
    for sent in conllu.parse(path.read_text(encoding="utf-8")):
        toks = [t for t in sent if isinstance(t["id"], int)]
        if not toks or len(toks) > max_tokens:
            continue

        # Remap heads through the filtered index space.
        id_to_pos = {t["id"]: i for i, t in enumerate(toks)}
        heads, rels, ok = [], [], True
        for t in toks:
            h = t["head"]
            if h is None:
                ok = False
                break
            if h == 0:
                heads.append(0)
            elif h in id_to_pos:
                heads.append(id_to_pos[h] + 1)   # keep 1-indexed
            else:
                ok = False
                break
            rels.append(t["deprel"] or "dep")
        if not ok:
            continue

        items.append(GoldItem(
            id=sent.metadata.get("sent_id", f"ud-{split}-{len(items):05d}"),
            tokens=[t["form"] for t in toks],
            layers={
                "pos": [t["upos"] or "X" for t in toks],
                "lemma": [t["lemma"] or t["form"] for t in toks],
                "morph": [_feats_to_string(t["feats"]) for t in toks],
                "dep": {"heads": heads, "rels": rels},
            },
        ))
        if limit and len(items) >= limit:
            break
    return items

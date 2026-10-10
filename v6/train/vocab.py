"""Label vocabularies: fixed inventories, plus the two derived from the corpus.

The fixed inventories are ``v6/schemas.py``'s, whose list position is the
class index -- the same lists the corpus gates validate against, and
identical to ``models/label_maps.json`` for POS, NER, DEP and SRL.

Morph and lemma are open-ended, so their vocabularies are derived from the
**train** split only (TRAINING.md §3). Anything outside them is masked out of
training rather than approximated, and the coverage is recorded so the loss
is a number rather than an assumption.

    uv run python -m v6.train.vocab            # writes data/v6-corpus/vocab.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ..config import DATA_DIR
from ..schemas import (CLS_LABELS, DEPRELS, NER_LABELS, SENTIMENT_LABELS,
                       SRL_TAGS, UPOS_TAGS)

VOCAB_PATH = DATA_DIR / "v6-corpus" / "vocab.json"
FIXED = {"pos": UPOS_TAGS, "ner": NER_LABELS, "dep": DEPRELS, "srl": SRL_TAGS,
         "cls": CLS_LABELS, "sentiment": SENTIMENT_LABELS}
# A script or FEATS string seen fewer times than this in train is masked:
# one-offs are mostly annotator noise and cannot be learned from one example.
MIN_COUNT = 2


# ── lemma edit scripts ───────────────────────────────────────────────
def edit_script(word: str, lemma: str) -> str:
    """``"<case>|<strip>|<append>"``: optionally lowercase, strip ``strip``
    characters from the end, append ``append``.

    Exact by construction: ``apply_script(word, edit_script(word, lemma))``
    returns ``lemma`` for every pair. Of the two case options the shorter edit
    wins, so "Has" -> "have" is ``l|2|ve`` rather than a whole-word rewrite.
    """
    best = None
    for case in ("", "l"):
        w = word.lower() if case else word
        p = 0
        while p < min(len(w), len(lemma)) and w[p] == lemma[p]:
            p += 1
        cand = (len(w) - p + len(lemma) - p, case, len(w) - p, lemma[p:])
        if best is None or cand[0] < best[0]:
            best = cand
    _, case, strip, append = best
    return f"{case}|{strip}|{append}"


def apply_script(word: str, script: str) -> str:
    case, strip, append = script.split("|", 2)
    w = word.lower() if case else word
    return w[:len(w) - int(strip)] + append


# ── vocabulary ──────────────────────────────────────────────────────
class Vocab:
    def __init__(self, labels: dict[str, list[str]], meta: dict | None = None):
        self.labels = labels
        self.index = {k: {l: i for i, l in enumerate(v)} for k, v in labels.items()}
        self.meta = meta or {}

    def size(self, layer: str) -> int:
        return len(self.labels[layer])

    def id(self, layer: str, label: str) -> int | None:
        return self.index[layer].get(label)

    def to_json(self) -> dict:
        return {"labels": self.labels, "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict) -> "Vocab":
        return cls(d["labels"], d.get("meta"))

    @classmethod
    def load(cls, path: Path = VOCAB_PATH) -> "Vocab":
        if not path.exists():
            raise SystemExit(f"{path} not found; run `python -m v6.train.vocab`")
        v = cls.from_json(json.loads(path.read_text()))
        for k, fixed in FIXED.items():
            if v.labels[k] != fixed:
                raise SystemExit(f"{path}: {k} inventory differs from "
                                 f"v6/schemas.py; rebuild the vocabulary")
        return v


def build(rows) -> Vocab:
    morph, lemma = Counter(), Counter()
    n_words = 0
    for r in rows:
        mask = json.loads(r["loss_mask"] or "{}")
        if r["morph"] is not None and not mask.get("morph"):
            morph.update(r["morph"])
        if r["lemma"] is not None and not mask.get("lemma"):
            lemma.update(edit_script(w, l) for w, l in zip(r["tokens"], r["lemma"]))
        n_words += r["n_tokens"]
    labels = dict(FIXED)
    # Phase 2. Relation names come from Re-DocRED's inventory, which needs the
    # Re-DocRED files, so they are resolved here and stored -- a VM reads them
    # from vocab.json rather than recomputing them.
    from ..schemas import relation_names
    from ..entities import TYPE_MAP
    labels["relations"] = relation_names()
    labels["ent_type"] = sorted(set(TYPE_MAP.values()))
    meta = {"min_count": MIN_COUNT, "train_words": n_words}
    for name, counts in (("morph", morph), ("lemma", lemma)):
        kept = sorted((k for k, c in counts.items() if c >= MIN_COUNT),
                      key=lambda k: (-counts[k], k))
        total = sum(counts.values())
        covered = sum(counts[k] for k in kept)
        labels[name] = kept
        meta[name] = {"classes": len(kept), "seen": len(counts),
                      "coverage": round(covered / max(total, 1), 6),
                      "masked_words": total - covered}
    return Vocab(labels, meta)


def main() -> int:
    from .data import load_rows
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=VOCAB_PATH)
    args = ap.parse_args()
    v = build(load_rows("train", columns=["tokens", "n_tokens", "lemma",
                                          "morph", "loss_mask"]))
    args.out.write_text(json.dumps(v.to_json(), indent=1))
    for k in ("morph", "lemma"):
        m = v.meta[k]
        print(f"{k}: {m['classes']} classes of {m['seen']} seen, "
              f"coverage {m['coverage']:.4%} ({m['masked_words']:,} words masked)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

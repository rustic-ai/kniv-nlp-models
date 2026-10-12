"""Batches for the schema-conditioned knowledge layer.

``LabelCollator`` turns a collated batch whose relation targets index the
whole label bank (``pair_labels [P, R+1]``) into one whose columns are the
batch's own label set, with the label texts tokenised for the label pass:

- ``rel_label_ids`` / ``rel_label_mask`` ``[K, L]``: the K labels' texts;
- ``rel_label_cols``: their bank ids, in column order (for decoding);
- ``pair_labels [P, K+1]``: column 0 is "no listed relation holds";
- ``pair_label_mask [P, K+1]``: 0 where a label was not in that pair's
  schema, so it is neither a positive nor a negative (``atlop_loss``).

Training samples the label set (label dropout): a sample of 16-96 of the
batch's labels. On an exhaustively labelled source (Re-DocRED) a dropped
positive leaves the pair "no listed relation", which is the right reading of
a smaller schema. Each label is read by one of several texts -- name only,
name and description, an alias and description -- so the head learns to read
descriptions rather than memorise columns. Evaluation uses every label, by
name and description, in bank order.
"""
from __future__ import annotations

import numpy as np
import torch

LABEL_MAX_SUBWORDS = 48
# name / name + description / alias + description
TEXT_MODES = (("name", 0.2), ("name_desc", 0.6), ("alias_desc", 0.2))


def label_text(rel: dict, mode: str = "name_desc", rng=None) -> str:
    name = rel["name"]
    if mode == "alias_desc" and rel.get("aliases"):
        name = rel["aliases"][int(rng.integers(len(rel["aliases"])))] if rng is not None \
            else rel["aliases"][0]
    if mode == "name" or not rel.get("description"):
        return f"relation: {name}"
    return f"relation: {name}. {rel['description']}"


class LabelCollator:
    def __init__(self, tok, bank: dict, train: bool, seed: int = 13,
                 min_labels: int = 16, max_labels: int = 96,
                 exclude_holdout: bool = False, only_holdout: bool = False,
                 sources: tuple[str, ...] = ("redocred",), text_modes=None):
        self.tok, self.train = tok, train
        self.text_modes = tuple(text_modes or TEXT_MODES)
        self.rels = bank["relations"]
        self.rng = np.random.default_rng(seed)
        self.min_labels, self.max_labels = min_labels, max_labels
        # Only the labels the batch's source was annotated against: a
        # Re-DocRED document says nothing about FewRel-only relations, so
        # offering them would teach false negatives. Never-trained labels
        # (FewRel val) are excluded from training batches.
        allowed = [r["id"] for r in self.rels
                   if r["source"] in sources
                   and (r.get("train", True) or not train)
                   and not (exclude_holdout and r["holdout"])
                   and not (only_holdout and not r["holdout"])]
        self.allowed = np.asarray(allowed, dtype=np.int64)
        self._tok_cache: dict[str, list[int]] = {}

    def _ids(self, text: str) -> list[int]:
        if text not in self._tok_cache:
            self._tok_cache[text] = self.tok(text, truncation=True,
                                             max_length=LABEL_MAX_SUBWORDS)["input_ids"]
        return self._tok_cache[text]

    def columns(self) -> np.ndarray:
        if not self.train:
            return self.allowed
        k = int(self.rng.integers(self.min_labels, min(self.max_labels, len(self.allowed)) + 1))
        return np.sort(self.rng.choice(self.allowed, size=k, replace=False))

    def __call__(self, batch: dict) -> dict:
        if "pair_labels" not in batch:
            return batch
        cols = self.columns()
        old = batch["pair_labels"]                                       # [P, R+1]
        new = old[:, 1 + torch.as_tensor(cols)]
        none = (new.sum(1, keepdim=True) == 0).to(new.dtype)
        batch["pair_labels"] = torch.cat([none, new], 1)
        batch["pair_label_mask"] = torch.ones_like(batch["pair_labels"])
        modes, probs = zip(*self.text_modes)
        texts = [label_text(self.rels[c], self.rng.choice(modes, p=probs) if self.train
                            else "name_desc", self.rng) for c in cols]
        ids = [self._ids(t) for t in texts]
        L = max(len(x) for x in ids)
        pad = self.tok.pad_token_id
        batch["rel_label_ids"] = torch.tensor([x + [pad] * (L - len(x)) for x in ids])
        batch["rel_label_mask"] = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids])
        batch["rel_label_cols"] = torch.as_tensor(cols)
        return batch

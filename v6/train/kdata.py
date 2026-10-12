"""Batches for the schema-conditioned knowledge layer.

``LabelCollator`` turns a collated batch whose relation targets index the
whole label bank (``pair_labels [P, R+1]``) into one whose columns are the
batch's own label set, with the label texts tokenised for the label pass:

- ``rel_label_ids`` / ``rel_label_mask`` ``[K, L]``: the K labels' texts;
- ``rel_label_cols``: their bank ids, in column order (for decoding);
- ``pair_labels [P, K+1]``: column 0 is "no listed relation holds";
- ``pair_label_mask [P, K+1]``: 0 where a label was not in that pair's
  schema, so it is neither a positive nor a negative (``atlop_loss``).

Each item carries the label-bank ids it was annotated against (its schema:
a Re-DocRED document, its source's relations; a teacher window, the 8-40
relations it was shown); a column outside an item's schema is masked for
its pairs. Teacher labels are also partial -- precise but incomplete (the
bake-off: P 0.71, R 0.31 with gold entities) -- so for those items a pair
whose positives were all dropped from the columns is masked whole, and a
pair with no teacher relation is kept as a negative only with probability
``unlabelled_keep`` (positive-unlabelled learning, downweighting the
unlabelled rather than trusting it).

Training samples the label set (label dropout): a sample of 16-96 of the
batch's labels. On an exhaustively labelled source (Re-DocRED) a dropped
positive leaves the pair "no listed relation", which is the right reading of
a smaller schema. Each label is read by one of several texts -- name only,
name and description, an alias and description -- so the head learns to read
descriptions rather than memorise columns. Evaluation uses every label, by
name and description, in bank order.
"""
from __future__ import annotations

from collections import Counter

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
                 sources: tuple[str, ...] = ("redocred",), text_modes=None,
                 unlabelled_keep: float = 0.5):
        self.tok, self.train = tok, train
        self.unlabelled_keep = unlabelled_keep
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
        # The same filters without the source one, for items with their own schema.
        self.usable = {r["id"] for r in self.rels
                       if (r.get("train", True) or not train)
                       and not (exclude_holdout and r["holdout"])
                       and not (only_holdout and not r["holdout"])}
        self._tok_cache: dict[str, list[int]] = {}

    def _ids(self, text: str) -> list[int]:
        if text not in self._tok_cache:
            self._tok_cache[text] = self.tok(text, truncation=True,
                                             max_length=LABEL_MAX_SUBWORDS)["input_ids"]
        return self._tok_cache[text]

    def schema_of(self, schema) -> np.ndarray:
        if schema is None:
            return self.allowed
        return np.asarray(sorted(set(int(x) for x in schema) & self.usable), dtype=np.int64)

    def columns(self, pool: np.ndarray | None = None) -> np.ndarray:
        pool = self.allowed if pool is None else pool
        if not self.train or len(pool) <= self.min_labels:
            return pool
        k = int(self.rng.integers(self.min_labels, min(self.max_labels, len(pool)) + 1))
        return np.sort(self.rng.choice(pool, size=k, replace=False))

    def __call__(self, batch: dict) -> dict:
        if "pair_labels" not in batch:
            return batch
        schemas = batch.pop("rel_schemas", None)
        partial = batch.pop("rel_partial", None)
        old = batch["pair_labels"]                                       # [P, R+1]
        if schemas is None:
            cols = self.columns()
            new = old[:, 1 + torch.as_tensor(cols)]
            none = (new.sum(1, keepdim=True) == 0).to(new.dtype)
            batch["pair_labels"] = torch.cat([none, new], 1)
            batch["pair_label_mask"] = torch.ones_like(batch["pair_labels"])
        else:
            per = [self.schema_of(x) for x in schemas]
            used = sorted({int(b) for b in batch["pair_batch"].tolist()})
            pool = np.unique(np.concatenate([per[b] for b in used] or [self.allowed]))
            cols = self.columns(pool)
            ct = torch.as_tensor(cols)
            known = torch.zeros(len(per), len(cols))                       # [B, K]
            for b, sc in enumerate(per):
                known[b] = torch.as_tensor(np.isin(cols, sc), dtype=known.dtype)
            pk = known[batch["pair_batch"]]                                # [P, K]
            new = old[:, 1 + ct] * pk
            none = (new.sum(1, keepdim=True) == 0).to(new.dtype)
            mask = torch.cat([torch.ones_like(none), pk], 1)
            part = torch.as_tensor(partial, dtype=torch.bool)[batch["pair_batch"]]
            had = old[:, 1:].sum(1) > 0
            drop = part & had & (none.squeeze(1) > 0)        # its positives fell outside
            if self.train:
                drop |= part & ~had & (torch.as_tensor(
                    self.rng.random(len(had))) >= self.unlabelled_keep)
            mask[drop] = 0
            batch["pair_labels"] = torch.cat([none, new], 1) * (~drop).unsqueeze(1)
            batch["pair_label_mask"] = mask
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


def teacher_features(item, payload: dict, tok, max_subwords: int, bank: dict, n_cls: int,
                     stats: Counter | None = None, markers: bool = True) -> dict | None:
    """One teacher-annotated window (a ``KGItem`` and its aligned ``kg``
    payload) as relation supervision: participants first (their mentions are
    the speaker tags of the marker pass), then the teacher's entities;
    triples by label-bank id; the window's schema; ``partial``. Every other
    head's label is IGNORE. Relations not (yet) in the bank are dropped and
    counted. ``None`` under two entities."""
    from .data import blank_window, marker_features, speaker_turns
    stats = Counter() if stats is None else stats
    spans = [tuple(x) for x in item.sentence_spans]
    f = blank_window(item.tokens, spans, tok, max_subwords, n_cls)
    W = f["_W"]
    by_name = {r["name"]: r["id"] for r in bank["relations"]}
    parts = list(payload.get("participants") or [])
    k = len(parts)
    turns = [(s, p) for s, p in speaker_turns(spans, item.speakers, parts)
             if s < W and f["word_ok"][s]]
    keep, ents = {}, []
    for p in range(k):
        ms = [(s, s + 1) for s, q in turns if q == p]     # plain pass: the turn's first word
        if ms:
            keep[p] = len(ents)
            ents.append(np.asarray(ms, dtype=np.int32))
    n_part = len(ents)
    for i, e in enumerate(payload["entities"]):
        ms = [(a, b) for a, b in e["spans"] if b <= W and f["word_ok"][a]]
        stats["kg_mentions_past_limit"] += len(e["spans"]) - len(ms)
        if ms:
            keep[k + i] = len(ents)
            ents.append(np.asarray(ms, dtype=np.int32))
    if len(ents) < 2:
        stats["kg_windows_under_two_entities"] += 1
        return None
    schema = sorted(by_name[r["name"]] for r in item.relations if r["name"] in by_name)
    stats["kg_schema_not_in_bank"] += len(item.relations) - len(schema)
    triples, attrs, ev = [], [], {}
    for x in payload["facts"]:
        r = by_name.get(x["relation"])
        if r is None:
            stats["kg_facts_not_in_bank"] += 1
            continue
        if x["head"] not in keep or x["tail"] not in keep:
            stats["kg_facts_past_limit"] += 1
            continue
        h, t = keep[x["head"]], keep[x["tail"]]
        triples.append((h, t, r))
        attrs.append({"h": h, "t": t, "r": r, "polarity": x["polarity"],
                      "modality": x["modality"],
                      "holder": keep.get(x["holder"], -1), "time": keep.get(x["time"], -1)})
        sents = [j for j in x["evidence"] if j < f["_S"] and spans[j][0] < W]
        if sents:
            ev.setdefault((h, t), set()).update(sents)
    f.update(coref=None, _n_rel=len(bank["relations"]),
             rel={"spans": ents, "types": np.zeros(len(ents), dtype=np.int32),
                  "triples": triples, "attrs": attrs, "n_participants": n_part,
                  "evidence": [(h, t, sorted(v)) for (h, t), v in sorted(ev.items())],
                  "schema": schema, "partial": True})
    if markers:
        part_turns = [(s, keep[p]) for s, p in turns if p in keep]
        f["rm"] = marker_features(item.tokens, spans, ents[n_part:], tok,
                                  turns=part_turns, participants=[parts[p] for p in keep if p < k])
        stats["kg_marker_mentions_past_limit"] += int(sum((p < 0).sum() for p in f["rm"]["men_pos"]))
    stats["kg_windows"] += 1
    stats["kg_triples"] += len(triples)
    return f

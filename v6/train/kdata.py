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

import hashlib
import json
from collections import Counter

import numpy as np
import torch

LABEL_MAX_SUBWORDS = 48
# name / name + description / alias + description
TEXT_MODES = (("name", 0.2), ("name_desc", 0.6), ("alias_desc", 0.2))


def type_text(t: dict) -> str:
    return f"entity type: {t['name']}. {t['description']}" if t.get("description") \
        else f"entity type: {t['name']}"


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
                 unlabelled_keep: float | None = None, exclude_ids=(), only_ids=None):
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
        # Relations held out of training altogether (``exclude_ids``), or the
        # only ones scored (``only_ids``, a zero-shot dev set).
        self.usable -= set(exclude_ids)
        if only_ids is not None:
            self.usable &= set(only_ids)
            self.allowed = np.asarray([i for i in self.allowed if i in self.usable], dtype=np.int64)
        self._tok_cache: dict[str, list[int]] = {}
        # Hard negatives (loop --hard-negatives): each relation's nearest
        # relations by label embedding. A batch with a true relation always
        # offers its siblings too -- within the pool, so only where the
        # source was annotated against them (never an invented negative).
        self.siblings: dict[int, np.ndarray] = {}

    def _ids(self, text: str) -> list[int]:
        if text not in self._tok_cache:
            self._tok_cache[text] = self.tok(text, truncation=True,
                                             max_length=LABEL_MAX_SUBWORDS)["input_ids"]
        return self._tok_cache[text]

    def schema_of(self, schema) -> np.ndarray:
        if schema is None:
            return self.allowed
        return np.asarray(sorted(set(int(x) for x in schema) & self.usable), dtype=np.int64)

    def columns(self, pool: np.ndarray | None = None, keep=()) -> np.ndarray:
        """Label dropout: a random subset of the pool, always including
        ``keep`` (the batch's true labels). Dropping a true label would mask
        its whole pair, losing the supervision instead of varying the
        negatives."""
        pool = self.allowed if pool is None else pool
        if not self.train or len(pool) <= self.min_labels:
            return pool
        keep = np.intersect1d(np.asarray(list(keep), dtype=np.int64), pool)
        if self.siblings and len(keep):
            sib = np.concatenate([self.siblings.get(int(k), np.zeros(0, dtype=np.int64))
                                  for k in keep])
            keep = np.union1d(keep, np.intersect1d(sib, pool))
        k = int(self.rng.integers(self.min_labels, min(self.max_labels, len(pool)) + 1))
        rest = np.setdiff1d(pool, keep)
        n = max(0, min(len(rest), k - len(keep)))
        return np.sort(np.concatenate([keep, self.rng.choice(rest, size=n, replace=False)]))

    def __call__(self, batch: dict) -> dict:
        if "pair_labels" not in batch:
            return batch
        schemas = batch.pop("rel_schemas", None)
        partial = batch.pop("rel_partial", None)
        keeps = batch.pop("rel_keep", None)
        old = batch["pair_labels"]                                       # [P, R+1]
        true = (old[:, 1:].sum(0) > 0).nonzero(as_tuple=True)[0].numpy()
        if schemas is None:
            cols = self.columns(keep=true)
            new = old[:, 1 + torch.as_tensor(cols)]
            none = (new.sum(1, keepdim=True) == 0).to(new.dtype)
            batch["pair_labels"] = torch.cat([none, new], 1)
            batch["pair_label_mask"] = torch.ones_like(batch["pair_labels"])
        else:
            per = [self.schema_of(x) for x in schemas]
            used = sorted({int(b) for b in batch["pair_batch"].tolist()})
            pool = np.unique(np.concatenate([per[b] for b in used] or [self.allowed]))
            cols = self.columns(pool, keep=true)
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
                # An explicit collator setting wins; otherwise each source's own.
                keep = torch.as_tensor([self.unlabelled_keep if self.unlabelled_keep is not None
                                        else (0.5 if k is None else k)
                                        for k in (keeps or [None] * len(per))],
                                       dtype=torch.float64)[batch["pair_batch"]]
                drop |= part & ~had & (torch.as_tensor(self.rng.random(len(had))) >= keep)
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
        if "attr_rel" in batch:                  # fact attributes: their relation's column
            where = {int(c): j for j, c in enumerate(cols)}
            batch["attr_col"] = torch.as_tensor([where.get(int(r), -1)
                                                 for r in batch.pop("attr_rel").tolist()])
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
                  "schema": schema, "partial": True,
                  "unlabelled_keep": UNLABELLED_KEEP["teacher"]})
    tname = {t["name"]: t["id"] for t in bank["entity_types"]}
    typed = [(tname[e["type"]], a, b) for e in payload["entities"] if e["type"] in tname
             for a, b in e["spans"]]
    f["_domain"] = item.domain
    f["ents"] = span_targets(typed, W, f["word_ok"],
                             [tname[t["name"]] for t in item.types if t["name"] in tname], True)
    if markers:
        part_turns = [(s, keep[p]) for s, p in turns if p in keep]
        f["rm"] = marker_features(item.tokens, spans, ents[n_part:], tok,
                                  turns=part_turns, participants=[parts[p] for p in keep if p < k])
        stats["kg_marker_mentions_past_limit"] += int(sum((p < 0).sum() for p in f["rm"]["men_pos"]))
    stats["kg_windows"] += 1
    stats["kg_triples"] += len(triples)
    return f


class TypeCollator:
    """Candidate spans and their typed targets for the span-type head, from
    the ``span_items`` collation left in a batch.

    Candidates are every span of up to ``max_width`` words inside one
    sentence. Training keeps every gold span and a sample of the rest
    (``max_neg`` per item); evaluation keeps all. Columns are the batch's
    entity types (a sample of at most ``max_types`` in training); a column
    outside an item's type schema is masked for its spans, and for a partial
    source (the teacher) an unlabelled span is kept as a negative only with
    probability ``unlabelled_keep``."""

    def __init__(self, tok, bank: dict, train: bool, seed: int = 17, max_width: int = 12,
                 max_neg: int = 128, max_types: int = 48, unlabelled_keep: float = 0.5):
        self.tok, self.train = tok, train
        self.types = bank["entity_types"]
        self.rng = np.random.default_rng(seed)
        self.max_width, self.max_neg, self.max_types = max_width, max_neg, max_types
        self.unlabelled_keep = unlabelled_keep
        self._tok_cache: dict[str, list[int]] = {}
        # Hard negatives (loop --hard-negatives): each relation's nearest
        # relations by label embedding. A batch with a true relation always
        # offers its siblings too -- within the pool, so only where the
        # source was annotated against them (never an invented negative).
        self.siblings: dict[int, np.ndarray] = {}

    def _ids(self, text: str) -> list[int]:
        if text not in self._tok_cache:
            self._tok_cache[text] = self.tok(text, truncation=True,
                                             max_length=LABEL_MAX_SUBWORDS)["input_ids"]
        return self._tok_cache[text]

    def candidates(self, sent_of_word: np.ndarray, word_ok: np.ndarray) -> list[tuple[int, int]]:
        out = []
        n = len(sent_of_word)
        for s in range(n):
            if not word_ok[s] or sent_of_word[s] < 0:
                continue
            for e in range(s + 1, min(s + self.max_width, n) + 1):
                if sent_of_word[e - 1] != sent_of_word[s]:
                    break
                if word_ok[e - 1]:
                    out.append((s, e))
        return out

    def __call__(self, batch: dict) -> dict:
        items = batch.pop("span_items", None)
        if not items or all(x is None for x in items):
            return batch
        sow = batch["sent_of_word"].numpy()
        wok = batch["word_mask"].numpy()
        pool = sorted({t for x in items if x is not None for t in x["schema"]})
        cols = np.asarray(pool, dtype=np.int64)
        if self.train and len(cols) > self.max_types:
            # Keep every type a span in the batch has; sample the rest.
            true = np.intersect1d(np.unique(np.concatenate(
                [x["spans"][:, 2] for x in items if x is not None and len(x["spans"])]
                or [np.zeros(0, dtype=np.int64)])).astype(np.int64), cols)
            rest = np.setdiff1d(cols, true)
            n = max(0, min(len(rest), self.max_types - len(true)))
            cols = np.sort(np.concatenate([true, self.rng.choice(rest, size=n, replace=False)]))
        col_of = {int(c): j for j, c in enumerate(cols)}
        sb, ss, se, lab, msk = [], [], [], [], []
        for b, x in enumerate(items):
            if x is None:
                continue
            gold: dict[tuple[int, int], set] = {}
            for s, e, t in x["spans"].tolist():
                gold.setdefault((s, e), set()).add(t)
            cand = self.candidates(sow[b], wok[b])
            if self.train:
                neg = [c for c in cand if c not in gold]
                if len(neg) > self.max_neg:
                    pick = self.rng.choice(len(neg), size=self.max_neg, replace=False)
                    neg = [neg[i] for i in sorted(pick)]
                cand = sorted(set(gold) | set(neg))
            known = np.zeros(len(cols) + 1, dtype=np.float32)
            known[0] = 1
            for t in x["schema"]:
                if t in col_of:
                    known[1 + col_of[t]] = 1
            for s, e in cand:
                row = np.zeros(len(cols) + 1, dtype=np.float32)
                types = gold.get((s, e), set())
                hit = [col_of[t] for t in types if t in col_of]
                if hit:
                    row[[1 + j for j in hit]] = 1
                else:
                    row[0] = 1
                m = known.copy()
                if types and not hit:                 # its types fell outside the columns
                    m[:] = 0
                elif x["partial"] and not types and self.train \
                        and self.rng.random() >= self.unlabelled_keep:
                    m[:] = 0
                sb.append(b); ss.append(s); se.append(e); lab.append(row); msk.append(m)
        L = torch.long
        batch["span_batch"] = torch.tensor(sb, dtype=L)
        batch["span_start"] = torch.tensor(ss, dtype=L)
        batch["span_end"] = torch.tensor(se, dtype=L)
        K = len(cols) + 1
        batch["span_labels"] = torch.tensor(np.asarray(lab).reshape(-1, K))
        batch["span_label_mask"] = torch.tensor(np.asarray(msk).reshape(-1, K))
        ids = [self._ids(type_text(self.types[c])) for c in cols]
        Lm = max((len(x) for x in ids), default=1)
        pad = self.tok.pad_token_id
        batch["type_label_ids"] = torch.tensor([x + [pad] * (Lm - len(x)) for x in ids]).reshape(len(ids), Lm)
        batch["type_label_mask"] = torch.tensor([[1] * len(x) + [0] * (Lm - len(x)) for x in ids]).reshape(len(ids), Lm)
        batch["type_label_cols"] = torch.as_tensor(cols)
        return batch


def decode_spans(logits: torch.Tensor, batch: dict,
                 offset: float = 0.0) -> list[list[tuple[int, int, int]]]:
    """Greedy flat decoding: per item, spans whose best type beats the
    threshold, highest margin first, skipping any that overlap one taken.
    ``(start, end_exclusive, bank type id)``."""
    if len(logits) == 0:
        return []
    # ``offset``: a span is kept when its best margin exceeds it (calibrated
    # for type names the head never trained on, like the label path's).
    margin = logits[:, 1:] - logits[:, :1] - offset
    best, j = margin.max(1)
    cols = batch["type_label_cols"]
    out: dict[int, list] = {}
    taken: dict[int, np.ndarray] = {}
    for i in torch.argsort(best, descending=True).tolist():
        if best[i] <= 0:
            break
        b = int(batch["span_batch"][i]); s = int(batch["span_start"][i]); e = int(batch["span_end"][i])
        t = taken.setdefault(b, np.zeros(int(batch["word_mask"].size(1)) + 1, dtype=bool))
        if t[s:e].any():
            continue
        t[s:e] = True
        out.setdefault(b, []).append((s, e, int(cols[j[i]])))
    return [sorted(out.get(b, [])) for b in range(int(batch["word_mask"].size(0)))]


# ── public sources for the knowledge layer ───────────────────────────
# Each is written once, locally, as a compact JSON beside the corpus (label
# ids already resolved against the bank), so a VM needs neither the network
# nor the raw datasets. ``knowledge_features`` turns any of them -- and the
# teacher export -- into feature dicts the collators batch together.
def _corpus():
    from ..config import DATA_DIR
    return DATA_DIR / "v6-corpus"


KNOWLEDGE_FILES = {"fewrel": "fewrel_rel.json", "fewnerd": "fewnerd_spans.json",
                   "docred_distant": "docred_distant_rel.json", "pretrain": "kg_pretrain.json",
                   "trex": "kg_trex.json"}
# Unlabelled pairs of a partial source kept as negatives, per source: FewRel
# labels one pair per sentence and says nothing of the others.
# Bump whenever a featuriser's output changes: the feature cache is keyed on it.
FEATURE_VERSION = 4
UNLABELLED_KEEP = {"teacher": 0.5, "docred_distant": 0.5, "fewrel": 0.0, "pretrain": 0.5,
                   "trex": 0.5}


def _digest(path):
    from .data import file_digest
    return file_digest(path)


def never_trained(bank: dict) -> set[int]:
    """The zero-shot holdout (``train: false``): never in a training item,
    as a fact, a negative or a schema column."""
    return {r["id"] for r in bank["relations"] if not r.get("train", True)}



def _ids(bank: dict) -> tuple[dict, dict, dict]:
    code = {r["code"]: r["id"] for r in bank["relations"] if r.get("code")}
    tname = {t["name"]: t["id"] for t in bank["entity_types"]}
    tcode = {t["code"]: t["id"] for t in bank["entity_types"] if t.get("code")}
    return code, tname, tcode


def prepare_knowledge(bank: dict, distant_docs: int = 40000) -> dict:
    """Write the public sources' JSONs; returns per-file counts."""
    from ..gold import docred_distant, fewnerd, fewrel
    code, _, tcode = _ids(bank)
    out = {}
    # Training items: relations that train. Zero-shot items ("val"): every
    # FewRel sentence of a never-trained relation, from either split.
    never = never_trained(bank)
    fr, val = [], []
    for sp in ("train", "val"):
        for x in fewrel.load(sp):
            if x["code"] not in code:
                continue
            item = {"tokens": x["tokens"], "head": x["head"], "tail": x["tail"],
                    "rel": code[x["code"]]}
            (val if item["rel"] in never else fr if sp == "train" else []).append(item)
    schema = sorted({r["rel"] for r in fr})
    (_corpus() / KNOWLEDGE_FILES["fewrel"]).write_text(
        json.dumps({"train": fr, "val": val, "schema": schema,
                    "val_schema": sorted({r["rel"] for r in val})}))
    out["fewrel"] = {"train": len(fr), "relations": len(schema)}
    fn = {}
    for split in ("train", "dev"):
        fn[split] = [{"tokens": x["tokens"],
                      "spans": [[tcode[c], s, e] for c, s, e in x["spans"]]}
                     for x in fewnerd.load(split)]
    fn["schema"] = sorted(t["id"] for t in bank["entity_types"] if t["source"] == "fewnerd")
    (_corpus() / KNOWLEDGE_FILES["fewnerd"]).write_text(json.dumps(fn))
    out["fewnerd"] = {k: len(v) for k, v in fn.items()}
    dd, missing = [], Counter()
    for d in docred_distant.load(distant_docs):
        trip = []
        for h, t, c in d["triples"]:
            if c in code:
                trip.append([h, t, code[c]])
            else:
                missing[c] += 1
        dd.append({"id": d["id"], "tokens": d["tokens"], "sentence_spans": d["sentence_spans"],
                   "entities": [{"type": tcode[e["type"]], "mentions": e["mentions"]}
                                for e in d["entities"]],
                   "triples": trip})
    rschema = sorted(r["id"] for r in bank["relations"] if r["source"] == "redocred")
    tschema = sorted(t["id"] for t in bank["entity_types"] if t["source"] == "docred")
    (_corpus() / KNOWLEDGE_FILES["docred_distant"]).write_text(
        json.dumps({"train": dd, "schema": rschema, "type_schema": tschema}))
    out["docred_distant"] = {"train": len(dd), "triples": sum(len(x["triples"]) for x in dd),
                             "codes_not_in_bank": dict(missing.most_common(5))}
    return out


HOLDOUT_FILE = "kg_holdout.json"
DEV_MOD = 40          # teacher windows whose id hashes to 0 mod 40 are dev, never trained


def choose_holdout(bank: dict, n_general: int = 32, n_business: int = 8,
                   min_facts: int = 60) -> dict:
    """A fixed zero-shot holdout: general and business relations the teacher
    used at least ``min_facts`` times, picked by hash. Training never sees
    them; held-out teacher windows score them every evaluation."""
    byname = {r["name"]: r for r in bank["relations"]}
    count = Counter()
    for it in json.loads((_corpus() / "kg_teacher.json").read_text())["items"]:
        for f in it["payload"]["facts"]:
            r = byname.get(f["relation"])
            if r is not None and r["source"] in ("general", "business"):
                count[r["id"]] += 1
    out = []
    for src, n in (("general", n_general), ("business", n_business)):
        elig = [i for i, c in count.items() if c >= min_facts and bank["relations"][i]["source"] == src]
        elig.sort(key=lambda i: hashlib.sha1(f"zs:{bank['relations'][i]['name']}".encode()).hexdigest())
        out += elig[:n]
    rec = {"relations": sorted(out), "names": [bank["relations"][i]["name"] for i in sorted(out)],
           "facts": {bank["relations"][i]["name"]: count[i] for i in sorted(out)},
           "dev_mod": DEV_MOD, "bank_hash": bank.get("hash")}
    (_corpus() / HOLDOUT_FILE).write_text(json.dumps(rec, indent=1))
    return rec


def load_holdout() -> dict | None:
    p = _corpus() / HOLDOUT_FILE
    return json.loads(p.read_text()) if p.exists() else None


def _is_dev(item_id: str) -> bool:
    w = item_id.split(":")[0]
    return int(hashlib.sha1(f"kgdev:{w}".encode()).hexdigest(), 16) % DEV_MOD == 0


def span_targets(spans, W: int, word_ok, schema, partial: bool,
                 stats: Counter | None = None, tag: str = "") -> dict:
    """Typed entity spans for the span-type head: ``(start, end_exclusive,
    type id)`` inside the window, the type schema the source was annotated
    against, and whether its spans are partial."""
    keep = [(int(s), int(e), int(t)) for t, s, e in spans if e <= W and word_ok[s]]
    if stats is not None:
        stats[f"{tag}_spans_past_limit"] += len(spans) - len(keep)
        stats[f"{tag}_spans"] += len(keep)
    return {"spans": np.asarray(sorted(set(keep)), dtype=np.int32).reshape(-1, 3),
            "schema": sorted(set(int(x) for x in schema)), "partial": partial}


def _rel(ents, triples, schema, n_rel, partial, keep, evidence=()):
    return {"spans": ents, "types": np.zeros(len(ents), dtype=np.int32), "triples": triples,
            "evidence": list(evidence), "schema": schema, "partial": partial,
            "unlabelled_keep": keep}


def fewrel_features(x: dict, schema, tok, max_subwords: int, n_rel: int, n_cls: int,
                    stats: Counter, markers: bool = True) -> dict | None:
    from .data import blank_window, marker_features
    spans = [(0, len(x["tokens"]))]
    f = blank_window(x["tokens"], spans, tok, max_subwords, n_cls)
    W = f["_W"]
    ents = []
    for side in (x["head"], x["tail"]):
        ms = [(a, b) for a, b in side if b <= W and f["word_ok"][a]]
        if not ms:
            stats["fewrel_past_limit"] += 1
            return None
        ents.append(np.asarray(ms, dtype=np.int32))
    f.update(coref=None, _n_rel=n_rel,
             rel=_rel(ents, [(0, 1, x["rel"])], schema, n_rel, True, UNLABELLED_KEEP["fewrel"]))
    if markers:
        f["rm"] = marker_features(x["tokens"], spans, ents, tok)
    stats["fewrel_items"] += 1
    return f


def fewnerd_features(x: dict, schema, tok, max_subwords: int, n_cls: int,
                     stats: Counter) -> dict:
    from .data import blank_window
    spans = [(0, len(x["tokens"]))]
    f = blank_window(x["tokens"], spans, tok, max_subwords, n_cls)
    f.update(coref=None, rel=None,
             ents=span_targets(x["spans"], f["_W"], f["word_ok"], schema, False, stats, "fewnerd"))
    stats["fewnerd_items"] += 1
    return f


def distant_features(d: dict, schema, type_schema, tok, max_subwords: int, n_rel: int,
                     n_cls: int, stats: Counter, markers: bool = True) -> dict | None:
    from .data import blank_window, marker_features
    spans = [tuple(s) for s in d["sentence_spans"]]
    f = blank_window(d["tokens"], spans, tok, max_subwords, n_cls)
    W = f["_W"]
    keep, ents, typed = {}, [], []
    for i, e in enumerate(d["entities"]):
        ms = [(a, b) for a, b in e["mentions"] if b <= W and f["word_ok"][a]]
        if ms:
            keep[i] = len(ents)
            ents.append(np.asarray(ms, dtype=np.int32))
            typed += [(e["type"], a, b) for a, b in ms]
    if len(ents) < 2:
        stats["distant_docs_dropped"] += 1
        return None
    trip = [(keep[h], keep[t], r) for h, t, r in d["triples"] if h in keep and t in keep]
    stats["distant_triples_past_limit"] += len(d["triples"]) - len(trip)
    f.update(coref=None, _n_rel=n_rel,
             rel=_rel(ents, trip, schema, n_rel, True, UNLABELLED_KEEP["docred_distant"]),
             ents=span_targets(typed, W, f["word_ok"], type_schema, False))
    if markers:
        f["rm"] = marker_features(d["tokens"], spans, ents, tok)
    stats["distant_docs"] += 1
    stats["distant_triples"] += len(trip)
    return f


def pretrain_features(d: dict, bank_ids: dict, tok, max_subwords: int, n_rel: int,
                      n_cls: int, stats: Counter, markers: bool = True) -> dict | None:
    """A relation-pretraining document (``v6.relation_pretrain``): Wikipedia
    text with its given entities (and any the teacher added), the teacher's
    facts by bank id, and the Wikidata-style schema it was asked about.
    Partial, like the teacher corpus."""
    from .data import blank_window, marker_features
    spans = [tuple(s) for s in d["sentence_spans"]]
    f = blank_window(d["tokens"], spans, tok, max_subwords, n_cls)
    W = f["_W"]
    keep, ents = {}, []
    for i, e in enumerate(d["entities"]):
        ms = sorted({(int(a), int(b)) for a, b in e["mentions"] if b <= W and f["word_ok"][a]})
        if ms:
            keep[i] = len(ents)
            ents.append(np.asarray(ms, dtype=np.int32))
    if len(ents) < 2:
        stats["pretrain_docs_dropped"] += 1
        return None
    trip = []
    for h, t, name in d["facts"]:
        r = bank_ids.get(name)
        if r is not None and h in keep and t in keep and keep[h] != keep[t]:
            trip.append((keep[h], keep[t], r))
    schema = sorted(bank_ids[n] for n in d["relations"] if n in bank_ids)
    f.update(coref=None, _n_rel=n_rel,
             rel=_rel(ents, sorted(set(trip)), schema, n_rel, True, UNLABELLED_KEEP["pretrain"]))
    if markers:
        f["rm"] = marker_features(d["tokens"], spans, ents, tok)
    stats["pretrain_docs"] += 1
    stats["pretrain_triples"] += len(set(trip))
    return f


def trex_features(d: dict, never: set, tok, max_subwords: int, n_rel: int, n_cls: int,
                  stats: Counter, markers: bool = True) -> dict | None:
    """A T-REx document (``gold.trex``): entities grouped by Wikidata id,
    their linked mentions typed with Wikidata classes (span head), and the
    aligned facts by bank id (relation head) -- both partial, against the
    document's relation and type schemas."""
    from .data import blank_window, marker_features
    spans = [tuple(s) for s in d["sentence_spans"]]
    f = blank_window(d["tokens"], spans, tok, max_subwords, n_cls)
    W = f["_W"]
    keep, ents, typed = {}, [], []
    for i, e in enumerate(d["entities"]):
        ms = sorted({(int(a), int(b)) for a, b in e["mentions"] if b <= W and f["word_ok"][a]})
        if ms:
            keep[i] = len(ents)
            ents.append(np.asarray(ms, dtype=np.int32))
        typed += [(t, a, b) for a, b in e["typed"] for t in e["types"]]
    f["ents"] = span_targets(typed, W, f["word_ok"], d["types"], True, stats, "trex")
    trip = sorted({(keep[h], keep[t], r) for h, t, r in d["facts"]
                   if r not in never and h in keep and t in keep and keep[h] != keep[t]})
    schema = sorted(r for r in d["relations"] if r not in never)
    if len(ents) >= 2:
        f.update(coref=None, _n_rel=n_rel,
                 rel=_rel(ents, trip, schema, n_rel, True, UNLABELLED_KEEP["trex"]))
        if markers:
            f["rm"] = marker_features(d["tokens"], spans, ents, tok)
    else:
        f.update(coref=None, rel=None)
    stats["trex_docs"] += 1
    stats["trex_triples"] += len(trip)
    return f


def knowledge_features(source: str, split: str, tok, max_subwords: int, n_cls: int,
                       bank: dict, markers: bool = True, limit: int | None = None,
                       stats: Counter | None = None, progress=None) -> list[dict]:
    """Feature dicts for one knowledge source, cached on disk per (source,
    split, tokenizer, limit, markers, bank hash, source file)."""
    import pickle
    from .data import FEATURES
    stats = Counter() if stats is None else stats
    if source == "teacher":
        path = _corpus() / "kg_teacher.json"
    else:
        path = _corpus() / KNOWLEDGE_FILES[source]
    if not path.exists():
        raise SystemExit(f"{path} missing; run python -m v6.train.kdata --prep")
    hold_sig = (load_holdout() or {}).get("relations") if source == "teacher" else None
    key = hashlib.sha256(json.dumps([FEATURE_VERSION, hold_sig, source, split, tok.name_or_path,
                                     max_subwords, markers, limit,
                                     bank.get("hash"), _digest(path)]).encode()
                         ).hexdigest()[:16]
    cache = FEATURES / f"knowledge-{source}-{split}-{key}.pkl"
    from .data import USED_CACHES
    USED_CACHES.append(cache)
    if cache.exists():
        with cache.open("rb") as fh:
            feats, saved = pickle.load(fh)
        stats.update(saved)
        return feats
    n_rel = len(bank["relations"])
    local = Counter()

    def tick(seq):
        # A long source on a slow VM CPU must not read as a dead run.
        for i, x in enumerate(seq):
            if progress is not None and i % 2000 == 0:
                progress(f"featurising {source}: {i:,}")
            yield x
    if source == "teacher":
        from ..kg_teacher import load_export
        pairs = [(it, p) for it, p in load_export(path) if _is_dev(it.id) == (split == "dev")]
        if split == "dev":
            # Only windows whose schema lists a held-out relation say anything
            # about zero-shot reading.
            hold = set((load_holdout() or {}).get("names", []))
            pairs = [(it, p) for it, p in pairs if hold & {r["name"] for r in it.relations}]
        feats = [f for it, p in tick(pairs[:limit])
                 if (f := teacher_features(it, p, tok, max_subwords, bank, n_cls, local, markers))]
    else:
        data = json.loads(path.read_text())
        rows = data[split][:limit] if split in data else []
        never = never_trained(bank)
        if source == "pretrain":
            ids = {r["name"]: r["id"] for r in bank["relations"] if r["id"] not in never}
            feats = [f for x in tick(data["docs"][:limit]) if (f := pretrain_features(
                x, ids, tok, max_subwords, n_rel, n_cls, local, markers))]
        elif source == "trex":
            feats = [f for x in tick(data["docs"][:limit]) if (f := trex_features(
                x, never, tok, max_subwords, n_rel, n_cls, local, markers))]
        elif source == "fewrel":
            sch = data["val_schema"] if split == "val" else data["schema"]
            if split != "val":
                rows = [x for x in rows if x["rel"] not in never]
                sch = [r for r in sch if r not in never]
            feats = [f for x in tick(rows) if (f := fewrel_features(
                x, sch, tok, max_subwords, n_rel, n_cls, local, markers))]
        elif source == "fewnerd":
            feats = [fewnerd_features(x, data["schema"], tok, max_subwords, n_cls, local)
                     for x in tick(rows)]
        else:
            feats = [f for x in tick(rows) if (f := distant_features(
                x, data["schema"], data["type_schema"], tok, max_subwords, n_rel, n_cls,
                local, markers))]
    FEATURES.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump((feats, dict(local)), fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache)
    stats.update(local)
    return feats


def main() -> int:
    import argparse
    import time
    from transformers import AutoTokenizer
    from ..label_bank import load as load_bank
    from .encoders import get
    from .vocab import Vocab
    ap = argparse.ArgumentParser(description="prepare and featurise the knowledge sources")
    ap.add_argument("--prep", action="store_true", help="write the public sources' JSONs")
    ap.add_argument("--featurise", default="teacher,fewrel,fewnerd,docred_distant")
    ap.add_argument("--encoder", default="deberta-v3-large")
    ap.add_argument("--distant-docs", type=int, default=40000)
    ap.add_argument("--holdout", action="store_true", help="choose the zero-shot relation holdout")
    args = ap.parse_args()
    bank = load_bank()
    if args.prep:
        print(json.dumps(prepare_knowledge(bank, args.distant_docs), indent=1), flush=True)
    if args.holdout or args.prep:
        rec = choose_holdout(bank)
        print(f"zero-shot holdout: {len(rec['relations'])} relations: {', '.join(rec['names'])}",
              flush=True)
    spec = get(args.encoder)
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    n_cls = Vocab.load().size("cls")
    for src in [s for s in args.featurise.split(",") if s]:
        for split in {"fewnerd": ("train", "dev"), "fewrel": ("train", "val"),
                      "teacher": ("train", "dev")}.get(src, ("train",)):
            t0, st = time.time(), Counter()
            feats = knowledge_features(src, split, tok, spec.max_subwords, n_cls, bank, stats=st)
            print(f"{src}/{split}: {len(feats):,} items in {time.time() - t0:.0f}s; "
                  + ", ".join(f"{k} {v:,}" for k, v in sorted(st.items())), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

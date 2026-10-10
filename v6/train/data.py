"""Corpus rows -> tensors, for any encoder.

    uv run python -m v6.train.data --verify --encoder ettin-1b --split dev

**Every encoder sees the same text.** Words are joined with single spaces and
the subwords are mapped back to words by character offset. Tokenizing
pre-split words instead (``is_split_into_words``) is not neutral: Ettin's
byte-level BPE then encodes each word as if it continued the previous one --
"argued" becomes ``arg ued`` -- which is out-of-distribution input and would
handicap it in the comparison. A word is represented by its first subword.

**The masking contract** (TRAINING.md §5). A label contributes no gradient
when any of these hold, and is ``IGNORE`` in the tensors:

- its layer is absent for the row (``loss_mask[layer] is True``);
- it is in a sentence that is not a tree (``dep_tokens``) or an ill-formed
  NER sequence (``ner_tokens``);
- its sentence's CLS or sentiment is missing (``<layer>_sentences``);
- its word is past this encoder's limit, or produced no subword at all;
- for DEP, its head is past the limit; for CLS and sentiment, its sentence
  ends past the limit;
- for morph and lemma, its class is outside the train vocabulary;
- for the SRL predicate head, it is a VERB with no frame (dropped as
  malformed) -- unknown, not negative.

The corpus's own ``encoder_word_limit`` is DeBERTa's and is not used: the
limit is recomputed per row with the tokenizer actually in use.

``--verify`` rebuilds every label from the collated batch and compares it to
the corpus row, computing the expected masks independently of the encoder.
This is where a silent corruption would otherwise live.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..config import DATA_DIR
from .vocab import Vocab, apply_script, edit_script

CORPUS = DATA_DIR / "v6-corpus" / "corpus"
FEATURES = DATA_DIR / "v6-corpus" / "features"
IGNORE = -100
TOKEN_LAYERS = ("pos", "morph", "lemma", "ner")
COLUMNS = ["window_id", "domain", "tokens", "sentence_spans", "n_tokens",
           "pos", "ner", "dep", "dep_heads", "lemma", "morph", "srl_frames",
           "cls", "sentiment", "loss_mask"]


def load_rows(split: str, columns: list[str] | None = None,
              domains: list[str] | None = None,
              limit: int | None = None) -> list[dict]:
    """Rows of one split, in a stable order (file path, then row)."""
    import pyarrow.parquet as pq
    files = sorted((CORPUS / f"split={split}").glob("domain=*/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {CORPUS}/split={split}")
    out = []
    for f in files:
        if domains and f.parent.name.removeprefix("domain=") not in domains:
            continue
        out.extend(pq.ParquetFile(f).read(columns=columns or COLUMNS).to_pylist())
        if limit and len(out) >= limit:
            return out[:limit]
    return out


# ── alignment ───────────────────────────────────────────────────────
def align(words: list[str], tok, max_subwords: int):
    """``(input_ids, first_subword, limit)`` for one window.

    ``first_subword[i]`` is the index of word ``i``'s first subword, or -1 if
    the tokenizer produced none for it. ``limit`` is the number of leading
    words the encoder sees: the first word whose first subword falls past
    ``max_subwords`` and everything after it are cut.
    """
    starts, pos = [], 0
    for w in words:
        starts.append(pos)
        pos += len(w) + 1
    enc = tok(" ".join(words), return_offsets_mapping=True,
              add_special_tokens=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    first = [-1] * len(words)
    for ti, (s, e) in enumerate(offs):
        if e <= s:
            continue                                   # special token
        # The token's last character identifies its word; a leading space in
        # the offset (byte-level BPE) belongs to the gap, not the word.
        wi = bisect.bisect_right(starts, e - 1) - 1
        if first[wi] < 0:
            first[wi] = ti
    limit = len(words)
    if len(ids) > max_subwords:
        cut = max_subwords - 1                         # room for [SEP]
        ids = ids[:cut] + [tok.sep_token_id]
        for i, f in enumerate(first):
            if f >= cut:
                limit = i
                break
    return ids, first[:limit], limit


def sentence_encodings(words, spans, starts, tok, max_subwords) -> dict:
    """``{sentence start: (input_ids, first_subword)}`` for the sentences that
    hold a predicate, each tokenised on its own -- whole, even where the
    window's encoder limit cut it."""
    out = {}
    for s, e in spans:
        if s in starts:
            ids, first, _ = align(words[s:e], tok, max_subwords)
            out[s] = (np.asarray(ids, dtype=np.int32), np.asarray(first, dtype=np.int32))
    return out


def _sentence_of(spans, n):
    out = np.full(n, -1, dtype=np.int32)
    for j, (s, e) in enumerate(spans):
        out[s:min(e, n)] = j
    return out


# ── one row ─────────────────────────────────────────────────────────
def encode_row(row: dict, tok, max_subwords: int, vocab: Vocab,
               stats: Counter, srl_sentences: bool = False,
               phase2: dict | None = None) -> dict:
    """One corpus row as features. ``srl_sentences`` also tokenises each
    predicate's sentence on its own, for the predicate-conditioned SRL head
    (``model.V6Model(srl_mode="conditioned")``), which runs one encoder pass
    per predicate over its sentence as v5 did."""
    words = row["tokens"]
    mask = json.loads(row["loss_mask"] or "{}")
    ids, first, W = align(words, tok, max_subwords)
    rep = np.array([f >= 0 for f in first], dtype=bool)
    stats["words"] += len(words)
    stats["words_past_limit"] += len(words) - W
    stats["words_without_subword"] += int((~rep).sum())

    def token_layer(layer, values, transform=None, extra=None):
        out = np.full(W, IGNORE, dtype=np.int16)
        if values is None or mask.get(layer) is True:
            return out
        for i in range(W):
            if not rep[i] or (extra is not None and extra[i]):
                continue
            lab = transform(i, values[i]) if transform else values[i]
            j = vocab.id(layer, lab)
            if j is None:
                stats[f"{layer}_out_of_vocab"] += 1
            else:
                out[i] = j
        return out

    feats = {
        "window_id": row["window_id"],
        "domain": row["domain"],
        "input_ids": np.asarray(ids, dtype=np.int32),
        "first_subword": np.asarray(first, dtype=np.int32),
        "word_ok": rep,
        "sent_of_word": _sentence_of(row["sentence_spans"], W),
        "pos": token_layer("pos", row["pos"]),
        "morph": token_layer("morph", row["morph"]),
        "lemma": token_layer("lemma", row["lemma"],
                             transform=lambda i, l: edit_script(words[i], l)),
        "ner": token_layer("ner", row["ner"], extra=mask.get("ner_tokens")),
    }

    # DEP: the head target is 0 for root and head+1 otherwise, so the root
    # sits in column 0 of a [W, W+1] score matrix.
    head = np.full(W, IGNORE, dtype=np.int16)
    rel = np.full(W, IGNORE, dtype=np.int16)
    if row["dep"] is not None and not mask.get("dep"):
        bad = mask.get("dep_tokens")
        for i in range(W):
            h = row["dep_heads"][i]
            if not rep[i] or (bad and bad[i]):
                continue
            if h >= 0 and (h >= W or not rep[h]):
                stats["dep_head_past_limit"] += 1
                continue
            head[i] = 0 if h < 0 else h + 1
            r = vocab.id("dep", row["dep"][i])
            if r is None:
                stats["dep_out_of_vocab"] += 1
            else:
                rel[i] = r
    feats["dep_head"], feats["dep_rel"] = head, rel

    # SRL: a predicate flag per word, and per frame the role tags inside the
    # predicate's own sentence (outside it they are trivially O, and the
    # model never scores across sentences). Stored as the sentence slice only.
    pred = np.full(W, IGNORE, dtype=np.int16)
    frames = []
    if row["srl_frames"] is not None and not mask.get("srl"):
        pos_ok = row["pos"] is not None and not mask.get("pos")
        if pos_ok:
            for i in range(W):
                if rep[i]:
                    pred[i] = IGNORE if row["pos"][i] == "VERB" else 0
        spans = row["sentence_spans"]
        for fr in row["srl_frames"]:
            p = fr["predicate_idx"]
            if p >= W or not rep[p]:
                stats["srl_frames_past_limit"] += 1
                continue
            pred[p] = 1
            s, e = spans[feats["sent_of_word"][p]]
            e = min(e, W)
            roles = np.full(e - s, IGNORE, dtype=np.int16)
            for i in range(s, e):
                if rep[i]:
                    j = vocab.id("srl", fr["tags"][i])
                    if j is None:
                        stats["srl_out_of_vocab"] += 1
                    else:
                        roles[i - s] = j
            frames.append((p, s, roles))
    feats["srl_pred"] = pred
    feats["frames"] = frames
    if srl_sentences:
        feats["srl_sents"] = sentence_encodings(
            words, row["sentence_spans"], {s for _, s, _ in frames}, tok, max_subwords)

    # Sentence layers. A sentence is pooled only if it ends inside the limit.
    S = len(row["sentence_spans"])
    whole = np.array([e <= W for _, e in row["sentence_spans"]], dtype=bool)
    cls = np.zeros((S, vocab.size("cls")), dtype=np.int8)
    cls_ok = np.zeros(S, dtype=bool)
    if row["cls"] is not None and not mask.get("cls"):
        miss = mask.get("cls_sentences") or [False] * S
        for j in range(S):
            if whole[j] and not miss[j]:
                cls_ok[j] = True
                for lab in row["cls"][j]:
                    k = vocab.id("cls", lab)
                    if k is None:
                        stats["cls_out_of_vocab"] += 1
                        cls_ok[j] = False
                    else:
                        cls[j, k] = 1
    sent = np.full(S, IGNORE, dtype=np.int16)
    if row["sentiment"] is not None and not mask.get("sentiment"):
        miss = mask.get("sentiment_sentences") or [False] * S
        for j in range(S):
            v = row["sentiment"][j]
            if whole[j] and not miss[j] and v is not None:
                k = vocab.id("sentiment", v)
                if k is None:
                    stats["sentiment_out_of_vocab"] += 1
                else:
                    sent[j] = k
    feats.update(sent_whole=whole, cls=cls, cls_ok=cls_ok, sentiment=sent)
    if phase2 is not None:
        feats["coref"] = coref_features(row, mask, W, rep, stats)
        feats["rel"] = relation_features(row, W, rep, vocab, phase2, stats)
        feats["_n_rel"] = vocab.size("relations")
    return feats


# ── phase 2: coreference and relations ──────────────────────────────
ENTITIES = DATA_DIR / "v6-corpus" / "entities_docred.json"


def load_phase2() -> dict:
    """``{"entities": {window_id: vertexSet}}`` -- the entity clusters ATLOP
    labelled, as the corpus built them (``v6.entities.build_entities``)."""
    docs = json.loads(ENTITIES.read_text())
    return {"entities": {d["title"]: d["vertexSet"] for d in docs}}


def _head_word(s: int, e: int, heads) -> int:
    """The span's syntactic head: the token whose DEP parent lies outside
    ``[s, e]``. Several (a fragment) -> the rightmost; no parse -> the last."""
    if heads is None:
        return e
    out = [i for i in range(s, e + 1) if heads[i] < 0 or not s <= heads[i] <= e]
    return out[-1] if out else e


def coref_features(row: dict, mask: dict, W: int, rep, stats: Counter) -> dict | None:
    """Word-level coreference targets (Dobrovolskii 2021): each mention as its
    head word, inclusive span and cluster. ``None`` when coref is absent."""
    if row.get("coref") is None or mask.get("coref"):
        return None
    heads, starts, ends, clusters = [], [], [], []
    taken: set[int] = set()
    for ci, chain in enumerate(row["coref"]):
        kept = []
        for s, e in chain:
            if e >= W or not rep[s]:
                stats["coref_mentions_past_limit"] += 1
                continue
            h = _head_word(s, e, row.get("dep_heads"))
            if h in taken:
                # Word-level coref represents a mention by its head word, so
                # two mentions sharing one cannot both be kept.
                stats["coref_mentions_shared_head"] += 1
                continue
            kept.append((h, s, e))
        if len(kept) < 2:                     # a chain needs two mentions
            stats["coref_chains_dropped"] += 1
            continue
        for h, s, e in kept:
            taken.add(h)
            heads.append(h); starts.append(s); ends.append(e); clusters.append(ci)
        stats["coref_chains"] += 1
    stats["coref_mentions"] += len(heads)
    return {"head": np.asarray(heads, dtype=np.int32), "start": np.asarray(starts, dtype=np.int32),
            "end": np.asarray(ends, dtype=np.int32), "cluster": np.asarray(clusters, dtype=np.int32)}


def relation_features(row: dict, W: int, rep, vocab: Vocab, phase2: dict,
                      stats: Counter) -> dict | None:
    """Entities (mention spans, coarse type) and triples for a window ATLOP
    labelled. Every such window supervises, including the 17,899 where ATLOP
    found nothing: those are negatives, not absences. ``None`` otherwise."""
    vertex = phase2["entities"].get(row["window_id"])
    if vertex is None:
        return None
    offs = [s for s, _ in row["sentence_spans"]]
    keep, spans, types = {}, [], []
    for ei, mentions in enumerate(vertex):
        ms = []
        for m in mentions:
            a = offs[m["sent_id"]] + m["pos"][0]
            b = offs[m["sent_id"]] + m["pos"][1]        # exclusive
            if b > W or not rep[a]:
                stats["rel_mentions_past_limit"] += 1
                continue
            ms.append((a, b))
        if not ms:
            stats["rel_entities_past_limit"] += 1
            continue
        keep[ei] = len(spans)
        spans.append(np.asarray(ms, dtype=np.int32))
        types.append(vocab.id("ent_type", mentions[0]["type"]))
    if len(spans) < 2:
        stats["rel_windows_under_two_entities"] += 1
        return None
    triples = []
    for r in row.get("relations") or []:
        if r["head"] not in keep or r["tail"] not in keep:
            stats["rel_triples_past_limit"] += 1
            continue
        k = vocab.id("relations", r["relation"])
        if k is None:
            stats["rel_out_of_vocab"] += 1
            continue
        triples.append((keep[r["head"]], keep[r["tail"]], k))
    stats["rel_windows"] += 1
    stats["rel_windows_negative_only"] += not triples
    stats["rel_triples"] += len(triples)
    return {"spans": spans, "types": np.asarray(types, dtype=np.int16),
            "triples": np.asarray(triples, dtype=np.int32).reshape(-1, 3)}


# ── features on disk ────────────────────────────────────────────────
def _key(spec, split: str, vocab: Vocab, limit: int | None,
         srl_sentences: bool = False, phase2: bool = False) -> str:
    manifest = DATA_DIR / "v6-corpus" / "MANIFEST.json"
    h = hashlib.sha256()
    h.update(json.dumps([spec.hf_id, spec.max_subwords, split, limit,
                         vocab.to_json()["labels"]] + (["srl_sentences"] if srl_sentences else [])
                        + (["phase2"] if phase2 else [])
                        ).encode())
    h.update(manifest.read_bytes() if manifest.exists() else b"")
    for f in sorted((CORPUS / f"split={split}").glob("domain=*/*.parquet")):
        st = f.stat()
        h.update(f"{f.name}{st.st_size}{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def featurise(spec, split: str, vocab: Vocab, tok=None,
              limit: int | None = None, srl_sentences: bool = False,
              phase2: bool = False) -> list[dict]:
    """Encoded rows for one split, cached per (tokenizer, limit, vocab, corpus).

    The key covers the corpus files' sizes and mtimes, so a re-assembled
    corpus is re-featurised rather than silently served from a stale cache.
    """
    import pickle
    path = FEATURES / (f"{spec.name}-{split}-"
                       f"{_key(spec, split, vocab, limit, srl_sentences, phase2)}.pkl")
    if path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec.hf_id)
    t0 = time.time()
    stats = Counter()
    rows = load_rows(split, limit=limit)
    p2 = load_phase2() if phase2 else None
    if phase2:
        rows = load_rows(split, limit=limit, columns=COLUMNS + ["coref", "relations"])
    out = [encode_row(r, tok, spec.max_subwords, vocab, stats, srl_sentences, p2)
           for r in rows]
    print(f"featurised {split}: {len(out)} windows in {time.time() - t0:.0f}s; "
          + ", ".join(f"{k} {v:,}" for k, v in sorted(stats.items())),
          flush=True)
    FEATURES.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return out


# ── batching ────────────────────────────────────────────────────────
def batch_order(feats: list[dict], batch_size: int, seed: int,
                epoch: int) -> list[list[int]]:
    """Length-bucketed batches, reproducible from (seed, epoch).

    Resume replays the same order and skips the batches already taken, so a
    resumed run sees the same sample order as an uninterrupted one.
    """
    rng = np.random.default_rng([seed, epoch])
    order = rng.permutation(len(feats))
    chunk = batch_size * 50
    batches = []
    for c in range(0, len(order), chunk):
        part = sorted(order[c:c + chunk], key=lambda i: len(feats[i]["input_ids"]))
        batches += [list(map(int, part[i:i + batch_size]))
                    for i in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


def collate(items: list[dict], pad_id: int):
    import torch
    B = len(items)
    T = max(len(x["input_ids"]) for x in items)
    W = max(len(x["first_subword"]) for x in items)
    S = max(len(x["sent_whole"]) for x in items)
    C = items[0]["cls"].shape[1]

    def full(shape, v, dtype):
        return torch.full(shape, v, dtype=dtype)

    out = {
        "input_ids": full((B, T), pad_id, torch.long),
        "attention_mask": torch.zeros(B, T, dtype=torch.long),
        "first_subword": torch.zeros(B, W, dtype=torch.long),
        "word_mask": torch.zeros(B, W, dtype=torch.bool),
        "sent_of_word": full((B, W), -1, torch.long),
        "sent_whole": torch.zeros(B, S, dtype=torch.bool),
        "cls": torch.zeros(B, S, C),
        "cls_ok": torch.zeros(B, S, dtype=torch.bool),
        "sentiment": full((B, S), IGNORE, torch.long),
    }
    for k in TOKEN_LAYERS + ("dep_head", "dep_rel", "srl_pred"):
        out[k] = full((B, W), IGNORE, torch.long)
    fb, fp, roles = [], [], []
    for b, x in enumerate(items):
        n, w, s = len(x["input_ids"]), len(x["first_subword"]), len(x["sent_whole"])
        out["input_ids"][b, :n] = torch.from_numpy(x["input_ids"].astype(np.int64))
        out["attention_mask"][b, :n] = 1
        out["first_subword"][b, :w] = torch.from_numpy(
            np.maximum(x["first_subword"], 0).astype(np.int64))
        out["word_mask"][b, :w] = torch.from_numpy(x["word_ok"])
        out["sent_of_word"][b, :w] = torch.from_numpy(x["sent_of_word"].astype(np.int64))
        for k in TOKEN_LAYERS + ("dep_head", "dep_rel", "srl_pred"):
            out[k][b, :w] = torch.from_numpy(x[k].astype(np.int64))
        out["sent_whole"][b, :s] = torch.from_numpy(x["sent_whole"])
        out["cls"][b, :s] = torch.from_numpy(x["cls"].astype(np.float32))
        out["cls_ok"][b, :s] = torch.from_numpy(x["cls_ok"])
        out["sentiment"][b, :s] = torch.from_numpy(x["sentiment"].astype(np.int64))
        for p, start, r in x["frames"]:
            row = torch.full((W,), IGNORE, dtype=torch.long)
            row[start:start + len(r)] = torch.from_numpy(r.astype(np.int64))
            fb.append(b); fp.append(p); roles.append(row)
    out["frame_batch"] = torch.tensor(fb, dtype=torch.long)
    out["frame_pred"] = torch.tensor(fp, dtype=torch.long)
    out["frame_roles"] = (torch.stack(roles) if roles
                          else torch.zeros(0, W, dtype=torch.long))
    if "srl_sents" in items[0]:
        out.update(_collate_srl_sentences(items, pad_id))
    if "coref" in items[0]:
        out.update(_collate_phase2(items, len(items[0]["rel"]["types"])
                                   if items[0].get("rel") else None))
    return out


def _collate_phase2(items: list[dict], _unused=None) -> dict:
    """Coreference mentions and relation entities/pairs, flattened over the
    batch with a per-item index, as the SRL frames are.

    Relation pairs are every ordered (head, tail) of distinct entities in a
    supervised window; ``pair_labels[:, 0]`` is the no-relation class, set
    when a pair has no triple -- ATLOP's adaptive threshold class.
    """
    import torch
    cb, ch, cs, ce, cc, cok = [], [], [], [], [], []
    eb, et, me, ms, mx = [], [], [], [], []
    pb, ph, pt, plab = [], [], [], []
    n_rel = None
    cluster_base = ent_base = 0
    for b, x in enumerate(items):
        c = x.get("coref")
        cok.append(c is not None)
        if c is not None and len(c["head"]):
            cb += [b] * len(c["head"]); ch += c["head"].tolist(); cs += c["start"].tolist()
            ce += c["end"].tolist(); cc += (c["cluster"] + cluster_base).tolist()
            cluster_base += int(c["cluster"].max()) + 1
        r = x.get("rel")
        if r is None:
            continue
        E = len(r["spans"])
        for k, spans in enumerate(r["spans"]):
            eb.append(b); et.append(int(r["types"][k]))
            for a, z in spans:
                me.append(ent_base + k); ms.append(int(a)); mx.append(int(z))
        n_rel = n_rel or x["_n_rel"]
        lab = {}
        for h, tl, k in r["triples"]:
            lab.setdefault((int(h), int(tl)), set()).add(int(k))
        for h in range(E):
            for tl in range(E):
                if h == tl:
                    continue
                pb.append(b); ph.append(ent_base + h); pt.append(ent_base + tl)
                plab.append(lab.get((h, tl), set()))
        ent_base += E
    L = torch.long
    out = {"coref_batch": torch.tensor(cb, dtype=L), "coref_head": torch.tensor(ch, dtype=L),
           "coref_start": torch.tensor(cs, dtype=L), "coref_end": torch.tensor(ce, dtype=L),
           "coref_cluster": torch.tensor(cc, dtype=L), "coref_ok": torch.tensor(cok),
           "ent_batch": torch.tensor(eb, dtype=L), "ent_type": torch.tensor(et, dtype=L),
           "men_ent": torch.tensor(me, dtype=L), "men_start": torch.tensor(ms, dtype=L),
           "men_end": torch.tensor(mx, dtype=L),
           "pair_batch": torch.tensor(pb, dtype=L), "pair_h": torch.tensor(ph, dtype=L),
           "pair_t": torch.tensor(pt, dtype=L)}
    R = (n_rel or items[0].get("_n_rel") or 0) + 1
    labels = torch.zeros(len(plab), R)
    for i, s in enumerate(plab):
        if s:
            labels[i, [k + 1 for k in s]] = 1.0
        else:
            labels[i, 0] = 1.0
    out["pair_labels"] = labels
    return out


def _collate_srl_sentences(items: list[dict], pad_id: int) -> dict:
    """Per frame, in ``frame_batch`` order: its sentence's subwords, the first
    subword of each of its words, the predicate's subword, and where the
    sentence starts in the window."""
    import torch
    encs = [(x["srl_sents"][s], p - s, s) for x in items for p, s, _ in x["frames"]]
    F_ = len(encs)
    Ls = max((len(e[0][0]) for e in encs), default=1)
    Ws = max((len(e[0][1]) for e in encs), default=1)
    ids = torch.full((F_, Ls), pad_id, dtype=torch.long)
    mask = torch.zeros(F_, Ls, dtype=torch.long)
    first = torch.zeros(F_, Ws, dtype=torch.long)
    wmask = torch.zeros(F_, Ws, dtype=torch.bool)
    psub = torch.zeros(F_, dtype=torch.long)
    start = torch.zeros(F_, dtype=torch.long)
    for f, ((sid, sfirst), plocal, s) in enumerate(encs):
        ids[f, :len(sid)] = torch.from_numpy(sid.astype(np.int64))
        mask[f, :len(sid)] = 1
        first[f, :len(sfirst)] = torch.from_numpy(np.maximum(sfirst, 0).astype(np.int64))
        wmask[f, :len(sfirst)] = torch.from_numpy(sfirst >= 0)
        psub[f] = int(sfirst[plocal]) if plocal < len(sfirst) and sfirst[plocal] >= 0 else 0
        start[f] = s
    return {"frame_sent_ids": ids, "frame_sent_mask": mask, "frame_sent_first": first,
            "frame_sent_wmask": wmask, "frame_pred_sub": psub, "frame_sent_start": start}


# ── round-trip verification ─────────────────────────────────────────
def _expected(row: dict, W: int, rep: list[bool], vocab: Vocab) -> dict:
    """What the batch must say, computed from the row alone.

    Deliberately written apart from ``encode_row``: the point is two
    derivations agreeing, not one derivation agreeing with itself.
    """
    m = json.loads(row["loss_mask"] or "{}")
    out = {}

    def ok(i, layer, extra=None):
        return (row[layer] is not None and not m.get(layer) and rep[i]
                and not (extra and m.get(extra) and m[extra][i]))

    for layer, extra in (("pos", None), ("morph", None), ("lemma", None),
                         ("ner", "ner_tokens")):
        exp = {}
        for i in range(W):
            if ok(i, layer, extra):
                lab = row[layer][i]
                if layer == "lemma":
                    if vocab.id("lemma", edit_script(row["tokens"][i], lab)) is None:
                        continue
                elif vocab.id(layer, lab) is None:
                    continue
                exp[i] = lab
        out[layer] = exp
    dep = {}
    for i in range(W):
        h = row["dep_heads"][i] if row["dep_heads"] else None
        if ok(i, "dep", "dep_tokens") and (h < 0 or (h < W and rep[h])):
            dep[i] = (h, row["dep"][i])
    out["dep"] = dep
    frames = {}
    for fr in (row["srl_frames"] or []) if not m.get("srl") else []:
        p = fr["predicate_idx"]
        if p < W and rep[p]:
            s, e = next((s, e) for s, e in row["sentence_spans"] if s <= p < e)
            frames[p] = {i: fr["tags"][i] for i in range(s, min(e, W)) if rep[i]}
    out["srl"] = frames
    pred = {}
    if (row["srl_frames"] is not None and not m.get("srl")
            and row["pos"] is not None and not m.get("pos")):
        pred = {i: 0 for i in range(W) if rep[i] and row["pos"][i] != "VERB"}
    for p in frames:
        pred[p] = 1
    out["srl_pred"] = pred
    spans = row["sentence_spans"]
    out["cls"] = {} if row["cls"] is None or m.get("cls") else {
        j: sorted(row["cls"][j]) for j, (s, e) in enumerate(spans)
        if e <= W and not (m.get("cls_sentences") or [False] * len(spans))[j]}
    out["sentiment"] = {} if row["sentiment"] is None or m.get("sentiment") else {
        j: row["sentiment"][j] for j, (s, e) in enumerate(spans)
        if e <= W and row["sentiment"][j] is not None
        and not (m.get("sentiment_sentences") or [False] * len(spans))[j]}
    return out


def _decoded(batch: dict, b: int, words: list[str], vocab: Vocab) -> dict:
    L = vocab.labels
    W = len(words)
    out = {}
    for layer in TOKEN_LAYERS:
        v = batch[layer][b].tolist()
        out[layer] = {i: (apply_script(words[i], L["lemma"][x]) if layer == "lemma"
                          else L[layer][x]) for i, x in enumerate(v[:W]) if x != IGNORE}
    hd, rl = batch["dep_head"][b].tolist(), batch["dep_rel"][b].tolist()
    out["dep"] = {i: (-1 if hd[i] == 0 else hd[i] - 1, L["dep"][rl[i]])
                  for i in range(W) if hd[i] != IGNORE}
    frames = {}
    for f in range(len(batch["frame_batch"])):
        if batch["frame_batch"][f] == b:
            r = batch["frame_roles"][f].tolist()
            frames[int(batch["frame_pred"][f])] = {
                i: L["srl"][x] for i, x in enumerate(r) if x != IGNORE}
    out["srl"] = frames
    sp = batch["srl_pred"][b].tolist()
    out["srl_pred"] = {i: x for i, x in enumerate(sp[:W]) if x != IGNORE}
    ok, c = batch["cls_ok"][b].tolist(), batch["cls"][b]
    out["cls"] = {j: sorted(L["cls"][k] for k in range(c.shape[1]) if c[j, k] > 0)
                  for j in range(len(ok)) if ok[j]}
    st = batch["sentiment"][b].tolist()
    out["sentiment"] = {j: L["sentiment"][x] for j, x in enumerate(st) if x != IGNORE}
    return out


def verify(spec, split: str, vocab: Vocab, limit: int | None,
           batch_size: int = 16) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    rows = load_rows(split, limit=limit)
    stats = Counter()
    feats = [encode_row(r, tok, spec.max_subwords, vocab, stats) for r in rows]
    by_id = {r["window_id"]: r for r in rows}
    bad, checked = Counter(), Counter()
    for idx in batch_order(feats, batch_size, seed=0, epoch=0):
        items = [feats[i] for i in idx]
        batch = collate(items, tok.pad_token_id)
        for b, x in enumerate(items):
            row = by_id[x["window_id"]]
            W = len(x["first_subword"])
            words = row["tokens"][:W]
            # Alignment: each word's first subword must start inside that word.
            enc = tok(" ".join(row["tokens"]), return_offsets_mapping=True)
            starts = np.cumsum([0] + [len(w) + 1 for w in row["tokens"]])
            for i, f in enumerate(x["first_subword"]):
                if f >= 0:
                    s, e = enc["offset_mapping"][f]
                    if not (starts[i] <= e - 1 < starts[i] + len(row["tokens"][i])):
                        bad["alignment"] += 1
                    checked["alignment"] += 1
            if not (batch["input_ids"][b, :len(x["input_ids"])].tolist()
                    == x["input_ids"].tolist()):
                bad["input_ids"] += 1
            exp = _expected(row, W, list(x["word_ok"]), vocab)
            got = _decoded(batch, b, words, vocab)
            for layer, e in exp.items():
                checked[layer] += len(e)
                if got[layer] != e:
                    bad[layer] += 1
                    if bad[layer] <= 3:
                        diff = {k: (e.get(k), got[layer].get(k))
                                for k in set(e) | set(got[layer])
                                if e.get(k) != got[layer].get(k)}
                        print(f"  MISMATCH {layer} {row['window_id']}: "
                              f"{dict(list(diff.items())[:5])}")
    print(f"{spec.name} {split}: {len(rows)} windows; "
          + ", ".join(f"{k} {v:,}" for k, v in sorted(stats.items())))
    for k in sorted(checked):
        print(f"  {k:10s} {checked[k]:>10,} labels checked   "
              f"{'OK' if not bad[k] else f'{bad[k]} WINDOWS DIFFER'}")
    if bad["input_ids"]:
        print(f"  input_ids  {bad['input_ids']} windows differ after collation")
    return 1 if bad else 0


def main() -> int:
    from . import encoders
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    spec, vocab = encoders.get(args.encoder), Vocab.load()
    if args.verify:
        return verify(spec, args.split, vocab, args.limit)
    featurise(spec, args.split, vocab, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

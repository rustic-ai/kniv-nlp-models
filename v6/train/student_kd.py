"""Distillation from both v7 teachers into a student (docs/v7-students-plan.md,
step 4): the student windows as training items, each carrying the two
teachers' targets mixed per head; a collate step that lays the mixtures out
on the student's own words, frames, spans and entity pairs; and the losses.

Mixture per head: ``p = w * p_UltraBERT + (1 - w) * p_DeBERTa`` with the
weights measured in step 1 (``WEIGHTS``). A position one teacher did not read
(its encoder stopped earlier) takes the other's target alone.

Loss per head (``kd_<head>``, normalised like every head by its mean over
the first steps, weighted by ``--head-weight``):
- POS, morphology, lemma, NER, DocRED entity tags, DEP arcs, DEP labels (at
  the mixed head), SRL predicates and roles, sentiment: soft cross-entropy;
- dialogue acts: binary cross-entropy against the mixed sigmoids;
- coreference: soft cross-entropy over the teachers' antecedents that are
  among the student's candidates;
- spans and relations: binary cross-entropy on the decision
  (``sigmoid(margin)``), teacher margins mixed with ``FLOOR`` where absent.
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import IGNORE
from .student_targets import FLOOR

TEACHERS = ("ultrabert-380m-v7b8ev", "deberta-v3-large-v7b8ev")
WEIGHTS = {"pos": .5, "morph": .5, "lemma": .5, "arc": .5, "deprel": .5, "srl": .5, "rel": .5,
           "ner": .75, "ent": .75, "cls": .75, "sentiment": .75, "coref": .75, "span": .75}
KD_HEADS = ("kd_pos", "kd_morph", "kd_lemma", "kd_ner", "kd_ent", "kd_arc", "kd_deprel",
            "kd_srl_pred", "kd_srl", "kd_cls", "kd_sentiment", "kd_coref", "kd_span", "kd_rel")
SIZES = {"pos": "pos", "morph": "morph", "lemma": "lemma", "ner": "ner", "deprel": "dep"}


def load_targets(paths: list[Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in paths:
        out.update(pickle.loads(Path(p).read_bytes()))
    return out


def _mix_topk(a, b, Wa, Wb, Ws, w):
    """Two top-k tables ``(idx [W,k], prob [W,k])`` -> concatenated weighted
    entries in student word positions ``(idx [Ws,2k], prob [Ws,2k])``; a
    position past one teacher's ``W`` takes the other's alone."""
    k = a[0].shape[1]
    idx = np.zeros((Ws, 2 * k), dtype=np.int32)
    pr = np.zeros((Ws, 2 * k), dtype=np.float32)
    for i in range(Ws):
        wa = w if i < Wa else 0.0
        wb = (1 - w) if i < Wb else 0.0
        z = wa + wb
        if z == 0:
            continue
        if wa:
            idx[i, :k], pr[i, :k] = a[0][i], a[1][i].astype(np.float32) * wa / z
        if wb:
            idx[i, k:], pr[i, k:] = b[0][i], b[1][i].astype(np.float32) * wb / z
    return idx, pr


def window_features(w: dict, tA: dict, tB: dict, tok, spec, vocab, rel_id: dict, type_id: dict,
                    markers: bool = True, n_rel: int = 0) -> dict | None:
    """A training item for one student window: no hard labels, the teachers'
    mixed targets under ``kd``."""
    from .data import blank_window, marker_features, speaker_turns
    from .kdata import span_targets
    tokens, sp = w["tokens"], [tuple(x) for x in w["sentence_spans"]]
    preds = sorted({int(p) for p, *_ in tA["srl"]} | {int(p) for p, *_ in tB["srl"]})
    f = blank_window(tokens, sp, tok, spec.max_subwords, vocab.size("cls"), predicates=preds)
    Ws, Wa, Wb = f["_W"], tA["W"], tB["W"]
    kd: dict = {"W": Ws}
    for h in ("pos", "morph", "lemma", "ner", "ent"):
        kd[h] = _mix_topk(tA[h], tB[h], Wa, Wb, Ws, WEIGHTS[h])
    kd["arc"] = _mix_topk(tA["arc"], tB["arc"], Wa, Wb, Ws, WEIGHTS["arc"])
    # DEP labels at the mixed arc's best head, from the teachers whose tree
    # head it is.
    ai, ap = kd["arc"]
    head = np.full(Ws, -1, dtype=np.int64)
    lab_i = np.zeros((Ws, 10), dtype=np.int32)
    lab_p = np.zeros((Ws, 10), dtype=np.float32)
    for i in range(Ws):
        acc = defaultdict(float)
        for j, p in zip(ai[i], ap[i]):
            if p > 0 and j <= Ws:
                acc[int(j)] += float(p)
        if not acc:
            continue
        hb = max(acc, key=acc.get)
        src = [(t, wt) for t, wt, Wt in ((tA, WEIGHTS["deprel"], Wa), (tB, 1 - WEIGHTS["deprel"], Wb))
               if i < Wt and int(t["tree_head"][i]) == hb]
        if not src:
            continue
        head[i] = hb
        z = sum(wt for _, wt in src)
        for n, (t, wt) in enumerate(src):
            lab_i[i, 5 * n:5 * n + 5] = t["deprel"][0][i]
            lab_p[i, 5 * n:5 * n + 5] = t["deprel"][1][i].astype(np.float32) * wt / z
    f["dep_head"] = np.where(head >= 0, head, IGNORE).astype(np.int16)   # the student's DEP labels are read here
    kd["deprel"] = (lab_i, lab_p, head >= 0)
    pa = np.zeros(Ws, np.float32); pb = np.zeros(Ws, np.float32)
    na, nb = min(Wa, Ws), min(Wb, Ws)
    pa[:na], pb[:nb] = tA["srl_pred"][:na], tB["srl_pred"][:nb]
    wts = np.where(np.arange(Ws) < Wa, WEIGHTS["srl"], 0) + np.where(np.arange(Ws) < Wb, 1 - WEIGHTS["srl"], 0)
    kd["srl_pred"] = ((np.where(np.arange(Ws) < Wa, WEIGHTS["srl"], 0) * pa
                       + np.where(np.arange(Ws) < Wb, 1 - WEIGHTS["srl"], 0) * pb) / np.maximum(wts, 1e-9),
                      wts > 0)
    frames = []
    ra = {int(p): (s, i, q) for p, s, i, q in tA["srl"]}
    rb = {int(p): (s, i, q) for p, s, i, q in tB["srl"]}
    for p, _, _ in f["frames"]:
        ent = [(r[p], wt) for r, wt in ((ra, WEIGHTS["srl"]), (rb, 1 - WEIGHTS["srl"])) if p in r]
        z = sum(wt for _, wt in ent)
        frames.append([(s, i, q.astype(np.float32) * wt / z) for (s, i, q), wt in ent])
    kd["srl"] = frames
    S = len(sp)
    kd["cls"] = (WEIGHTS["cls"] * tA["cls"].astype(np.float32) + (1 - WEIGHTS["cls"]) * tB["cls"].astype(np.float32))[:S]
    kd["sentiment"] = (WEIGHTS["sentiment"] * tA["sentiment"].astype(np.float32)
                       + (1 - WEIGHTS["sentiment"]) * tB["sentiment"].astype(np.float32))[:S]
    co_i = np.full((Ws, 16), -2, dtype=np.int32)
    co_p = np.zeros((Ws, 16), dtype=np.float32)
    for i in range(Ws):
        acc = defaultdict(float)
        for t, wt, Wt in ((tA, WEIGHTS["coref"], Wa), (tB, 1 - WEIGHTS["coref"], Wb)):
            if i < Wt:
                for c, p in zip(t["coref"][0][i], t["coref"][1][i]):
                    if c < Ws:
                        acc[int(c)] += wt * float(p)
        z = sum(acc.values())
        for n, (c, p) in enumerate(sorted(acc.items(), key=lambda x: -x[1])[:16]):
            co_i[i, n], co_p[i, n] = c, p / z
    kd["coref"] = (co_i, co_p)
    f.update(coref=None, rel=None, _n_rel=n_rel or vocab.size("relations"))   # columns index the whole bank
    # Typed spans: the window's types by bank id.
    if w.get("types") and "span" in tA and "span" in tB:
        cols = [type_id.get(n) for n in w["types"]]
        f["ents"] = span_targets([], Ws, f["word_ok"], [c for c in cols if c is not None], True)
        f["ents"]["unlabelled_keep"] = 0.0          # no hard "not an entity" labels: the targets teach
        acc: dict = defaultdict(lambda: [FLOOR, FLOOR])
        for k_, t in enumerate((tA, tB)):
            s_, e_, j_, m_ = t["span"]
            for s, e, j, m in zip(s_, e_, j_, m_):
                if cols[j] is not None and e <= Ws:
                    acc[(int(s), int(e), cols[j])][k_] = float(m)
        kd["span"] = {key: WEIGHTS["span"] * a + (1 - WEIGHTS["span"]) * b for key, (a, b) in acc.items()}
    # Relations: the entities both teachers kept, in the runtime's order.
    if w.get("entities") and w.get("relations") and "rel" in tA and "rel" in tB:
        both = set(tA["rel_ents"].tolist()) & set(tB["rel_ents"].tolist())
        ok = f["word_ok"]
        speakers = w.get("speakers") or []
        parts = list(dict.fromkeys(s for s in speakers if s))
        turns = [(s, p) for s, p in speaker_turns(sp, speakers, parts) if s < Ws and ok[s]]
        ids, spans = [], []
        for p, _ in enumerate(parts):
            ms = [(s, s + 1) for s, q in turns if q == p]
            if ms and (-1 - p) in both:
                ids.append(-1 - p); spans.append(np.asarray(ms, dtype=np.int32))
        n_part = len(spans)
        for ei, e in enumerate(w["entities"]):
            ms = [(int(a), int(b)) for a, b in e["spans"] if b <= Ws and ok[a]]
            if ms and ei in both:
                ids.append(ei); spans.append(np.asarray(ms, dtype=np.int32))
        if len(spans) >= 2:
            cols = [rel_id.get(n) for n in w["relations"]]
            schema = [c for c in cols if c is not None]
            f["rel"] = {"spans": spans, "types": np.zeros(len(spans), dtype=np.int32), "triples": [],
                        "evidence": [], "schema": schema, "partial": True, "unlabelled_keep": 0.0,
                        "ids": ids}
            if markers:
                part_turns = [(s, p) for s, p in turns if p < n_part]
                f["rm"] = marker_features(tokens, sp, spans[n_part:], tok, turns=part_turns,
                                          participants=parts[:n_part])
            pos = {e: k for k, e in enumerate(ids)}
            acc = defaultdict(lambda: [FLOOR, FLOOR])
            for k_, t in enumerate((tA, tB)):
                for h, tl, r, m in zip(*t["rel"]):
                    if int(h) in pos and int(tl) in pos and cols[r] is not None:
                        acc[(pos[int(h)], pos[int(tl)], cols[r])][k_] = float(m)
            kd["rel"] = {key: WEIGHTS["rel"] * a + (1 - WEIGHTS["rel"]) * b for key, (a, b) in acc.items()}
            kd["rel_n"] = len(spans)
    f["kd"] = kd
    return f


def build(windows_path: Path, target_paths: dict[str, list[Path]], tok, spec, vocab, bank,
          markers: bool = True, limit: int | None = None) -> list[dict]:
    """Training items for every student window both teachers scored."""
    T = {r: load_targets(ps) for r, ps in target_paths.items()}
    A, B = (T[r] for r in TEACHERS)
    rel_id = {r["name"]: r["id"] for r in bank["relations"]}
    type_id = {t["name"]: t["id"] for t in bank["entity_types"]}
    out = []
    for line in windows_path.read_text().splitlines():
        w = json.loads(line)
        if w["id"] in A and w["id"] in B:
            f = window_features(w, A[w["id"]], B[w["id"]], tok, spec, vocab, rel_id, type_id, markers,
                                n_rel=len(bank["relations"]))
            if f is not None:
                out.append(f)
        if limit and len(out) >= limit:
            break
    return out


# ── collate: targets on the batch's own layout ──────────────────────
def kd_collate(items: list[dict], b: dict, vocab) -> dict:
    """Dense target tensors for a collated (and label-collated) batch."""
    if not any("kd" in x for x in items):
        return b
    B = len(items)
    W = b["first_subword"].size(1)
    S = b["sent_whole"].size(1)
    out = {}
    for h in ("pos", "morph", "lemma", "ner", "ent"):
        C = len(vocab.labels[SIZES.get(h, "ner")]) if h != "ent" else 13
        t = torch.zeros(B, W, C)
        for j, x in enumerate(items):
            if "kd" not in x:
                continue
            idx, pr = x["kd"][h]
            n = idx.shape[0]
            t[j, :n].scatter_add_(1, torch.from_numpy(idx.astype(np.int64)), torch.from_numpy(pr))
        out[f"kd_{h}"] = t
    t = torch.zeros(B, W, W + 1)
    for j, x in enumerate(items):
        if "kd" in x:
            idx, pr = x["kd"]["arc"]
            n = idx.shape[0]
            keep = idx <= W
            t[j, :n].scatter_add_(1, torch.from_numpy(np.where(keep, idx, 0).astype(np.int64)),
                                  torch.from_numpy(np.where(keep, pr, 0)))
    out["kd_arc"] = t
    C = len(vocab.labels["dep"])
    t, m = torch.zeros(B, W, C), torch.zeros(B, W, dtype=torch.bool)
    for j, x in enumerate(items):
        if "kd" in x:
            li, lp, ok = x["kd"]["deprel"]
            n = li.shape[0]
            t[j, :n].scatter_add_(1, torch.from_numpy(li.astype(np.int64)), torch.from_numpy(lp))
            m[j, :n] = torch.from_numpy(ok)
    out["kd_deprel"], out["kd_deprel_mask"] = t, m
    t, m = torch.zeros(B, W), torch.zeros(B, W, dtype=torch.bool)
    for j, x in enumerate(items):
        if "kd" in x:
            p, ok = x["kd"]["srl_pred"]
            t[j, :len(p)], m[j, :len(p)] = torch.from_numpy(p), torch.from_numpy(ok)
    out["kd_srl_pred"], out["kd_srl_pred_mask"] = t, m
    C = len(vocab.labels["srl"])
    rows = []
    for x in items:
        for fr in (x["kd"]["srl"] if "kd" in x else [None] * len(x["frames"])):
            r = torch.zeros(W, C)
            for s, idx, pr in fr or []:
                n = idx.shape[0]
                r[s:s + n].scatter_add_(1, torch.from_numpy(idx.astype(np.int64)), torch.from_numpy(pr))
            rows.append(r)
    out["kd_srl"] = torch.stack(rows) if rows else torch.zeros(0, W, C)
    C = len(vocab.labels["cls"])
    t, m = torch.zeros(B, S, C), torch.zeros(B, S, dtype=torch.bool)
    t2 = torch.zeros(B, S, len(vocab.labels["sentiment"]))
    for j, x in enumerate(items):
        if "kd" in x:
            c = x["kd"]["cls"]
            t[j, :len(c)] = torch.from_numpy(c)
            t2[j, :len(c)] = torch.from_numpy(x["kd"]["sentiment"])
            m[j, :len(c)] = torch.from_numpy(x["sent_whole"][:len(c)])
    out["kd_cls"], out["kd_sentiment"], out["kd_sent_mask"] = t, t2, m
    ci, cp = torch.full((B, W, 16), -2, dtype=torch.long), torch.zeros(B, W, 16)
    for j, x in enumerate(items):
        if "kd" in x:
            i_, p_ = x["kd"]["coref"]
            ci[j, :len(i_)], cp[j, :len(p_)] = torch.from_numpy(i_.astype(np.int64)), torch.from_numpy(p_)
    out["kd_coref_cand"], out["kd_coref_p"] = ci, cp
    if "span_batch" in b and "type_label_cols" in b:
        col = {int(c): k for k, c in enumerate(b["type_label_cols"].tolist())}
        t = torch.full((len(b["span_batch"]), len(col)), float("nan"))
        where = {(int(bb), int(s), int(e)): n for n, (bb, s, e) in
                 enumerate(zip(b["span_batch"].tolist(), b["span_start"].tolist(), b["span_end"].tolist()))}
        for j, x in enumerate(items):
            sp = x.get("kd", {}).get("span")
            if not sp:
                continue
            types = [c for c in col]
            for (s, e, c), mg in sp.items():
                n = where.get((j, s, e))
                if n is not None and c in col:
                    t[n, col[c]] = mg
            # every candidate of this window, for its own types, not above the floor: the floor
            rows_j = [n for (bb, _, _), n in where.items() if bb == j]
            own = [col[c] for c in set(c for (_, _, c) in sp) if c in col]
            if rows_j and own:
                sub = t[rows_j][:, own]
                t[torch.tensor(rows_j).unsqueeze(1), torch.tensor(own)] = torch.where(
                    torch.isnan(sub), torch.full_like(sub, FLOOR), sub)
            del types
        out["kd_span"] = t
    if "pair_batch" in b and "rel_label_cols" in b and len(b["pair_batch"]):
        col = {int(c): k for k, c in enumerate(b["rel_label_cols"].tolist())}
        t = torch.full((len(b["pair_batch"]), len(col)), float("nan"))
        base, pair_base = 0, 0
        pair_of = defaultdict(dict)
        for n, (bb, h, tl) in enumerate(zip(b["pair_batch"].tolist(), b["pair_h"].tolist(),
                                            b["pair_t"].tolist())):
            pair_of[bb][(h, tl)] = n
        ent_base = 0
        for j, x in enumerate(items):
            r = x.get("rel")
            if r is None:
                continue
            kdr = x.get("kd", {}).get("rel")
            if kdr is not None:
                own = [col[c] for c in r["schema"] if c in col]
                rows = [pair_of[j][(ent_base + h, ent_base + tl)] for h in range(len(r["spans"]))
                        for tl in range(len(r["spans"])) if h != tl]
                if rows and own:
                    t[torch.tensor(rows).unsqueeze(1), torch.tensor(own)] = FLOOR
                for (h, tl, c), mg in kdr.items():
                    if c in col:
                        t[pair_of[j][(ent_base + h, ent_base + tl)], col[c]] = mg
            ent_base += len(r["spans"])
        out["kd_rel"] = t
    b.update(out)
    return b


# ── losses ──────────────────────────────────────────────────────────
def _soft_ce(logits, target, mask):
    """Mean over masked positions of ``-sum target * log_softmax(logits)``."""
    z = target.sum(-1)
    m = mask & (z > 0)
    if not bool(m.any()):
        return logits.sum() * 0.0, 0
    lp = F.log_softmax(logits.float(), -1)
    tgt = target / z.clamp(min=1e-9).unsqueeze(-1)
    return -(tgt * lp).sum(-1)[m].mean(), int(m.sum())


def _decision(margins, teacher):
    ok = torch.isfinite(teacher)
    if not bool(ok.any()):
        return margins.sum() * 0.0, 0
    return F.binary_cross_entropy_with_logits(margins[ok].float(), torch.sigmoid(teacher[ok].float())), int(ok.sum())


def student_losses(o: dict, b: dict) -> dict:
    out = {}
    if "kd_pos" not in b:
        return {h: (o["pos"].sum() * 0.0, 0) for h in KD_HEADS}
    wm = b["word_mask"]
    for h in ("pos", "morph", "lemma", "ner"):
        out[f"kd_{h}"] = _soft_ce(o[h], b[f"kd_{h}"], wm)
    out["kd_ent"] = (_soft_ce(o["ent"], b["kd_ent"], wm) if "ent" in o and o["ent"] is not None
                     else (o["pos"].sum() * 0.0, 0))
    out["kd_arc"] = _soft_ce(o["arc"], b["kd_arc"], wm)
    out["kd_deprel"] = _soft_ce(o["rel"], b["kd_deprel"], wm & b["kd_deprel_mask"])
    sp = torch.stack([1 - b["kd_srl_pred"], b["kd_srl_pred"]], -1)
    out["kd_srl_pred"] = _soft_ce(o["srl_pred"], sp, wm & b["kd_srl_pred_mask"])
    if len(o["srl_role"]) and len(b["kd_srl"]) == len(o["srl_role"]):
        out["kd_srl"] = _soft_ce(o["srl_role"], b["kd_srl"], b["kd_srl"].sum(-1) > 0)
    else:
        out["kd_srl"] = (o["pos"].sum() * 0.0, 0)
    m = b["kd_sent_mask"]
    if bool(m.any()):
        out["kd_cls"] = (F.binary_cross_entropy_with_logits(o["cls"][m].float(), b["kd_cls"][m]), int(m.sum()))
    else:
        out["kd_cls"] = (o["pos"].sum() * 0.0, 0)
    out["kd_sentiment"] = _soft_ce(o["sentiment"], b["kd_sentiment"], m)
    if "coref_scores" in o:
        lp = F.log_softmax(o["coref_scores"].float(), -1)                     # [B,W,k+1]
        cand = torch.cat([torch.full_like(o["coref_cand"][..., :1], -1), o["coref_cand"]], -1)
        hit = (b["kd_coref_cand"].unsqueeze(-1) == cand.unsqueeze(-2))        # [B,W,16,k+1]
        hit = hit & (b["kd_coref_cand"] >= -1).unsqueeze(-1)
        lp_t = (lp.unsqueeze(-2) * hit).sum(-1)                               # log-prob of each target
        got = hit.any(-1)
        p = b["kd_coref_p"] * got
        z = p.sum(-1)
        msk = wm & (z > 0)
        out["kd_coref"] = ((-(p * lp_t).sum(-1) / z.clamp(min=1e-9))[msk].mean(), int(msk.sum())) \
            if bool(msk.any()) else (o["pos"].sum() * 0.0, 0)
    else:
        out["kd_coref"] = (o["pos"].sum() * 0.0, 0)
    if "span" in o and "kd_span" in b and len(o["span"]):
        out["kd_span"] = _decision(o["span"][:, 1:] - o["span"][:, :1], b["kd_span"])
    else:
        out["kd_span"] = (o["pos"].sum() * 0.0, 0)
    if "relation" in o and "kd_rel" in b and len(o["relation"]):
        out["kd_rel"] = _decision(o["relation"][:, 1:] - o["relation"][:, :1], b["kd_rel"])
    else:
        out["kd_rel"] = (o["pos"].sum() * 0.0, 0)
    return out

"""Train one encoder on the v6 corpus.

    uv run python -m v6.train.loop --encoder ettin-1b --hf-repo dragonscale-ai/kniv-v6-runs
    uv run python -m v6.train.loop --encoder ettin-1b --probe 20     # A100 fit check

Every head trains jointly on every window (TRAINING.md §1, §6). Per-head
losses start uniform; after ``--calib-steps`` each is divided by its own
early mean so a head with more classes or more tokens does not dominate.
Per-head losses are logged every interval: a head flat from step zero is
invisible in the total.

Dev evaluation scores agreement with the corpus annotators and is used only
to select ``best.pt``. Headline numbers come from public gold (TRAINING.md §8).

Operations follow TRAINING.md §7 and OPERATIONS.md: run in the kernel, never
detached; resume reads ``latest.pt`` (pulled from the hub if this VM is new)
and replays the same sample order; SIGTERM checkpoints and exits.

**Control files** in the output directory are checked every step, so a run
can be steered without a kernel interrupt (``colab.py stop`` / ``checkpoint``
upload them):

``STOP``        checkpoint and exit cleanly; its text is recorded as the reason
``CHECKPOINT``  write ``latest.pt`` now and carry on

**Non-finite steps are skipped, never applied.** A NaN or inf loss or
gradient norm skips the update (the data is still consumed, so the sample
order is unchanged) and is recorded. ``--max-nonfinite`` in a row stops the
run *without* writing ``latest.pt``, so the last good resume point survives.
Resuming replays the same data, so a run stopped this way needs a change --
usually a lower learning rate, under a new run name -- not a plain relaunch.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from ..config import RUNS_DIR
from . import encoders
from .data import IGNORE, batch_order, collate, featurise
from .model import HEAD_WEIGHT, HEADS, PHASE2_HEADS, RELATION_HEADS, V6Model, losses
from .runstate import Hub, RunState, memory
from .vocab import Vocab


# ── evaluation ──────────────────────────────────────────────────────
def _spans(tags: list[str]) -> set[tuple]:
    out, cur = set(), None
    for i, t in enumerate(tags + ["O"]):
        if cur and not (t.startswith("I-") and t[2:] == cur[0]):
            out.add((cur[0], cur[1], i))
            cur = None
        if t.startswith("B-") or (t.startswith("I-") and cur is None):
            cur = (t[2:], i)
    return out


@torch.no_grad()
def relation_siblings(model, tok, device, bank, k: int) -> dict:
    """Each trainable relation's ``k`` nearest relations by the model's own
    label embeddings (cosine): the confusable ones (member of / member of
    sports team, country / country of origin)."""
    from .gold_eval import schema_embeddings
    ids = [r["id"] for r in bank["relations"] if r.get("train", True)]
    was = model.training
    model.eval()
    emb = torch.cat([schema_embeddings(model, tok, device, ids[i:i + 64]).float()
                     for i in range(0, len(ids), 64)])
    model.train(was)
    emb = torch.nn.functional.normalize(emb, dim=-1)
    sim = emb @ emb.T
    sim.fill_diagonal_(-1)
    top = sim.topk(k, dim=-1).indices.cpu().numpy()
    arr = np.asarray(ids, dtype=np.int64)
    return {int(arr[i]): arr[top[i]] for i in range(len(ids))}


GOLD_DEV_KEYS = ("pos", "morph", "lemma", "ner_f1", "las", "srl_pred_f1", "srl_role_f1")


def _f1(tp: float, n_pred: float, n_gold: float) -> float:
    return 2 * tp / max(n_pred + n_gold, 1)


@torch.no_grad()
def evaluate(model, feats, vocab: Vocab, pad_id: int, device, amp,
             batch_size: int, rel_gold: list | None = None,
             label_coll=None, label_coll_zs=None, span_gold: list | None = None,
             extra_rel: dict | None = None, beat=None, knowledge_only: bool = False,
             gold_dev: list | None = None) -> dict:
    """Dev scores and their composite. ``rel_gold``: human-annotated relation
    documents (Re-DocRED dev), scored as ``relation_gold_f1`` and counted in
    the composite. ``label_coll``: the label-conditioned head's batch labels;
    ``label_coll_zs``: the held-out relations only, scored on ``rel_gold`` as
    ``relation_gold_zs_f1`` (reported, never in the composite).

    With the hybrid, Re-DocRED through the frozen path is a constant, so the
    label path is scored on its own too (``relation_gold_label_f1``), and
    ``span_gold`` (Few-NERD dev) scores the span head (``span_gold_f1``):
    both join the composite, which then moves only with what trains.

    ``gold_dev``: human-annotated linguistic dev items (``golddata.gold_dev``);
    their POS, morphology, lemma, LAS, NER and SRL scores (``gold_*``) replace
    the corpus-dev ones in the composite, so best.pt follows gold, not the
    corpus's own annotators."""
    on_gold = on_zs = on_span = on_lang = None
    if gold_dev:
        on_lang = evaluate(model, gold_dev, vocab, pad_id, device, amp, batch_size, beat=beat)
    sticky = model.rel_lora is not None and model.rel_lora.sticky
    if sticky:                       # the plain pass must be plain while evaluating
        model.rel_lora.sticky = model.rel_lora.on = False
    if rel_gold:
        on_gold = evaluate(model, rel_gold, vocab, pad_id, device, amp, max(1, batch_size // 4),
                           label_coll=label_coll, beat=beat)
        if label_coll_zs is not None:
            on_zs = evaluate(model, rel_gold, vocab, pad_id, device, amp, max(1, batch_size // 4),
                             label_coll=label_coll_zs, beat=beat)
    extra = {}
    for name, (f_, c_) in (extra_rel or {}).items():
        if f_:
            e_ = evaluate(model, f_, vocab, pad_id, device, amp, max(1, batch_size // 4),
                          label_coll=c_, beat=beat)
            extra[name] = e_.get("relation_rank_acc", 0.0)
            # Fact attributes are scored on the teacher dev windows (their
            # held-out relations' facts).
            extra.update({k: v for k, v in e_.items() if k.startswith("attr_")})
            extra[name.replace("_acc", "_thr_f1")] = e_.get("relation_f1", 0.0)
    if span_gold:
        on_span = evaluate(model, span_gold, vocab, pad_id, device, amp, batch_size,
                           label_coll=label_coll, beat=beat)
    model.eval()
    L = vocab.labels
    c = defaultdict(float)
    order = sorted(range(len(feats)), key=lambda i: len(feats[i]["input_ids"]))
    for k in range(0, len(order), batch_size):
        if beat is not None and (k // batch_size) % 20 == 0:
            beat()                       # a long evaluation must not read as a dead run
        b = collate([feats[i] for i in order[k:k + batch_size]], pad_id)
        b = _to(label_coll(b) if label_coll is not None else b, device)
        with amp():
            o = model(b, predict_heads=True)
        for h in ("pos", "morph", "lemma"):
            t, m = b[h], b[h] != IGNORE
            c[f"{h}_ok"] += (o[h].argmax(-1)[m] == t[m]).sum().item()
            c[f"{h}_n"] += m.sum().item()
        t, m = b["dep_head"], b["dep_head"] != IGNORE
        ua = (o["rel_head"] == t) & m
        c["uas_ok"] += ua.sum().item()
        c["las_ok"] += (ua & (o["rel"].argmax(-1) == b["dep_rel"])).sum().item()
        c["dep_n"] += m.sum().item()
        t, m = b["srl_pred"], b["srl_pred"] != IGNORE
        p = o["srl_pred"].argmax(-1)
        c["pred_tp"] += ((p == 1) & (t == 1) & m).sum().item()
        c["pred_p"] += ((p == 1) & m).sum().item()
        c["pred_g"] += ((t == 1) & m).sum().item()
        ner_p, ner_g = o["ner"].argmax(-1).tolist(), b["ner"].tolist()
        for pp, gg in zip(ner_p, ner_g):
            g = [L["ner"][x] if x != IGNORE else "O" for x in gg]
            q = [L["ner"][x] if y != IGNORE else "O" for x, y in zip(pp, gg)]
            gs, ps = _spans(g), _spans(q)
            c["ner_tp"] += len(gs & ps); c["ner_p"] += len(ps); c["ner_g"] += len(gs)
        if len(b["frame_batch"]):
            rp, rg = o["srl_role"].argmax(-1).tolist(), b["frame_roles"].tolist()
            for pp, gg in zip(rp, rg):
                g = [L["srl"][x] if x != IGNORE else "O" for x in gg]
                q = [L["srl"][x] if y != IGNORE else "O" for x, y in zip(pp, gg)]
                gs = {s for s in _spans(g) if s[0] != "V"}
                ps = {s for s in _spans(q) if s[0] != "V"}
                c["role_tp"] += len(gs & ps); c["role_p"] += len(ps); c["role_g"] += len(gs)
        ok = b["cls_ok"]
        pc, gc = (o["cls"][ok] > 0), b["cls"][ok] > 0.5
        c["cls_tp"] += (pc & gc).sum().item()
        c["cls_p"] += pc.sum().item(); c["cls_g"] += gc.sum().item()
        t, m = b["sentiment"], b["sentiment"] != IGNORE
        c["sent_ok"] += (o["sentiment"].argmax(-1)[m] == t[m]).sum().item()
        c["sent_n"] += m.sum().item()
        if "coref_scores" in o:
            _coref_counts(o, b, c)
        if "ent" in o and "ent" in b:
            from .data import ENT_TAGS
            for pp, gg in zip(o["ent"].argmax(-1).tolist(), b["ent"].tolist()):
                g = [ENT_TAGS[x] if x != IGNORE else "O" for x in gg]
                q = [ENT_TAGS[x] if y != IGNORE else "O" for x, y in zip(pp, gg)]
                gs, ps = _spans(g), _spans(q)
                c["ent_tp"] += len(gs & ps); c["ent_p"] += len(ps); c["ent_g"] += len(gs)
        if "relation" in o and len(o["relation"]):
            lg, gold = o["relation"], b["pair_labels"][:, 1:] > 0.5
            pred = lg[:, 1:] > lg[:, :1]
            c["rel_tp"] += (pred & gold).sum().item()
            c["rel_p"] += pred.sum().item(); c["rel_g"] += gold.sum().item()
            if "relation_label" in o:
                lg = o["relation_label"]
                pred = lg[:, 1:] > lg[:, :1]
                c["rlab_tp"] += (pred & gold).sum().item()
                c["rlab_p"] += pred.sum().item(); c["rlab_g"] += gold.sum().item()
        if "relation" in o and len(o["relation"]):
            # Ranking: for each pair holding a listed relation, is a true one
            # the best-scoring column? Threshold-free, as FewRel is scored --
            # unseen labels rarely clear the threshold before calibration, so
            # thresholded F1 sits at zero and cannot select a checkpoint.
            lg, gold = o["relation"], b["pair_labels"][:, 1:] > 0.5
            pos = gold.any(1)
            if bool(pos.any()) and lg.size(1) > 2:
                cand = lg[pos, 1:].float()
                if "pair_label_mask" in b:       # only the pair's own schema competes
                    cand = cand.masked_fill(b["pair_label_mask"][pos, 1:] == 0, float("-inf"))
                top = cand.argmax(1)
                c["rank_ok"] += gold[pos].gather(1, top.unsqueeze(1)).sum().item()
                c["rank_n"] += int(pos.sum())
        if "fact_attr" in o:
            fa, ok = o["fact_attr"], o["fact_attr_ok"]
            pp, gp = fa["polarity"].argmax(-1), b["attr_polarity"][ok]
            c["neg_tp"] += int(((pp == 1) & (gp == 1)).sum())
            c["neg_p"] += int((pp == 1).sum()); c["neg_g"] += int((gp == 1).sum())
            c["mod_ok"] += int((fa["modality"].argmax(-1) == b["attr_modality"][ok]).sum())
            c["hold_ok"] += int((fa["holder"].argmax(-1) - 1 == b["attr_holder"][ok]).sum())
            c["time_ok"] += int((fa["time"].argmax(-1) - 1 == b["attr_time"][ok]).sum())
            c["attr_n"] += int(ok.sum())
        if "span" in o and len(o["span"]):
            from .kdata import decode_spans
            pred = {(b_, s_, e_, t_) for b_, items in enumerate(decode_spans(o["span"].float(), b))
                    for s_, e_, t_ in items}
            g = b["span_labels"][:, 1:] > 0.5
            rows, cols_ = g.nonzero(as_tuple=True)
            tc = b["type_label_cols"]
            gold = {(int(b["span_batch"][i]), int(b["span_start"][i]), int(b["span_end"][i]),
                     int(tc[j])) for i, j in zip(rows.tolist(), cols_.tolist())}
            c["span_tp"] += len(pred & gold); c["span_p"] += len(pred); c["span_g"] += len(gold)
    model.train()
    if sticky:
        model.rel_lora.sticky = True
    r = {h: c[f"{h}_ok"] / max(c[f"{h}_n"], 1) for h in ("pos", "morph", "lemma")}
    r.update(uas=c["uas_ok"] / max(c["dep_n"], 1), las=c["las_ok"] / max(c["dep_n"], 1),
             ner_f1=_f1(c["ner_tp"], c["ner_p"], c["ner_g"]),
             srl_pred_f1=_f1(c["pred_tp"], c["pred_p"], c["pred_g"]),
             srl_role_f1=_f1(c["role_tp"], c["role_p"], c["role_g"]),
             cls_f1=_f1(c["cls_tp"], c["cls_p"], c["cls_g"]),
             sentiment=c["sent_ok"] / max(c["sent_n"], 1))
    keys = ["pos", "morph", "lemma", "ner_f1", "las", "srl_pred_f1", "srl_role_f1",
            "cls_f1", "sentiment"]
    if on_lang is not None:
        for k in GOLD_DEV_KEYS:
            r[f"gold_{k}"] = on_lang[k]
        keys = [f"gold_{k}" if k in GOLD_DEV_KEYS else k for k in keys]
    if c["link_g"] or c["rel_g"]:
        r.update(coref_link_f1=_f1(c["link_tp_p"], c["link_p"], c["link_g"])
                 if c["link_g"] else 0.0,
                 relation_f1=_f1(c["rel_tp"], c["rel_p"], c["rel_g"]))
        keys += ["coref_link_f1", "relation_f1"]
    if c["ent_g"]:
        r["ent_f1"] = _f1(c["ent_tp"], c["ent_p"], c["ent_g"])
    if c["rlab_g"]:
        r["relation_label_f1"] = _f1(c["rlab_tp"], c["rlab_p"], c["rlab_g"])
    if c["span_g"]:
        r["span_f1"] = _f1(c["span_tp"], c["span_p"], c["span_g"])
    if c["rank_n"]:
        r["relation_rank_acc"] = c["rank_ok"] / c["rank_n"]
    if c["attr_n"]:
        r["attr_negation_f1"] = 2 * c["neg_tp"] / max(c["neg_p"] + c["neg_g"], 1)
        for k in ("mod", "hold", "time"):
            r[f"attr_{k}_acc"] = c[f"{k}_ok"] / c["attr_n"]
    if on_zs is not None:
        r["relation_gold_zs_f1"] = on_zs["relation_f1"]
    if on_gold is not None:
        r["relation_gold_f1"] = on_gold["relation_f1"]
        keys.append("relation_gold_f1")
        if "ent_f1" in on_gold:
            r["ent_gold_f1"] = on_gold["ent_f1"]
            keys.append("ent_gold_f1")
        if "relation_label_f1" in on_gold:
            r["relation_gold_label_f1"] = on_gold["relation_label_f1"]
            keys.append("relation_gold_label_f1")
    if on_span is not None and "span_f1" in on_span:
        r["span_gold_f1"] = on_span["span_f1"]
        keys.append("span_gold_f1")
    for name, v in extra.items():
        r[name] = round(v, 5)
    if "relation_zs_acc" in extra:
        # Zero-shot relations select best.pt; the label path on Re-DocRED's
        # own relations (answered by the frozen path) is reported only.
        keys = [k for k in keys if k != "relation_gold_label_f1"] + ["relation_zs_acc"]
    if knowledge_only:
        # A run that trains only the knowledge layer (--train-scope
        # knowledge-full): the linguistic heads are untrained and drift at
        # random, so they would decide best.pt. Only what trains counts.
        keys = [k for k in ("relation_gold_f1", "span_gold_f1", "relation_zs_acc",
                            "fewrel_zs_acc") if k in r]
    r = {k: round(v, 5) for k, v in r.items()}
    # One number to select best.pt on: every head counts equally.
    r["composite"] = round(sum(r[k] for k in keys) / len(keys), 5)
    return r


def _coref_counts(o: dict, b: dict, c: dict) -> None:
    """Antecedent-link F1 on gold mentions: a predicted link (word -> earlier
    word) is right when both are gold mentions of one cluster; recall is over
    gold mentions that have an earlier mention in their cluster."""
    s, cand = o["coref_scores"], o["coref_cand"]
    B, W, _ = cand.shape
    cl = torch.full((B, W), -1, dtype=torch.long, device=cand.device)
    cl[b["coref_batch"], b["coref_head"]] = b["coref_cluster"]
    ok = b["coref_ok"].unsqueeze(1) & b["word_mask"]
    best = s.argmax(-1)                                                # 0 = dummy
    ante = torch.gather(cand, 2, (best - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1)
    linked = (best > 0) & ok
    right = linked & (cl >= 0) & (torch.gather(cl, 1, ante) == cl)
    # gold mentions with an earlier mention of their cluster
    has_prev = torch.zeros(B, W, dtype=torch.bool, device=cand.device)
    order = torch.argsort(b["coref_head"] + b["coref_batch"] * (W + 1))
    seen = set()
    for i in order.tolist():
        key = int(b["coref_cluster"][i])
        if key in seen:
            has_prev[b["coref_batch"][i], b["coref_head"][i]] = True
        seen.add(key)
    c["link_tp_p"] += right.sum().item()
    c["link_p"] += linked.sum().item()
    c["link_g"] += (has_prev & ok).sum().item()


def _rank_of(state: dict) -> int:
    return next(v.size(0) for k, v in state.items() if k.startswith("rel_lora.A."))


def _load_init(model, init: str, vocab: Vocab, relation: str = "drop",
               partial: bool = False) -> dict:
    """Weights from a checkpoint; heads it lacks start fresh. A vocabulary
    that disagrees on any shared inventory refuses to load.

    ``relation``, for a marker-pass model: ``drop`` starts the relation head
    fresh (from a shared-pass checkpoint, whose head means nothing here);
    ``keep`` loads it as is; ``transplant`` loads what the label-conditioned
    head shares with an ATLOP head (its extractors) and leaves the rest to
    ``LabelRelationHead.transplant``. Returns the checkpoint's state dict."""
    path = Path(init)
    if not path.exists():
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download("dragonscale-ai/kniv-v6-runs", f"{init}/best.pt"))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k, labels in ck["vocab"]["labels"].items():
        if vocab.labels.get(k) != labels:
            raise SystemExit(f"--init vocabulary differs from this one on {k!r}")
    state = ck["model"]
    if getattr(model, "rel_markers", False):
        if relation == "drop":
            # ATLOP's head shares no meaning with the shared-pass head's weights
            # (only the bilinear's name and shape coincide): it starts fresh.
            state = {k: v for k, v in state.items() if not k.startswith("relation.")}
        elif relation == "transplant":
            state = {k: v for k, v in state.items() if not k.startswith("relation.bilinear")}
        elif relation == "extractors":
            # A new head over the same pair pooling: only ATLOP's extractors carry over.
            state = {k: v for k, v in state.items() if not k.startswith("relation.")
                     or k.startswith(("relation.head_extractor", "relation.tail_extractor"))}
    lora = getattr(model, "rel_lora", None)
    if lora is not None:
        key = next((k for k in state if k.startswith("rel_lora.A.")), None)
        if key is not None and state[key].size(0) != lora.A[key[len("rel_lora.A."):]].size(0):
            n = lora.load_grown(state, "rel_lora.")
            state = {k: v for k, v in state.items() if not k.startswith("rel_lora.")}
            print(f"relation adapter grown from rank {_rank_of(ck['model'])} "
                  f"to {next(iter(lora.A.values())).size(0)} ({n} tensors, output unchanged)",
                  flush=True)
    if partial:
        own = set(model.state_dict())
        dropped = [k for k in state if k not in own]
        state = {k: v for k, v in state.items() if k in own}
        if dropped:
            print(f"--init-partial: {len(dropped)} tensors of heads this model lacks dropped", flush=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # What --init actually supplied: the optimiser gives only the rest the
    # new-head learning rate.
    model._init_loaded = set(state) - set(unexpected)
    fresh = [m for m in missing if not m.startswith(("coref.", "relation.",
                                                      "mix.coref.", "mix.relation.", "ent.",
                                                      "mix.ent.", "rel_lora.", "label_enc.",
                                                      "legacy_", "span.", "mix.ent_span.",
                                                      "label_lora.", "attr."))]
    if fresh or unexpected:
        raise SystemExit(f"--init does not fit this model: missing {fresh[:5]}, "
                         f"unexpected {unexpected[:5]}")
    print(f"initialised from {init}: {len(ck['model'])} tensors loaded, "
          f"{len(missing)} phase-2 tensors fresh", flush=True)
    return ck["model"]


def _load_legacy(model, run: str) -> None:
    """The hybrid's frozen ATLOP path from ``run``: its relation adapter and
    head become ``legacy_lora`` / ``legacy_relation``, and its entity-mention
    head replaces the model's, so Re-DocRED end to end is that run's
    exactly."""
    path = Path(run)
    if not path.exists():
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download("dragonscale-ai/kniv-v6-runs", f"{run}/best.pt"))
    st = torch.load(path, map_location="cpu", weights_only=False)["model"]
    mapped = {}
    for k, v in st.items():
        if k.startswith("rel_lora."):
            mapped["legacy_lora." + k[len("rel_lora."):]] = v
        elif k.startswith("relation."):
            mapped["legacy_relation." + k[len("relation."):]] = v
        elif k.startswith(("ent.", "mix.ent.")):
            mapped[k] = v
    want = {k for k in model.state_dict() if k.startswith(("legacy_lora.", "legacy_relation.",
                                                           "ent.", "mix.ent."))}
    if set(mapped) != want:
        raise SystemExit(f"--legacy-relation {run} does not fit: missing "
                         f"{sorted(want - set(mapped))[:5]}, extra {sorted(set(mapped) - want)[:5]}")
    model.load_state_dict(mapped, strict=False)
    print(f"legacy relation path and entity head from {run}: {len(mapped)} tensors", flush=True)


class Chain:
    """Batch transforms applied in order (type targets, then relation labels)."""

    def __init__(self, *fns):
        self.fns = [f for f in fns if f is not None]

    def __call__(self, batch):
        for f in self.fns:
            batch = f(batch)
        return batch


@torch.no_grad()
def _transplant(model, init_state: dict, coll, device) -> None:
    """Make the label-conditioned head reproduce the checkpoint's ATLOP head:
    embed the bank's relations (in the bilinear's column order, by name and
    description) with the starting model, then solve for the projections."""
    from .model import LabelRelationHead
    if not isinstance(model.relation, LabelRelationHead):
        raise SystemExit("transplant needs the label-conditioned relation head")
    bil = torch.nn.Linear(init_state["relation.bilinear.weight"].size(1),
                          init_state["relation.bilinear.weight"].size(0))
    bil.weight.copy_(init_state["relation.bilinear.weight"])
    bil.bias.copy_(init_state["relation.bilinear.bias"])
    cols = list(range(bil.out_features - 1))
    texts = [coll.rels[c] for c in cols]
    from .kdata import label_text
    tok = coll.tok
    enc = tok([label_text(r) for r in texts], padding=True, truncation=True,
              max_length=48, return_tensors="pt").to(device)
    was = model.training
    model.eval()
    emb = model.label_embeddings(enc["input_ids"], enc["attention_mask"])
    model.train(was)
    resid = model.relation.transplant(bil.to(device), emb.float())
    print(f"transplanted the ATLOP head into the label head: {len(cols)} labels, "
          f"least-squares residual {resid:.2e}", flush=True)


def _frozen_digest(model) -> str:
    """sha1 over every parameter not being trained, in name order."""
    import hashlib
    h = hashlib.sha1()
    for n, p in sorted(model.named_parameters(), key=lambda x: x[0]):
        if not p.requires_grad:
            h.update(n.encode())
            h.update(p.detach().to("cpu").contiguous().numpy().tobytes())
    return h.hexdigest()


def _to(batch: dict, device) -> dict:
    # non_blocking only on CUDA: on MPS a non-blocking copy from unpinned
    # memory races with the source being freed and delivers garbage indices.
    nb = device.type == "cuda"
    # Python-side leftovers (span targets with no span head) stay behind.
    return {k: v.to(device, non_blocking=nb) for k, v in batch.items() if hasattr(v, "to")}


# ── training ────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--encoder", required=True, choices=list(encoders.ENCODERS))
    ap.add_argument("--run-name", help="default: the encoder name")
    ap.add_argument("--out", type=Path, help="default: v6/runs/train/<run-name>")
    ap.add_argument("--hf-repo", help="private HF repo to mirror checkpoints to")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, help="stop after this many steps")
    ap.add_argument("--encoder-lr", type=float)
    ap.add_argument("--head-lr", type=float)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--grad-accum", type=int)
    ap.add_argument("--phase2", action="store_true",
                    help="add the coreference and relation heads (TRAINING.md §4)")
    ap.add_argument("--init", help="start from a checkpoint: a best.pt path, or a run "
                                   "name in the hub repo (e.g. deberta-v3-large-srlft-full)")
    ap.add_argument("--rel-head", choices=["bilinear", "label", "zs"], default="bilinear",
                    help="label: relations are label texts read by the encoder (schema-"
                         "conditioned); bilinear: ATLOP's fixed 96-way projection")
    ap.add_argument("--init-relation", choices=["drop", "keep", "transplant", "extractors"],
                    help="with --init and --rel-markers: drop the checkpoint's relation "
                         "head (default; from a shared-pass checkpoint), keep it, or "
                         "transplant its ATLOP bilinear into the label head exactly")
    ap.add_argument("--label-pass", choices=["plain", "adapter"], default="plain",
                    help="read label texts with the plain frozen encoder (only the label "
                         "projection learns) or through the relation adapter")
    ap.add_argument("--label-text", choices=["mixed", "canonical"], default="mixed",
                    help="training label texts: mixed (name / name+description / alias) "
                         "or canonical (name+description only, as at evaluation)")
    ap.add_argument("--kd-weight", type=float, default=0.0,
                    help="with --init-relation transplant: distil the new head's margins "
                         "toward the transplanted ATLOP head's, at this weight (relative "
                         "to the calibrated relation loss) decaying linearly to 0 over the "
                         "first half of training")
    ap.add_argument("--label-holdout", action="store_true",
                    help="train without the label bank's held-out relations and report "
                         "zero-shot F1 on them (relation_gold_zs_f1); diagnostic runs")
    ap.add_argument("--assert-frozen", action="store_true",
                    help="at every evaluation, check every untrained tensor is still "
                         "bit-identical to its value at the start; abort if not")
    ap.add_argument("--rel-markers", action="store_true",
                    help="ATLOP's relation head over a marker pass (entity mentions "
                         "wrapped in *); the entity head moves to the plain pass")
    ap.add_argument("--rel-adapter", type=int, default=0, metavar="R",
                    help="rank of a LoRA adapter for a second, relation-only encoder pass "
                         "feeding the relation and entity heads (0 = none)")
    ap.add_argument("--train-scope", choices=["all", "phase2", "relation", "relation-full",
                                              "knowledge-full"],
                    default="all",
                    help="phase2: train only the phase-2 heads and their layer mixes; "
                         "everything else stays exactly as --init left it")
    ap.add_argument("--relation-data", choices=["all", "redocred"], default="all",
                    help="redocred: train on Re-DocRED documents alone, without the "
                         "corpus windows and their teacher relation labels "
                         "(with --train-scope relation, where nothing else trains)")
    ap.add_argument("--redocred-train", type=int, default=0, metavar="N",
                    help="add Re-DocRED train, N times per epoch, as relation-only "
                         "supervision; its dev then joins best.pt selection (0 = off)")
    ap.add_argument("--kg-data", default="", metavar="SRC=N,...",
                    help="knowledge sources, each N times per epoch: teacher, fewrel, fewnerd, "
                         "docred_distant (needs --rel-head label; python -m v6.train.kdata --prep)")
    ap.add_argument("--gold-data", default="", metavar="SRC=N,...",
                    help="human-annotated linguistic data, each N times per epoch (N < 1: that "
                         "fraction): ud (UD EWT + PropBank EWT frames), v5ner (v5's "
                         "gold-filtered NER); see golddata.py")
    ap.add_argument("--srl-pred-hidden", type=int, default=0,
                    help="MLP predicate head of this width (as srl_finetune --pred-hidden built)")
    ap.add_argument("--srl-ctx", type=int, default=0,
                    help="SRL context role head of this width (srl_finetune --role-ctx)")
    ap.add_argument("--ner-lstm", type=int, default=0,
                    help="BiLSTM NER head of this width (head_finetune --ner-lstm)")
    ap.add_argument("--hard-negatives", type=int, default=0, metavar="K",
                    help="offer each true relation's K nearest relations (by the model's label "
                         "embeddings) as negatives in its batch, within the source's schema")
    ap.add_argument("--gold-dev", action="store_true",
                    help="score UD EWT dev, PropBank dev and v5 gold NER dev every eval; their "
                         "linguistic scores replace the corpus dev ones in the composite")
    ap.add_argument("--kg-oversample", default="", metavar="DOMAIN=N,...",
                    help="repeat teacher windows of a domain N times, e.g. business=2")
    ap.add_argument("--legacy-relation", metavar="RUN",
                    help="the hybrid: RUN's ATLOP adapter, head and entity head, frozen, answer "
                         "the relations they were trained on; the label head the rest")
    ap.add_argument("--new-head-lr", type=float,
                    help="learning rate for heads trained from scratch (the span head) when "
                         "--head-lr is kept low for the heads being fine-tuned")
    ap.add_argument("--kg-holdout", action="store_true",
                    help="never train the zero-shot relation holdout (kg_holdout.json); score it "
                         "every eval on held-out teacher windows (relation_zs_acc: is a true relation "
                         "ranked first; selects best.pt) and report FewRel's never-trained "
                         "relations (fewrel_zs_acc)")
    ap.add_argument("--label-adapter", type=int, default=0, metavar="R",
                    help="read label texts through their own rank-R adapter (v6.2 zero-shot head)")
    ap.add_argument("--legacy-rank", type=int, default=16,
                    help="the --legacy-relation run's adapter rank")
    ap.add_argument("--span-head", action="store_true",
                    help="train the schema-conditioned entity span head")
    ap.add_argument("--kg-teacher", type=int, default=0, metavar="N",
                    help="add the exported teacher windows (kg_teacher.json), N times per "
                         "epoch, as partial relation supervision (needs --rel-head label)")
    ap.add_argument("--head-weight", action="append", default=[], metavar="HEAD=W",
                    help="fixed weight on a head's scaled loss, e.g. relation=3 to give "
                         "it a larger share of the encoder update (repeatable)")
    ap.add_argument("--p2-context", type=int, default=0, metavar="N",
                    help="transformer layers over words inside each phase-2 head "
                         "(for a frozen encoder; 0 = none)")
    ap.add_argument("--srl-mode", choices=["joint", "conditioned"], default="joint",
                    help="joint: every predicate in one pass (v6); conditioned: one "
                         "pass per predicate over its sentence (v5), for MODEL_CHANGES §1")
    ap.add_argument("--encoder-lr-scale", type=float, default=1.0,
                    help="multiply the encoder's registered learning rate (bake-offs)")
    ap.add_argument("--attn-implementation", choices=["eager", "sdpa"],
                    help="override the encoder's attention implementation (eager: the marker "
                         "pass can read attention maps on Ettin/ModernBERT)")
    ap.add_argument("--attr-head", action="store_true",
                    help="fact attributes (polarity, modality, holder, time) on the zero-shot "
                         "relation head, trained from the teacher corpus")
    ap.add_argument("--max-batch-tokens", type=int, metavar="N",
                    help="batch up to --batch-size items while items x longest input <= N "
                         "(set it to the encoder's window: short documents batch, long ones "
                         "go alone, memory stays at batch 1's)")
    ap.add_argument("--word-pool", choices=["first", "mean"], default="first",
                    help="word vectors: the first subword, or the mean of a word's subwords "
                         "(for tokenizers that split words often)")
    ap.add_argument("--prepare-only", action="store_true",
                    help="build every feature cache this run needs, list them in "
                         "features/manifest-<run>.json, and exit before training")
    ap.add_argument("--init-partial", action="store_true",
                    help="--init may carry heads this model lacks (a knowledge-pretrained "
                         "encoder starting phase 1): they are dropped")
    ap.add_argument("--grad-checkpointing", choices=["on", "off"],
                    help="override the encoder's default")
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--calib-steps", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-minutes", type=float, default=30.0)
    ap.add_argument("--push-minutes", type=float, default=5.0,
                    help="mirror status, events and train.log to the hub this often")
    ap.add_argument("--max-nonfinite", type=int, default=5,
                    help="stop after this many consecutive non-finite steps")
    ap.add_argument("--train-limit", type=int)
    ap.add_argument("--dev-limit", type=int)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device")
    ap.add_argument("--probe", type=int, metavar="STEPS",
                    help="memory/speed probe on the longest batches; saves nothing")
    args = ap.parse_args()

    spec = encoders.get(args.encoder)
    over = {k: v for k, v in (("encoder_lr", args.encoder_lr), ("head_lr", args.head_lr),
                              ("batch_size", args.batch_size),
                              ("grad_accum", args.grad_accum),
                              ("grad_checkpointing", None if args.grad_checkpointing is None
                               else args.grad_checkpointing == "on"),
                              ("attn_implementation", args.attn_implementation)) if v is not None}
    spec = dataclasses.replace(spec, **over)
    if args.encoder_lr_scale != 1.0:
        # One factor for every encoder keeps a bake-off fair to each one's own
        # swept rate (DeBERTa 8e-5, Ettin 2.8e-4 for phase 1).
        spec = dataclasses.replace(spec, encoder_lr=spec.encoder_lr * args.encoder_lr_scale)
    run_name = args.run_name or spec.name
    out = args.out or RUNS_DIR / "train" / run_name
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available()
                                          else "cpu"))
    if device.type == "cuda":
        # Without gradient checkpointing (knowledge-full) activation sizes vary
        # widely between batches; the caching allocator then fragments until
        # a small request fails with most of the card reserved but unused
        # (v7 bake-off: OOM at 38.4 GB reserved, ~6 GB in use). Expandable
        # segments let freed blocks be reused at any size.
        try:
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        except Exception as exc:                        # older torch: env var only
            print(f"allocator settings not applied: {exc}", flush=True)
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    def amp():
        return (torch.autocast("cuda", dtype=torch.bfloat16) if bf16
                else torch.autocast(device.type, enabled=False))

    torch.manual_seed(args.seed)
    t_start = time.time()

    def preparing(note: str) -> None:
        # Featurising can outlast a launcher's first-heartbeat wait on a
        # slow VM CPU: say the run is alive, and what it is doing, until
        # RunState takes over status.json.
        from .runstate import atomic_write_json
        Path(out).mkdir(parents=True, exist_ok=True)
        atomic_write_json(Path(out) / "status.json",
                          {"state": "preparing", "note": note, "started_at": t_start,
                           "heartbeat": time.time(), "step": 0, "total_steps": 0})
    preparing("loading the tokenizer")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    vocab = Vocab.load()
    cond = args.srl_mode == "conditioned"
    preparing("featurising the corpus")
    train = featurise(spec, "train", vocab, tok, limit=args.train_limit, srl_sentences=cond,
                      phase2=args.phase2, progress=preparing)
    dev = (featurise(spec, "dev", vocab, tok, limit=args.dev_limit, srl_sentences=cond,
                     phase2=args.phase2) if not args.probe else [])
    heads = HEADS + (PHASE2_HEADS if args.phase2 else ())
    weight = dict(HEAD_WEIGHT)
    for hw in args.head_weight:
        h, _, w = hw.partition("=")
        if h not in heads or not w:
            raise SystemExit(f"--head-weight {hw!r}: expected HEAD=W with HEAD one of {heads}")
        weight[h] = float(w)
    rd_dev = []
    if args.redocred_train:
        if not args.phase2:
            raise SystemExit("--redocred-train needs --phase2")
        from .data import redocred_features
        st = Counter()
        rd = redocred_features("train", tok, spec.max_subwords, vocab, st,
                               markers=args.rel_markers)
        if args.relation_data == "redocred":
            if not (args.train_scope.startswith("relation") or args.train_scope == "knowledge-full"):
                raise SystemExit("--relation-data redocred needs --train-scope relation: "
                                 "the corpus windows supervise every other head")
            train = rd * args.redocred_train
        else:
            train = train + rd * args.redocred_train
        rd_dev = (redocred_features("dev", tok, spec.max_subwords, vocab, st,
                                    markers=args.rel_markers)
                  if not args.probe else [])
        print("Re-DocRED: " + ", ".join(f"{k} {v:,}" for k, v in sorted(st.items()))
              + f"; train x{args.redocred_train}", flush=True)

    kg_data = dict(x.split("=") for x in args.kg_data.split(",") if x)
    if args.kg_teacher:
        kg_data.setdefault("teacher", str(args.kg_teacher))
    if kg_data:
        if args.rel_head not in ("label", "zs") or not args.phase2:
            raise SystemExit("--kg-data needs --phase2 and a label relation head")
        from ..label_bank import load as load_bank
        from .kdata import knowledge_features
        over = {d: int(n) for d, n in (x.split("=") for x in args.kg_oversample.split(",") if x)}
        kg_bank = load_bank()
        for src, n in kg_data.items():
            preparing(f"featurising {src}")
            st = Counter()
            feats = knowledge_features(src, "train", tok, spec.max_subwords, vocab.size("cls"),
                                       kg_bank, markers=args.rel_markers, limit=args.train_limit,
                                       stats=st, progress=preparing)
            if src == "teacher" and over:
                feats = [f for f in feats for _ in range(over.get(f.get("_domain"), 1))]
            w = float(n)
            if w < 1:                      # a fixed random fraction of the source
                rng = np.random.default_rng([args.seed, len(src)])
                feats = [feats[i] for i in sorted(rng.choice(len(feats), int(len(feats) * w),
                                                             replace=False))]
                w = 1
            train = train + feats * int(w)
            print(f"{src}: {len(feats):,} items x{n}; "
                  + ", ".join(f"{k} {v:,}" for k, v in sorted(st.items())), flush=True)

    gold_data = dict(x.split("=") for x in args.gold_data.split(",") if x)
    gold_dev_items = None
    if gold_data or args.gold_dev:
        from .golddata import gold_dev, gold_features
        for src, n in gold_data.items():
            preparing(f"featurising gold {src}")
            feats = gold_features(src, "train", tok, spec, vocab, phase2=args.phase2)
            w = float(n)
            if w < 1:
                rng = np.random.default_rng([args.seed, len(src), 7])
                feats = [feats[i] for i in sorted(rng.choice(len(feats), int(len(feats) * w),
                                                             replace=False))]
                w = 1
            train = train + feats * int(w)
            print(f"gold {src}: {len(feats):,} items x{n}", flush=True)
        if args.gold_dev and not args.probe:
            gold_dev_items = gold_dev(tok, spec, vocab, phase2=args.phase2)

    model = V6Model(spec.hf_id, {k: vocab.size(k) for k in vocab.labels},
                    attn_implementation=spec.attn_implementation,
                    srl_mode=args.srl_mode, phase2=args.phase2,
                    p2_context=args.p2_context, rel_adapter=args.rel_adapter,
                    rel_markers=args.rel_markers, rel_head=args.rel_head,
                    label_plain=args.label_pass == "plain",
                    legacy_relation=bool(args.legacy_relation),
                    span_head=args.span_head, label_adapter=args.label_adapter,
                    legacy_rank=args.legacy_rank,
                    trust_remote_code=spec.trust_remote_code,
                    word_pool=args.word_pool, attr_head=args.attr_head,
                    srl_pred_hidden=args.srl_pred_hidden, srl_ctx=args.srl_ctx,
                    ner_lstm=args.ner_lstm).to(device)
    init_relation = args.init_relation or "drop"
    if init_relation == "transplant" and args.rel_head != "label":
        raise SystemExit("--init-relation transplant needs --rel-head label")
    init_state = (_load_init(model, args.init, vocab, init_relation, partial=args.init_partial)
                  if args.init else None)
    preparing("building the model")
    if args.legacy_relation:
        _load_legacy(model, args.legacy_relation)
    extra_dev, hold_ids = {}, []
    if args.kg_holdout:
        from .kdata import load_holdout
        rec = load_holdout()
        if rec is None:
            raise SystemExit("--kg-holdout needs kg_holdout.json; run python -m v6.train.kdata --holdout")
        hold_ids = rec["relations"]
    label_coll = label_coll_dev = label_coll_zs = None
    if args.rel_head in ("label", "zs"):
        from ..label_bank import load as load_bank
        from .kdata import LabelCollator
        bank = load_bank()
        label_coll = LabelCollator(tok, bank, train=True, seed=args.seed,
                                   exclude_holdout=args.label_holdout, exclude_ids=hold_ids,
                                   text_modes=([("name_desc", 1.0)] if args.label_text == "canonical"
                                               else None))
        if args.hard_negatives:
            label_coll.siblings = relation_siblings(model, tok, device, bank, args.hard_negatives)
            print(f"hard negatives: {args.hard_negatives} siblings per relation, e.g. " + "; ".join(
                f"{bank['relations'][i]['name']} -> "
                + ", ".join(bank["relations"][j]["name"] for j in label_coll.siblings[i][:3])
                for i in list(label_coll.siblings)[:4]), flush=True)
        label_coll_dev = LabelCollator(tok, bank, train=False, exclude_holdout=args.label_holdout)
        if args.label_holdout:
            label_coll_zs = LabelCollator(tok, bank, train=False, only_holdout=True)
    span_dev = []
    if args.span_head and not args.probe:
        from .kdata import knowledge_features
        span_dev = knowledge_features("fewnerd", "dev", tok, spec.max_subwords, vocab.size("cls"),
                                      load_bank(), markers=args.rel_markers, limit=2000)
    if args.rel_head in ("label", "zs"):
        if args.span_head:
            from .kdata import TypeCollator
            label_coll = Chain(TypeCollator(tok, bank, train=True, seed=args.seed), label_coll)
            label_coll_dev = Chain(TypeCollator(tok, bank, train=False), label_coll_dev)
        if hold_ids and not args.probe:
            from .kdata import knowledge_features
            preparing("featurising the zero-shot dev sets")
            tdev = knowledge_features("teacher", "dev", tok, spec.max_subwords, vocab.size("cls"),
                                      bank, markers=args.rel_markers, limit=1500)
            fz = [r["id"] for r in bank["relations"] if not r.get("train", True)]
            fdev = [f for f in knowledge_features("fewrel", "val", tok, spec.max_subwords,
                                                  vocab.size("cls"), bank, markers=args.rel_markers)
                    if f["rel"]["triples"][0][2] in fz]
            fdev = [fdev[i] for i in range(0, len(fdev), max(1, len(fdev) // 800))]
            tc = TypeCollator(tok, bank, train=False) if args.span_head else None
            extra_dev = {
                "relation_zs_acc": (tdev, Chain(tc, LabelCollator(tok, bank, train=False,
                                                                 only_ids=hold_ids))),
                "fewrel_zs_acc": (fdev, Chain(tc, LabelCollator(tok, bank, train=False,
                                                               only_ids=fz)))}
            print(f"zero-shot dev: {len(tdev)} held-out teacher windows on {len(hold_ids)} relations; "
                  f"{len(fdev)} FewRel sentences on {len(fz)} never-trained relations", flush=True)
        if init_relation == "transplant":
            if init_state is None or "relation.bilinear.weight" not in init_state:
                raise SystemExit("--init-relation transplant needs an --init with an ATLOP head")
            _transplant(model, init_state, label_coll_dev, device)
            if args.kd_weight > 0:
                bil = torch.nn.Linear(1, 1)
                bil.weight = torch.nn.Parameter(init_state["relation.bilinear.weight"].to(device))
                bil.bias = torch.nn.Parameter(init_state["relation.bilinear.bias"].to(device))
                model.relation.keep_teacher(bil)
    if args.prepare_only:
        from .data import FEATURES, USED_CACHES
        caches = sorted({str(p.relative_to(FEATURES)) for p in USED_CACHES if p.exists()})
        man = FEATURES / f"manifest-{args.run_name or spec.name}.json"
        man.write_text(json.dumps({"encoder": spec.name, "args": vars(args), "caches": caches},
                                  indent=1, default=str))
        print(f"prepared {len(caches)} feature caches; wrote {man}", flush=True)
        return 0
    if args.train_scope == "phase2":
        if not args.phase2:
            raise SystemExit("--train-scope phase2 needs --phase2")
        for n, prm in model.named_parameters():
            prm.requires_grad = n.startswith(("coref.", "relation.", "mix.coref.",
                                              "mix.relation.", "ent.", "mix.ent."))
        model.freeze_encoder = True
        model.train()
    elif args.train_scope == "relation-full":
        # Diagnostic only: the whole encoder fine-tuned through the relation
        # pass, as ATLOP is. It bounds what a relation adapter could reach
        # with this head, and breaks the shared encoder for every other head:
        # its checkpoint is a measurement, never a deployment candidate.
        if args.rel_adapter:
            raise SystemExit("--train-scope relation-full trains the encoder itself; drop --rel-adapter")
        for n, prm in model.named_parameters():
            prm.requires_grad = n.startswith(("encoder.", "relation.", "mix.relation.",
                                              "ent.", "mix.ent."))
        heads = RELATION_HEADS
    elif args.train_scope == "knowledge-full":
        # v7 knowledge pretraining: the encoder itself trains, through the
        # plain pass (spans), the marker pass (relations, with its adapter)
        # and the label pass (with its adapter). Adapters switch by a global
        # flag, so a gradient-checkpointing recompute could run under the
        # wrong adapter: checkpointing must be off.
        if not (args.rel_adapter and args.label_adapter and args.rel_head == "zs"):
            raise SystemExit("--train-scope knowledge-full needs --rel-head zs, --rel-adapter "
                             "and --label-adapter")
        if spec.grad_checkpointing:
            raise SystemExit("--train-scope knowledge-full needs --grad-checkpointing off")
        for n, prm in model.named_parameters():
            prm.requires_grad = n.startswith(("encoder.", "rel_lora.", "relation.",
                                              "mix.relation.", "label_enc.", "label_lora.",
                                              "span.", "mix.ent_span.", "attr."))
        model.plain_grad = True
        heads = RELATION_HEADS
    elif args.train_scope == "relation":
        # The relation adapter, relation and entity heads alone, on the adapted
        # pass only: encoder weights, phase 1 and coref stay bit-identical.
        if not args.rel_adapter:
            raise SystemExit("--train-scope relation needs --rel-adapter")
        # With the hybrid, the entity-mention head belongs to the frozen
        # legacy path (exact Re-DocRED end to end); the span head is the
        # trainable entity path.
        trainable = ("rel_lora.", "relation.", "mix.relation.", "label_enc.",
                     "span.", "mix.ent_span.", "label_lora.") + \
            (() if args.legacy_relation else ("ent.", "mix.ent."))
        for n, prm in model.named_parameters():
            prm.requires_grad = n.startswith(trainable)
        heads = RELATION_HEADS
        model.rel_lora.sticky = True
    frozen_digest = _frozen_digest(model) if args.assert_frozen else None
    if frozen_digest is not None:
        n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"--assert-frozen: {n_frozen:,} untrained parameters fingerprinted", flush=True)
    wrong = {str(p.dtype) for p in model.parameters()} - {"torch.float32"}
    if wrong:
        raise SystemExit(f"parameters must be fp32 master weights; found {wrong}")
    if spec.grad_checkpointing:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    enc_decay, enc_plain = [], []
    for n, p in model.encoder.named_parameters():
        if p.requires_grad:
            (enc_plain if p.ndim < 2 or "norm" in n.lower() else enc_decay).append(p)
    # Heads trained from scratch beside heads being fine-tuned can take
    # their own rate (--new-head-lr): the fine-tuned ones keep --head-lr.
    new_prefixes = ("span.", "mix.ent_span.", "label_lora.", "label_enc.score") + \
        (("relation.pair.", "relation.lab_in.", "relation.doc_proj.", "relation.xattn.",
          "relation.th", "relation.bias.", "relation.log_scale") if args.rel_head == "zs" else ())
    loaded = getattr(model, "_init_loaded", None)
    if loaded is not None:
        # From a checkpoint: "new" means not loaded from it -- a pretrained
        # span or relation head is not new in stage B, a head it lacks is.
        new = {id(p) for n, p in model.named_parameters() if n not in loaded and p.requires_grad}
    else:
        new = {id(p) for n, p in model.named_parameters()
               if n.startswith(new_prefixes) and p.requires_grad}
    print(f"new-head learning rate on {sum(p.numel() for p in model.parameters() if id(p) in new):,} "
          f"parameters", flush=True)
    head_params = [p for p in model.head_parameters() if p.requires_grad]
    groups = [{"params": enc_decay, "lr": spec.encoder_lr, "weight_decay": args.weight_decay},
              {"params": enc_plain, "lr": spec.encoder_lr, "weight_decay": 0.0},
              {"params": [p for p in head_params if id(p) not in new or not args.new_head_lr],
               "lr": spec.head_lr, "weight_decay": 0.0},
              {"params": [p for p in head_params if id(p) in new and args.new_head_lr],
               "lr": args.new_head_lr or spec.head_lr, "weight_decay": 0.0}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, 0.98),
                            eps=1e-6, fused=device.type == "cuda")

    micro = spec.batch_size
    if args.probe and args.max_batch_tokens:
        # Worst cases among the batches training will actually form: the
        # largest attention (items x longest^2), the most entity pairs, the
        # most items.
        from .data import _length
        real = batch_order(train, micro, args.seed, 0, args.max_batch_tokens)

        def pairs(b):
            return sum(len(train[i]["rel"]["spans"]) ** 2 for i in b if train[i].get("rel"))
        def words(b):                   # candidate spans and type columns grow with words
            return sum(len(train[i]["first_subword"]) for i in b)
        picks = [max(real, key=lambda b: len(b) * max(_length(train[i]) for i in b) ** 2),
                 max(real, key=pairs), max(real, key=len), max(real, key=words)]
        print(f"probe batches: {[len(b) for b in picks]} items, longest "
              f"{[max(_length(train[i]) for i in b) for b in picks]} subwords, "
              f"{[pairs(b) for b in picks]} pairs, {[words(b) for b in picks]} words; "
              f"{len(real):,} batches per epoch "
              f"for {len(train):,} items", flush=True)
        order = (picks * spec.grad_accum)[:max(spec.grad_accum, len(picks))]
        return _probe(model, opt, train, order, args.probe, spec, tok, device, amp, label_coll)
    if args.probe:
        # Worst case: the longest windows, every step.
        longest = sorted(range(len(train)), key=lambda i: -len(train[i]["input_ids"]))
        order = [longest[i:i + micro] for i in range(0, micro * spec.grad_accum, micro)]
        if args.phase2:
            # The relation head grows with entity pairs, not length: the last
            # micro-batch is the windows with the most pairs instead.
            def n_pairs(i):
                r = train[i].get("rel")
                return len(r["spans"]) ** 2 if r else 0
            order[-1] = sorted(range(len(train)), key=lambda i: -n_pairs(i))[:micro]
        return _probe(model, opt, train, order, args.probe, spec, tok, device, amp, label_coll)

    per_epoch = len(batch_order(train, micro, args.seed, 0, args.max_batch_tokens)) // spec.grad_accum
    total = args.max_steps or per_epoch * args.epochs
    warm = max(1, int(total * args.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(total - warm, 1))))

    hub = Hub(args.hf_repo, run_name) if args.hf_repo else None
    rs = RunState(out, run_id=f"{run_name}-{int(time.time())}", hub=hub,
                  meta={"encoder": spec.asdict(), "args": vars(args) | {"out": str(out)},
                        "train_windows": len(train), "dev_windows": len(dev),
                        "steps_per_epoch": per_epoch})
    rs.push_minutes = args.push_minutes
    scale, calib = {h: 1.0 for h in heads}, defaultdict(list)
    ck = rs.try_resume(model, opt, sched)
    if ck:
        scale, calib = ck["loss_scale"], defaultdict(list, ck["calib"])
    rs.status(state="training", total_steps=total, push=True)
    print(f"{spec.name} on {device}{' bf16' if bf16 else ''}: {len(train)} train / "
          f"{len(dev)} dev windows, {per_epoch} steps/epoch, {total} steps, "
          f"batch {micro}x{spec.grad_accum}", flush=True)

    def extra():
        return {"loss_scale": scale, "calib": dict(calib), "vocab": vocab.to_json(),
                "encoder": spec.asdict(),
                "model_config": {"srl_mode": args.srl_mode, "phase2": args.phase2,
                                 "p2_context": args.p2_context, "ent_head": True,
                                 "rel_adapter": args.rel_adapter, "rel_markers": args.rel_markers,
                                 "rel_head": args.rel_head,
                                 "label_plain": args.label_pass == "plain",
                                 "legacy_relation": bool(args.legacy_relation),
                                 "span_head": args.span_head,
                                 "label_adapter": args.label_adapter,
                                 "legacy_rank": args.legacy_rank,
                                 "srl_pred_hidden": args.srl_pred_hidden,
                                 "srl_ctx": args.srl_ctx, "ner_lstm": args.ner_lstm,
                                 "word_pool": args.word_pool,
                                 "attr_head": args.attr_head,
                                 "kg_holdout": hold_ids,
                                 "head_lr": spec.head_lr, "new_head_lr": args.new_head_lr,
                                 "redocred_train": args.redocred_train,
                                 "relation_data": args.relation_data,
                                 "init": args.init, "train_scope": args.train_scope}}

    def stop(state, reason=""):
        # Checkpoint upload first, status last: a mirrored "stopped" or "done"
        # then means latest.pt is already on the hub, so the VM may be released.
        rs.save_latest(model, opt, sched, extra())
        if hub:
            hub.wait()
        rs.status(state=state, reason=reason, push=True)

    def control() -> str | None:
        """Act on a control file; return a stop reason if the run must end."""
        ckf, stopf = out / "CHECKPOINT", out / "STOP"
        if ckf.exists():
            ckf.unlink()
            rs.event("control", action="checkpoint")
            rs.save_latest(model, opt, sched, extra())
            rs.status(push=True)
            print(f"  checkpoint requested: saved at step {rs.p.step}", flush=True)
        if stopf.exists():
            why = stopf.read_text().strip() or "STOP file"
            stopf.unlink()                 # or a relaunch would stop at once
            rs.event("control", action="stop", reason=why)
            return why
        return None

    model.train()
    last_save = time.time()
    log, n_log = defaultdict(float), 0
    try:
        while rs.p.step < total:
            epoch = rs.p.epoch
            batches = batch_order(train, micro, args.seed, epoch, args.max_batch_tokens)
            start = (rs.p.step - epoch * per_epoch) * spec.grad_accum
            for m0 in range(start, per_epoch * spec.grad_accum, spec.grad_accum):
                this, seen, bad = defaultdict(float), set(), []
                for m in range(m0, m0 + spec.grad_accum):
                    b = collate([train[i] for i in batches[m]], tok.pad_token_id)
                    b = _to(label_coll(b) if label_coll is not None else b, device)
                    try:
                        with amp():
                            o = model(b, relation_only=args.train_scope.startswith("relation")
                                      or args.train_scope == "knowledge-full")
                        L = losses(o, b)
                    except torch.OutOfMemoryError:
                        # One oversized batch must not end a run that fits
                        # otherwise: drop this step (like a non-finite one),
                        # free the memory and carry on.
                        o = L = b = None
                        opt.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        bad = ["out_of_memory"]
                        break
                    vals = {h: l.item() for h, (l, n) in L.items() if n}
                    bad = [h for h, v in vals.items() if not math.isfinite(v)]
                    if bad:
                        break                  # never backpropagate a non-finite loss
                    if args.kd_weight:
                        # Relative to the calibrated relation loss, so a
                        # weight of 1 means "as strong as the relation term".
                        weight["relation_kd"] = args.kd_weight * scale.get("relation", 1.0) * max(
                            0.0, 1.0 - rs.p.step / max(total / 2, 1))
                    loss = sum(scale.get(h, 1.0) * weight.get(h, 1.0) * l
                               for h, (l, _) in L.items()) / spec.grad_accum
                    try:
                        loss.backward()
                    except torch.OutOfMemoryError:
                        o = L = b = loss = None
                        opt.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        bad = ["out_of_memory"]
                        break
                    for h, v in vals.items():
                        this[h] += v / spec.grad_accum
                        seen.add(h)
                    this["total"] += loss.item()
                gn = None
                if not bad:
                    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                    if not math.isfinite(gn):
                        bad = ["grad_norm"]
                if bad:
                    opt.zero_grad(set_to_none=True)
                    sched.step()               # the step is consumed, not applied
                else:
                    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                rs.p.step += 1
                step = rs.p.step
                rs.p.lr = sched.get_last_lr()[0]
                if bad:
                    rs.p.nonfinite_steps += 1
                    rs.p.nonfinite_streak += 1
                    rs.event("nonfinite", step=step, heads=bad,
                             streak=rs.p.nonfinite_streak)
                    print(f"  NON-FINITE at step {step} ({', '.join(bad)}): update "
                          f"skipped, {rs.p.nonfinite_streak} in a row", flush=True)
                    if rs.p.nonfinite_streak >= args.max_nonfinite:
                        why = (f"{rs.p.nonfinite_streak} consecutive non-finite steps "
                               f"at step {step}; latest.pt left at the last good state")
                        rs.event("abort", reason=why)
                        rs.status(state="failed", reason=why, push=True)
                        if hub:
                            hub.wait()
                        print(f"STOPPED: {why}", flush=True)
                        return 2
                    rs.status()
                else:
                    rs.p.nonfinite_streak = 0
                    rs.p.grad_norm = round(gn, 4)
                    for k, v in this.items():
                        log[k] += v
                    n_log += 1

                # Loss scale: each head divided by its own mean over the first
                # calib_steps steps, so the heads start on an equal footing.
                # A skipped step's partial losses are not evidence.
                if step <= args.calib_steps and not bad:
                    for h in seen:
                        calib[h].append(this[h])
                if step == args.calib_steps:
                    scale = {h: 1.0 / max(sum(calib[h]) / len(calib[h]), 1e-3)
                             if calib[h] else 1.0 for h in heads}
                    # Distillation starts at ~0 (the transplant is exact), so
                    # calibrating it would multiply it a thousandfold; its
                    # strength is --kd-weight's alone.
                    scale["relation_kd"] = 1.0
                    rs.event("loss_scale", scale=scale)
                    print("  loss scales: " + " ".join(
                        f"{h} {s:.3f}" for h, s in scale.items()), flush=True)

                if (step % args.log_every == 0 or step == total) and n_log:
                    hl = {h: round(log[h] / n_log, 4) for h in heads}
                    rs.status(loss=round(log["total"] / n_log, 4), head_loss=hl)
                    p = rs.p
                    print(f"step {step}/{total} ep {epoch} loss {p.loss:.4f} | "
                          + " ".join(f"{h} {v:.3f}" for h, v in hl.items())
                          + f" | gn {p.grad_norm} lr {p.lr:.2e} | {p.steps_per_sec or 0:.2f} "
                          f"st/s eta {p.eta_minutes}m | mem {p.memory} disk {p.disk_free_gb}G",
                          flush=True)
                    log, n_log = defaultdict(float), 0

                if step % args.eval_every == 0 or step == total:
                    rs.status(state="evaluating")
                    r = evaluate(model, dev, vocab, tok.pad_token_id, device, amp, micro * 2,
                                 rel_gold=rd_dev, label_coll=label_coll_dev,
                                 label_coll_zs=label_coll_zs, span_gold=span_dev,
                                 extra_rel=extra_dev, gold_dev=gold_dev_items,
                                 beat=lambda: rs.status(state="evaluating"),
                                 knowledge_only=args.train_scope == "knowledge-full")
                    if hasattr(getattr(model, "relation", None), "gates"):
                        r["xattn_gates"] = model.relation.gates()
                    if frozen_digest is not None and _frozen_digest(model) != frozen_digest:
                        why = f"a frozen tensor changed by step {step}"
                        rs.event("abort", reason=why)
                        rs.status(state="failed", reason=why, push=True)
                        raise SystemExit(f"STOPPED: {why}")
                    rs.event("eval", step=step, **r)
                    better = rs.maybe_save_best(model, r["composite"], extra())
                    rs.status(state="training", last_eval=r, push=True)
                    print(f"  eval step {step}: " + " ".join(f"{k} {v:.4f}" if isinstance(v, (int, float)) else f"{k} {v}"
                                      for k, v in r.items())
                          + ("  * best" if better else ""), flush=True)

                why = control()
                if why:
                    stop("stopped", why)
                    print(f"stopped at step {step} ({why}); checkpointed", flush=True)
                    return 0
                if rs.stop_requested:
                    stop("interrupted", "signal")
                    print(f"interrupted at step {step}; checkpointed", flush=True)
                    return 0
                if time.time() - last_save > args.save_minutes * 60:
                    rs.save_latest(model, opt, sched, extra())
                    last_save = time.time()
                if step >= total:
                    break
            else:
                rs.p.epoch = epoch + 1
                rs.save_latest(model, opt, sched, extra())
                last_save = time.time()
                continue
            break
    except Exception as exc:
        rs.event("error", error=f"{type(exc).__name__}: {exc}", step=rs.p.step,
                 memory=memory())
        rs.status(state="failed", reason=f"{type(exc).__name__}: {exc}"[:300], push=True)
        if hub:
            hub.wait()
        raise
    stop("done")
    print(f"done: {rs.p.step} steps, best composite {rs.p.best:.4f} "
          f"at step {rs.p.best_step}", flush=True)
    return 0


def _probe(model, opt, train, order, steps, spec, tok, device, amp, label_coll=None) -> int:
    """Run ``steps`` optimiser steps on the longest windows and report peak
    memory and speed. This is the A100 fit check: batch size, gradient
    checkpointing and bf16 are measured here, not assumed."""
    import json
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for s in range(steps):
        t0 = time.time()
        for idx in order:
            b = collate([train[i] for i in idx], tok.pad_token_id)
            b = _to(label_coll(b) if label_coll is not None else b, device)
            with amp():
                L = losses(model(b), b)
            (sum(l for l, _ in L.values()) / len(order)).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        print(f"  probe step {s + 1}/{steps}: {times[-1]:.2f}s {memory()}", flush=True)
    n_words = sum(len(train[i]["first_subword"]) for idx in order for i in idx)
    rep = {"encoder": spec.name, "batch_size": spec.batch_size,
           "grad_accum": spec.grad_accum, "grad_checkpointing": spec.grad_checkpointing,
           "longest_subwords": max(len(train[i]["input_ids"]) for idx in order for i in idx),
           "sec_per_step": round(sum(times[1:]) / max(len(times) - 1, 1), 3),
           "words_per_sec": round(n_words / (sum(times[1:]) / max(len(times) - 1, 1)), 1),
           **memory()}
    print("PROBE " + json.dumps(rep), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Score v6 checkpoints on public gold -- the evaluation that picks the encoder.

    uv run python -m v6.train.gold_eval                       # the three full runs
    uv run python -m v6.train.gold_eval --runs ettin-1b --limit 200   # quick look

Dev scores during training are agreement with our own annotators. These are
human-annotated test sets none of the models trained on (TRAINING.md §8):

| benchmark | layers | metric |
|---|---|---|
| UD English EWT test | POS, lemma, morph, DEP | accuracy, accuracy, feat-F1, UAS/LAS |
| PropBank EWT test | SRL | span-F1, gold predicate, ``V`` excluded |
| OntoNotes 5.0 test | NER | span-F1 over 18 types |
| SGD test | CLS | micro-F1 over the five scorable acts |

The scorers are ``v6/score`` and ``v6/gold/sgd_cls``'s, the same ones the
annotator bake-off used, so v5's figures in DECISIONS.md are a like-for-like
reference. NER and SRL are decoded with constrained BIO Viterbi, as v5 was.
DEP heads are the arc argmax, as in training-time evaluation; the share of
sentences that come out as well-formed trees is reported beside them.

Every input is one sentence (one SGD turn with its preceding turns as
context), so no encoder gains from its longer window.

Differences between encoders come with **paired bootstrap 95% intervals** over
items: a difference whose interval spans zero is a tie, however it looks.

Phase 2, for checkpoints with the coref and relation heads:

| benchmark | layer | metric |
|---|---|---|
| LitBank, 400-word windows | coref | CoNLL-F1 per window, averaged |
| Re-DocRED test, docs ≤ 512 words | relations | triple-F1, gold entities; and end to end |

These two inputs are whole documents, not single sentences. Their teachers'
scores (LingMess, ATLOP) are on the same items.

Sentiment has no public gold here.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from ..config import DATA_DIR, RUNS_DIR
from ..entities import bio_spans as bio_spans_t
from ..score.metrics import _feats_set, bio_spans
from .data import IGNORE, blank_window, collate
from .encoders import EncoderSpec
from .vocab import Vocab, apply_script

CKPT_DIR = DATA_DIR / "v6-eval" / "checkpoints"
OUT_DIR = RUNS_DIR / "gold-eval"
HF_REPO = "dragonscale-ai/kniv-v6-runs"
RUNS = ("deberta-v3-large", "ettin-400m", "ettin-1b")
CLS_SCORED = ("Question", "Inform", "Directive", "Commissive", "Social")


# ── model ───────────────────────────────────────────────────────────
def load_run(run: str, repo: str, device, local: Path | None = None) -> tuple:
    """``best.pt`` from the hub, or from ``local/<run>/`` when given."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from .model import V6Model
    if local:
        path, metric_path = local / run / "best.pt", local / run / "best.metric"
    else:
        path = hf_hub_download(repo, f"{run}/best.pt", local_dir=str(CKPT_DIR))
        metric_path = hf_hub_download(repo, f"{run}/best.metric", local_dir=str(CKPT_DIR))
    metric = json.loads(Path(metric_path).read_text())
    ck = torch.load(path, map_location="cpu", weights_only=False)
    spec = EncoderSpec(**ck["encoder"])
    vocab = Vocab.from_json(ck["vocab"])
    cfg = ck.get("model_config") or {}
    model = V6Model(spec.hf_id, {k: vocab.size(k) for k in vocab.labels},
                    pretrained=False, srl_mode=cfg.get("srl_mode", "joint"),
                    phase2=cfg.get("phase2", False), p2_context=cfg.get("p2_context", 0),
                    ent_head=cfg.get("ent_head", False), rel_adapter=cfg.get("rel_adapter", 0),
                    rel_markers=cfg.get("rel_markers", False),
                    rel_head=cfg.get("rel_head", "bilinear"),
                    label_plain=cfg.get("label_plain", False),
                    legacy_relation=cfg.get("legacy_relation", False),
                    span_head=cfg.get("span_head", False),
                    label_adapter=cfg.get("label_adapter", 0),
                    legacy_rank=cfg.get("legacy_rank"),
                    attn_implementation=spec.attn_implementation,
                    trust_remote_code=spec.trust_remote_code,
                    word_pool=cfg.get("word_pool", "first"),
                    attr_head=cfg.get("attr_head", False))
    model.load_state_dict(ck["model"])          # strict: a mismatch raises
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    return model, tok, spec, vocab, metric


def window(tokens: list[str], spans: list[tuple[int, int]], tok, spec,
           vocab: Vocab, predicates: list[int] = (), srl_sentences: bool = False) -> dict:
    """A label-free feature dict: the gold is scored outside the model."""
    return blank_window(tokens, spans, tok, spec.max_subwords, vocab.size("cls"),
                        predicates, srl_sentences)


@torch.no_grad()
def run_model(model, feats: list[dict], tok, device, batch: int = 32) -> list[dict]:
    """Per window: numpy logits/predictions, in input order."""
    order = sorted(range(len(feats)), key=lambda i: len(feats[i]["input_ids"]))
    out: list[dict | None] = [None] * len(feats)
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        b = {n: v.to(device) for n, v in collate([feats[i] for i in idx],
                                                 tok.pad_token_id).items()}
        o = model(b, predict=False, predict_heads=True)
        fb = o["frame_batch"].tolist()
        for j, i in enumerate(idx):
            W = feats[i]["_W"]
            out[i] = {
                "pos": o["pos"][j, :W].argmax(-1).cpu().numpy(),
                "morph": o["morph"][j, :W].argmax(-1).cpu().numpy(),
                "lemma": o["lemma"][j, :W].argmax(-1).cpu().numpy(),
                "ner": torch.log_softmax(o["ner"][j, :W].float(), -1).cpu().numpy(),
                "head": o["rel_head"][j, :W].cpu().numpy(),
                "rel": o["rel"][j, :W].argmax(-1).cpu().numpy(),
                "srl": [torch.log_softmax(o["srl_role"][f, :W].float(), -1).cpu().numpy()
                        for f in range(len(fb)) if fb[f] == j],
                # Padded to the batch's longest window: keep this window's own.
                "cls": torch.sigmoid(o["cls"][j, :feats[i]["_S"]].float()).cpu().numpy(),
                "srl_pred": o["srl_pred"][j, :W].argmax(-1).cpu().numpy(),
            }
    return out


# ── decoding ────────────────────────────────────────────────────────
def bio_constraints(tags: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """``allowed[i, j]``: may tag j follow tag i; ``start[j]``: may j begin."""
    C = len(tags)
    allowed = np.ones((C, C), dtype=bool)
    start = np.ones(C, dtype=bool)
    for j, t in enumerate(tags):
        if t.startswith("I-"):
            start[j] = False
            for i, s in enumerate(tags):
                allowed[i, j] = s in (f"B-{t[2:]}", t)
    return allowed, start


def viterbi(logp: np.ndarray, allowed: np.ndarray, start: np.ndarray,
            keep: np.ndarray | None = None) -> np.ndarray:
    """Best BIO-consistent path. ``keep`` limits which tags may be emitted at
    all (SRL: ``V`` only at the predicate)."""
    NEG = -1e9
    T, C = logp.shape
    lp = logp.copy()
    if keep is not None:
        lp[~keep] = NEG
    trans = np.where(allowed, 0.0, NEG)
    score = np.where(start, lp[0], NEG)
    back = np.zeros((T, C), dtype=np.int32)
    for t in range(1, T):
        cand = score[:, None] + trans
        back[t] = cand.argmax(0)
        score = cand.max(0) + lp[t]
    path = np.zeros(T, dtype=np.int32)
    path[-1] = score.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


# ── benchmarks: each returns per-item counts for the bootstrap ──────
def eval_ud(model, tok, spec, vocab, device, limit) -> dict:
    from ..gold.ud_ewt import load_ud_items
    items = load_ud_items("test", limit=limit)
    feats = [window(it.tokens, [(0, len(it.tokens))], tok, spec, vocab) for it in items]
    outs = run_model(model, feats, tok, device)
    L = vocab.labels
    rows = []
    for it, f, o in zip(items, feats, outs):
        W, g = f["_W"], it.layers
        n = len(it.tokens)
        assert W == n, "a UD sentence was truncated"
        pos = [L["pos"][x] for x in o["pos"]]
        lem = [apply_script(w, L["lemma"][x]) for w, x in zip(it.tokens, o["lemma"])]
        mor = [L["morph"][x] for x in o["morph"]]
        heads = o["head"].tolist()            # 0 = root, else 1-indexed: UD's convention
        rels = [L["dep"][x] for x in o["rel"]]
        r = {"pos": [sum(a == b for a, b in zip(pos, g["pos"])), n],
             "lemma": [sum(a == b for a, b in zip(lem, g["lemma"])), n],
             "morph_exact": [sum(a == b for a, b in zip(mor, g["morph"])), n]}
        tp = npred = ngold = 0
        for a, b in zip(mor, g["morph"]):
            ps, gs = _feats_set(a), _feats_set(b)
            tp += len(ps & gs); npred += len(ps); ngold += len(gs)
        r["morph_f1"] = [tp, npred, ngold]
        dg = g["dep"]
        uas = sum(h == gh for h, gh in zip(heads, dg["heads"]))
        las = sum(h == gh and rl == gr for h, gh, rl, gr
                  in zip(heads, dg["heads"], rels, dg["rels"]))
        r["uas"], r["las"] = [uas, n], [las, n]
        r["tree"] = [int(_is_tree(heads)), 1]
        rows.append(r)
    return {"items": rows, "kinds": {"pos": "acc", "lemma": "acc",
                                     "morph_exact": "acc", "morph_f1": "f1",
                                     "uas": "acc", "las": "acc", "tree": "acc"}}


def _is_tree(heads: list[int]) -> bool:
    if sum(h == 0 for h in heads) != 1:
        return False
    for i in range(len(heads)):
        seen, j = set(), i
        while heads[j] != 0:
            if j in seen:
                return False
            seen.add(j)
            j = heads[j] - 1
    return True


def eval_srl(model, tok, spec, vocab, device, limit) -> dict:
    from ..gold.propbank import load_srl_items
    items = load_srl_items(limit=limit)
    # One forward per sentence, all its gold predicates at once.
    by_sent: dict[tuple, list] = defaultdict(list)
    for it in items:
        by_sent[tuple(it.tokens)].append(it)
    sents = list(by_sent)
    cond = getattr(model, "srl_mode", "joint") == "conditioned"
    feats = [window(list(s), [(0, len(s))], tok, spec, vocab,
                    predicates=[it.predicate_idx for it in by_sent[s]],
                    srl_sentences=cond) for s in sents]
    outs = run_model(model, feats, tok, device)
    tags = vocab.labels["srl"]
    allowed, start = bio_constraints(tags)
    v = tags.index("V")
    # The predicate's part of speech, from the gold file: v6 trains on VERB
    # predicates only, so adjectival and nominal ones are scored apart.
    from ..config import PREPARED_DIR
    raw = json.loads((PREPARED_DIR / "srl_test.json").read_text())
    pos_of = {(tuple(e["words"]), e["predicate_idx"]): e["pos_tags"][e["predicate_idx"]]
              for e in raw if e.get("words") and e.get("predicate_idx") is not None
              and e.get("pos_tags")}
    rows = []
    for s, o in zip(sents, outs):
        for it, lp in zip(by_sent[s], o["srl"]):
            keep = np.ones_like(lp, dtype=bool)
            keep[:, v] = False
            keep[it.predicate_idx, :] = False
            keep[it.predicate_idx, v] = True     # the predicate is given
            pred = [tags[x] for x in viterbi(lp, allowed, start, keep)]
            ps, gs = bio_spans(pred), bio_spans(it.layers["srl"])
            c = [len(ps & gs), len(ps), len(gs)]
            verb = pos_of.get((tuple(it.tokens), it.predicate_idx)) == "VERB"
            rows.append({"srl_f1": c, "srl_f1_verb": c if verb else [0, 0, 0],
                         "srl_f1_other": [0, 0, 0] if verb else c})
    return {"items": rows, "kinds": {"srl_f1": "f1", "srl_f1_verb": "f1",
                                     "srl_f1_other": "f1"}}


POS_GROUPS = ("VERB", "AUX", "NOUN", "ADJ")


def _group(upos: str) -> str:
    return upos if upos in POS_GROUPS else "other"


def eval_srl_full(model, tok, spec, vocab, device, limit) -> dict:
    """PropBank EWT test with every predicate annotated (``gold.propbank_full``).

    Three measurements per sentence:
    - **predicate identification**: the predicate head's flags against every
      gold predicate, overall and by UPOS group;
    - **roles given the gold predicate**, as ``eval_srl`` but over all frames;
    - **end to end**: roles for the predicates the model itself found. An
      argument counts only if predicate, role and span all match, so a missed
      predicate loses all its arguments and a spurious one adds false ones.
    """
    path = DATA_DIR / "prepared" / "propbank-full" / "test.json"
    if not path.exists():
        raise SystemExit(f"{path} missing; run python -m v6.gold.propbank_full")
    rows = json.loads(path.read_text())[:limit] if limit else json.loads(path.read_text())
    tags = vocab.labels["srl"]
    allowed, start = bio_constraints(tags)
    v = tags.index("V")
    cond = getattr(model, "srl_mode", "joint") == "conditioned"

    def decode(lp, p):
        keep = np.ones_like(lp, dtype=bool)
        keep[:, v] = False
        keep[p, :] = False
        keep[p, v] = True
        return bio_spans([tags[x] for x in viterbi(lp, allowed, start, keep)])

    # Pass 1: gold predicates -> role scores, and the predicate head's flags.
    gold_preds = [[f["predicate_idx"] for f in r["frames"]] for r in rows]
    feats = [window(r["words"], [(0, len(r["words"]))], tok, spec, vocab,
                    predicates=gp, srl_sentences=cond) for r, gp in zip(rows, gold_preds)]
    outs = run_model(model, feats, tok, device)
    found = [sorted(np.nonzero(o["srl_pred"] == 1)[0].tolist()) for o in outs]
    # Pass 2: the model's own predicates -> roles, for the end-to-end score.
    feats2 = [window(r["words"], [(0, len(r["words"]))], tok, spec, vocab,
                     predicates=fp, srl_sentences=cond) for r, fp in zip(rows, found)]
    outs2 = run_model(model, feats2, tok, device)

    items = []
    for r, gp, o, fp, o2 in zip(rows, gold_preds, outs, found, outs2):
        row = {}
        gset, fset = set(gp), set(fp)
        row["pred_id"] = [len(gset & fset), len(fset), len(gset)]
        for g in POS_GROUPS:
            gg = {p for p in gset if _group(r["upos"][p]) == g}
            fg = {p for p in fset if _group(r["upos"][p]) == g}
            row[f"pred_id_{g.lower()}"] = [len(gg & fg), len(fg), len(gg)]
        tp = n_p = n_g = 0
        gold_args = set()
        for f, lp in zip(r["frames"], o["srl"]):
            gs = bio_spans(f["srl_tags"])
            ps = decode(lp, f["predicate_idx"])
            tp += len(gs & ps); n_p += len(ps); n_g += len(gs)
            gold_args |= {(f["predicate_idx"],) + s for s in gs}
        row["role_f1"] = [tp, n_p, n_g]
        pred_args = set()
        for p, lp in zip(fp, o2["srl"]):
            pred_args |= {(p,) + s for s in decode(lp, p)}
        row["e2e_f1"] = [len(gold_args & pred_args), len(pred_args), len(gold_args)]
        items.append(row)
    kinds = {"pred_id": "f1", "role_f1": "f1", "e2e_f1": "f1",
             **{f"pred_id_{g.lower()}": "f1" for g in POS_GROUPS}}
    return {"items": items, "kinds": kinds}


def eval_ner(model, tok, spec, vocab, device, limit) -> dict:
    from ..gold.ner import load_ner_items
    items = load_ner_items("ontonotes5", limit=limit)
    feats = [window(it.tokens, [(0, len(it.tokens))], tok, spec, vocab) for it in items]
    outs = run_model(model, feats, tok, device)
    tags = vocab.labels["ner"]
    allowed, start = bio_constraints(tags)
    rows = []
    for it, o in zip(items, outs):
        pred = [tags[x] for x in viterbi(o["ner"], allowed, start)]
        ps, gs = bio_spans(pred), bio_spans(it.layers["ner"])
        rows.append({"ner_f1": [len(ps & gs), len(ps), len(gs)]})
    return {"items": rows, "kinds": {"ner_f1": "f1"}}


_WORD = re.compile(r"\w+(?=n't)|n't|\w+(?:[-.']\w+)*|'\w+|[^\w\s]")


def eval_cls(model, tok, spec, vocab, device, limit, context: int = 6) -> dict:
    """SGD test turns, each the last sentence of a window holding up to
    ``context`` preceding turns -- the conversation domain's shape in training."""
    from ..gold.sgd_cls import load_gold
    gold = load_gold("test")
    dialogues: dict[str, list] = defaultdict(list)
    for g in gold:
        dialogues[g["dialogue_id"]].append(g)
    turns = [g for d in sorted(dialogues) for g in sorted(dialogues[d], key=lambda x: x["turn_idx"])]
    if limit:
        rng = np.random.default_rng(13)
        keep = set(rng.choice(len(turns), size=min(limit, len(turns)), replace=False).tolist())
        targets = [i for i in range(len(turns)) if i in keep]
    else:
        targets = list(range(len(turns)))
    feats = []
    for i in targets:
        g = turns[i]
        prev = [t for t in dialogues[g["dialogue_id"]] if t["turn_idx"] < g["turn_idx"]][-context:]
        sents = [_WORD.findall(t["text"]) for t in prev] + [_WORD.findall(g["text"])]
        sents = [s for s in sents[:-1] if s] + [sents[-1] or ["."]]
        while True:
            toks, spans = [], []
            for s in sents:
                spans.append((len(toks), len(toks) + len(s)))
                toks += s
            f = window(toks, spans, tok, spec, vocab)
            if f["sent_whole"][-1] or len(sents) == 1:
                break
            sents = sents[1:]                       # drop context until the turn fits
        feats.append(f)
    outs = run_model(model, feats, tok, device)
    labels = vocab.labels["cls"]
    rows = []
    for i, o in zip(targets, outs):
        pred = {labels[k] for k, p in enumerate(o["cls"][-1]) if p > 0.5}
        g = set(turns[i]["cls"])
        tp = sum(l in pred and l in g for l in CLS_SCORED)
        rows.append({"cls_f1": [tp, sum(l in pred for l in CLS_SCORED),
                                sum(l in g for l in CLS_SCORED)],
                     "cls_exact": [int(pred == g), 1],
                     "_pred": sorted(pred), "_gold": sorted(g)})
    return {"items": rows, "kinds": {"cls_f1": "f1", "cls_exact": "acc"}}


# ── phase 2: coreference and relations ──────────────────────────────
def phase2_window(tokens: list[str], spans: list[tuple[int, int]], tok, spec,
                  vocab: Vocab, entities: list[list[tuple[int, int]]] | None = None,
                  markers: bool = False) -> dict:
    """``window`` plus the phase-2 inputs: no coref targets, and ``entities``
    (each a list of ``(start, end_exclusive)`` mentions) as the relation
    head's input with every pair unlabelled. ``_ent_keep`` maps the kept
    entities back to the caller's indices: an entity whose every mention
    falls past the subword limit cannot be scored."""
    f = window(tokens, spans, tok, spec, vocab)
    W = f["_W"]
    f.update(coref=None, rel=None, _n_rel=vocab.size("relations"), _ent_keep=[])
    kept, keep = [], []
    for ei, ms in enumerate(entities or []):
        ms = [(a, b) for a, b in ms if b <= W and f["word_ok"][a]]
        if ms:
            kept.append(np.asarray(ms, dtype=np.int32)); keep.append(ei)
    if len(kept) >= 2:
        f["rel"] = {"spans": kept, "types": np.zeros(len(kept), dtype=np.int32),
                    "triples": []}
        f["_ent_keep"] = keep
        if markers:
            from .data import marker_features
            f["rm"] = marker_features(tokens, spans, kept, tok)
    return f


def _sentences(tokens: list[str]) -> list[tuple[int, int]]:
    """Sentence spans at terminal punctuation, for gold without boundaries
    (LitBank). Only the coref span head reads them: a span cannot cross one."""
    spans, s = [], 0
    for i, t in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if t in (".", "!", "?") and nxt not in ('"', "'", "”", "’", "''"):
            spans.append((s, i + 1)); s = i + 1
        elif t in ('"', "”", "''") and i > 0 and tokens[i - 1] in (".", "!", "?"):
            spans.append((s, i + 1)); s = i + 1
    if s < len(tokens):
        spans.append((s, len(tokens)))
    return spans


def link_clusters(antecedent: dict[int, int]) -> list[list[int]]:
    """Word clusters from each word's chosen antecedent (union-find)."""
    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for w, a in antecedent.items():
        parent[find(w)] = find(a)
    groups: dict[int, list[int]] = defaultdict(list)
    for w in parent:
        groups[find(w)].append(w)
    return sorted(sorted(g) for g in groups.values() if len(g) > 1)


def relation_triples(logits: np.ndarray, pairs: list[tuple[int, int]],
                     names: list[str], top: int = 4, offset: float = 0.0) -> set[tuple]:
    """ATLOP decoding: a relation holds when its logit beats the threshold
    class (column 0) lowered by ``offset``, at most ``top`` per pair."""
    out = set()
    for (h, t), lg in zip(pairs, logits):
        above = [k for k in np.argsort(-lg[1:])[:top] if lg[k + 1] > lg[0] - offset]
        out |= {(h, t, names[k]) for k in above}
    return out


@torch.no_grad()
def schema_embeddings(model, tok, device, cols: list[int] | None = None):
    """The label pass over a schema, once: the bank's relations (all, in bank
    order = the vocabulary's, unless ``cols`` picks some), by name and
    description. ``[K, label_dim]``, passed to the model as ``rel_label_emb``."""
    from ..label_bank import load as load_bank
    from .kdata import label_text
    rels = load_bank()["relations"]
    if cols is None:                    # Re-DocRED's own 96, in vocabulary order
        cols = [r["id"] for r in rels if r["source"] == "redocred"]
    enc = tok([label_text(rels[c]) for c in cols], padding=True, truncation=True,
              max_length=48, return_tensors="pt").to(device)
    return model.label_embeddings(enc["input_ids"], enc["attention_mask"])


@torch.no_grad()
def run_phase2(model, feats: list[dict], tok, device, batch: int = 4,
               cols: list[int] | None = None) -> list[dict]:
    """Per window: predicted coref clusters (inclusive word spans), relation
    logits per ordered entity pair, and NER log-probabilities."""
    order = sorted(range(len(feats)), key=lambda i: len(feats[i]["input_ids"]))
    out: list[dict | None] = [None] * len(feats)
    label_emb = (schema_embeddings(model, tok, device, cols) if model.rel_kind in ("label", "zs")
                 else None)
    if label_emb is not None:
        from ..label_bank import load as load_bank
        ids = cols if cols is not None else [r["id"] for r in load_bank()["relations"]
                                              if r["source"] == "redocred"]
        label_cols = torch.tensor(ids, device=device)
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        b = {n: v.to(device) for n, v in collate([feats[i] for i in idx],
                                                 tok.pad_token_id).items()}
        if label_emb is not None:
            b["rel_label_emb"] = label_emb
            b["rel_label_cols"] = label_cols     # the hybrid routes columns by bank id
        o = model(b, predict=False, predict_heads=True)
        best = o["coref_scores"].argmax(-1).cpu()                    # [B,W]
        cand = o["coref_cand"].cpu()
        rel = (o["relation"].float().cpu().numpy() if "relation" in o
               else np.zeros((0, model.sizes["relations"] + 1), dtype=np.float32))
        pair_b = b["pair_batch"].cpu().numpy()
        for j, i in enumerate(idx):
            W = feats[i]["_W"]
            ante = {w: int(cand[j, w, best[j, w] - 1]) for w in range(W) if best[j, w] > 0}
            words = link_clusters(ante)
            heads = [w for c in words for w in c]
            span = {}
            if heads:
                h = torch.tensor(heads, device=device)
                lg, pos = model.coref.spans(o["coref_a"], torch.full_like(h, j), h,
                                            b["sent_of_word"])
                st = lg[..., 0].masked_fill(pos > h.unsqueeze(1), -1e9).argmax(1)
                en = lg[..., 1].masked_fill(pos < h.unsqueeze(1), -1e9).argmax(1)
                st = pos.gather(1, st.unsqueeze(1)).squeeze(1).tolist()
                en = pos.gather(1, en.unsqueeze(1)).squeeze(1).tolist()
                span = {w: (s, e) for w, s, e in zip(heads, st, en)}
            out[i] = {"coref": [[span[w] for w in c] for c in words],
                      "relation": rel[pair_b == j],
                      "ner": torch.log_softmax(o["ner"][j, :W].float(), -1).cpu().numpy(),
                      "ent": (torch.log_softmax(o["ent"][j, :W].float(), -1).cpu().numpy()
                              if "ent" in o else None)}
    return out


def eval_litbank(model, tok, spec, vocab, device, limit) -> dict:
    """LitBank coreference, the bake-off's 400-word windows: CoNLL-F1 (mean of
    MUC, B³, CEAF-e) per window, averaged -- the teacher LingMess's protocol."""
    from ..gold.litbank import load_coref_items
    from ..score.coref import conll_f1
    items = load_coref_items(limit=limit)
    feats = [phase2_window(it.tokens, _sentences(it.tokens), tok, spec, vocab)
             for it in items]
    outs = run_phase2(model, feats, tok, device)
    rows = []
    for it, f, o in zip(items, feats, outs):
        gold = it.layers["coref"]
        r = conll_f1(gold, o["coref"]) if gold else {"conll_f1": 1.0, "b3_f1": 1.0, "muc_f1": 1.0}
        n_gold = sum(len(c) for c in gold)
        reach = sum(e < f["_W"] for c in gold for _, e in c)
        rows.append({"coref_conll": [r["conll_f1"], 1], "coref_b3": [r["b3_f1"], 1],
                     "coref_muc": [r["muc_f1"], 1], "coref_reach": [reach, n_gold]})
    return {"items": rows, "kinds": {"coref_conll": "acc", "coref_b3": "acc",
                                     "coref_muc": "acc", "coref_reach": "acc"}}


def _pairs(n: int) -> list[tuple[int, int]]:
    return [(h, t) for h in range(n) for t in range(n) if h != t]


OFFSETS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)


def calibrate_offset(model, tok, spec, vocab, device) -> float:
    """The threshold offset that maximises triple-F1 on **Re-DocRED dev**,
    for test. ATLOP's threshold class is learned on the training labels'
    density; a head trained on sparse teacher labels sets it too high for
    Re-DocRED's dense gold."""
    from ..gold.redocred import load_rel_items
    items = load_rel_items("dev")
    names = vocab.labels["relations"]
    feats = [phase2_window(it.tokens, [tuple(x) for x in it.layers["sentence_spans"]], tok,
                           spec, vocab, [[(a, e + 1) for a, e in en["mentions"]]
                                         for en in it.entities],
                           markers=model.rel_markers) for it in items]
    outs = run_phase2(model, feats, tok, device, batch=1)

    def f1(off):
        tp = p = g = 0
        for it, f, o in zip(items, feats, outs):
            keep = f["_ent_keep"]
            pred = {(keep[h], keep[t], r) for h, t, r in
                    relation_triples(o["relation"], _pairs(len(keep)), names, offset=off)}
            gold = {tuple(x) for x in it.layers["rel"]}
            tp += len(pred & gold); p += len(pred); g += len(gold)
        return 2 * tp / max(p + g, 1)
    return max(OFFSETS, key=f1)


def eval_redocred(model, tok, spec, vocab, device, limit, oracle: bool = False) -> dict:
    """Re-DocRED test (documents up to 512 words).

    ``rel_f1``: gold entities given, micro triple-F1 -- the teacher ATLOP's
    protocol. Triples on an entity past the subword limit count as misses.

    ``rel_e2e_f1``: entities found by the model itself -- its NER and coref
    through ``entities.build_entities``, as the corpus built them for the
    teacher. A predicted entity stands for the gold entity sharing the most
    exactly-matching mention spans (none: its triples are false positives),
    so this is a relaxed end-to-end score, not the strict all-mentions one.
    It decodes with the dev-calibrated threshold, as a deployment would.

    ``rel_f1_cal``: gold entities, threshold offset chosen on Re-DocRED dev
    (``calibrate_offset``); test never informs it.

    ``oracle`` (benchmark ``redocred_oracle``) locates the end-to-end loss by
    replacing one stage with gold: ``rel_e2e_goldclust`` -- the model's
    mentions grouped by the gold entity each matches exactly (unmatched ones
    stay alone): what coreference and merging cost; ``rel_e2e_goldment`` --
    gold mention spans and types, merged by the model's coreference: what
    mention finding costs. Entities and scoring otherwise as ``rel_e2e_f1``.
    """
    from ..entities import build_entities
    from ..gold.redocred import load_rel_items
    items = load_rel_items(limit=limit)
    names = vocab.labels["relations"]
    tags = vocab.labels["ner"]
    allowed, start = bio_constraints(tags)
    feats = []
    for it in items:
        sp = [tuple(x) for x in it.layers["sentence_spans"]]
        ents = [[(a, e + 1) for a, e in en["mentions"]] for en in it.entities]
        feats.append(phase2_window(it.tokens, sp, tok, spec, vocab, ents,
                                   markers=model.rel_markers))
    # One document at a time: a 40-entity document is 1,560 pairs, each a
    # 49k-wide bilinear input, and four of them exhaust a 48 GB Mac.
    outs = run_phase2(model, feats, tok, device, batch=1)
    offset = calibrate_offset(model, tok, spec, vocab, device)
    from .data import ENT_TAGS
    ent_allowed, ent_start = bio_constraints(ENT_TAGS)
    rows, feats2, order2 = [], [], []
    for n, (it, f, o) in enumerate(zip(items, feats, outs)):
        keep = f["_ent_keep"]
        gold = {tuple(x) for x in it.layers["rel"]}
        row = {}
        for key, off in (("rel_f1", 0.0), ("rel_f1_cal", offset)):
            pred = relation_triples(o["relation"], _pairs(len(keep)), names, offset=off)
            pred = {(keep[h], keep[t], r) for h, t, r in pred}
            row[key] = [len(pred & gold), len(pred), len(gold)]
        row["rel_reach"] = [sum(h in keep and t in keep for h, t, _ in gold), len(gold)]
        row["_offset"] = offset
        rows.append(row)
        # End to end: the model's own entities -- from the DocRED-style
        # entity-mention head when the model has one, else OntoNotes NER.
        W = f["_W"]
        sp = [tuple(x) for x in it.layers["sentence_spans"]]
        if o["ent"] is not None:
            ner = [[ENT_TAGS[x] for x in viterbi(o["ent"][s:min(e, W)], ent_allowed, ent_start)]
                   if s < W else None for s, e in sp]
        else:
            ner = [[tags[x] for x in viterbi(o["ner"][s:min(e, W)], allowed, start)]
                   if s < W else None for s, e in sp]
        pm = {(s + a, s + b) for (s, _), t in zip(sp, ner) if t for _, a, b in bio_spans_t(t)}
        gm = {(a, e + 1) for en in it.entities for a, e in en["mentions"]}
        rows[-1]["ent_mention_f1"] = [len(pm & gm), len(pm), len(gm)]
        modes = {"rel_e2e_f1": (ner, o["coref"])}
        if oracle:
            # Gold clustering of the model's own mentions.
            pspans = [(t_, s + a, s + b) for (s, _), t in zip(sp, ner) if t
                      for t_, a, b in bio_spans_t(t)]
            ent_of = {}
            for ei, en in enumerate(it.entities):
                for a, e in en["mentions"]:
                    ent_of[(a, e + 1)] = ei
            groups = defaultdict(list)
            for k, (_, a, b) in enumerate(pspans):
                groups[ent_of.get((a, b), ("solo", k))].append((a, b))
            modes["rel_e2e_goldclust"] = ("clusters", list(groups.values()))
            # Gold mentions (with gold types), the model's coreference.
            gtags = []
            for s, e in sp:
                t = ["O"] * (e - s)
                for en in it.entities:
                    for a, b in en["mentions"]:
                        if s <= a and b < e:
                            t[a - s] = f"B-{en['type']}"
                            for i in range(a + 1, b + 1):
                                t[i - s] = f"I-{en['type']}"
                gtags.append(t if s < W else None)
            modes["rel_e2e_goldment"] = (gtags, o["coref"])
            # The model's mentions merged by name only (no coreference chains).
            modes["rel_e2e_nocoref"] = (ner, [])
        for key, (tg, chains) in modes.items():
            if tg == "clusters":
                ents = [c for c in chains]
            else:
                sents, vertex = build_entities(it.tokens, [list(x) for x in sp], tg,
                                               [[list(m) for m in c] for c in chains])
                if vertex is None:
                    continue
                ents = [[(sp[m["sent_id"]][0] + m["pos"][0], sp[m["sent_id"]][0] + m["pos"][1])
                         for m in v] for v in vertex]
            if key == "rel_e2e_f1" and oracle:
                # Grouping errors of the deployed entities: a predicted entity
                # is "split" when another predicted entity shares its gold
                # entity, "merged" when its mentions belong to two gold ones.
                ent_of_m = {(a, e + 1): gi for gi, en in enumerate(it.entities)
                            for a, e in en["mentions"]}
                owners = [{ent_of_m[m] for m in en if m in ent_of_m} for en in ents]
                single = [next(iter(o_)) for o_ in owners if len(o_) == 1]
                rows[n]["ent_split"] = [sum(single.count(g) - 1 for g in set(single)),
                                        max(len(single), 1)]
                rows[n]["ent_merged"] = [sum(len(o_) > 1 for o_ in owners), max(len(ents), 1)]
            if len(ents) < 2:
                continue
            feats2.append(phase2_window(it.tokens, sp, tok, spec, vocab, ents,
                                        markers=model.rel_markers))
            order2.append((n, ents, key))
    outs2 = run_phase2(model, feats2, tok, device, batch=1)
    keys = ["rel_e2e_f1"] + (["rel_e2e_goldclust", "rel_e2e_goldment", "rel_e2e_nocoref"]
                              if oracle else [])
    for r in rows:
        for key in keys:
            r[key] = [0, 0, r["rel_f1"][2]]
        r["rel_e2e_docs"] = [0, 1]          # did the model find two entities?
    for (n, ents, key), f, o in zip(order2, feats2, outs2):
        it, keep = items[n], f["_ent_keep"]
        gold_m = [{(a, e + 1) for a, e in en["mentions"]} for en in it.entities]
        match = {}
        for k, ei in enumerate(keep):
            ov = [len(set(ents[ei]) & g) for g in gold_m]
            m = max(ov) if ov else 0
            match[k] = ov.index(m) if m and ov.count(m) == 1 else None
        pred = relation_triples(o["relation"], _pairs(len(keep)), names, offset=offset)
        pred_n = len({(h, t, r) for h, t, r in pred})
        mapped = {(match[h], match[t], r) for h, t, r in pred
                  if match[h] is not None and match[t] is not None and match[h] != match[t]}
        gold = {tuple(x) for x in it.layers["rel"]}
        rows[n][key] = [len(mapped & gold), pred_n, len(gold)]
        if key == "rel_e2e_f1":
            rows[n]["rel_e2e_docs"] = [1, 1]
    for r in rows:
        r.setdefault("ent_mention_f1", [0, 0, 0])
    kinds = {"rel_f1": "f1", "rel_f1_cal": "f1", "rel_reach": "acc", "ent_mention_f1": "f1",
             "rel_e2e_f1": "f1", "rel_e2e_docs": "acc"}
    if oracle:
        kinds.update(rel_e2e_goldclust="f1", rel_e2e_goldment="f1", rel_e2e_nocoref="f1",
                     ent_split="acc", ent_merged="acc")
        for r in rows:
            r.setdefault("ent_split", [0, 1]); r.setdefault("ent_merged", [0, 1])
    return {"items": rows, "kinds": kinds}


def eval_fewrel(model, tok, spec, vocab, device, limit, per_relation: int = 100) -> dict:
    """FewRel validation (MIT), relation classification read from label texts.

    Each sentence has one head and one tail entity; the model scores the
    head->tail pair against the candidate relations' names and descriptions
    and the highest-scoring one is its answer (the threshold class is not
    consulted: FewRel always holds one of the candidates).

    ``fewrel_zs8``: the 8 validation relations no training source has, 8-way
    -- true zero-shot. ``fewrel_all16``: all 16, 16-way; 8 of them are
    Re-DocRED relations the head was trained on. Bilinear-head checkpoints
    have no label texts to read and are skipped.
    """
    from ..gold import fewrel
    from ..label_bank import load as load_bank
    rels = load_bank()["relations"]
    val = [r for r in rels if r.get("fewrel") == "val"]
    unseen = [r for r in val if not r.get("train", True)]
    by_code = {r["code"]: r["id"] for r in val}
    items = []
    per = defaultdict(int)
    for x in fewrel.load("val"):
        if per[x["code"]] >= (limit or per_relation):
            continue
        per[x["code"]] += 1
        items.append(x)
    rows = []
    for name, cand in (("fewrel_all16", val), ("fewrel_zs8", unseen)):
        cols = [r["id"] for r in cand]
        col_of = {c: j for j, c in enumerate(cols)}
        its = [x for x in items if by_code[x["code"]] in col_of]
        feats = [phase2_window(x["tokens"], [(0, len(x["tokens"]))], tok, spec, vocab,
                               [x["head"], x["tail"]], markers=model.rel_markers) for x in its]
        outs = run_phase2(model, feats, tok, device, batch=8, cols=cols)
        for k, (x, o) in enumerate(zip(its, outs)):
            lg = o["relation"]
            ok = 0
            if len(lg):                          # pair 0 is head -> tail
                ok = int(int(lg[0][1:].argmax()) == col_of[by_code[x["code"]]])
            while len(rows) <= k:
                rows.append({})
            rows[k][name] = [ok, 1]
    for r in rows:
        r.setdefault("fewrel_all16", [0, 0]); r.setdefault("fewrel_zs8", [0, 0])
    return {"items": rows, "kinds": {"fewrel_all16": "acc", "fewrel_zs8": "acc"}}


BENCHMARKS = {"ud": eval_ud, "srl": eval_srl, "srl_full": eval_srl_full,
              "ner": eval_ner, "cls": eval_cls,
              "litbank": eval_litbank, "redocred": eval_redocred,
              "redocred_oracle": lambda *a: eval_redocred(*a, oracle=True),
              "fewrel": eval_fewrel}
PHASE2_BENCHMARKS = ("litbank", "redocred", "redocred_oracle", "fewrel")
# Benchmarks that read label texts: only for label-conditioned heads.
LABEL_BENCHMARKS = ("fewrel",)


# ── aggregation and the bootstrap ───────────────────────────────────
def metric(rows: list[dict], key: str, kind: str, idx=None) -> float:
    a = np.array([r[key] for r in rows], dtype=np.float64)
    if idx is not None:
        a = a[idx]
    s = a.sum(0)
    if kind == "acc":
        return s[0] / max(s[1], 1)
    p = s[0] / max(s[1], 1)
    r = s[0] / max(s[2], 1)
    return 2 * p * r / max(p + r, 1e-12)


def paired_bootstrap(a: list[dict], b: list[dict], key: str, kind: str,
                     n: int = 1000, seed: int = 13) -> tuple[float, float, float]:
    """Difference a - b, with a 95% interval from resampling items jointly."""
    rng = np.random.default_rng(seed)
    A = np.array([r[key] for r in a], dtype=np.float64)
    B = np.array([r[key] for r in b], dtype=np.float64)

    def m(arr, idx):
        s = arr[idx].sum(0)
        if kind == "acc":
            return s[0] / max(s[1], 1)
        p, r = s[0] / max(s[1], 1), s[0] / max(s[2], 1)
        return 2 * p * r / max(p + r, 1e-12)

    full = np.arange(len(A))
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(A), len(A))
        diffs.append(m(A, idx) - m(B, idx))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return m(A, full) - m(B, full), float(lo), float(hi)


REPORTED = [("ud", "pos", "POS acc"), ("ud", "lemma", "lemma acc"),
            ("ud", "morph_f1", "morph feat-F1"), ("ud", "morph_exact", "morph exact"),
            ("ud", "uas", "UAS"), ("ud", "las", "LAS"), ("ud", "tree", "tree rate"),
            ("srl", "srl_f1", "SRL F1"), ("srl", "srl_f1_verb", "SRL F1, VERB predicates"),
            ("srl", "srl_f1_other", "SRL F1, ADJ/NOUN/other"),
            ("srl_full", "pred_id", "predicate ID F1 (all predicates)"),
            ("srl_full", "pred_id_verb", "predicate ID F1, VERB"),
            ("srl_full", "pred_id_aux", "predicate ID F1, AUX"),
            ("srl_full", "pred_id_noun", "predicate ID F1, NOUN"),
            ("srl_full", "pred_id_adj", "predicate ID F1, ADJ"),
            ("srl_full", "role_f1", "roles F1, gold predicates (all)"),
            ("srl_full", "e2e_f1", "end-to-end SRL F1"),
            ("ner", "ner_f1", "NER F1"),
            ("cls", "cls_f1", "CLS micro-F1"), ("cls", "cls_exact", "CLS exact"),
            ("litbank", "coref_conll", "coref CoNLL-F1"),
            ("litbank", "coref_reach", "coref gold mentions inside the window"),
            ("redocred", "rel_f1", "relation triple-F1, gold entities"),
            ("redocred", "rel_f1_cal", "same, threshold calibrated on Re-DocRED dev"),
            ("redocred", "rel_e2e_f1", "relation triple-F1, own entities (relaxed)"),
            ("redocred", "rel_reach", "relation gold triples inside the window"),
            ("redocred", "ent_mention_f1", "entity mentions, exact span (the e2e entities)"),
            ("fewrel", "fewrel_zs8", "FewRel val, 8 never-trained relations (zero-shot acc)"),
            ("fewrel", "fewrel_all16", "FewRel val, all 16 relations (acc)")]
# v5 from DECISIONS.md; its SRL split from its cached bake-off predictions on
# the same items (EVALUATION.md).
V5 = {"pos": 0.987, "uas": 0.949, "las": 0.925, "ner_f1": 0.889, "srl_f1": 0.843,
      "srl_f1_verb": 0.857, "srl_f1_other": 0.742}
# The phase-2 teachers' scores on the same items (DECISIONS.md); v5 had no
# coref or relation heads.
TEACHER = {"coref_conll": 0.699, "rel_f1": 0.790}


def report(results: dict, out: Path, source: str) -> str:
    runs = list(results)
    lines = ["# v6 public-gold evaluation", "",
             f"{time.strftime('%Y-%m-%d %H:%M')} · checkpoints `best.pt` from {source}", "",
             "| benchmark | metric | " + " | ".join(runs) + " | v5 |",
             "|---|---|" + "---|" * len(runs) + "---|"]
    for bench, key, name in REPORTED:
        if not all(bench in results[r] for r in runs):
            continue
        vals = [metric(results[r][bench]["items"], key, results[r][bench]["kinds"][key])
                for r in runs]
        best = max(vals) if not key.endswith(("tree", "reach")) else None
        cells = [f"**{v:.4f}**" if v == best else f"{v:.4f}" for v in vals]
        n = len(results[runs[0]][bench]["items"])
        ref = (f"{V5[key]:.3f}" if key in V5 else
               f"{TEACHER[key]:.3f} (teacher)" if key in TEACHER else "—")
        lines.append(f"| {bench} (n={n}) | {name} | " + " | ".join(cells) + f" | {ref} |")
    lines += ["", "## Paired differences (95% bootstrap interval; a tie if it spans 0)", ""]
    pairs = [(a, b) for i, a in enumerate(runs) for b in runs[i + 1:]]
    lines += ["| metric | " + " | ".join(f"{a} − {b}" for a, b in pairs) + " |",
              "|---|" + "---|" * len(pairs)]
    for bench, key, name in REPORTED:
        if key in ("tree", "coref_reach", "rel_reach") or \
                not all(bench in results[r] for r in runs):
            continue
        cells = []
        for a, b in pairs:
            d, lo, hi = paired_bootstrap(results[a][bench]["items"], results[b][bench]["items"],
                                         key, results[a][bench]["kinds"][key])
            sig = "" if lo <= 0 <= hi else " ✱"
            cells.append(f"{d * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]{sig}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Differences in points (×100). ✱ = interval excludes zero."]
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--limit", type=int, help="items per benchmark (quick look)")
    ap.add_argument("--cls-limit", type=int, default=5000,
                    help="SGD test turns, sampled with a fixed seed (0 = all)")
    ap.add_argument("--device")
    ap.add_argument("--local", type=Path, help="read <dir>/<run>/best.pt instead of the hub")
    ap.add_argument("--zs-offset", type=float,
                    help="label-path threshold offset (calibrate_zs); default: the run's "
                         "calibration.json on the hub, if any, else 0")
    ap.add_argument("--no-xattn", action="store_true",
                    help="v6.2 zero-shot head: switch the label cross-attention off, to measure "
                         "what it adds")
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    out = OUT_DIR / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for run in args.runs.split(","):
        t0 = time.time()
        model, tok, spec, vocab, best = load_run(run, args.repo, device, args.local)
        if hasattr(model, "zs_offset"):
            off = args.zs_offset
            if off is None and not args.local:
                from huggingface_hub import hf_hub_download
                try:
                    off = json.loads(Path(hf_hub_download(args.repo, f"{run}/calibration.json",
                                                          local_dir=str(CKPT_DIR))).read_text())["offset"]
                except Exception:
                    off = 0.0
            model.zs_offset = float(off or 0.0)
            if model.zs_offset:
                print(f"{run}: label-path offset {model.zs_offset:+.2f}", flush=True)
        if args.no_xattn and hasattr(model.relation, "xattn_off"):
            model.relation.xattn_off = True
            print(f"{run}: label cross-attention switched off", flush=True)
        print(f"{run}: best.pt from step {best['step']} (dev {best['value']}), "
              f"SRL head {model.srl_mode}, on {device}",
              flush=True)
        results[run] = {}
        for bench in args.benchmarks.split(","):
            if bench in PHASE2_BENCHMARKS and not model.phase2:
                continue                    # no coref or relation heads
            if bench in LABEL_BENCHMARKS and getattr(model, "rel_kind", "bilinear") not in ("label", "zs"):
                continue                    # no label texts to read
            t1 = time.time()
            lim = args.limit if bench != "cls" else (args.limit or args.cls_limit or None)
            r = BENCHMARKS[bench](model, tok, spec, vocab, device, lim)
            results[run][bench] = r
            summary = ", ".join(f"{k} {metric(r['items'], k, kd):.4f}"
                                for k, kd in r["kinds"].items())
            print(f"  {bench}: {len(r['items'])} items in {time.time() - t1:.0f}s; {summary}",
                  flush=True)
        (out / f"{run}.json").write_text(json.dumps(
            {"run": run, "best": best, "encoder": spec.asdict(), "results": results[run]}))
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        print(f"  {run} done in {time.time() - t0:.0f}s", flush=True)
    source = f"`{args.local}`" if args.local else f"`{args.repo}`"
    print("\n" + report(results, out, source))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

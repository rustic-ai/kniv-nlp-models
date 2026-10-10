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

Sentiment has no public gold here; coref and relations are phase 2.
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
from ..score.metrics import _feats_set, bio_spans
from .data import IGNORE, align, collate, sentence_encodings
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
    srl_mode = (ck.get("model_config") or {}).get("srl_mode", "joint")
    model = V6Model(spec.hf_id, {k: vocab.size(k) for k in vocab.labels},
                    pretrained=False, srl_mode=srl_mode)
    model.load_state_dict(ck["model"])          # strict: a mismatch raises
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    return model, tok, spec, vocab, metric


def window(tokens: list[str], spans: list[tuple[int, int]], tok, spec,
           vocab: Vocab, predicates: list[int] = (), srl_sentences: bool = False) -> dict:
    """A label-free feature dict ``collate`` accepts: the gold is scored
    outside the model, so every label is IGNORE."""
    ids, first, W = align(tokens, tok, spec.max_subwords)
    sent = np.full(W, -1, dtype=np.int32)
    for j, (s, e) in enumerate(spans):
        sent[s:min(e, W)] = j
    blank = np.full(W, IGNORE, dtype=np.int16)
    S = len(spans)
    frames = []
    for p in predicates:
        if p < W:
            s, e = spans[sent[p]]
            frames.append((p, s, np.full(min(e, W) - s, IGNORE, dtype=np.int16)))
    return {"input_ids": np.asarray(ids, dtype=np.int32),
            "first_subword": np.asarray(first, dtype=np.int32),
            "word_ok": np.asarray([f >= 0 for f in first], dtype=bool),
            "sent_of_word": sent, "pos": blank, "morph": blank, "lemma": blank,
            "ner": blank, "dep_head": blank, "dep_rel": blank, "srl_pred": blank,
            "frames": frames,
            "sent_whole": np.array([e <= W for _, e in spans], dtype=bool),
            "cls": np.zeros((S, vocab.size("cls")), dtype=np.int8),
            "cls_ok": np.zeros(S, dtype=bool),
            "sentiment": np.full(S, IGNORE, dtype=np.int16),
            "_W": W, "_S": S,
            **({"srl_sents": sentence_encodings(tokens, spans, {s for _, s, _ in frames},
                                                tok, spec.max_subwords)}
               if srl_sentences else {})}


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


BENCHMARKS = {"ud": eval_ud, "srl": eval_srl, "srl_full": eval_srl_full,
              "ner": eval_ner, "cls": eval_cls}


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
            ("cls", "cls_f1", "CLS micro-F1"), ("cls", "cls_exact", "CLS exact")]
# v5 from DECISIONS.md; its SRL split from its cached bake-off predictions on
# the same items (EVALUATION.md).
V5 = {"pos": 0.987, "uas": 0.949, "las": 0.925, "ner_f1": 0.889, "srl_f1": 0.843,
      "srl_f1_verb": 0.857, "srl_f1_other": 0.742}


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
        best = max(vals) if key != "tree" else None
        cells = [f"**{v:.4f}**" if v == best else f"{v:.4f}" for v in vals]
        n = len(results[runs[0]][bench]["items"])
        lines.append(f"| {bench} (n={n}) | {name} | " + " | ".join(cells)
                     + f" | {V5[key]:.3f} |" if key in V5 else
                     f"| {bench} (n={n}) | {name} | " + " | ".join(cells) + " | — |")
    lines += ["", "## Paired differences (95% bootstrap interval; a tie if it spans 0)", ""]
    pairs = [(a, b) for i, a in enumerate(runs) for b in runs[i + 1:]]
    lines += ["| metric | " + " | ".join(f"{a} − {b}" for a, b in pairs) + " |",
              "|---|" + "---|" * len(pairs)]
    for bench, key, name in REPORTED:
        if key == "tree" or not all(bench in results[r] for r in runs):
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
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    out = OUT_DIR / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for run in args.runs.split(","):
        t0 = time.time()
        model, tok, spec, vocab, best = load_run(run, args.repo, device, args.local)
        print(f"{run}: best.pt from step {best['step']} (dev {best['value']}), "
              f"SRL head {model.srl_mode}, on {device}",
              flush=True)
        results[run] = {}
        for bench in args.benchmarks.split(","):
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

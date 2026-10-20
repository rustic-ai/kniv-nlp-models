"""How much does each head gain from both v7 models? The teacher weights for
the students (docs/v7-students-plan.md, step 1).

    uv run --extra export python -m v6.train.ensemble_heads [--n 400]

Both rc2 models run side by side through the release runtime (``kniv_v7``,
PyTorch graphs). Per head, their output distributions are mixed,
``p = w * p_UltraBERT + (1 - w) * p_DeBERTa`` for w in 0, 0.25, 0.5, 0.75 and
1, then decoded exactly as the runtime decodes and scored on dev:

- POS, morphology, lemma, UAS / LAS: UD English EWT dev
- NER: the gold-filtered v5 NER dev sentences
- SRL end to end: PropBank EWT dev (every predicate)
- dialogue acts: SGD dev turns, with up to 6 preceding turns as context
- coreference: GUM dev (CoNLL F1), each model's calibrated link bias

Relations, spans and FewRel come from ``ensemble_probe``'s dumps (same
weights, ``ensemble_probe score --weights``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..config import RUNS_DIR

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
RUNS = ("ultrabert-380m-v7b8ev", "deberta-v3-large-v7b8ev")
RELEASE = {"ultrabert-380m-v7b8ev": Path("release/ultrabert-380m"),
           "deberta-v3-large-v7b8ev": Path("release/deberta-v3-large")}


def softmax(x, axis=-1):
    x = x.astype(np.float64)
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def mix(pa, pb, w):
    """log of the weighted mixture of two probability arrays."""
    return np.log(np.clip(w * pa + (1 - w) * pb, 1e-30, None)).astype(np.float32)


class Teacher:
    """One rc2 model behind the runtime, with its raw graph outputs exposed."""

    def __init__(self, run: str):
        from kniv_v7 import Kniv
        from v6.export.torch_backend import TorchBackend
        self.k = Kniv(RELEASE[run], backend=TorchBackend.load(run))
        self.g = self.k.backend.g

    def plain(self, tokens, spans):
        from kniv_v7.runtime import align, word_ends
        k = self.k
        ids, first, W = align(tokens, k.tok, k.meta["max_subwords"])
        sent = np.full(W, -1, dtype=np.int64)
        for j, (s, e) in enumerate(spans):
            sent[s:min(e, W)] = j
        first = np.asarray(first, dtype=np.int64)
        wm = first >= 0
        x = {"input_ids": np.asarray([ids], dtype=np.int64),
             "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
             "first_subword": np.maximum(first, 0)[None], "word_end": word_ends(first, len(ids))[None],
             "word_mask": wm[None], "sent_of_word": sent[None],
             "sent_ids": np.arange(len(spans), dtype=np.int64)}
        o = dict(zip(k.meta["graphs"]["plain"]["outputs"], k.run("plain", **x)))
        o.update(_W=W, _sent=sent, _wm=wm)
        return o

    @torch.no_grad()
    def dep_labels(self, o):
        """Label probabilities for every candidate head: ``[W, W+1, labels]``."""
        rel = self.g["dep_rel"].rel
        s = rel.pairwise(torch.from_numpy(o["rel_dep"]), torch.from_numpy(o["rel_head"]))[0]
        return softmax(s.numpy())

    def roles(self, o, preds):
        """Per predicate, role probabilities over the window ``[W, tags]`` (as
        ``Kniv.srl`` lays them out before decoding)."""
        from kniv_v7.runtime import NEG
        tags = self.k.labels["srl"]
        sent, wm, W = o["_sent"], o["_wm"], o["_W"]
        s = o["srl_words"][0]
        same = [(sent == sent[p]) & wm & (sent[p] >= 0) for p in preds]
        start = [int(np.argmax(m)) for m in same]
        end = [len(m) - 1 - int(np.argmax(m[::-1])) for m in same]
        n = [e - a + 1 for a, e in zip(start, end)]
        L = max(n)
        words = np.zeros((len(preds), L, s.shape[-1]), dtype=np.float32)
        mask = np.zeros((len(preds), L), dtype=bool)
        for f, (a, c) in enumerate(zip(start, n)):
            words[f, :c] = s[a:a + c]
            mask[f, :c] = True
        lg = self.k.run("srl_roles", words=words, pred=np.asarray(
            [p - a for p, a in zip(preds, start)], dtype=np.int64), mask=mask)[0]
        out = []
        for f in range(len(preds)):
            role = np.full((W, len(tags)), NEG, dtype=np.float32)
            role[start[f] + np.arange(n[f])] = lg[f, :n[f]]
            role[~same[f]] = NEG
            out.append(softmax(role))
        return out

    def antecedents(self, o):
        """Per word, ``{candidate word (-1 = none): probability}`` with the
        model's calibrated link bias."""
        sc = o["coref_scores"][0].astype(np.float64).copy()
        sc[:, 0] += self.k.coref_bias
        p = softmax(sc)
        cand = o["coref_cand"][0]
        out = []
        for w in range(o["_W"]):
            d = {-1: p[w, 0]}
            for j in range(cand.shape[1]):
                if sc[w, j + 1] > -5e3:
                    d[int(cand[w, j])] = d.get(int(cand[w, j]), 0.0) + p[w, j + 1]
            out.append(d)
        return out


# ── tasks ────────────────────────────────────────────────────────────
def ud(T, n):
    from ..gold.ud_ewt import load_ud_items
    from kniv_v7.runtime import apply_script, decode_trees
    items = load_ud_items("dev", limit=n)
    L = T[0].k.labels
    c = defaultdict(lambda: np.zeros(2))
    for it in items:
        sp = [(0, len(it.tokens))]
        oa, ob = (t.plain(it.tokens, sp) for t in T)
        W = oa["_W"]
        if W != ob["_W"] or W != len(it.tokens):
            continue
        g = it.layers
        la, lb = T[0].dep_labels(oa), T[1].dep_labels(ob)
        for w in WEIGHTS:
            pos = mix(softmax(oa["pos"][0]), softmax(ob["pos"][0]), w).argmax(-1)
            mor = mix(softmax(oa["morph"][0]), softmax(ob["morph"][0]), w).argmax(-1)
            lem = mix(softmax(oa["lemma"][0]), softmax(ob["lemma"][0]), w).argmax(-1)
            arc = mix(softmax(oa["arc"][0]), softmax(ob["arc"][0]), w)
            head = decode_trees(arc, oa["_sent"], oa["_wm"])
            lab = mix(la[np.arange(W), head], lb[np.arange(W), head], w).argmax(-1)
            c[("pos", w)] += [sum(L["pos"][x] == y for x, y in zip(pos, g["pos"])), W]
            c[("morph", w)] += [sum(L["morph"][x] == y for x, y in zip(mor, g["morph"])), W]
            c[("lemma", w)] += [sum(apply_script(t, L["lemma"][x]) == y
                                    for t, x, y in zip(it.tokens, lem, g["lemma"])), W]
            c[("uas", w)] += [sum(int(h) == gh for h, gh in zip(head, g["dep"]["heads"])), W]
            c[("las", w)] += [sum(int(h) == gh and L["dep"][x] == gr for h, x, gh, gr
                                  in zip(head, lab, g["dep"]["heads"], g["dep"]["rels"])), W]
    return {f"{k}@{w}": round(v[0] / max(v[1], 1), 4) for (k, w), v in c.items()}


def f1(c):
    return round(2 * c[0] / max(c[1] + c[2], 1), 4)


def ner(T, n):
    from ..train.golddata import v5ner_rows
    from kniv_v7.entities import bio_spans
    from kniv_v7.runtime import bio_constraints, viterbi
    rows = v5ner_rows("dev")[:n]
    tags = T[0].k.labels["ner"]
    al, st = bio_constraints(tags)
    c = defaultdict(lambda: np.zeros(3))
    for r in rows:
        toks = r["tokens"]
        sp = [(0, len(toks))]
        oa, ob = (t.plain(toks, sp) for t in T)
        if oa["_W"] != ob["_W"] or oa["_W"] != len(toks):
            continue
        gold = set(bio_spans(r["ner"]))
        for w in WEIGHTS:
            lp = mix(softmax(oa["ner"][0]), softmax(ob["ner"][0]), w)
            pred = set(bio_spans([tags[k] for k in viterbi(lp, al, st)]))
            c[w] += [len(pred & gold), len(pred), len(gold)]
    return {f"ner@{w}": f1(v) for w, v in c.items()}


def srl(T, n):
    from kniv_v7.entities import bio_spans
    from kniv_v7.runtime import bio_constraints, viterbi
    rows = json.loads(Path("data/prepared/propbank-full/dev.json").read_text())[:n]
    tags = T[0].k.labels["srl"]
    al, st = bio_constraints(tags)
    v = tags.index("V")
    c = defaultdict(lambda: np.zeros(3))
    for r in rows:
        toks = r["words"]
        sp = [(0, len(toks))]
        oa, ob = (t.plain(toks, sp) for t in T)
        W = oa["_W"]
        if W != ob["_W"] or W != len(toks):
            continue
        gold = {(f["predicate_idx"],) + s for f in r["frames"]
                for s in bio_spans(f["srl_tags"]) if s[0] != "V"}
        pa, pb = softmax(oa["srl_pred"][0])[:, 1], softmax(ob["srl_pred"][0])[:, 1]
        for w in WEIGHTS:
            preds = [int(p) for p in np.nonzero((w * pa + (1 - w) * pb > 0.5) & oa["_wm"])[0]]
            pred = set()
            if preds:
                ra, rb = T[0].roles(oa, preds), T[1].roles(ob, preds)
                for p, x, y in zip(preds, ra, rb):
                    lp = mix(x, y, w)
                    keep = np.ones_like(lp, dtype=bool)
                    keep[:, v] = False
                    keep[p, :] = False
                    keep[p, v] = True
                    pred |= {(p,) + s for s in bio_spans([tags[k] for k in viterbi(lp, al, st, keep)])
                             if s[0] != "V"}
            c[w] += [len(pred & gold), len(pred), len(gold)]
    return {f"srl_e2e@{w}": f1(v_) for w, v_ in c.items()}


def cls(T, n):
    from ..gold.sgd_cls import load_gold
    from .gold_eval import CLS_SCORED, _WORD
    gold = load_gold("validation")
    dialogues = defaultdict(list)
    for g in gold:
        dialogues[g["dialogue_id"]].append(g)
    turns = [g for d in sorted(dialogues) for g in sorted(dialogues[d], key=lambda x: x["turn_idx"])]
    rng = np.random.default_rng(13)
    pick = sorted(rng.choice(len(turns), size=min(n, len(turns)), replace=False).tolist())
    labels = T[0].k.labels["cls"]
    c = defaultdict(lambda: np.zeros(3))
    for i in pick:
        g = turns[i]
        prev = [t for t in dialogues[g["dialogue_id"]] if t["turn_idx"] < g["turn_idx"]][-6:]
        sents = [_WORD.findall(t["text"]) for t in prev] + [_WORD.findall(g["text"])]
        sents = [s for s in sents[:-1] if s] + [sents[-1] or ["."]]
        toks, sp = [], []
        for s in sents:
            sp.append((len(toks), len(toks) + len(s)))
            toks += s
        oa, ob = (t.plain(toks, sp) for t in T)
        if sp[-1][1] > min(oa["_W"], ob["_W"]):
            continue
        sa = 1 / (1 + np.exp(-oa["cls"][0][-1].astype(np.float64)))
        sb = 1 / (1 + np.exp(-ob["cls"][0][-1].astype(np.float64)))
        gs = set(g["cls"])
        for w in WEIGHTS:
            p = {labels[k] for k, x in enumerate(w * sa + (1 - w) * sb) if x > 0.5}
            c[w] += [sum(l in p and l in gs for l in CLS_SCORED), sum(l in p for l in CLS_SCORED),
                     sum(l in gs for l in CLS_SCORED)]
    return {f"cls@{w}": f1(v) for w, v in c.items()}


def coref(T, n):
    from ..score.coref import conll_f1
    from .golddata import gum_rows
    from kniv_v7.runtime import link_clusters
    rows = gum_rows("dev")[:n]
    res = defaultdict(list)
    for r in rows:
        toks, sp = r["tokens"], [tuple(x) for x in r["sentence_spans"]]
        oa, ob = (t.plain(toks, sp) for t in T)
        W = min(oa["_W"], ob["_W"])
        aa, ab = T[0].antecedents(oa), T[1].antecedents(ob)
        gold = [[tuple(m) for m in cl if m[1] < W] for cl in r["coref"]]
        gold = [cl for cl in gold if len(cl) >= 2]
        for w in WEIGHTS:
            ante = {}
            for i in range(W):
                d = defaultdict(float)
                for k, p in aa[i].items():
                    d[k] += w * p
                for k, p in ab[i].items():
                    d[k] += (1 - w) * p
                best = max(d, key=d.get)
                if best >= 0 and best < W:
                    ante[i] = best
            clusters = link_clusters(ante)
            heads = [x for cl in clusters for x in cl]
            span = {}
            if heads:
                h = np.asarray(heads, dtype=np.int64)
                outs = [t.k.run("coref_spans", coref_words=o["coref_words"][:, :t_W], heads=h,
                                sent_of_word=o["_sent"][None, :t_W])
                        for t, o, t_W in ((T[0], oa, oa["_W"]), (T[1], ob, ob["_W"]))]
                (lga, pos), (lgb, _) = outs
                lg = np.log(np.clip(w * softmax(lga, 1) + (1 - w) * softmax(lgb, 1), 1e-30, None))
                st = np.where(pos > h[:, None], -1e9, lg[..., 0]).argmax(1)
                en = np.where(pos < h[:, None], -1e9, lg[..., 1]).argmax(1)
                span = {x: (int(pos[i, st[i]]), int(pos[i, en[i]])) for i, x in enumerate(heads)}
            pred = [[span[x] for x in cl] for cl in clusters]
            res[w].append(conll_f1(gold, pred)["conll_f1"] if gold else 1.0)
    return {f"coref@{w}": round(float(np.mean(v)), 4) for w, v in res.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--tasks", default="ud,ner,srl,cls,coref")
    args = ap.parse_args()
    sys.path.insert(0, str(RELEASE[RUNS[0]]))
    torch.set_grad_enabled(False)
    T = [Teacher(r) for r in RUNS]
    out = {"weights": "p = w * UltraBERT + (1 - w) * DeBERTa"}
    for task in args.tasks.split(","):
        t0 = time.time()
        r = {"ud": ud, "ner": ner, "srl": srl, "cls": cls, "coref": coref}[task](
            T, args.n if task != "coref" else 999)
        out.update(r)
        print(task, json.dumps(r), f"[{time.time() - t0:.0f}s]", flush=True)
    path = RUNS_DIR / "ensemble-probe" / f"heads-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

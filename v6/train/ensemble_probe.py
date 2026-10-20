"""Do the two v7 models make the same mistakes? And would a teacher made of
both beat either?

    uv run python -m v6.train.ensemble_probe dump --run ultrabert-380m-v7b8ev --local data/v6-eval/checkpoints
    uv run python -m v6.train.ensemble_probe dump --run deberta-v3-large-v7b8ev --local data/v6-eval/checkpoints
    uv run python -m v6.train.ensemble_probe score --runs ultrabert-380m-v7b8ev,deberta-v3-large-v7b8ev

``dump`` writes one model's raw scores (one process per model):

- **Re-DocRED dev, gold entities**: every ordered pair's relation logits.
- **FewRel zero-shot**: the 15 unseen relations' logits per instance (a
  label's logit does not depend on which other labels are listed, so m = 5
  and 10 are read from the same logits).
- **CrossNER (5 domains) and Few-NERD**: every candidate span's type margins.

``score`` compares the two models and their average, decoded as deployed:

- Re-DocRED is cross-fitted on dev: documents split in two halves, threshold
  offsets (global, then per relation) chosen on one and scored on the other,
  then swapped; the two halves' counts are summed. Test is never read.
- FewRel: averaged log-probabilities over the candidate relations, macro F1
  as ``zs_protocol``.
- Spans: averaged margins, the offset chosen on CrossNER politics as the
  scorecard does, CrossNER reported as the mean of the other four.

Error overlap: for each gold item, which model got it (one, both, neither);
the share of one model's errors the other gets right is the room a
combined teacher has.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from ..config import RUNS_DIR
from .gold_eval import (HF_REPO, OFFSETS, REL_OFFSETS, MIN_DEV_FACTS, _pairs, load_run,
                        phase2_window, run_phase2)

OUT = RUNS_DIR / "ensemble-probe"
SEEDS = 5


# ── dump ─────────────────────────────────────────────────────────────
def dump_redocred(model, tok, spec, vocab, device) -> list[dict]:
    from ..gold.redocred import load_rel_items
    items = load_rel_items("dev")
    feats = [phase2_window(it.tokens, [tuple(x) for x in it.layers["sentence_spans"]], tok, spec,
                           vocab, [[(a, e + 1) for a, e in en["mentions"]] for en in it.entities],
                           markers=model.rel_markers) for it in items]
    outs = run_phase2(model, feats, tok, device, batch=1)
    rows = []
    for it, f, o in zip(items, feats, outs):
        keep = list(f["_ent_keep"])
        pairs = [(keep[h], keep[t]) for h, t in _pairs(len(keep))]
        lg = o["relation"]
        rows.append({"pairs": pairs, "margin": (lg[:, 1:] - lg[:, :1]).astype(np.float16),
                     "gold": sorted({tuple(x) for x in it.layers["rel"]}),
                     "n_ent": len(it.entities)})
    return rows


def fewrel_items() -> tuple[list[dict], list[int]]:
    from ..gold import fewrel
    from ..label_bank import load as load_bank
    unseen = [r for r in load_bank()["relations"] if not r.get("train", True)]
    code = {r["code"]: r["id"] for r in unseen}
    per: dict[int, int] = {}
    items = []
    for x in fewrel.load("train") + fewrel.load("val"):
        rid = code.get(x["code"])
        if rid is None or per.get(rid, 0) >= 700:
            continue
        per[rid] = per.get(rid, 0) + 1
        items.append({"tokens": x["tokens"], "head": x["head"], "tail": x["tail"], "rel": rid})
    return items, sorted(code.values())


def dump_fewrel(model, tok, spec, vocab, device) -> dict:
    items, cols = fewrel_items()
    feats = [phase2_window(x["tokens"], [(0, len(x["tokens"]))], tok, spec, vocab,
                           [x["head"], x["tail"]], markers=model.rel_markers) for x in items]
    outs = run_phase2(model, feats, tok, device, batch=8, cols=cols)
    lg = np.stack([o["relation"][0, 1:] if len(o["relation"]) else np.full(len(cols), -1e4)
                   for o in outs]).astype(np.float32)
    return {"cols": cols, "gold": [x["rel"] for x in items], "logits": lg}


@torch.no_grad()
def span_margins(model, tok, spec, vocab, device, items: list[dict], labels: list[str],
                 batch: int = 16) -> list[dict]:
    """Per sentence, every candidate span ``(start, end)`` and its margin over
    the threshold for each type, in ``labels`` order (``kniv_predict``'s
    batching, without the decoding)."""
    from .data import blank_window, collate
    from .kdata import TypeCollator, span_targets
    bank = {"entity_types": [{"id": i, "name": n, "description": ""} for i, n in enumerate(labels)]}
    coll = TypeCollator(tok, bank, train=False)
    lid = {n: i for i, n in enumerate(labels)}
    feats = []
    for it in items:
        sp = [(0, len(it["tokens"]))]
        f = blank_window(it["tokens"], sp, tok, spec.max_subwords, vocab.size("cls"))
        f.update(coref=None, rel=None,
                 ents=span_targets([(lid[t], s, e) for t, s, e in it["spans"] if t in lid],
                                   f["_W"], f["word_ok"], list(range(len(labels))), False))
        feats.append(f)
    out: list[dict] = [{} for _ in items]
    order = sorted(range(len(items)), key=lambda i: len(feats[i]["input_ids"]))
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        b = coll(collate([feats[i] for i in idx], tok.pad_token_id))
        b = {n: v.to(device) for n, v in b.items() if hasattr(v, "to")}
        lg = model(b, relation_only=True)["span"].float().cpu()
        cols = b["type_label_cols"].cpu().numpy()
        m = np.full((len(lg), len(labels)), -1e4, dtype=np.float32)
        m[:, cols] = (lg[:, 1:] - lg[:, :1]).numpy()
        sb, ss, se = (b[n].cpu().numpy() for n in ("span_batch", "span_start", "span_end"))
        for j, i in enumerate(idx):
            sel = sb == j
            out[i] = {"spans": np.stack([ss[sel], se[sel]], 1).astype(np.int32),
                      "margin": m[sel].astype(np.float16)}
    return out


def span_sets() -> dict:
    from .span_eval import DOMAINS, load_crossner, load_fewnerd
    sets = {"fewnerd": load_fewnerd(3000)}
    for d in DOMAINS:
        sets[f"crossner_{d}"] = load_crossner(d)
    return sets


def dump_spans(model, tok, spec, vocab, device) -> dict:
    out = {}
    for name, items in span_sets().items():
        labels = sorted({t for it in items for t, _, _ in it["spans"]})
        out[name] = {"labels": labels,
                     "gold": [{(a, b, t) for t, a, b in it["spans"]} for it in items],
                     "items": span_margins(model, tok, spec, vocab, device, items, labels)}
        print(f"  {name}: {len(items)} sentences", flush=True)
    return out


def dump(args) -> int:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    model, tok, spec, vocab, _ = load_run(args.run, args.repo, device, args.local)
    path = OUT / f"{args.run}.pkl"
    res = pickle.loads(path.read_bytes()) if path.exists() else {}
    for part, fn in (("redocred", dump_redocred), ("fewrel", dump_fewrel), ("spans", dump_spans)):
        if part in res or part not in args.parts.split(","):
            continue
        t0 = time.time()
        res[part] = fn(model, tok, spec, vocab, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(res))             # after each part: a crash keeps the rest
        print(f"{args.run} {part}: {time.time() - t0:.0f}s", flush=True)
    return 0


# ── score: Re-DocRED ─────────────────────────────────────────────────
def candidates(docs: list[dict], K: int, top: int = 4) -> tuple[list[np.ndarray], np.ndarray]:
    """Per relation, every (doc, pair) where it is among the pair's ``top``:
    ``[margin, is_gold, doc, head, tail]``; and the gold count per relation."""
    marg = [[] for _ in range(K)]
    gold_n = np.zeros(K)
    for d, doc in enumerate(docs):
        gold = doc["gold_idx"]
        for _, _, k in gold:
            gold_n[k] += 1
        for (h, t), m in zip(doc["pairs"], doc["margin"]):
            for k in np.argsort(-m)[:top]:
                marg[k].append((float(m[k]), (h, t, int(k)) in gold, d, h, t))
    return [np.array(x, dtype=np.float64).reshape(-1, 5) for x in marg], gold_n


def choose_offsets(docs: list[dict], K: int) -> np.ndarray:
    """``calibrate_offsets`` on the given documents: the global offset, then
    per relation by coordinate ascent on micro-F1."""
    marg, gold_n = candidates(docs, K)
    G = gold_n.sum()

    def counts(k, off):
        m = marg[k]
        hit = m[:, 0] > -off
        return float((hit & (m[:, 1] > 0)).sum()), float(hit.sum())

    f1 = lambda tp, p: 2 * tp / max(p + G, 1)
    tab = {k: {off: counts(k, off) for off in REL_OFFSETS} for k in range(K)}
    glob = max(OFFSETS, key=lambda off: f1(*map(sum, zip(*(counts(k, off) for k in range(K))))))
    per = np.full(K, glob, dtype=np.float64)
    cur = {k: counts(k, glob) for k in range(K)}
    for _ in range(3):
        for k in range(K):
            if gold_n[k] < MIN_DEV_FACTS:
                continue
            TP = sum(c[0] for c in cur.values()) - cur[k][0]
            P = sum(c[1] for c in cur.values()) - cur[k][1]
            best = max(REL_OFFSETS, key=lambda off: f1(TP + tab[k][off][0], P + tab[k][off][1]))
            per[k], cur[k] = best, tab[k][best]
    return per


def predict(docs: list[dict], off: np.ndarray, top: int = 4) -> list[set]:
    out = []
    for doc in docs:
        p = set()
        for (h, t), m in zip(doc["pairs"], doc["margin"]):
            for k in np.argsort(-m)[:top]:
                if m[k] + off[k] > 0:
                    p.add((h, t, int(k)))
        out.append(p)
    return out


def combine_docs(a: list[dict], b: list[dict], wa: float = 0.5) -> list[dict]:
    """Average margins pair by pair. A pair one model could not reach (an
    entity past its subword limit) keeps the other model's margin."""
    out = []
    for x, y in zip(a, b):
        my = dict(zip(map(tuple, y["pairs"]), y["margin"]))
        mx = dict(zip(map(tuple, x["pairs"]), x["margin"]))
        pairs = sorted(set(mx) | set(my))
        marg = np.stack([wa * mx[p].astype(np.float32) + (1 - wa) * my[p].astype(np.float32)
                         if p in mx and p in my else (mx.get(p, my.get(p))).astype(np.float32)
                         for p in pairs]) if pairs else np.zeros((0, x["margin"].shape[1]))
        out.append({**x, "pairs": pairs, "margin": marg})
    return out


def score_redocred(A: list[dict], B: list[dict], names: list[str]) -> dict:
    idx = {r: k for k, r in enumerate(names)}
    for docs in (A, B):
        for d in docs:
            d["gold_idx"] = {(h, t, idx[r]) for h, t, r in d["gold"] if r in idx}
            d["margin"] = d["margin"].astype(np.float32)
    K = len(names)
    halves = [np.arange(len(A)) % 2 == 0, np.arange(len(A)) % 2 == 1]
    systems = {"A": A, "B": B, "avg": combine_docs(A, B)}
    preds = {s: [None] * len(A) for s in systems}
    for cal, ev in (halves, halves[::-1]):
        for s, docs in systems.items():
            off = choose_offsets([d for d, c in zip(docs, cal) if c], K)
            for i in np.where(ev)[0]:
                preds[s][i] = predict([docs[i]], off)[0]
    res = {}
    for s in systems:
        tp = sum(len(p & d["gold_idx"]) for p, d in zip(preds[s], A))
        np_ = sum(len(p) for p in preds[s])
        ng = sum(len(d["gold_idx"]) for d in A)
        res[s] = {"f1": 2 * tp / max(np_ + ng, 1), "p": tp / max(np_, 1), "r": tp / max(ng, 1)}
    # Overlap: gold facts each model finds, false facts each makes.
    both = only_a = only_b = none = fp_a = fp_b = fp_both = 0
    for pa, pb, d in zip(preds["A"], preds["B"], A):
        g = d["gold_idx"]
        for f in g:
            ia, ib = f in pa, f in pb
            both += ia and ib; only_a += ia and not ib; only_b += ib and not ia
            none += not ia and not ib
        fa, fb = pa - g, pb - g
        fp_a += len(fa); fp_b += len(fb); fp_both += len(fa & fb)
    res["overlap"] = {"gold": both + only_a + only_b + none, "both": both, "only_A": only_a,
                      "only_B": only_b, "neither": none, "false_A": fp_a, "false_B": fp_b,
                      "false_shared": fp_both}
    return res


# ── score: FewRel ────────────────────────────────────────────────────
def score_fewrel(A: dict, B: dict) -> dict:
    from .zs_protocol import macro
    cols = A["cols"]
    assert cols == B["cols"] and A["gold"] == B["gold"]
    gold = np.asarray(A["gold"])
    rng = np.random.default_rng(13)
    splits = {f"m={m}": [sorted(rng.choice(cols, size=m, replace=False).tolist())
                         for _ in range(SEEDS)] for m in (5, 10)}
    splits[f"m={len(cols)}"] = [cols]
    pos = {c: j for j, c in enumerate(cols)}

    def logp(lg, sel):
        x = lg[:, sel]
        return x - np.logaddexp.reduce(x, axis=1, keepdims=True)
    res = {}
    for name, seeds in splits.items():
        f = {"A": [], "B": [], "avg": []}
        ov = np.zeros(4)
        for cs in seeds:
            sel = [pos[c] for c in cs]
            rows = np.isin(gold, cs)
            la, lb = logp(A["logits"][rows], sel), logp(B["logits"][rows], sel)
            g = gold[rows]
            pa, pb, pe = (np.asarray(cs)[x.argmax(1)] for x in (la, lb, (la + lb) / 2))
            for s, p in (("A", pa), ("B", pb), ("avg", pe)):
                f[s].append(macro(g.tolist(), p.tolist(), cs)["f1"])
            ca, cb = pa == g, pb == g
            ov += [(ca & cb).sum(), (ca & ~cb).sum(), (~ca & cb).sum(), (~ca & ~cb).sum()]
        res[name] = {s: float(np.mean(v)) for s, v in f.items()}
        res[name]["overlap"] = dict(zip(("both", "only_A", "only_B", "neither"), ov.tolist()))
    return res


# ── score: spans ─────────────────────────────────────────────────────
def decode(item: dict, offset: float) -> set:
    """``decode_spans`` on one sentence's margins."""
    m = item["margin"].astype(np.float32) - offset
    if not len(m):
        return set()
    best, j = m.max(1), m.argmax(1)
    taken = np.zeros(int(item["spans"][:, 1].max()) + 1, dtype=bool)
    out = set()
    for i in np.argsort(-best, kind="stable"):
        if best[i] <= 0:
            break
        s, e = item["spans"][i]
        if taken[s:e].any():
            continue
        taken[s:e] = True
        out.add((int(s), int(e), int(j[i])))
    return out


def combine_items(a: list[dict], b: list[dict]) -> list[dict]:
    out = []
    for x, y in zip(a, b):
        if not len(x.get("spans", [])) or not len(y.get("spans", [])):
            out.append(x if len(x.get("spans", [])) else y)
            continue
        ky = {(int(s), int(e)): k for k, (s, e) in enumerate(y["spans"])}
        keep = [k for k, (s, e) in enumerate(x["spans"]) if (int(s), int(e)) in ky]
        sp = x["spans"][keep]
        m = (x["margin"][keep].astype(np.float32)
             + y["margin"][[ky[(int(s), int(e))] for s, e in sp]].astype(np.float32)) / 2
        out.append({"spans": sp, "margin": m})
    return out


def span_f1(items: list[dict], gold: list[set], labels: list[str], offset: float) -> tuple:
    pred = [{(s, e, labels[t]) for s, e, t in decode(it, offset)} for it in items]
    tp = sum(len(g & p) for g, p in zip(gold, pred))
    np_, ng = sum(len(p) for p in pred), sum(len(g) for g in gold)
    pr, rc = tp / max(np_, 1), tp / max(ng, 1)
    return 2 * pr * rc / max(pr + rc, 1e-12), pred


SPAN_OFFSETS = np.round(np.arange(-5.0, 1.01, 0.5), 2)


def score_spans(A: dict, B: dict) -> dict:
    systems = {"A": {n: v["items"] for n, v in A.items()}, "B": {n: v["items"] for n, v in B.items()},
               "avg": {n: combine_items(A[n]["items"], B[n]["items"]) for n in A}}
    res = {}
    for s, sets in systems.items():
        cal = "crossner_politics"
        off = max(SPAN_OFFSETS, key=lambda o: span_f1(sets[cal], A[cal]["gold"], A[cal]["labels"], o)[0])
        res[s] = {"offset": float(off)}
        for n in sets:
            res[s][n], _ = span_f1(sets[n], A[n]["gold"], A[n]["labels"], off)
        held = [n for n in sets if n.startswith("crossner_") and n != cal]
        res[s]["crossner_mean4"] = float(np.mean([res[s][n] for n in held]))
    # Overlap of gold spans found, at each model's own offset.
    ov = {}
    for n in A:
        _, pa = span_f1(A[n]["items"], A[n]["gold"], A[n]["labels"], res["A"]["offset"])
        _, pb = span_f1(B[n]["items"], A[n]["gold"], A[n]["labels"], res["B"]["offset"])
        c = np.zeros(4)
        for g, x, y in zip(A[n]["gold"], pa, pb):
            for sp in g:
                ia, ib = sp in x, sp in y
                c += [ia and ib, ia and not ib, ib and not ia, not ia and not ib]
        ov[n] = dict(zip(("both", "only_A", "only_B", "neither"), c.tolist()))
    res["overlap"] = ov
    return res


def score(args) -> int:
    ra, rb = args.runs.split(",")
    A = pickle.loads((OUT / f"{ra}.pkl").read_bytes())
    B = pickle.loads((OUT / f"{rb}.pkl").read_bytes())
    from ..label_bank import load as load_bank
    names = [r["name"] for r in load_bank()["relations"] if r["source"] == "redocred"]
    res = {"A": ra, "B": rb}
    if "redocred" in A and "redocred" in B:
        res["redocred_dev_crossfit"] = score_redocred(A["redocred"], B["redocred"], names)
    if "fewrel" in A and "fewrel" in B:
        res["fewrel"] = score_fewrel(A["fewrel"], B["fewrel"])
    if "spans" in A and "spans" in B:
        res["spans"] = score_spans(A["spans"], B["spans"])
    out = OUT / f"score-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"written to {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--run", required=True)
    d.add_argument("--repo", default=HF_REPO)
    d.add_argument("--local", type=Path)
    d.add_argument("--parts", default="redocred,fewrel,spans")
    d.add_argument("--device")
    s = sub.add_parser("score")
    s.add_argument("--runs", required=True, help="A,B")
    args = ap.parse_args()
    return dump(args) if args.cmd == "dump" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())

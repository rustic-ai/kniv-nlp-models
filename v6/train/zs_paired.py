"""Paired zero-shot relation comparison on FewRel's unseen relations.

    uv run python -m v6.train.zs_paired --runs ultrabert-380m-v7b8ev,ultrabert-380m-v8e-w1half \
        --base ultrabert-380m-v7b8ev

Every instance of the label bank's 15 unseen relations (both FewRel splits,
700 each) is scored once against all 15 label texts, and the logits are
cached per run and label-text hash. Every setting is read from that one
matrix by restricting the argmax to a subset of the columns (each column's
logit depends only on its own label text, which ``--check`` verifies against
a direct run on a subset):

- ``m15``: 15-way, macro F1 and accuracy (the published m = 15 setting);
- ``zs8``: 8-way among the 8 FewRel-validation relations no training source
  has -- the clean zero-shot test (``gold_eval fewrel_zs8`` at 700 per
  relation instead of 100);
- ``ho7``: 7-way among the 7 FewRel-train relations held out of the bank;
- ``m5`` / ``m10``: macro F1 averaged over ``--draws`` fixed random subsets
  (one seed's spread at m = 5 is about +-0.15 F1, so 5 seeds cannot carry a
  1-point gate).

Each run is compared with ``--base`` on the same instances: the difference
with a 95% interval from a bootstrap that resamples instances within each
relation (the same resample for both runs, and for m5 / m10 every draw is
recomputed on it). Argmax over unseen columns ignores the calibration
offsets: they are the same for every unseen relation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..config import RUNS_DIR
from .gold_eval import HF_REPO, load_run, phase2_window, run_phase2
from .zs_protocol import load_items

CACHE = RUNS_DIR / "zs-protocol" / "logits"


def label_hash(unseen: list[dict]) -> str:
    txt = json.dumps([(r["id"], r["name"], r.get("description", "")) for r in unseen])
    return hashlib.sha1(txt.encode()).hexdigest()[:10]


def score(run: str, items: list[dict], ids: list[int], repo: str, device, local, tag: str,
          batch: int) -> np.ndarray:
    path = CACHE / f"{run}-{tag}.npy"
    if path.exists():
        return np.load(path)
    t0 = time.time()
    model, tok, spec, vocab, _ = load_run(run, repo, device, local)
    feats = [phase2_window(x["tokens"], [(0, len(x["tokens"]))], tok, spec, vocab,
                           [x["head"], x["tail"]], markers=model.rel_markers) for x in items]
    L = np.full((len(items), len(ids)), -np.inf, dtype=np.float32)
    step = 2000                         # chunks keep MPS graph caches small
    for k in range(0, len(feats), step):
        outs = run_phase2(model, feats[k:k + step], tok, device, batch=batch, cols=ids)
        for j, o in enumerate(outs):
            if len(o["relation"]):
                L[k + j] = o["relation"][0][1:]
        print(f"  {run}: {min(k + step, len(feats))}/{len(feats)} [{time.time() - t0:.0f}s]",
              flush=True)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, L)
    return L


def macro_f1(W: np.ndarray, L: np.ndarray, gold: np.ndarray, cols: list[int]) -> tuple:
    """Macro F1 and accuracy over the instances of ``cols`` (indices into
    L's columns), for every row of the weight matrix W [B, N]."""
    keep = np.isin(gold, cols)
    sub = L[keep][:, cols]
    pred = np.asarray(cols)[sub.argmax(1)]
    g = gold[keep]
    Wk = W[:, keep]
    C = len(cols)
    P = np.zeros((keep.sum(), C), np.float32)
    G = np.zeros_like(P)
    P[np.arange(len(pred)), np.searchsorted(cols, pred)] = 1
    G[np.arange(len(g)), np.searchsorted(cols, g)] = 1
    tp, np_, ng = Wk @ (P * G), Wk @ P, Wk @ G
    pr, rc = tp / np.maximum(np_, 1e-9), tp / np.maximum(ng, 1e-9)
    f1 = (2 * pr * rc / np.maximum(pr + rc, 1e-12)).mean(1)
    acc = tp.sum(1) / np.maximum(Wk.sum(1), 1e-9)
    return f1, acc


def metrics(W, L, gold, settings) -> dict:
    out = {}
    for name, draws in settings.items():
        f1s, accs = zip(*(macro_f1(W, L, gold, cols) for cols in draws))
        out[f"{name}_f1"] = np.mean(f1s, 0)
        out[f"{name}_acc"] = np.mean(accs, 0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--base", help="run every other run is paired against (default: the first)")
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--local", type=Path)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device")
    ap.add_argument("--check", action="store_true",
                    help="verify subset argmax against a direct run on 5 columns (first run)")
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    items, unseen = load_items(700)
    ids = sorted(r["id"] for r in unseen)
    tag = label_hash(unseen)
    gold = np.searchsorted(ids, [x["rel"] for x in items])          # column index per instance
    val_cols = sorted({int(g) for g, x in zip(gold, items) if x["origin"] == "val"})
    trn_cols = sorted(set(range(len(ids))) - set(val_cols))
    rng = np.random.default_rng(13)
    settings = {"m15": [list(range(len(ids)))], "zs8": [val_cols], "ho7": [trn_cols]}
    for m in (5, 10):
        settings[f"m{m}"] = [sorted(rng.choice(len(ids), size=m, replace=False).tolist())
                             for _ in range(args.draws)]
    print(f"{len(items)} instances, {len(ids)} unseen relations ({len(val_cols)} never trained), "
          f"label texts {tag}", flush=True)

    runs = args.runs.split(",")
    base = args.base or runs[0]
    if base not in runs:
        runs = [base] + runs
    Ls = {r: score(r, items, ids, args.repo, device, args.local, tag, args.batch) for r in runs}

    if args.check:
        model, tok, spec, vocab, _ = load_run(runs[0], args.repo, device, args.local)
        cols = settings["m5"][0]
        sel = [i for i, g in enumerate(gold) if g in cols][:400]
        feats = [phase2_window(items[i]["tokens"], [(0, len(items[i]["tokens"]))], tok, spec, vocab,
                               [items[i]["head"], items[i]["tail"]], markers=model.rel_markers)
                 for i in sel]
        outs = run_phase2(model, feats, tok, device, batch=args.batch, cols=[ids[c] for c in cols])
        direct = np.array([o["relation"][0][1:] for o in outs])
        diff = np.abs(direct - Ls[runs[0]][sel][:, cols]).max()
        same = (direct.argmax(1) == Ls[runs[0]][sel][:, cols].argmax(1)).mean()
        print(f"check: max |logit diff| {diff:.2e}, argmax agreement {same:.4f}", flush=True)
        del model

    # Bootstrap weights: resample instances within each relation.
    brng = np.random.default_rng(7)
    W = np.zeros((args.boot + 1, len(items)), np.float32)
    W[0] = 1                                                     # row 0: the point estimate
    for c in range(len(ids)):
        idx = np.flatnonzero(gold == c)
        for b in range(1, args.boot + 1):
            W[b] += np.bincount(brng.choice(idx, size=len(idx)), minlength=len(items))[:len(items)]
    M = {r: metrics(W, Ls[r], gold, settings) for r in runs}
    keys = list(M[base])
    report = {"label_texts": tag, "instances": len(items), "draws": args.draws, "boot": args.boot,
              "runs": {r: {k: float(M[r][k][0]) for k in keys} for r in runs}, "paired": {}}
    print("\n| metric | " + " | ".join(runs) + " |\n|---|" + "---|" * len(runs))
    for k in keys:
        print(f"| {k} | " + " | ".join(f"{M[r][k][0]:.4f}" for r in runs) + " |")
    print(f"\nPaired against {base} (points, 95% bootstrap interval; * = excludes zero)\n")
    print("| run | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys))
    for r in runs:
        if r == base:
            continue
        cells, report["paired"][r] = [], {}
        for k in keys:
            d = (M[r][k] - M[base][k]) * 100
            lo, hi = np.percentile(d[1:], [2.5, 97.5])
            report["paired"][r][k] = [float(d[0]), float(lo), float(hi)]
            cells.append(f"{d[0]:+.1f} [{lo:+.1f}, {hi:+.1f}]{'*' if lo > 0 or hi < 0 else ''}")
        print(f"| {r} | " + " | ".join(cells) + " |")
    out = RUNS_DIR / "zs-protocol" / ("paired-" + time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(report, indent=1))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

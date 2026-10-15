"""Zero-shot relation classification on FewRel, scored the way ZS-BERT,
RE-Matching and GLiREL report it: m unseen relations, every test instance of
them classified among the m relation labels (the highest-scoring label is the
answer), macro precision / recall / F1 over the m relations, mean and
standard deviation over seeds.

    uv run python -m v6.train.zs_protocol --runs deberta-v3-large-p2-zs-v62-ftr

What differs from the published protocol, and why: there, m relations of
FewRel's 80 are drawn per seed and the model trains on the other 80 - m. Our
models are trained once, on FewRel's training relations, Re-DocRED and
Wikidata-labelled documents, which cover nearly all of FewRel; only the 8
validation relations no training source has (``train: false`` in the label
bank) are unseen. So the m unseen relations are drawn from those 8: m = 5 over
5 seeds, and m = 8 (one split). Comparable in metric and task, not in which
relations are unseen -- relation sets differ in difficulty.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..config import RUNS_DIR
from ..label_bank import load as load_bank
from .gold_eval import HF_REPO, load_run, phase2_window, run_phase2


def macro(gold: list[int], pred: list[int], classes: list[int]) -> dict:
    ps, rs, fs = [], [], []
    g, p = np.asarray(gold), np.asarray(pred)
    for c in classes:
        tp = int(((p == c) & (g == c)).sum())
        np_, ng = int((p == c).sum()), int((g == c).sum())
        pr, rc = tp / max(np_, 1), tp / max(ng, 1)
        ps.append(pr); rs.append(rc); fs.append(2 * pr * rc / max(pr + rc, 1e-12))
    return {"p": float(np.mean(ps)), "r": float(np.mean(rs)), "f1": float(np.mean(fs)),
            "acc": float((g == p).mean())}


def evaluate(model, tok, spec, vocab, device, splits: dict, items: list[dict]) -> dict:
    feats = {}
    out = {}
    for name, seeds in splits.items():
        runs = []
        for cols in seeds:
            its = [i for i, x in enumerate(items) if x["rel"] in cols]
            for i in its:
                if i not in feats:
                    x = items[i]
                    feats[i] = phase2_window(x["tokens"], [(0, len(x["tokens"]))], tok, spec, vocab,
                                             [x["head"], x["tail"]], markers=model.rel_markers)
            outs = run_phase2(model, [feats[i] for i in its], tok, device, batch=8, cols=cols)
            pred, gold = [], []
            for i, o in zip(its, outs):
                lg = o["relation"]
                pred.append(cols[int(lg[0][1:].argmax())] if len(lg) else -1)
                gold.append(items[i]["rel"])
            runs.append(macro(gold, pred, cols))
        out[name] = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs])))
                     for k in ("p", "r", "f1", "acc")}
        out[name]["seeds"] = len(seeds)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--local", type=Path)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--per-relation", type=int, default=700, help="FewRel has 700 per relation")
    ap.add_argument("--device")
    args = ap.parse_args()
    from ..gold import fewrel
    bank = load_bank()
    unseen = [r for r in bank["relations"] if not r.get("train", True)]
    code = {r["code"]: r["id"] for r in unseen}
    per: dict[int, int] = {}
    items = []
    for x in fewrel.load("val"):
        rid = code.get(x["code"])
        if rid is None or per.get(rid, 0) >= args.per_relation:
            continue
        per[rid] = per.get(rid, 0) + 1
        items.append({"tokens": x["tokens"], "head": x["head"], "tail": x["tail"], "rel": rid})
    ids = sorted(code.values())
    rng = np.random.default_rng(13)
    splits = {"m=5": [sorted(rng.choice(ids, size=5, replace=False).tolist())
                      for _ in range(args.seeds)],
              "m=8": [ids]}
    print(f"{len(items)} instances over {len(ids)} unseen relations: "
          f"{', '.join(r['name'] for r in unseen)}", flush=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    results = {}
    for run in args.runs.split(","):
        t0 = time.time()
        model, tok, spec, vocab, best = load_run(run, args.repo, device, args.local)
        results[run] = evaluate(model, tok, spec, vocab, device, splits, items)
        for name, r in results[run].items():
            print(f"  {run} {name} ({r['seeds']} seeds): macro F1 {r['f1'][0]:.4f} +- {r['f1'][1]:.4f}"
                  f" (P {r['p'][0]:.4f}, R {r['r'][0]:.4f}); accuracy {r['acc'][0]:.4f}"
                  f"  [{time.time() - t0:.0f}s]", flush=True)
        del model
    out = RUNS_DIR / "zs-protocol" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"splits": splits, "results": results}, indent=1))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

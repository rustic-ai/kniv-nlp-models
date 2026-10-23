"""Failure analysis, steps 0 and 2 (docs/v7-failure-analysis-plan.md): where a
model's errors fall, and how much of each an oracle threshold recovers.

    uv run python -m v6.train.failure_analysis --runs ultrabert-380m-v7b8ev,deberta-v3-large-v7b8ev

**Relations** (Re-DocRED test, gold entities): the error taxonomy of
``error_analysis.relations`` (threshold / confused / pair missed; confused,
reversed, spurious) scored at three settings --
- **deployed**: per-relation offsets chosen on dev (``calibrate_offsets``);
- **oracle global**: one offset chosen on test;
- **oracle per relation**: offsets chosen on test (the upper bound any
  threshold calibration can reach);
and the threshold-free ceilings: the share of gold facts whose relation is
the pair's best (rank 1) or among its best four.

**Coreference** (LitBank windows): ``error_analysis.coref``'s taxonomy at the
deployed link bias, and CoNLL-F1 over a sweep of biases (oracle = the best on
test).

**Entity spans** (Few-NERD test sample, CrossNER's five domains): per set,
exact-match F1 at the deployed span offset and at the best offset on that
set (oracle), and the error types: right span with the wrong type, a
boundary overlapping a gold span, a gold span missed, a predicted span
overlapping nothing.

Writes ``runs/failure/<run>.json`` and prints one ``FA`` line per finding.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np
import torch

from ..config import RUNS_DIR
from . import error_analysis as EA
from . import span_eval as SE
from .gold_eval import (HF_REPO, _pairs, apply_calibration, eval_litbank, fit_offsets, load_run,
                        metric, relation_pass)

COREF_BIASES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
SPAN_OFFSETS = tuple(np.round(np.arange(-5.0, 2.01, 0.5), 2))


def _f1(c: dict) -> float:
    fn = c.get("fn_threshold", 0) + c.get("fn_confused", 0) + c.get("fn_pair_missed", 0)
    fp = c.get("fp_confused", 0) + c.get("fp_reversed", 0) + c.get("fp_spurious", 0)
    tp = c.get("tp", 0)
    return 2 * tp / max(2 * tp + fn + fp, 1)


def relations(model, tok, spec, vocab, device) -> dict:
    names = vocab.labels["relations"]
    t0 = time.time()
    dev = relation_pass(model, tok, spec, vocab, device, "dev")
    test = relation_pass(model, tok, spec, vocab, device, "test")
    print(f"FA relations: dev and test passes in {time.time() - t0:.0f}s", flush=True)
    _, dev_per = fit_offsets(*dev, names)
    t_glob, t_per = fit_offsets(*test, names)
    out = {}
    shipped = getattr(model, "rel_offsets", None)
    for key, off in ((("shipped", shipped),) if shipped is not None else ()) + (
            ("deployed", dev_per), ("oracle_global", np.full(len(names), t_glob)),
                     ("oracle_per_relation", t_per)):
        r = EA.relations(model, tok, spec, vocab, device, data=test, offset=off)
        r["f1"] = _f1(r["counts"])
        out[key] = r
        print(f"FA relations {key}: F1 {r['f1']:.4f}", flush=True)
        for line in EA.summarise(r):
            print(f"FA   {line}", flush=True)
    # Threshold-free: is the gold relation the pair's best, or among its best four?
    idx = {r: k for k, r in enumerate(names)}
    rank = Counter()
    for it, f, o in zip(*test):
        keep = f["_ent_keep"]
        lg = {(keep[h], keep[t]): x for (h, t), x in zip(_pairs(len(keep)), o["relation"])}
        for h, t, r in (tuple(x) for x in it.layers["rel"]):
            if r not in idx:
                continue
            rank["gold"] += 1
            x = lg.get((h, t))
            if x is None:
                continue
            order = list(np.argsort(-x[1:]))
            pos = order.index(idx[r])
            rank["rank1"] += int(pos == 0)
            rank["top4"] += int(pos < 4)
    out["ceiling"] = {"rank1_recall": rank["rank1"] / max(rank["gold"], 1),
                      "top4_recall": rank["top4"] / max(rank["gold"], 1), "gold": rank["gold"]}
    print(f"FA relations ceiling: gold relation ranked first {out['ceiling']['rank1_recall']:.3f}, "
          f"in the best four {out['ceiling']['top4_recall']:.3f} (of {rank['gold']:,})", flush=True)
    return out


def coref(model, tok, spec, vocab, device) -> dict:
    tax = EA.coref(model, tok, spec, vocab, device)
    c = tax["counts"]
    kinds = ("pronoun", "name", "nominal")
    for k in kinds:
        g = max(c.get(f"gold_{k}", 0), 1)
        print(f"FA coref {k}: gold {c.get(f'gold_{k}', 0)}, correct {c.get(f'correct_{k}', 0) / g:.3f}, "
              f"wrong link {c.get(f'wrong_{k}', 0) / g:.3f}, unlinked {c.get(f'unlinked_{k}', 0) / g:.3f}, "
              f"missed {c.get(f'missed_{k}', 0) / g:.3f} (boundary {c.get(f'missed_overlap_{k}', 0) / g:.3f})",
              flush=True)
    print(f"FA coref entities: split {c.get('split_entities', 0) / max(c.get('gold_entities', 1), 1):.3f} "
          f"of gold, fused {c.get('fused_entities', 0) / max(c.get('pred_entities', 1), 1):.3f} of predicted",
          flush=True)
    deployed = getattr(model, "coref_link_bias", 0.0)
    sweep = {}
    for b in sorted(set(COREF_BIASES) | {float(deployed)}):
        model.coref_link_bias = b
        r = eval_litbank(model, tok, spec, vocab, device, None)
        sweep[b] = metric(r["items"], "coref_conll", "acc")
    model.coref_link_bias = deployed
    best = max(sweep, key=sweep.get)
    print(f"FA coref bias: deployed {deployed:+.1f} -> {sweep[float(deployed)]:.4f}; oracle {best:+.1f} -> "
          f"{sweep[best]:.4f} ({', '.join(f'{k:+.1f}:{v:.3f}' for k, v in sorted(sweep.items()))})",
          flush=True)
    return {"taxonomy": tax, "bias_sweep": sweep, "deployed_bias": deployed, "oracle_bias": best}


def _span_errors(gold: list[set], pred: list[set]) -> dict:
    c = Counter()
    for g, p in zip(gold, pred):
        gs = {(a, b): t for a, b, t in g}
        ps = {(a, b): t for a, b, t in p}
        c["gold"] += len(g); c["pred"] += len(p)
        for (a, b), t in gs.items():
            if ps.get((a, b)) == t:
                c["correct"] += 1
            elif (a, b) in ps:
                c["wrong_type"] += 1
            elif any(x < b and a < y for x, y in ps):
                c["boundary"] += 1
            else:
                c["missed"] += 1
        for (a, b) in ps:
            if (a, b) not in gs and not any(x < b and a < y for x, y in gs):
                c["spurious"] += 1
    return dict(c)


def spans(model, tok, spec, vocab, device, fewnerd_n: int = 3000) -> dict:
    SE.DESCRIBE = True                                  # the scorecard's setting (--describe)
    sets = {"fewnerd": SE.load_fewnerd(fewnerd_n)}
    for d in SE.DOMAINS:
        sets[f"crossner_{d}"] = SE.load_crossner(d)
    deployed = float(getattr(model, "span_offset", 0.0) or 0.0)
    out = {}
    for n, s in sets.items():
        labels = sorted({t for it in s for t, _, _ in it["spans"]})
        gold = [{(a, b, t) for t, a, b in it["spans"]} for it in s]
        sweep = {}
        for off in sorted(set(SPAN_OFFSETS) | {deployed}):
            sweep[float(off)] = SE.f1(gold, SE.kniv_predict(model, tok, spec, vocab, device, s, labels,
                                                            offset=float(off)))["f1"]
        best = max(sweep, key=sweep.get)
        err = {k: _span_errors(gold, SE.kniv_predict(model, tok, spec, vocab, device, s, labels, offset=o))
               for k, o in (("deployed", deployed), ("oracle", best))}
        out[n] = {"deployed": sweep[deployed], "oracle": sweep[best], "oracle_offset": best,
                  "errors": err, "sweep": sweep}
        for k, e in err.items():
            g = max(e.get("gold", 1), 1)
            print(f"FA spans {n} {k} ({deployed if k == 'deployed' else best:+.1f}): F1 "
                  f"{sweep[deployed if k == 'deployed' else best]:.4f}; of gold spans: correct "
                  f"{e.get('correct', 0) / g:.3f}, wrong type {e.get('wrong_type', 0) / g:.3f}, boundary "
                  f"{e.get('boundary', 0) / g:.3f}, missed {e.get('missed', 0) / g:.3f}; spurious "
                  f"{e.get('spurious', 0)} of {e.get('pred', 0)} predicted", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--parts", default="relations,coref,spans")
    ap.add_argument("--device")
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    parts = args.parts.split(",")
    (RUNS_DIR / "failure").mkdir(parents=True, exist_ok=True)
    for run in args.runs.split(","):
        model, tok, spec, vocab, _ = load_run(run, HF_REPO, device)
        model.eval()
        cal = apply_calibration(model, run, vocab)
        print(f"FA === {run}: calibration {json.dumps({k: v for k, v in cal.items() if k != 'rel_offsets'})}",
              flush=True)
        res = {}
        with torch.no_grad():
            for p in parts:
                res[p] = {"relations": relations, "coref": coref, "spans": spans}[p](
                    model, tok, spec, vocab, device)
                (RUNS_DIR / "failure" / f"{run}.json").write_text(json.dumps(res, indent=1, default=str))
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

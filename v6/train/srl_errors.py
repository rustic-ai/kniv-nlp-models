"""SRL errors by kind, role and word fragmentation, and a decoding-only
calibration -- on PropBank EWT (gold predicates, as ``gold_eval``'s ``srl``).

    uv run python -m v6.train.srl_errors --runs deberta-v3-large-v7bk-p1,ultrabert-380m-v7bk-p1

Per gold argument span: **correct** (span and role), **label** (exact span,
wrong role), **boundary** (overlapping span, same role), **both** (overlapping,
different role) or **missed**; per predicted span with no overlapping gold:
**spurious**. Per role F1 (B-/I- merged). Recall by the most subwords any word
of the argument splits into under the run's own tokenizer (1, 2, 3+): does a
tokenizer that fragments words lose arguments?

**Calibration**: a constant added to the ``O`` tag's log-probability before
Viterbi (negative: more arguments), chosen on PropBank dev, scored on test.
It changes no weight -- if it recovers the gap, the gap is the head's
argument/non-argument balance, not what the encoder represents.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from ..config import RUNS_DIR
from .gold_eval import (HF_REPO, bio_constraints, bio_spans, load_run, run_model, viterbi,
                        window)

BIASES = np.round(np.arange(-3.0, 1.01, 0.25), 2)


def logprobs(model, tok, spec, vocab, device, items):
    by_sent: dict[tuple, list] = defaultdict(list)
    for it in items:
        by_sent[tuple(it.tokens)].append(it)
    sents = list(by_sent)
    feats = [window(list(s), [(0, len(s))], tok, spec, vocab,
                    predicates=[it.predicate_idx for it in by_sent[s]]) for s in sents]
    outs = run_model(model, feats, tok, device)
    return [(it, lp) for s, o in zip(sents, outs) for it, lp in zip(by_sent[s], o["srl"])]


def decode(pairs, tags, bias):
    allowed, start = bio_constraints(tags)
    v, o_ = tags.index("V"), tags.index("O")
    res = []
    for it, lp in pairs:
        lp = lp.copy()
        lp[:, o_] += bias
        keep = np.ones_like(lp, dtype=bool)
        keep[:, v] = False
        keep[it.predicate_idx, :] = False
        keep[it.predicate_idx, v] = True
        pred = [tags[x] for x in viterbi(lp, allowed, start, keep)]
        res.append((it, bio_spans(pred), bio_spans(it.layers["srl"])))
    return res


def f1(res):
    tp = sum(len(p & g) for _, p, g in res)
    np_, ng = sum(len(p) for _, p, _ in res), sum(len(g) for _, _, g in res)
    pr, rc = tp / max(np_, 1), tp / max(ng, 1)
    return {"p": pr, "r": rc, "f1": 2 * pr * rc / max(pr + rc, 1e-12)}


def _span(x):
    """bio_spans items as (label, start, end_exclusive), whatever their order."""
    lab = next(e for e in x if isinstance(e, str))
    a, b = sorted(e for e in x if not isinstance(e, str))
    return lab, a, b


def analyse(res, tok):
    kinds, roles, frag = Counter(), defaultdict(Counter), defaultdict(Counter)
    for it, ps, gs in res:
        P = [_span(x) for x in ps]
        G = [_span(x) for x in gs]
        npieces = [max(1, len(tok.tokenize(" " + w))) for w in it.tokens]
        for g in G:
            lab, a, b = g
            if lab == "V":
                continue
            bucket = min(max(npieces[a:b] or [1]), 3)
            ok = g in P
            frag[bucket]["n"] += 1
            frag[bucket]["ok"] += ok
            roles[lab]["g"] += 1
            roles[lab]["tp"] += ok
            if ok:
                kinds["correct"] += 1
                continue
            ov = [p for p in P if p[0] != "V" and p[1] < b and a < p[2]]
            if any(p[1] == a and p[2] == b for p in ov):
                kinds["label"] += 1
            elif any(p[0] == lab for p in ov):
                kinds["boundary"] += 1
            elif ov:
                kinds["both"] += 1
            else:
                kinds["missed"] += 1
        for p in P:
            if p[0] == "V":
                continue
            roles[p[0]]["p"] += 1
            if not any(g[1] < p[2] and p[1] < g[2] for g in G if g[0] != "V"):
                kinds["spurious"] += 1
    role_f1 = {}
    for r, c in roles.items():
        pr, rc = c["tp"] / max(c["p"], 1), c["tp"] / max(c["g"], 1)
        role_f1[r] = {"n": c["g"], "f1": 2 * pr * rc / max(pr + rc, 1e-12), "p": pr, "r": rc}
    return {"kinds": dict(kinds), "roles": role_f1,
            "recall_by_subwords": {k: {"n": c["n"], "recall": c["ok"] / c["n"]}
                                   for k, c in sorted(frag.items())}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--local", type=Path)
    ap.add_argument("--device")
    args = ap.parse_args()
    from ..config import PREPARED_DIR
    from ..gold.propbank import load_srl_items
    test = load_srl_items()
    dev = load_srl_items(path=PREPARED_DIR / "srl_dev.json")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    out = {}
    for run in args.runs.split(","):
        t0 = time.time()
        model, tok, spec, vocab, _ = load_run(run, args.repo, device, args.local)
        tags = vocab.labels["srl"]
        lp_dev = logprobs(model, tok, spec, vocab, device, dev)
        lp_test = logprobs(model, tok, spec, vocab, device, test)
        sweep = {float(b): f1(decode(lp_dev, tags, b))["f1"] for b in BIASES}
        best = max(sweep, key=sweep.get)
        base, cal = decode(lp_test, tags, 0.0), decode(lp_test, tags, best)
        out[run] = {"dev_sweep": sweep, "bias": best,
                    "test": f1(base), "test_calibrated": f1(cal),
                    "analysis": analyse(base, tok)}
        r = out[run]
        print(f"{run}: test F1 {r['test']['f1']:.4f} (P {r['test']['p']:.3f} R {r['test']['r']:.3f}); "
              f"O bias {best:+.2f} on dev ({sweep[0.0]:.4f} -> {sweep[best]:.4f}) -> test "
              f"{r['test_calibrated']['f1']:.4f} (P {r['test_calibrated']['p']:.3f} "
              f"R {r['test_calibrated']['r']:.3f})  [{time.time() - t0:.0f}s]", flush=True)
        print(f"  errors: {r['analysis']['kinds']}", flush=True)
        print(f"  recall by subwords per word: {r['analysis']['recall_by_subwords']}", flush=True)
        del model
    path = RUNS_DIR / "srl-errors" / time.strftime("%Y%m%d-%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.json").write_text(json.dumps(out, indent=1))
    print(f"written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

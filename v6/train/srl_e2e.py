"""Where end-to-end SRL loses its F1, and what decoding alone recovers -- on
PropBank EWT with every predicate (``gold_eval``'s ``srl_full``).

    uv run python -m v6.train.srl_e2e --local data/v6-eval/checkpoints --runs R1,R2

End to end, an argument counts only if predicate, span and role all match.
Gold arguments are lost three ways: their predicate was **missed**, the
predicate was found but the argument's role or span was wrong (**role**
errors), or arguments were added for a **spurious** predicate. Each is
counted, overall and by the predicate's part of speech, with per-role F1 on
correctly found predicates.

**Decoding search** (no weight changes): a predicate threshold on the
predicate head's probability (instead of argmax, i.e. 0.5) and a constant
added to the ``O`` tag's log-probability before Viterbi (negative: more
arguments). Chosen on PropBank dev end-to-end F1, scored on test once.

One encoder pass per split suffices: the joint SRL head scores each listed
predicate's frame independently, so the pass lists the gold predicates and
every word above the lowest threshold searched, and any predicate set is then
decoded from it.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from ..config import DATA_DIR, RUNS_DIR
from .gold_eval import (HF_REPO, POS_GROUPS, _group, bio_constraints, bio_spans, load_run,
                        run_model, viterbi, window)

THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
BIASES = (-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5)


def frames_for(model, tok, spec, vocab, device, rows):
    """Per sentence: predicate probabilities and, for gold and candidate
    predicates, the role log-probabilities ``{pred: [W, tags]}``."""
    feats = [window(r["words"], [(0, len(r["words"]))], tok, spec, vocab) for r in rows]
    probe = run_model(model, feats, tok, device)
    cands = []
    for r, o in zip(rows, probe):
        c = {f["predicate_idx"] for f in r["frames"]}
        c |= set(np.nonzero(o["srl_pred_p"] >= min(THRESHOLDS))[0].tolist())
        cands.append(sorted(c))
    feats = [window(r["words"], [(0, len(r["words"]))], tok, spec, vocab, predicates=c)
             for r, c in zip(rows, cands)]
    outs = run_model(model, feats, tok, device)
    return [{"p": o["srl_pred_p"], "lp": dict(zip(c, o["srl"]))}
            for c, o in zip(cands, outs)]


def score(rows, res, tags, thr, bias):
    allowed, start = bio_constraints(tags)
    v, o_ = tags.index("V"), tags.index("O")
    c = Counter()
    per_role = defaultdict(Counter)
    for r, x in zip(rows, res):
        gold = {f["predicate_idx"]: bio_spans(f["srl_tags"]) for f in r["frames"]}
        found = set(np.nonzero(x["p"] >= thr)[0].tolist())
        for p in set(gold) | found:
            g = _group(r["upos"][p])
            gs = gold.get(p, set())
            if p not in found:
                c["missed_pred_args"] += len(gs); c[f"missed_pred_args_{g}"] += len(gs)
                c["gold"] += len(gs); c[f"gold_{g}"] += len(gs)
                continue
            lp = x["lp"][p].copy()
            lp[:, o_] += bias
            keep = np.ones_like(lp, dtype=bool)
            keep[:, v] = False
            keep[p, :] = False
            keep[p, v] = True
            ps = bio_spans([tags[k] for k in viterbi(lp, allowed, start, keep)])
            c["gold"] += len(gs); c[f"gold_{g}"] += len(gs)
            c["pred"] += len(ps); c[f"pred_{g}"] += len(ps)
            tp = len(gs & ps)
            c["tp"] += tp; c[f"tp_{g}"] += tp
            if p not in gold:
                c["spurious_pred_args"] += len(ps); c[f"spurious_pred_args_{g}"] += len(ps)
                continue
            # Role errors on a correctly found predicate.
            for role, a, b in gs:
                if (role, a, b) in ps:
                    c["correct"] += 1
                    per_role[role]["tp"] += 1
                    continue
                ov = [(r2, a2, b2) for r2, a2, b2 in ps if a2 <= b and a <= b2]
                if any((a2, b2) == (a, b) for _, a2, b2 in ov):
                    c["label"] += 1
                elif any(r2 == role for r2, _, _ in ov):
                    c["boundary"] += 1
                elif ov:
                    c["both"] += 1
                else:
                    c["missed_arg"] += 1
                per_role[role]["fn"] += 1
            for role, a, b in ps:
                if (role, a, b) not in gs:
                    per_role[role]["fp"] += 1
                    if not any(a2 <= b and a <= b2 for _, a2, b2 in gs):
                        c["spurious_arg"] += 1
    f1 = 2 * c["tp"] / max(c["pred"] + c["gold"], 1)
    return f1, c, per_role


def splits(split):
    return [r for r in json.loads((DATA_DIR / "prepared" / "propbank-full" / f"{split}.json")
                                  .read_text())]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--local", type=Path)
    ap.add_argument("--device")
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    out = RUNS_DIR / "srl_e2e"
    out.mkdir(parents=True, exist_ok=True)
    for run in args.runs.split(","):
        model, tok, spec, vocab, _ = load_run(run, args.repo, device, args.local)
        tags = vocab.labels["srl"]
        dev, test = splits("dev"), splits("test")
        rd = frames_for(model, tok, spec, vocab, device, dev)
        rt = frames_for(model, tok, spec, vocab, device, test)
        grid = {(t, b): score(dev, rd, tags, t, b)[0] for t in THRESHOLDS for b in BIASES}
        best = max(grid, key=grid.get)
        base_f1, base, roles = score(test, rt, tags, 0.5, 0.0)
        best_f1, cal, _ = score(test, rt, tags, *best)
        n = base["gold"]
        print(f"\n{run}: end-to-end F1 test {base_f1:.4f} (argmax predicates, no bias)", flush=True)
        print(f"  gold arguments {n:,}: lost to missed predicates {base['missed_pred_args']:,} "
              f"({base['missed_pred_args'] / n:.1%}); on found predicates: label "
              f"{base['label']:,}, boundary {base['boundary']:,}, both {base['both']:,}, "
              f"missed {base['missed_arg']:,}; false arguments: of spurious predicates "
              f"{base['spurious_pred_args']:,}, spurious spans {base['spurious_arg']:,}")
        for g in POS_GROUPS + ("other",):
            if base[f"gold_{g}"]:
                fg = 2 * base[f"tp_{g}"] / max(base[f"pred_{g}"] + base[f"gold_{g}"], 1)
                print(f"  {g:5s}: gold args {base[f'gold_{g}']:5,}  e2e F1 {fg:.3f}  "
                      f"lost to missed predicates {base[f'missed_pred_args_{g}']:,}  "
                      f"args of spurious predicates {base[f'spurious_pred_args_{g}']:,}")
        worst = sorted(roles.items(), key=lambda kv: -(kv[1]["fn"] + kv[1]["fp"]))[:10]
        print("  roles by errors (found predicates): " + "  ".join(
            f"{r} F1 {2 * c['tp'] / max(2 * c['tp'] + c['fn'] + c['fp'], 1):.3f} "
            f"(fn {c['fn']}, fp {c['fp']})" for r, c in worst))
        print(f"  decoding chosen on dev: predicate threshold {best[0]}, O bias {best[1]:+} "
              f"(dev {grid[best]:.4f} vs {grid[(0.5, 0.0)]:.4f}) -> test {best_f1:.4f} "
              f"({best_f1 - base_f1:+.4f})", flush=True)
        (out / f"{run}.json").write_text(json.dumps({
            "test_f1": base_f1, "test_f1_calibrated": best_f1,
            "chosen": {"pred_threshold": best[0], "o_bias": best[1]},
            "dev_grid": {f"{t},{b}": v for (t, b), v in grid.items()},
            "breakdown": dict(base), "roles": {r: dict(c) for r, c in roles.items()}}, indent=1))
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

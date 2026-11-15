"""V3's order sensitivity, read on V1c's held-out examples (no training,
nothing tuned): how much the reshuffle costs, and whether a fixed order
removes it.

    uv run python -m v6.eval.v3_order            # the pilot's checkpoint, held and held_long

Readings, each candidate inside the list the extractor would send:

- AUC under three shuffles of the lists (seeds 1, 2, 3; seed 1 is what the
  pilot evaluated) and under the fixed order (``canon``: each list sorted by
  the head's first mention, the tail's, then the relation), built from two
  different input orders;
- between shuffles 1 and 2: keep decisions at 0.6 that flip, how far the
  flipped candidates sit from 0.6, the mean change in the keep probability,
  and precision and recall at 0.6 under each;
- between the two fixed-order builds: verdicts that differ (should be 0).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from ..train import verifier3 as V
from ..train.verifier import HEDGED_ANY, VERDICTS_K, coarse


def _sets(recs, extra, tok, bank, args, canon):
    out = {}
    for split in ("held", "held_long"):
        keys = V.v1c_held_keys(recs, tok, bank, split, args.dev_limit)
        out[split] = lambda rng, keys=keys: [p for i, ks in keys.items() for p in V.record_passes(
            recs[i], extra.get(recs[i]["id"]), tok, bank, args.budget, args.cap, rng, keep=ks, canon=canon)]
    return out


def _arr(pred, ks):
    P = np.stack([pred[k][0] for k in ks])
    Y = np.array([pred[k][2] for k in ks])
    keep = 1 - coarse(P)[:, 1]
    Yc = np.where((Y >= 3) | (Y == HEDGED_ANY), 3, Y)
    return P.argmax(1), keep, Yc != 1


def _pr(keep, pos, t=0.6):
    k = keep >= t
    return {"precision": round(float(pos[k].mean()), 4), "recall": round(float(k[pos].mean()), 4), "kept": int(k.sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=V.RUN)
    ap.add_argument("--ver", default=f"{V.RUN}-ver3")
    ap.add_argument("--repo", default="dragonscale-ai/kniv-v6-runs")
    ap.add_argument("--device")
    ap.add_argument("--data", default="verifier_v1c.json")
    ap.add_argument("--budget", type=int, default=448)
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument("--dev-limit", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--batch-tokens", type=int, default=384)
    ap.add_argument("--out", type=Path, default=Path("v6/runs/v3/order.json"))
    a = ap.parse_args()
    a.local, a.out_run = None, a.ver
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else
                                       "mps" if torch.backends.mps.is_available() else "cpu"))
    base, tok, bank, recs = V._setup(a, device)
    extra = V.load_lists(a)
    from collections import Counter
    ids = Counter(r["id"] for r in recs)
    for k in [k for k, v in ids.items() if v > 1]:
        extra.pop(k, None)
    from huggingface_hub import hf_hub_download
    from ..train.gold_eval import CKPT_DIR
    p = CKPT_DIR / a.ver / "verifier.pt"
    if not p.exists():
        p = Path(hf_hub_download(a.repo, f"{a.ver}/verifier.pt", local_dir=str(CKPT_DIR)))
    ck = torch.load(p, map_location="cpu", weights_only=False)
    model = V.Verifier3(base, r=64, n_out=len(VERDICTS_K)).to(device)
    model.ver.load_state_dict(ck["ver"]); model.head.load_state_dict(ck["head"])
    b2 = a.batch_tokens ** 2 * a.batch
    shuf, fixed = _sets(recs, extra, tok, bank, a, False), _sets(recs, extra, tok, bank, a, True)
    res = {}
    for split in ("held", "held_long"):
        runs = {f"shuffle{s}": V.predict(model, shuf[split](random.Random(s)), tok.pad_token_id, device, b2)
                for s in (1, 2, 3)}
        for s in (5, 7):
            runs[f"fixed{s}"] = V.predict(model, fixed[split](random.Random(s)), tok.pad_token_id, device, b2)
        ks = sorted(set.intersection(*(set(r) for r in runs.values())), key=str)
        R = {n: _arr(r, ks) for n, r in runs.items()}
        pos = R["shuffle1"][2]
        out = {"n": len(ks)}
        for n, (v, keep, _) in R.items():
            out[f"auc_{n}"] = round(V._auc(keep, pos), 4)
            out[f"pr_{n}"] = _pr(keep, pos)
        for x, y in (("shuffle1", "shuffle2"), ("shuffle1", "shuffle3"), ("fixed5", "fixed7"), ("shuffle1", "fixed5")):
            (va, ka, _), (vb, kb, _) = R[x], R[y]
            flip = (ka >= 0.6) != (kb >= 0.6)
            near = np.minimum(np.abs(ka - 0.6), np.abs(kb - 0.6))
            out[f"{x}_vs_{y}"] = {
                "verdict_changed": round(float((va != vb).mean()), 4),
                "keep_flipped": round(float(flip.mean()), 4),
                "mean_abs_dkeep": round(float(np.abs(ka - kb).mean()), 4),
                "flipped_within_0.05": round(float((near[flip] <= 0.05).mean()), 3) if flip.any() else None,
                "flipped_within_0.10": round(float((near[flip] <= 0.10).mean()), 3) if flip.any() else None,
                "flipped_true": int((flip & pos).sum()), "flipped_false": int((flip & ~pos).sum()),
            }
        res[split] = out
        print(split, json.dumps(out), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

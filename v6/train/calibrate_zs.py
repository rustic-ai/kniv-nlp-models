"""Threshold calibration for relations the label path never trained on.

    uv run python -m v6.train.calibrate_zs --run deberta-v3-large-p2-zs-v62-ftr

The label path ranks unseen relations well (FewRel 8-way 0.48) but their
scores rarely clear the threshold column, so thresholded extraction of an
unseen relation finds almost nothing. A single offset ``c`` fixes the scale:
a label-path relation is predicted when its margin (logit minus the
threshold logit) exceeds ``c`` instead of 0. Frozen-path relations (Re-DocRED's
96, answered by the legacy head) are not offset.

``c`` is chosen on half of the held-out teacher windows -- the 40 relations no
training run saw, scored against the teacher's labels -- and reported on the
other half, so the number is not fitted to its own test. Teacher labels are
incomplete, so absolute F1 is an underestimate; the offset is what matters.
Written to ``<run>/calibration.json`` beside the run's other outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..config import RUNS_DIR
from ..label_bank import load as load_bank
from .data import collate
from .gold_eval import HF_REPO, load_run
from .kdata import LabelCollator, knowledge_features, load_holdout

OFFSETS = np.round(np.arange(-8.0, 4.01, 0.25), 2)


@torch.no_grad()
def margins(model, tok, device, feats, coll, batch: int = 2):
    """Per pair: label-path margins, gold labels and the schema mask."""
    rows = []
    for k in range(0, len(feats), batch):
        b = coll(collate(feats[k:k + batch], tok.pad_token_id))
        b = {n: v.to(device) for n, v in b.items() if hasattr(v, "to")}
        o = model(b, relation_only=True)
        lg = o.get("relation_label", o["relation"]).float()
        if not len(lg):
            continue
        m = (lg[:, 1:] - lg[:, :1]).cpu().numpy()
        rows.append((m, b["pair_labels"][:, 1:].cpu().numpy() > 0.5,
                     b["pair_label_mask"][:, 1:].cpu().numpy() > 0))
    return rows


def sweep(rows) -> list[dict]:
    out = []
    for c in OFFSETS:
        tp = np_ = ng = 0
        for m, g, k in rows:
            p = (m > c) & k
            tp += int((p & g).sum()); np_ += int(p.sum()); ng += int((g & k).sum())
        pr, rc = tp / max(np_, 1), tp / max(ng, 1)
        out.append({"offset": float(c), "p": pr, "r": rc, "f1": 2 * pr * rc / max(pr + rc, 1e-12),
                    "predicted": np_})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--local", type=Path)
    ap.add_argument("--device")
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    model, tok, spec, vocab, _ = load_run(args.run, args.repo, device, args.local)
    bank = load_bank()
    hold = load_holdout()["relations"]
    feats = knowledge_features("teacher", "dev", tok, spec.max_subwords, vocab.size("cls"), bank,
                               markers=model.rel_markers, limit=1500)
    for f in feats:
        f.pop("ents", None)                     # spans are not scored here
    half = [hashlib.sha1(str(i).encode()).digest()[0] % 2 for i in range(len(feats))]
    coll = LabelCollator(tok, bank, train=False, only_ids=hold)
    search = margins(model, tok, device, [f for f, h in zip(feats, half) if h == 0], coll)
    held = margins(model, tok, device, [f for f, h in zip(feats, half) if h == 1], coll)
    s, h = sweep(search), sweep(held)
    best = max(s, key=lambda x: x["f1"])
    at = next(x for x in h if x["offset"] == best["offset"])
    zero = next(x for x in h if x["offset"] == 0.0)
    rec = {"run": args.run, "offset": best["offset"], "search_f1": best["f1"],
           "held_out": at, "held_out_at_zero": zero,
           "note": "label-path relations: predict when margin > offset; frozen-path relations unchanged"}
    out = RUNS_DIR / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.run}.json").write_text(json.dumps({**rec, "search": s, "held": h}, indent=1))
    print(f"offset {best['offset']:+.2f} (chosen on {sum(1 for x in half if x == 0)} windows): "
          f"held-out F1 {at['f1']:.4f} (P {at['p']:.4f} R {at['r']:.4f}, {at['predicted']} predicted) "
          f"vs at 0: F1 {zero['f1']:.4f} (P {zero['p']:.4f} R {zero['r']:.4f}, {zero['predicted']} predicted)")
    for x in h[::4]:
        print(f"  offset {x['offset']:+.2f}: P {x['p']:.3f} R {x['r']:.3f} F1 {x['f1']:.3f} ({x['predicted']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

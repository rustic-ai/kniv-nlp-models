"""Per-relation zero-shot breakdown: which relations a checkpoint gets right,
and what it confuses them with.

    uv run python -m v6.train.rel_breakdown --runs deberta-v3-large-p2-zs-v62-ftr,ettin-400m-v7bk-k2

Two threshold-free tests, scored per gold relation:

- **held-out teacher relations** (the 40 of ``kg_holdout.json``, never
  trained): for every pair in the teacher dev windows holding one, is a true
  relation the best-scoring of the 40? Split by window domain and by whether
  the pair involves a speaker (conversation facts) -- the in-loop
  ``relation_zs_acc``, broken down;
- **FewRel's 8 unseen relations**: every instance classified among the 8, per
  relation accuracy and its most frequent wrong answer.
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
from ..label_bank import load as load_bank
from .gold_eval import HF_REPO, load_run, phase2_window, run_phase2


@torch.no_grad()
def heldout(model, tok, spec, vocab, device, bank, hold_ids, limit):
    from .data import collate
    from .kdata import LabelCollator, knowledge_features
    feats = knowledge_features("teacher", "dev", tok, spec.max_subwords, vocab.size("cls"),
                               bank, markers=model.rel_markers, limit=limit)
    coll = LabelCollator(tok, bank, train=False, only_ids=hold_ids)
    rows = []
    model.eval()
    for f in feats:
        b = coll(collate([f], tok.pad_token_id))
        b = {k: v.to(device) if hasattr(v, "to") else v for k, v in b.items()}
        o = model(b, relation_only=True)
        if "relation" not in o or not len(o["relation"]):
            continue
        lg, gold = o["relation"][:, 1:].float(), b["pair_labels"][:, 1:] > 0.5
        if "pair_label_mask" in b:
            lg = lg.masked_fill(b["pair_label_mask"][:, 1:] == 0, float("-inf"))
        cols = b["rel_label_cols"].tolist()
        n_part = f["rel"].get("n_participants", 0)
        for p in gold.any(1).nonzero(as_tuple=True)[0].tolist():
            top = int(lg[p].argmax())
            speaker = int(b["pair_h"][p]) < n_part or int(b["pair_t"][p]) < n_part
            for g in gold[p].nonzero(as_tuple=True)[0].tolist():
                rows.append({"gold": cols[g], "pred": cols[top], "ok": bool(gold[p, top]),
                             "domain": f.get("_domain", "?"), "speaker": speaker})
    return rows


def fewrel_rows(model, tok, spec, vocab, device, bank, per):
    from ..gold import fewrel
    unseen = [r for r in bank["relations"] if not r.get("train", True)]
    code = {r["code"]: r["id"] for r in unseen}
    count: Counter = Counter()
    items = []
    for x in fewrel.load("val"):
        rid = code.get(x["code"])
        if rid is None or count[rid] >= per:
            continue
        count[rid] += 1
        items.append((x, rid))
    cols = sorted(code.values())
    feats = [phase2_window(x["tokens"], [(0, len(x["tokens"]))], tok, spec, vocab,
                           [x["head"], x["tail"]], markers=model.rel_markers) for x, _ in items]
    outs = run_phase2(model, feats, tok, device, batch=8, cols=cols)
    rows = []
    for (_, rid), o in zip(items, outs):
        lg = o["relation"]
        pred = cols[int(lg[0][1:].argmax())] if len(lg) else -1
        rows.append({"gold": rid, "pred": pred, "ok": pred == rid})
    return rows


def summarise(rows, key):
    by = defaultdict(list)
    for r in rows:
        by[key(r)].append(r)
    return {k: {"n": len(v), "acc": sum(r["ok"] for r in v) / len(v),
                "confused_with": Counter(r["pred"] for r in v if not r["ok"]).most_common(2)}
            for k, v in by.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--local", type=Path)
    ap.add_argument("--teacher-limit", type=int, default=1500)
    ap.add_argument("--fewrel-per-relation", type=int, default=300)
    ap.add_argument("--device")
    args = ap.parse_args()
    from .kdata import load_holdout
    bank = load_bank()
    name = {r["id"]: r["name"] for r in bank["relations"]}
    hold_ids = load_holdout()["relations"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    out = {}
    for run in args.runs.split(","):
        t0 = time.time()
        model, tok, spec, vocab, _ = load_run(run, args.repo, device, args.local)
        h = heldout(model, tok, spec, vocab, device, bank, hold_ids, args.teacher_limit)
        fr = fewrel_rows(model, tok, spec, vocab, device, bank, args.fewrel_per_relation)
        out[run] = {
            "heldout_all": sum(r["ok"] for r in h) / max(len(h), 1),
            "heldout_by_relation": summarise(h, lambda r: name[r["gold"]]),
            "heldout_by_domain": summarise(h, lambda r: r["domain"]),
            "heldout_by_speaker": summarise(h, lambda r: "speaker pair" if r["speaker"] else "other pair"),
            "fewrel_all": sum(r["ok"] for r in fr) / max(len(fr), 1),
            "fewrel_by_relation": summarise(fr, lambda r: name[r["gold"]]),
        }
        for sec in ("heldout_by_relation", "fewrel_by_relation"):
            for v in out[run][sec].values():
                v["confused_with"] = [(name.get(k, str(k)), n) for k, n in v["confused_with"]]
        print(f"{run}: held-out {out[run]['heldout_all']:.3f} ({len(h)} facts), "
              f"FewRel 8-way {out[run]['fewrel_all']:.3f}  [{time.time() - t0:.0f}s]", flush=True)
        del model
    path = RUNS_DIR / "rel-breakdown" / time.strftime("%Y%m%d-%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.json").write_text(json.dumps(out, indent=1))
    print(f"written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

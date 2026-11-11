"""P0 fit test (v8 plan r2): can the relation head learn the pair decision
when its labels are clean?

    uv run python -m v6.train.fit_test --labels v6/runs/kg-judge/labels-astra-c1.json

The fit diagnostic found every checkpoint at F1 0.55 on short texts it was
trained on (E3, E13), with most errors in the yes / no decision on a pair,
and many "false" facts true (E4). Two readings: the teacher's labels are
inconsistent, so nothing can fit them; or the head cannot learn the pair
decision. This separates them.

Labels per text (``kg_judge``): **clean** = the judged candidates and
teacher facts the judge marks expressed (supported, negated, hedged);
**teacher** = the teacher's facts as trained today. In both, every other
pair and relation of the text's schema is a negative (gold entities).

The marker pass (encoder + relation adapter, frozen) and the label pass are
run once per text and cached; only the relation head (``relation.*``:
pooling, extractors, pair MLP, label cross-attention, threshold) trains, from
the run's own weights, once on each label set with the same budget. Scored
every epoch on the training texts and the held-out (dev) texts, against both
label sets, at the head's own threshold and at the best single offset.

Pass (plan r2): training F1 against clean labels >= 0.85. Read beside it:
step 0 (the run unchanged) against clean labels, the teacher-trained control
on its own labels, and held-out F1, since 300 texts can be memorised.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..annotate.kg import KGItem
from ..kg_judge import POSITIVE, load_texts
from ..label_bank import load as load_bank
from .data import collate
from .gold_eval import HF_REPO, load_run
from .kdata import LabelCollator, teacher_features
from .model import atlop_loss

GRID = np.arange(-4, 6.01, 0.25)


def payload_of(it: dict, facts: list[dict]) -> dict:
    return {"participants": it["participants"], "entities": it["entities"],
            "facts": [{"head": f["head"], "tail": f["tail"], "relation": f["relation"],
                       "polarity": f.get("polarity", "positive"), "modality": "asserted",
                       "holder": None, "time": None, "evidence": []} for f in facts]}


@torch.no_grad()
def build_cache(model, tok, spec, bank, labels: list[dict], device) -> list[dict]:
    texts = {s: {t["id"]: t for t in load_texts(s)} for s in ("train", "dev")}
    coll = LabelCollator(tok, bank, train=False)
    out = []
    t0 = time.time()
    for n, r in enumerate(labels):
        if not r["judged"] and r["facts"]:
            continue                                   # a failed judge call: no clean labels
        it = texts[r["split"]][r["id"]]
        item = KGItem(id=it["id"], tokens=it["tokens"], sentence_spans=it["sentence_spans"],
                      speakers=it.get("speakers"), participants=it["participants"], types=it["types"],
                      relations=it["relations"], domain=it.get("domain") or "")
        clean = [f for f in r["facts"] if f["verdict"] in POSITIVE]
        feats = {}
        for kind, facts in (("clean", clean), ("teacher", it["facts"])):
            f = teacher_features(item, payload_of(it, facts), tok, spec.max_subwords, bank, 1,
                                 keep_unlabelled=1.0)
            if f is None:
                break
            feats[kind] = coll(collate([f], tok.pad_token_id))
        if len(feats) < 2 or len(feats["clean"]["pair_h"]) == 0:
            continue
        b = {k: v.to(device) if hasattr(v, "to") else v for k, v in feats["clean"].items()}
        seq, att = model._marker_pass(b)
        emb = model.label_embeddings(b["rel_label_ids"], b["rel_label_mask"])
        keep = ("ent_batch", "men_ent", "rm_men_pos", "pair_h", "pair_t", "pair_batch",
                "rm_attention_mask", "rel_label_cols")
        assert torch.equal(feats["clean"]["pair_h"], feats["teacher"]["pair_h"])
        out.append({"split": r["split"], "id": r["id"],
                    "seq": seq.half().cpu(), "att": att.half().cpu(), "emb": emb.half().cpu(),
                    "b": {k: feats["clean"][k] for k in keep},
                    "y": {k: feats[k]["pair_labels"][:, 1:].bool() for k in feats}})
        if n % 50 == 0:
            print(f"  cached {n + 1}/{len(labels)} [{time.time() - t0:.0f}s]", flush=True)
            if device.type == "mps":
                torch.mps.empty_cache()
    return out


def logits_of(head, x, device):
    b = {k: v.to(device) for k, v in x["b"].items()}
    lg, _ = head(x["seq"].to(device).float(), x["att"].to(device).float(), b, x["emb"].to(device).float())
    return lg


@torch.no_grad()
def margins(head, items, device) -> list[np.ndarray]:
    head.eval()
    out = []
    for x in items:
        lg = logits_of(head, x, device).float()
        out.append((lg[:, 1:] - lg[:, :1]).cpu().numpy())
    return out


def f1_at(M, Y, d):
    tp = p = g = 0
    for m, y in zip(M, Y):
        pr = m > d
        tp += int((pr & y).sum()); p += int(pr.sum()); g += int(y.sum())
    pr, rc = tp / max(p, 1), tp / max(g, 1)
    return 2 * pr * rc / max(pr + rc, 1e-9), pr, rc


def report(head, items, device) -> dict:
    out = {}
    M = margins(head, items, device)
    for split in ("train", "dev"):
        idx = [i for i, x in enumerate(items) if x["split"] == split]
        Ms = [M[i] for i in idx]
        for lab in ("clean", "teacher"):
            Ys = [items[i]["y"][lab].numpy() for i in idx]
            f0 = f1_at(Ms, Ys, 0.0)
            best = max(GRID, key=lambda d: f1_at(Ms, Ys, d)[0])
            fb = f1_at(Ms, Ys, best)
            out[f"{split}/{lab}"] = {"f1": round(f0[0], 4), "p": round(f0[1], 3), "r": round(f0[2], 3),
                                     "best_f1": round(fb[0], 4), "best_offset": float(best)}
    return out


def line(tag, rep):
    cells = [f"{k} {v['f1']:.3f} (best {v['best_f1']:.3f} @ {v['best_offset']:+.2f})"
             for k, v in rep.items()]
    print(f"{tag:14s} " + " | ".join(cells), flush=True)


def train(head0, items, lab: str, epochs: int, lr: float, accum: int, device, seed: int) -> list:
    head = copy.deepcopy(head0).to(device)
    for p in head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    tr = [x for x in items if x["split"] == "train"]
    rng = np.random.default_rng(seed)
    hist = [{"epoch": 0, **report(head, items, device)}]
    line(f"{lab} ep 0", {k: v for k, v in hist[-1].items() if k != "epoch"})
    for ep in range(1, epochs + 1):
        head.train()
        order = rng.permutation(len(tr))
        tot, n = 0.0, 0
        opt.zero_grad()
        for j, i in enumerate(order):
            x = tr[i]
            lg = logits_of(head, x, device)
            y = x["y"][lab].float().to(device)
            none = (y.sum(1, keepdim=True) == 0).float()
            loss = atlop_loss(lg.float(), torch.cat([none, y], 1)) / accum
            loss.backward()
            tot += float(loss) * accum; n += 1
            if (j + 1) % accum == 0 or j + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step(); opt.zero_grad()
        rep = report(head, items, device)
        hist.append({"epoch": ep, "loss": tot / max(n, 1), **rep})
        line(f"{lab} ep {ep}", rep)
    return hist


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--run", default="ultrabert-380m-v7b8ev")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else
                                       "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(a.seed)
    bank = load_bank()
    model, tok, spec, _, _ = load_run(a.run, HF_REPO, device)
    model.eval()
    labels = json.loads(a.labels.read_text())
    items = build_cache(model, tok, spec, bank, labels, device)
    n_tr = sum(x["split"] == "train" for x in items)
    pos = {k: sum(int(x["y"][k].sum()) for x in items if x["split"] == "train") for k in ("clean", "teacher")}
    print(f"{n_tr} train / {len(items) - n_tr} dev texts; train positives: clean {pos['clean']}, "
          f"teacher {pos['teacher']}", flush=True)
    head0 = model.relation.cpu()
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    res = {"run": a.run, "labels": str(a.labels), "epochs": a.epochs, "lr": a.lr, "accum": a.accum,
           "texts": {"train": n_tr, "dev": len(items) - n_tr}, "positives": pos}
    for lab in ("clean", "teacher"):
        res[lab] = train(head0, items, lab, a.epochs, a.lr, a.accum, device, a.seed)
    out = a.out or a.labels.parent / f"fit-{a.run}-{a.labels.stem}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

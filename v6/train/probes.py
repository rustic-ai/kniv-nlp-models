"""Failure analysis, step 4 (docs/v7-failure-analysis-plan.md): head or
encoder? Probes trained on a model's frozen vectors, against its own head.

    uv run python -m v6.train.probes --runs ultrabert-380m-v7b8ev --parts relations

**Relations** (Re-DocRED, gold entities). Vectors per ordered entity pair
from the frozen model: the relation head's **pair vector** (what its final
scoring reads) and the **encoder vectors** it is built from (head entity,
tail entity, localised context). Probes -- linear and a two-layer MLP,
multi-label over the 96 relations -- train on Re-DocRED train (the head's own
data: every pair with a relation, a fixed sample of the rest), choose their
epoch and threshold on dev, and are scored on test:
- triple F1 (one global threshold chosen on dev; the head scored the same
  way, so the comparison is like for like);
- "is there any relation" for a pair: ROC AUC and best F1, from the best
  relation's margin -- the decision behind most of the head's errors (true
  facts under the threshold, false facts on unrelated pairs).

Reading: a probe that beats the head on its own pair vector -> the head's
scoring or loss; one that beats it only on the encoder vectors -> how the
head builds the pair; none beats it -> the encoder's vectors (step 5).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from ..config import RUNS_DIR
from .gold_eval import HF_REPO, _pairs, apply_calibration, load_run, relation_pass

NEG_KEEP = 0.15          # share of unrelated training pairs kept (all related ones are)
THRESHOLDS = np.round(np.arange(-4.0, 4.01, 0.25), 2)


def _collect(items, feats, outs, idx, rng=None):
    """Per pair: pair vector, encoder vectors, head logits and gold labels
    (multi-hot over the relations); ``rng``: subsample unrelated pairs."""
    X1, X2, L, Y = [], [], [], []
    K = len(idx)
    for it, f, o in zip(items, feats, outs):
        keep = f["_ent_keep"]
        gold = {}
        for h, t, r in (tuple(x) for x in it.layers["rel"]):
            if r in idx:
                gold.setdefault((h, t), set()).add(idx[r])
        for n, (h, t) in enumerate(_pairs(len(keep))):
            g = gold.get((keep[h], keep[t]), set())
            if rng is not None and not g and rng.random() > NEG_KEEP:
                continue
            y = np.zeros(K, dtype=np.uint8)
            y[list(g)] = 1
            X1.append(o["pair"][n]); X2.append(o["pair_in"][n]); L.append(o["relation"][n]); Y.append(y)
    return (np.stack(X1), np.stack(X2), np.stack(L).astype(np.float32), np.stack(Y))


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    pos = y.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def _best_f1(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(-score)
    tp = np.cumsum(y[order]); k = np.arange(1, len(score) + 1)
    return float((2 * tp / (k + y.sum())).max())


def _triple_f1(margin: np.ndarray, Y: np.ndarray, thr: float, top: int = 4) -> float:
    """Decisions as the runtime makes them: a pair's best ``top`` relations
    whose margin beats ``-thr``."""
    rank = np.argsort(-margin, 1)[:, :top]
    m = np.take_along_axis(margin, rank, 1)
    hit = m > -thr
    tp = np.take_along_axis(Y, rank, 1)[hit].sum()
    return float(2 * tp / max(hit.sum() + Y.sum(), 1))


def _score(margin_dev, Y_dev, margin_test, Y_test) -> dict:
    thr = max(THRESHOLDS, key=lambda t: _triple_f1(margin_dev, Y_dev, t))
    any_t = margin_test.max(1)
    return {"triple_f1": _triple_f1(margin_test, Y_test, thr), "threshold": float(thr),
            "any_auc": _auc(any_t, Y_test.any(1)), "any_best_f1": _best_f1(any_t, Y_test.any(1))}


class _Probe(nn.Module):
    def __init__(self, d: int, K: int, hidden: int | None):
        super().__init__()
        self.net = (nn.Linear(d, K + 1) if hidden is None else
                    nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, K + 1)))

    def forward(self, x):
        return self.net(x)


def _atlop(logits, y):
    """ATLOP's adaptive-threshold loss, as the head is trained: column 0 is
    the threshold."""
    lab = torch.cat([torch.zeros_like(y[:, :1]), y], 1).float()
    th = torch.zeros_like(lab); th[:, 0] = 1
    p_mask = lab + th
    n_mask = 1 - lab
    l1 = -(torch.log_softmax(logits - (1 - p_mask) * 1e30, 1) * lab).sum(1)
    l2 = -(torch.log_softmax(logits - (1 - n_mask) * 1e30, 1) * th).sum(1)
    return (l1 + l2).mean()


def train_probe(Xtr, Ytr, Xdv, Ydv, Xte, Yte, hidden, device, epochs: int = 12, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    K = Ytr.shape[1]
    mu = Xtr.astype(np.float32).mean(0); sd = Xtr.astype(np.float32).std(0) + 1e-4
    norm = lambda X: torch.from_numpy(((X.astype(np.float32) - mu) / sd))
    tr_x, tr_y = norm(Xtr), torch.from_numpy(Ytr)
    net = _Probe(Xtr.shape[1], K, hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3 if hidden else 3e-3, weight_decay=0.01)

    def margins(X):
        out = []
        with torch.no_grad():
            net.eval()
            for k in range(0, len(X), 8192):
                lg = net(norm(X[k:k + 8192]).to(device)).float().cpu().numpy()
                out.append(lg[:, 1:] - lg[:, :1])
        return np.concatenate(out)
    best, best_state = -1.0, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(tr_x))
        for k in range(0, len(perm), 1024):
            i = perm[k:k + 1024]
            loss = _atlop(net(tr_x[i].to(device)), tr_y[i].to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        dv = margins(Xdv)
        thr = max(THRESHOLDS, key=lambda t: _triple_f1(dv, Ydv, t))
        f = _triple_f1(dv, Ydv, thr)
        if f > best:
            best, best_state = f, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return _score(margins(Xdv), Ydv, margins(Xte), Yte) | {"dev_triple_f1": best}


def relations(model, tok, spec, vocab, device, train_docs: int | None) -> dict:
    names = vocab.labels["relations"]
    idx = {r: k for k, r in enumerate(names)}
    t0 = time.time()
    data = {}
    for split in ("train", "dev", "test"):
        items, feats, outs = relation_pass(model, tok, spec, vocab, device, split, keep_pairs=True,
                                           limit=train_docs if split == "train" else None)
        data[split] = _collect(items, feats, outs, idx,
                               np.random.default_rng(0) if split == "train" else None)
        del items, feats, outs
        X1, X2, L, Y = data[split]
        print(f"PR relations {split}: {len(Y):,} pairs ({int(Y.any(1).sum()):,} related), pair vector "
              f"{X1.shape[1]}, encoder vectors {X2.shape[1]} [{time.time() - t0:.0f}s]", flush=True)
    res = {"head": _score(data["dev"][2][:, 1:] - data["dev"][2][:, :1], data["dev"][3],
                          data["test"][2][:, 1:] - data["test"][2][:, :1], data["test"][3])}
    print(f"PR relations head: {json.dumps({k: round(v, 4) for k, v in res['head'].items()})}", flush=True)
    for feat, j in (("pair_vector", 0), ("encoder_vectors", 1)):
        for kind, hidden in (("linear", None), ("mlp", 1024)):
            r = train_probe(data["train"][j], data["train"][3], data["dev"][j], data["dev"][3],
                            data["test"][j], data["test"][3], hidden, device)
            res[f"{kind}_{feat}"] = r
            print(f"PR relations {kind} on {feat}: {json.dumps({k: round(v, 4) for k, v in r.items()})} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--parts", default="relations")
    ap.add_argument("--train-docs", type=int, help="first N Re-DocRED training documents (a quick check)")
    ap.add_argument("--device")
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    (RUNS_DIR / "probes").mkdir(parents=True, exist_ok=True)
    for run in args.runs.split(","):
        model, tok, spec, vocab, _ = load_run(run, HF_REPO, device)
        model.eval()
        apply_calibration(model, run, vocab)
        path = RUNS_DIR / "probes" / f"{run}.json"
        res = json.loads(path.read_text()) if path.exists() else {}
        for p in args.parts.split(","):
            res[p] = {"relations": relations}[p](model, tok, spec, vocab, device, args.train_docs)
            path.write_text(json.dumps(res, indent=1))
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

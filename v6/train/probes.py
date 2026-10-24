"""Failure analysis, step 4 (docs/v7-failure-analysis-plan.md): head or
encoder? Probes trained on a model's frozen vectors, against its own head.

    uv run python -m v6.train.probes --runs ultrabert-380m-v7b8ev --parts relations

**Entity types** (gold spans, so boundaries are out of the question). Per
gold span: the span head's **span vector** and the **encoder vector** it is
built from (first word, last word, mean, width). The head's typing accuracy
(its best type among the set's, threshold ignored) against a linear and an
MLP classifier on each -- on Few-NERD trained on Few-NERD train (the head's
data), on each CrossNER domain by 5-fold cross-validation on its test set (a
diagnostic upper bound: what the frozen vectors separate when told the
types; the head is zero-shot there).

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
from .gold_eval import HF_REPO, _pairs, apply_calibration, load_run, phase2_window, run_phase2

NEG_KEEP = 0.15          # share of unrelated training pairs kept (all related ones are)
THRESHOLDS = np.round(np.arange(-4.0, 4.01, 0.25), 2)
# Probes train on the CPU: the MLP probes (LayerNorm) collapse to one class on MPS.
PROBE_DEVICE = torch.device("cpu")
RUN = ""                 # the run being probed (cache file names)


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
    caches = {}
    for split in ("train", "dev", "test"):
        cache = RUNS_DIR / "probes" / f"{RUN}-relations-{split}{'-' + str(train_docs) if train_docs and split == 'train' else ''}.npz"
        caches[split] = cache
        if cache.exists():
            continue
        # One split at a time, written and freed before the next: the vectors of
        # all three held together with a pass's activations run MPS out of memory.
        # 300 documents at a time: the pass keeps every pair's vectors, which for
        # all of a split's documents outgrow memory before they are subsampled.
        from ..gold.redocred import load_rel_items
        docs = load_rel_items(split, limit=train_docs if split == "train" else None)
        rng = np.random.default_rng(0) if split == "train" else None
        parts = []
        for k in range(0, len(docs), 300):
            items = docs[k:k + 300]
            feats = [phase2_window(it.tokens, [tuple(x) for x in it.layers["sentence_spans"]], tok, spec,
                                   vocab, [[(a, e + 1) for a, e in en["mentions"]] for en in it.entities],
                                   markers=model.rel_markers) for it in items]
            outs = run_phase2(model, feats, tok, device, batch=1, keep_pairs=True)
            parts.append(_collect(items, feats, outs, idx, rng))
            del feats, outs
            if device.type == "mps":
                torch.mps.empty_cache()
        X1, X2, L, Y = (np.concatenate([p[j] for p in parts]) for j in range(4))
        del parts
        np.savez(cache, X1=X1, X2=X2, L=L, Y=Y)
        del X1, X2, L, Y
        if device.type == "mps":
            torch.mps.empty_cache()
    for split, cache in caches.items():
        z = np.load(cache)
        data[split] = (z["X1"], z["X2"], z["L"], z["Y"])
        X1, X2, L, Y = data[split]
        print(f"PR relations {split}: {len(Y):,} pairs ({int(Y.any(1).sum()):,} related), pair vector "
              f"{X1.shape[1]}, encoder vectors {X2.shape[1]} [{time.time() - t0:.0f}s]", flush=True)
    res = {"head": _score(data["dev"][2][:, 1:] - data["dev"][2][:, :1], data["dev"][3],
                          data["test"][2][:, 1:] - data["test"][2][:, :1], data["test"][3])}
    print(f"PR relations head: {json.dumps({k: round(v, 4) for k, v in res['head'].items()})}", flush=True)
    for feat, j in (("pair_vector", 0), ("encoder_vectors", 1)):
        for kind, hidden in (("linear", None), ("mlp", 1024)):
            r = train_probe(data["train"][j], data["train"][3], data["dev"][j], data["dev"][3],
                            data["test"][j], data["test"][3], hidden, PROBE_DEVICE)
            res[f"{kind}_{feat}"] = r
            print(f"PR relations {kind} on {feat}: {json.dumps({k: round(v, 4) for k, v in r.items()})} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    return res


@torch.no_grad()
def span_vectors(model, tok, spec, vocab, device, items: list[dict], labels: list[str], describe=None,
                 keys: list | None = None):
    """``keys``: filled with ``(item index, span index)`` of every span returned."""
    """Per gold span (up to the head's width): span vector, encoder vector,
    the head's best type and the gold type (indices into ``labels``)."""
    from . import span_eval as SE
    from .data import blank_window, collate
    from .kdata import TypeCollator, span_targets
    desc = describe or (lambda n: f"a {n}")
    bank = {"entity_types": [{"id": i, "name": n, "description": desc(n)} for i, n in enumerate(labels)]}
    coll = TypeCollator(tok, bank, train=False)
    lid = {n: i for i, n in enumerate(labels)}
    X, R, H, G = [], [], [], []
    for k in range(0, len(items), 16):
        chunk = items[k:k + 16]
        feats = []
        for it in chunk:
            f = blank_window(it["tokens"], [(0, len(it["tokens"]))], tok, spec.max_subwords, vocab.size("cls"))
            f.update(coref=None, rel=None,
                     ents=span_targets([(lid[t], s, e) for t, s, e in it["spans"] if t in lid],
                                       f["_W"], f["word_ok"], list(range(len(labels))), False))
            feats.append(f)
        b = coll(collate(feats, tok.pad_token_id))
        b = {n: v.to(device) for n, v in b.items() if hasattr(v, "to")}
        o = model(b, relation_only=True)
        where = {(int(bb), int(s), int(e)): n for n, (bb, s, e) in enumerate(zip(
            b["span_batch"].tolist(), b["span_start"].tolist(), b["span_end"].tolist()))}
        cols = b["type_label_cols"].tolist()           # column -> bank id (= index into labels)
        lg = o["span"].float()
        for j, it in enumerate(chunk):
            for q, (t, s, e) in enumerate(it["spans"]):
                n = where.get((j, s, e))
                if n is None or t not in lid:
                    continue
                if keys is not None:
                    keys.append((k + j, q))
                X.append(model.span._x[n].half().cpu().numpy())
                R.append(model.span._rep[n].half().cpu().numpy())
                H.append(int(cols[int(lg[n, 1:].argmax())]))   # indices into labels
                G.append(lid[t])
    return np.stack(X), np.stack(R), np.array(H), np.array(G)


def _classifier(Xtr, ytr, Xte, yte, K, hidden, device, epochs: int = 30, seed: int = 0) -> float:
    torch.manual_seed(seed)
    mu = Xtr.astype(np.float32).mean(0); sd = Xtr.astype(np.float32).std(0) + 1e-4
    norm = lambda X: torch.from_numpy((X.astype(np.float32) - mu) / sd).to(device)
    net = _Probe(Xtr.shape[1], K - 1, hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3 if hidden else 3e-3, weight_decay=0.05)
    xt, yt = norm(Xtr), torch.from_numpy(ytr).to(device)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), device=device)
        for k in range(0, len(perm), 512):
            i = perm[k:k + 512]
            loss = nn.functional.cross_entropy(net(xt[i]), yt[i])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return float((net(norm(Xte)).argmax(1).cpu().numpy() == yte).mean())


def spans(model, tok, spec, vocab, device, train_docs: int | None) -> dict:
    from . import span_eval as SE
    from ..gold import fewnerd
    res = {}
    t0 = time.time()
    test = SE.load_fewnerd(3000)
    labels = sorted({t for it in test for t, _, _ in it["spans"]})
    rows = fewnerd.load("train")
    rng = np.random.default_rng(3)
    pick = rng.choice(len(rows), size=min(train_docs or 20000, len(rows)), replace=False)
    fix = {"GPE": "geopolitical entity", "education": "educational institution",
           "government or governmentagency": "government or government agency"}
    name = lambda c: fix.get(fewnerd.type_text(c)[0], fewnerd.type_text(c)[0])
    train = [{"tokens": rows[i]["tokens"], "spans": [(name(c), s, e) for c, s, e in rows[i]["spans"]]}
             for i in pick]
    Xtr, Rtr, _, Gtr = span_vectors(model, tok, spec, vocab, device, train, labels)
    Xte, Rte, Hte, Gte = span_vectors(model, tok, spec, vocab, device, test, labels)
    K = len(labels) + 1
    r = {"spans_train": int(len(Gtr)), "spans_test": int(len(Gte)), "head": float((Hte == Gte).mean())}
    for feat, (a, b) in (("span_vector", (Xtr, Xte)), ("encoder_vector", (Rtr, Rte))):
        for kind, hidden in (("linear", None), ("mlp", 1024)):
            r[f"{kind}_{feat}"] = _classifier(a, Gtr, b, Gte, K, hidden, PROBE_DEVICE)
    res["fewnerd"] = r
    print(f"PR spans fewnerd: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    for d in SE.DOMAINS:
        items = SE.load_crossner(d)
        labels = sorted({t for it in items for t, _, _ in it["spans"]})
        X, R, Hd, G = span_vectors(model, tok, spec, vocab, device, items, labels)
        K = len(labels) + 1
        folds = np.random.default_rng(5).integers(0, 5, len(G))
        r = {"spans": int(len(G)), "types": len(labels), "head": float((Hd == G).mean())}
        for feat, A in (("span_vector", X), ("encoder_vector", R)):
            for kind, hidden in (("linear", None), ("mlp", 512)):
                acc = [_classifier(A[folds != k], G[folds != k], A[folds == k], G[folds == k], K, hidden,
                                   PROBE_DEVICE)
                       * (folds == k).sum() for k in range(5)]
                r[f"{kind}_{feat}_cv"] = float(sum(acc) / len(G))
        res[f"crossner_{d}"] = r
        print(f"PR spans crossner_{d}: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})} "
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
        global RUN
        RUN = run
        path = RUNS_DIR / "probes" / f"{run}.json"
        res = json.loads(path.read_text()) if path.exists() else {}
        for p in args.parts.split(","):
            res[p] = {"relations": relations, "spans": spans}[p](model, tok, spec, vocab, device, args.train_docs)
            path.write_text(json.dumps(res, indent=1))
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train the entity linker: which entity mentions name the same entity.

    uv run python -m v6.train.entity_link --local data/v6-eval/checkpoints \\
        --run R --out-run R-link [--no-upload]

End-to-end relations lost 7-16 points to grouping mentions into entities
(``gold_eval redocred_oracle``): coreference chains fused distinct entities,
and exact-name merging left fragments. The linker (``model.EntityLinker``)
scores mention pairs from the frozen plain pass's entity-span vectors plus
cheap features, and average-link clustering groups them. Head-only: the
encoder and every other head stay as the base run left them.

Data, every source with mentions grouped by entity (gold or distant):
Re-DocRED train (MIT), DocRED distant (MIT; a deterministic subset), T-REx
(CC BY-SA; entities grouped by Wikidata id, a subset). The coreference
feature uses the base model's own chains, as at inference. The clustering
threshold is chosen on Re-DocRED dev end-to-end relation F1 and stored in
``calibration.json`` (``link_threshold``).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..entities import _FUNCTION, _acronym, names_compatible

DOCRED_TYPES = ("LOC", "MISC", "NUM", "ORG", "PER", "TIME")


# ── features and clustering (also used at inference) ────────────────────
def pair_features(tokens, spans, types, chain_of, sent_of):
    """``spans`` ``[(start, end_exclusive)]``, ``types`` per mention (None:
    unknown), ``chain_of`` per mention the coreference chain id or -1,
    ``sent_of`` per mention its sentence. Returns pair index ``[P,2]``
    (i < j), features ``[P,N_FEAT]`` and sentence-distance buckets ``[P]``."""
    n = len(spans)
    names = [tuple(t.lower() for t in tokens[a:b]) for a, b in spans]
    content = [tuple(t for t in nm if t not in _FUNCTION) for nm in names]
    pairs, feats, dist = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = set(content[i]), set(content[j])
            ti, tj = types[i], types[j]
            raw_i, raw_j = tokens[spans[i][0]:spans[i][1]], tokens[spans[j][0]:spans[j][1]]
            short, long_ = (raw_i, raw_j) if len(raw_i) <= len(raw_j) else (raw_j, raw_i)
            feats.append([
                float(names[i] == names[j]),
                float(names_compatible(tokens, spans[i], spans[j])),
                float(_acronym((), short, long_)),
                float(bool(content[i]) and bool(content[j]) and content[i][-1] == content[j][-1]),
                float(bool(content[i]) and bool(content[j]) and content[i][0] == content[j][0]),
                len(ci & cj) / max(len(ci | cj), 1),
                0.5 if ti is None or tj is None else float(ti == tj),
                float(chain_of[i] >= 0 and chain_of[i] == chain_of[j]),
                min(len(names[i]), len(names[j])) / max(len(names[i]), len(names[j]), 1),
                float(spans[i] == spans[j]),
            ])
            pairs.append((i, j))
            dist.append(min(abs(sent_of[i] - sent_of[j]), 7))
    return (np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
            np.asarray(feats, dtype=np.float32).reshape(-1, 10),
            np.asarray(dist, dtype=np.int64))


def cluster(n: int, pairs, probs, threshold: float) -> list[list[int]]:
    """Average-link agglomerative clustering: merge the two groups with the
    highest mean pair probability while it is at least ``threshold``."""
    P = np.zeros((n, n))
    for (i, j), p in zip(pairs, probs):
        P[i, j] = P[j, i] = p
    groups = [[i] for i in range(n)]
    while len(groups) > 1:
        best, bi, bj = threshold, -1, -1
        for x in range(len(groups)):
            for y in range(x + 1, len(groups)):
                s = P[np.ix_(groups[x], groups[y])].mean()
                if s >= best:
                    best, bi, bj = s, x, y
        if bi < 0:
            break
        groups[bi] += groups[bj]
        del groups[bj]
    return groups


def chain_ids(spans, chains) -> list[int]:
    """Each mention's coreference chain (any overlap), or -1."""
    out = [-1] * len(spans)
    for ci, ch in enumerate(chains or []):
        for cs, ce in ch:                        # inclusive word span
            for k, (a, b) in enumerate(spans):
                if out[k] < 0 and a <= ce and cs <= b - 1:
                    out[k] = ci
    return out


class Linker:
    """``build_entities(link=...)``: groups typed mention spans with a model's
    linker head, given the window's link words and coreference chains."""

    def __init__(self, model, tokens, sentence_spans, link_words, chains, threshold: float):
        self.m, self.tokens, self.sp = model, tokens, sentence_spans
        self.words, self.chains, self.thr = link_words, chains, threshold

    def __call__(self, typed):
        spans = [(a, b) for _, a, b in typed]
        W = self.words.size(0)
        ok = [b <= W for a, b in spans]
        idx = [k for k, o in enumerate(ok) if o]
        if len(idx) < 2:
            return [[k] for k in range(len(spans))]
        sub = [spans[k] for k in idx]
        offs = [s for s, _ in self.sp]
        sent = [max(i for i, o in enumerate(offs) if o <= a) for a, _ in sub]
        pairs, feats, dist = pair_features(self.tokens, sub, [typed[k][0] for k in idx],
                                           chain_ids(sub, self.chains), sent)
        lk = self.m.linker
        dev = next(lk.parameters()).device
        with torch.no_grad():
            mv = lk.mention_vectors(self.words.to(dev), sub)
            p = torch.sigmoid(lk(mv, torch.from_numpy(pairs).to(dev),
                                 torch.from_numpy(feats).to(dev),
                                 torch.from_numpy(dist).to(dev))).cpu().numpy()
        groups = [[idx[k] for k in g] for g in cluster(len(sub), pairs, p, self.thr)]
        return groups + [[k] for k, o in enumerate(ok) if not o]


# ── training data ───────────────────────────────────────────────────────
def load_docs(source: str, limit: int, seed: int) -> list[dict]:
    """``{tokens, sentence_spans, entities: [{type, mentions [(a, b_excl)]}]}``."""
    from ..config import DATA_DIR
    if source == "redocred":
        from ..gold.redocred import load_rel_items
        out = []
        for it in load_rel_items("train", limit=limit or None):
            out.append({"tokens": it.tokens, "sentence_spans": [tuple(x) for x in it.layers["sentence_spans"]],
                        "entities": [{"type": en["type"], "mentions": [(a, e + 1) for a, e in en["mentions"]]}
                                     for en in it.entities]})
        return out
    if source == "docred_distant":
        docs = json.loads((DATA_DIR / "v6-corpus" / "docred_distant_rel.json").read_text())["train"]
        out = [{"tokens": d["tokens"], "sentence_spans": [tuple(x) for x in d["sentence_spans"]],
                "entities": [{"type": DOCRED_TYPES[e["type"]] if isinstance(e["type"], int) else e["type"],
                              "mentions": [tuple(m) for m in e["mentions"]]} for e in d["entities"]]}
               for d in docs]
    elif source == "trex":
        docs = json.loads((DATA_DIR / "v6-corpus" / "kg_trex.json").read_text())["docs"]
        out = [{"tokens": d["tokens"], "sentence_spans": [tuple(x) for x in d["sentence_spans"]],
                "entities": [{"type": None, "mentions": [tuple(m) for m in e["mentions"]]}
                             for e in d["entities"]]} for d in docs]
    else:
        raise SystemExit(f"unknown source {source}")
    # Re-DocRED's own cap: the coreference scorer's pairwise tensor grows with
    # the window, and long T-REx abstracts exhausted the MPS pool.
    out = [d for d in out if len(d["tokens"]) <= 512]
    rng = np.random.default_rng(seed)
    keep = sorted(rng.choice(len(out), min(limit, len(out)), replace=False))
    return [out[i] for i in keep]


@torch.no_grad()
def cache(model, tok, spec, vocab, device, docs, max_pairs: int = 3000, seed: int = 0):
    """Per document: mention vectors [M,3H] fp16, pairs, features, distance
    buckets and same-entity labels. The model's plain pass and coreference
    run once per document, through gold_eval's phase-2 path."""
    from .gold_eval import phase2_window, run_phase2
    rng = np.random.default_rng(seed)
    model.emit_link_words = True
    out = []
    feats = [phase2_window(d["tokens"], d["sentence_spans"], tok, spec, vocab, [],
                           markers=model.rel_markers) for d in docs]
    for k in range(0, len(docs), 64):
        res = run_phase2(model, feats[k:k + 64], tok, device, batch=2)
        for d, f, o in zip(docs[k:k + 64], feats[k:k + 64], res):
            W = f["_W"]
            ments, ent, typ = [], [], []
            for ei, e in enumerate(d["entities"]):
                for a, b in e["mentions"]:
                    if b <= W and f["word_ok"][a]:
                        ments.append((a, b)); ent.append(ei); typ.append(e["type"])
            if len(ments) < 2 or len(set(ent)) < 1:
                continue
            offs = [s for s, _ in d["sentence_spans"]]
            sent = [max(i for i, o_ in enumerate(offs) if o_ <= a) for a, _ in ments]
            pairs, pf, dist = pair_features(d["tokens"], ments, typ, chain_ids(ments, o["coref"]), sent)
            y = np.array([ent[i] == ent[j] for i, j in pairs], dtype=np.float32)
            if len(pairs) > max_pairs:              # keep every positive, sample negatives
                pos, neg = np.nonzero(y)[0], np.nonzero(y == 0)[0]
                neg = rng.choice(neg, max(0, max_pairs - len(pos)), replace=False) if len(neg) else neg
                sel = np.sort(np.concatenate([pos, neg]))
                pairs, pf, dist, y = pairs[sel], pf[sel], dist[sel], y[sel]
            mv = model.linker.mention_vectors(o["link_words"], ments) if model.linker is not None \
                else torch.stack([torch.cat([o["link_words"][a], o["link_words"][b - 1],
                                             o["link_words"][a:b].mean(0)]) for a, b in ments])
            out.append({"mv": mv.half(), "pairs": torch.from_numpy(pairs), "feats": torch.from_numpy(pf),
                        "dist": torch.from_numpy(dist), "y": torch.from_numpy(y)})
        if device.type == "mps":
            torch.mps.empty_cache()
        print(f"  cached {min(k + 64, len(docs)):,}/{len(docs):,} documents", flush=True)
    model.emit_link_words = False
    return out


def main() -> int:
    from .gold_eval import CKPT_DIR, HF_REPO, load_run
    from ..train.model import EntityLinker
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-run", required=True)
    ap.add_argument("--local", type=Path)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--redocred", type=int, default=0, help="Re-DocRED train documents (0: all)")
    ap.add_argument("--dev-limit", type=int, default=0, help="Re-DocRED dev documents (0: all)")
    ap.add_argument("--distant", type=int, default=8000, help="DocRED distant documents")
    ap.add_argument("--trex", type=int, default=8000, help="T-REx documents")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    model, tok, spec, vocab, best = load_run(args.run, HF_REPO, device, args.local)
    model.eval()
    t0 = time.time()
    train_docs = (load_docs("redocred", args.redocred, args.seed) + load_docs("docred_distant", args.distant, args.seed)
                  + load_docs("trex", args.trex, args.seed))
    from ..gold.redocred import load_rel_items
    dev_docs = [{"tokens": it.tokens, "sentence_spans": [tuple(x) for x in it.layers["sentence_spans"]],
                 "entities": [{"type": en["type"], "mentions": [(a, e + 1) for a, e in en["mentions"]]}
                              for en in it.entities]} for it in load_rel_items("dev", limit=args.dev_limit or None)]
    tr = cache(model, tok, spec, vocab, device, train_docs, seed=args.seed)
    dv = cache(model, tok, spec, vocab, device, dev_docs, max_pairs=10 ** 9)
    H = tr[0]["mv"].size(1) // 3
    print(f"cached {len(tr):,} train / {len(dv):,} dev documents, "
          f"{sum(len(x['y']) for x in tr):,} train pairs ({sum(float(x['y'].sum()) for x in tr):,.0f} "
          f"positive) in {time.time() - t0:.0f}s", flush=True)
    cpu = torch.device("cpu")
    model.to(cpu)
    if device.type == "mps":
        torch.mps.empty_cache()
    lk = EntityLinker(H, d=args.d)
    opt = torch.optim.AdamW(lk.parameters(), lr=args.lr, weight_decay=0.01)

    def pair_f1(docs, thr=0.5):
        lk.eval(); tp = p_ = g_ = 0
        with torch.no_grad():
            for x in docs:
                pr = torch.sigmoid(lk(x["mv"].float(), x["pairs"], x["feats"], x["dist"])) >= thr
                tp += int((pr & (x["y"] > 0.5)).sum()); p_ += int(pr.sum()); g_ += int((x["y"] > 0.5).sum())
        lk.train()
        return 2 * tp / max(p_ + g_, 1)

    rng = np.random.default_rng(args.seed)
    best_f1, stale = pair_f1(dv), 0
    best_state = {k: v.clone() for k, v in lk.state_dict().items()}
    print(f"dev pair F1 before (untrained): {best_f1:.4f}", flush=True)
    for ep in range(1, args.epochs + 1):
        tot = n = 0
        for k in rng.permutation(len(tr)):
            x = tr[k]
            lg = lk(x["mv"].float(), x["pairs"], x["feats"], x["dist"])
            pos = x["y"].sum().clamp(min=1)
            w = torch.where(x["y"] > 0.5, (len(x["y"]) - pos) / pos, torch.ones_like(x["y"])).clamp(max=20)
            loss = F.binary_cross_entropy_with_logits(lg, x["y"], weight=w)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(lk.parameters(), 1.0); opt.step()
            tot += loss.item(); n += 1
        f1 = pair_f1(dv)
        mark = ""
        if f1 > best_f1:
            best_f1, stale, mark = f1, 0, "  * best"
            best_state = {k: v.clone() for k, v in lk.state_dict().items()}
        else:
            stale += 1
        print(f"epoch {ep}: loss {tot / n:.4f}  dev same-entity pair F1 {f1:.4f}{mark}", flush=True)
        if stale >= args.patience:
            break
    lk.load_state_dict(best_state)
    src = (args.local or CKPT_DIR) / args.run / "best.pt"
    ck = torch.load(src, map_location="cpu", weights_only=False)
    for k, v in lk.state_dict().items():
        ck["model"][f"linker.{k}"] = v.detach().cpu().float()
    ck.setdefault("model_config", {})["linker"] = args.d
    ck["entity_link"] = {"base": args.run, "dev_pair_f1": best_f1, "args": vars(args)}
    out = CKPT_DIR / args.out_run
    out.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out / "best.pt")
    (out / "best.metric").write_text(json.dumps({
        "value": best["value"], "step": best["step"],
        "metric": f"dev composite of the base run; entity linker trained (dev pair F1 {best_f1:.4f})"}))
    print(f"wrote {out / 'best.pt'}", flush=True)
    from .srl_finetune import upload
    upload(out, args.out_run, not args.no_upload, "entity linker")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

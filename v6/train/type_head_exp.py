"""Failure analysis, step 4 for entity typing: can a different typing head or
different training schemas close the gap between the span head (0.55-0.83
typing accuracy on CrossNER's gold spans) and what the frozen vectors hold
(probes 0.84-0.93)? Everything on a model's frozen vectors; CrossNER's types
are never trained, so it stays zero-shot.

    uv run python -m v6.train.type_head_exp --run ultrabert-380m-v7b8ev

Per gold span, the encoder vector the span head reads (first word, last
word, mean, width); per type, its label-pass embedding of ``entity type:
<name>. <description>`` (the label bank's descriptions, as trained). Heads,
trained from scratch on the cached vectors, loss ``-log P(any correct type)``
over the example's schema:
- ``dot``: the span head's form (span MLP . type projection + type bias) --
  the control;
- ``mlp``: an MLP over ``[span; type; span * type]``;
- ``set``: ``mlp`` after the schema's types attend to each other (a
  transformer layer), so a type is scored against its rivals.
Training data:
- ``fn``: Few-NERD train, schema = its 66 fine types (one gold each);
- ``fn+trex``: plus T-REx documents (Wikidata types: each document's own
  type list as the schema, every type of the entity correct) -- many more
  type names, general and specific together;
- ``fnh+trex``: Few-NERD schemas sampled with coarse parents ("person"
  beside "politician") and "miscellaneous"; gold the most specific listed.
Scored: typing accuracy on gold spans -- Few-NERD test (66 fine types), each
CrossNER domain (its own types, "a <name>").
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from ..config import DATA_DIR, RUNS_DIR
from . import span_eval as SE
from .gold_eval import HF_REPO, load_run
from .probes import span_vectors

DEVICE = torch.device("cpu")          # LayerNorm-based heads collapse on MPS
COARSE = {"art": "work of art", "building": "building", "event": "event", "location": "location",
          "organization": "organization", "other": "other named thing", "person": "person",
          "product": "product"}


def type_embeddings(model, tok, device, types: list[tuple[str, str]]) -> np.ndarray:
    out = []
    for k in range(0, len(types), 128):
        texts = [f"entity type: {n}. {d}" if d else f"entity type: {n}" for n, d in types[k:k + 128]]
        enc = tok(texts, truncation=True, max_length=48, padding=True, return_tensors="pt")
        with torch.no_grad():
            out.append(model.label_embeddings(enc["input_ids"].to(device),
                                              enc["attention_mask"].to(device)).float().cpu().numpy())
    return np.concatenate(out)


def fewnerd_items(split: str, n: int, seed: int):
    from ..gold import fewnerd
    rows = fewnerd.load(split)
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(rows), size=min(n, len(rows)), replace=False).tolist())
    return [{"tokens": rows[i]["tokens"], "spans": [(c, s, e) for c, s, e in rows[i]["spans"]]} for i in idx]


def trex_items(n: int, seed: int):
    """T-REx documents: every typed mention as a span; the entity's bank type
    ids and the document's type list kept beside it."""
    d = json.loads((DATA_DIR / "v6-corpus" / "kg_trex.json").read_text())["docs"]
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.choice(len(d), size=min(n, len(d)), replace=False):
        doc = d[int(i)]
        spans, types = [], []
        for e in doc["entities"]:
            if not e.get("types"):
                continue
            for s, t in e["typed"]:
                spans.append(("entity", int(s), int(t)))
                types.append([int(x) for x in e["types"]])
        if spans:
            out.append({"tokens": doc["tokens"], "spans": spans, "_types": types,
                        "_schema": [int(x) for x in doc["types"]]})
    return out


class Head(nn.Module):
    def __init__(self, dr: int, dt: int, kind: str, d: int = 512):
        super().__init__()
        self.kind = kind
        self.span = nn.Sequential(nn.LayerNorm(dr), nn.Linear(dr, d), nn.GELU(), nn.Linear(d, d))
        self.typ = nn.Sequential(nn.LayerNorm(dt), nn.Linear(dt, d))
        if kind == "dot":
            self.bias = nn.Linear(dt, 1)
        else:
            self.score = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        if kind == "set":
            self.mix = nn.TransformerEncoderLayer(d, 8, 2 * d, dropout=0.1, batch_first=True)

    def forward(self, r, t, tmask):
        """``r [B, dr]``, ``t [B, S, dt]``, ``tmask [B, S]`` (True = a type)."""
        x = self.span(r)
        q = self.typ(t)
        if self.kind == "dot":
            s = (q @ x.unsqueeze(-1)).squeeze(-1) + self.bias(t).squeeze(-1)
        else:
            if self.kind == "set":
                q = self.mix(q, src_key_padding_mask=~tmask)
            xx = x.unsqueeze(1).expand_as(q)
            s = self.score(torch.cat([xx, q, xx * q], -1)).squeeze(-1)
        return s.masked_fill(~tmask, -1e4)


def fewnerd_examples(G, fn_ids, coarse_ids, misc, rng, hier: bool):
    """``(row, schema, positives)`` per Few-NERD span; ``G`` indexes ``fn_ids``
    (the 66 fine types' table ids), ``coarse_ids`` each fine type's parent."""
    out = []
    for i, g in enumerate(G):
        g = int(g)
        if not hier:
            out.append((i, fn_ids, {fn_ids[g]}))
            continue
        fine = set(rng.choice(66, size=int(rng.integers(5, 21)), replace=False).tolist())
        (fine.add if rng.random() < 0.75 else fine.discard)(g)
        sch = [fn_ids[f] for f in sorted(fine)]
        parents = {coarse_ids[f] for f in fine if rng.random() < 0.6}
        if rng.random() < 0.7:
            parents.add(coarse_ids[g])
        sch += sorted(parents)
        if rng.random() < 0.5:
            sch.append(misc)
        gold = (fn_ids[g] if fn_ids[g] in sch else coarse_ids[g] if coarse_ids[g] in sch
                else misc if misc in sch else None)
        if gold is not None:
            rng.shuffle(sch)
            out.append((i, sch, {gold}))
    return out


def batchify(R, T, ex, idx):
    S = max(len(ex[i][1]) for i in idx)
    t = np.zeros((len(idx), S, T.shape[1]), dtype=np.float32)
    m = np.zeros((len(idx), S), dtype=bool)
    pos = np.zeros((len(idx), S), dtype=bool)
    rows = []
    for j, i in enumerate(idx):
        row, sch, p = ex[i]
        t[j, :len(sch)] = T[sch]; m[j, :len(sch)] = True
        pos[j, :len(sch)] = [x in p for x in sch]
        rows.append(row)
    return (torch.from_numpy(R[rows].astype(np.float32)), torch.from_numpy(t), torch.from_numpy(m),
            torch.from_numpy(pos))


def evaluate(head, R, T, G) -> float:
    head.eval()
    t = torch.from_numpy(T).float().unsqueeze(0)
    m = torch.ones(1, len(T), dtype=torch.bool)
    acc = 0
    with torch.no_grad():
        for k in range(0, len(R), 1024):
            r = torch.from_numpy(R[k:k + 1024].astype(np.float32))
            s = head(r, t.expand(len(r), -1, -1), m.expand(len(r), -1))
            acc += int((s.argmax(1).numpy() == G[k:k + 1024]).sum())
    return acc / max(len(R), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--fewnerd", type=int, default=30000, help="Few-NERD training sentences")
    ap.add_argument("--trex", type=int, default=4000, help="T-REx training documents")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--heads", default="dot,mlp,set")
    ap.add_argument("--data", default="fn,fn+trex,fnh+trex")
    args = ap.parse_args()
    from ..gold import fewnerd
    from ..label_bank import load as load_bank
    bank = load_bank()["entity_types"]
    by_name = {t["name"]: t["id"] for t in bank}
    fine_codes = sorted(fewnerd.fine_types())
    assert len(fine_codes) == 66
    # The type table: the bank's entity types, then any coarse parent it lacks.
    table = [(t["name"], t.get("description", "")) for t in bank]
    for c in sorted(COARSE):
        if COARSE[c] not in by_name:
            by_name[COARSE[c]] = len(table)
            table.append((COARSE[c], f"a {COARSE[c]}"))
    fn_ids = [by_name[fewnerd.type_text(c)[0]] for c in fine_codes]
    coarse_ids = [by_name[COARSE[c.split("-")[0]]] for c in fine_codes]
    misc = by_name["miscellaneous"]
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, tok, spec, vocab, _ = load_run(args.run, HF_REPO, dev)
    model.eval()
    t0 = time.time()
    cache = RUNS_DIR / "probes" / f"{args.run}-typehead-{args.fewnerd}-{args.trex}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)["data"].item()
    else:
        data = {}
        _, R, _, G = span_vectors(model, tok, spec, vocab, dev, fewnerd_items("train", args.fewnerd, 3),
                                  fine_codes)
        data["fn"] = (R, G)
        tx = trex_items(args.trex, 5)
        keys: list = []
        _, R, _, _ = span_vectors(model, tok, spec, vocab, dev, tx, ["entity"], keys=keys)
        data["trex"] = (R, [(tx[i]["_schema"], tx[i]["_types"][q]) for i, q in keys])
        _, R, _, G = span_vectors(model, tok, spec, vocab, dev, fewnerd_items("test", 3000, 7), fine_codes)
        data["fewnerd"] = (R, G)
        for d in SE.DOMAINS:
            items = SE.load_crossner(d)
            labels = sorted({t for it in items for t, _, _ in it["spans"]})
            _, R, _, G = span_vectors(model, tok, spec, vocab, dev, items, labels)
            data[f"crossner_{d}"] = (R, G, labels)
        np.savez(cache, data=np.array(data, dtype=object))
    T = type_embeddings(model, tok, dev, table)
    evals = {"fewnerd": (data["fewnerd"][0], T[fn_ids], data["fewnerd"][1])}
    for d in SE.DOMAINS:
        R, G, labels = data[f"crossner_{d}"]
        evals[f"crossner_{d}"] = (R, type_embeddings(model, tok, dev, [(n, f"a {n}") for n in labels]), G)
    del model
    Rfn, Gfn = data["fn"]
    Rtx, meta = data["trex"]
    R = np.concatenate([Rfn, Rtx])
    tx_ex = [(len(Rfn) + i, sch, set(ty) & set(sch)) for i, (sch, ty) in enumerate(meta) if set(ty) & set(sch)]
    print(f"TH vectors [{time.time() - t0:.0f}s]: Few-NERD {len(Gfn):,} spans, T-REx {len(tx_ex):,} spans, "
          f"{len({x for _, s, _ in tx_ex for x in s}):,} T-REx types", flush=True)
    res = {}
    for dk in args.data.split(","):
        ex = fewnerd_examples(Gfn, fn_ids, coarse_ids, misc, np.random.default_rng(11), dk.startswith("fnh"))
        if "trex" in dk:
            ex += tx_ex
        for hk in args.heads.split(","):
            torch.manual_seed(0)
            head = Head(R.shape[1], T.shape[1], hk).to(DEVICE)
            opt = torch.optim.AdamW(head.parameters(), lr=5e-4, weight_decay=0.01)
            for ep in range(args.epochs):
                head.train()
                order = np.random.default_rng(ep).permutation(len(ex))
                for k in range(0, len(order), 256):
                    r, t, m, pos = batchify(R, T, ex, order[k:k + 256])
                    s = head(r, t, m)
                    loss = (torch.logsumexp(s, 1) - torch.logsumexp(s.masked_fill(~pos, -1e4), 1)).mean()
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            r = {n: evaluate(head, *v) for n, v in evals.items()}
            r["crossner_mean"] = float(np.mean([r[f"crossner_{d}"] for d in SE.DOMAINS]))
            res[f"{hk}/{dk}"] = r
            print(f"TH {args.run} {hk}/{dk}: " + "  ".join(f"{k} {v:.3f}" for k, v in r.items())
                  + f" [{time.time() - t0:.0f}s]", flush=True)
            (RUNS_DIR / "failure" / f"{args.run}-typehead.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

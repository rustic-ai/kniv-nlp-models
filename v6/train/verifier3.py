"""List verifier (plan r4 V3): candidates written into a shared pass.

    uv run python -m v6.train.verifier3 prep  [--out-run RUN-ver3] [--limit N] [--upload]   # GPU: lists and soft targets
    uv run python -m v6.train.verifier3 smoke [--device cpu]
    uv run python -m v6.train.verifier3 train --out-run RUN-ver3 [--probe 40] [--upload]

Why. V1c works because the candidate is written into the text and the full
encoder reads both; V2 (a reader over a pass without the candidate) could
not recover that. V3 keeps the candidates in the pass and shares the pass.

Pass. A chunk of text, then a list of candidates, then a legend:

    [CLS] chunk with every mention between * [SEP] h1 | r1 | t1 [SEP] h2 | r2 | t2 [SEP] ... r1: desc ; r2: desc [SEP]

The chunk is the sentence, or the shortest run of sentences, holding a
mention of both arguments; a chunk longer than ``cap`` subwords, a
conversation, and a broken-argument variant use the whole window (V1c's
window form). Each candidate's slot is its segment (its words and the
``[SEP]`` after them). A pass with one candidate is written exactly as V1c
writes its input (``h | r | t | description [SEP]``, no legend), so on the
window form it is V1c's input. Candidates per pass: as many as fit
``budget`` candidate subwords, the pass at most 1,024.

Model. V1c's computation per slot: the run's frozen encoder through V3's
own adapter (started from V1c's), the relation head's pooling for the
slot's pair (entity vectors from the chunk's start markers, the context
from the pair's attention), the mean of the slot's segment, V1c's head
(started from V1c's) to the eight verdicts.

Lists (``prep``, on the GPU). Per record of ``verifier_v1c.json``, the
candidates the extractor sends to the verifier on the record's entities
and schema (d123, floor 3, the deployed per-pair cut), each with V1c's
probabilities. In training, a judged or rule-labelled candidate keeps its
label; every other candidate of the list takes V1c's probabilities as a
soft target. Long windows keep at most ``max_extra`` unlabelled candidates,
those in chunks holding a labelled one first. Lists are shuffled in
training. Written to ``data/v6-corpus/verifier_v3_lists.json`` (and to the
runs repo with ``--upload``).

Readings on V1c's held-out examples (the same records and candidates,
each inside its own list): AUC on expressed / not against candidates seen,
AUC by the number of candidates in the pass, and the share of verdicts
(and of keep / drop at 0.6) that change when every list is reshuffled.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .verifier import (DATA, HEDGED_ANY, VERDICTS, VERDICTS_K, _corpus, _markers, coarse, encode, k_loss)
from .verifier2 import v1c_held_keys

RUN, V1C = "ultrabert-380m-v8p3-d123", "ultrabert-380m-v8p3-d123-ver1c"
LISTS = "verifier_v3_lists.json"
SOFT = -2           # label of a candidate trained on V1c's probabilities
MAX_PASS = 1024


# ── schemas ────────────────────────────────────────────────────────
def _schemas() -> dict:
    """record id -> schema relation names (kg_short texts, their minimal-pair
    variants, the stage-4 windows); records outside them use their own
    candidates' relations."""
    out = {}
    S = json.loads((_corpus() / "kg_short.json").read_text())
    import ast
    for sp in ("train", "dev"):
        for e in S[sp]:
            r = e["relations"]
            out[e["id"]] = ast.literal_eval(r) if isinstance(r, str) else list(r)
    p = Path("v6/runs/s4-pilot/items.json")
    if p.exists():
        for it in json.loads(p.read_text()):
            out[it["id"]] = [r["name"] for r in it["relations"]]
    return out


def schema_of(rec, schemas):
    rid = rec["id"]
    s = schemas.get(rid) or schemas.get(rid.split(":var:")[0])
    return s if s else sorted({c["rel"] for c in rec["cands"]})


# ── prep: the extractor's lists and V1c's probabilities ────────────
def prep(args) -> int:
    from ..infer import Extractor
    from ..label_bank import load as load_bank
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bank = {r["name"]: r for r in load_bank()["relations"]}
    recs = json.loads((_corpus() / args.data).read_text())["records"]
    if args.merge:
        return merge_parts(args)
    # held-out records: only those V1c's held-out examples read
    from .gold_eval import load_run
    _, tok0, *_ = load_run(RUN, args.repo, torch.device("cpu"))
    need = {recs[i]["id"] for sp in ("held", "held_long") for i in v1c_held_keys(recs, tok0, bank, sp, args.dev_limit)}
    recs = [r for r in recs if r["split"] == "train" or r["id"] in need]
    if args.limit:
        recs = recs[:args.limit]
    recs = [r for k, r in enumerate(recs) if k % args.shards == args.shard]
    schemas = _schemas()
    ex = Extractor(RUN, device=str(device))
    ex.use_verifier(V1C, threshold=0.6)
    ex.margin_shift = 0.0
    cap = {}
    orig = ex._verify

    def grab(w, nodes, spans, parts, cands):
        cap["nodes"], cap["cands"] = nodes, list(cands)
        ex._last_verdicts = {}
        return [None] * len(cands)
    ex._verify = grab
    out, t0, n_all, n_extra = {}, time.time(), 0, 0
    queue: list = []
    t_rel = t_ver = 0.0
    from .verifier import collate as vcollate

    def flush(q):
        """V1c over queued candidates, sorted by length, batches within a token budget."""
        q = sorted(q, key=lambda x: len(x[2]["ids"]))
        i = 0
        while i < len(q):
            L = len(q[i][2]["ids"])
            k = max(1, min(256, (256 * 128 * 128) // max(L * L, 1)))
            chunk = q[i:i + k]
            k = len(chunk)
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                lg = ex.verifier(vcollate([c[2] for c in chunk], ex.tok.pad_token_id, device))
            for (rid, (a, b, r), _), pr in zip(chunk, torch.softmax(lg.float(), -1).cpu().tolist()):
                out.setdefault(rid, []).append({"h": a, "t": b, "rel": r, "soft": [round(x, 5) for x in pr]})
            i += k
        return len(q)
    for n, rec in enumerate(recs):
        parts = rec.get("participants") or []
        P = len(parts)
        w = {"tokens": rec["tokens"], "sentence_spans": rec["sentence_spans"]}
        if rec.get("speakers"):
            w["sentence_speakers"] = rec["speakers"]
        ents = [{"type": "entity", "mentions": [tuple(m) for m in ms]} for ms in rec["entities"][P:]]
        rels = [{"name": x, "description": bank.get(x, {}).get("description", "")} for x in schema_of(rec, schemas)]
        cap.clear()
        t1 = time.time()
        with torch.no_grad():
            ex._relations(w, ents, rels, ex.top)
        t_rel += time.time() - t1
        if not cap:
            continue
        # node index -> record entity index (speakers by name, entities by spans)
        span_ix = {tuple(sorted(tuple(m) for m in ms)): P + k for k, ms in enumerate(rec["entities"][P:])}
        node_ix = []
        for nd in cap["nodes"]:
            if nd.get("speaker"):
                node_ix.append(parts.index(nd["text"]) if nd["text"] in parts else None)
            else:
                node_ix.append(span_ix.get(tuple(sorted(tuple(m) for m in nd["spans"]))))
        labelled = {(c["h"], c["t"], c["rel"]) for c in rec["cands"] if not c.get("swap")}
        extra = []
        for h, t, r in cap["cands"]:
            a, b = node_ix[h], node_ix[t]
            if a is None or b is None or (a, b, r) in labelled:
                continue
            extra.append((a, b, r))
        n_all += len(cap["cands"])
        if len(extra) > args.max_extra:             # long windows: chunks holding a labelled candidate first
            near = {x for c in rec["cands"] for x in (c["h"], c["t"])}
            extra.sort(key=lambda c: (not (c[0] in near or c[1] in near), random.Random(f"{rec['id']}{c}").random()))
            extra = extra[:args.max_extra]
        if extra:                                   # V1c's probabilities, its own input form: queued, run in batches
            cache, keep = {}, []
            for a, b, r in extra:
                e = encode(rec, {"h": a, "t": b, "rel": r, "y": 0}, ex.tok, bank, None, None, cache=cache)
                if e is not None:
                    queue.append((rec["id"], (a, b, r), e))
        if len(queue) >= args.queue or n == len(recs) - 1:
            t1 = time.time()
            n_extra += flush(queue)
            t_ver += time.time() - t1
            queue = []
        if n % 2000 == 0:
            print(f"prep {n}/{len(recs)}: {n_all:,} extractor candidates, {n_extra:,} unlabelled with V1c's "
                  f"probabilities [{time.time() - t0:.0f}s: relation step {t_rel:.0f}s, V1c {t_ver:.0f}s]", flush=True)
    if queue:
        n_extra += flush(queue)
    ex._verify = orig
    path = _corpus() / LISTS
    if args.shards > 1:
        path = path.with_suffix(f".part{args.shard}.json")
    path.write_text(json.dumps({"run": RUN, "verifier": V1C, "max_extra": args.max_extra, "extra": out}))
    print(f"prep done: {len(recs):,} records, {n_all:,} extractor candidates, {n_extra:,} unlabelled kept "
          f"-> {path}", flush=True)
    if args.shards > 1:
        return 0
    return _upload(args, path)


def merge_parts(args) -> int:
    path = _corpus() / LISTS
    out = {}
    for k in range(args.shards):
        d = json.loads(path.with_suffix(f".part{k}.json").read_text())
        out.update(d["extra"])
        meta = d
    path.write_text(json.dumps({**meta, "extra": out}))
    print(f"prep merged: {len(out):,} records with unlabelled candidates, "
          f"{sum(len(v) for v in out.values()):,} candidates -> {path}", flush=True)
    return _upload(args, path)


def _upload(args, path) -> int:
    if args.upload:
        from huggingface_hub import HfApi
        gz = path.with_suffix(".json.gz")
        gz.write_bytes(gzip.compress(path.read_bytes()))
        HfApi().upload_file(path_or_fileobj=str(gz), path_in_repo=f"{args.out_run}/{gz.name}", repo_id=args.repo,
                            commit_message=f"Add {args.out_run} list data")
        print(f"uploaded {gz.name}", flush=True)
    return 0


def load_lists(args) -> dict:
    path = _corpus() / LISTS
    if not path.exists():
        from huggingface_hub import hf_hub_download
        gz = Path(hf_hub_download(args.repo, f"{args.out_run}/{LISTS.replace('.json', '.json.gz')}",
                                  local_dir=str(_corpus().parent / "v6-eval" / "checkpoints")))
        path.write_bytes(gzip.decompress(gz.read_bytes()))
    return json.loads(path.read_text())["extra"]


# ── passes ─────────────────────────────────────────────────────────
def _chunk(rec, P, h, t):
    """Shortest sentence run (first, last) holding a mention of both."""
    sp = rec["sentence_spans"]
    so = {}
    for k, (a, b) in enumerate(sp):
        for i in range(a, b):
            so[i] = k
    best = None
    last = len(sp) - 1
    for a in rec["entities"][h]:
        for b in rec["entities"][t]:
            # every sentence either mention touches (a mention can cross a boundary)
            ks = [so.get(x, 0 if x < sp[0][0] else last) for x in (a[0], a[1] - 1, b[0], b[1] - 1)]
            s = (min(ks), max(ks))
            if best is None or s[1] - s[0] < best[1] - best[0]:
                best = (s[0], s[1])
    return best


def _chunk_rec(rec, P, a, b):
    """The chunk as a record (tokens, spans, entities inside it) and the map
    from the record's entity index to the chunk's."""
    sp = rec["sentence_spans"]
    s0, e0 = sp[a][0], sp[b][1]
    ents, names, ix = [], [], {}
    for k, ms in enumerate(rec["entities"]):
        if k < P:
            continue
        inside = [(m[0] - s0, m[1] - s0) for m in ms if m[0] >= s0 and m[1] <= e0]
        if inside:
            ix[k] = len(ents); ents.append(inside); names.append(rec["names"][k])
    return ({"tokens": rec["tokens"][s0:e0], "sentence_spans": [[x - s0, y - s0] for x, y in sp[a:b + 1]],
             "speakers": None, "participants": [], "entities": ents, "names": names}, ix)


def _segments(cands, rec_names, bank, tok):
    """Each candidate's subwords: ``h | r | t`` (one candidate: V1c's
    ``h | r | t | description``)."""
    out = []
    for c in cands:
        out.append(tok(f"{rec_names[c['h']]} | {c['rel']} | {rec_names[c['t']]}", add_special_tokens=False)["input_ids"])
    return out


def _legend(rels, bank, tok):
    txt = " ; ".join(f"{r}: {bank.get(r, {}).get('description', '')}" for r in rels)
    return tok(txt, add_special_tokens=False)["input_ids"][:192]


def build_passes(rec, cands, tok, bank, budget, cap, rng=None, cache=None):
    """One record's passes for ``cands`` (each ``{h, t, rel, y}`` or with
    ``soft``, and optionally ``swap``, ``key``). ``rng``: shuffle the lists."""
    P = len(rec.get("participants") or [])
    groups = defaultdict(list)
    for c in cands:
        if c.get("swap"):
            groups[("swap", c["swap"][0], tuple(c["swap"][1]))].append(c)
        elif P or rec.get("speakers"):
            groups[("window",)].append(c)
        else:
            groups[("chunk",) + _chunk(rec, P, c["h"], c["t"])].append(c)
    cache = {} if cache is None else cache
    out = []
    for key, cs in groups.items():
        if key[0] == "chunk":
            if key not in cache:
                cr, ix = _chunk_rec(rec, P, key[1], key[2])
                rm = _markers(cr, tok)
                cache[key] = (cr, ix, rm) if len(rm["ids"]) <= cap else None
            if cache[key] is None:
                key = ("window",)
        if key[0] == "chunk":
            cr, ix, rm = cache[key]
            cs = [dict(c, h=ix[c["h"]], t=ix[c["t"]]) for c in cs]
            names = cr["names"]
        else:
            wk = key
            if wk not in cache:
                sw = None if key[0] == "window" else [key[1], list(key[2])]
                rm = _markers(rec, tok, sw)
                names = list(rec["names"])
                if sw is not None:
                    names[sw[0]] = " ".join(rec["tokens"][sw[1][0]:sw[1][1]])
                cache[wk] = (None, None, rm, names)
            _, _, rm, names = cache[wk]
        men = rm["men_pos"]
        cs = [c for c in cs if not any(len(men[k]) == 0 or men[k].max() < 0 for k in (c["h"], c["t"]))]
        if not cs:
            continue
        if rng is not None:
            rng.shuffle(cs)
        segs = _segments(cs, names, bank, tok)
        # pack: candidate subwords within budget, the pass within MAX_PASS
        cur, cur_n = [], 0
        packs = []
        for c, s in zip(cs, segs):
            if cur and (cur_n + len(s) + 1 > budget):
                packs.append(cur); cur, cur_n = [], 0
            cur.append((c, s)); cur_n += len(s) + 1
        if cur:
            packs.append(cur)
        base = [int(x) for x in rm["ids"]]
        for pk in packs:
            if len(pk) == 1:                        # V1c's input exactly
                c, _ = pk[0]
                rel = bank.get(c["rel"], {"name": c["rel"], "description": ""})
                seg = tok(f"{names[c['h']]} | {rel['name']} | {names[c['t']]} | {rel.get('description', '')}",
                          add_special_tokens=False)["input_ids"][:96]
                ids = base + seg + [tok.sep_token_id]
                slots = [(len(base), len(ids) - 1)]        # V1c's candidate mask: its words, not the [SEP]
            else:
                ids, slots = list(base), []
                for c, s in pk:
                    a = len(ids); ids += s + [tok.sep_token_id]; slots.append((a, len(ids)))
                ids += _legend(sorted({c["rel"] for c, _ in pk}), bank, tok) + [tok.sep_token_id]
            if len(ids) > MAX_PASS:
                if len(pk) == 1:
                    continue
                half = len(pk) // 2                 # too long: split the list
                for sub in (pk[:half], pk[half:]):
                    out += build_passes(rec, [c for c, _ in sub], tok, bank, budget, cap, None, cache)
                continue
            men_ent, men_pos = [], []
            for k, pos in enumerate(men):
                for q in (pos.tolist() if len(pos) else [-1]):
                    men_ent.append(k); men_pos.append(q)
            out.append({"ids": ids, "men_pos": men_pos, "men_ent": men_ent, "E": len(men), "id": rec["id"],
                        "slots": [{"h": c["h"], "t": c["t"], "a": a, "b": b, "y": c.get("y", SOFT),
                                   "soft": c.get("soft"), "key": c.get("key"), "src": c.get("src", ""), "n": len(pk)}
                                  for (c, _), (a, b) in zip(pk, slots)]})
    return out


def record_passes(rec, extra, tok, bank, budget, cap, rng=None, keep=None, cache=None):
    """A record's labelled candidates (``keep``: their indices, else all) and
    its unlabelled list members (training: all; evaluation: those in a chunk
    with a kept candidate, so each kept candidate sits in its own list)."""
    cands = []
    for j, c in enumerate(rec["cands"]):
        if keep is not None and j not in keep:
            continue
        cands.append({**c, "key": (rec["id"], j)})
    for e in extra or []:
        cands.append({"h": e["h"], "t": e["t"], "rel": e["rel"], "y": SOFT, "soft": e["soft"]})
    ps = build_passes(rec, cands, tok, bank, budget, cap, rng, cache)
    if keep is not None:
        ps = [p for p in ps if any(s["key"] is not None for s in p["slots"])]
    return ps


# ── model ──────────────────────────────────────────────────────────
class Verifier3(nn.Module):
    def __init__(self, base, r: int = 64, hidden: int = 512, n_out: int = len(VERDICTS_K)):
        super().__init__()
        from .model import RelationAdapter
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.ver = RelationAdapter(base.encoder, r)
        H = base.encoder.config.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(4 * H), nn.Linear(4 * H, hidden), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(hidden, n_out))

    def warm_start(self, ck: dict):
        """V1c's adapter and head."""
        self.ver.load_state_dict(ck["ver"]); self.head.load_state_dict(ck["head"])

    def trainable(self):
        return list(self.ver.parameters()) + list(self.head.parameters())

    def forward(self, b: dict) -> torch.Tensor:
        base = self.base
        if getattr(base, "rel_lora", None) is not None:
            base.rel_lora.on = False
        self.ver.on = True
        try:
            seq, att = base._encode_with_attn(b["ids"], b["mask"])
        finally:
            self.ver.on = False
        base.relation.pair_features(seq, att, b)
        hs, ts, rs = base.relation._pair_in
        cs = torch.cumsum(seq.float(), 1)                                        # slot means by prefix sums
        pb, sa, sb = b["pair_batch"], b["seg_a"], b["seg_b"]
        tot = cs[pb, sb - 1] - torch.where((sa > 0).unsqueeze(-1), cs[pb, (sa - 1).clamp(min=0)], torch.zeros_like(cs[pb, 0]))
        cand = tot / (sb - sa).clamp(min=1).unsqueeze(-1).float()
        x = torch.cat([hs.float(), ts.float(), rs.float(), cand], -1)
        return self.head(x).float()


def collate(ps: list[dict], pad: int, device) -> dict:
    T = max(len(p["ids"]) for p in ps)
    ids = torch.full((len(ps), T), pad, dtype=torch.long)
    mask = torch.zeros(len(ps), T, dtype=torch.long)
    ent_batch, men_ent, men_pos, ph, pt, pb, sa, sb, ys, soft, smask = ([] for _ in range(11))
    for b, p in enumerate(ps):
        n = len(p["ids"])
        ids[b, :n] = torch.tensor(p["ids"]); mask[b, :n] = 1
        base = len(ent_batch)
        ent_batch += [b] * p["E"]
        men_ent += [base + k for k in p["men_ent"]]; men_pos += p["men_pos"]
        for s in p["slots"]:
            ph.append(base + s["h"]); pt.append(base + s["t"]); pb.append(b); sa.append(s["a"]); sb.append(s["b"])
            ys.append(s["y"])
            soft.append(s["soft"] if s["soft"] is not None else [0.0] * len(VERDICTS_K))
    out = {"ids": ids, "mask": mask, "ent_batch": torch.tensor(ent_batch), "men_ent": torch.tensor(men_ent),
           "rm_men_pos": torch.tensor(men_pos), "pair_h": torch.tensor(ph), "pair_t": torch.tensor(pt),
           "pair_batch": torch.tensor(pb), "seg_a": torch.tensor(sa), "seg_b": torch.tensor(sb),
           "y": torch.tensor(ys), "soft": torch.tensor(soft, dtype=torch.float32)}
    return {k: v.to(device) for k, v in out.items()}


def loss_fn(lg, y, soft, w, soft_weight):
    lab = y != SOFT
    loss = lg.new_zeros(())
    denom = 0.0
    if lab.any():
        l1 = k_loss(lg[lab], y[lab], w)
        loss = loss + l1 * float(lab.sum())
        denom += float(lab.sum())
    if (~lab).any() and soft_weight > 0:
        lp = torch.log_softmax(lg[~lab].float(), -1)
        q = soft[~lab]
        q = q[:, :lp.size(1)] if q.size(1) >= lp.size(1) else q
        l2 = -(q * lp).sum(-1).mean()
        loss = loss + soft_weight * l2 * float((~lab).sum())
        denom += soft_weight * float((~lab).sum())
    return loss / max(denom, 1.0)


# ── evaluation and training ────────────────────────────────────────
def _batches(ps, max_tokens2: int, max_n: int = 32):
    """Passes in batches within ``sum`` of length² (attention maps kept)."""
    i = 0
    while i < len(ps):
        k = 1
        while i + k < len(ps) and k < max_n and (k + 1) * max(len(p["ids"]) for p in ps[i:i + k + 1]) ** 2 <= max_tokens2:
            k += 1
        yield ps[i:i + k]
        i += k


def _auc(s, y):
    o = np.asarray(s).argsort().argsort() + 1
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    return float((o[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


@torch.no_grad()
def predict(model, ps, pad, device, max_tokens2):
    """Per kept slot (``key``): verdict probabilities and the pass size."""
    model.eval()
    out = {}
    for chunk in _batches(sorted(ps, key=lambda p: len(p["ids"])), max_tokens2):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(collate(chunk, pad, device))
        pr = torch.softmax(lg.float(), -1).cpu().numpy()
        j = 0
        for p in chunk:
            for s in p["slots"]:
                if s["key"] is not None:
                    out[s["key"]] = (pr[j], s["n"], s["y"])
                j += 1
    model.train()
    return out


def read(pred: dict) -> dict:
    P = np.stack([v[0] for v in pred.values()])
    Y = np.array([v[2] for v in pred.values()])
    N = np.array([v[1] for v in pred.values()])
    Pc = coarse(P)
    Yc = np.where((Y >= 3) | (Y == HEDGED_ANY), 3, Y)
    ex = 1 - Pc[:, 1]
    res = {"n": len(Y), "auc_expressed": _auc(ex, Yc != 1), "acc": float((Pc.argmax(1) == Yc).mean())}
    for lo, hi in ((1, 1), (2, 4), (5, 8), (9, 16), (17, 999)):
        sel = (N >= lo) & (N <= hi)
        if sel.sum() >= 50 and 0 < (Yc[sel] != 1).sum() < sel.sum():
            res[f"auc_n{lo}-{hi}"] = _auc(ex[sel], Yc[sel] != 1)
            res[f"count_n{lo}-{hi}"] = int(sel.sum())
    return res


def _setup(args, device):
    from .gold_eval import load_run
    from ..label_bank import load as load_bank
    base, tok, spec, vocab, _ = load_run(args.run, args.repo, device, args.local)
    base.eval()
    bank = {r["name"]: r for r in load_bank()["relations"]}
    recs = json.loads((_corpus() / args.data).read_text())["records"]
    return base, tok, bank, recs


def _v1c_ck(args):
    from .gold_eval import CKPT_DIR, HF_REPO
    from huggingface_hub import hf_hub_download
    p = CKPT_DIR / V1C / "verifier.pt"
    if not p.exists():
        p = Path(hf_hub_download(args.repo or HF_REPO, f"{V1C}/verifier.pt", local_dir=str(CKPT_DIR)))
    return torch.load(p, map_location="cpu", weights_only=False)


def held_sets(recs, extra, tok, bank, args):
    """V1c's exact held-out examples, each in its list (``rng`` for order)."""
    out = {}
    for split in ("held", "held_long"):
        keys = v1c_held_keys(recs, tok, bank, split, args.dev_limit)
        out[split] = (keys, lambda rng, keys=keys: [p for i, ks in keys.items() for p in record_passes(
            recs[i], extra.get(recs[i]["id"]), tok, bank, args.budget, args.cap, rng, keep=ks)])
    return out


def train(args) -> int:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    base, tok, bank, recs = _setup(args, device)
    extra = load_lists(args)
    # the lists are keyed by record id, and 1,405 ids are shared by two training
    # records with different entities: those records train on their labelled
    # candidates only
    ids = Counter(r["id"] for r in recs)
    dup = {k for k, v in ids.items() if v > 1}
    n_dup = sum(len(extra.pop(k, [])) for k in dup)
    print(f"{len(dup):,} ids shared by two records: their {n_dup:,} unlabelled list candidates dropped", flush=True)
    verdicts = VERDICTS_K
    rng = random.Random(args.seed)
    train_recs = [r for r in recs if r["split"] == "train"]
    if args.limit:
        train_recs = train_recs[:args.limit]
    H = held_sets(recs, extra, tok, bank, args)
    held = {s: f(random.Random(1)) for s, (_, f) in H.items()}
    print(f"{len(train_recs):,} training records; held out as V1c: "
          + ", ".join(f"{s} {sum(1 for p in ps for x in p['slots'] if x['key'] is not None):,} kept candidates "
                      f"in {len(ps):,} passes" for s, ps in held.items()), flush=True)
    model = Verifier3(base, r=args.rank, n_out=len(verdicts)).to(device)
    model.warm_start(_v1c_ck(args))
    opt = torch.optim.AdamW([{"params": list(model.ver.parameters()), "lr": args.lr},
                             {"params": list(model.head.parameters()), "lr": args.head_lr}], weight_decay=0.01)
    w = torch.tensor([1.0, 1.0, args.neg_weight] + [args.hedge_weight] * (len(verdicts) - 3), device=device)
    out = Path(args.out_dir or f"data/v6-eval/checkpoints/{args.out_run}")
    out.mkdir(parents=True, exist_ok=True)
    budget2 = args.batch_tokens ** 2 * args.batch
    # steps from the target number of candidates: an estimate per pass, refined as it runs
    est = sum(len(r["cands"]) + len(extra.get(r["id"], [])) for r in train_recs)
    print(f"{est:,} list candidates per epoch; target {args.candidates:,} labelled candidates seen", flush=True)
    seen, seen_all, step, best, t0, peak = 0, 0, 0, -1.0, time.time(), 0.0
    curve = []

    def evaluate():
        res = {}
        for s, ps in held.items():
            r = read(predict(model, ps, tok.pad_token_id, device, budget2))
            res.update({(f"long_{k}" if s == "held_long" else k): v for k, v in r.items()})
        return res
    res = evaluate()                     # step 0: the warm start
    curve.append({"seen": 0, **res})
    print(f"eval 0 (warm start, V1c's adapter and head): {json.dumps({k: round(v, 4) for k, v in res.items()})}", flush=True)
    order: list = []
    buf: list = []
    if args.probe:          # the worst batches first: the longest training records' passes, longest popped first
        cache = {}
        for r in sorted(train_recs, key=lambda r: -len(r["tokens"]))[:300]:
            buf += record_passes(r, extra.get(r["id"]), tok, bank, args.budget, args.cap, rng, cache=cache)
            cache = {}
        buf.sort(key=lambda p: (len(p["ids"]), len(p["slots"])))
    while seen < args.candidates:
        if not buf:
            if not order:
                order = list(range(len(train_recs))); rng.shuffle(order)
            cache = {}
            while order and len(buf) < 512:
                r = train_recs[order.pop()]
                buf += record_passes(r, extra.get(r["id"]), tok, bank, args.budget, args.cap, rng, cache=cache)
                cache = {}
            rng.shuffle(buf)
        # one batch within the length² budget
        bt, n2 = [], 0
        while buf and len(bt) < args.batch * 4:
            L = len(buf[-1]["ids"])
            if bt and (len(bt) + 1) * max(L, *(len(p["ids"]) for p in bt)) ** 2 > budget2:
                break
            bt.append(buf.pop())
        b = collate(bt, tok.pad_token_id, device)
        # learning rate by candidates seen: 3% warm-up, then linear to zero at the target
        f = min(1.0, seen / (0.03 * args.candidates)) * max(0.0, 1 - seen / args.candidates)
        for g, lr in zip(opt.param_groups, (args.lr, args.head_lr)):
            g["lr"] = lr * max(f, 0.02)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(b)
        loss = loss_fn(lg, b["y"], b["soft"], w, args.soft_weight)
        step += 1
        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss, skipped", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable(), 1.0)
        opt.step()
        n_before = seen
        seen += int((b["y"] != SOFT).sum())        # labelled candidates: V1c's data, as V1c saw it
        seen_all += len(b["y"])
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated() / 2 ** 30)
        if step % 50 == 0 or step <= args.probe:
            print(f"step {step} seen {seen:,} (all {seen_all:,}) loss {loss.item():.4f} passes {len(bt)} T {b['ids'].size(1)} "
                  f"cands {len(b['y'])} peak {peak:.1f} GiB [{time.time() - t0:.0f}s]", flush=True)
        if args.probe and step >= args.probe:
            print(f"probe done: {step} steps, peak {peak:.1f} GiB, {(time.time() - t0) / step:.2f} s per step, "
                  f"{seen / step:.0f} candidates per step")
            return 0
        if seen // args.eval_every > n_before // args.eval_every or seen >= args.candidates:
            res = evaluate()
            curve.append({"seen": seen, "seen_all": seen_all, **res})
            print(f"eval {step} seen {seen:,}: {json.dumps({k: round(v, 4) for k, v in res.items()})}", flush=True)
            if res["auc_expressed"] > best:
                best = res["auc_expressed"]
                torch.save({"form": "v3", "ver": model.ver.state_dict(), "head": model.head.state_dict(),
                            "base_run": args.run, "rank": args.rank, "step": step, "seen": seen, "eval": res,
                            "data": args.data, "verdicts": verdicts, "budget": args.budget, "cap": args.cap},
                           out / "verifier.pt")
            (out / "curve.json").write_text(json.dumps(curve))
            (out / "status.json").write_text(json.dumps({"state": "running", "step": step, "seen": seen,
                                                         "best": best, "peak_gib": peak, "time": time.time()}))
    # order check: every list reshuffled, on the best checkpoint
    ck = torch.load(out / "verifier.pt", map_location=device, weights_only=False)
    model.ver.load_state_dict(ck["ver"]); model.head.load_state_dict(ck["head"])
    orders = {}
    for s, (_, f) in H.items():
        a = predict(model, held[s], tok.pad_token_id, device, budget2)
        b2 = predict(model, f(random.Random(2)), tok.pad_token_id, device, budget2)
        ks = [k for k in a if k in b2]
        va = np.array([a[k][0].argmax() for k in ks]); vb = np.array([b2[k][0].argmax() for k in ks])
        pa = np.array([1 - coarse(a[k][0][None])[0, 1] >= 0.6 for k in ks])
        pb = np.array([1 - coarse(b2[k][0][None])[0, 1] >= 0.6 for k in ks])
        multi = np.array([a[k][1] > 1 for k in ks])
        orders[s] = {"n": len(ks), "verdict_changed": float((va != vb).mean()), "keep_changed": float((pa != pb).mean()),
                     "verdict_changed_in_lists": float((va != vb)[multi].mean()) if multi.any() else 0.0}
    print(f"order check (best, step {ck['step']}): {json.dumps(orders)}", flush=True)
    ck["order"] = orders; ck["curve"] = curve
    torch.save(ck, out / "verifier.pt")
    (out / "status.json").write_text(json.dumps({"state": "done", "step": step, "seen": seen, "best": best,
                                                 "peak_gib": peak, "order": orders, "time": time.time()}))
    if args.upload:
        from huggingface_hub import HfApi
        HfApi().upload_folder(repo_id=args.repo, folder_path=str(out), path_in_repo=args.out_run,
                              allow_patterns=["verifier.pt", "status.json", "curve.json", "train.log"],
                              commit_message=f"Add {args.out_run} (list verifier)")
        print(f"uploaded to {args.repo}/{args.out_run}", flush=True)
    print(f"done: best held-out AUC {best:.4f}")
    return 0


# ── smoke (local) ──────────────────────────────────────────────────
def smoke(args) -> int:
    """PASS / FAIL: a one-candidate pass in the window form is V1c's input and,
    with the warm start, V1c's logits; a candidate's logits alone and in a
    batch of passes; the loss is finite; the gradient reaches only the
    adapter and the head."""
    from .verifier import Verifier, collate as vcollate
    device = torch.device(args.device or "cpu")
    base, tok, bank, recs = _setup(args, device)
    ck = _v1c_ck(args)
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name, flush=True)
        ok &= bool(cond)
    m3 = Verifier3(base, r=ck["rank"], n_out=len(VERDICTS_K)).to(device)
    m3.warm_start(ck)
    m1 = Verifier(base, r=ck["rank"], n_out=len(VERDICTS_K)).to(device)
    m1.ver.load_state_dict(ck["ver"]); m1.head.load_state_dict(ck["head"])
    m3.eval(); m1.eval()
    rs = [r for r in recs if r["split"] == "train" and not r.get("participants")][:30]
    diffs, n = [], 0
    with torch.no_grad():
        for r in rs:
            for j, c in enumerate(r["cands"][:3]):
                if c.get("swap"):
                    continue
                e = encode(r, c, tok, bank, None, None)
                if e is None:
                    continue
                ps = build_passes(r, [dict(c, key=(r["id"], j))], tok, bank, budget=448, cap=10 ** 6)
                ps_w = [dict(p) for p in ps]
                # the window form: rebuild with the chunk cap at 0
                ps_w = build_passes(r, [dict(c, key=(r["id"], j))], tok, bank, budget=448, cap=0)
                if not ps_w:
                    continue
                same = ps_w[0]["ids"] == e["ids"]
                a = m1(vcollate([e], tok.pad_token_id, device))
                b3 = m3(collate(ps_w, tok.pad_token_id, device))
                diffs.append((same, float((a - b3).abs().max())))
                n += 1
    check(f"one candidate, window form: input identical to V1c's on {sum(s for s, _ in diffs)}/{len(diffs)}",
          all(s for s, _ in diffs))
    check(f"one candidate, window form: logits = V1c's, max diff {max(d for _, d in diffs):.2e}",
          max(d for _, d in diffs) < 1e-3)
    # a candidate alone and in a batch; lists with several candidates
    r = next(r for r in rs if len(r["cands"]) >= 6)
    cands = [dict(c, key=(r["id"], j)) for j, c in enumerate(r["cands"]) if not c.get("swap")]
    ps = build_passes(r, cands, tok, bank, budget=448, cap=256)
    other = build_passes(rs[0], [dict(c, key=(rs[0]["id"], j)) for j, c in enumerate(rs[0]["cands"])
                                 if not c.get("swap")], tok, bank, budget=448, cap=256)
    with torch.no_grad():
        alone = m3(collate(ps[:1], tok.pad_token_id, device))
        batch = m3(collate(ps[:1] + other, tok.pad_token_id, device))
    k = len(ps[0]["slots"])
    check(f"pass alone vs in a batch ({k} slots): max diff {float((alone - batch[:k]).abs().max()):.2e}",
          torch.allclose(alone, batch[:k], atol=1e-3))
    check(f"lists of several candidates built: {[len(p['slots']) for p in ps]}", any(len(p["slots"]) > 1 for p in ps))
    m3.train()
    b = collate(ps + other, tok.pad_token_id, device)
    w = torch.ones(len(VERDICTS_K), device=device)
    b["y"][0] = SOFT
    b["soft"][0] = torch.full((len(VERDICTS_K),), 1 / len(VERDICTS_K))
    loss = loss_fn(m3(b), b["y"], b["soft"], w, 0.5)
    loss.backward()
    check(f"loss finite ({loss.item():.3f}) with a soft target", torch.isfinite(loss))
    g_ver = sum(float(p.grad.abs().sum()) for p in m3.ver.parameters() if p.grad is not None)
    g_head = sum(float(p.grad.abs().sum()) for p in m3.head.parameters() if p.grad is not None)
    g_base = sum(1 for p in base.parameters() if p.grad is not None)
    check(f"gradient on the adapter ({g_ver:.2e}) and the head ({g_head:.2e})", g_ver > 0 and g_head > 0)
    check(f"no gradient on frozen parameters ({g_base} tensors)", g_base == 0)
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["prep", "smoke", "train"])
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--out-run", default=f"{RUN}-ver3")
    ap.add_argument("--out-dir")
    ap.add_argument("--repo", default="dragonscale-ai/kniv-v6-runs")
    ap.add_argument("--local", type=Path)
    ap.add_argument("--device")
    ap.add_argument("--data", default="verifier_v1c.json")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-extra", type=int, default=60)
    ap.add_argument("--queue", type=int, default=4096, help="prep: candidates queued before V1c runs on them")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true", help="prep: merge the shards' parts (and upload)")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=2e-4)
    ap.add_argument("--budget", type=int, default=448, help="candidate subwords per pass")
    ap.add_argument("--cap", type=int, default=256, help="chunk subwords; longer chunks use the window")
    ap.add_argument("--batch", type=int, default=32, help="batch budget: batch x batch-tokens squared")
    ap.add_argument("--batch-tokens", type=int, default=384)
    ap.add_argument("--candidates", type=int, default=460_000, help="train: candidates to see")
    ap.add_argument("--eval-every", type=int, default=40_000, help="train: candidates between readings")
    ap.add_argument("--soft-weight", type=float, default=0.5)
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--dev-limit", type=int, default=4000)
    ap.add_argument("--neg-weight", type=float, default=2.0)
    ap.add_argument("--hedge-weight", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    if a.cmd == "prep":
        return prep(a)
    if a.cmd == "smoke":
        return smoke(a)
    return train(a)


if __name__ == "__main__":
    raise SystemExit(main())

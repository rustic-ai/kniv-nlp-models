"""Shared-pass fact verifier (plan r4 V2): the text is read once, every
candidate is scored from that pass.

    uv run python -m v6.train.verifier2 train --run RUN --out-run RUN-ver2 [--upload]
    uv run python -m v6.train.verifier2 smoke [--device cpu]

Model. The run's encoder and heads stay frozen. One encoder pass per window
over the marker text (every entity mention between ``*``, as the relation
pass), through a new LoRA adapter started as a copy of the run's relation
adapter; nothing about any candidate is in that pass, so its cost does not
depend on how many candidates there are. From it, per window: each entity's
vector (log-sum-exp of its start markers, as the relation head) and its
last-layer attention row. Per candidate (head, tail, relation), eight query
tokens: a learned CLS, the head and tail entity vectors, the pair's context
(the text weighted by the product of the two entities' attention, as the
relation head), and four slots projected from the relation's label vector
(the run's label pass over "relation: name. description", the vector the
release already computes once per schema). Two pre-norm blocks: the queries
attend to each other, then to the cached text states, with three learned
per-head biases on the attention scores: at the head's mention markers, at
the tail's, and on the log of the pair's own attention map. An MLP over the
CLS, head and tail outputs gives V1c's eight verdicts.

Per candidate that is a few (8 x width) products against the window's
cached keys and values: no encoder pass. Training reads each window (and
each broken-argument variant of it, whose markers differ) once per step
with all its candidates.

Data: ``verifier_v1c.json`` as V1c, the same splits (``held`` and
``held_long``) and labels; the held-out readings are taken on exactly
V1c's held-out examples (the records V1c's ``examples`` read, ``dev_limit``
candidates) and on every held-out candidate, so the two compare directly.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .verifier import (DATA, HEDGED_ANY, VERDICTS, VERDICTS_K, _corpus, _markers, coarse, encode,
                       k_loss, next_batch)

N_Q, SLOTS = 8, 4      # query tokens per candidate: CLS, head, tail, context, four label slots


# ── model ──────────────────────────────────────────────────────────
class Cross(nn.Module):
    """Queries of each candidate attend to one window's cached states; role
    biases at the head's and tail's mention markers and on the pair map."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.b_head = nn.Parameter(torch.zeros(heads))
        self.b_tail = nn.Parameter(torch.zeros(heads))
        self.g_pair = nn.Parameter(torch.zeros(heads))

    def forward(self, x, mem, key_ok, hm, tm, lpair):
        """``x [n,Q,d]``, ``mem [T,d]``, ``key_ok [T]`` bool, ``hm, tm, lpair [n,T]``."""
        n, Q, d = x.shape
        T = mem.size(0)
        q = self.q(x).view(n, Q, self.h, self.dh).transpose(1, 2)               # n,h,Q,dh
        k = self.k(mem).view(T, self.h, self.dh).permute(1, 2, 0)               # h,dh,T
        v = self.v(mem).view(T, self.h, self.dh).transpose(0, 1)                # h,T,dh
        s = torch.matmul(q, k.unsqueeze(0)) / math.sqrt(self.dh)                # n,h,Q,T
        bias = (self.b_head.view(1, -1, 1) * hm.unsqueeze(1) + self.b_tail.view(1, -1, 1) * tm.unsqueeze(1)
                + self.g_pair.view(1, -1, 1) * lpair.unsqueeze(1))              # n,h,T
        s = s + bias.unsqueeze(2).to(s.dtype)
        s = s.masked_fill(~key_ok.view(1, 1, 1, T), -1e4)
        a = torch.softmax(s.float(), -1).to(v.dtype)
        out = torch.matmul(a, v.unsqueeze(0))                                   # n,h,Q,dh
        return self.o(out.transpose(1, 2).reshape(n, Q, d))


class SelfAttn(nn.Module):
    """Attention among one candidate's query tokens (written out: the fused
    ``nn.MultiheadAttention`` path does not export to ONNX)."""

    def __init__(self, d: int, heads: int, p: float = 0.1):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv, self.o, self.drop = nn.Linear(d, 3 * d), nn.Linear(d, d), nn.Dropout(p)

    def forward(self, x):
        n, Q, d = x.shape
        q, k, v = self.qkv(x).view(n, Q, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)   # n,h,Q,dh each
        a = torch.softmax((torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dh)).float(), -1).to(v.dtype)
        return self.o(torch.matmul(self.drop(a), v).transpose(1, 2).reshape(n, Q, d))


class Block(nn.Module):
    def __init__(self, d: int, heads: int, p: float = 0.1):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = SelfAttn(d, heads, p)
        self.ca = Cross(d, heads)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(p), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(p)

    def forward(self, x, mem, key_ok, hm, tm, lpair):
        y = self.ln1(x)
        x = x + self.drop(self.sa(y))
        x = x + self.drop(self.ca(self.ln2(x), mem, key_ok, hm, tm, lpair))
        return x + self.drop(self.ff(self.ln3(x)))


class Reader(nn.Module):
    """The candidate reader over one window's pass (the part per candidate)."""

    def __init__(self, H: int, label_dim: int, d: int = 768, layers: int = 2, heads: int = 12,
                 n_out: int = len(VERDICTS_K), hidden: int = 512):
        super().__init__()
        self.d = d
        self.mem = nn.Sequential(nn.Linear(H, d), nn.LayerNorm(d))
        self.q_h, self.q_t, self.q_r = nn.Linear(H, d), nn.Linear(H, d), nn.Linear(H, d)
        self.q_l = nn.Linear(label_dim, SLOTS * d)
        self.cls = nn.Parameter(torch.randn(d) * 0.02)
        self.pos = nn.Parameter(torch.randn(N_Q, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.Linear(3 * d, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, n_out))

    def forward(self, seq, key_ok, hs, ts, rs, lab, hm, tm, lpair):
        """One window: ``seq [T,H]``; per candidate ``hs, ts, rs [n,H]``,
        ``lab [n,label_dim]``, ``hm, tm, lpair [n,T]``. Logits ``[n, n_out]``."""
        n = hs.size(0)
        dt = self.q_h.weight.dtype
        mem = self.mem(seq.to(dt))
        x = torch.cat([self.cls.view(1, 1, -1).expand(n, 1, self.d),
                       self.q_h(hs.to(dt)).unsqueeze(1), self.q_t(ts.to(dt)).unsqueeze(1),
                       self.q_r(rs.to(dt)).unsqueeze(1), self.q_l(lab.to(dt)).view(n, SLOTS, self.d)], 1)
        x = x + self.pos.unsqueeze(0)
        for b in self.blocks:
            x = b(x, mem, key_ok, hm, tm, lpair)
        x = self.ln(x)
        return self.out(torch.cat([x[:, 0], x[:, 1], x[:, 2]], -1)).float()


def _member(men_pos, men_ent, E, device):
    """``[E,M]`` 1 where mention ``m`` (a usable start marker) is entity ``e``'s."""
    ok = men_pos >= 0
    return ((men_ent.unsqueeze(0) == torch.arange(E, device=device).unsqueeze(1)) & ok.unsqueeze(0)).float()


def pool(seq, att, men_pos, men_ent, E):
    """One window's entities (ATLOP, as ``MarkerRelationHead.pair_features``):
    ``ent [E,H]`` log-sum-exp of the start markers' states, ``e_att
    [E,heads,T]`` the mean of their attention rows. Written with a membership
    matrix (no scatter), so it exports to ONNX."""
    seq = seq.float()
    pc = men_pos.clamp(min=0)
    member = _member(men_pos, men_ent, E, seq.device)                            # [E,M]
    m_emb = seq[pc]                                                              # [M,H]
    fill = torch.where(member.bool().unsqueeze(-1), m_emb.unsqueeze(0), torch.full_like(m_emb, -1e4).unsqueeze(0))
    ent = torch.logsumexp(fill, 1)                                               # [E,H]
    has = member.sum(1, keepdim=True) > 0
    ent = torch.where(has, ent, torch.zeros_like(ent))
    cnt = member.sum(1).clamp(min=1)
    e_att = torch.einsum("em,hmt->eht", member, att[:, pc].float()) / cnt.view(-1, 1, 1)
    return ent, e_att


def pair_inputs(seq, ent, e_att, men_pos, men_ent, h, t, T_ok, E):
    """Per candidate: ``hs, ts, rs [n,H]``, role masks ``hm, tm [n,T]`` (1 at
    the entity's start markers) and ``lpair [n,T]`` (log of the pair map)."""
    hs, ts = ent[h], ent[t]
    ht = (e_att[h] * e_att[t]).mean(1)                                           # [n,T]
    ht = ht / (ht.sum(1, keepdim=True) + 1e-5)
    rs = ht @ seq.float()
    T = seq.size(0)
    pc = men_pos.clamp(min=0)
    onehot = (pc.unsqueeze(1) == torch.arange(T, device=seq.device).unsqueeze(0)).float()   # [M,T]
    ent_pos = (_member(men_pos, men_ent, E, seq.device) @ onehot).clamp(max=1)   # [E,T]
    lpair = torch.log(ht * T_ok + 1e-4)
    return hs, ts, rs, ent_pos[h], ent_pos[t], lpair


class Verifier2(nn.Module):
    def __init__(self, base, r: int = 64, d: int = 768, layers: int = 2, heads: int = 12,
                 n_out: int = len(VERDICTS_K), label_dim: int | None = None):
        super().__init__()
        from .model import RelationAdapter
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.ver = RelationAdapter(base.encoder, r)
        if getattr(base, "rel_lora", None) is not None:      # start as the relation pass
            st = {f"A.{k}": v for k, v in base.rel_lora.A.items()}
            st.update({f"B.{k}": v for k, v in base.rel_lora.B.items()})
            self.ver.load_grown(st, "")
        H = base.encoder.config.hidden_size
        label_dim = label_dim or H
        self.cfg = {"rank": r, "d": d, "layers": layers, "heads": heads, "n_out": n_out, "label_dim": label_dim}
        self.reader = Reader(H, label_dim, d, layers, heads, n_out)

    def trainable(self):
        return list(self.ver.parameters()) + list(self.reader.parameters())

    def encode_pass(self, ids, mask):
        base = self.base
        if getattr(base, "rel_lora", None) is not None:
            base.rel_lora.on = False
        self.ver.on = True
        try:
            return base._encode_with_attn(ids, mask)
        finally:
            self.ver.on = False

    def forward(self, b: dict) -> torch.Tensor:
        """``b``: ``collate`` of passes. Logits ``[N, n_out]`` in candidate order."""
        seq, att = self.encode_pass(b["ids"], b["mask"])
        out = []
        for p, w in enumerate(b["passes"]):
            T = int(b["mask"][p].sum())
            mp, me = w["men_pos"], w["men_ent"]
            ent, e_att = pool(seq[p, :T], att[p, :, :T, :T], mp, me, w["E"])
            hs, ts, rs, hm, tm, lp = pair_inputs(seq[p, :T], ent, e_att, mp, me, w["h"], w["t"], T, w["E"])
            key_ok = torch.ones(T, dtype=torch.bool, device=seq.device)
            out.append(self.reader(seq[p, :T], key_ok, hs, ts, rs, b["lab"][w["lab"]], hm, tm, lp))
        return torch.cat(out, 0)


# ── data ───────────────────────────────────────────────────────────
def label_text_of(rel: dict, name: str | None = None, desc: str | None = None) -> str:
    from .kdata import label_text
    return label_text({"name": name or rel["name"], "description": rel.get("description", "") if desc is None
                       else desc})


def passes_of(rec: dict, tok, bank: dict, para: dict | None, rng: random.Random | None,
              keep=None, max_subwords: int = 1024) -> list[dict]:
    """One record's passes: the plain window and each broken-argument variant
    (its markers differ), each with its candidates (``keep``: a set of
    candidate indices to keep, for V1c's exact held-out examples)."""
    groups: dict = {}
    for j, c in enumerate(rec["cands"]):
        if keep is not None and j not in keep:
            continue
        sw = c.get("swap")
        key = None if sw is None else (sw[0], tuple(sw[1]))
        groups.setdefault(key, []).append((j, c))
    out = []
    for key, cs in groups.items():
        rm = _markers(rec, tok, None if key is None else [key[0], list(key[1])])
        if len(rm["ids"]) > max_subwords:
            continue
        men = rm["men_pos"]
        cands = []
        for j, c in cs:
            if any(len(men[k]) == 0 or men[k].max() < 0 for k in (c["h"], c["t"])):
                continue
            rel = bank.get(c["rel"], {"name": c["rel"], "description": ""})
            name, desc = rel["name"], rel.get("description", "")
            if para and rng is not None and rng.random() < 0.2 and rel.get("id") in para:
                pn, pd = rng.choice(para[rel["id"]])
                name, desc = pn, pd or desc
            cands.append({"j": j, "h": c["h"], "t": c["t"], "y": c["y"], "src": c.get("src", ""),
                          "label": label_text_of(rel, name, desc)})
        if not cands:
            continue
        men_ent, men_pos = [], []
        for k, pos in enumerate(men):
            for q in (pos.tolist() if len(pos) else [-1]):
                men_ent.append(k); men_pos.append(q)
        out.append({"ids": [int(x) for x in rm["ids"]], "men_pos": men_pos, "men_ent": men_ent,
                    "E": len(men), "cands": cands, "id": rec.get("id")})
    return out


def v1c_held_keys(recs, tok, bank, split: str, limit: int | None) -> dict:
    """The (record index -> candidate indices) V1c's ``examples`` read for
    ``split`` with ``limit`` (the same records, the same candidates)."""
    keys, n = {}, 0
    for i, r in enumerate(recs):
        if r["split"] != split:
            continue
        cache: dict = {}
        ks = {j for j, c in enumerate(r["cands"]) if encode(r, c, tok, bank, None, None, cache=cache) is not None}
        if ks:
            keys[i] = ks
            n += len(ks)
        if limit and n >= limit:
            break
    return keys


class LabelCache:
    """Label vectors from the frozen label pass, one per label text."""

    def __init__(self, base, tok, device):
        self.base, self.tok, self.device, self.c = base, tok, device, {}

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> torch.Tensor:
        miss = sorted({t for t in texts if t not in self.c})
        for i in range(0, len(miss), 64):
            chunk = miss[i:i + 64]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=48, return_tensors="pt")
            v = self.base.label_embeddings(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
            for t, x in zip(chunk, v.float()):
                self.c[t] = x
        return torch.stack([self.c[t] for t in texts])


def collate(ps: list[dict], pad: int, device, labels: LabelCache) -> dict:
    T = max(len(p["ids"]) for p in ps)
    ids = torch.full((len(ps), T), pad, dtype=torch.long)
    mask = torch.zeros(len(ps), T, dtype=torch.long)
    texts, passes, ys = [], [], []
    for b, p in enumerate(ps):
        n = len(p["ids"])
        ids[b, :n] = torch.tensor(p["ids"]); mask[b, :n] = 1
        k0 = len(texts)
        texts += [c["label"] for c in p["cands"]]
        passes.append({"men_pos": torch.tensor(p["men_pos"], device=device),
                       "men_ent": torch.tensor(p["men_ent"], device=device), "E": p["E"],
                       "h": torch.tensor([c["h"] for c in p["cands"]], device=device),
                       "t": torch.tensor([c["t"] for c in p["cands"]], device=device),
                       "lab": torch.arange(k0, len(texts), device=device)})
        ys += [c["y"] for c in p["cands"]]
    return {"ids": ids.to(device), "mask": mask.to(device), "passes": passes, "lab": labels(texts),
            "y": torch.tensor(ys, device=device)}


# ── evaluation and training ────────────────────────────────────────
@torch.no_grad()
def evaluate(model, ps, pad, device, labels, bs: int = 32, max_tokens: int = 384) -> dict:
    """V1c's ``evaluate`` readings over passes (every candidate of a pass)."""
    model.eval()
    P, Y, S = [], [], []
    i = 0
    while i < len(ps):
        k = 1
        while (i + k < len(ps) and k < bs
               and (k + 1) * max(len(p["ids"]) for p in ps[i:i + k + 1]) ** 2 <= bs * max_tokens ** 2):
            k += 1
        chunk = ps[i:i + k]
        i += k
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(collate(chunk, pad, device, labels))
        P.append(torch.softmax(lg, -1).cpu().numpy())
        Y += [c["y"] for p in chunk for c in p["cands"]]
        S += [c["src"] for p in chunk for c in p["cands"]]
    model.train()
    P, Y = np.concatenate(P), np.array(Y)
    res_k = {}
    if P.shape[1] == len(VERDICTS_K):
        kk = Y >= 3
        if kk.any():
            res_k["acc_kind"] = float((P[kk, 3:].argmax(1) + 3 == Y[kk]).mean())
            res_k["n_kind"] = int(kk.sum())
        P = coarse(P)
        Y = np.where((Y >= 3) | (Y == HEDGED_ANY), 3, Y)
    expressed = 1 - P[:, 1]
    yb = Y != 1
    o = expressed.argsort().argsort() + 1
    n1, n0 = yb.sum(), (~yb).sum()
    auc = (o[yb].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)
    pred = P.argmax(1)
    res = {"n": len(Y), "acc": float((pred == Y).mean()), "auc_expressed": float(auc), **res_k}
    for k, v in enumerate(VERDICTS):
        sel = Y == k
        if sel.any():
            res[f"recall_{v}"] = float((pred[sel] == k).mean())
    return res


def _setup(args, device):
    from .gold_eval import load_run
    from ..label_bank import load as load_bank
    base, tok, spec, vocab, _ = load_run(args.run, args.repo, device, args.local)
    base.eval()
    bank = {r["name"]: r for r in load_bank()["relations"]}
    para_raw = json.loads((_corpus() / "label_para.json").read_text()) if (_corpus() / "label_para.json").exists() else {}
    para = {int(k): v for k, v in para_raw.items()}
    D = json.loads((_corpus() / args.data).read_text())
    return base, tok, bank, para, D["records"], D.get("verdicts", VERDICTS)


def train(args) -> int:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    base, tok, bank, para, recs, verdicts = _setup(args, device)
    rng = random.Random(args.seed)
    tr = []
    for r in recs:
        if r["split"] == "train":
            tr += passes_of(r, tok, bank, para, rng)
            if args.limit and len(tr) >= args.limit:
                break
    # held out: V1c's exact examples, and every held-out candidate
    kh = v1c_held_keys(recs, tok, bank, "held", args.dev_limit)
    held = [p for i, ks in kh.items() for p in passes_of(recs[i], tok, bank, None, None, keep=ks)]
    kl = v1c_held_keys(recs, tok, bank, "held_long", args.dev_limit)
    held_long = [p for i, ks in kl.items() for p in passes_of(recs[i], tok, bank, None, None, keep=ks)]
    held_all = [p for r in recs if r["split"] == "held" for p in passes_of(r, tok, bank, None, None)] \
        if not args.limit else []
    nc = lambda ps: sum(len(p["cands"]) for p in ps)
    print(f"{len(tr):,} training passes ({nc(tr):,} candidates); held out as V1c {len(held):,} passes "
          f"({nc(held):,}), long {len(held_long):,} ({nc(held_long):,}), all held {nc(held_all):,}; "
          f"{Counter(c['y'] for p in tr for c in p['cands'])}", flush=True)
    labels = LabelCache(base, tok, device)
    model = Verifier2(base, r=args.rank, d=args.d, layers=args.layers, heads=args.heads,
                      n_out=len(verdicts), label_dim=labels(["relation: x"]).shape[-1]).to(device)
    opt = torch.optim.AdamW([{"params": list(model.ver.parameters()), "lr": args.lr},
                             {"params": list(model.reader.parameters()), "lr": args.head_lr}], weight_decay=0.01)
    steps = int(args.steps or (len(tr) * args.epochs) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.head_lr], total_steps=steps,
                                                pct_start=0.05, anneal_strategy="linear")
    w = torch.tensor([1.0, 1.0, args.neg_weight] + [args.hedge_weight] * (len(verdicts) - 3), device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w) if len(verdicts) == len(VERDICTS) else (lambda lg, y: k_loss(lg, y, w))
    out = Path(args.out_dir or f"data/v6-eval/checkpoints/{args.out_run}")
    out.mkdir(parents=True, exist_ok=True)
    best, t0, order, longq = -1.0, time.time(), [], []
    peak = 0.0
    model.train()

    def ev(step):
        res = evaluate(model, held, tok.pad_token_id, device, labels)
        if held_long:
            res.update({f"long_{k}": v for k, v in evaluate(model, held_long, tok.pad_token_id, device, labels).items()
                        if k in ("n", "auc_expressed", "acc", "acc_kind")})
        return res

    if args.probe:          # the worst batches first: passes in order of length, the longest popped first
        order = sorted(range(len(tr)), key=lambda j: (len(tr[j]["ids"]), len(tr[j]["cands"])))
    for step in range(1, steps + 1):
        if not order:
            order = list(range(len(tr))); rng.shuffle(order)
        idx = next_batch(tr, order, longq, args.batch, args.max_tokens)
        if not idx:
            order = list(range(len(tr))); rng.shuffle(order)
            idx = next_batch(tr, order, longq, args.batch, args.max_tokens)
        b = collate([tr[i] for i in idx], tok.pad_token_id, device, labels)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(b)
        loss = loss_fn(lg, b["y"])
        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss, skipped", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable(), 1.0)
        opt.step(); sched.step()
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated() / 2 ** 30)
        if step % 50 == 0 or step <= args.probe:
            print(f"step {step}/{steps} loss {loss.item():.4f} passes {len(idx)} T {b['ids'].size(1)} "
                  f"cands {len(b['y'])} peak {peak:.1f} GiB [{time.time() - t0:.0f}s]", flush=True)
        if args.probe and step >= args.probe:
            print(f"probe done: {step} steps, peak {peak:.1f} GiB, {(time.time() - t0) / step:.2f} s per step")
            return 0
        if step % args.eval_every == 0 or step == steps:
            res = ev(step)
            print(f"eval {step}: {json.dumps({k: round(v, 4) for k, v in res.items()})}", flush=True)
            if res["auc_expressed"] > best:
                best = res["auc_expressed"]
                torch.save({"form": "v2", "ver": model.ver.state_dict(), "reader": model.reader.state_dict(),
                            "cfg": model.cfg, "base_run": args.run, "rank": args.rank, "step": step, "eval": res,
                            "data": args.data, "verdicts": verdicts}, out / "verifier.pt")
                (out / "best.metric").write_text(json.dumps({"value": best, "step": step,
                                                             "metric": "held-out AUC, expressed (V1c's examples)"}))
            (out / "status.json").write_text(json.dumps({"state": "running", "step": step, "of": steps,
                                                         "best": best, "peak_gib": peak, "time": time.time()}))
    if held_all:            # the best checkpoint on every held-out candidate
        ck = torch.load(out / "verifier.pt", map_location=device, weights_only=False)
        model.ver.load_state_dict(ck["ver"]); model.reader.load_state_dict(ck["reader"])
        res = evaluate(model, held_all, tok.pad_token_id, device, labels)
        print(f"best (step {ck['step']}) on every held-out candidate: {json.dumps({k: round(v, 4) for k, v in res.items()})}",
              flush=True)
        ck["eval_all_held"] = res
        torch.save(ck, out / "verifier.pt")
    (out / "status.json").write_text(json.dumps({"state": "done", "step": steps, "best": best,
                                                 "peak_gib": peak, "time": time.time()}))
    if args.upload:
        from huggingface_hub import HfApi
        HfApi().upload_folder(repo_id=args.repo, folder_path=str(out), path_in_repo=args.out_run,
                              allow_patterns=["verifier.pt", "best.metric", "status.json", "train.log"],
                              commit_message=f"Add {args.out_run} (shared-pass fact verifier)")
        print(f"uploaded to {args.repo}/{args.out_run}", flush=True)
    print(f"done: best held-out AUC {best:.4f}")
    return 0


def load_v2(path: Path, base, device) -> Verifier2:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    m = Verifier2(base, r=c["rank"], d=c["d"], layers=c["layers"], heads=c["heads"], n_out=c["n_out"],
                  label_dim=c["label_dim"]).to(device)
    m.ver.load_state_dict(ck["ver"]); m.reader.load_state_dict(ck["reader"])
    return m.eval()


@torch.no_grad()
def score_window(model, rec: dict, cands: list, tok, bank: dict, labels: LabelCache, device):
    """The extractor's call: one window (``rec``: tokens, sentence spans,
    speakers, participants, entity spans) and its candidates ``(head, tail,
    relation)``. One pass, every candidate from it. Returns verdict
    probabilities per candidate (None: not readable, the margin decides)."""
    r = dict(rec, cands=[{"h": h, "t": t, "rel": rel, "y": 0} for h, t, rel in cands])
    ps = passes_of(r, tok, bank, None, None)
    out: list = [None] * len(cands)
    if not ps:
        return out
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        lg = model(collate(ps, tok.pad_token_id, device, labels))
    pr = torch.softmax(lg.float(), -1).cpu().tolist()
    for c, p in zip(ps[0]["cands"], pr):
        out[c["j"]] = p
    return out


# ── smoke (local, tiny) ────────────────────────────────────────────
def smoke(args) -> int:
    """PASS / FAIL checks on a few real records with the run's model on
    ``--device``: shapes; a pass's logits do not depend on the other passes
    in its batch, nor on the other candidates of its own pass; the loss is
    finite and its gradient reaches the adapter and the reader and nothing
    frozen; train / eval readings run."""
    device = torch.device(args.device or "cpu")
    base, tok, bank, para, recs, verdicts = _setup(args, device)
    rng = random.Random(0)
    ps = [p for r in recs[:40] if r["split"] == "train" for p in passes_of(r, tok, bank, para, rng)][:6]
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name, flush=True)
        ok &= bool(cond)
    labels = LabelCache(base, tok, device)
    model = Verifier2(base, r=64, d=128, layers=2, heads=4, n_out=len(verdicts),
                      label_dim=labels(["relation: x"]).shape[-1]).to(device)
    model.eval()
    with torch.no_grad():
        b = collate(ps, tok.pad_token_id, device, labels)
        lg = model(b)
        check(f"logits shape {tuple(lg.shape)} = ({len(b['y'])}, {len(verdicts)})",
              lg.shape == (len(b["y"]), len(verdicts)))
        one = model(collate(ps[:1], tok.pad_token_id, device, labels))
        n0 = len(ps[0]["cands"])
        check(f"pass alone vs in a batch: max diff {float((one - lg[:n0]).abs().max()):.2e}",
              torch.allclose(one, lg[:n0], atol=1e-3))
        p1 = dict(ps[0], cands=ps[0]["cands"][:1])
        solo = model(collate([p1], tok.pad_token_id, device, labels))
        check(f"candidate alone vs with its pass's others: max diff {float((solo - one[:1]).abs().max()):.2e}",
              torch.allclose(solo, one[:1], atol=1e-3))
        # the pooling is the relation head's: same head, tail and context vectors on the same pass
        p0 = ps[0]
        ids = torch.tensor([p0["ids"]], device=device)
        seq, att = model.encode_pass(ids, torch.ones_like(ids))
        mp, me = torch.tensor(p0["men_pos"], device=device), torch.tensor(p0["men_ent"], device=device)
        h = torch.tensor([c["h"] for c in p0["cands"]], device=device)
        t = torch.tensor([c["t"] for c in p0["cands"]], device=device)
        ent, e_att = pool(seq[0], att[0], mp, me, p0["E"])
        hs, ts, rs, *_ = pair_inputs(seq[0], ent, e_att, mp, me, h, t, ids.size(1), p0["E"])
        rb = {"ent_batch": torch.zeros(p0["E"], dtype=torch.long, device=device), "men_ent": me, "rm_men_pos": mp,
              "pair_h": h, "pair_t": t, "pair_batch": torch.zeros(len(h), dtype=torch.long, device=device)}
        base.relation.pair_features(seq, att, rb)
        rh, rt, rr = base.relation._pair_in
        diff = max(float((a - b_).abs().max()) for a, b_ in ((hs, rh), (ts, rt), (rs, rr)))
        check(f"pooling = the relation head's pair_features: max diff {diff:.2e}", diff < 1e-3)
    model.train()
    b = collate(ps, tok.pad_token_id, device, labels)
    w = torch.ones(len(verdicts), device=device)
    loss = k_loss(model(b), b["y"], w)
    loss.backward()
    check(f"loss finite ({loss.item():.3f})", torch.isfinite(loss))
    g_ver = sum(float(p.grad.abs().sum()) for p in model.ver.parameters() if p.grad is not None)
    g_rd = sum(float(p.grad.abs().sum()) for p in model.reader.parameters() if p.grad is not None)
    g_base = sum(1 for p in base.parameters() if p.grad is not None)
    check(f"gradient reaches the adapter ({g_ver:.2e}) and the reader ({g_rd:.2e})", g_ver > 0 and g_rd > 0)
    check(f"no gradient on frozen parameters ({g_base} tensors)", g_base == 0)
    res = evaluate(model, ps, tok.pad_token_id, device, labels)
    check(f"evaluate runs: {res['n']} candidates, AUC {res['auc_expressed']:.3f}", res["n"] == len(b["y"]))
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["train", "smoke"])
    ap.add_argument("--run", default="ultrabert-380m-v8p3-d123")
    ap.add_argument("--out-run", default="ultrabert-380m-v8p3-d123-ver2")
    ap.add_argument("--out-dir")
    ap.add_argument("--repo", default="dragonscale-ai/kniv-v6-runs")
    ap.add_argument("--local", type=Path)
    ap.add_argument("--device")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=32, help="passes per step")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--probe", type=int, default=0, help="train: stop after N steps and print peak memory")
    ap.add_argument("--limit", type=int, help="train: first N training passes (pipeline test)")
    ap.add_argument("--dev-limit", type=int, default=4000)
    ap.add_argument("--neg-weight", type=float, default=2.0)
    ap.add_argument("--hedge-weight", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--data", default="verifier_v1c.json")
    a = ap.parse_args()
    return train(a) if a.cmd == "train" else smoke(a)


if __name__ == "__main__":
    raise SystemExit(main())

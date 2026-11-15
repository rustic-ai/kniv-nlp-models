"""What the list verifier (plan r4 V3) costs per text on CPU, before it is
trained: its graph over each tier's encoder, ONNX INT8, on the real spread
of candidates over sentences.

    uv run python -m v6.export.verifier3_onnx spread                 # candidates' chunks, M1 short and stage-4 long
    uv run --extra export python -m v6.export.verifier3_onnx bench --tier ultrabert-380m
    uv run python -m v6.export.verifier3_onnx simulate [--cap 128] [--budgets 112 224 448]

V3 reads one pass per chunk and list: the chunk (the sentence, or shortest
run of sentences, holding a mention of both arguments; every mention in it
marked as the marker pass marks them) followed by up to a budget of
candidates, each ``[slot] head | relation | tail``, then a legend of the
relations present (``name: description``). A chunk longer than ``cap``
subwords is replaced by the whole window (V1c's window form), whose
candidates are listed in the same way.

``spread``: every candidate the deployed verifier reads, with its chunk.
M1 short texts: the stage-3 cache (V1c, floor 3; the entity step's own
mention spans). Stage 4's long windows: every logged candidate, its
mentions found by their words in the window (a name found nowhere drops
the candidate). Subwords are counted with the run's tokenizer; markers add
two per mention. Writes ``v6/runs/verifier3-onnx/spread.pkl``.

``bench``: the V3 graph untrained (encoder with its last-layer attention,
the relation head's pooling for every slot's pair, V1c's head shape over
head, tail, context and the slot's segment) at a grid of pass lengths,
batch 1, median of three warm runs; and the release's end-to-end time at
64 and 500 words, in the same run. Writes ``cost3-<tier>.json``.

``simulate``: per text, its chunks, their candidates packed into passes
under each budget (candidate subwords per pass), each pass timed by
interpolating the bench, against the release at the same length. Reported
on M1's texts (mean 64 words, 22 candidates) and on stage 4's long windows
rescaled to 570 candidates per 500 words (each window's candidates
resampled with replacement to 570, the chunk spread kept).
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OUT = Path("v6/runs/verifier3-onnx")
TIERS = {"ettin-17m": "jhu-clsp/ettin-encoder-17m", "ettin-68m": "jhu-clsp/ettin-encoder-68m",
         "modernbert-base": "answerdotai/ModernBERT-base", "ultrabert-380m": "dmis-lab/UltraBERT-380M"}
GRID = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]


# ── spread ─────────────────────────────────────────────────────────
def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("dmis-lab/UltraBERT-380M", trust_remote_code=True)


def _sent_of(sp):
    """token -> sentence index."""
    out = {}
    for k, (a, b) in enumerate(sp):
        for i in range(a, b):
            out[i] = k
    return out


def _chunk(sp, so, hm, tm):
    """Shortest sentence run holding a mention of each argument: (first, last)."""
    best = None
    for a in hm:
        for b in tm:
            s = sorted([so.get(a[0], 0), so.get(b[0], 0)])
            if best is None or s[1] - s[0] < best[1] - best[0]:
                best = (s[0], s[1])
    return best


def _find(tokens, name):
    low = [w.lower() for w in tokens]
    mt = name.lower().split()
    return [(a, a + len(mt)) for a in range(len(low) - len(mt) + 1) if low[a:a + len(mt)] == mt]


def spread() -> int:
    tok = _tok()
    nsub = lambda words: len(tok(" ".join(words), add_special_tokens=False)["input_ids"])
    from ..label_bank import load as load_bank
    bank = {r["name"]: r for r in load_bank()["relations"]}
    leg = {}

    def legend(rel):
        if rel not in leg:
            d = bank.get(rel, {}).get("description", "")
            leg[rel] = len(tok(f"{rel}: {d} ;", add_special_tokens=False)["input_ids"])
        return leg[rel]
    texts = {"short": [], "long": []}
    # M1 short: the stage-3 cache (entity step's spans)
    C = pickle.loads(Path("v6/runs/m1-precision/s3cost-cache-flat-v1c.pkl").read_bytes())
    items = {it["id"]: it for it in pickle.loads(Path("v6/runs/m1-precision/preds-m1v1c-n500.pkl").read_bytes())["items"]}
    by = defaultdict(list)
    for (tid, h, t, rel) in C["ver"]:
        by[tid].append((h, t, rel))
    for tid, cands in by.items():
        it, ents = items[tid], C["ents"][tid]
        texts["short"].append(_text(it["tokens"], it["sentence_spans"],
                                    [[tuple(m) for m in e["mentions"]] for e in ents], cands, nsub, legend))
    # stage 4 long windows: mentions by their words
    S4 = Path("v6/runs/s4-pilot")
    its = {it["id"]: it for it in json.loads((S4 / "items.json").read_text())}
    for p in sorted(S4.glob("pred-*.pkl")):
        for r in pickle.loads(p.read_bytes()):
            it = its[r["id"]]
            names, ments, cands = {}, [], []
            for c in r["cands"]:
                hs = [s for n in c["head_m"] for s in _find(it["tokens"], n)]
                ts = [s for n in c["tail_m"] for s in _find(it["tokens"], n)]
                if not hs or not ts:
                    continue
                for k, ms in (("h", hs), ("t", ts)):
                    key = tuple(sorted(ms))
                    if key not in names:
                        names[key] = len(ments); ments.append(sorted(set(ms)))
                cands.append((names[tuple(sorted(hs))], names[tuple(sorted(ts))], c["relation"]))
            if cands:
                texts["long"].append(_text(it["tokens"], it["sentence_spans"], ments, cands, nsub, legend))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spread.pkl").write_bytes(pickle.dumps(texts))
    for k, ts in texts.items():
        n = [len(t["cands"]) for t in ts]
        one = np.mean([c["chunk"][0] == c["chunk"][1] for t in ts for c in t["cands"]])
        print(f"{k}: {len(ts)} texts, {np.mean([t['words'] for t in ts]):.0f} words, {np.mean(n):.1f} candidates; "
              f"{one:.2f} of candidates have both arguments in one sentence; chunk subwords median "
              f"{np.median([c['chunk_sub'] for t in ts for c in t['cands']]):.0f}, 90th "
              f"{np.percentile([c['chunk_sub'] for t in ts for c in t['cands']], 90):.0f}")
    return 0


def _text(tokens, sp, ments, cands, nsub, legend):
    """One text: its subwords, and per candidate its chunk, the chunk's
    subwords with markers, the candidate's own subwords and its legend's."""
    so = _sent_of(sp)
    sent_sub = [nsub(tokens[a:b]) for a, b in sp]
    marks = defaultdict(int)                         # markers per sentence: two per mention
    for ms in ments:
        for m in ms:
            marks[so.get(m[0], 0)] += 2
    out = []
    for h, t, rel in cands:
        if h >= len(ments) or t >= len(ments) or not ments[h] or not ments[t]:
            continue
        a, b = _chunk(sp, so, ments[h], ments[t])
        sub = sum(sent_sub[a:b + 1]) + sum(marks[k] for k in range(a, b + 1))
        hn = " ".join(tokens[ments[h][0][0]:ments[h][0][1]])
        tn = " ".join(tokens[ments[t][0][0]:ments[t][0][1]])
        out.append({"chunk": (a, b), "chunk_sub": sub, "rel": rel,
                    "cand_sub": 1 + nsub(f"{hn} | {rel} | {tn}".split()) + 1, "leg": legend(rel)})
    return {"words": len(tokens), "sub": sum(sent_sub) + sum(marks.values()), "cands": out}


# ── bench ──────────────────────────────────────────────────────────
class V3Graph(nn.Module):
    """One pass: chunk + slots. Out: verdict logits per slot."""

    def __init__(self, enc, H):
        super().__init__()
        self.enc = enc
        self.head = nn.Sequential(nn.LayerNorm(4 * H), nn.Linear(4 * H, 512), nn.GELU(), nn.Linear(512, 8))

    def forward(self, ids, mask, men_pos, member, pair_h, pair_t, seg):
        from .graphs import _mask
        out = self.enc(input_ids=ids, attention_mask=_mask(self.enc, mask), output_attentions=True)
        seq, att = out.last_hidden_state[0].float(), out.attentions[-1][0].float()     # [T,H], [h,T,T]
        m_emb = seq[men_pos]                                                          # [M,H]
        fill = torch.where(member.bool().unsqueeze(-1), m_emb.unsqueeze(0), torch.full_like(m_emb, -1e4).unsqueeze(0))
        ent = torch.logsumexp(fill, 1)
        e_att = torch.einsum("em,hmt->eht", member, att[:, men_pos]) / member.sum(1).clamp(min=1).view(-1, 1, 1)
        ht = (e_att[pair_h] * e_att[pair_t]).mean(1)
        ht = ht / (ht.sum(1, keepdim=True) + 1e-5)
        rs = ht @ seq
        cand = (seg @ seq) / seg.sum(1, keepdim=True).clamp(min=1)                  # [P,H] slot segments
        return self.head(torch.cat([ent[pair_h], ent[pair_t], rs, cand], -1))


def _release_times(tier):
    from .verifier2_onnx import _release_times as rt
    return rt(tier)


def bench(tier: str) -> int:
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModel
    torch.set_grad_enabled(False)
    rel = _release_times(tier)
    print(f"{tier}: release INT8 end to end {rel['short']:.3f} s at 64 words, {rel['long']:.3f} s at 500", flush=True)
    enc = AutoModel.from_pretrained(TIERS[tier], attn_implementation="eager", trust_remote_code=True).eval()
    H = enc.config.hidden_size
    g = V3Graph(enc, H).eval()
    rng = np.random.default_rng(0)

    def inputs(T, P=16, E=12):
        M = 2 * E
        pos = np.sort(rng.choice(np.arange(1, max(T - 1, M + 2)), size=M, replace=False)).astype(np.int64) % T
        member = np.zeros((E, M), np.float32)
        member[np.repeat(np.arange(E), 2), np.arange(M)] = 1
        seg = np.zeros((P, T), np.float32)
        for p in range(P):
            a = int(rng.integers(0, max(T - 12, 1))); seg[p, a:a + 12] = 1
        return {"ids": rng.integers(1000, 20000, size=(1, T)).astype(np.int64), "mask": np.ones((1, T), np.int64),
                "men_pos": pos, "member": member, "pair_h": rng.integers(0, E, P).astype(np.int64),
                "pair_t": rng.integers(0, E, P).astype(np.int64), "seg": seg}
    names = ["ids", "mask", "men_pos", "member", "pair_h", "pair_t", "seg"]
    d = OUT / tier
    d.mkdir(parents=True, exist_ok=True)
    x = inputs(128)
    p = d / "v3.onnx"
    axes = {"ids": {1: "T"}, "mask": {1: "T"}, "men_pos": {0: "M"}, "member": {0: "E", 1: "M"},
            "pair_h": {0: "P"}, "pair_t": {0: "P"}, "seg": {0: "P", 1: "T"}, "logits": {0: "P"}}
    torch.onnx.export(g, tuple(torch.from_numpy(x[k]) for k in names), str(p), input_names=names,
                      output_names=["logits"], dynamic_axes=axes, opset_version=17, do_constant_folding=True,
                      dynamo=False)
    q = d / "v3-int8.onnx"
    quantize_dynamic(str(p), str(q), weight_type=QuantType.QInt8)
    s = ort.InferenceSession(str(q), providers=["CPUExecutionProvider"])
    times = {}
    for T in GRID:
        x = inputs(T)
        s.run(None, x)
        r = []
        for _ in range(3):
            t0 = time.perf_counter(); s.run(None, x); r.append(time.perf_counter() - t0)
        times[T] = float(np.median(r))
        print(f"  pass of {T} subwords: {times[T] * 1000:.1f} ms", flush=True)
    for f in d.glob("v3*"):
        f.unlink()
    (OUT / f"cost3-{tier}.json").write_text(json.dumps({"tier": tier, "release_s": rel, "pass_s": times}, indent=1))
    return 0


# ── simulate ───────────────────────────────────────────────────────
def _passes(text, cap, budget, win_cap=1024):
    """Pass lengths (subwords) for one text: candidates grouped by chunk
    (the window for chunks over ``cap``), each group packed into passes of at
    most ``budget`` candidate subwords, each pass with its relations' legend."""
    groups = defaultdict(list)
    for c in text["cands"]:
        groups[c["chunk"] if c["chunk_sub"] <= cap else "window"].append(c)
    out = []
    for key, cs in groups.items():
        base = min(text["sub"], win_cap) if key == "window" else cs[0]["chunk_sub"]
        cur, rels = 0, set()
        for c in cs:
            if cur and cur + c["cand_sub"] > budget:
                out.append(base + 2 + cur + sum(r[1] for r in rels))
                cur, rels = 0, set()
            cur += c["cand_sub"]; rels.add((c["rel"], c["leg"]))
        if cur:
            out.append(base + 2 + cur + sum(r[1] for r in rels))
    return out


def simulate(cap: int, budgets: list[int], seed: int = 7) -> int:
    S = pickle.loads((OUT / "spread.pkl").read_bytes())
    rng = np.random.default_rng(seed)
    long_570 = []
    for t in S["long"]:
        if t["words"] < 300 or not t["cands"]:
            continue
        idx = rng.integers(0, len(t["cands"]), 570)
        long_570.append({**t, "cands": [t["cands"][i] for i in idx]})
    print(f"M1 short: {len(S['short'])} texts, {np.mean([len(t['cands']) for t in S['short']]):.1f} candidates; "
          f"stage 4 long: {len(long_570)} windows of 300+ words, rescaled to 570 candidates "
          f"({np.mean([t['words'] for t in long_570]):.0f} words)")
    for b in budgets:
        ps = [_passes(t, cap, b) for t in S["short"]]
        pl = [_passes(t, cap, b) for t in long_570]
        print(f"budget {b} subwords: short {np.mean([len(p) for p in ps]):.1f} passes per text "
              f"(mean {np.mean([x for p in ps for x in p]):.0f} subwords); long {np.mean([len(p) for p in pl]):.1f} "
              f"passes (mean {np.mean([x for p in pl for x in p]):.0f}), window-form passes "
              f"{np.mean([sum(x > cap + b + 200 for x in p) for p in pl]):.1f}")
    print("\n| tier | release s, 64 / 500 words | " + " | ".join(f"budget {b}: short / long" for b in budgets) + " |")
    print("|---|---|" + "---|" * len(budgets))
    for tier in TIERS:
        p = OUT / f"cost3-{tier}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        g = np.array(sorted(int(k) for k in r["pass_s"]))
        tv = np.array([r["pass_s"][str(k)] for k in g])
        tt = lambda L: float(np.interp(L, g, tv)) if L <= g[-1] else float(tv[-1] * (L / g[-1]) ** 1.5)
        rs, rl = r["release_s"]["short"], r["release_s"]["long"]
        cells = []
        for b in budgets:
            # short: each text against the release at 64 words scaled by its length
            s = np.mean([1 + sum(tt(x) for x in _passes(t, cap, b)) / (rs * max(t["words"], 1) / 64)
                         for t in S["short"]])
            l_ = np.mean([1 + sum(tt(x) for x in _passes(t, cap, b)) / (rl * t["words"] / 500) for t in long_570])
            cells.append(f"{s:.2f}x / {l_:.2f}x")
        print(f"| {tier} | {rs:.3f} / {rl:.3f} | " + " | ".join(cells) + " |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["spread", "bench", "simulate", "all"])
    ap.add_argument("--tier", choices=list(TIERS))
    ap.add_argument("--cap", type=int, default=128)
    ap.add_argument("--budgets", type=int, nargs="+", default=[112, 224, 448])
    a = ap.parse_args()
    if a.cmd == "spread":
        return spread()
    if a.cmd == "bench":
        return bench(a.tier)
    if a.cmd == "all":
        for t in TIERS:
            subprocess.run([sys.executable, "-m", "v6.export.verifier3_onnx", "bench", "--tier", t], check=True)
    return simulate(a.cap, a.budgets)


if __name__ == "__main__":
    raise SystemExit(main())

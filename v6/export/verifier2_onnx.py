"""What the shared-pass verifier (plan r4 V2) costs per text on CPU, before
it is trained: its graph over each tier's encoder, ONNX INT8.

    uv run --extra export python -m v6.export.verifier2_onnx cost --tier ultrabert-380m
    uv run --extra export python -m v6.export.verifier2_onnx table

``cost``: one tier, in its own process (each release ships its own
``kniv_v7``). The release's end-to-end time (INT8, as ``v6.export.bench``:
full linguistic layer, 12 relations) on Re-DocRED dev text of 64 and 500
words, then the V2 graph untrained -- the marker pass (encoder with its
last-layer attention, as the release's marker graph) and the candidate
reader (``v6.train.verifier2.Reader``) -- on random ids of the same
lengths: 22 candidates on 64 words (128 subwords, 12 entities) and 570 on
500 words (768 subwords, 60 entities), three mentions each. Also the
reader alone, the cost if V2 read the release's own marker pass instead of
running its own. Median of three warm runs. Writes
``v6/runs/verifier2-onnx/cost-<tier>.json``.

``table``: every tier's total over the release's, short and long.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OUT = Path("v6/runs/verifier2-onnx")
TIERS = {"ettin-17m": ("jhu-clsp/ettin-encoder-17m", 256, 4), "ettin-68m": ("jhu-clsp/ettin-encoder-68m", 512, 8),
         "modernbert-base": ("answerdotai/ModernBERT-base", 768, 12),
         "ultrabert-380m": ("dmis-lab/UltraBERT-380M", 768, 12)}
CASES = {"short": (64, 128, 22, 12), "long": (500, 768, 570, 60)}     # words, subwords, candidates, entities
NAMES = ["ids", "mask", "men_pos", "men_ent", "pair_h", "pair_t", "lab"]


class V2Graph(nn.Module):
    """One window: the marker pass, then every candidate from it."""

    def __init__(self, enc, reader, E: int, reader_only: bool = False):
        super().__init__()
        self.enc, self.reader, self.E, self.reader_only = enc, reader, E, reader_only

    def forward(self, ids, mask, men_pos, men_ent, pair_h, pair_t, lab):
        from ..train.verifier2 import pair_inputs, pool
        if self.reader_only:            # the release's marker pass states, given
            seq, att = ids, mask
        else:
            from .graphs import _mask
            out = self.enc(input_ids=ids, attention_mask=_mask(self.enc, mask), output_attentions=True)
            seq, att = out.last_hidden_state[0].float(), out.attentions[-1][0].float()
        ent, e_att = pool(seq, att, men_pos, men_ent, self.E)
        T = seq.size(0)
        hs, ts, rs, hm, tm, lp = pair_inputs(seq, ent, e_att, men_pos, men_ent, pair_h, pair_t, T, self.E)
        key_ok = torch.ones(T, dtype=torch.bool)
        return self.reader(seq, key_ok, hs, ts, rs, lab, hm, tm, lp)


REPOS = {"ettin-17m": "tiny", "ettin-68m": "medium", "modernbert-base": "large", "ultrabert-380m": "xlarge-u"}


def _release_times(tier: str) -> dict:
    """The release's end-to-end INT8 time at 64 and 500 words (as bench.py),
    the release repo at its current tag."""
    from huggingface_hub import snapshot_download
    from .bench import RELATIONS
    from ..gold.redocred import load_rel_items
    rel = Path(snapshot_download(f"dragonscale-ai/kniv-v7-{REPOS[tier]}-en", revision="v7.1.2",
                                 ignore_patterns=["model.pt", "*-fp16.onnx"]))
    for m in [m for m in sys.modules if m == "kniv_v7" or m.startswith("kniv_v7.")]:
        del sys.modules[m]
    sys.path.insert(0, str(rel))
    from kniv_v7 import Kniv
    from kniv_v7.runtime import OnnxBackend
    k = Kniv(rel, backend=OnnxBackend(rel / "onnx", variant="int8"))
    words = [w for it in load_rel_items("dev")[:40] for w in it.tokens]
    out = {}
    for case, (n, *_) in CASES.items():
        text = " ".join(words[:n])
        k.analyse(text, relations=RELATIONS)
        runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            k.analyse(text, relations=RELATIONS)
            runs.append(time.perf_counter() - t0)
        out[case] = float(np.median(runs))
    return out


def _inputs(H, T, N, E, label_dim, rng, reader_only=False):
    M = 3 * E
    pos = np.sort(rng.choice(np.arange(1, T - 1), size=M, replace=False)).astype(np.int64)
    x = {"men_pos": pos, "men_ent": np.repeat(np.arange(E), 3).astype(np.int64),
         "pair_h": rng.integers(0, E, size=N).astype(np.int64), "pair_t": rng.integers(0, E, size=N).astype(np.int64),
         "lab": rng.standard_normal((N, label_dim)).astype(np.float32)}
    if reader_only:                     # the pass's states and attention stand in for ids / mask
        x["ids"] = rng.standard_normal((T, H)).astype(np.float32)
    else:
        x["ids"] = rng.integers(1000, 20000, size=(1, T)).astype(np.int64)
        x["mask"] = np.ones((1, T), dtype=np.int64)
    return x


def _time(sess, x):
    sess.run(None, x)
    runs = []
    for _ in range(3):
        t0 = time.perf_counter()
        sess.run(None, x)
        runs.append(time.perf_counter() - t0)
    return float(np.median(runs))


def cost(tier: str) -> int:
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModel
    from ..train.verifier2 import Reader
    torch.set_grad_enabled(False)
    rel = _release_times(tier)
    print(f"{tier}: release INT8 end to end {rel['short']:.3f} s at 64 words, {rel['long']:.3f} s at 500", flush=True)
    hf, d, heads = TIERS[tier]
    enc = AutoModel.from_pretrained(hf, attn_implementation="eager", trust_remote_code=True).eval()
    H = enc.config.hidden_size
    reader = Reader(H, H, d=d, layers=2, heads=heads).eval()
    rng = np.random.default_rng(0)
    d_out = OUT / tier
    d_out.mkdir(parents=True, exist_ok=True)
    res = {"tier": tier, "encoder": hf, "reader_d": d, "release_s": rel}
    for ro in (False, True):
        for case, (_, T, N, E) in CASES.items():
            g = V2Graph(enc, reader, E, reader_only=ro).eval()
            x = _inputs(H, T, N, E, H, rng, ro)
            if ro:
                x["mask"] = rng.random((enc.config.num_attention_heads, T, T)).astype(np.float32)
                x["mask"] /= x["mask"].sum(-1, keepdims=True)
            name = f"v2-{case}{'-reader' if ro else ''}"
            p = d_out / f"{name}.onnx"
            torch.onnx.export(g, tuple(torch.from_numpy(x[k]) for k in NAMES), str(p), input_names=NAMES,
                              output_names=["logits"], opset_version=17, do_constant_folding=True, dynamo=False)
            q = d_out / f"{name}-int8.onnx"
            quantize_dynamic(str(p), str(q), weight_type=QuantType.QInt8)
            s = ort.InferenceSession(str(q), providers=["CPUExecutionProvider"])
            t = _time(s, x)
            key = ("reader_" if ro else "v2_") + case
            res[key] = t
            print(f"  {key}: {N} candidates on {T} subwords: {t:.3f} s ({t / N * 1000:.2f} ms per candidate); "
                  f"release + this = {(rel[case] + t) / rel[case]:.2f}x", flush=True)
            p.unlink(); q.unlink()
            for f in d_out.glob(f"{name}*.data"):
                f.unlink()
    (OUT / f"cost-{tier}.json").write_text(json.dumps(res, indent=1))
    return 0


def table() -> int:
    print("| tier | release s, 64 / 500 words | V2 (own pass) s, 22 / 570 candidates | total over release | "
          "reader alone s | total over release, reading the release's pass |")
    print("|---|---|---|---|---|---|")
    for tier in TIERS:
        p = OUT / f"cost-{tier}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        rs, rl = r["release_s"]["short"], r["release_s"]["long"]
        print(f"| {tier} | {rs:.3f} / {rl:.3f} | {r['v2_short']:.3f} / {r['v2_long']:.3f} | "
              f"{(rs + r['v2_short']) / rs:.2f}x / {(rl + r['v2_long']) / rl:.2f}x | "
              f"{r['reader_short']:.3f} / {r['reader_long']:.3f} | "
              f"{(rs + r['reader_short']) / rs:.2f}x / {(rl + r['reader_long']) / rl:.2f}x |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["cost", "table", "all"])
    ap.add_argument("--tier", choices=list(TIERS))
    a = ap.parse_args()
    if a.cmd == "cost":
        return cost(a.tier)
    if a.cmd == "all":
        for t in TIERS:
            subprocess.run([sys.executable, "-m", "v6.export.verifier2_onnx", "cost", "--tier", t], check=True)
    return table()


if __name__ == "__main__":
    raise SystemExit(main())

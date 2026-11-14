"""The fact verifier as one ONNX graph, and what it costs per text on CPU.

    uv run --extra export python -m v6.export.verifier_onnx export [--run RUN] [--ver VER] [--out DIR]
    uv run --extra export python -m v6.export.verifier_onnx bench  [--out DIR] [--int8]
    uv run --extra export python -m v6.export.verifier_onnx student --encoder HF_ID

``export``: the verifier (plan r4 V1) over its base run, LoRA folded into a
copy of the encoder, as ``verifier.onnx``; checked against the PyTorch
verifier on real candidates (M1 short texts and stage-4 long windows).

``bench``: onnxruntime on CPU, per text, the verifier over that text's
verified candidates (batches of ``BATCH``), on M1 short-dev texts
(candidates as the deployed flat floor 3 and floor 1 read them) and on the
stage-4 pilot's long windows; set beside the release's measured end-to-end
time for texts of that length (``release/*/onnx/bench-*.json``).

``student``: the same graph shape over another encoder with untrained
adapter and head: its cost only, for a tier that has no verifier yet.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .graphs import merged_encoder, _mask

RUN, VER = "ultrabert-380m-v8p3-d123", "ultrabert-380m-v8p3-d123-ver1"
OUT = Path("v6/runs/verifier-onnx")
BATCH = 8
NAMES = (["ids", "mask", "cand_mask", "men_pos", "men_ent", "ent_ids", "pair_h", "pair_t"], ["logits"])
AXES = {"ids": {0: "B", 1: "T"}, "mask": {0: "B", 1: "T"}, "cand_mask": {0: "B", 1: "T"},
        "men_pos": {0: "M"}, "men_ent": {0: "M"}, "ent_ids": {0: "E"}, "pair_h": {0: "B"},
        "pair_t": {0: "B"}, "logits": {0: "B"}}


class VerifierGraph(nn.Module):
    """``B`` candidates of one text: marker text + candidate per row
    (``ids, mask, cand_mask [B,T]``); the text's mentions (``men_pos [M]``,
    the same in every row since the candidate comes after the text;
    ``men_ent [M]`` in 0..E-1, ``ent_ids [E]``); each row's pair
    (``pair_h, pair_t [B]``). Out: verdict logits ``[B,4]``."""

    def __init__(self, enc, relation, head):
        super().__init__()
        self.enc, self.r, self.head = enc, relation, head

    def forward(self, ids, mask, cand_mask, men_pos, men_ent, ent_ids, pair_h, pair_t):
        out = self.enc(input_ids=ids, attention_mask=_mask(self.enc, mask), output_attentions=True)
        seq, att = out.last_hidden_state.float(), out.attentions[-1].float()   # [B,T,H], [B,heads,T,T]
        ok = men_pos >= 0
        pc = men_pos.clamp(min=0)
        member = ((men_ent.unsqueeze(0) == ent_ids.unsqueeze(1)) & ok.unsqueeze(0)).float()   # [E,M]
        m_emb = seq[:, pc].masked_fill(~ok.view(1, -1, 1), -1e4)                # [B,M,H]
        e_max = torch.where(member.bool().view(1, member.size(0), -1, 1), m_emb.unsqueeze(1),
                            torch.full_like(m_emb, -1e4).unsqueeze(1)).amax(2)  # [B,E,H]
        ex = torch.exp(m_emb - e_max[:, men_ent]) * ok.view(1, -1, 1).float()
        ent = e_max + torch.log(torch.einsum("em,bmh->beh", member, ex).clamp(min=1e-30))
        ent = torch.where(ent > -1e3, ent, torch.zeros_like(ent))
        cnt = member.sum(1).clamp(min=1)
        m_att = att[:, :, pc]                                                    # [B,heads,M,T]
        e_att = torch.einsum("em,bkmt->bket", member, m_att) / cnt.view(1, 1, -1, 1)   # [B,heads,E,T]
        rows = torch.arange(ids.size(0))
        ah = e_att[rows, :, pair_h]                                              # [B,heads,T]
        at = e_att[rows, :, pair_t]
        ht = (ah * at).mean(1)
        ht = ht / (ht.sum(1, keepdim=True) + 1e-5)
        rs = torch.einsum("bt,bth->bh", ht, seq)
        hs, ts = ent[rows, pair_h], ent[rows, pair_t]
        cm = cand_mask.float().unsqueeze(-1)
        cand = (seq * cm).sum(1) / cm.sum(1).clamp(min=1)
        return self.head(torch.cat([hs, ts, rs, cand], -1)).float()


def _inputs(exs, pad):
    """One text's candidates (``verifier.encode`` examples) as graph inputs."""
    T = max(len(e["ids"]) for e in exs)
    B = len(exs)
    ids = np.full((B, T), pad, dtype=np.int64)
    mask = np.zeros((B, T), dtype=np.int64)
    cm = np.zeros((B, T), dtype=bool)
    for b, e in enumerate(exs):
        n = len(e["ids"])
        ids[b, :n] = e["ids"]; mask[b, :n] = 1; cm[b, e["n_text"]:n - 1] = True
    men_ent, men_pos = [], []
    for k, pos in enumerate(exs[0]["men"]):
        for p in pos.tolist():
            men_ent.append(k); men_pos.append(p)
    E = len(exs[0]["men"])
    return {"ids": ids, "mask": mask, "cand_mask": cm, "men_pos": np.asarray(men_pos, dtype=np.int64),
            "men_ent": np.asarray(men_ent, dtype=np.int64), "ent_ids": np.arange(E, dtype=np.int64),
            "pair_h": np.asarray([e["h"] for e in exs], dtype=np.int64),
            "pair_t": np.asarray([e["t"] for e in exs], dtype=np.int64)}


def _texts(limit_short=30, limit_long=6):
    """Real texts with the candidates the deployed verifier reads: M1 short
    dev (the stage-3 cost sweep's cache, floor 3 and floor 1 four per pair)
    and stage-4 pilot long windows (every logged candidate)."""
    out = []
    C = pickle.loads(Path("v6/runs/m1-precision/s3cost-cache-flat.pkl").read_bytes())
    items = {it["id"]: it for it in pickle.loads(Path("v6/runs/m1-precision/preds-m1s3b-n500.pkl")
                                                  .read_bytes())["items"]}
    by = {}
    for (tid, h, t, rel), v in C["ver"].items():
        by.setdefault(tid, []).append((h, t, rel))
    for tid in list(by)[:limit_short]:
        out.append(("short", items[tid], C["ents"][tid], by[tid]))
    S = json.loads(Path("v6/runs/s4-pilot/items.json").read_text())
    rows = pickle.loads(Path("v6/runs/s4-pilot/pred-000.pkl").read_bytes())
    its = {it["id"]: it for it in S}
    for r in rows[:limit_long]:
        out.append(("long", its[r["id"]], None, [(c["head"], c["tail"], c["relation"]) for c in r["cands"]]))
    return out


def _encode_text(ex, it, ents, cands):
    """The verifier's examples for one text, built as ``Extractor._verify``
    builds them (entities from the extractor's own entity step)."""
    from ..train.verifier import encode
    w = {"tokens": it["tokens"], "sentence_spans": it["sentence_spans"]}
    if it.get("speakers"):
        w["sentence_speakers"] = it["speakers"]
    cap = {}
    orig = ex._verify

    def grab(w_, nodes, spans, parts, cs):
        cap.update(nodes=nodes, spans=spans, parts=parts)
        ex._last_verdicts = {}
        return [None] * len(cs)
    ex._verify = grab
    try:
        e = ents if ents is not None else ex._entities(w, it.get("types"))
        ex._relations(w, e, list(it["relations"]), ex.top)
    finally:
        ex._verify = orig
    if not cap:
        return []
    rec = {"tokens": w["tokens"], "sentence_spans": w["sentence_spans"],
           "speakers": w.get("sentence_speakers"), "participants": cap["parts"],
           "entities": [[tuple(int(x) for x in m) for m in ms] for ms in cap["spans"]],
           "names": [n["text"] for n in cap["nodes"]]}
    cache, exs = {}, []
    for h, t, rel in cands:
        e = encode(rec, {"h": h, "t": t, "rel": rel, "y": 0}, ex.tok, ex.ver_bank, None, None, cache=cache)
        if e is not None:
            exs.append(e)
    return exs


def _extractor():
    from ..infer import Extractor
    ex = Extractor(RUN, device="cpu")
    ex.use_verifier(VER, threshold=0.6)
    return ex


def export(out: Path) -> int:
    from ..train.verifier import collate
    ex = _extractor()
    v = ex.verifier
    g = VerifierGraph(merged_encoder(ex.model, v.ver), ex.model.relation, v.head).eval()
    texts = _texts(limit_short=12, limit_long=3)
    sets = []
    for kind, it, ents, cands in texts:
        exs = _encode_text(ex, it, ents, cands)[:BATCH]
        if exs:
            sets.append((kind, exs))
    pad = ex.tok.pad_token_id
    worst = 0.0
    with torch.no_grad():
        for kind, exs in sets:
            ref = v(collate(exs, pad, "cpu")).numpy()
            got = g(**{k: torch.from_numpy(x) for k, x in _inputs(exs, pad).items()}).numpy()
            worst = max(worst, float(np.abs(ref - got).max()))
    print(f"graph vs PyTorch verifier: max |diff| {worst:.2e} over {len(sets)} texts")
    out.mkdir(parents=True, exist_ok=True)
    x = {k: torch.from_numpy(a) for k, a in _inputs(sets[0][1], pad).items()}
    torch.onnx.export(g, tuple(x[k] for k in NAMES[0]), str(out / "verifier.onnx"), input_names=NAMES[0],
                      output_names=NAMES[1], dynamic_axes=AXES, opset_version=17, do_constant_folding=True,
                      dynamo=False)
    import onnxruntime as ort
    s = ort.InferenceSession(str(out / "verifier.onnx"), providers=["CPUExecutionProvider"])
    worst = 0.0
    with torch.no_grad():
        for kind, exs in sets:
            ref = v(collate(exs, pad, "cpu")).numpy()
            got = s.run(None, _inputs(exs, pad))[0]
            worst = max(worst, float(np.abs(ref - got).max()))
    print(f"onnxruntime vs PyTorch: max |diff| {worst:.2e}")
    return 0


def quantize(out: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    q = out / "verifier-int8.onnx"
    if not q.exists():
        quantize_dynamic(str(out / "verifier.onnx"), str(q), weight_type=QuantType.QInt8)
    return q


def _time(sess, exs, pad):
    t0 = time.perf_counter()
    for i in range(0, len(exs), BATCH):
        sess.run(None, _inputs(exs[i:i + BATCH], pad))
    return time.perf_counter() - t0


def bench(out: Path, int8: bool) -> int:
    import onnxruntime as ort
    ex = _extractor()
    pad = ex.tok.pad_token_id
    path = quantize(out) if int8 else out / "verifier.onnx"
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    floor1 = pickle.loads(Path("v6/runs/m1-precision/s3cost-flat-f1k4.pkl").read_bytes())["n_read"] \
        if Path("v6/runs/m1-precision/s3cost-flat-f1k4.pkl").exists() else None
    res = {"short": [], "long": []}
    for kind, it, ents, cands in _texts():
        exs = _encode_text(ex, it, ents, cands)
        if not exs:
            continue
        _time(sess, exs[:BATCH], pad)                       # warm-up
        t = _time(sess, exs, pad)
        res[kind].append({"words": len(it["tokens"]), "cands": len(exs), "s": t,
                          "subwords": int(np.mean([len(e["ids"]) for e in exs]))})
        print(f"  {kind} {len(it['tokens'])} words, {len(exs)} candidates: {t:.2f} s", flush=True)
    rel = json.loads(Path("release/ultrabert-380m/onnx/bench-int8.json").read_text())
    print(f"verifier {'int8' if int8 else 'fp32'} onnxruntime CPU, batches of {BATCH}")
    for kind, rs in res.items():
        if not rs:
            continue
        w = np.mean([r["words"] for r in rs]); n = np.mean([r["cands"] for r in rs])
        t = np.mean([r["s"] for r in rs]); per = t / n
        print(f"  {kind}: {len(rs)} texts, {w:.0f} words, {np.mean([r['subwords'] for r in rs]):.0f} subwords "
              f"per input, {n:.1f} candidates per text, {t:.2f} s per text ({per * 1000:.0f} ms per candidate)")
    print("release (int8) end to end: " + ", ".join(f"{k} {v['total_s']:.3f} s (marker {v['graphs_s']['marker']:.3f})"
                                                  for k, v in rel.items() if k.endswith("words")))
    (out / f"bench-{'int8' if int8 else 'fp32'}.json").write_text(json.dumps(res))
    return 0


def student(enc_id: str, int8: bool) -> int:
    """Cost only: the verifier's shape over ``enc_id`` (untrained), timed on
    random ids of the short and long inputs' lengths."""
    import onnxruntime as ort
    from transformers import AutoModel
    enc = AutoModel.from_pretrained(enc_id, attn_implementation="eager").eval()
    H = enc.config.hidden_size

    class R(nn.Module):
        pass
    head = nn.Sequential(nn.LayerNorm(4 * H), nn.Linear(4 * H, 512), nn.GELU(), nn.Linear(512, 4))
    g = VerifierGraph(enc, R(), head).eval()
    out = OUT / enc_id.replace("/", "--")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    def fake(B, T, E=12):
        ids = rng.integers(1000, 20000, size=(B, T)).astype(np.int64)
        cm = np.zeros((B, T), dtype=bool); cm[:, T - 20:T - 1] = True
        pos = np.sort(rng.choice(T - 30, size=3 * E, replace=False)).astype(np.int64)
        return {"ids": ids, "mask": np.ones((B, T), dtype=np.int64), "cand_mask": cm, "men_pos": pos,
                "men_ent": np.repeat(np.arange(E), 3).astype(np.int64), "ent_ids": np.arange(E, dtype=np.int64),
                "pair_h": rng.integers(0, E, size=B).astype(np.int64), "pair_t": rng.integers(0, E, size=B).astype(np.int64)}
    x = fake(BATCH, 128)
    p = out / "verifier.onnx"
    torch.onnx.export(g, tuple(torch.from_numpy(x[k]) for k in NAMES[0]), str(p), input_names=NAMES[0],
                      output_names=NAMES[1], dynamic_axes=AXES, opset_version=17, dynamo=False)
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q = out / "verifier-int8.onnx"
        quantize_dynamic(str(p), str(q), weight_type=QuantType.QInt8)
        p = q
    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    for T in (128, 768):
        x = fake(BATCH, T)
        s.run(None, x)
        t0 = time.perf_counter()
        for _ in range(3):
            s.run(None, x)
        print(f"{enc_id} {'int8' if int8 else 'fp32'}: {T} subwords, "
              f"{(time.perf_counter() - t0) / 3 / BATCH * 1000:.0f} ms per candidate")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["export", "bench", "student"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--encoder")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    if a.cmd == "export":
        return export(a.out)
    if a.cmd == "bench":
        return bench(a.out, a.int8)
    return student(a.encoder, a.int8)


if __name__ == "__main__":
    raise SystemExit(main())

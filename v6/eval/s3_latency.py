"""Stage 3 (v8 plan r4): what the verifier costs per text on CPU.

    uv run python -m v6.eval.s3_latency [--n 50] [--threads 8]

The extractor (PyTorch, CPU) on the first ``n`` M1 short-dev texts, each
decode with and without the verifier: flat (the deployed entity step) and
nested (ner + np + nest + merge). Time per text (median and mean, after one
warm-up text) and candidates the verifier read per text. The ratio is what
transfers to the ONNX runtime; the absolute times are PyTorch's.
"""
from __future__ import annotations

import argparse
import statistics
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="ultrabert-380m-v8p3-d123")
    ap.add_argument("--verifier", default="ultrabert-380m-v8p3-d123-ver1")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--settings", default="3:4", help="nested with the verifier: FLOOR:VER_TOP,... (each its own row)")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    from .m1_precision import _configure, load_set
    from ..infer import Extractor
    items = load_set("short", a.n + 1)
    ex = Extractor(a.run, device="cpu")
    boost0 = ex.span_boost
    ex.use_verifier(a.verifier, threshold=0.6)
    n_read = []
    orig = ex._verify

    def counted(w, nodes, spans, parts, cands):
        n_read.append(len(cands))
        return orig(w, nodes, spans, parts, cands)
    ex._verify = counted
    print("| decode | verifier | median s / text | mean s / text | candidates verified / text |\n|---|---|---|---|---|")
    base = {}
    runs = [("base", False, None), ("base", True, None), ("ner+np+nest+merge:2", False, None)]
    for s in a.settings.split(","):         # FLOOR:VER_TOP for nested, flat:FLOOR:VER_TOP for flat
        flat = s.startswith("flat:")
        runs.append(("base" if flat else "ner+np+nest+merge:2", True,
                     tuple(float(x) for x in s.split(":")[1 if flat else 0:])))
    for dec, ver, st in runs:
        if True:
            name, _, boost = dec.partition(":")
            cfg = ("ver" if ver else "base") if name == "base" else name + ("+ver" if ver else "") + (":" + boost if boost else "")
            _configure(ex, cfg, boost0)
            ex.margin_shift = 0.0
            ex.ver_floor, ex.ver_top = (st[0], int(st[1])) if st else (3.0, None)
            ts = []
            n_read.clear()
            for j, it in enumerate(items):
                w = {"tokens": it["tokens"], "sentence_spans": it["sentence_spans"]}
                if it.get("speakers"):
                    w["sentence_speakers"] = it["speakers"]
                t0 = time.perf_counter()
                ents = ex._entities(w, it["types"])
                ex._relations(w, ents, list(it["relations"]), ex.top)
                if j:
                    ts.append(time.perf_counter() - t0)
            med, mean = statistics.median(ts), statistics.mean(ts)
            if not ver:
                base[dec] = mean
            ratio = f" ({mean / base[dec]:.2f}x)" if ver else ""
            lab = ('flat' if dec == 'base' else 'nested') + (f" f{st[0]:g}k{int(st[1])}" if st else "")
            print(f"| {lab} | {'yes' if ver else 'no'} | {med:.2f} | {mean:.2f}{ratio} | "
                  f"{sum(n_read) / len(items):.1f} |", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

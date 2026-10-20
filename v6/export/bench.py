"""How fast is a release on this machine? Per graph and end to end.

    uv run --extra export python -m v6.export.bench --release release/ultrabert-380m [--variant fp16]

Windows of about 40, 200 and 500 words (Re-DocRED dev text), analysed
with the full linguistic layer and 12 relations over the DocRED entity head's
entities; three warm runs each after one warm-up, median reported.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

RELATIONS = [{"name": n} for n in ("place of birth", "employer", "country", "member of",
                                   "located in", "founded by", "author", "spouse",
                                   "part of", "date of birth", "headquarters location", "child")]


class Timed:
    """A backend wrapper that times each graph call."""

    def __init__(self, inner):
        self.inner, self.t = inner, defaultdict(float)

    def run(self, name, **x):
        t0 = time.perf_counter()
        out = self.inner.run(name, **x)
        self.t[name] += time.perf_counter() - t0
        return out


def texts() -> dict[str, str]:
    from ..gold.redocred import load_rel_items
    words = [w for it in load_rel_items("dev")[:40] for w in it.tokens]
    out = {}
    for n in (40, 200, 500):
        out[f"{n} words"] = " ".join(words[:n])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--variant")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, str(args.release))
    from kniv_v7 import Kniv
    from kniv_v7.runtime import OnnxBackend
    be = OnnxBackend(args.release / "onnx", variant=args.variant)
    if args.threads:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        orig = be._session

        def session(name):
            if name not in be._sess:
                path = be.dir / f"{name}-{be.variant}.onnx"
                if not (be.variant and path.exists()):
                    path = be.dir / f"{name}.onnx"
                be._sess[name] = ort.InferenceSession(str(path), so, providers=be.providers)
            return be._sess[name]
        be._session = session
        del orig
    tb = Timed(be)
    k = Kniv(args.release, backend=tb)
    res = {"machine": platform.processor() or platform.machine(), "variant": args.variant or "fp32",
           "threads": args.threads or "default"}
    for label, text in texts().items():
        k.analyse(text, relations=RELATIONS)                # warm-up (sessions, shapes)
        runs, graphs = [], []
        for _ in range(3):
            tb.t.clear()
            t0 = time.perf_counter()
            out = k.analyse(text, relations=RELATIONS)
            runs.append(time.perf_counter() - t0)
            graphs.append(dict(tb.t))
        med = int(np.argsort(runs)[1])
        g = graphs[med]
        res[label] = {"total_s": round(runs[med], 3), "host_s": round(runs[med] - sum(g.values()), 3),
                      "graphs_s": {n: round(v, 3) for n, v in sorted(g.items(), key=lambda kv: -kv[1])},
                      "entities": len(out[0]["entities"]), "facts": len(out[0]["facts"])}
        print(label, json.dumps(res[label]), flush=True)
    (args.release / "onnx" / f"bench-{args.variant or 'fp32'}.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

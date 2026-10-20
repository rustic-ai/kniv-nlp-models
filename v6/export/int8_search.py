"""INT8 that holds accuracy: find the layers that break it, keep them fp32.

    uv run --extra export python -m v6.export.int8_search stats --run R --release DIR
    uv run --extra export python -m v6.export.int8_search build --release DIR --variant V \
        --graphs plain,label,marker --exclude-ratio 20 [--per-channel] [--static]

``stats``: every encoder linear's input activations in each pass (plain,
label, marker) on real windows -- absolute max, the 99.9th percentile of
per-token absolute max, and their ratio (an outlier ratio: one scale per
tensor spends its range on the max). Written to ``<release>/onnx/act_stats.json``.

``build``: INT8 copies ``<graph>-<variant>.onnx`` of the encoder graphs with
every linear whose input outlier ratio exceeds ``--exclude-ratio`` (or whose
absmax exceeds ``--exclude-absmax``) left fp32.
- dynamic (default): per-tensor activation scales at run time; ``--per-channel``
  for per-channel weights (2-D weights only).
- ``--static``: QDQ with activation ranges calibrated on real windows
  (percentile calibration), ``--smooth`` adding SmoothQuant's rescaling.

Score with ``int8_eval --variants V``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ENCODERS = ("plain", "label", "marker")


def texts():
    """Real windows: Re-DocRED train documents and UD sentences (not the dev
    items int8_eval scores), plus conversations."""
    from ..gold.redocred import load_rel_items
    from ..gold.ud_ewt import load_ud_items
    docs = load_rel_items("train", limit=40)
    sents = load_ud_items("train", limit=60)
    out = [{"tokens": d.tokens, "sentence_spans": [list(x) for x in d.layers["sentence_spans"]],
            "entities": [{"type": e["type"], "mentions": [(a, b + 1) for a, b in e["mentions"]]}
                         for e in d.entities]} for d in docs]
    out += [{"tokens": s.tokens, "sentence_spans": [[0, len(s.tokens)]], "entities": []} for s in sents]
    return out


class Recorder:
    """A backend wrapper that keeps every encoder graph's inputs."""

    def __init__(self, inner, keep=ENCODERS):
        self.inner, self.keep, self.inputs = inner, keep, {k: [] for k in keep}

    def run(self, name, **x):
        if name in self.keep:
            self.inputs[name].append({k: np.asarray(v) for k, v in x.items()})
        return self.inner.run(name, **x)


RELATIONS = [{"name": n} for n in ("place of birth", "employer", "country", "member of", "located in",
                                   "founded by", "author", "spouse", "part of", "date of birth")]
TYPES = [{"name": n} for n in ("person", "location", "organization", "date", "event")]


def record(release: Path, backend) -> dict[str, list[dict]]:
    sys.path.insert(0, str(release))
    from kniv_v7 import Kniv
    rec = Recorder(backend)
    k = Kniv(release, backend=rec)
    for w in texts():
        out = k.window({"tokens": w["tokens"], "sentence_spans": w["sentence_spans"]}, RELATIONS,
                       TYPES if not w["entities"] else None, True, 4)
        del out
        if w["entities"]:      # the marker pass over gold entities too
            k.relations({}, w["tokens"], [tuple(x) for x in w["sentence_spans"]], w["entities"],
                        RELATIONS, 4)
        k._label_cache.clear()
    return rec.inputs


def stats(run: str, release: Path) -> int:
    from ..train.gold_eval import HF_REPO, load_run
    from .graphs import GRAPHS
    from .onnx_v7 import SPEC
    from .torch_backend import TorchBackend
    torch.backends.mha.set_fastpath_enabled(False)
    model, *_ = load_run(run, HF_REPO, torch.device("cpu"), Path("data/v6-eval/checkpoints"))
    tb = TorchBackend(model)
    inputs = record(release, tb)
    out = {}
    for g in ENCODERS:
        enc = tb.g[g].enc
        acc: dict[str, list] = {}

        def hook(name):
            def f(mod, inp, _):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1]).abs()
                acc.setdefault(name, []).append((float(x.max()), x.max(1).values.numpy()))
            return f
        hs = [m.register_forward_hook(hook(n)) for n, m in enc.named_modules()
              if isinstance(m, torch.nn.Linear)]
        for x in inputs[g]:
            with torch.no_grad():
                tb.g[g](*[torch.from_numpy(np.asarray(x[k])) if k in x else torch.zeros(1, 1, dtype=torch.long)
                          for k in SPEC[g][0]])
        for h in hs:
            h.remove()
        rows = {}
        for n, v in acc.items():
            amax = max(a for a, _ in v)
            tok = np.concatenate([t for _, t in v])
            p999 = float(np.percentile(tok, 99.9))
            rows[n] = {"absmax": round(amax, 2), "p99.9_token_max": round(p999, 2),
                       "median_token_max": round(float(np.median(tok)), 2),
                       "ratio": round(amax / max(float(np.median(tok)), 1e-6), 2)}
        out[g] = rows
        top = sorted(rows.items(), key=lambda kv: -kv[1]["ratio"])[:8]
        print(g, "highest outlier ratios:", [(n, r["absmax"], r["ratio"]) for n, r in top], flush=True)
    (release / "onnx" / "act_stats.json").write_text(json.dumps(out, indent=1))
    return 0


def onnx_names(path: Path, linears: set[str]) -> list[str]:
    """ONNX MatMul/Gemm node names for the given encoder linears (exported
    under ``/enc/<module path>/``)."""
    import onnx
    m = onnx.load(str(path), load_external_data=False)
    out = []
    for n in m.graph.node:
        if n.op_type not in ("MatMul", "Gemm"):
            continue
        mod = re.sub(r"^/enc/", "", n.name).rsplit("/", 1)[0].replace("/", ".")
        if mod in linears:
            out.append(n.name)
    return out


def two_d_weights(path: Path) -> list[str]:
    """MatMul nodes whose weight is a 2-D initializer (per-channel needs it)."""
    import onnx
    m = onnx.load(str(path))
    inits = {i.name: len(i.dims) for i in m.graph.initializer}
    return [n.name for n in m.graph.node if n.op_type == "MatMul"
            and any(inits.get(i) == 2 for i in n.input)]


class Reader:
    """Calibration inputs for one graph (static quantisation)."""

    def __init__(self, items, names):
        self.items = [{k: v for k, v in x.items() if k in names} for x in items]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.items):
            return None
        self.i += 1
        return self.items[self.i - 1]

    def rewind(self):
        self.i = 0


def build(args) -> int:
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                          quantize_dynamic, quantize_static)
    d = args.release / "onnx"
    st = json.loads((d / "act_stats.json").read_text())
    calib = None
    if args.static:
        sys.path.insert(0, str(args.release))
        from kniv_v7.runtime import OnnxBackend
        calib = record(args.release, OnnxBackend(d))
    for g in args.graphs.split(","):
        src = d / f"{g}.onnx"
        lin = {n for n, r in st[g].items()
               if r["ratio"] > args.exclude_ratio or r["absmax"] > args.exclude_absmax
               or any(p in n for p in args.exclude.split(",") if p)
               or (args.exclude_re and re.search(args.exclude_re, n))}
        excl = onnx_names(src, lin)
        dst = d / f"{g}-{args.variant}.onnx"
        if args.static:
            import onnxruntime as ort
            names = {i.name for i in ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
                     .get_inputs()}
            extra = {"ActivationSymmetric": False, "WeightSymmetric": True}
            if args.smooth:
                extra.update(SmoothQuant=True, SmoothQuantAlpha=args.alpha)
            quantize_static(str(src), str(dst), Reader(calib[g][:args.calib], names),
                            quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                            weight_type=QuantType.QInt8, per_channel=args.per_channel,
                            nodes_to_exclude=excl, op_types_to_quantize=["MatMul"],
                            calibrate_method=CalibrationMethod.Percentile,
                            use_external_data_format=True, extra_options=extra)
        else:
            nodes = None
            if args.per_channel:
                nodes = [n for n in two_d_weights(src) if n not in set(excl)]
            quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8,
                             per_channel=args.per_channel, nodes_to_exclude=excl,
                             nodes_to_quantize=nodes, op_types_to_quantize=["MatMul"])
        print(f"{dst.name}: {len(excl)} linears kept fp32 of {len(st[g])}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--run", required=True)
    s.add_argument("--release", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--release", type=Path, required=True)
    b.add_argument("--variant", required=True)
    b.add_argument("--graphs", default="plain,label,marker")
    b.add_argument("--exclude-ratio", type=float, default=float("inf"))
    b.add_argument("--exclude-absmax", type=float, default=float("inf"))
    b.add_argument("--exclude", default="", help="comma-separated substrings of linear names")
    b.add_argument("--exclude-re", default="", help="regex over linear names")
    b.add_argument("--per-channel", action="store_true")
    b.add_argument("--static", action="store_true")
    b.add_argument("--smooth", action="store_true")
    b.add_argument("--alpha", type=float, default=0.5)
    b.add_argument("--calib", type=int, default=64)
    args = ap.parse_args()
    return stats(args.run, args.release) if args.cmd == "stats" else build(args)


if __name__ == "__main__":
    raise SystemExit(main())

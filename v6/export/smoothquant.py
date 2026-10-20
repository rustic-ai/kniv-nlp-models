"""SmoothQuant on the layers INT8 had to keep fp32 (step 4 of the v7
compression plan).

    uv run --extra export python -m v6.export.smoothquant --run R --release DIR \
        --alpha 0.5,0.75 [--graphs plain,marker]

The feed-forward down-projections carry the activation outliers (UltraBERT
layer 14: inputs ~8,000 against a typical token maximum of ~4), so per-tensor
INT8 had to leave them fp32. SmoothQuant (Xiao et al. 2023) moves the
difficulty into the weights: per input channel ``j``, ``s_j = max|x_j|^a /
max|W_:j|^(1-a)``, and the layer computes ``(x / s) @ (W * s)^T`` -- the same
function, with the outlier channels shrunk. Here the division is an explicit
fp32 multiply before the layer (the preceding operation is a nonlinearity,
so it cannot be folded), and the rescaled weight is an ordinary 2-D weight
that per-channel INT8 then quantises like every other.

Per alpha, for each graph: the smoothed fp32 graph (checked against the
unsmoothed PyTorch graph), then ``<graph>-sq<alpha>.onnx``: per-channel INT8
of every linear. Score with ``int8_eval --variants sq<alpha>,fp16`` (the
label graph stays FP16).
"""
from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .int8_search import record, two_d_weights

DOWN = {"ultrabert": r"mlp\.down_proj$", "deberta": r"layer\.\d+\.output\.dense$"}


class SmoothLinear(nn.Module):
    """``lin(x / s)`` with the scale folded into the weight: ``x * inv_s`` then
    a linear with ``W * s``."""

    def __init__(self, lin: nn.Linear, s: torch.Tensor):
        super().__init__()
        self.register_buffer("inv_s", (1.0 / s).to(lin.weight.dtype))
        self.lin = nn.Linear(lin.in_features, lin.out_features, bias=lin.bias is not None)
        with torch.no_grad():
            self.lin.weight.copy_(lin.weight * s.to(lin.weight.dtype))
            if lin.bias is not None:
                self.lin.bias.copy_(lin.bias)

    def forward(self, x):
        return self.lin(x * self.inv_s)


def channel_absmax(graph, spec_inputs, items, pattern: str) -> dict[str, torch.Tensor]:
    """Per input channel absmax of the matching linears of ``graph.enc``."""
    acc: dict[str, torch.Tensor] = {}

    def hook(name):
        def f(_, inp, __):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1]).abs().amax(0)
            acc[name] = torch.maximum(acc[name], x) if name in acc else x
        return f
    hs = [m.register_forward_hook(hook(n)) for n, m in graph.enc.named_modules()
          if isinstance(m, nn.Linear) and re.search(pattern, n)]
    for x in items:
        with torch.no_grad():
            graph(*[torch.from_numpy(np.asarray(x[k])) if k in x else torch.zeros(1, 1, dtype=torch.long)
                    for k in spec_inputs])
    for h in hs:
        h.remove()
    return acc


def smooth(graph, amax: dict[str, torch.Tensor], alpha: float):
    """A copy of ``graph`` with each measured linear replaced by ``SmoothLinear``."""
    g = copy.deepcopy(graph)
    for name, a in amax.items():
        parent, _, leaf = name.rpartition(".")
        mod = g.enc.get_submodule(parent) if parent else g.enc
        lin = getattr(mod, leaf)
        w = lin.weight.detach().float().abs().amax(0).clamp(min=1e-5)
        s = (a.clamp(min=1e-5) ** alpha / w ** (1 - alpha)).clamp(min=1e-5)
        setattr(mod, leaf, SmoothLinear(lin, s))
    return g.eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--alpha", default="0.5,0.75")
    ap.add_argument("--graphs", default="plain,marker")
    args = ap.parse_args()
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from ..train.gold_eval import HF_REPO, load_run
    from .check_graphs import diff
    from .onnx_v7 import OPSET, SPEC
    from .torch_backend import TorchBackend
    torch.backends.mha.set_fastpath_enabled(False)
    model, *_ = load_run(args.run, HF_REPO, torch.device("cpu"), Path("data/v6-eval/checkpoints"))
    tb = TorchBackend(model)
    pattern = DOWN["ultrabert" if "ultrabert" in args.run else "deberta"]
    inputs = record(args.release, tb)
    d = args.release / "onnx"
    for g in args.graphs.split(","):
        ins, outs, axes = SPEC[g]
        items = inputs[g]
        calib, check = items[: len(items) * 3 // 4], items[len(items) * 3 // 4:]
        amax = channel_absmax(tb.g[g], ins, calib, pattern)
        print(f"{g}: {len(amax)} linears smoothed; largest input channel "
              f"{max(float(a.max()) for a in amax.values()):.0f}", flush=True)
        for alpha in map(float, args.alpha.split(",")):
            sg = smooth(tb.g[g], amax, alpha)
            worst = 0.0
            for x in check[:6]:                       # same function: ~fp32 rounding only
                feed = [torch.from_numpy(np.asarray(x[k])) if k in x else torch.zeros(1, 1, dtype=torch.long)
                        for k in ins]
                with torch.no_grad():
                    a, b = sg(*feed), tb.g[g](*feed)
                for u, v in zip(a, b):
                    if u.is_floating_point() and u.numel():
                        worst = max(worst, float(((u - v).abs() / (v.abs() + 1)).max()))
            ex = check[0]
            example = tuple(torch.from_numpy(np.asarray(ex[k])) if k in ex
                            else torch.zeros(1, 1, dtype=torch.long) for k in ins)
            tag = f"sq{int(alpha * 100)}"
            fp = d / f"{g}-{tag}-fp32.onnx"
            torch.onnx.export(sg, example, fp, input_names=ins, output_names=outs, dynamic_axes=axes,
                              opset_version=OPSET, dynamo=False, external_data=True)
            q = d / f"{g}-{tag}.onnx"
            quantize_dynamic(str(fp), str(q), weight_type=QuantType.QInt8, per_channel=True,
                             nodes_to_quantize=two_d_weights(fp), op_types_to_quantize=["MatMul"])
            ort.InferenceSession(str(q), providers=["CPUExecutionProvider"])     # loads
            for p in (fp, fp.with_name(fp.name + ".data")):
                if p.exists():
                    p.unlink()
            print(f"  {q.name}: smoothing changes outputs by at most {worst:.1e} (relative); "
                  f"{q.stat().st_size / 2 ** 30:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

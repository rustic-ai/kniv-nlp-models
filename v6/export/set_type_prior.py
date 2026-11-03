"""Flatten a release's entity-type prior in place: ``span.type_bias`` reads
nothing from the type embedding (weight 0) and adds one constant.

    uv run --extra export python -m v6.export.set_type_prior --release release/ettin-17m \
        --bias -3.0734 --span-offset -3.5

The learned prior favours generic types (miscellaneous, location, person) by
about 2 points over specific ones, which costs zero-shot typing on the
smaller models. ``--bias`` is the prior's mean over the evaluated types, so
the span offset moves little; ``--span-offset`` is the one chosen on
CrossNER politics with the flat prior. Writes ``model.pt``, every
``span*.onnx`` and ``calibration.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--bias", type=float, required=True)
    ap.add_argument("--span-offset", type=float, required=True)
    args = ap.parse_args()
    ck_path = args.release / "model.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    w, b = ck["model"]["span.type_bias.weight"], ck["model"]["span.type_bias.bias"]
    ck["model"]["span.type_bias.weight"] = torch.zeros_like(w)
    ck["model"]["span.type_bias.bias"] = torch.full_like(b, args.bias)
    torch.save(ck, ck_path)
    print(f"{ck_path}: type prior flat at {args.bias:+.4f}")
    for p in sorted((args.release / "onnx").glob("span*.onnx")):
        m = onnx.load(str(p))
        init = {i.name: i for i in m.graph.initializer}
        for name, fill in (("span.type_bias.weight", 0.0), ("span.type_bias.bias", args.bias)):
            a = numpy_helper.to_array(init[name])
            init[name].CopyFrom(numpy_helper.from_array(np.full_like(a, fill), name))
        onnx.save(m, str(p))
        print(f"{p}: type prior flat at {args.bias:+.4f}")
    cal_path = args.release / "calibration.json"
    cal = json.loads(cal_path.read_text())
    cal["span_offset"] = args.span_offset
    cal_path.write_text(json.dumps(cal, indent=1))
    print(f"{cal_path}: span_offset {args.span_offset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

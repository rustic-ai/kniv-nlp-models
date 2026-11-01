"""INT8 for what the encoder recipe leaves in fp32: the embedding tables and
the head graphs. Matters for small students, where those are most of the
weights (Ettin-17M: a 13M-parameter embedding table, 3.9M in the transformer
layers, about 38M in the heads).

    uv run --extra export python -m v6.export.compact_int8 --release release/ettin-17m

- ``plain-int8`` / ``marker-int8`` (already INT8 in their linears): each large
  embedding table (``Gather`` of a float initializer) becomes INT8 with one
  scale per row, gathered and rescaled in the graph.
- Head graphs (``attr``, ``srl_roles``, ``span``, ``dep_rel``, ``coref_spans``)
  -> ``<graph>-int8``: weight-only INT8 (``DequantizeLinear``, one scale per
  output channel) for every large weight, and INT8 embedding tables; the
  arithmetic stays fp32. (Dynamic INT8 MatMuls failed in the attribute graph
  on some inputs, and the heads are a small share of the time.)
- The label graph stays FP16: in INT8 it cost Ettin-17M 2.1 points of
  relation F1 and 2.4 of span F1.

Score with ``int8_eval --variants int8,fp16``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HEADS = ("attr", "srl_roles", "span", "dep_rel", "coref_spans")


def int8_tables(m, min_size: int) -> int:
    """Each ``Gather`` of a 2-D float initializer with at least ``min_size``
    values: INT8 rows and a per-row scale, gathered and multiplied back."""
    from onnx import TensorProto, helper, numpy_helper
    inits = {i.name: i for i in m.graph.initializer}
    nodes, n = [], 0
    for node in m.graph.node:
        w = inits.get(node.input[0]) if node.op_type == "Gather" else None
        if (w is None or len(w.dims) != 2 or w.data_type != TensorProto.FLOAT
                or np.prod(w.dims) < min_size
                or any(a.name == "axis" and helper.get_attribute_value(a) != 0 for a in node.attribute)):
            nodes.append(node)
            continue
        W = numpy_helper.to_array(w)
        scale = (np.maximum(np.abs(W).max(1, keepdims=True), 1e-8) / 127.0).astype(np.float32)
        q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
        qn, sn, o = w.name + "_q8", w.name + "_s8", node.output[0]
        m.graph.initializer.remove(w)
        m.graph.initializer.extend([numpy_helper.from_array(q, qn), numpy_helper.from_array(scale, sn)])
        nodes += [helper.make_node("Gather", [qn, node.input[1]], [o + "_gq"], axis=0),
                  helper.make_node("Gather", [sn, node.input[1]], [o + "_gs"], axis=0),
                  helper.make_node("Cast", [o + "_gq"], [o + "_gf"], to=TensorProto.FLOAT),
                  helper.make_node("Mul", [o + "_gf", o + "_gs"], [o])]
        n += 1
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    return n


def int8_weights(m, min_size: int = 4096) -> int:
    """Weight-only INT8 for every float initializer of 2+ dimensions not read
    by a ``Gather``: one scale per output channel (the last axis of a
    MatMul's weight, else the first), ``DequantizeLinear`` in the graph."""
    from onnx import TensorProto, helper, numpy_helper
    use: dict[str, list[tuple[str, int]]] = {}
    for node in m.graph.node:
        for j, i in enumerate(node.input):
            use.setdefault(i, []).append((node.op_type, j))
    n = 0
    for w in list(m.graph.initializer):
        ops = use.get(w.name, [])
        if (w.data_type != TensorProto.FLOAT or len(w.dims) < 2 or np.prod(w.dims) < min_size
                or not ops or any(op == "Gather" for op, _ in ops)):
            continue
        W = numpy_helper.to_array(w)
        axis = W.ndim - 1 if all(op == "MatMul" and j == 1 for op, j in ops) else 0
        scale = (np.maximum(np.abs(W).max(tuple(a for a in range(W.ndim) if a != axis)), 1e-8)
                 / 127.0).astype(np.float32)
        shape = [1] * W.ndim
        shape[axis] = -1
        q = np.clip(np.round(W / scale.reshape(shape)), -127, 127).astype(np.int8)
        m.graph.initializer.remove(w)
        m.graph.initializer.extend([numpy_helper.from_array(q, w.name + "_w8"),
                                    numpy_helper.from_array(scale, w.name + "_w8s"),
                                    numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8),
                                                            w.name + "_w8z")])
        m.graph.node.insert(0, helper.make_node(
            "DequantizeLinear", [w.name + "_w8", w.name + "_w8s", w.name + "_w8z"], [w.name], axis=axis))
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", type=Path, required=True)
    args = ap.parse_args()
    import onnx
    import onnxruntime as ort
    d = args.release / "onnx"

    def save(m, path):
        onnx.save(m, str(path))
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])     # loads

    for g in ("plain", "marker"):
        p = d / f"{g}-int8.onnx"
        before = p.stat().st_size
        m = onnx.load(str(p))
        n = int8_tables(m, 100_000)
        save(m, p)
        print(f"{p.name}: {n} embedding tables INT8, {before / 1e6:.1f} -> {p.stat().st_size / 1e6:.1f} MB",
              flush=True)
    for g in HEADS:
        src, out = d / f"{g}.onnx", d / f"{g}-int8.onnx"
        m = onnx.load(str(src))
        n_t, n_w = int8_tables(m, 10_000), int8_weights(m)
        save(m, out)
        print(f"{out.name}: {n_w} weights and {n_t} tables weight-only INT8, "
              f"{src.stat().st_size / 1e6:.1f} -> {out.stat().st_size / 1e6:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

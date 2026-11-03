"""Set the coreference antecedent shortlist of a release's plain graphs in place.

    uv run --extra export python -m v6.export.set_coref_k --release release/ettin-17m --k 100

K is three constants in ``plain*.onnx``: the top-k itself and the shapes of
the NEG and zero pads ``Plain.antecedents`` adds so top-k never needs
K <= W. Nothing else depends on it, so the fp16 and INT8 graphs keep their
weights and quantisation exactly. Writes ``coref_k`` into ``onnx_meta.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def set_k(path: Path, old: int, new: int) -> None:
    m = onnx.load(str(path), load_external_data=False)
    g = m.graph
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    topk = [n for n in g.node if n.op_type == "TopK"]
    assert len(topk) == 1, f"{path.name}: {len(topk)} TopK nodes"

    def is_k(n) -> bool:
        if n.op_type != "Constant":
            return False
        v = numpy_helper.to_array(n.attribute[0].t)
        return v.dtype == np.int64 and v.shape == (1,) and int(v[0]) == old

    hits = []
    for n in g.node:
        if not is_k(n):
            continue
        for c in consumers.get(n.output[0], []):
            if c is topk[0] or (c.op_type == "Concat" and any(
                    d.op_type == "ConstantOfShape" for d in consumers.get(c.output[0], []))):
                hits.append(n)
                break
    assert len(hits) == 3, f"{path.name}: {len(hits)} shortlist constants, expected 3"
    for n in hits:
        n.attribute[0].t.CopyFrom(numpy_helper.from_array(np.array([new], dtype=np.int64)))
    onnx.save(m, str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--k", type=int, required=True)
    args = ap.parse_args()
    onnx_dir = args.release / "onnx"
    meta_path = onnx_dir / "onnx_meta.json"
    meta = json.loads(meta_path.read_text())
    old = int(meta["coref_k"])
    for p in sorted(onnx_dir.glob("plain*.onnx")):
        set_k(p, old, args.k)
        print(f"{p}: coreference shortlist {old} -> {args.k}")
    meta["coref_k"] = args.k
    meta_path.write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

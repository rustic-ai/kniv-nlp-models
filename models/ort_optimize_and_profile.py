"""One-off: run ORT transformers optimizer on cascade.onnx, then profile.

Read-only with respect to the source `cascade.onnx`. Writes a sibling
`cascade-opt.onnx` (and `cascade-opt-fp16.onnx` if `--fp16` is passed) and
two `*.profile.json` Chrome traces (open in chrome://tracing).

Usage:
    uv run models/ort_optimize_and_profile.py \\
        --model-dir models/kniv-deberta-nlp-base-en-base \\
        [--fp16] [--seq-len 128] [--batch 1] [--warmup 5] [--iters 50]

Prints:
    - Fused operator statistics from the optimizer
    - Per-EP latency (mean / p50 / p95) for original and optimized models
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# Force line-buffered stdout so log tails reflect progress live.
sys.stdout.reconfigure(line_buffering=True)

print("[boot] importing numpy/onnxruntime...", flush=True)
import numpy as np
import onnxruntime as ort
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
print(f"[boot] onnxruntime {ort.__version__}", flush=True)


SIZE_TO_HEADS_HIDDEN = {
    "xsmall": (6, 384),
    "small":  (12, 768),
    "base":   (12, 768),
    "large":  (16, 1024),
}


def infer_size_from_dir(model_dir: Path) -> str:
    for size in SIZE_TO_HEADS_HIDDEN:
        if model_dir.name.endswith(f"-{size}"):
            return size
    raise ValueError(
        f"Cannot infer size from {model_dir.name}; expected suffix one of {list(SIZE_TO_HEADS_HIDDEN)}"
    )


def run_optimizer(src: Path, dst: Path, num_heads: int, hidden: int) -> dict:
    opts = FusionOptions("bert")
    opts.enable_attention = True
    opts.enable_skip_layer_norm = True
    opts.enable_embed_layer_norm = True
    opts.enable_bias_gelu = True
    opts.enable_gelu = True
    opts.enable_layer_norm = True

    opt_model = optimizer.optimize_model(
        str(src),
        model_type="bert",
        num_heads=num_heads,
        hidden_size=hidden,
        optimization_options=opts,
    )
    opt_model.save_model_to_file(str(dst))
    return opt_model.get_fused_operator_statistics()


def build_feeds(session: ort.InferenceSession, batch: int, seq_len: int) -> dict:
    feeds = {}
    for inp in session.get_inputs():
        shape = []
        for i, d in enumerate(inp.shape):
            if isinstance(d, int):
                shape.append(d)
            elif i == 0:
                shape.append(batch)
            else:
                shape.append(seq_len)
        if inp.type == "tensor(int64)":
            if "mask" in inp.name:
                arr = np.ones(shape, dtype=np.int64)
            elif "predicate" in inp.name:
                arr = np.zeros(shape, dtype=np.int64)
            else:
                arr = np.random.randint(0, 30000, size=shape, dtype=np.int64)
        elif inp.type == "tensor(float)":
            arr = np.random.randn(*shape).astype(np.float32)
        else:
            raise RuntimeError(f"Unhandled input type {inp.type} for {inp.name}")
        feeds[inp.name] = arr
    return feeds


def bench(model_path: Path, providers: list[str], batch: int, seq_len: int,
          warmup: int, iters: int, enable_profiling: bool = False) -> dict:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if enable_profiling:
        so.enable_profiling = True

    sess = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
    actual = sess.get_providers()[0]
    feeds = build_feeds(sess, batch, seq_len)

    for _ in range(warmup):
        sess.run(None, feeds)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1000.0)

    result = {
        "ep": actual,
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "p95_ms": sorted(times)[int(len(times) * 0.95) - 1],
    }
    if enable_profiling:
        result["profile_file"] = sess.end_profiling()
    return result


def fmt_row(label: str, r: dict) -> str:
    return (f"  {label:<28} ep={r['ep']:<25} "
            f"mean={r['mean_ms']:7.2f}ms  p50={r['p50_ms']:7.2f}ms  p95={r['p95_ms']:7.2f}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True,
                    help="Path to a models/kniv-deberta-nlp-base-en-{size}/ directory")
    ap.add_argument("--fp16", action="store_true",
                    help="Also produce cascade-opt-fp16.onnx and bench it on CUDA")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--profile", action="store_true",
                    help="Emit Chrome trace JSON for each run (much slower)")
    args = ap.parse_args()

    onnx_dir = args.model_dir / "onnx"
    src = onnx_dir / "cascade.onnx"
    dst = onnx_dir / "cascade-opt.onnx"
    dst_fp16 = onnx_dir / "cascade-opt-fp16.onnx"
    if not src.exists():
        raise FileNotFoundError(src)

    size = infer_size_from_dir(args.model_dir)
    num_heads, hidden = SIZE_TO_HEADS_HIDDEN[size]
    print(f"[optimize] {src.name} -> {dst.name}  (size={size}, heads={num_heads}, hidden={hidden})")

    stats = run_optimizer(src, dst, num_heads, hidden)
    print("[optimize] fused operator statistics:")
    print(json.dumps(stats, indent=2))

    fused_attn = stats.get("Attention") or stats.get("MultiHeadAttention") or 0
    if fused_attn == 0:
        print("[optimize] WARNING: attention did NOT fuse. "
              "Try re-exporting at opset 17/19, or fall back to TensorRT EP on CUDA.")
    else:
        print(f"[optimize] attention fused into {fused_attn} ops — good")

    if args.fp16:
        try:
            opt_model_fp16 = optimizer.optimize_model(
                str(src), model_type="bert",
                num_heads=num_heads, hidden_size=hidden,
            )
            opt_model_fp16.convert_float_to_float16(keep_io_types=True)
            opt_model_fp16.save_model_to_file(str(dst_fp16))
            print(f"[optimize] wrote {dst_fp16.name}")
        except Exception as e:
            print(f"[optimize] FP16 conversion failed: {e}")

    avail = ort.get_available_providers()
    print(f"\n[bench] available providers: {avail}")
    print(f"[bench] batch={args.batch} seq_len={args.seq_len} warmup={args.warmup} iters={args.iters}\n")

    targets: list[tuple[str, Path, list[str]]] = []
    targets.append(("original (CPU)", src, ["CPUExecutionProvider"]))
    targets.append(("optimized (CPU)", dst, ["CPUExecutionProvider"]))
    if "CUDAExecutionProvider" in avail:
        cuda = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        targets.append(("original (CUDA)", src, cuda))
        targets.append(("optimized (CUDA)", dst, cuda))
        if args.fp16 and dst_fp16.exists():
            targets.append(("optimized-fp16 (CUDA)", dst_fp16, cuda))

    print("[bench] results:")
    for label, path, providers in targets:
        try:
            r = bench(path, providers, args.batch, args.seq_len,
                      args.warmup, args.iters, enable_profiling=args.profile)
            print(fmt_row(label, r))
            if "profile_file" in r:
                print(f"      trace: {r['profile_file']}")
        except Exception as e:
            print(f"  {label:<28} FAILED: {e}")


if __name__ == "__main__":
    main()

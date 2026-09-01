"""Performance (latency / throughput) benchmark for kniv distilled students.

Measures inference latency across runtime backends:
  - PyTorch CPU
  - PyTorch CUDA (if available)
  - ONNX Runtime CPU (FP32 + INT8)
  - ONNX Runtime CUDA (FP32 + INT8, if onnxruntime-gpu installed and GPU available)

Sweeps batch size and sequence length, reports mean / p50 / p95 / p99 latency
and throughput (sentences/sec).

Usage:
    # All available backends, all sizes
    python models/student_benchmark_perf.py --model-dir models/kniv-deberta-nlp-base-en-xsmall

    # Custom sweep
    python models/student_benchmark_perf.py --model-dir ... \
        --batch-sizes 1 8 32 --seq-lens 64 128

    # Only specific runtimes
    python models/student_benchmark_perf.py --model-dir ... \
        --runtimes pytorch_cuda onnx_int8_cuda
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from student_loader import load_student


def percentile_ms(times_sec, p):
    return float(np.percentile(times_sec, p) * 1000)


def warm_and_time(fn, n_warmup=5, n_iters=30, sync_cuda=False):
    """Run fn() n_warmup+n_iters times, return list of n_iters timings (seconds)."""
    for _ in range(n_warmup):
        fn()
    if sync_cuda:
        torch.cuda.synchronize()
    timings = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        if sync_cuda:
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)
    return timings


def benchmark_pytorch(model, tokenizer, batch_size, seq_len, device, n_iters):
    text = ["The quick brown fox jumps over the lazy dog ."] * batch_size
    enc = tokenizer(text, return_tensors="pt", padding="max_length",
                    max_length=seq_len, truncation=True)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    pred = torch.zeros(batch_size, dtype=torch.long, device=device)

    def run():
        with torch.no_grad():
            _ = model(ids, mask, pred)

    sync = device.type == "cuda"
    timings = warm_and_time(run, n_warmup=5, n_iters=n_iters, sync_cuda=sync)
    return {
        "mean_ms": float(np.mean(timings) * 1000),
        "p50_ms": percentile_ms(timings, 50),
        "p95_ms": percentile_ms(timings, 95),
        "p99_ms": percentile_ms(timings, 99),
        "throughput_sps": batch_size / float(np.mean(timings)),
    }


def benchmark_onnx(session, tokenizer, batch_size, seq_len, n_iters, is_gpu=False):
    text = ["The quick brown fox jumps over the lazy dog ."] * batch_size
    enc = tokenizer(text, return_tensors="np", padding="max_length",
                    max_length=seq_len, truncation=True)
    ids = enc["input_ids"].astype(np.int64)
    mask = enc["attention_mask"].astype(np.int64)
    pred = np.zeros(batch_size, dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": mask, "predicate_idx": pred}

    def run():
        session.run(None, feed)

    # ORT CUDA execution is async — sync via torch.cuda for fair timing
    sync = is_gpu and torch.cuda.is_available()
    timings = warm_and_time(run, n_warmup=5, n_iters=n_iters, sync_cuda=sync)
    return {
        "mean_ms": float(np.mean(timings) * 1000),
        "p50_ms": percentile_ms(timings, 50),
        "p95_ms": percentile_ms(timings, 95),
        "p99_ms": percentile_ms(timings, 99),
        "throughput_sps": batch_size / float(np.mean(timings)),
    }


def discover_runtimes(args):
    """Decide which runtimes are available based on environment + flags.

    Returns list of (runtime_id, runtime_kind, providers_or_device) tuples.
    runtime_kind ∈ {"pytorch", "onnx_fp32", "onnx_int8"}.
    """
    runtimes = []

    # PyTorch CPU
    runtimes.append(("pytorch_cpu", "pytorch", "cpu"))
    # PyTorch CUDA
    if torch.cuda.is_available():
        runtimes.append(("pytorch_cuda", "pytorch", "cuda"))

    # ONNX Runtime
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        # CPU FP32 / INT8
        if "CPUExecutionProvider" in avail:
            runtimes.append(("onnx_fp32_cpu", "onnx_fp32", ["CPUExecutionProvider"]))
            runtimes.append(("onnx_int8_cpu", "onnx_int8", ["CPUExecutionProvider"]))
        # CUDA FP32 / INT8
        if "CUDAExecutionProvider" in avail:
            cuda_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            runtimes.append(("onnx_fp32_cuda", "onnx_fp32", cuda_providers))
            runtimes.append(("onnx_int8_cuda", "onnx_int8", cuda_providers))
    except ImportError:
        print("WARNING: onnxruntime not installed; skipping ONNX backends")

    # Filter to user request
    if args.runtimes:
        runtimes = [r for r in runtimes if r[0] in args.runtimes]
    return runtimes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128])
    parser.add_argument("--n-iters", type=int, default=30)
    parser.add_argument("--runtimes", nargs="+", default=None,
                        help="Subset of runtimes to benchmark. Choices: "
                             "pytorch_cpu, pytorch_cuda, onnx_fp32_cpu, onnx_int8_cpu, "
                             "onnx_fp32_cuda, onnx_int8_cuda. Default: all available.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    onnx_dir = model_dir / "onnx"
    fp32_path = onnx_dir / "cascade.onnx"
    int8_path = onnx_dir / "cascade-int8.onnx"

    print("=" * 70)
    print(f"PERFORMANCE BENCHMARK — {model_dir.name}")
    print("=" * 70)
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  CUDA: not available")

    runtimes = discover_runtimes(args)
    print(f"  Runtimes to test: {', '.join(r[0] for r in runtimes)}\n")

    results = {
        "model_dir": str(model_dir),
        "n_iters": args.n_iters,
        "batch_sizes": args.batch_sizes,
        "seq_lens": args.seq_lens,
        "runtimes": {},
    }

    # ── Load tokenizer once (shared) ──
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    for runtime_id, kind, target in runtimes:
        print(f"\n--- {runtime_id} ---")

        if kind == "pytorch":
            device = torch.device(target)
            model, _, info = load_student(model_dir, checkpoint=args.checkpoint,
                                           device=device)
            rt_results = []
            for bs in args.batch_sizes:
                for sl in args.seq_lens:
                    r = benchmark_pytorch(model, tokenizer, bs, sl, device, args.n_iters)
                    r.update({"batch_size": bs, "seq_len": sl})
                    rt_results.append(r)
                    print(f"  bs={bs:3d} seq={sl:3d}  "
                          f"mean={r['mean_ms']:7.2f}ms  p50={r['p50_ms']:7.2f}ms  "
                          f"p95={r['p95_ms']:7.2f}ms  thr={r['throughput_sps']:8.1f} sent/s")
            results["runtimes"][runtime_id] = rt_results
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        else:  # onnx_fp32 or onnx_int8
            import onnxruntime as ort
            path = fp32_path if kind == "onnx_fp32" else int8_path
            if not path.exists():
                print(f"  Skipping: {path} does not exist (run student_export_onnx.py)")
                continue
            print(f"  File: {path} ({path.stat().st_size / 1e6:.0f} MB)")
            print(f"  Providers: {target}")
            session = ort.InferenceSession(str(path), providers=target)
            actual_provider = session.get_providers()[0]
            print(f"  Active provider: {actual_provider}")
            is_gpu = "CUDA" in actual_provider or "Tensorrt" in actual_provider
            rt_results = []
            for bs in args.batch_sizes:
                for sl in args.seq_lens:
                    r = benchmark_onnx(session, tokenizer, bs, sl, args.n_iters, is_gpu=is_gpu)
                    r.update({"batch_size": bs, "seq_len": sl})
                    rt_results.append(r)
                    print(f"  bs={bs:3d} seq={sl:3d}  "
                          f"mean={r['mean_ms']:7.2f}ms  p50={r['p50_ms']:7.2f}ms  "
                          f"p95={r['p95_ms']:7.2f}ms  thr={r['throughput_sps']:8.1f} sent/s")
            results["runtimes"][runtime_id] = rt_results
            del session

    # ── Summary tables ──
    if not results["runtimes"]:
        print("\nNo runtimes successfully benchmarked.")
        return

    runtime_names = list(results["runtimes"].keys())
    col_w = 12

    # Latency at bs=1 across seq_lens
    print(f"\n{'=' * 80}")
    print(f"  LATENCY (mean ms per call, batch_size=1)")
    print(f"{'=' * 80}")
    header = f"  {'Runtime':<28}"
    for sl in args.seq_lens:
        header += f"{'seq=' + str(sl):>{col_w}}"
    print(header)
    print(f"  {'─'*28}" + "".join(f" {'─'*(col_w-1)}" for _ in args.seq_lens))
    for rt_name in runtime_names:
        row = f"  {rt_name:<28}"
        for sl in args.seq_lens:
            match = next((r for r in results["runtimes"][rt_name]
                          if r["batch_size"] == 1 and r["seq_len"] == sl), None)
            row += f"{match['mean_ms']:>{col_w}.2f}" if match else f"{'—':>{col_w}}"
        print(row)

    # Throughput at largest batch
    largest_bs = max(args.batch_sizes)
    print(f"\n{'=' * 80}")
    print(f"  THROUGHPUT (sentences/sec, batch_size={largest_bs})")
    print(f"{'=' * 80}")
    header = f"  {'Runtime':<28}"
    for sl in args.seq_lens:
        header += f"{'seq=' + str(sl):>{col_w}}"
    print(header)
    print(f"  {'─'*28}" + "".join(f" {'─'*(col_w-1)}" for _ in args.seq_lens))
    for rt_name in runtime_names:
        row = f"  {rt_name:<28}"
        for sl in args.seq_lens:
            match = next((r for r in results["runtimes"][rt_name]
                          if r["batch_size"] == largest_bs and r["seq_len"] == sl), None)
            row += f"{match['throughput_sps']:>{col_w}.1f}" if match else f"{'—':>{col_w}}"
        print(row)

    # Speedup table relative to pytorch_cpu (most common comparison baseline)
    if "pytorch_cpu" in runtime_names:
        baseline_id = "pytorch_cpu"
    else:
        baseline_id = runtime_names[0]
    largest_seq = max(args.seq_lens)
    print(f"\n{'=' * 80}")
    print(f"  SPEEDUP vs {baseline_id} at bs=1, seq={largest_seq}")
    print(f"{'=' * 80}")
    base = next((r for r in results["runtimes"][baseline_id]
                 if r["batch_size"] == 1 and r["seq_len"] == largest_seq), None)
    if base is not None:
        base_ms = base["mean_ms"]
        print(f"  {'Runtime':<28} {'mean ms':>10} {'speedup':>10}")
        print(f"  {'─'*28} {'─'*10} {'─'*10}")
        for rt_name in runtime_names:
            match = next((r for r in results["runtimes"][rt_name]
                          if r["batch_size"] == 1 and r["seq_len"] == largest_seq), None)
            if match:
                speedup = base_ms / match["mean_ms"]
                marker = " ★" if rt_name == baseline_id else ""
                print(f"  {rt_name:<28} {match['mean_ms']:>10.2f} {speedup:>10.2f}x{marker}")

    # ── Save ──
    out_path = model_dir / "perf_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'=' * 80}")
    print(f"Saved performance results: {out_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

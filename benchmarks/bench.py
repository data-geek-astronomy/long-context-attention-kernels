"""Benchmark sliding-window+global attention vs. a naive dense-masked
baseline across sequence lengths, to demonstrate near-linear vs quadratic
scaling.

Usage:
    python benchmarks/bench.py --seq-len 4096 8192 16384 --window 256
"""

import argparse
import math
import sys
import os
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from long_context_attention import sliding_window_attention, _reference_sliding_window_attention


def bench(fn, *args, warmup=3, iters=10, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0  # ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"device={device}  dtype={dtype}  window={args.window}")
    print(f"{'seq_len':>8} | {'sparse (ms)':>12} | {'dense-masked (ms)':>18} | {'speedup':>8}")

    for seq_len in args.seq_len:
        Q = torch.randn(args.batch, args.heads, seq_len, args.head_dim, dtype=dtype, device=device)
        K = torch.randn(args.batch, args.heads, seq_len, args.head_dim, dtype=dtype, device=device)
        V = torch.randn(args.batch, args.heads, seq_len, args.head_dim, dtype=dtype, device=device)
        scale = 1.0 / math.sqrt(args.head_dim)
        global_tokens = [0, 1, 2]

        try:
            t_sparse = bench(sliding_window_attention, Q, K, V, args.window, global_tokens, scale)
        except Exception as e:  # pragma: no cover
            t_sparse = float("nan")
            print(f"  [sparse kernel unavailable: {e}]")

        t_dense = bench(_reference_sliding_window_attention, Q, K, V, args.window, global_tokens, scale)

        speedup = t_dense / t_sparse if t_sparse == t_sparse and t_sparse > 0 else float("nan")
        print(f"{seq_len:>8} | {t_sparse:>12.3f} | {t_dense:>18.3f} | {speedup:>7.2f}x")


if __name__ == "__main__":
    main()

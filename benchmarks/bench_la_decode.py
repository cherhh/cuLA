#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License")

"""
bench_la_decode.py — Benchmark: TMA vs cp.async for linear_attention_decode
                      (state-based linear attention decode, single token per request)

Compares kernel latency (ms) for the two implementations:
  - la_cpasync : original cp.async-based state load (la_decode.py)
  - la_tma     : TMA bulk-async state load           (la_decode_tma.py)

Usage:
  python benchmarks/bench_la_decode.py            # default configs
  python benchmarks/bench_la_decode.py --ncu      # single iteration for ncu profiling
  ncu -k regex:'cutlass' -o la_decode.ncu-rep python benchmarks/bench_la_decode.py --ncu
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from cula.lightning.la_decode import linear_attention_decode as la_cpasync
from cula.lightning.la_decode_tma import linear_attention_decode_tma as la_tma

# ============================================================
# Constants
# ============================================================
K_DIM = 128
WARMUP = 20
N_ITERS = 200


def time_kernel(fn, warmup=WARMUP, n_iters=N_ITERS):
    """CUDA-event based timing. Returns ms/call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n_iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / n_iters


def make_tensors(B, H, K, device="cuda"):
    torch.manual_seed(42)
    q = torch.randn(B, H, K, device=device, dtype=torch.float32)
    k = torch.randn(B, H, K, device=device, dtype=torch.float32)
    v = torch.randn(B, H, K, device=device, dtype=torch.float32)
    # state: (B*H, V, K) = (B*H, K, K) since V=K
    s = torch.randn(B * H, K, K, device=device, dtype=torch.float32) * 0.01
    decay = torch.rand(H, device=device, dtype=torch.float32) * 0.1 + 0.9
    offsets = torch.arange(B * H, device=device, dtype=torch.int32)
    out = torch.zeros(B, H, K, device=device, dtype=torch.bfloat16)
    return q, k, v, s, decay, offsets, out


def bench(B, H, K, ncu_mode=False):
    warmup = 1 if ncu_mode else WARMUP
    n_iters = 1 if ncu_mode else N_ITERS
    SCALE = K ** -0.5

    q, k, v, s_base, decay, offsets, _ = make_tensors(B, H, K)

    # ── warm up / compile ──────────────────────────────────────────────────
    s_w = s_base.clone()
    o_w = torch.zeros(B, H, K, device="cuda", dtype=torch.bfloat16)
    la_cpasync(q, k, v, s_w, o_w, SCALE, 0, 0, 0, 0, 0, offsets, decay, K, K, K)
    torch.cuda.synchronize()

    s_w = s_base.clone()
    o_w = torch.zeros(B, H, K, device="cuda", dtype=torch.bfloat16)
    la_tma(q, k, v, s_w, o_w, SCALE, 0, 0, 0, 0, 0, offsets, decay, K, K, K)
    torch.cuda.synchronize()

    # ── correctness (one-shot) ────────────────────────────────────────────
    s_ref = s_base.clone()
    o_ref = torch.zeros(B, H, K, device="cuda", dtype=torch.bfloat16)
    la_cpasync(q, k, v, s_ref, o_ref, SCALE, 0, 0, 0, 0, 0, offsets, decay, K, K, K)
    torch.cuda.synchronize()

    s_tma = s_base.clone()
    o_tma = torch.zeros(B, H, K, device="cuda", dtype=torch.bfloat16)
    la_tma(q, k, v, s_tma, o_tma, SCALE, 0, 0, 0, 0, 0, offsets, decay, K, K, K)
    torch.cuda.synchronize()

    diff = (o_ref.float() - o_tma.float()).abs().max().item()
    correct = "✓" if diff < 0.01 else f"✗(Δ={diff:.4f})"

    # ── timing ─────────────────────────────────────────────────────────────
    # Note: state is modified in-place by both kernels (writeback).
    # We clone fresh state each call to avoid measuring decay accumulation.
    # Pre-allocate output to avoid allocation overhead in the timed section.
    s_bench_ref = s_base.clone()
    o_bench = torch.zeros(B, H, K, device="cuda", dtype=torch.bfloat16)

    def run_cpasync():
        nonlocal s_bench_ref
        la_cpasync(q, k, v, s_bench_ref, o_bench, SCALE, 0, 0, 0, 0, 0, offsets, decay, K, K, K)

    def run_tma():
        nonlocal s_bench_ref
        la_tma(q, k, v, s_bench_ref, o_bench, SCALE, 0, 0, 0, 0, 0, offsets, decay, K, K, K)

    t_ref = time_kernel(run_cpasync, warmup=warmup, n_iters=n_iters)
    t_tma = time_kernel(run_tma,     warmup=warmup, n_iters=n_iters)
    speedup = t_ref / t_tma

    return t_ref, t_tma, speedup, correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncu", action="store_true", help="NCU profiling mode (warmup=1, iters=1)")
    args = parser.parse_args()

    torch.cuda.init()

    # Configs: (B, H)  with K fixed at 128
    configs = [
        (1,  16),
        (4,  16),
        (8,  16),
        (16, 16),
        (32, 16),
        (64, 16),
        (128, 16),
        (256, 16),
    ]

    K = K_DIM

    print()
    print("=" * 70)
    print(f"  la_decode benchmark: TMA vs cp.async  (K={K}, V={K}, fp32 state)")
    print("=" * 70)
    print(f"  {'B':>5}  {'H':>4}  {'BxH':>6}  {'cp.async(ms)':>14}  {'TMA(ms)':>10}  {'Speedup':>9}  {'Correct'}")
    print("-" * 70)

    for B, H in configs:
        t_ref, t_tma, speedup, correct = bench(B, H, K, ncu_mode=args.ncu)
        print(
            f"  {B:>5}  {H:>4}  {B*H:>6}  "
            f"{t_ref:>14.4f}  {t_tma:>10.4f}  {speedup:>8.2f}x  {correct}"
        )

    print()


if __name__ == "__main__":
    main()

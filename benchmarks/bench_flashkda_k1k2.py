#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark FlashKDA prefill (K1 + K2): CuTeDSL vs CUTLASS C++.

Compares the full prefill path (K1 prepare + K2 recurrence) head-to-head:

  - cute:  ``cula.ops.flashkda_prefill.flash_kda_prefill`` with
           ``CULA_FLASHKDA_USE_CUTE=1`` (currently runs ``launch_k1_full`` +
           ``launch_k2_phaseB``).
  - cpp :  ``flash_kda_C.fwd`` (the CUTLASS C++ baseline shipped with
           MoonshotAI/FlashKDA).

Settings mirror ``benchmarks/bench_kda.py`` (H=64, D=128, bf16, safe gate
inputs). Only fixed-length / no-state shapes are exercised because the
CuTeDSL path does not yet support varlen / state I/O.

Usage:
    CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_flashkda_k1k2.py
    CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_flashkda_k1k2.py --ncu
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "FlashKDA"))

# Enable the CuTeDSL prefill path BEFORE importing flashkda_prefill so the
# module-level ``_USE_CUTE`` flag is set correctly.
os.environ["CULA_FLASHKDA_USE_CUTE"] = "1"

import torch  # noqa: E402

from benchmarks.utils import SEED, set_seed  # noqa: E402
from cula.ops.flashkda_prefill import flash_kda_prefill  # noqa: E402

try:
    import flash_kda_C as flash_kda_cpp  # noqa: F401

    HAS_CPP = True
except Exception as e:  # pragma: no cover
    print(f"[warn] flash_kda_C not importable: {e!r}", file=sys.stderr)
    HAS_CPP = False


# ============================================================
# Constants (mirror bench_kda.py)
# ============================================================
H, D = 64, 128
WARMUP = 10
N_ITERS = 30
NCU_MODE = False


# ============================================================
# Helpers
# ============================================================
def time_kernel(fn, warmup=None, n_iters=None):
    if warmup is None:
        warmup = 1 if NCU_MODE else WARMUP
    if n_iters is None:
        n_iters = 1 if NCU_MODE else N_ITERS
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters  # ms


def accuracy_stats(ref, out):
    ref_f = ref.float()
    out_f = out.float()
    diff = (ref_f - out_f).abs()
    rmse = diff.pow(2).mean().sqrt().item()
    max_diff = diff.max().item()
    denom = ref_f.abs().max().item()
    rel_max = max_diff / denom if denom > 0 else 0.0
    return rmse, rel_max


def make_inputs(B, T, device="cuda"):
    """Build (B,T,H,D) bf16 inputs matching the C++ ``fwd`` signature.

    Both ``flash_kda_prefill`` and ``flash_kda_C.fwd`` expect 4-D
    ``q/k/v/g`` of shape ``(B, T, H, D)``, ``beta`` of shape ``(B, T, H)``
    in bf16, and ``A_log`` ``(H,)`` / ``dt_bias`` ``(H, D)`` in fp32.
    """
    set_seed(SEED)
    dtype = torch.bfloat16
    q = torch.randn(B, T, H, D, dtype=dtype, device=device) * 0.5
    k = torch.randn(B, T, H, D, dtype=dtype, device=device) * 0.5
    v = torch.randn(B, T, H, D, dtype=dtype, device=device) * 0.5
    g = torch.randn(B, T, H, D, dtype=dtype, device=device) * 0.1
    beta = torch.randn(B, T, H, dtype=dtype, device=device) * 0.1
    A_log = torch.randn(H, dtype=torch.float, device=device) * 0.1
    dt_bias = torch.randn(H, D, dtype=torch.float, device=device) * 0.1
    return q, k, v, g, beta, A_log, dt_bias


def cpp_workspace_for(B, T, H_, device):
    """Allocate the bytes-buffer required by ``flash_kda_C.fwd``."""
    n_bytes = flash_kda_cpp.get_workspace_size(B, T, H_)
    return torch.empty(n_bytes, dtype=torch.uint8, device=device)


# ============================================================
# Fixed-length benchmark
# ============================================================
def bench_fixed(configs):
    print("\n" + "=" * 100)
    print(" Fixed-Length Benchmark: CuTeDSL (K1+K2) vs CUTLASS C++ (flash_kda_C.fwd)")
    print("=" * 100)
    results = []
    scale = D**-0.5
    lower_bound = -5.0
    device = torch.device("cuda")

    for B, T in configs:
        torch.cuda.empty_cache()
        q, k, v, g, beta, A_log, dt_bias = make_inputs(B, T, device)
        out_cute = torch.empty_like(v)
        out_cpp = torch.empty_like(v)

        # --- CuTeDSL path ---
        try:
            flash_kda_prefill(q, k, v, g, beta, scale, out_cute, A_log, dt_bias, lower_bound)
            torch.cuda.synchronize()
            cute_ok = True
        except Exception as e:
            print(f"  [warn] cute failed for B={B} T={T}: {e!r}", file=sys.stderr)
            cute_ok = False

        # --- C++ path ---
        if HAS_CPP:
            ws = cpp_workspace_for(B, T, H, device)
            flash_kda_cpp.fwd(
                q,
                k,
                v,
                g,
                beta,
                scale,
                out_cpp,
                ws,
                A_log,
                dt_bias,
                lower_bound,
                None,
                None,
                None,
            )
            torch.cuda.synchronize()
            cpp_ok = True
        else:
            cpp_ok = False

        # Accuracy
        if cute_ok and cpp_ok:
            rmse, rel_max = accuracy_stats(out_cpp, out_cute)
        else:
            rmse, rel_max = float("nan"), float("nan")

        # Timing
        ms_cute = (
            time_kernel(lambda: flash_kda_prefill(q, k, v, g, beta, scale, out_cute, A_log, dt_bias, lower_bound))
            if cute_ok
            else float("nan")
        )

        if cpp_ok:
            ws = cpp_workspace_for(B, T, H, device)

            def _cpp():
                flash_kda_cpp.fwd(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    scale,
                    out_cpp,
                    ws,
                    A_log,
                    dt_bias,
                    lower_bound,
                    None,
                    None,
                    None,
                )

            ms_cpp = time_kernel(_cpp)
        else:
            ms_cpp = float("nan")

        speedup = (ms_cpp / ms_cute) if (ms_cute > 0 and ms_cpp == ms_cpp) else float("nan")

        results.append(
            dict(
                B=B,
                T=T,
                rmse=rmse,
                rel_max=rel_max,
                ms_cute=ms_cute,
                ms_cpp=ms_cpp,
                speedup=speedup,
            )
        )

        del q, k, v, g, beta, A_log, dt_bias, out_cute, out_cpp
        torch.cuda.empty_cache()

    return results


# ============================================================
# Report
# ============================================================
def print_report(fixed_results):
    sep = "=" * 100
    print(f"\n\n{sep}")
    print("                BENCHMARK REPORT: FlashKDA prefill (K1 + K2)")
    print("                CuTeDSL (k1_full + k2_phaseB)  vs  CUTLASS C++ (flash_kda_C.fwd)")
    print(f"                H={H}  D={D}  dtype=bf16")
    wu = 1 if NCU_MODE else WARMUP
    ni = 1 if NCU_MODE else N_ITERS
    print(f"                Warmup={wu}  Iters={ni}{'  [NCU]' if NCU_MODE else ''}")
    print(sep)

    print(f"\n  {'─' * 90}")
    print(
        f"  {'B':>3s}  {'T':>6s}  │  {'RMSE':>10s}  {'rel_max':>10s}  │  {'cute(ms)':>10s}  {'cpp(ms)':>10s}  {'speedup':>10s}"
    )
    print(f"  {'─' * 90}")
    for r in fixed_results:
        print(
            f"  {r['B']:3d}  {r['T']:6d}  │  "
            f"{r['rmse']:10.6f}  {r['rel_max']:10.6f}  │  "
            f"{r['ms_cute']:10.4f}  {r['ms_cpp']:10.4f}  "
            f"{r['speedup']:9.3f}x"
        )
    print(f"  {'─' * 90}")
    print(f"\n{sep}\n")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="bench_flashkda_k1k2: CuTeDSL vs CUTLASS C++")
    parser.add_argument("--ncu", action="store_true", help="NCU profiling mode (warmup=1, iters=1)")
    args = parser.parse_args()

    global NCU_MODE
    if args.ncu:
        NCU_MODE = True
        print("[NCU mode] warmup=1, iters=1")

    # Fixed-length configs: keep B*T moderate so the CuTeDSL build/cache time
    # does not dominate. Mirrors the small/medium end of bench_kda.py.
    fixed_configs = [
        (1, 512),
        (1, 1024),
        (1, 2048),
        (1, 4096),
        (1, 8192),
        (2, 1024),
        (2, 2048),
        (2, 4096),
    ]

    fixed_res = bench_fixed(fixed_configs)
    print_report(fixed_res)


if __name__ == "__main__":
    main()

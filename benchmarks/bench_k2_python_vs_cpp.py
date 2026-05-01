#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F821
"""Benchmark K2 kernel: Python CuteDSL (Phase A/B) vs C++ CUTLASS baseline.

This benchmark isolates K2 (recurrence) performance by:
1. Running K1 once to generate workspace tensors
2. Timing only K2 kernel execution (Phase A and Phase B separately)
3. Comparing against C++ flash_kda_C.fwd (full K1+K2 pipeline)

Usage:
    python benchmarks/bench_k2_python_vs_cpp.py
    python benchmarks/bench_k2_python_vs_cpp.py --ncu  # for profiling
    python benchmarks/bench_k2_python_vs_cpp.py --phase B  # test Phase B only
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "FlashKDA"))

import torch

from benchmarks.utils import SEED, set_seed

try:
    import flash_kda_C as flash_kda_cpp

    HAS_CPP = True
except Exception as e:
    print(f"[warn] flash_kda_C not importable: {e!r}", file=sys.stderr)
    HAS_CPP = False

# ============================================================
# Constants
# ============================================================
H, D = 64, 128
CHUNK = 16
WARMUP = 30
N_ITERS = 100
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

    times = []
    for _ in range(n_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    mean = sum(times) / len(times)
    median = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    return mean, median, p95


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
    """Build (B,T,H,D) bf16 inputs."""
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
    """Allocate workspace for C++ kernel."""
    if not HAS_CPP:
        return None
    n_bytes = flash_kda_cpp.get_workspace_size(B, T, H_)
    return torch.empty(n_bytes, dtype=torch.uint8, device=device)


# ============================================================
# K2 Benchmark
# ============================================================
def bench_k2(configs, phase="A"):
    from cula.ops.flashkda_k1 import launch_k1_full
    from cula.ops.flashkda_k2 import launch_k2_phaseA, launch_k2_phaseB

    launch_k2 = launch_k2_phaseA if phase == "A" else launch_k2_phaseB
    phase_name = f"Phase {phase}"

    print("\n" + "=" * 120)
    print(f" K2 {phase_name} Benchmark: Python CuteDSL vs C++ CUTLASS")
    print("=" * 120)
    print(f" H={H}, D={D}, CHUNK={CHUNK}, dtype=bf16")
    print(f" Warmup={WARMUP if not NCU_MODE else 1}, Iters={N_ITERS if not NCU_MODE else 1}")
    print("=" * 120)

    results = []
    scale = D**-0.5
    lower_bound = -5.0
    device = torch.device("cuda")

    for B, T in configs:
        torch.cuda.empty_cache()
        q, k, v, g, beta, A_log, dt_bias = make_inputs(B, T, device)

        # Prepare workspace (run K1 once)
        total_tiles = (B * T) // CHUNK
        beta_flat = beta.view(B * T, H).permute(1, 0).contiguous().view(-1)

        n_qk = total_tiles * H * CHUNK * D
        n_cc = total_tiles * H * CHUNK * CHUNK
        ws_qd = torch.zeros(n_qk, dtype=torch.bfloat16, device=device)
        ws_kd = torch.zeros_like(ws_qd)
        ws_kr = torch.zeros_like(ws_qd)
        ws_gt = torch.zeros(total_tiles * H * D, dtype=torch.float32, device=device)
        ws_inv = torch.zeros(n_cc, dtype=torch.bfloat16, device=device)
        ws_mqk = torch.zeros_like(ws_inv)

        # Run K1 to populate workspace
        launch_k1_full(q, k, g, A_log, dt_bias, beta_flat, scale, -1.0, ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk)
        torch.cuda.synchronize()

        # --- Python K2 ---
        out_python = torch.zeros_like(v)
        try:
            launch_k2(v, beta_flat, ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk, out_python)
            torch.cuda.synchronize()
            python_ok = True
        except Exception as e:
            print(f"  [warn] Python K2 failed for B={B} T={T}: {e!r}", file=sys.stderr)
            python_ok = False

        # --- C++ full pipeline (K1+K2) ---
        out_cpp = torch.empty_like(v)
        if HAS_CPP:
            ws_cpp = cpp_workspace_for(B, T, H, device)
            try:
                flash_kda_cpp.fwd(q, k, v, g, beta, scale, out_cpp, ws_cpp, A_log, dt_bias, lower_bound, None, None, None)
                torch.cuda.synchronize()
                cpp_ok = True
            except Exception as e:
                print(f"  [warn] C++ failed for B={B} T={T}: {e!r}", file=sys.stderr)
                cpp_ok = False
        else:
            cpp_ok = False

        # Accuracy
        if python_ok and cpp_ok:
            rmse, rel_max = accuracy_stats(out_cpp, out_python)
        else:
            rmse, rel_max = float("nan"), float("nan")

        # Timing - Python K2 only
        if python_ok:
            mean_py, median_py, p95_py = time_kernel(
                lambda: launch_k2(v, beta_flat, ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk, out_python)
            )
        else:
            mean_py = median_py = p95_py = float("nan")

        # Timing - C++ full (K1+K2)
        if cpp_ok:
            ws_cpp = cpp_workspace_for(B, T, H, device)
            mean_cpp, median_cpp, p95_cpp = time_kernel(
                lambda: flash_kda_cpp.fwd(
                    q, k, v, g, beta, scale, out_cpp, ws_cpp, A_log, dt_bias, lower_bound, None, None, None
                )
            )
        else:
            mean_cpp = median_cpp = p95_cpp = float("nan")

        speedup = (mean_cpp / mean_py) if (mean_py > 0 and not torch.isnan(torch.tensor(mean_cpp))) else float("nan")

        results.append(
            {
                "B": B,
                "T": T,
                "rmse": rmse,
                "rel_max": rel_max,
                "py_mean": mean_py,
                "py_median": median_py,
                "py_p95": p95_py,
                "cpp_mean": mean_cpp,
                "cpp_median": median_cpp,
                "cpp_p95": p95_cpp,
                "speedup": speedup,
            }
        )

        del q, k, v, g, beta, A_log, dt_bias, out_python, out_cpp
        del ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk
        torch.cuda.empty_cache()

    return results


# ============================================================
# Report
# ============================================================
def print_report(results, phase="A"):
    sep = "=" * 120
    print(f"\n\n{sep}")
    print(f"                BENCHMARK REPORT: K2 Phase {phase}")
    print("                Python CuteDSL  vs  C++ CUTLASS (full K1+K2)")
    print(f"                H={H}  D={D}  CHUNK={CHUNK}  dtype=bf16")
    wu = 1 if NCU_MODE else WARMUP
    ni = 1 if NCU_MODE else N_ITERS
    print(f"                Warmup={wu}  Iters={ni}{'  [NCU]' if NCU_MODE else ''}")
    print(sep)

    print(f"\n  {'─' * 110}")
    print(
        f"  {'B':>3s}  {'T':>6s}  │  {'RMSE':>10s}  {'rel_max':>10s}  │  "
        f"{'Py(ms)':>10s}  {'C++(ms)':>10s}  {'speedup':>10s}  │  "
        f"{'Py_p95':>10s}  {'C++_p95':>10s}"
    )
    print(f"  {'─' * 110}")

    for r in results:
        print(
            f"  {r['B']:3d}  {r['T']:6d}  │  "
            f"{r['rmse']:10.6f}  {r['rel_max']:10.6f}  │  "
            f"{r['py_mean']:10.4f}  {r['cpp_mean']:10.4f}  {r['speedup']:9.3f}x  │  "
            f"{r['py_p95']:10.4f}  {r['cpp_p95']:10.4f}"
        )

    print(f"  {'─' * 110}")

    # Summary statistics
    valid_speedups = [r["speedup"] for r in results if not torch.isnan(torch.tensor(r["speedup"]))]
    if valid_speedups:
        avg_speedup = sum(valid_speedups) / len(valid_speedups)
        print(f"\n  Average speedup (C++ full / Python K2): {avg_speedup:.3f}x")
        print("  Note: C++ time includes K1+K2, Python time is K2 only")

    print(f"\n{sep}\n")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="bench_k2_python_vs_cpp: K2 Phase A/B vs C++ baseline")
    parser.add_argument("--ncu", action="store_true", help="NCU profiling mode (warmup=1, iters=1)")
    parser.add_argument(
        "--phase", choices=["A", "B", "both"], default="both", help="Which phase to benchmark (A=scalar, B=TMA)"
    )
    args = parser.parse_args()

    global NCU_MODE
    if args.ncu:
        NCU_MODE = True
        print("[NCU mode] warmup=1, iters=1")

    # Test configs - focus on shapes where K2 dominates
    configs = [
        (1, 512),
        (1, 1024),
        (1, 2048),
        (1, 4096),
        (2, 1024),
        (2, 2048),
        (4, 1024),
    ]

    if args.phase in ["A", "both"]:
        results_a = bench_k2(configs, phase="A")
        print_report(results_a, phase="A")

    if args.phase in ["B", "both"]:
        results_b = bench_k2(configs, phase="B")
        print_report(results_b, phase="B")


if __name__ == "__main__":
    main()

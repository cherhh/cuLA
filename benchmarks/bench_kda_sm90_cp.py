#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
bench_kda_sm90_cp.py — Benchmark: SM90 intracard CP speedup (CP-on vs CP-off)

Measures the speedup of the SM90 intracard context-parallel path against
the serial K1+K2 baseline across varlen configurations.

Usage:
  python bench_kda_sm90_cp.py [--ncu] [--sanitizer]

With --ncu, warmup=1 and iters=1 for ncu profiling:
  ncu --set full -o report python bench_kda_sm90_cp.py --ncu
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from benchmarks.utils import SEED, exclusive_cumsum, prepare_safe_gate_inputs, set_seed
from cula.kda import flashkda_prefill as cula_kda_prefill
from cula.utils import assert_hopper, get_device_sm_count

D = 128
H_VALUES = [4, 8]
WARMUP = 10
N_ITERS = 100
NCU_MODE = False
SANITIZER_MODE = False

# (tag, seq_lens) — each entry is tested at every H in H_VALUES.
# SM90 CHUNK=16, so sequences need to be long enough (>= ~8K tiles) for CP to pay off.
CONFIGS = [
    ("T=4K", [4096]),
    ("T=8K", [8192]),
    ("T=16K", [16384]),
    ("T=32K", [32768]),
    ("T=64K", [65536]),
    ("2x16K", [16384, 16384]),
    ("32K+4K", [32768, 4096]),
    ("32K+1K", [32768, 1024]),
    ("64K+1K", [65536, 1024]),
    ("64K+2x1K", [65536, 1024, 1024]),
    ("64K+5x1K", [65536] + [1024] * 5),
]


def time_kernel(fn, warmup=None, n_iters=None):
    if warmup is None:
        warmup = 1 if (NCU_MODE or SANITIZER_MODE) else WARMUP
    if n_iters is None:
        n_iters = 1 if (NCU_MODE or SANITIZER_MODE) else N_ITERS
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(n_iters):
        fn()
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) / n_iters


def run_kernel(q, k, v, g, beta, scale, A_log, dt_bias, cu_seqlens, lower_bound, *, use_cp):
    cula_kda_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        output_final_state=False,
        safe_gate=True,
        lower_bound=lower_bound,
        use_intracard_cp="auto" if use_cp else False,
    )


def bench_cp(h_values, configs):
    print("\n" + "=" * 100)
    print(" SM90 Intracard CP Benchmark: CP-on vs CP-off")
    print("=" * 100)

    device = torch.device("cuda")
    assert_hopper(device)
    num_sms = get_device_sm_count(device)
    print(f" [Device] {torch.cuda.get_device_name(0)}  SMs={num_sms}")
    results = []

    for H in h_values:
        for tag, seq_lens in configs:
            set_seed(SEED)
            torch.cuda.empty_cache()

            total_T = sum(seq_lens)
            cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)
            inputs = prepare_safe_gate_inputs(1, total_T, H, D, device, cu_seqlens=cu_seqlens, seed=SEED)
            q, k, v, g, beta = inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
            A_log, dt_bias = inputs["A_log"], inputs["dt_bias"]
            scale, lower_bound = inputs["scale"], inputs["lower_bound"]

            common = dict(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                A_log=A_log,
                dt_bias=dt_bias,
                cu_seqlens=cu_seqlens,
                lower_bound=lower_bound,
            )

            ms_off = time_kernel(lambda: run_kernel(**common, use_cp=False))
            ms_on = time_kernel(lambda: run_kernel(**common, use_cp=True))

            speedup = ms_off / ms_on if ms_on > 0 else float("inf")
            r = dict(tag=tag, H=H, total_T=total_T, ms_off=ms_off, ms_on=ms_on, speedup=speedup)
            results.append(r)

            del q, k, v, g, beta, A_log, dt_bias, inputs
            torch.cuda.empty_cache()

    return results


def print_report(results, h_values):
    sep = "=" * 95
    print(f"\n\n{sep}")
    print("               BENCHMARK REPORT: SM90 Intracard CP")
    print("               CP-on vs CP-off (intracard_prefill vs flash_kda_fwd)")
    print(f"               D={D}  dtype=bf16  safe_gate=True")
    wu = 1 if (NCU_MODE or SANITIZER_MODE) else WARMUP
    ni = 1 if (NCU_MODE or SANITIZER_MODE) else N_ITERS
    mode_tag = "  [NCU mode]" if NCU_MODE else ("  [Sanitizer mode]" if SANITIZER_MODE else "")
    print(f"               Warmup={wu}  Iters={ni}{mode_tag}")
    print(sep)

    for H_val in h_values:
        h_results = [r for r in results if r["H"] == H_val]
        if not h_results:
            continue
        print(f"\n  [H={H_val}]")
        print(f"  {'─' * 80}")
        print(f"  {'config':<20s} {'T':>7s}  │  {'CP_off(ms)':>10s}  {'CP_on(ms)':>10s}  {'Speedup':>8s}")
        print(f"  {'─' * 80}")
        for r in h_results:
            print(f"  {r['tag']:<20s} {r['total_T']:>7d}  │  {r['ms_off']:>10.4f}  {r['ms_on']:>10.4f}  {r['speedup']:>7.2f}x")
        print(f"  {'─' * 80}")

    speedups = [r["speedup"] for r in results]
    if speedups:
        geo = 1.0
        for s in speedups:
            geo *= s
        geo = geo ** (1 / len(speedups))
        print(f"\n  All configs: geo-mean={geo:.2f}x  best={max(speedups):.2f}x  worst={min(speedups):.2f}x")

    print(f"\n{sep}\n")


def main():
    parser = argparse.ArgumentParser(description="bench_kda_sm90_cp: SM90 intracard CP speedup")
    parser.add_argument("--ncu", action="store_true", help="NCU profiling mode: warmup=1, iters=1")
    parser.add_argument("--sanitizer", action="store_true", help="Sanitizer mode: warmup=1, iters=1")
    args = parser.parse_args()

    global NCU_MODE, SANITIZER_MODE
    if args.ncu:
        NCU_MODE = True
        print("[NCU mode] warmup=1, iters=1")
    if args.sanitizer:
        SANITIZER_MODE = True
        print("[Sanitizer mode] warmup=1, iters=1")

    results = bench_cp(H_VALUES, CONFIGS)
    print_report(results, H_VALUES)
    return results


if __name__ == "__main__":
    main()

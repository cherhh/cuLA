#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F821
"""3-way FlashKDA prefill benchmark:

  * cute : CuTeDSL ``cula.ops.flashkda_prefill.flash_kda_prefill``
           (env ``CULA_FLASHKDA_USE_CUTE=1`` -> ``launch_k1_full + launch_k2_phaseB``)
  * cpp  : CUTLASS C++ ``flash_kda_C.fwd``
  * fla  : Triton ``fla.ops.kda.chunk_kda``

Settings mirror ``benchmarks/bench_kda.py`` (H=64, D=128, bf16, safe_gate
inputs). Only fixed-length / no-state shapes are exercised because the
CuTeDSL and C++ paths do not support varlen / state I/O.

Usage:
    CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_flashkda_3way.py
    CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_flashkda_3way.py --ncu
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
os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))

import torch  # noqa: E402

from benchmarks.utils import SEED, exclusive_cumsum, set_seed  # noqa: E402
from cula.ops.flashkda_prefill import flash_kda_prefill  # noqa: E402

try:
    import flash_kda_C as flash_kda_cpp  # noqa: F401

    HAS_CPP = True
except Exception as e:  # pragma: no cover
    print(f"[warn] flash_kda_C not importable: {e!r}", file=sys.stderr)
    HAS_CPP = False

try:
    from fla.ops.kda import chunk_kda as fla_chunk_kda

    HAS_FLA = True
except Exception as e:  # pragma: no cover
    print(f"[warn] fla.ops.kda not importable: {e!r}", file=sys.stderr)
    HAS_FLA = False


# ============================================================
# Constants
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
    """Build inputs for all three paths.

    cute / cpp: ``(B, T, H, D)`` bf16 q/k/v/g, ``(B, T, H)`` bf16 beta,
                ``(H,)`` fp32 A_log, ``(H, D)`` fp32 dt_bias.
    fla       : ``(1, B*T, H, D)`` bf16 q/k/v/g, ``(1, B*T, H)`` fp32 beta
                (sigmoid pre-applied per ``prepare_safe_gate_inputs``),
                cu_seqlens int32 ``(B+1,)``.
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

    # FLA-format views (flatten batch into time, sigmoid beta as fp32).
    q_fla = q.reshape(1, B * T, H, D)
    k_fla = k.reshape(1, B * T, H, D)
    v_fla = v.reshape(1, B * T, H, D)
    g_fla = g.reshape(1, B * T, H, D)
    beta_fla = beta.float().sigmoid().reshape(1, B * T, H)
    cu_seqlens = torch.tensor(exclusive_cumsum([T] * B), dtype=torch.int32, device=device)

    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        q_fla=q_fla,
        k_fla=k_fla,
        v_fla=v_fla,
        g_fla=g_fla,
        beta_fla=beta_fla,
        cu_seqlens=cu_seqlens,
    )


def cpp_workspace_for(B, T, H_, device):
    n_bytes = flash_kda_cpp.get_workspace_size(B, T, H_)
    return torch.empty(n_bytes, dtype=torch.uint8, device=device)


# ============================================================
# Bench loop
# ============================================================
def bench_fixed(configs):
    print("\n" + "=" * 110)
    print(" Fixed-Length Benchmark: CuTeDSL (K1+K2) vs CUTLASS C++ vs FLA Triton  (H=64, D=128, bf16)")
    print("=" * 110)
    results = []
    scale = D**-0.5
    lower_bound = -5.0
    device = torch.device("cuda")

    for B, T in configs:
        torch.cuda.empty_cache()
        x = make_inputs(B, T, device)
        q, k, v, g, beta = x["q"], x["k"], x["v"], x["g"], x["beta"]
        A_log, dt_bias = x["A_log"], x["dt_bias"]
        out_cute = torch.empty_like(v)
        out_cpp = torch.empty_like(v)

        # --- CuTeDSL path ---
        cute_ok = True
        try:
            flash_kda_prefill(q, k, v, g, beta, scale, out_cute, A_log, dt_bias, lower_bound)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  [warn] cute failed B={B} T={T}: {e!r}", file=sys.stderr)
            cute_ok = False

        # --- C++ path ---
        cpp_ok = HAS_CPP
        if cpp_ok:
            try:
                ws = cpp_workspace_for(B, T, H, device)
                flash_kda_cpp.fwd(q, k, v, g, beta, scale, out_cpp, ws, A_log, dt_bias, lower_bound, None, None, None)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"  [warn] cpp failed B={B} T={T}: {e!r}", file=sys.stderr)
                cpp_ok = False

        # --- FLA path ---
        fla_ok = HAS_FLA
        if fla_ok:
            try:
                o_fla, _ = fla_chunk_kda(
                    q=x["q_fla"],
                    k=x["k_fla"],
                    v=x["v_fla"],
                    g=x["g_fla"],
                    beta=x["beta_fla"],
                    scale=scale,
                    A_log=A_log,
                    dt_bias=dt_bias.reshape(-1),
                    initial_state=None,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=x["cu_seqlens"],
                    use_gate_in_kernel=True,
                    safe_gate=True,
                    lower_bound=lower_bound,
                    disable_recompute=False,
                )
                torch.cuda.synchronize()
            except Exception as e:
                print(f"  [warn] fla failed B={B} T={T}: {e!r}", file=sys.stderr)
                fla_ok = False

        # Accuracy: cute vs cpp share input format
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

        ms_cpp = float("nan")
        if cpp_ok:
            ws = cpp_workspace_for(B, T, H, device)

            def _cpp():
                flash_kda_cpp.fwd(q, k, v, g, beta, scale, out_cpp, ws, A_log, dt_bias, lower_bound, None, None, None)

            ms_cpp = time_kernel(_cpp)

        ms_fla = float("nan")
        if fla_ok:
            dt_bias_flat = dt_bias.reshape(-1)

            def _fla():
                fla_chunk_kda(
                    q=x["q_fla"],
                    k=x["k_fla"],
                    v=x["v_fla"],
                    g=x["g_fla"],
                    beta=x["beta_fla"],
                    scale=scale,
                    A_log=A_log,
                    dt_bias=dt_bias_flat,
                    initial_state=None,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=x["cu_seqlens"],
                    use_gate_in_kernel=True,
                    safe_gate=True,
                    lower_bound=lower_bound,
                    disable_recompute=False,
                )

            ms_fla = time_kernel(_fla)

        sp_cpp_over_cute = (ms_cpp / ms_cute) if (ms_cute > 0 and ms_cpp == ms_cpp) else float("nan")
        sp_fla_over_cute = (ms_fla / ms_cute) if (ms_cute > 0 and ms_fla == ms_fla) else float("nan")

        r = dict(
            B=B,
            T=T,
            rmse=rmse,
            rel_max=rel_max,
            ms_cute=ms_cute,
            ms_cpp=ms_cpp,
            ms_fla=ms_fla,
            sp_cpp=sp_cpp_over_cute,
            sp_fla=sp_fla_over_cute,
        )
        results.append(r)
        del x, q, k, v, g, beta, A_log, dt_bias, out_cute, out_cpp

    return results


# ============================================================
# Report
# ============================================================
def print_report(rows):
    sep = "=" * 120
    print(f"\n\n{sep}")
    print("                       3-WAY BENCHMARK REPORT: FlashKDA Prefill")
    print("                       cute (K1+K2 CuTeDSL)  |  cpp (CUTLASS C++)  |  fla (Triton)")
    print(
        f"                       H={H}  D={D}  dtype=bf16   warmup={WARMUP if not NCU_MODE else 1}"
        f"   iters={N_ITERS if not NCU_MODE else 1}"
    )
    print(sep)
    print(
        f"  {'B':>3s}  {'T':>6s}  │  {'RMSE':>10s}  {'rel_max':>10s}"
        f"  │  {'cute(ms)':>10s}  {'cpp(ms)':>10s}  {'fla(ms)':>10s}"
        f"  │  {'cpp/cute':>9s}  {'fla/cute':>9s}"
    )
    print(f"  {'-' * 116}")
    for r in rows:
        print(
            f"  {r['B']:3d}  {r['T']:6d}  │  "
            f"{r['rmse']:10.6f}  {r['rel_max']:10.6f}  │  "
            f"{r['ms_cute']:10.4f}  {r['ms_cpp']:10.4f}  {r['ms_fla']:10.4f}  │  "
            f"{r['sp_cpp']:8.2f}x  {r['sp_fla']:8.2f}x"
        )
    print(f"  {'-' * 116}\n")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="3-way FlashKDA prefill benchmark")
    parser.add_argument("--ncu", action="store_true", help="NCU mode: warmup=1 iters=1")
    args = parser.parse_args()

    global NCU_MODE
    if args.ncu:
        NCU_MODE = True
        print("[NCU mode] warmup=1, iters=1")

    fixed_configs = [
        (1, 512),
        (1, 1024),
        (1, 4096),
        (1, 8192),
        (1, 16384),
        (2, 512),
        (2, 1024),
        (2, 4096),
        (2, 8192),
        (2, 16384),
    ]

    rows = bench_fixed(fixed_configs)
    print_report(rows)
    return rows


if __name__ == "__main__":
    main()

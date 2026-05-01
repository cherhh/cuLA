#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
bench_flashkda_cutedsl_vs_chunk.py
===================================
Benchmark & accuracy comparison:

  cuLA CuTeDSL K1+K2 (flash_kda_prefill)
    vs
  cuLA Triton chunk_kda_fwd (chunk_kda_fwd)

Both paths perform the *same* full forward pass (gate activation inside kernel,
safe_gate=True, use_qk_l2norm implicitly via K1), so only kernel implementation
differs.  Backward is NOT compared (CuTeDSL K1+K2 is forward-only).

Accuracy metrics (vs Triton chunk_kda as reference):
  rel_max  — max(|ref - out|) / max(|ref|)
  err_ratio — RMS error / RMS reference
  mean_diff — mean absolute error

Performance metrics:
  FWD time (ms) via CUDA events.

Usage:
  python benchmarks/bench_flashkda_cutedsl_vs_chunk.py [--mode fixed|varlen|both] [--ncu]
"""

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

os.environ["CULA_FLASHKDA_USE_CUTE"] = "1"

from benchmarks.utils import (
    SEED,
    build_varlen_configs,
    get_env_info,
    set_seed,
)
from cula.kda import chunk_kda as cula_chunk_kda
from cula.ops.flashkda_prefill import flash_kda_prefill

# ============================================================
# Constants
# ============================================================
H, D = 64, 128
WARMUP = 20
N_ITERS = 100
NCU_MODE = False


# ============================================================
# Input preparation
# ============================================================
def make_inputs(seq_lens, H, D, device, seed=SEED, has_init_state=False):
    """Generate raw inputs compatible with both APIs.

    Both APIs use:
      use_qk_l2norm_in_kernel=True  (L2-norm q,k inside the kernel)
      use_gate_in_kernel=True       (gate activation inside the kernel)
      safe_gate=True
    So raw (un-normalised) q/k/g inputs are passed to both.
    """
    T_total = sum(seq_lens)
    N = len(seq_lens)
    set_seed(seed)

    # BF16 q/k/v/g — both APIs accept BF16
    q = torch.randn(1, T_total, H, D, dtype=torch.bfloat16, device=device) * 0.3
    k = torch.randn_like(q) * 0.3
    v = torch.randn_like(q) * 0.3
    g = torch.randn_like(q) * 0.1

    # beta: BF16 for both (chunk_kda accepts bf16 too)
    beta = torch.randn(1, T_total, H, dtype=torch.bfloat16, device=device) * 0.3

    A_log = torch.full((H,), -0.1, dtype=torch.float32, device=device)
    # chunk_kda expects dt_bias of shape [H*D]; flash_kda_prefill expects [H, D]
    dt_bias_flat = torch.zeros(H * D, dtype=torch.float32, device=device)
    dt_bias_2d = dt_bias_flat.view(H, D)

    scale = D**-0.5
    lower_bound = -5.0

    cu_seqlens = torch.tensor(
        [0] + [sum(seq_lens[: i + 1]) for i in range(N)],
        dtype=torch.int32,
        device=device,
    )

    h0_f32 = None
    if has_init_state:
        h0_f32 = torch.randn(N, H, D, D, dtype=torch.float32, device=device) * 0.02

    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias_flat=dt_bias_flat,
        dt_bias_2d=dt_bias_2d,
        scale=scale,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        h0_f32=h0_f32,
        N=N,
        T=T_total,
    )


# ============================================================
# Kernel wrappers
# ============================================================
def run_chunk(inp, output_final=False):
    """Run cuLA Triton chunk_kda (reference) — uses same l2norm + gate path as flash_kda_prefill."""
    out, ht = cula_chunk_kda(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        g=inp["g"],
        beta=inp["beta"],
        scale=inp["scale"],
        initial_state=inp["h0_f32"],
        output_final_state=output_final,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=inp["lower_bound"],
        cu_seqlens=inp["cu_seqlens"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias_flat"],
    )
    torch.cuda.synchronize()
    return out, ht


def run_cute(inp, output_final=False):
    """Run CuTeDSL flash_kda_prefill (K1+K2)."""
    out = torch.zeros_like(inp["v"])
    ht = torch.zeros(inp["N"], H, D, D, dtype=torch.float32, device=inp["v"].device) if output_final else None
    flash_kda_prefill(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        g=inp["g"],
        beta=inp["beta"],
        scale=inp["scale"],
        out=out,
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias_2d"],
        lower_bound=inp["lower_bound"],
        initial_state=inp["h0_f32"],
        final_state=ht,
        cu_seqlens=inp["cu_seqlens"],
    )
    torch.cuda.synchronize()
    return out, ht


# ============================================================
# Timing helpers
# ============================================================
def time_kernel(fn):
    wu = 1 if NCU_MODE else WARMUP
    ni = 1 if NCU_MODE else N_ITERS
    for _ in range(wu):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ni):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ni


# ============================================================
# Accuracy helpers
# ============================================================
def accuracy_stats(ref, out):
    ref_f = ref.float()
    out_f = out.float()
    diff = (ref_f - out_f).abs()
    err = diff.flatten().pow(2).mean().sqrt().item()
    base = ref_f.flatten().pow(2).mean().sqrt().item()
    err_ratio = err / (base + 1e-8)
    max_diff = diff.max().item()
    denom = ref_f.abs().max().item()
    rel_max = max_diff / denom if denom > 0 else 0.0
    mean_diff = diff.mean().item()
    return err_ratio, rel_max, mean_diff


# ============================================================
# Fixed-length benchmark
# ============================================================
def bench_fixed(configs):
    results = []
    device = torch.device("cuda")

    for B, T in configs:
        torch.cuda.empty_cache()
        seq_lens = [T] * B
        inp = make_inputs(seq_lens, H, D, device, has_init_state=True)

        # Accuracy (with final state)
        ref_o, ref_ht = run_chunk(inp, output_final=True)
        cut_o, cut_ht = run_cute(inp, output_final=True)

        acc = {}
        for name, ref, got in [("o", ref_o, cut_o), ("ht", ref_ht, cut_ht)]:
            acc[name] = dict(zip(["err_ratio", "rel_max", "mean_diff"], accuracy_stats(ref, got)))

        # Timing
        ms_chunk = time_kernel(lambda: run_chunk(inp, output_final=False))
        ms_cute = time_kernel(lambda: run_cute(inp, output_final=False))
        speedup = ms_chunk / ms_cute if ms_cute > 0 else float("inf")

        results.append(dict(B=B, T=T, acc=acc, ms_chunk=ms_chunk, ms_cute=ms_cute, speedup=speedup))
        del ref_o, ref_ht, cut_o, cut_ht
        torch.cuda.empty_cache()

    return results


# ============================================================
# Varlen benchmark
# ============================================================
def bench_varlen(configs):
    results = []
    device = torch.device("cuda")

    for seq_lens, total_len, dist in configs:
        torch.cuda.empty_cache()
        N = len(seq_lens)
        inp = make_inputs(seq_lens, H, D, device, has_init_state=True)

        ref_o, ref_ht = run_chunk(inp, output_final=True)
        cut_o, cut_ht = run_cute(inp, output_final=True)

        acc = {}
        for name, ref, got in [("o", ref_o, cut_o), ("ht", ref_ht, cut_ht)]:
            acc[name] = dict(zip(["err_ratio", "rel_max", "mean_diff"], accuracy_stats(ref, got)))

        ms_chunk = time_kernel(lambda: run_chunk(inp, output_final=False))
        ms_cute = time_kernel(lambda: run_cute(inp, output_final=False))
        speedup = ms_chunk / ms_cute if ms_cute > 0 else float("inf")

        min_l, max_l = min(seq_lens), max(seq_lens)
        tag = f"{dist:>7s} {N:>2d}seqs T={total_len} [{min_l}..{max_l}]"
        results.append(dict(tag=tag, T=total_len, N=N, acc=acc, ms_chunk=ms_chunk, ms_cute=ms_cute, speedup=speedup))
        del ref_o, ref_ht, cut_o, cut_ht
        torch.cuda.empty_cache()

    return results


# ============================================================
# Report
# ============================================================
def print_report(fixed_results, varlen_results):
    env = get_env_info()
    sep = "=" * 130
    print(f"\n{sep}")
    print("   FlashKDA CuTeDSL K1+K2 (flash_kda_prefill) vs Triton chunk_kda_fwd")
    print(f"   H={H}  D={D}  dtype=bf16/f32  safe_gate=True  use_gate_in_kernel=True")
    print(f"   GPU={env['gpu']}  CUDA={env['cuda']}  PyTorch={env['torch']}")
    wu = 1 if NCU_MODE else WARMUP
    ni = 1 if NCU_MODE else N_ITERS
    print(f"   Warmup={wu}  Iters={ni}{'  [NCU mode]' if NCU_MODE else ''}")
    print("   Accuracy reference: cuLA Triton chunk_kda (use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True)")
    print(sep)

    acc_keys = ["o", "ht"]
    hdr_acc = "  ".join(f"{'rel_max_' + k:>12s}  {'err_ratio_' + k:>13s}" for k in acc_keys)

    def fmt_acc(acc):
        parts = []
        for k in acc_keys:
            a = acc.get(k, {})
            parts.append(f"{a.get('rel_max', float('nan')):12.6f}  {a.get('err_ratio', float('nan')):13.6f}")
        return "  ".join(parts)

    if fixed_results:
        print("\n  [Fixed-Length]  (B seqs, each length T)")
        head = f"  {'B':>3s}  {'T':>6s}  │  {'chunk(ms)':>10s}  {'cute(ms)':>10s}  {'Speedup':>8s}  │  {hdr_acc}"
        print(f"  {'─' * (len(head) - 2)}")
        print(head)
        print(f"  {'─' * (len(head) - 2)}")
        for r in fixed_results:
            print(
                f"  {r['B']:3d}  {r['T']:6d}  │  "
                f"{r['ms_chunk']:10.3f}  {r['ms_cute']:10.3f}  {r['speedup']:7.2f}x  │  "
                f"{fmt_acc(r['acc'])}"
            )
        print(f"  {'─' * (len(head) - 2)}")

    if varlen_results:
        print("\n  [Varlen]")
        head = f"  {'Config':>45s}  │  {'chunk(ms)':>10s}  {'cute(ms)':>10s}  {'Speedup':>8s}  │  {hdr_acc}"
        print(f"  {'─' * (len(head) - 2)}")
        print(head)
        print(f"  {'─' * (len(head) - 2)}")
        for r in varlen_results:
            print(
                f"  {r['tag']:>45s}  │  "
                f"{r['ms_chunk']:10.3f}  {r['ms_cute']:10.3f}  {r['speedup']:7.2f}x  │  "
                f"{fmt_acc(r['acc'])}"
            )
        print(f"  {'─' * (len(head) - 2)}")

    print(f"\n{sep}\n")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="FlashKDA CuTeDSL K1+K2 vs Triton chunk_kda_fwd")
    parser.add_argument("--mode", default="both", choices=["fixed", "varlen", "both"])
    parser.add_argument("--ncu", action="store_true", help="NCU mode: warmup=1, iters=1")
    args = parser.parse_args()

    global NCU_MODE
    if args.ncu:
        NCU_MODE = True

    fixed_configs = [
        (1, 512),
        (1, 1024),
        (1, 2048),
        (1, 4096),
        (1, 8192),
        (1, 16384),
        (2, 1024),
        (2, 4096),
        (2, 8192),
        (2, 16384),
    ]

    # flash_kda_prefill (CuTeDSL K1) requires all seq_lens to be multiples of CHUNK=16.
    # Snap each seq_len up to the next multiple of 16 and update total_len accordingly.
    CHUNK = 16
    raw_varlen = build_varlen_configs(
        num_seqs_list=(10, 20),
        total_lens=(4096, 8192, 16384),
        dists=("uniform", "random", "skewed"),
    )
    varlen_configs = []
    for seq_lens, total_len, dist in raw_varlen:
        snapped = [((s + CHUNK - 1) // CHUNK) * CHUNK for s in seq_lens]
        varlen_configs.append((snapped, sum(snapped), dist))

    fixed_res, varlen_res = [], []

    if args.mode in ("fixed", "both"):
        fixed_res = bench_fixed(fixed_configs)

    if args.mode in ("varlen", "both"):
        varlen_res = bench_varlen(varlen_configs)

    print_report(fixed_res, varlen_res)


if __name__ == "__main__":
    main()

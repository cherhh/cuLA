#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end benchmark: FlashKDA CuTeDSL port vs FlashKDA C++.

Uses cuLA's standard settings (mirrors bench_kda_fused_fwd.py):
  - H=64, D=128, bf16
  - Fixed (B, T) ∈ {1, 2} × {512, 1024, 4096, 8192, 16384}
  - Varlen num_seqs ∈ {10, 20} × total_len ∈ {4K, 8K, 16K} × dists
  - warmup=25, iters=100

FlashKDA CuTeDSL (K1) requires seq_lens % 16 == 0; varlen lens snapped up
to the next multiple of 16. Both implementations see the same snapped inputs.
"""

import argparse
import os
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ["CULA_FLASHKDA_USE_CUTE"] = "1"

import numpy as np
import torch

from cula.ops.flashkda.prefill import flash_kda_prefill

H, D = 64, 128
CHUNK = 16
WARMUP = 25
N_ITERS = 100
NCU_MODE = False
SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def exclusive_cumsum(lens):
    out = [0]
    for x in lens:
        out.append(out[-1] + x)
    return out


def gen_uniform(N, T):
    per = T // N
    lens = [per] * N
    lens[0] += T - per * N
    return lens


def gen_skewed(N, T):
    if N == 1:
        return [T]
    short = max(1, T // (2 * (N - 1)))
    long_len = T - short * (N - 1)
    return [long_len] + [short] * (N - 1)


def gen_random(N, T, seed=42):
    rng = np.random.RandomState(seed)
    raw = rng.dirichlet(np.ones(N))
    lens = np.maximum(1, np.round(raw * T).astype(int))
    lens[0] += T - lens.sum()
    return np.maximum(1, lens).tolist()


def snap_seq_lens(seq_lens, chunk):
    return [((s + chunk - 1) // chunk) * chunk for s in seq_lens]


def make_inputs(B, T, device):
    """Allocate inputs of shape [B, T, H, D]."""
    set_seed(SEED)
    dtype = torch.bfloat16
    scale = D ** -0.5
    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device)
    g = torch.randn(B, T, H, D, dtype=dtype, device=device)
    beta = torch.randn(B, T, H, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(H, dtype=torch.float, device=device)
    dt_bias = torch.randn(H, D, dtype=torch.float, device=device)
    return dict(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log,
                dt_bias=dt_bias, scale=scale, lower_bound=-5.0)


def time_kernel(fn):
    wu = 1 if NCU_MODE else WARMUP
    ni = 1 if NCU_MODE else N_ITERS
    for _ in range(wu):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ni):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ni


def _bench_pair(q, k, v, g, beta, A_log, dt_bias, scale, lower_bound, cu_seqlens):
    out_cpp = torch.empty_like(v)
    out_cute = torch.empty_like(v)

    def run_cpp():
        import flash_kda
        flash_kda.fwd(q, k, v, g, beta, scale, out_cpp,
                      A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
                      cu_seqlens=cu_seqlens)

    def run_cute():
        flash_kda_prefill(q, k, v, g, beta,
                          scale=scale, out=out_cute,
                          A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
                          cu_seqlens=cu_seqlens)

    run_cpp(); run_cute()  # JIT warmup
    torch.cuda.synchronize()
    ms_cpp = time_kernel(run_cpp)
    ms_cute = time_kernel(run_cute)
    rel_max = (out_cute.float() - out_cpp.float()).abs().max().item()
    rel_max /= max(out_cpp.float().abs().max().item(), 1e-6)
    return ms_cpp, ms_cute, rel_max


def run_fixed(B, T):
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    inp = make_inputs(B, T, device)
    ms_cpp, ms_cute, rel_max = _bench_pair(
        inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"],
        inp["A_log"], inp["dt_bias"], inp["scale"], inp["lower_bound"],
        cu_seqlens=None,
    )
    tag = f"fixed   B={B:>2d}  T={T:>5d}"
    del inp
    torch.cuda.empty_cache()
    return {"tag": tag, "ms_cpp": ms_cpp, "ms_cute": ms_cute, "rel_max": rel_max}


def run_varlen(seq_lens, dist):
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    T = sum(seq_lens)
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.long, device=device)
    inp = make_inputs(1, T, device)
    ms_cpp, ms_cute, rel_max = _bench_pair(
        inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"],
        inp["A_log"], inp["dt_bias"], inp["scale"], inp["lower_bound"],
        cu_seqlens=cu_seqlens,
    )
    n_seqs = len(seq_lens)
    min_l, max_l = min(seq_lens), max(seq_lens)
    avg_l = T // n_seqs
    tag = f"{dist:>7s} {n_seqs:>2d}seqs T={T:>5d} [{min_l}..{max_l}] avg={avg_l}"
    del inp
    torch.cuda.empty_cache()
    return {"tag": tag, "ms_cpp": ms_cpp, "ms_cute": ms_cute, "rel_max": rel_max}


FIXED_CONFIGS = [
    (1, 512), (1, 1024), (1, 4096), (1, 8192), (1, 16384),
    (2, 512), (2, 1024), (2, 4096), (2, 8192), (2, 16384),
]


def build_varlen_configs():
    num_seqs_list = (10, 20)
    total_lens = (4096, 8192, 16384)
    configs = []
    for T in total_lens:
        for N in num_seqs_list:
            for dist in ("uniform", "random", "skewed"):
                if dist == "uniform":
                    lens = gen_uniform(N, T)
                elif dist == "skewed":
                    lens = gen_skewed(N, T)
                else:
                    lens = gen_random(N, T, seed=SEED)
                configs.append((snap_seq_lens(lens, CHUNK), dist))
    return configs


def print_section(title, results):
    sep = "=" * 100
    print(f"\n{sep}")
    print(f" {title}")
    print(sep)
    hdr = f"  {'Config':>45s}  │  {'cute(ms)':>9s}  {'cpp(ms)':>9s}  {'cute/cpp':>9s}  {'rel_max':>10s}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    ratios = []
    for r in results:
        ratio = r["ms_cute"] / r["ms_cpp"]
        ratios.append(ratio)
        print(f"  {r['tag']:>45s}  │  {r['ms_cute']:9.3f}  {r['ms_cpp']:9.3f}  {ratio:8.2f}x  {r['rel_max']:10.2e}")
    print("  " + "─" * (len(hdr) - 2))
    if ratios:
        from math import exp, log
        geomean = exp(sum(log(r) for r in ratios) / len(ratios))
        wins = sum(1 for r in ratios if r < 1.0)
        draws = sum(1 for r in ratios if 1.0 <= r <= 1.02)
        loses = len(ratios) - wins - draws
        print(f"  Geomean cute/cpp = {geomean:.3f}x  |  cute wins {wins}/{len(ratios)}, draws {draws}, loses {loses}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="E2E bench: FlashKDA CuTeDSL vs C++ (fixed + varlen)")
    parser.add_argument("--mode", choices=["fixed", "varlen", "both"], default="both")
    parser.add_argument("--ncu", action="store_true")
    args = parser.parse_args()

    global NCU_MODE
    NCU_MODE = args.ncu

    wu = 1 if NCU_MODE else WARMUP
    ni = 1 if NCU_MODE else N_ITERS
    print(f"[Device] {torch.cuda.get_device_name(0)}")
    print(f"[Config] H={H} D={D} bf16  warmup={wu} iters={ni}")

    fixed_results = []
    varlen_results = []

    if args.mode in ("fixed", "both"):
        for B, T in FIXED_CONFIGS:
            try:
                r = run_fixed(B, T)
                fixed_results.append(r)
                print(f"  {r['tag']}  │  cute={r['ms_cute']:7.3f}ms  cpp={r['ms_cpp']:7.3f}ms  rel_max={r['rel_max']:.2e}")
            except Exception as e:
                print(f"  [FAIL] fixed B={B} T={T}: {e}")

    if args.mode in ("varlen", "both"):
        for seq_lens, dist in build_varlen_configs():
            try:
                r = run_varlen(seq_lens, dist)
                varlen_results.append(r)
                print(f"  {r['tag']}  │  cute={r['ms_cute']:7.3f}ms  cpp={r['ms_cpp']:7.3f}ms  rel_max={r['rel_max']:.2e}")
            except Exception as e:
                print(f"  [FAIL] {dist} {len(seq_lens)}seqs T={sum(seq_lens)}: {e}")

    if fixed_results:
        print_section("E2E Fixed — FlashKDA CuTeDSL (cuLA-ported) vs FlashKDA C++ (upstream)", fixed_results)
    if varlen_results:
        print_section("E2E Varlen — FlashKDA CuTeDSL (cuLA-ported) vs FlashKDA C++ (upstream)", varlen_results)


if __name__ == "__main__":
    main()

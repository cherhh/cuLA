#!/usr/bin/env python3
"""Full-config benchmark: state_transposed=False (K-last) vs True (V-last).

Covers the same 10 fixed + 18 varlen configs as bench_flashkda_e2e.py, but
with initial_state + final_state provided so the state_transposed flag is
actually exercised. Has-state path disables CUDA Graph, so both runs go
through direct K1+K2 launch.
"""
import os
import sys
import pathlib
import random

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
os.environ["CULA_FLASHKDA_USE_CUTE"] = "1"

import numpy as np
import torch
from cula.ops.sm90.flashkda.prefill import flash_kda_prefill

H, D = 64, 128
CHUNK = 16
WARMUP = 25
ITERS = 100
SEED = 42


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def exclusive_cumsum(lens):
    out = [0]
    for x in lens: out.append(out[-1] + x)
    return out


def gen_uniform(N, T):
    per = T // N; lens = [per] * N; lens[0] += T - per * N; return lens


def gen_skewed(N, T):
    if N == 1: return [T]
    short = max(1, T // (2 * (N - 1))); long_len = T - short * (N - 1)
    return [long_len] + [short] * (N - 1)


def gen_random(N, T, seed=42):
    rng = np.random.RandomState(seed)
    raw = rng.dirichlet(np.ones(N))
    lens = np.maximum(1, np.round(raw * T).astype(int))
    lens[0] += T - lens.sum()
    return np.maximum(1, lens).tolist()


def snap(lens, c): return [((s + c - 1) // c) * c for s in lens]


def time_kernel(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS


def bench_one(B, T, cu_seqlens, N_state):
    """Time both state layouts for one config."""
    device = torch.device("cuda")
    set_seed(SEED)
    dtype = torch.bfloat16
    scale = D ** -0.5
    lb = -5.0

    if cu_seqlens is None:  # fixed mode
        q = torch.randn(B, T, H, D, dtype=dtype, device=device)
        k = torch.randn(B, T, H, D, dtype=dtype, device=device)
        v = torch.randn(B, T, H, D, dtype=dtype, device=device)
        g = torch.randn(B, T, H, D, dtype=dtype, device=device)
        beta = torch.randn(B, T, H, dtype=dtype, device=device)
    else:  # varlen (B=1, T=T_total)
        q = torch.randn(1, T, H, D, dtype=dtype, device=device)
        k = torch.randn(1, T, H, D, dtype=dtype, device=device)
        v = torch.randn(1, T, H, D, dtype=dtype, device=device)
        g = torch.randn(1, T, H, D, dtype=dtype, device=device)
        beta = torch.randn(1, T, H, dtype=dtype, device=device)
    A_log = torch.randn(H, dtype=torch.float32, device=device)
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=device)

    out_f = torch.empty_like(v); out_t = torch.empty_like(v)
    # initial_state shape is [N, H, D, D] — D=D regardless of which is K/V
    init_f = torch.randn(N_state, H, D, D, dtype=torch.float32, device=device)
    init_t = torch.randn(N_state, H, D, D, dtype=torch.float32, device=device)
    fin_f = torch.empty_like(init_f); fin_t = torch.empty_like(init_t)

    def run_false():
        flash_kda_prefill(q, k, v, g, beta, scale=scale, out=out_f,
                          A_log=A_log, dt_bias=dt_bias, lower_bound=lb,
                          initial_state=init_f, final_state=fin_f,
                          cu_seqlens=cu_seqlens, state_transposed=False)

    def run_true():
        flash_kda_prefill(q, k, v, g, beta, scale=scale, out=out_t,
                          A_log=A_log, dt_bias=dt_bias, lower_bound=lb,
                          initial_state=init_t, final_state=fin_t,
                          cu_seqlens=cu_seqlens, state_transposed=True)

    run_false(); run_true()  # JIT warmup
    torch.cuda.synchronize()
    ms_f = time_kernel(run_false)
    ms_t = time_kernel(run_true)
    return ms_f, ms_t


FIXED_CONFIGS = [
    (1, 512), (1, 1024), (1, 4096), (1, 8192), (1, 16384),
    (2, 512), (2, 1024), (2, 4096), (2, 8192), (2, 16384),
]


def build_varlen_configs():
    cfgs = []
    for T in (4096, 8192, 16384):
        for N in (10, 20):
            for dist in ("uniform", "random", "skewed"):
                if dist == "uniform": lens = gen_uniform(N, T)
                elif dist == "skewed": lens = gen_skewed(N, T)
                else: lens = gen_random(N, T, seed=SEED)
                cfgs.append((snap(lens, CHUNK), dist))
    return cfgs


def print_section(title, results):
    sep = "=" * 100
    print(f"\n{sep}\n {title}\n{sep}")
    hdr = f"  {'Config':>45s}  │  {'False(ms)':>10s}  {'True(ms)':>10s}  {'T/F ratio':>10s}"
    print(hdr); print("  " + "─" * (len(hdr) - 2))
    ratios = []
    for tag, ms_f, ms_t in results:
        r = ms_t / ms_f; ratios.append(r)
        print(f"  {tag:>45s}  │  {ms_f:>10.3f}  {ms_t:>10.3f}  {r:>9.3f}x")
    print("  " + "─" * (len(hdr) - 2))
    if ratios:
        from math import exp, log
        gmean = exp(sum(log(r) for r in ratios) / len(ratios))
        spread = (max(ratios) - min(ratios)) * 100
        print(f"  Geomean T/F = {gmean:.4f}x  |  spread = {spread:.2f}%  |  max |Δ| = {max(abs(r-1.0) for r in ratios)*100:.2f}%")
    print(sep)


def main():
    print(f"[Device] {torch.cuda.get_device_name(0)}")
    print(f"[Config] H={H} D={D} bf16  warmup={WARMUP} iters={ITERS}  WITH initial_state + final_state\n")
    print("Note: has-state path disables CUDA Graph; both runs go through direct K1+K2 launch.\n")

    fixed_res = []
    for B, T in FIXED_CONFIGS:
        try:
            ms_f, ms_t = bench_one(B, T, cu_seqlens=None, N_state=B)
            tag = f"fixed   B={B:>2d}  T={T:>5d}"
            fixed_res.append((tag, ms_f, ms_t))
            print(f"  {tag}  │  False={ms_f:7.3f}ms  True={ms_t:7.3f}ms  T/F={ms_t/ms_f:.3f}x")
        except Exception as e:
            print(f"  fixed B={B} T={T}  FAIL: {e}")

    print()
    varlen_res = []
    for seq_lens, dist in build_varlen_configs():
        T = sum(seq_lens); N = len(seq_lens)
        try:
            device = torch.device("cuda")
            cu = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.long, device=device)
            ms_f, ms_t = bench_one(1, T, cu_seqlens=cu, N_state=N)
            min_l, max_l = min(seq_lens), max(seq_lens); avg = T // N
            tag = f"{dist:>7s} {N:>2d}seqs T={T:>5d} [{min_l}..{max_l}] avg={avg}"
            varlen_res.append((tag, ms_f, ms_t))
            print(f"  {tag}  │  False={ms_f:7.3f}ms  True={ms_t:7.3f}ms  T/F={ms_t/ms_f:.3f}x")
        except Exception as e:
            print(f"  varlen {dist} N={N} T={T}  FAIL: {e}")

    if fixed_res: print_section("Fixed — K-last (False) vs V-last (True), WITH state", fixed_res)
    if varlen_res: print_section("Varlen — K-last (False) vs V-last (True), WITH state", varlen_res)


if __name__ == "__main__":
    main()

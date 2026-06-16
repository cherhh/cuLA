#!/usr/bin/env python3
"""Micro-bench: state_transposed=False vs True (K-contiguous vs V-contiguous).

Has-state path disables CUDA Graph (has_state_in), so both runs go through the
direct K1+K2 launch path. Pure measurement of kernel + dispatch differences.
"""
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
os.environ["CULA_FLASHKDA_USE_CUTE"] = "1"

import torch
from cula.ops.sm90.flashkda.prefill import flash_kda_prefill

H, D = 64, 128
WARMUP = 25
ITERS = 100


def time_kernel(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def bench(B, T):
    device = torch.device("cuda")
    torch.manual_seed(0)
    dtype = torch.bfloat16
    scale = D ** -0.5
    lower_bound = -5.0

    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device)
    g = torch.randn(B, T, H, D, dtype=dtype, device=device)
    beta = torch.randn(B, T, H, dtype=dtype, device=device)
    A_log = torch.randn(H, dtype=torch.float32, device=device)
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=device)

    out_a = torch.empty_like(v)
    out_b = torch.empty_like(v)

    # state_transposed=False : [N=B, H, V=D, K=D]  (K-contiguous)
    init_state_a = torch.randn(B, H, D, D, dtype=torch.float32, device=device)
    final_state_a = torch.empty_like(init_state_a)

    # state_transposed=True : [N=B, H, K=D, V=D]  (V-contiguous)
    init_state_b = torch.randn(B, H, D, D, dtype=torch.float32, device=device)
    final_state_b = torch.empty_like(init_state_b)

    def run_a():
        flash_kda_prefill(q, k, v, g, beta,
                          scale=scale, out=out_a,
                          A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
                          initial_state=init_state_a, final_state=final_state_a,
                          state_transposed=False)

    def run_b():
        flash_kda_prefill(q, k, v, g, beta,
                          scale=scale, out=out_b,
                          A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
                          initial_state=init_state_b, final_state=final_state_b,
                          state_transposed=True)

    # JIT warmup (each layout gets its own JIT compile)
    run_a(); run_b()
    torch.cuda.synchronize()

    ms_a = time_kernel(run_a)
    ms_b = time_kernel(run_b)

    return ms_a, ms_b


def main():
    print(f"[Device] {torch.cuda.get_device_name(0)}")
    print(f"[Config] H={H} D={D} bf16  warmup={WARMUP} iters={ITERS}")
    print()
    print(f"{'B':>3} {'T':>6}  │  {'False (K-cont)':>15}  {'True (V-cont)':>15}  {'B/A ratio':>10}")
    print("─" * 65)
    for B, T in [(1, 1024), (1, 4096), (1, 8192), (1, 16384),
                 (2, 4096), (2, 8192), (4, 8192)]:
        try:
            ms_a, ms_b = bench(B, T)
            ratio = ms_b / ms_a
            print(f"{B:>3} {T:>6}  │  {ms_a:>13.3f}ms  {ms_b:>13.3f}ms  {ratio:>9.3f}x")
        except Exception as e:
            print(f"{B:>3} {T:>6}  │  FAIL: {e}")


if __name__ == "__main__":
    main()

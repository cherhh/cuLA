"""Profile harness for FlashKDA CuTeDSL prefill.

Runs K1 + K2 on a representative varlen workload (8 seqs × 1024 tokens, H=64,
D=128) with cudaProfilerStart/Stop wrapping the timed iteration, so nsys/ncu
only capture the kernels we care about (warmup runs outside the range).

Usage:
  nsys profile --capture-range cudaProfilerApi --capture-range-end stop \
      -o flashkda_cute python harness.py
  ncu --target-processes all --set full --replay-mode kernel \
      -k 'regex:k[12]_kernel' -c 1 \
      -o flashkda_cute python harness.py --iters 1
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ["CULA_FLASHKDA_USE_CUTE"] = "1"

import torch
from cula.ops.sm90.flashkda.prefill import flash_kda_prefill


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--N", type=int, default=8, help="num sequences")
    parser.add_argument("--T-per-seq", type=int, default=1024)
    parser.add_argument("--H", type=int, default=64)
    args = parser.parse_args()

    H, D, CHUNK = args.H, 128, 16
    N, T_per_seq = args.N, args.T_per_seq
    assert T_per_seq % CHUNK == 0, "FlashKDA requires seq_len % 16 == 0"

    device = torch.device("cuda")
    torch.manual_seed(0)
    T_total = N * T_per_seq
    cu_seqlens = torch.arange(0, T_total + 1, T_per_seq, dtype=torch.long, device=device)

    dtype = torch.bfloat16
    q = torch.randn(1, T_total, H, D, dtype=dtype, device=device)
    k = torch.randn(1, T_total, H, D, dtype=dtype, device=device)
    v = torch.randn(1, T_total, H, D, dtype=dtype, device=device)
    g = torch.randn(1, T_total, H, D, dtype=dtype, device=device)
    beta = torch.randn(1, T_total, H, dtype=dtype, device=device)
    A_log = torch.randn(H, dtype=torch.float32, device=device)
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=device)
    out = torch.empty_like(v)

    scale = D ** -0.5
    lower_bound = -5.0

    def run():
        flash_kda_prefill(q, k, v, g, beta,
                          scale=scale, out=out,
                          A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
                          cu_seqlens=cu_seqlens)

    # JIT warmup (5x outside the profiled range)
    for _ in range(5):
        run()
    torch.cuda.synchronize()

    # Profiled region
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.iters):
        run()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(f"[harness] N={N} T_per_seq={T_per_seq} H={H} D={D}  iters={args.iters}")
    print(f"[harness] T_total={T_total}  out.shape={tuple(out.shape)}")


if __name__ == "__main__":
    main()

# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""3-way bench (cuLA serial / cuLA CP / FLA chunk_kda) for SM90 intracard-CP comparison."""

import csv
import math
import os
import sys

os.environ.setdefault("CULA_FLASHKDA_USE_CUTE", "1")
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import torch  # noqa: E402

from cula.ops.sm90.flashkda.cp import flash_kda_prefill_cp  # noqa: E402
from cula.ops.sm90.flashkda.k2 import D  # noqa: E402
from cula.ops.sm90.flashkda.prefill import flash_kda_prefill  # noqa: E402

try:
    from fla.ops.kda import chunk_kda
except ImportError:
    chunk_kda = None

SCALE = 1.0 / math.sqrt(D)
LB = -5.0
WARMUP = 10
ITERS = 10

CONFIGS = [
    ("4x256", [256] * 4),
    ("8x256", [256] * 8),
    ("16x256", [256] * 16),
    ("4x1K", [1024] * 4),
    ("8x1K", [1024] * 8),
    ("4x2K", [2048] * 4),
    ("1K+512+256+128", [1024, 512, 256, 128]),
    ("2K+1K+512+256", [2048, 1024, 512, 256]),
    ("1K+1+63+65+129", [1024, 1, 63, 65, 129]),
    ("T=4K", [4096]),
    ("T=8K", [8192]),
    ("T=32K", [32768]),
    ("T=64K", [65536]),
    ("T=128K", [131072]),
    ("8x4K", [4096] * 8),
    ("4x8K", [8192] * 4),
    ("2x16K", [16384] * 2),
    ("16K+16K", [16384, 16384]),
    ("24K+8K", [24576, 8192]),
    ("28K+4K", [28672, 4096]),
    ("32K+256+256", [32768, 256, 256]),
    ("40K+1K+8K", [40960, 1024, 8192]),
    ("64K+512+256+128", [65536, 512, 256, 128]),
    ("128K+1K", [131072, 1024]),
    ("128K+2x1K", [131072, 1024, 1024]),
    ("128K+5x1K", [131072] + [1024] * 5),
    ("128K+10x1K", [131072] + [1024] * 10),
]
H_VALUES = [4, 8]


def bench(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def make_inputs(T, H, dev):
    torch.manual_seed(0)
    mk = lambda *s: torch.randn(*s, dtype=torch.bfloat16, device=dev)
    q, k, v, g = (mk(1, T, H, D) for _ in range(4))
    beta = mk(1, T, H)
    A_log = torch.randn(H, dtype=torch.float32, device=dev)
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=dev)
    return q, k, v, g, beta, A_log, dt_bias


def main():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "benchmark_3way_cula_fla.csv")
    dev = torch.device("cuda")
    prop = torch.cuda.get_device_properties(dev)
    print(f"device={prop.name} SMs={prop.multi_processor_count} "
          f"D={D} warmup={WARMUP} iters={ITERS} fla={'yes' if chunk_kda else 'NO'}",
          flush=True)

    fcsv = open(csv_path, "w", newline="")
    wr = csv.writer(fcsv)
    wr.writerow(["H", "config", "total_T", "n_seqs",
                 "cula_serial_ms", "cula_cp_ms", "fla_ms"])

    for H in H_VALUES:
        print(f"\n  [H={H}]", flush=True)
        print(f"  {'config':<24s} {'T':>7s}  {'serial':>8s}  {'cp_auto':>8s}  {'fla':>8s}",
              flush=True)
        for tag, seq_lens in CONFIGS:
            T_total = sum(seq_lens)
            n_seqs = len(seq_lens)
            q, k, v, g, beta, A_log, dt_bias = make_inputs(T_total, H, dev)
            out = torch.empty_like(v)
            fin = torch.empty(n_seqs, H, D, D, dtype=torch.float32, device=dev)
            cu32 = cu64 = None
            if n_seqs > 1:
                cl = [0]
                for sl in seq_lens:
                    cl.append(cl[-1] + sl)
                cu32 = torch.tensor(cl, dtype=torch.int32, device=dev)
                cu64 = cu32.to(torch.int64)

            def run_serial():
                flash_kda_prefill(q, k, v, g, beta, scale=SCALE, out=out,
                                  A_log=A_log, dt_bias=dt_bias, lower_bound=LB,
                                  final_state=fin, cu_seqlens=cu32)

            def run_cp():
                flash_kda_prefill_cp(q, k, v, g, beta, scale=SCALE, out=out,
                                     A_log=A_log, dt_bias=dt_bias, lower_bound=LB,
                                     final_state=fin, cu_seqlens=cu32, s_split=None)

            try:
                ms_ser = bench(run_serial)
            except Exception as exc:
                print(f"    [{tag}] serial error: {exc}", flush=True)
                ms_ser = float("nan")

            try:
                ms_cp = bench(run_cp)
            except Exception as exc:
                print(f"    [{tag}] cp skip: {type(exc).__name__}", flush=True)
                ms_cp = float("nan")

            ms_fla = float("nan")
            if chunk_kda is not None:
                try:
                    def run_fla():
                        chunk_kda(q=q, k=k, v=v, g=g, beta=beta, scale=SCALE,
                                  initial_state=None, output_final_state=True,
                                  use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True,
                                  use_beta_sigmoid_in_kernel=True,
                                  A_log=A_log, dt_bias=dt_bias, lower_bound=LB,
                                  transpose_state_layout=True, cu_seqlens=cu64)
                    ms_fla = bench(run_fla)
                except Exception as exc:
                    print(f"    [{tag}] fla error: {exc}", flush=True)

            print(f"  {tag:<24s} {T_total:>7d}  {ms_ser:>8.3f}  {ms_cp:>8.3f}  {ms_fla:>8.3f}",
                  flush=True)
            wr.writerow([H, tag, T_total, n_seqs,
                         f"{ms_ser:.4f}", f"{ms_cp:.4f}", f"{ms_fla:.4f}"])
            fcsv.flush()

            del q, k, v, g, beta, out, fin
            torch.cuda.empty_cache()

    fcsv.close()
    print(f"\nwrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()

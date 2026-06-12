# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Mirror hyaloid intcd-cp bench (bench_intracard_cp_sm90.py): same 17 configs
× H in {4, 8}, same warmup/iters, time cuLA's CuTeDSL serial vs CP[auto].
Merged with hyaloid's CP-off/CP-on numbers offline (CSV)."""

import argparse
import csv
import math
import os
import sys

os.environ.setdefault("CULA_FLASHKDA_USE_CUTE", "1")
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import torch  # noqa: E402

from cula.ops.flashkda.cp import flash_kda_prefill_cp  # noqa: E402
from cula.ops.flashkda.k2 import D  # noqa: E402
from cula.ops.flashkda.prefill import flash_kda_prefill  # noqa: E402

SCALE = 1.0 / math.sqrt(D)
LB = -5.0
WARMUP = 25
ITERS = 100

CONFIGS = [
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
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
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
    # Match hyaloid's safe_gate input convention closely enough for cost
    # parity: beta is a (1, T, H) bf16 (cuLA's K1 expects bf16; hyaloid's
    # kernel does the sigmoid internally — for cuLA we feed pre-sigmoided bf16
    # so kernel work is identical shape-wise).
    beta = torch.randn(1, T, H, dtype=torch.float32, device=dev).sigmoid().to(torch.bfloat16)
    A_log = torch.randn(H, dtype=torch.float32, device=dev)
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=dev)
    return q, k, v, g, beta, A_log, dt_bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_vs_hyaloid.csv"),
    )
    args = ap.parse_args()
    dev = torch.device("cuda")
    prop = torch.cuda.get_device_properties(dev)
    print(f"device={prop.name} SMs={prop.multi_processor_count} "
          f"cap={prop.major}.{prop.minor} D={D} warmup={WARMUP} iters={ITERS}",
          flush=True)

    fcsv = open(args.csv, "w", newline="")
    wr = csv.writer(fcsv)
    wr.writerow(["H", "config", "total_T", "n_seqs",
                 "cula_serial_ms", "cula_cp_auto_ms", "cula_cp_speedup"])

    for H in H_VALUES:
        print(f"\n  [H={H}]", flush=True)
        print(f"  {'config':<24s} {'T':>7s}  "
              f"{'serial_ms':>10s}  {'cp_auto_ms':>10s}  {'speedup':>8s}",
              flush=True)
        for tag, seq_lens in CONFIGS:
            T_total = sum(seq_lens)
            n_seqs = len(seq_lens)
            q, k, v, g, beta, A_log, dt_bias = make_inputs(T_total, H, dev)
            out = torch.empty_like(v)
            fin = torch.empty(n_seqs, H, D, D, dtype=torch.float32, device=dev)
            cu32 = None
            if n_seqs > 1:
                cl = [0]
                for sl in seq_lens:
                    cl.append(cl[-1] + sl)
                cu32 = torch.tensor(cl, dtype=torch.int32, device=dev)

            def run_serial():
                flash_kda_prefill(q, k, v, g, beta, scale=SCALE, out=out,
                                  A_log=A_log, dt_bias=dt_bias, lower_bound=LB,
                                  final_state=fin, cu_seqlens=cu32)

            def run_cp():
                flash_kda_prefill_cp(q, k, v, g, beta, scale=SCALE, out=out,
                                     A_log=A_log, dt_bias=dt_bias,
                                     lower_bound=LB, final_state=fin,
                                     cu_seqlens=cu32, s_split=None)

            try:
                ms_ser = bench(run_serial)
                ms_cp = bench(run_cp)
                sp = ms_ser / ms_cp if ms_cp > 0 else float("inf")
            except torch.cuda.OutOfMemoryError:
                ms_ser = ms_cp = sp = float("nan")

            print(f"  {tag:<24s} {T_total:>7d}  "
                  f"{ms_ser:>10.4f}  {ms_cp:>10.4f}  {sp:>7.2f}x", flush=True)
            wr.writerow([H, tag, T_total, n_seqs,
                         f"{ms_ser:.4f}", f"{ms_cp:.4f}", f"{sp:.2f}"])
            fcsv.flush()

            del q, k, v, g, beta, out, fin
            torch.cuda.empty_cache()

    fcsv.close()
    print(f"\nwrote {args.csv}", flush=True)


if __name__ == "__main__":
    main()

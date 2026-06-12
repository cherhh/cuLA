# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Same-card cross-check: fla chunk_kda vs flash_kda ext vs cuLA serial vs CP.

FLA invocation copied from FlashKDA/benchmarks/bench_fwd.py (use_gate_in_kernel
etc.), so the FLA side matches the methodology behind BENCHMARK_H20.md. All
implementations run prefill-from-zero + final_state at the deployment configs.

Usage: python profile/flashkda_intracard_cp/bench_fla_cross.py [--iters 20]
"""

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

try:
    import flash_kda as _fk
except ImportError:
    _fk = None
try:
    from fla.ops.kda import chunk_kda
except ImportError:
    chunk_kda = None

SCALE = 1.0 / math.sqrt(D)
LB = -5.0

CONFIGS = [
    ("TP8_h8_T32768", 8, [32768]),
    ("TP8_h8_T16384", 8, [16384]),
    ("TP8_h8_varlen_24576_8192", 8, [24576, 8192]),
    ("TP4_h16_T32768", 16, [32768]),
    ("TP4_h16_T16384", 16, [16384]),
    ("TP2_h24_T32768", 24, [32768]),
    ("Sym_h32_T32768", 32, [32768]),
]


def bench(fn, iters, warmup=5):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_fla_cross.csv"),
    )
    args = ap.parse_args()
    dev = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(dev)} "
          f"SMs={torch.cuda.get_device_properties(dev).multi_processor_count} "
          f"iters={args.iters} fla={'yes' if chunk_kda else 'NO'} "
          f"cpp={'yes' if _fk else 'NO'}", flush=True)

    fcsv = open(args.csv, "w", newline="")
    wr = csv.writer(fcsv)
    wr.writerow(["config", "impl", "ms", "speedup_vs_fla"])

    for name, H, seq_lens in CONFIGS:
        T_total = sum(seq_lens)
        n_seqs = len(seq_lens)
        torch.manual_seed(0)
        mk = lambda *s: torch.randn(*s, dtype=torch.bfloat16, device=dev)
        q, k, v, g = (mk(1, T_total, H, D) for _ in range(4))
        beta = mk(1, T_total, H)
        A_log = torch.rand(H, dtype=torch.float32, device=dev)
        dt_bias = torch.rand(H, D, dtype=torch.float32, device=dev)
        out = torch.empty_like(v)
        fin = torch.empty(n_seqs, H, D, D, dtype=torch.float32, device=dev)
        cu64 = cu32 = None
        if n_seqs > 1:
            cl = [0]
            for sl in seq_lens:
                cl.append(cl[-1] + sl)
            cu64 = torch.tensor(cl, dtype=torch.int64, device=dev)
            cu32 = cu64.to(torch.int32)

        rows = {}

        if chunk_kda is not None:
            def run_fla():
                chunk_kda(q=q, k=k, v=v, g=g, beta=beta, scale=SCALE,
                          initial_state=None, output_final_state=True,
                          use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True,
                          use_beta_sigmoid_in_kernel=True,
                          A_log=A_log, dt_bias=dt_bias, lower_bound=LB,
                          transpose_state_layout=True, cu_seqlens=cu64)
            try:
                rows["fla_chunk_kda"] = bench(run_fla, args.iters)
            except TypeError as exc:
                print(f"[{name}] fla flags unsupported ({exc}); skipping", flush=True)

        if _fk is not None:
            def run_ext():
                _fk.fwd(q, k, v, g, beta, SCALE, out, A_log, dt_bias, LB,
                        None, fin, cu64)
            rows["flash_kda_ext"] = bench(run_ext, args.iters)

        def run_serial():
            flash_kda_prefill(q, k, v, g, beta, scale=SCALE, out=out,
                              A_log=A_log, dt_bias=dt_bias, lower_bound=LB,
                              final_state=fin, cu_seqlens=cu32)
        rows["cula_serial_cute"] = bench(run_serial, args.iters)

        def run_cp():
            flash_kda_prefill_cp(q, k, v, g, beta, scale=SCALE, out=out,
                                 A_log=A_log, dt_bias=dt_bias, lower_bound=LB,
                                 final_state=fin, cu_seqlens=cu32, s_split=None)
        rows["cula_cp_auto"] = bench(run_cp, args.iters)

        base = rows.get("fla_chunk_kda")
        print(f"\n[{name}]", flush=True)
        for impl, ms in rows.items():
            sp = f"{base / ms:.2f}" if base else ""
            print(f"  {impl:<18} {ms:.3f} ms" + (f"  ({sp}x vs fla)" if sp else ""), flush=True)
            wr.writerow([name, impl, f"{ms:.3f}", sp])
        fcsv.flush()

        del q, k, v, g, beta, out, fin
        torch.cuda.empty_cache()

    fcsv.close()
    print(f"\nwrote {args.csv}", flush=True)


if __name__ == "__main__":
    main()

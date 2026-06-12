# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""S6 bench: serial cute vs cpp ext vs intracard-CP sweep -> benchmark.csv.

Configs follow the deployment settings table (user-provided FlashQLA
comparison, 2026-06-10): H := h_v per TP degree, T in {16K,32K}, plus the
varlen 24576+8192 split. CP swept at s_split in {4,8,16,32,auto}.
(H=64 exploratory sweep archived in benchmark_h64.csv.)

Usage:  python profile/flashkda_intracard_cp/bench_cp.py [--iters 10] [--csv PATH]
"""

import argparse
import csv
import os
import sys

os.environ.setdefault("CULA_FLASHKDA_USE_CUTE", "1")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import torch  # noqa: E402

from cula.ops.flashkda.cp import (  # noqa: E402
    AUTO_MIN_SEG_TILES,
    _auto_s_split,
    _plan_segments,
    flash_kda_prefill_cp,
)
from cula.ops.flashkda.k2 import CHUNK, D  # noqa: E402
from cula.ops.flashkda.prefill import flash_kda_prefill  # noqa: E402

try:
    import flash_kda as _fk
except ImportError:
    _fk = None

SCALE = D**-0.5
S_SPLITS = (4, 8, 16, 32, None)  # None = auto


def bench(fn, iters, warmup=3):
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


def make_inputs(T_total, H, device, seed=0):
    torch.manual_seed(seed)
    mk = lambda *shape: torch.randn(*shape, dtype=torch.bfloat16, device=device)
    q, k, v, g = (mk(1, T_total, H, D) for _ in range(4))
    beta = mk(1, T_total, H)
    A_log = torch.randn(H, dtype=torch.float32, device=device)
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=device)
    out = torch.empty_like(v)
    fin = torch.empty(0)  # placeholder, sized per config below
    return q, k, v, g, beta, A_log, dt_bias, out, fin


def plan_info(device, seq_lens, H, s_split):
    seq_tiles = [sl // CHUNK for sl in seq_lens]
    if s_split is None:
        eff = _auto_s_split(device, len(seq_tiles), H)
        seg_cu, _ = _plan_segments(seq_tiles, eff, AUTO_MIN_SEG_TILES)
    else:
        eff = s_split
        seg_cu, _ = _plan_segments(seq_tiles, eff)
    return eff, len(seg_cu) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--auto-only", action="store_true",
                    help="only serial + cp[auto] rows (skip cpp + s_split sweep)")
    ap.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.csv"),
    )
    args = ap.parse_args()
    s_splits = (None,) if args.auto_only else S_SPLITS
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    print(f"device={torch.cuda.get_device_name(dev)} SMs={sms} D={D} "
          f"iters={args.iters} cpp={'yes' if _fk else 'NO'}", flush=True)

    # (name, H=h_v, seq_lens, mode) from the deployment table.
    # h_qk<h_v rows are GQA; cuLA kernels share H across q/k/v/g, so H:=h_v is
    # the state/v-side-equivalent proxy (overestimates q/k traffic for TP8/TP4).
    # TP1 2B (h16, 1x32768) coincides with the TP4 32K row — not duplicated.
    configs = [
        ("TP8_h8_T32768", 8, [32768], None),
        ("TP8_h8_T16384", 8, [16384], None),
        ("TP8_h8_varlen_24576_8192", 8, [24576, 8192], "varlen"),
        ("TP4_h16_T32768", 16, [32768], None),
        ("TP4_h16_T16384", 16, [16384], None),
        ("TP2_h24_T32768", 24, [32768], None),
        ("Sym_h32_T32768", 32, [32768], None),
    ]

    fcsv = open(args.csv, "w", newline="")
    wr = csv.writer(fcsv)
    wr.writerow(["config", "impl", "s_split_req", "s_split_eff", "n_seg_total",
                 "ms", "speedup_vs_serial_cute", "note"])

    for name, H, seq_lens, mode in configs:
        T_total = sum(seq_lens)
        n_seqs = len(seq_lens)
        q, k, v, g, beta, A_log, dt_bias, out, _ = make_inputs(T_total, H, dev)
        fin = torch.empty(n_seqs, H, D, D, dtype=torch.float32, device=dev)
        cu = None
        if mode == "varlen":
            cu_list = [0]
            for sl in seq_lens:
                cu_list.append(cu_list[-1] + sl)
            cu = torch.tensor(cu_list, dtype=torch.int32, device=dev)

        def run_serial():
            flash_kda_prefill(q, k, v, g, beta, scale=SCALE, out=out,
                              A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
                              final_state=fin, cu_seqlens=cu)

        t_ser = bench(run_serial, args.iters)
        print(f"\n[{name}] serial-cute {t_ser:.3f} ms", flush=True)
        wr.writerow([name, "serial-cute", "", "", "", f"{t_ser:.3f}", "1.00", ""])
        fcsv.flush()

        if _fk is not None and not args.auto_only:
            cu64 = cu.long() if cu is not None else None  # cpp ext requires int64
            try:
                def run_cpp():
                    _fk.fwd(q, k, v, g, beta, SCALE, out, A_log, dt_bias, -5.0,
                            None, fin, cu64)
                t_cpp = bench(run_cpp, args.iters)
                print(f"[{name}] cpp        {t_cpp:.3f} ms ({t_ser / t_cpp:.2f}x)", flush=True)
                wr.writerow([name, "cpp", "", "", "", f"{t_cpp:.3f}",
                             f"{t_ser / t_cpp:.2f}", ""])
            except Exception as exc:  # noqa: BLE001
                print(f"[{name}] cpp        ERR {type(exc).__name__}: {exc}", flush=True)
                wr.writerow([name, "cpp", "", "", "", "", "", f"ERR {type(exc).__name__}"])
            fcsv.flush()

        for ss in s_splits:
            eff, n_seg = plan_info(dev, seq_lens, H, ss)
            tag = f"cp[{ss if ss is not None else f'auto={eff}'}]"
            note = "delegates-to-serial" if n_seg == n_seqs else ""
            try:
                def run_cp():
                    flash_kda_prefill_cp(q, k, v, g, beta, scale=SCALE, out=out,
                                         A_log=A_log, dt_bias=dt_bias,
                                         lower_bound=-5.0, final_state=fin,
                                         cu_seqlens=cu, s_split=ss)
                t_cp = bench(run_cp, args.iters)
                print(f"[{name}] {tag:<13} {t_cp:.3f} ms ({t_ser / t_cp:.2f}x) "
                      f"segs={n_seg} {note}", flush=True)
                wr.writerow([name, tag, ss if ss is not None else "auto", eff, n_seg,
                             f"{t_cp:.3f}", f"{t_ser / t_cp:.2f}", note])
            except Exception as exc:  # noqa: BLE001
                print(f"[{name}] {tag:<13} ERR {type(exc).__name__}: {exc}", flush=True)
                wr.writerow([name, tag, ss if ss is not None else "auto", eff, n_seg,
                             "", "", f"ERR {type(exc).__name__}"])
            fcsv.flush()

        del q, k, v, g, beta, out, fin
        torch.cuda.empty_cache()

    fcsv.close()
    print(f"\nwrote {args.csv}", flush=True)


if __name__ == "__main__":
    main()

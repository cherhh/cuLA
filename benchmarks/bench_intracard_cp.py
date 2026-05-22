#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""Intracard-CP benchmark — end-to-end chunk_kda.

Measures the speedup of cuLA's intracard context-parallel path against the
non-CP baseline across a range of varlen configurations.  Also verifies that
the heuristic does not regress throughput when CP is correctly bypassed.

Usage:
    python benchmarks/bench_intracard_cp.py
    python benchmarks/bench_intracard_cp.py --warmup 5 --n-iters 50
    python benchmarks/bench_intracard_cp.py --ncu
"""

from __future__ import annotations

import argparse
import contextlib
import os
import pathlib
import sys
from dataclasses import dataclass

os.environ.setdefault("CULA_INTRACARD_CP", "1")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cula.ops.cp.chunk_delta_h import (  # noqa: E402
    compute_subseq_len,
    prepare_subseq_cu_seqlens,
    should_use_intracard_cp,
)
from cula.utils import get_device_sm_count  # noqa: E402

BT, K, V, D = 64, 128, 128, 128

WARMUP = 10
N_ITERS = 100
NCU_MODE = False  # set by --ncu; forces warmup=1, n_iters=1


# ============================== env toggle ==============================


@contextlib.contextmanager
def cp_on(enable: bool):
    old = os.environ.get("CULA_INTRACARD_CP")
    os.environ["CULA_INTRACARD_CP"] = "1" if enable else "0"
    try:
        if enable:
            with torch.inference_mode():
                yield
        else:
            yield
    finally:
        if old is None:
            os.environ.pop("CULA_INTRACARD_CP", None)
        else:
            os.environ["CULA_INTRACARD_CP"] = old


# ============================== inputs ==============================


def make_inputs(seq_lens, H, seed=42, device="cuda", dtype=torch.bfloat16):
    total = sum(seq_lens)
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)
    torch.manual_seed(seed)
    q = torch.randn(1, total, H, D, dtype=dtype, device=device)
    k = F.normalize(torch.randn(1, total, H, D, dtype=torch.float32, device=device), p=2, dim=-1).to(dtype)
    v = torch.randn(1, total, H, D, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(1, total, H, D, dtype=torch.float32, device=device)).clamp(-5, 0)
    beta = torch.randn(1, total, H, dtype=torch.float32, device=device).sigmoid()
    cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
    return q, k, v, g, beta, cu_t


# ============================== bench harness ==============================


def time_kernel(fn, warmup, n_iters) -> float:
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
    return start.elapsed_time(end) / n_iters


def run_chunk_kda(q, k, v, g, beta, cu, *, enable_cp: bool) -> None:
    from cula.kda.chunk_fwd import chunk_kda_fwd

    scale = D**-0.5
    with cp_on(enable_cp):
        chunk_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu.cpu(),
            safe_gate=True,
            lower_bound=-5.0,
            use_gate_in_kernel=False,
        )


# ============================== strategy predict ==============================


def predict_cp(seq_lens, H, num_sms):
    cu = torch.tensor(
        [0] + list(torch.tensor(seq_lens).cumsum(0).tolist()),
        dtype=torch.int32,
    )
    if not should_use_intracard_cp(cu, num_sms, H, BT):
        return False, 0
    max_len = int(torch.diff(cu).max().item())
    subseq_len = compute_subseq_len(max_len, num_sms, H, BT, num_seqs=len(seq_lens))
    _, split_info, total_subseqs = prepare_subseq_cu_seqlens(cu, subseq_len, BT)
    return bool(split_info), total_subseqs


# ============================== configs ==============================

# (tag, seq_lens) — each entry is tested at every H in H_VALUES
CONFIGS = [
    # --- single seq (ascending length) ---
    ("T=4K", [4096]),
    ("T=8K", [8192]),
    ("T=32K", [32768]),
    ("T=64K", [65536]),
    ("T=128K", [131072]),
    # --- equal-length batches (~32K total) ---
    ("8x4K", [4096] * 8),
    ("4x8K", [8192] * 4),
    ("2x16K", [16384] * 2),
    # --- asymmetric multi-seq ---
    ("16K+16K", [16384, 16384]),
    ("24K+8K", [24576, 8192]),
    ("28K+4K", [28672, 4096]),
    ("32K+256+256", [32768, 256, 256]),
    ("40K+1K+8K", [40960, 1024, 8192]),
    ("64K+512+256+128", [65536, 512, 256, 128]),
    ("128K+1K", [131072, 1024]),
    # --- 128K + several short seqs ---
    ("128K+2x1K", [131072, 1024, 1024]),
    ("128K+5x1K", [131072] + [1024] * 5),
    ("128K+10x1K", [131072] + [1024] * 10),
]

H_VALUES = [4, 8]


# ============================== row + report ==============================


@dataclass
class Row:
    tag: str
    H: int
    total_T: int
    pred: bool
    n_sub: int
    ms_off: float
    ms_on: float

    @property
    def speedup(self) -> float:
        return self.ms_off / self.ms_on


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--n-iters", type=int, default=None, dest="n_iters")
    ap.add_argument("--ncu", action="store_true", help="NCU mode: warmup=1, n_iters=1")
    args = ap.parse_args()

    global NCU_MODE
    NCU_MODE = args.ncu

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    num_sms = get_device_sm_count(device)

    warmup = 1 if NCU_MODE else (args.warmup or WARMUP)
    n_iters = 1 if NCU_MODE else (args.n_iters or N_ITERS)

    print(f"Device: {torch.cuda.get_device_name(device)} (SM={num_sms})")
    print(f"Bench : warmup={warmup}, n_iters={n_iters}")
    print()

    hdr = f"{'config':<24s} {'T':>7} {'pred':>4} {'sub':>4}  {'CP_off':>8}  {'CP_on':>8}  {'speedup':>8}"
    sep = "-" * len(hdr)

    all_rows: list[Row] = []
    for H in H_VALUES:
        print(f"--- H={H} ---")
        print(hdr)
        print(sep)
        for tag, seq_lens in CONFIGS:
            pred, n_sub = predict_cp(seq_lens, H, num_sms)
            q, k, v, g, beta, cu = make_inputs(seq_lens, H)
            ms_off = time_kernel(lambda: run_chunk_kda(q, k, v, g, beta, cu, enable_cp=False), warmup, n_iters)
            ms_on = time_kernel(lambda: run_chunk_kda(q, k, v, g, beta, cu, enable_cp=True), warmup, n_iters)
            r = Row(tag=tag, H=H, total_T=sum(seq_lens), pred=pred, n_sub=n_sub, ms_off=ms_off, ms_on=ms_on)
            all_rows.append(r)
            pred_s = "Y" if pred else "N"
            print(
                f"{r.tag:<24s} {r.total_T:>7}    {pred_s}  {r.n_sub:>4d}  "
                f"{r.ms_off:>8.3f}  {r.ms_on:>8.3f}  {r.speedup:>7.2f}x"
            )
        print()

    triggered = [r for r in all_rows if r.pred]
    bypassed = [r for r in all_rows if not r.pred]

    if triggered:
        speedups = [r.speedup for r in triggered]
        geo = 1.0
        for s in speedups:
            geo *= s
        geo = geo ** (1 / len(speedups))
        print(
            f"CP triggered ({len(triggered)} configs): "
            f"geo-mean={geo:.2f}x  best={max(speedups):.2f}x  worst={min(speedups):.2f}x"
        )

    if bypassed:
        ratios = [r.ms_on / r.ms_off for r in bypassed]
        print(
            f"CP bypassed  ({len(bypassed)} configs): "
            f"mean overhead={sum(ratios) / len(ratios):.3f}x  max={max(ratios):.3f}x  "
            f"(1.00 = no regression)"
        )


if __name__ == "__main__":
    main()

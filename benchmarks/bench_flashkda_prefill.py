# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark FlashKDA prefill: CuteDSL wrapper vs FlashKDA C++.

Until the CuteDSL K1/K2 ports land, the wrapper currently delegates to the
FlashKDA C++ extension when ``CULA_FLASHKDA_USE_CUTE=1`` is set, so the two
columns of the table will report similar timings. As individual phases of K1
and K2 are migrated to CuteDSL, the delta will grow.

Usage:
    CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_flashkda_prefill.py
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "FlashKDA"))

from cula.ops.flashkda_prefill import D as HEAD_DIM
from cula.ops.flashkda_prefill import flash_kda_prefill

try:
    import flash_kda  # noqa: F401  (presence check)
    HAS_FLASH_KDA = True
except Exception as e:
    print(f"[warn] flash_kda not importable: {e!r}", file=sys.stderr)
    HAS_FLASH_KDA = False


# (B, T, H) sweep. T is total tokens / B. Keep H small for fast sweeps.
SHAPES = [
    (1, 512, 4),
    (1, 1024, 4),
    (1, 2048, 4),
    (1, 4096, 4),
    (2, 2048, 8),
    (4, 1024, 8),
]
WARMUP_ITERS = 5
TIMING_ITERS = 20


def _make_inputs(B: int, T: int, H: int, device: str = "cuda"):
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(B, T, H, HEAD_DIM, generator=g, device=device, dtype=torch.bfloat16) * 0.5
    k = torch.randn(B, T, H, HEAD_DIM, generator=g, device=device, dtype=torch.bfloat16) * 0.5
    v = torch.randn(B, T, H, HEAD_DIM, generator=g, device=device, dtype=torch.bfloat16) * 0.5
    g_pre = torch.randn(B, T, H, HEAD_DIM, generator=g, device=device, dtype=torch.bfloat16) * 0.1
    beta = torch.randn(B, T, H, generator=g, device=device, dtype=torch.bfloat16) * 0.1
    A_log = torch.randn(H, generator=g, device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H, HEAD_DIM, generator=g, device=device, dtype=torch.float32) * 0.1
    return q, k, v, g_pre, beta, A_log, dt_bias


def _time_us(fn, *, warmup: int = WARMUP_ITERS, iters: int = TIMING_ITERS) -> float:
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # ms -> us


def main():
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Head dim: {HEAD_DIM}, dtype=bf16")
    print()
    header = f"{'B':>3} {'T':>6} {'H':>3} {'cute (us)':>12} {'cpp (us)':>12} {'speedup':>10}"
    print(header)
    print("-" * len(header))

    scale = HEAD_DIM ** -0.5
    lower_bound = -5.0

    for B, T, H in SHAPES:
        q, k, v, g_pre, beta, A_log, dt_bias = _make_inputs(B, T, H)
        out_cute = torch.empty_like(v)
        out_cpp = torch.empty_like(v)

        os.environ["CULA_FLASHKDA_USE_CUTE"] = "1"
        # Re-import to pick up env var
        import importlib
        import cula.ops.flashkda_prefill as _mod
        importlib.reload(_mod)

        fn_cute = lambda: _mod.flash_kda_prefill(
            q, k, v, g_pre, beta, scale, out_cute, A_log, dt_bias, lower_bound
        )

        try:
            t_cute = _time_us(fn_cute)
        except Exception as e:
            t_cute = float("nan")
            print(f"[warn] cute path failed for ({B},{T},{H}): {e!r}", file=sys.stderr)

        if HAS_FLASH_KDA:
            fn_cpp = lambda: flash_kda.fwd(
                q, k, v, g_pre, beta, scale, out_cpp, A_log, dt_bias, lower_bound
            )
            t_cpp = _time_us(fn_cpp)
        else:
            t_cpp = float("nan")

        speedup = (t_cpp / t_cute) if t_cute > 0 and t_cpp == t_cpp else float("nan")
        print(f"{B:>3} {T:>6} {H:>3} {t_cute:>12.2f} {t_cpp:>12.2f} {speedup:>10.3f}x")

    os.environ.pop("CULA_FLASHKDA_USE_CUTE", None)


if __name__ == "__main__":
    main()

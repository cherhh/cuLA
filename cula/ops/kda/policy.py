# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel dispatch policy for KDA wrappers."""

from __future__ import annotations

import os
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

from cula.ops.kda.sm90.cp.plan import CHUNK as SM90_CP_CHUNK
from cula.ops.kda.sm90.cp.plan import CP_ENGAGE_MARGIN, auto_plan_segments, estimate_cp_speedup

IntracardCPMode = Literal["auto"] | bool


@dataclass(frozen=True)
class IntracardCPDecision:
    enabled: bool
    reason: str | None = None
    force: bool = False


class NotSplittableError(ValueError):
    """Raised when intracard CP cannot meaningfully split the given shape.

    Subclasses ValueError so existing ``except ValueError`` callers keep working,
    while new code can catch it narrowly and fall back to the serial path.
    """


def normalize_intracard_cp_mode(mode: IntracardCPMode) -> IntracardCPMode:
    # Identity checks (not `in`): `1 == True` / `0 == False` would match stray ints.
    if mode != "auto" and mode is not True and mode is not False:
        raise ValueError(f'use_intracard_cp must be "auto", True, or False, got {mode!r}')
    return mode


def resolve_intracard_cp_mode(
    use_intracard_cp: IntracardCPMode | None,
    use_cp_alias: IntracardCPMode | None,
) -> IntracardCPMode | None:
    if use_intracard_cp is not None and use_cp_alias is not None:
        raise TypeError("Pass only one of use_intracard_cp or use_cp.")
    mode = use_intracard_cp if use_intracard_cp is not None else use_cp_alias
    if mode is None:
        return None
    return normalize_intracard_cp_mode(mode)


def _reject_or_disable(mode: IntracardCPMode, reason: str) -> IntracardCPDecision:
    if mode is True:
        raise ValueError(reason)
    return IntracardCPDecision(False, reason)


_SEQ_LENS_CACHE: dict = {}


def _seq_lens_from_cu(cu_seqlens: torch.Tensor, cu_seqlens_cpu: torch.Tensor | None) -> list[int]:
    """Per-sequence lengths from cu_seqlens, cached by tensor identity so the auto router
    pays a GPU->host sync only on a cache miss, not on every decision. Passing
    cu_seqlens_cpu avoids the sync entirely even on a miss."""
    key = id(cu_seqlens)
    stamp = (cu_seqlens.data_ptr(), int(cu_seqlens._version), cu_seqlens.numel())
    cached = _SEQ_LENS_CACHE.get(key)
    if cached is not None:
        ref, cstamp, seq_lens = cached
        if ref() is cu_seqlens and cstamp == stamp:
            return seq_lens
        _SEQ_LENS_CACHE.pop(key, None)
    src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.cpu()
    cu_list = [int(x) for x in src.tolist()]
    seq_lens = [cu_list[i + 1] - cu_list[i] for i in range(len(cu_list) - 1)]
    if len(_SEQ_LENS_CACHE) >= 32:
        _SEQ_LENS_CACHE.pop(next(iter(_SEQ_LENS_CACHE)))
    _SEQ_LENS_CACHE[key] = (weakref.ref(cu_seqlens), stamp, seq_lens)
    return seq_lens


def _sm90_seq_tiles(
    q: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    mode: IntracardCPMode,
) -> list[int] | IntracardCPDecision:
    # Non-CHUNK-aligned lengths are supported by the backend (pad-before-CP), so the
    # per-sequence tile count is the ceil; no alignment rejection here.
    B, T, _H, _K = q.shape
    if cu_seqlens is None:
        return [(T + SM90_CP_CHUNK - 1) // SM90_CP_CHUNK] * B

    if B != 1:
        return _reject_or_disable(mode, "SM90 intracard CP varlen mode requires packed B=1.")
    seq_lens = _seq_lens_from_cu(cu_seqlens, cu_seqlens_cpu)
    return [(sl + SM90_CP_CHUNK - 1) // SM90_CP_CHUNK for sl in seq_lens]


def sm90_intracard_cp_decision(
    q: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    mode: IntracardCPMode | None,
) -> IntracardCPDecision:
    # SM90 has no env-gated legacy default; unspecified (None) means CP off.
    if mode is None:
        mode = False
    mode = normalize_intracard_cp_mode(mode)
    if mode is False:
        return IntracardCPDecision(False, "disabled")

    seq_tiles_or_decision = _sm90_seq_tiles(q, cu_seqlens, cu_seqlens_cpu, mode)
    if isinstance(seq_tiles_or_decision, IntracardCPDecision):
        return seq_tiles_or_decision
    seq_tiles = seq_tiles_or_decision
    if not seq_tiles:
        return _reject_or_disable(mode, "SM90 intracard CP requires at least one sequence.")

    _s_split, seg_cu, per_seq = auto_plan_segments(q.device, seq_tiles, q.shape[2])
    n_seg_total = len(seg_cu) - 1
    max_n_seg = max(n_seg for _first, n_seg in per_seq)
    if n_seg_total == len(seq_tiles) or max_n_seg <= 2:
        return _reject_or_disable(
            mode,
            "SM90 intracard CP is not meaningfully splittable for this shape.",
        )
    # auto-only gate: engage only when the calibrated cost model predicts the
    # CP pipeline (pre_scan + merge + segment-K2) beats the serial K2 chain by
    # at least CP_ENGAGE_MARGIN. force (True) still runs: the shape IS splittable.
    if mode is not True:
        speedup = estimate_cp_speedup(q.device, seq_tiles, seg_cu, per_seq, q.shape[2])
        if speedup < CP_ENGAGE_MARGIN:
            return IntracardCPDecision(
                False,
                f"intracard CP not beneficial: predicted {speedup:.2f}x (< {CP_ENGAGE_MARGIN:.2f}x margin)",
            )
    return IntracardCPDecision(True)


def _sm100_env_cp_enabled() -> bool:
    return os.environ.get("CULA_INTRACARD_CP", "0") != "0"


def sm100_intracard_cp_decision(
    *,
    mode: IntracardCPMode | None,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    g: torch.Tensor | None,
    num_qk_heads: int,
    chunk_size: int,
    is_inference: bool,
    sm_count_provider: Callable[[], int],
    no_cp: bool = False,
) -> IntracardCPDecision:
    if mode is None:
        mode = "auto" if _sm100_env_cp_enabled() else False
    mode = normalize_intracard_cp_mode(mode)
    # no_cp is the recursion guard: intracard_fwd_h re-invokes fwd_h with _no_cp=True
    # so sub-sequences do not recursively re-trigger CP.
    if mode is False or no_cp:
        return IntracardCPDecision(False, "disabled")

    if cu_seqlens is None:
        return _reject_or_disable(mode, "SM100 intracard CP requires varlen cu_seqlens.")
    if g is not None:
        return _reject_or_disable(mode, "SM100 intracard CP requires g is None; pass gate through gk.")
    if not is_inference:
        return _reject_or_disable(mode, "SM100 intracard CP is inference-only.")

    if mode is True:
        return IntracardCPDecision(True, force=True)

    # auto: consult the CPU-only perf heuristic.
    from cula.ops.kda.sm100.cp.chunk_delta_h import should_use_intracard_cp

    cpu = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.cpu()
    if should_use_intracard_cp(cpu, sm_count_provider(), num_qk_heads, chunk_size):
        return IntracardCPDecision(True)
    return IntracardCPDecision(False, "SM100 intracard CP heuristic declined for this shape.")

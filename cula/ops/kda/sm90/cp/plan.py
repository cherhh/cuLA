# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Segment planning for SM90 KDA intracard context-parallel (CP) prefill.

CP re-runs the recurrence (pre_scan + segment-K2) to cut the serial-K2
critical path, so a split must shrink the longest chain by more than the
re-run cost. plan_* return the plan the executor runs verbatim; a trivial
plan (nothing split) means "serial path".

Decisions use whole tiles + SM count. The two constants below are chain-cost
ratios fitted on H100; ratios of same-species chains cancel absolute speed,
so no per-SKU recalibration.
"""

from __future__ import annotations

import os
import weakref
from dataclasses import dataclass

import torch

from cula.ops.kda.cp_mode import CPMode, NotSplittableError
from cula.utils import get_device_sm_count

CHUNK = 16

# Manual-plan floor (s_split given): keeps hand-crafted splits non-degenerate.
MIN_SEG_TILES = 4

# Floor on auto segment length, tuned on H100.
AUTO_MIN_SEG_TILES = int(os.environ.get("CULA_KDA_CP_AUTO_MIN_SEG_TILES", "32"))

# Per-tile cost of the CP re-run (pre_scan + segment-K2) relative to the
# serial K2 chain, tuned on H100: splitting engages only when the critical
# path shrinks by more than this.
RERUN_RATIO = float(os.environ.get("CULA_KDA_CP_RERUN_RATIO", "3"))


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class CPPlan:
    """Segmentation in whole tiles: segment i covers packed tile range
    [seg_cu[i], seg_cu[i+1]); per_seq[s] = (first_segment, n_segments).
    Trivial (nothing split) means "serial path"; reason says why."""

    seq_tiles: tuple[int, ...]
    seg_cu: tuple[int, ...]
    per_seq: tuple[tuple[int, int], ...]
    reason: str = ""

    @classmethod
    def serial(cls, seq_tiles=(), reason: str = "") -> CPPlan:
        return cls(tuple(seq_tiles), (0,), (), reason)

    @property
    def trivial(self) -> bool:
        return self.n_seg_total <= self.n_seqs

    @property
    def n_seqs(self) -> int:
        return len(self.seq_tiles)

    @property
    def n_seg_total(self) -> int:
        return len(self.seg_cu) - 1

    @property
    def total_tiles(self) -> int:
        return self.seg_cu[-1]

    @property
    def seg_tiles(self) -> tuple[int, ...]:
        return tuple(self.seg_cu[i + 1] - self.seg_cu[i] for i in range(self.n_seg_total))

    @property
    def max_seg_tiles(self) -> int:
        return max(self.seg_tiles, default=0)


def _materialize(seq_tiles: list[int], n_segs: list[int]) -> CPPlan:
    """Split each sequence into its requested number of near-equal segments."""
    seg_cu = [0]
    per_seq = []
    for tiles, n in zip(seq_tiles, n_segs):
        n = min(n, tiles)  # zero-tile sequences get zero segments
        first = len(seg_cu) - 1
        base, rem = divmod(tiles, max(n, 1))
        for i in range(n):
            seg_cu.append(seg_cu[-1] + base + (1 if i < rem else 0))
        per_seq.append((first, n))
    return CPPlan(tuple(seq_tiles), tuple(seg_cu), tuple(per_seq))


def split_balanced(seq_tiles: list[int], H: int, sm_count: int) -> CPPlan:
    """One-wave split: target segment length spreads the total load over the
    chain slots (sm_count // H); each sequence gets ceil(tiles/target)
    segments. Equal lengths (not counts) minimize the critical path; a full
    machine pushes the target past every sequence, so nothing splits."""
    parallel_segs = max(1, sm_count // H)  # segments the card can run at once
    target_seg_tiles = max(AUTO_MIN_SEG_TILES, _ceil_div(sum(seq_tiles), parallel_segs))
    n_segs = [max(1, min(_ceil_div(tiles, target_seg_tiles), tiles // AUTO_MIN_SEG_TILES)) for tiles in seq_tiles]
    return _materialize(seq_tiles, n_segs)


def plan_auto(seq_tiles: list[int], H: int, sm_count: int) -> CPPlan:
    """split_balanced + profitability: wall = max(critical chain, per-slot
    load) on both sides, with CP tiles costing RERUN_RATIO more; engage only
    when the serial wall exceeds the CP wall.

    Example: seq_tiles = [896] + [8] * 16, H=16, 132 SMs:
      parallel_segs    = 132 // 16 = 8 segments running at once
      target_seg_tiles = max(32, ceil(1024 / 8)) = 128
      split  = 896 -> 7 segments of 128; the 8-tile sequences stay whole
      engage = serial wall max(896, 128) >= CP wall 3 * max(128, 128) -> CP

    Counter-example: seq_tiles = [64], H=16 -> 2 segments of 32, but
    serial wall 64 < CP wall 3 * 32 = 96 -> trivial plan (serial path).
    """
    if not seq_tiles:
        return CPPlan.serial(seq_tiles, "empty input")
    plan = split_balanced(seq_tiles, H, sm_count)
    if plan.trivial:
        return CPPlan.serial(seq_tiles, "machine already full or sequences too short to split")
    parallel_segs = max(1, sm_count // H)
    # No schedule finishes before the total work spread over the parallel
    # slots, nor before its longest chain: wall = max of the two bounds.
    load_bound = _ceil_div(sum(seq_tiles), parallel_segs)
    serial_wall = max(max(seq_tiles), load_bound)
    cp_wall = RERUN_RATIO * max(plan.max_seg_tiles, load_bound)
    if serial_wall < cp_wall:
        return CPPlan.serial(
            seq_tiles,
            f"serial wall {serial_wall} tiles vs CP wall ~{cp_wall:g}: "
            f"splitting does not pay the {RERUN_RATIO:g}x re-run cost",
        )
    return plan


def plan_manual(seq_tiles: list[int], s_split: int) -> CPPlan:
    """Up to s_split near-equal segments per sequence; no profitability
    judgment (tests, experiments)."""
    n_segs = [max(1, min(s_split, tiles // max(1, MIN_SEG_TILES))) for tiles in seq_tiles]
    return _materialize(seq_tiles, n_segs)


_SEQ_LENS_CACHE: dict = {}


def _seq_lens_from_cu(cu_seqlens: torch.Tensor, cu_seqlens_cpu: torch.Tensor | None) -> list[int]:
    """Per-sequence lengths from cu_seqlens, cached by tensor identity so the
    planner pays a GPU->host sync only on a cache miss, not on every call.
    Passing cu_seqlens_cpu avoids the sync entirely even on a miss."""
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


def plan_prefill(
    q: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    mode: CPMode | None = None,
    *,
    _seq_tiles: list[int] | None = None,
) -> CPPlan:
    """Planning entry for the prefill wrapper. Trivial plan -> serial path.
    AUTO declines unprofitable splits; FORCE splits whenever possible and
    raises NotSplittableError otherwise."""
    if mode is None or mode is CPMode.OFF:
        return CPPlan.serial((), "disabled")
    B, T, H, _K = q.shape
    if _seq_tiles is not None:
        seq_tiles = list(_seq_tiles)
    elif cu_seqlens is None:
        # Non-CHUNK-aligned lengths run CP via pad-to-tile, hence ceil tiles.
        seq_tiles = [_ceil_div(T, CHUNK)] * B
    elif B != 1:
        if mode is CPMode.FORCE:
            raise NotSplittableError("SM90 intracard CP varlen mode requires packed B=1.")
        return CPPlan.serial((), "varlen requires packed B=1")
    else:
        seq_lens = _seq_lens_from_cu(cu_seqlens, cu_seqlens_cpu)
        seq_tiles = [_ceil_div(sl, CHUNK) for sl in seq_lens]

    if mode is CPMode.FORCE:
        if not seq_tiles:
            raise NotSplittableError("SM90 intracard CP requires at least one sequence.")
        plan = split_balanced(seq_tiles, H, get_device_sm_count(q.device))
        if plan.trivial:
            raise NotSplittableError("SM90 intracard CP cannot split this shape.")
        return plan
    return plan_auto(seq_tiles, H, get_device_sm_count(q.device))

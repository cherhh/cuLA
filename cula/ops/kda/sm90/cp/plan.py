# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Segment planning for SM90 KDA intracard context-parallel (CP) prefill.

CP shortens the K2 critical path by splitting sequences into segments that
recur in parallel, at the price of re-running the recurrence (pre_scan +
segment-K2 instead of one serial K2). The planner answers "how to split" and
"whether to" in one step: it returns the plan the executor runs verbatim, and
a *trivial* plan (nothing split) means "take the serial path".

All decisions are dimensionless -- whole tiles, plus the GPU's SM count. The
engage rule needs no calibrated time constants because it only compares
same-species MMA chains against each other, so it holds across Hopper SKUs.
"""

from __future__ import annotations

import os
import warnings
import weakref
from dataclasses import dataclass

import torch

from cula.ops.kda.cp_mode import CPMode, NotSplittableError
from cula.utils import get_device_sm_count

CHUNK = 16

# Lenient floor for *manual* plans (s_split given): just enough to keep
# hand-crafted test splits non-degenerate. Not a tunable.
MIN_SEG_TILES = 4

# ---------------------------------------------------------------------------
# Decision constants. Exactly two dimensionless numbers enter the engage
# decision (plus the runtime SM count). Both are RATIOS of the same species
# of dependent-MMA chain, so absolute chain speed cancels and they hold
# across Hopper SKUs. Derivations use the H100 SXM e2e-pipeline chain fits
# (CUDA-event timing, 2026-07; "fixed" is the EFFECTIVE per-launch cost --
# it includes the merge step and inter-kernel gaps, not just the isolated
# kernel intercept):
#   serial K2   1.48 us/tile + 84 us fixed
#   pre_scan    2.39 us/tile + 27 us fixed
#   segment-K2  1.69 us/tile + 19 us fixed
# ---------------------------------------------------------------------------

# Floor on auto segment length. Each segment restarts its chain cold (TMA
# warmup, state init, fp32 state epilogue, plus its share of merge and
# launch gaps): (27/2.39 + 19/1.69) ~= 22 tiles of effective chain work
# across the two CP passes; 32 rounds up with headroom so segments always
# amortize their cold start. Validated behaviorally by the local engage-
# boundary probes rather than by intercept fits.
AUTO_MIN_SEG_TILES = int(os.environ.get("CULA_KDA_CP_AUTO_MIN_SEG_TILES", "32"))

# CP re-runs the recurrence: pre_scan + segment-K2 cost (2.39 + 1.69) / 1.48
# = 2.76x the serial K2 chain per tile; 3 rounds up to absorb the merge step
# and fit error. Splitting pays off only when the critical path shrinks by
# more than this.
RERUN_RATIO = float(os.environ.get("CULA_KDA_CP_RERUN_RATIO", "3"))

_REMOVED_KNOBS = sorted(
    k
    for k in os.environ
    if k.startswith("CULA_KDA_CP_COST_")
    or k
    in (
        "CULA_KDA_CP_ENGAGE_MARGIN",
        "CULA_KDA_CP_MIN_SEG",
        "CULA_KDA_CP_ENGAGE_MIN_TILES",
        "CULA_KDA_CP_MIN_SEG_TILES",
    )
)
if _REMOVED_KNOBS:
    warnings.warn(
        f"{_REMOVED_KNOBS} no longer have any effect: the calibrated CP cost model "
        "was replaced by the dimensionless engage rule (see CULA_KDA_CP_RERUN_RATIO).",
        stacklevel=2,
    )


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class CPPlan:
    """How sequences split into segments, in whole tiles.

    Segments are numbered globally in packed order; segment ``i`` covers the
    tile range ``[seg_cu[i], seg_cu[i+1])``. ``per_seq[s]`` is
    ``(first_segment, n_segments)`` for sequence ``s``. A trivial plan means
    "run the serial path"; ``reason`` says why.
    """

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
    for r, n in zip(seq_tiles, n_segs):
        n = min(n, r)  # zero-tile sequences get zero segments
        first = len(seg_cu) - 1
        base, rem = divmod(r, max(n, 1))
        for i in range(n):
            seg_cu.append(seg_cu[-1] + base + (1 if i < rem else 0))
        per_seq.append((first, n))
    return CPPlan(tuple(seq_tiles), tuple(seg_cu), tuple(per_seq))


def split_balanced(seq_tiles: list[int], H: int, sm_count: int) -> CPPlan:
    """Split so equal-length segment chains fill one SM wave.

    ``target`` is the global segment length that spreads the total load over
    the machine's chain slots (SM count // H); every sequence gets
    ``ceil(tiles / target)`` segments, floored so segments amortize their cold
    start. Balancing segment lengths (not counts) minimizes the critical path,
    and a full machine pushes ``target`` past every sequence so nothing splits.
    """
    slots = max(1, sm_count // H)
    target = max(AUTO_MIN_SEG_TILES, _ceil_div(sum(seq_tiles), slots))
    n_segs = [max(1, min(_ceil_div(r, target), r // AUTO_MIN_SEG_TILES)) for r in seq_tiles]
    return _materialize(seq_tiles, n_segs)


def plan_auto(seq_tiles: list[int], H: int, sm_count: int) -> CPPlan:
    """split_balanced plus the profitability judgment.

    Both paths are chain-bound, so each wall is max(critical chain, per-slot
    load) in tiles -- with CP's chains costing RERUN_RATIO more per tile.
    Engage only when the serial wall exceeds the CP wall.
    """
    if not seq_tiles:
        return CPPlan.serial(seq_tiles, "empty input")
    plan = split_balanced(seq_tiles, H, sm_count)
    if plan.trivial:
        return CPPlan.serial(seq_tiles, "machine already full or sequences too short to split")
    slots = max(1, sm_count // H)
    load = _ceil_div(sum(seq_tiles), slots)
    serial_wall = max(max(seq_tiles), load)
    cp_wall = RERUN_RATIO * max(plan.max_seg_tiles, load)
    if serial_wall < cp_wall:
        return CPPlan.serial(
            seq_tiles,
            f"serial wall {serial_wall} tiles vs CP wall ~{cp_wall:g}: "
            f"splitting does not pay the {RERUN_RATIO:g}x re-run cost",
        )
    return plan


def plan_manual(seq_tiles: list[int], s_split: int) -> CPPlan:
    """Split every sequence into up to ``s_split`` near-equal segments, no
    profitability judgment -- the caller has decided (tests, experiments)."""
    n_segs = [max(1, min(s_split, r // max(1, MIN_SEG_TILES))) for r in seq_tiles]
    return _materialize(seq_tiles, n_segs)


# ---------------------------------------------------------------------------
# Tensor-level entry for the prefill wrapper
# ---------------------------------------------------------------------------
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
) -> CPPlan:
    """Mode-aware planning entry for the SM90 prefill wrapper.

    Returns the plan the executor runs verbatim; a trivial plan means "take
    the serial path". AUTO declines unprofitable splits; FORCE skips the
    profitability judgment and raises NotSplittableError when the shape
    cannot split at all.
    """
    if mode is None or mode is CPMode.OFF:
        return CPPlan.serial((), "disabled")
    B, T, H, _K = q.shape
    if cu_seqlens is None:
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

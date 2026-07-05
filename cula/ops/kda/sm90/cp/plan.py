# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Segment planning for SM90 KDA intracard context-parallel (CP) prefill."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

CHUNK = 16
MIN_SEG_TILES = int(os.environ.get("CULA_KDA_CP_MIN_SEG_TILES", "4"))
AUTO_MIN_SEG_TILES = int(os.environ.get("CULA_KDA_CP_AUTO_MIN_SEG_TILES", "32"))
# Superseded by estimate_cp_speedup for the auto engage decision; kept for
# env-var compatibility and external callers.
MIN_BENEFICIAL_SEG = int(os.environ.get("CULA_KDA_CP_MIN_SEG", "5"))
ENGAGE_MIN_TILES = int(os.environ.get("CULA_KDA_CP_ENGAGE_MIN_TILES", "640"))

# ---------------------------------------------------------------------------
# Engage cost model
# ---------------------------------------------------------------------------
# Per-CTA chain wall time is modeled as `per_tile * tiles + fixed` (us),
# fitted on H100 SXM (D=128, CHUNK=16, bf16) against the serial K2 chain,
# the (interleaved) pre_scan S+M chain, and the per-segment K2 rerun. K1 and
# driver host time are identical/hidden on both sides and cancel out of the
# comparison. Only the serial-vs-CP ratio drives the decision, so absolute
# miscalibration on other SM90 parts shifts the break-even point mildly
# without changing the asymptotics; override via env for other silicon.
CP_COST_SERIAL_PER_TILE_US = float(os.environ.get("CULA_KDA_CP_COST_SERIAL_PER_TILE_US", "1.48"))
CP_COST_SERIAL_FIXED_US = float(os.environ.get("CULA_KDA_CP_COST_SERIAL_FIXED_US", "84"))
CP_COST_PRESCAN_PER_TILE_US = float(os.environ.get("CULA_KDA_CP_COST_PRESCAN_PER_TILE_US", "2.39"))
CP_COST_PRESCAN_FIXED_US = float(os.environ.get("CULA_KDA_CP_COST_PRESCAN_FIXED_US", "27"))
CP_COST_K2SEG_PER_TILE_US = float(os.environ.get("CULA_KDA_CP_COST_K2SEG_PER_TILE_US", "1.69"))
CP_COST_K2SEG_FIXED_US = float(os.environ.get("CULA_KDA_CP_COST_K2SEG_FIXED_US", "19"))
CP_COST_MERGE_FIXED_US = float(os.environ.get("CULA_KDA_CP_COST_MERGE_FIXED_US", "8"))
CP_COST_MERGE_PER_SEG_US = float(os.environ.get("CULA_KDA_CP_COST_MERGE_PER_SEG_US", "3.3"))
# Required predicted serial/CP ratio before auto engages CP: absorbs model
# error, the (hidden) extra driver host work, and measurement noise.
CP_ENGAGE_MARGIN = float(os.environ.get("CULA_KDA_CP_ENGAGE_MARGIN", "1.10"))

_SM_COUNT_CACHE: dict[int, int] = {}


def _sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    v = _SM_COUNT_CACHE.get(idx)
    if v is None:
        v = torch.cuda.get_device_properties(idx).multi_processor_count
        _SM_COUNT_CACHE[idx] = v
    return v


def _auto_s_split(device: torch.device, seq_tiles: list[int], H: int) -> int:
    """How many segments to split each long sequence into.

    Target: fill one SM wave (total segments ≈ SM count).
    Short sequences (< 2*AUTO_MIN_SEG_TILES tiles) stay as 1 segment.
    """
    sm_count = _sm_count(device)
    n_seqs = len(seq_tiles)
    n_nosplit = sum(1 for r in seq_tiles if r < 2 * AUTO_MIN_SEG_TILES)
    n_split = n_seqs - n_nosplit
    if n_split == 0:
        return 1
    remaining = max(n_split * H, sm_count - n_nosplit * H)
    return max(1, remaining // (H * n_split))


def _plan_segments(
    seq_tiles: list[int], s_split: int, min_seg_tiles: int | None = None
) -> tuple[list[int], list[tuple[int, int]]]:
    """Split each sequence's tile range into <= s_split near-equal segments."""
    if min_seg_tiles is None:
        min_seg_tiles = MIN_SEG_TILES
    seg_cu = [0]
    per_seq: list[tuple[int, int]] = []
    for r in seq_tiles:
        n_seg = max(1, min(s_split, r // max(1, min_seg_tiles)))
        n_seg = min(n_seg, r)
        first = len(seg_cu) - 1
        base, rem = divmod(r, n_seg)
        for i in range(n_seg):
            seg_cu.append(seg_cu[-1] + base + (1 if i < rem else 0))
        per_seq.append((first, n_seg))
    return seg_cu, per_seq


def _plan_balanced(device: torch.device, seq_tiles: list[int], H: int) -> tuple[list[int], list[tuple[int, int]]]:
    """Duration-weighted plan: aim for equal segment LENGTHS across the batch.

    The legacy planner hands every splittable sequence the same segment COUNT
    and budgets short sequences as if they occupied their SMs for the whole
    kernel; with ragged batches that leaves the critical path on the longest
    segment (or refuses to split at all). Here each sequence gets
    ceil(tiles / target_len) segments, with target_len sized so one SM wave
    covers the total load — short sequences drain early and the scheduler
    backfills their SMs.
    """
    sm_count = _sm_count(device)
    slots = max(1, sm_count // H)
    total = sum(seq_tiles)
    target_len = max(AUTO_MIN_SEG_TILES, -(-total // slots))
    seg_cu = [0]
    per_seq: list[tuple[int, int]] = []
    for r in seq_tiles:
        n_seg = max(1, min(-(-r // target_len), r // max(1, AUTO_MIN_SEG_TILES)))
        n_seg = min(n_seg, r)
        first = len(seg_cu) - 1
        base, rem = divmod(r, n_seg)
        for i in range(n_seg):
            seg_cu.append(seg_cu[-1] + base + (1 if i < rem else 0))
        per_seq.append((first, n_seg))
    return seg_cu, per_seq


def _plan_is_splittable(seq_tiles: list[int], seg_cu: list[int], per_seq: list[tuple[int, int]]) -> bool:
    """Same structural criterion policy/driver apply before engaging CP."""
    return len(seg_cu) - 1 > len(seq_tiles) and max(n_seg for _first, n_seg in per_seq) > 2


def auto_plan_segments(device: torch.device, seq_tiles: list[int], H: int) -> tuple[int, list[int], list[tuple[int, int]]]:
    """Return the automatic segment cap and planned segments.

    Builds two candidates — the legacy equal-count plan and the
    duration-balanced plan — and keeps whichever the cost model predicts
    faster. Ties keep the legacy plan, so uniform/dense batches plan exactly
    as before; ragged batches (where the legacy planner under-splits or
    refuses) switch to the balanced plan.
    """
    s_split = _auto_s_split(device, seq_tiles, H)
    seg_cu, per_seq = _plan_segments(seq_tiles, s_split, AUTO_MIN_SEG_TILES)
    seg_cu_b, per_seq_b = _plan_balanced(device, seq_tiles, H)

    legacy_ok = _plan_is_splittable(seq_tiles, seg_cu, per_seq)
    balanced_ok = _plan_is_splittable(seq_tiles, seg_cu_b, per_seq_b)
    if balanced_ok and (
        not legacy_ok or _cp_cost_us(device, seg_cu_b, per_seq_b, H) < _cp_cost_us(device, seg_cu, per_seq, H)
    ):
        seg_cu, per_seq = seg_cu_b, per_seq_b
        s_split = max(n_seg for _first, n_seg in per_seq)
    return s_split, seg_cu, per_seq


def _makespan_us(chain_tiles: list[int], per_tile_us: float, fixed_us: float, H: int, sm_count: int) -> float:
    """Lower-bound makespan of one chain-per-(chain, head) kernel: the slowest
    chain, or the average per-SM load when chains x H oversubscribe the SMs."""
    if not chain_tiles:
        return 0.0
    walls = [per_tile_us * t + fixed_us for t in chain_tiles]
    return max(max(walls), sum(walls) * H / sm_count)


def _cp_cost_us(device: torch.device, seg_cu: list[int], per_seq: list[tuple[int, int]], H: int) -> float:
    """Predicted CP pipeline wall time (pre_scan + merge + segment-K2), us."""
    sm_count = _sm_count(device)
    seg_tiles = [seg_cu[i + 1] - seg_cu[i] for i in range(len(seg_cu) - 1)]
    max_n_seg = max(n_seg for _first, n_seg in per_seq)
    return (
        _makespan_us(seg_tiles, CP_COST_PRESCAN_PER_TILE_US, CP_COST_PRESCAN_FIXED_US, H, sm_count)
        + _makespan_us(seg_tiles, CP_COST_K2SEG_PER_TILE_US, CP_COST_K2SEG_FIXED_US, H, sm_count)
        + CP_COST_MERGE_FIXED_US
        + CP_COST_MERGE_PER_SEG_US * max_n_seg
    )


def estimate_cp_speedup(
    device: torch.device,
    seq_tiles: list[int],
    seg_cu: list[int],
    per_seq: list[tuple[int, int]],
    H: int,
) -> float:
    """Predicted serial-K2 / (pre_scan + merge + segment-K2) wall-time ratio.

    > 1 means CP is predicted faster. K1 and host-side driver work are the
    same on both sides and are excluded.
    """
    sm_count = _sm_count(device)
    serial_us = _makespan_us(seq_tiles, CP_COST_SERIAL_PER_TILE_US, CP_COST_SERIAL_FIXED_US, H, sm_count)
    cp_us = _cp_cost_us(device, seg_cu, per_seq, H)
    if cp_us <= 0.0:
        return 0.0
    return serial_us / cp_us


@dataclass
class CPPlan:
    """Result of segment planning for intracard CP.

    Each sequence is split into contiguous segments of tiles.
    Segments across all sequences are numbered globally 0..n_seg_total-1.
    """

    n_seqs: int
    n_seg_total: int
    seq_tiles: list[int]
    seg_cu: list[int]  # cumulative tile boundary per segment (len n_seg_total+1)
    per_seq: list[tuple[int, int]]  # (first_segment, n_segments) for each sequence
    total_tiles: int  # ceil tile count (for workspace sizing)
    s_split: int
    # varlen partial-tile metadata (None for dense or aligned varlen)
    v_tile_starts: torch.Tensor | None
    v_tile_actual_lens: torch.Tensor | None
    v_is_varlen: bool


def plan_cp(
    device: torch.device,
    n_seqs: int,
    seq_tiles: list[int],
    T_total: int,
    H: int,
    s_split: int | None,
    varlen_meta=None,
) -> CPPlan:
    """Plan how to split sequences into parallel segments.

    Returns a CPPlan. Caller should check plan.n_seg_total > n_seqs
    and max segments > 2 to decide whether CP is worth it.
    """
    v_tile_starts = None
    v_tile_actual_lens = None
    v_is_varlen = False
    total_tiles = sum(seq_tiles)

    if varlen_meta is not None and varlen_meta.needs_padding:
        v_is_varlen = True
        v_tile_starts = varlen_meta.tile_starts
        v_tile_actual_lens = varlen_meta.tile_actual_lens

    if s_split is None:
        # Must match the plan the policy layer scored (same entry point).
        s_split, seg_cu, per_seq = auto_plan_segments(device, seq_tiles, H)
    else:
        seg_cu, per_seq = _plan_segments(seq_tiles, s_split, MIN_SEG_TILES)

    return CPPlan(
        n_seqs=n_seqs,
        n_seg_total=len(seg_cu) - 1,
        seq_tiles=seq_tiles,
        seg_cu=seg_cu,
        per_seq=per_seq,
        total_tiles=total_tiles,
        s_split=s_split,
        v_tile_starts=v_tile_starts,
        v_tile_actual_lens=v_tile_actual_lens,
        v_is_varlen=v_is_varlen,
    )

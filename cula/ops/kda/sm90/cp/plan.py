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
MIN_BENEFICIAL_SEG = int(os.environ.get("CULA_KDA_CP_MIN_SEG", "5"))
ENGAGE_MIN_TILES = int(os.environ.get("CULA_KDA_CP_ENGAGE_MIN_TILES", "640"))

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


def auto_plan_segments(device: torch.device, seq_tiles: list[int], H: int) -> tuple[int, list[int], list[tuple[int, int]]]:
    """Return the automatic segment cap and planned segments."""
    s_split = _auto_s_split(device, seq_tiles, H)
    seg_cu, per_seq = _plan_segments(seq_tiles, s_split, AUTO_MIN_SEG_TILES)
    return s_split, seg_cu, per_seq


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
        s_split = _auto_s_split(device, seq_tiles, H)
        min_seg_tiles = AUTO_MIN_SEG_TILES
    else:
        min_seg_tiles = MIN_SEG_TILES

    seg_cu, per_seq = _plan_segments(seq_tiles, s_split, min_seg_tiles)

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

# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python segment planning for SM90 KDA intracard context-parallel (CP) prefill."""

from __future__ import annotations

import os

import torch

CHUNK = 16  # match k2.CHUNK
MIN_SEG_TILES = int(os.environ.get("CULA_KDA_CP_MIN_SEG_TILES", "4"))
# Per-segment tile floor for auto planning. Low enough that a single long sequence
# can split into ~SM_count/H segments (one full SM wave) instead of staying coarse.
AUTO_MIN_SEG_TILES = int(os.environ.get("CULA_KDA_CP_AUTO_MIN_SEG_TILES", "32"))
# Auto-router perf gate: skip CP when a sequence plans into fewer than this many
# segments — measured to regress vs serial (pre_scan/merge overhead > parallelism)
# at <=4 segments/seq. force (use_cp=True) ignores this and still runs.
MIN_BENEFICIAL_SEG = int(os.environ.get("CULA_KDA_CP_MIN_SEG", "5"))
# Auto-router perf gate: CP only engages once the longest sequence reaches this many
# tiles. Below this the serial K1+K2 already finishes fast (its short per-(seq,head)
# chain isn't the bottleneck), so CP's pre_scan/merge overhead is a net loss. This is
# decoupled from AUTO_MIN_SEG_TILES so the split can be fine (fill SMs) while the
# engage threshold stays where serial actually starts to lose.
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
    sm_count = _sm_count(device)
    # Target one full SM wave (not two): fill the array while keeping the segment
    # count — and thus the pre_scan/merge work — as low as possible.
    target_ctas = sm_count
    n_seqs = len(seq_tiles)
    # Short sequences (< 2*AUTO_MIN_SEG_TILES) get 1 segment; exclude from SM budget.
    n_nosplit = sum(1 for r in seq_tiles if r < 2 * AUTO_MIN_SEG_TILES)
    n_split = n_seqs - n_nosplit
    if n_split == 0:
        return 1
    remaining = max(n_split * H, target_ctas - n_nosplit * H)
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

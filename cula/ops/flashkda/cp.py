# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""FlashKDA intracard-CP (intra-card sequence parallelism) prefill driver.

Three-stage pipeline over chunk-affine recurrence S' = M_c*S + B_c:

  1. pre_scan : per segment (contiguous tile range) compute
                  B_seg (S-chain, S0=0) and M_seg (M-chain, M0=I, V:=0 duality).
  2. merge    : host-side fold of segment carries
                  carry_{i+1} = carry_i @ G_M_i + G_B_i      (bhvk layout trick:
                GMEM "vk" layout stores G = S^T, turning the left-multiplied
                recurrence S' = M*S + B into a right-multiplication, so the
                merge is a plain torch.baddbmm with no transposes).
  3. rerun    : full K2 per segment with the correct carry as initial_state,
                writing the real `out` (global tile indexing places segment
                outputs at their final positions).

K1 runs ONCE for the whole batch: its workspaces are per-tile with no
cross-chunk recurrence, shared by pre_scan and rerun.

All internal CP state buffers use layout False ("bhvk", [*, V, K] K-contiguous).
A user-facing ``state_transposed=True`` is handled by transposing the last two
dims at driver entry/exit.
"""

from __future__ import annotations

import os

import torch

from cula.ops.flashkda.k1 import launch_k1
from cula.ops.flashkda.k2 import CHUNK, D, launch_k2
from cula.ops.flashkda.prefill import (
    _copy_beta_flat,
    _ensure_cute_arch_for_device,
    _get_or_alloc_workspaces,
    _get_or_build_seq_lens,
    flash_kda_prefill,
)

MIN_SEG_TILES = int(os.environ.get("CULA_FLASHKDA_CP_MIN_SEG_TILES", "4"))
# Auto planner: never split below this many tiles per segment (S6 sweep: the
# 2-CTA/SM target + this floor hits the measured optimum on every deployment
# config, H in {8,16,24,32}, T in {16K,32K}, varlen).
AUTO_MIN_SEG_TILES = int(os.environ.get("CULA_FLASHKDA_CP_AUTO_MIN_SEG_TILES", "128"))


# ---------------------------------------------------------------------------
# Segment planning
# ---------------------------------------------------------------------------
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
    target_ctas = 2 * sm_count
    n_seqs = len(seq_tiles)
    # Budget-aware: short sequences (tiles < 2*AUTO_MIN_SEG_TILES) always get
    # 1 segment regardless of s_split; don't let them consume SM budget that
    # could go to long sequences. Without this, 128K+10x1K at H=8 computes
    # s=floor(264/88)=3 → the 128K seq gets only 3 segments instead of 23.
    n_nosplit = sum(1 for r in seq_tiles if r < 2 * AUTO_MIN_SEG_TILES)
    n_split = n_seqs - n_nosplit
    if n_split == 0:
        return 1
    remaining = max(n_split * H, target_ctas - n_nosplit * H)
    return max(1, remaining // (H * n_split))


def _plan_segments(
    seq_tiles: list[int], s_split: int, min_seg_tiles: int | None = None
) -> tuple[list[int], list[tuple[int, int]]]:
    """Split each sequence's tile range into <= s_split near-equal segments.

    Returns (seg_cu_tiles, per_seq) where seg_cu_tiles is the exclusive prefix
    sum over ALL segments (global tile units) and per_seq[s] = (first_seg, n_seg).
    """
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


# ---------------------------------------------------------------------------
# Merge (host)
# ---------------------------------------------------------------------------
def _merge_carries_(
    carries: torch.Tensor,  # [S, H, D, D] fp32 scratch, fully overwritten
    m_seg: torch.Tensor,  # [S, H, D, D] fp32, bhvk (= M_seg^T per head)
    b_seg: torch.Tensor,  # [S, H, D, D] fp32, bhvk (= B_seg^T per head)
    per_seq: list[tuple[int, int]],
    init_bhvk: torch.Tensor | None,  # [N, H, D, D] fp32 or None
) -> torch.Tensor:
    """Per-sequence affine fold: carries[i] = state entering segment i (bhvk).

    In-place into preallocated scratch: out= baddbmm per step, no per-step
    allocations or extra copies (the fold is launch-count-bound on host).
    """
    for s, (first, n_seg) in enumerate(per_seq):
        if init_bhvk is None:
            carries[first].zero_()
        else:
            carries[first].copy_(init_bhvk[s])
        for i in range(first, first + n_seg - 1):
            # G_S' = G_S @ G_M + G_B  (right-multiply in bhvk layout)
            torch.baddbmm(b_seg[i], carries[i], m_seg[i], out=carries[i + 1])
    return carries


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------
_EYE_CACHE: dict = {}
_SCRATCH_CACHE: dict = {}
_PLAN_TENSOR_CACHE: dict = {}


def _get_plan_tensor(values: tuple, dtype, device: torch.device) -> torch.Tensor:
    """Small host-derived index tensor (seg_cu_tiles, last-segment idx), cached
    per plan so steady-state calls skip the torch.tensor alloc + HtoD copy."""
    key = (values, dtype, str(device))
    cached = _PLAN_TENSOR_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(values, dtype=dtype, device=device)
        _PLAN_TENSOR_CACHE[key] = cached
    return cached


def _get_eye(n_seg: int, H: int, device: torch.device) -> torch.Tensor:
    key = (n_seg, H, str(device))
    cached = _EYE_CACHE.get(key)
    if cached is None:
        eye = torch.eye(D, dtype=torch.float32, device=device)
        cached = eye.expand(n_seg, H, D, D).contiguous()
        _EYE_CACHE[key] = cached
    return cached


def _get_scratch(key_name: str, shape: tuple, dtype, device, zero_on_alloc: bool = False) -> torch.Tensor:
    key = (key_name, shape, dtype, str(device))
    cached = _SCRATCH_CACHE.get(key)
    if cached is None:
        alloc = torch.zeros if zero_on_alloc else torch.empty
        cached = alloc(shape, dtype=dtype, device=device)
        _SCRATCH_CACHE[key] = cached
    return cached


def _get_prescan_launcher():
    if os.environ.get("CULA_FLASHKDA_CP_V0", "0") == "1":
        return None
    try:
        from cula.ops.flashkda.k2_prescan import launch_k2_prescan

        return launch_k2_prescan
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def flash_kda_prefill_cp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    state_transposed: bool = False,
    s_split: int | None = None,
) -> None:
    """FlashKDA prefill with intracard sequence parallelism.

    Same semantics as ``flash_kda_prefill`` (CuTeDSL path), restricted to
    CHUNK-aligned sequence lengths. ``s_split`` caps the number of segments
    per sequence (None = auto from SM count).
    """
    assert q.is_cuda and q.dtype == torch.bfloat16
    B, T, H, K = q.shape
    assert K == D
    device = q.device
    _ensure_cute_arch_for_device(device)

    if cu_seqlens is None:
        assert T % CHUNK == 0, f"T={T} must be a multiple of {CHUNK}"
        n_seqs = B
        seq_tiles = [T // CHUNK] * B
        T_total = B * T
    else:
        assert B == 1, "varlen requires packed B=1"
        # weakref-identity cached in prefill.py: no DtoH sync on repeat calls
        seq_lens = _get_or_build_seq_lens(cu_seqlens)
        n_seqs = len(seq_lens)
        assert all(sl % CHUNK == 0 for sl in seq_lens), (
            "intracard-CP requires CHUNK-aligned sequence lengths; "
            "use flash_kda_prefill for the padded-repack path"
        )
        seq_tiles = [sl // CHUNK for sl in seq_lens]
        T_total = T

    min_seg_tiles = None
    if s_split is None:
        s_split = _auto_s_split(device, seq_tiles, H)
        min_seg_tiles = AUTO_MIN_SEG_TILES

    seg_cu, per_seq = _plan_segments(seq_tiles, s_split, min_seg_tiles)
    n_seg_total = len(seg_cu) - 1

    # Bypass: if no sequence gets more than 2 segments, the parallelism gain
    # is too small to overcome prescan+merge overhead (~0.3-0.4 ms).
    max_n_seg = max(n_seg for _, n_seg in per_seq)
    if n_seg_total == n_seqs or max_n_seg <= 2:
        # Degenerate plan (1 segment per sequence): CP is pure overhead, the
        # serial path is bit-identical semantics. Delegate.
        flash_kda_prefill(
            q, k, v, g, beta, scale=scale, out=out, A_log=A_log,
            dt_bias=dt_bias, lower_bound=lower_bound,
            initial_state=initial_state, final_state=final_state,
            cu_seqlens=cu_seqlens, state_transposed=state_transposed,
        )
        return

    seg_cu_tiles = _get_plan_tensor(tuple(seg_cu), torch.int32, device)
    total_tiles = T_total // CHUNK

    # ---- K1 once (workspaces are per-tile, segment-agnostic) ----
    n_qk = total_tiles * H * CHUNK * D
    n_cc = total_tiles * H * CHUNK * CHUNK
    ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk, beta_flat = _get_or_alloc_workspaces(
        n_qk, n_cc, total_tiles * H * D, T_total * H, device, beta.dtype
    )
    _copy_beta_flat(beta, beta_flat, H, T_total)
    launch_k1(q, k, g, A_log, dt_bias, beta_flat, scale, lower_bound,
              ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk)

    # ---- user initial_state -> internal bhvk fp32 ----
    init_bhvk = None
    if initial_state is not None:
        assert initial_state.shape == (n_seqs, H, D, D)
        init_bhvk = initial_state.to(torch.float32)
        if state_transposed:
            init_bhvk = init_bhvk.transpose(-1, -2)
        init_bhvk = init_bhvk.contiguous()

    # ---- stage 1: pre_scan -> B_seg, M_seg (bhvk fp32) ----
    b_seg = _get_scratch("b_seg", (n_seg_total, H, D, D), torch.float32, device)
    m_seg = _get_scratch("m_seg", (n_seg_total, H, D, D), torch.float32, device)
    v_flat = v.view(1, T_total, H, D) if B > 1 else v

    prescan = _get_prescan_launcher()
    if prescan is not None:
        prescan(v_flat, beta_flat, ws_kd, ws_kr, ws_gt, ws_inv,
                b_seg, m_seg, seg_cu_tiles)
    else:
        # v0: two passes through the unmodified K2.
        scratch_out = _get_scratch("scratch_out", v_flat.shape, torch.bfloat16, device)
        launch_k2(v_flat, beta_flat, ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk,
                  scratch_out, seg_cu_tiles,
                  initial_state=None, final_state=b_seg, state_transposed=False)
        # Read-only inside K2; zeroed once at allocation.
        zeros_v = _get_scratch("zeros_v", v_flat.shape, torch.bfloat16, device, zero_on_alloc=True)
        launch_k2(zeros_v, beta_flat, ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk,
                  scratch_out, seg_cu_tiles,
                  initial_state=_get_eye(n_seg_total, H, device), final_state=m_seg,
                  state_transposed=False)

    # ---- stage 2: merge ----
    carries = _get_scratch("carries", (n_seg_total, H, D, D), torch.float32, device)
    _merge_carries_(carries, m_seg, b_seg, per_seq, init_bhvk)

    # ---- stage 3: rerun with correct carries ----
    out_flat = out.view(1, T_total, H, D) if B > 1 else out
    seg_final = None
    if final_state is not None:
        seg_final = _get_scratch("seg_final", (n_seg_total, H, D, D), torch.float32, device)
    launch_k2(v_flat, beta_flat, ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk,
              out_flat, seg_cu_tiles,
              initial_state=carries, final_state=seg_final, state_transposed=False)

    if final_state is not None:
        last_idx = _get_plan_tensor(
            tuple(first + n_seg - 1 for first, n_seg in per_seq), torch.long, device
        )
        if (
            not state_transposed
            and final_state.dtype == torch.float32
            and final_state.is_contiguous()
            and final_state.shape == (n_seqs, H, D, D)
        ):
            # Gather straight into the user buffer: no alloc, no extra copy.
            torch.index_select(seg_final, 0, last_idx, out=final_state)
        else:
            fin = seg_final.index_select(0, last_idx)
            if state_transposed:
                fin = fin.transpose(-1, -2)
            final_state.copy_(fin.to(final_state.dtype))

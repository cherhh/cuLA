# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2026 MoonshotAI
# Licensed under the MIT License.
# Based on MoonshotAI/FlashKDA (https://github.com/MoonshotAI/FlashKDA)

"""FlashKDA intracard-CP prefill driver.

Three-stage pipeline (pre_scan, merge, rerun) over chunk-affine recurrence.
Internal state buffers use bhvk layout; user-facing state_transposed handled
at entry/exit.
"""

from __future__ import annotations

import os

import torch

from cula.ops.sm90.k1 import launch_k1
from cula.ops.sm90.k2 import CHUNK, D, launch_k2
from cula.ops.sm90.cp.pre_scan import launch_pre_scan
from cula.ops.sm90.cp.merge import launch_merge
from cula.ops.sm90.fwd import (
    _copy_beta_flat,
    _cute_arch_for_device,
    _get_or_alloc_workspaces,
    _get_or_build_seq_lens,
    flash_kda_fwd,
)

MIN_SEG_TILES = int(os.environ.get("CULA_FLASHKDA_CP_MIN_SEG_TILES", "4"))
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


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------
_SCRATCH_CACHE: dict = {}
_PLAN_TENSOR_CACHE: dict = {}
_SCRATCH_CACHE_MAXSIZE = 8
_PLAN_TENSOR_CACHE_MAXSIZE = 64


def _get_plan_tensor(values: tuple, dtype, device: torch.device) -> torch.Tensor:
    key = (values, dtype, str(device))
    cached = _PLAN_TENSOR_CACHE.get(key)
    if cached is None:
        if len(_PLAN_TENSOR_CACHE) >= _PLAN_TENSOR_CACHE_MAXSIZE:
            _PLAN_TENSOR_CACHE.pop(next(iter(_PLAN_TENSOR_CACHE)))
        cached = torch.tensor(values, dtype=dtype, device=device)
        _PLAN_TENSOR_CACHE[key] = cached
    return cached


def _get_scratch(key_name: str, shape: tuple, dtype, device, zero_on_alloc: bool = False) -> torch.Tensor:
    key = (key_name, shape, dtype, str(device))
    cached = _SCRATCH_CACHE.get(key)
    if cached is None:
        if len(_SCRATCH_CACHE) >= _SCRATCH_CACHE_MAXSIZE:
            _SCRATCH_CACHE.pop(next(iter(_SCRATCH_CACHE)))
        alloc = torch.zeros if zero_on_alloc else torch.empty
        cached = alloc(shape, dtype=dtype, device=device)
        _SCRATCH_CACHE[key] = cached
    return cached



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def flash_kda_fwd_cp(*args, **kwargs) -> None:
    q = args[0] if args else kwargs["q"]
    with _cute_arch_for_device(q.device):
        _flash_kda_fwd_cp_impl(*args, **kwargs)


def _flash_kda_fwd_cp_impl(
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

    Same semantics as ``flash_kda_fwd``, restricted to CHUNK-aligned
    sequence lengths. ``s_split`` caps segments per sequence (None = auto).
    """
    assert q.is_cuda and q.dtype == torch.bfloat16
    B, T, H, K = q.shape
    assert K == D
    device = q.device

    if cu_seqlens is None:
        assert T % CHUNK == 0, f"T={T} must be a multiple of {CHUNK}"
        n_seqs = B
        seq_tiles = [T // CHUNK] * B
        T_total = B * T
    else:
        assert B == 1, "varlen requires packed B=1"
        seq_lens = _get_or_build_seq_lens(cu_seqlens)
        n_seqs = len(seq_lens)
        assert all(sl % CHUNK == 0 for sl in seq_lens), (
            "intracard-CP requires CHUNK-aligned sequence lengths; "
            "use flash_kda_fwd for the padded-repack path"
        )
        seq_tiles = [sl // CHUNK for sl in seq_lens]
        T_total = T

    min_seg_tiles = None
    if s_split is None:
        s_split = _auto_s_split(device, seq_tiles, H)
        min_seg_tiles = AUTO_MIN_SEG_TILES

    seg_cu, per_seq = _plan_segments(seq_tiles, s_split, min_seg_tiles)
    n_seg_total = len(seg_cu) - 1

    # Bypass: <= 2 segments per sequence => CP overhead outweighs parallelism.
    max_n_seg = max(n_seg for _, n_seg in per_seq)
    if n_seg_total == n_seqs or max_n_seg <= 2:
        flash_kda_fwd(
            q, k, v, g, beta, scale=scale, out=out, A_log=A_log,
            dt_bias=dt_bias, lower_bound=lower_bound,
            initial_state=initial_state, final_state=final_state,
            cu_seqlens=cu_seqlens, state_transposed=state_transposed,
        )
        return

    seg_cu_tiles = _get_plan_tensor(tuple(seg_cu), torch.int32, device)
    total_tiles = T_total // CHUNK

    # ---- K1 once ----
    n_qk = total_tiles * H * CHUNK * D
    n_cc = total_tiles * H * CHUNK * CHUNK
    ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk, beta_flat = _get_or_alloc_workspaces(
        n_qk, n_cc, total_tiles * H * D, T_total * H, device, beta.dtype
    )
    _copy_beta_flat(beta, beta_flat, H, T_total)
    launch_k1(q, k, g, A_log, dt_bias, beta_flat, scale, lower_bound,
              ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk)

    # ---- initial_state -> bhvk fp32 ----
    init_bhvk = None
    if initial_state is not None:
        assert initial_state.shape == (n_seqs, H, D, D)
        init_bhvk = initial_state.to(torch.float32)
        if state_transposed:
            init_bhvk = init_bhvk.transpose(-1, -2)
        init_bhvk = init_bhvk.contiguous()

    # ---- stage 1: pre_scan ----
    b_seg = _get_scratch("b_seg", (n_seg_total, H, D, D), torch.float32, device)
    m_seg = _get_scratch("m_seg", (n_seg_total, H, D, D), torch.float32, device)
    v_flat = v.view(1, T_total, H, D) if B > 1 else v

    launch_pre_scan(v_flat, beta_flat, ws_kd, ws_kr, ws_gt, ws_inv,
                      b_seg, m_seg, seg_cu_tiles)

    # ---- stage 2: merge ----
    carries = _get_scratch("carries", (n_seg_total, H, D, D), torch.float32, device)
    launch_merge(carries, m_seg, b_seg, per_seq, init_bhvk)

    # ---- stage 3: rerun ----
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
            torch.index_select(seg_final, 0, last_idx, out=final_state)
        else:
            fin = seg_final.index_select(0, last_idx)
            if state_transposed:
                fin = fin.transpose(-1, -2)
            final_state.copy_(fin.to(final_state.dtype))

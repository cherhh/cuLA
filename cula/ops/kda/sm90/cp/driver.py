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


"""
Intracard-CP prefill driver.

Three-stage pipeline (pre_scan, merge, rerun) over chunk-affine recurrence.
Internal state buffers use bhvk layout; user-facing state_transposed handled
at entry/exit.
"""

from __future__ import annotations

import torch

from cula.ops.kda.sm90.cp.merge import launch_merge
from cula.ops.kda.sm90.cp.plan import AUTO_MIN_SEG_TILES, _auto_s_split, _plan_segments
from cula.ops.kda.sm90.cp.pre_scan import launch_pre_scan
from cula.ops.kda.sm90.fwd import (
    _copy_beta_flat,
    _cute_arch_for_device,
    _get_or_alloc_workspaces,
    _get_or_build_varlen_metadata,
    flash_kda_fwd,
)
from cula.ops.kda.sm90.k1 import launch_k1
from cula.ops.kda.sm90.k2 import CHUNK, D, launch_k2

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
def intracard_prefill(*args, **kwargs) -> None:
    q = args[0] if args else kwargs["q"]
    with _cute_arch_for_device(q.device):
        _intracard_prefill_impl(*args, **kwargs)


# ---------------------------------------------------------------------------
# Dense partial-tile (non-CHUNK-aligned) support — pad to a CHUNK multiple.
# (Varlen is handled natively in _intracard_prefill_impl: ceil tiles + mask.)
# ---------------------------------------------------------------------------
def _pad_cp_inputs(pad, q, k, v, g, beta):
    """Pad the 5 KDA inputs with no-op CP sentinels via pad(tensor, fill):
    q/k/v=0, g=-1e6 (decay~1 keeps state), beta=-80 (sigmoid~0, no update)."""
    return pad(q, 0.0), pad(k, 0.0), pad(v, 0.0), pad(g, -1e6), pad(beta, -80.0)


def _intracard_prefill_padded_dense(
    q,
    k,
    v,
    g,
    beta,
    scale,
    out,
    A_log,
    dt_bias,
    lower_bound,
    initial_state,
    final_state,
    state_transposed,
    s_split,
    allow_fallback,
) -> None:
    B, T, H, _ = q.shape
    pad_t = ((T + CHUNK - 1) // CHUNK) * CHUNK - T

    def _pad(x: torch.Tensor, fill: float) -> torch.Tensor:
        spec = (0, 0, 0, pad_t) if x.ndim == 3 else (0, 0, 0, 0, 0, pad_t)
        return torch.nn.functional.pad(x, spec, value=fill)

    pq, pk, pv, pg, pbeta = _pad_cp_inputs(_pad, q, k, v, g, beta)
    pout = out.new_empty((B, T + pad_t, H, D))
    _intracard_prefill_impl(
        pq,
        pk,
        pv,
        pg,
        pbeta,
        scale,
        pout,
        A_log,
        dt_bias,
        lower_bound,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=None,
        state_transposed=state_transposed,
        s_split=s_split,
        allow_fallback=allow_fallback,
    )
    out.copy_(pout[:, :T])


def _intracard_prefill_impl(
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
    allow_fallback: bool = True,
) -> None:
    """Prefill with intracard sequence parallelism.

    Non-CHUNK-aligned sequence lengths are padded up to CHUNK and 
    run through the aligned CP pipeline. 
    
    ``s_split`` caps subsequences per sequence (None = auto).
    """
    assert q.is_cuda and q.dtype == torch.bfloat16
    B, T, H, K = q.shape
    assert K == D
    device = q.device

    # dense non-CHUNK-aligned: pad to a CHUNK multiple (cheap tail F.pad, no gather).
    # varlen is handled natively below (ceil tiles + in-kernel partial-tile mask, no pad).
    if cu_seqlens is None:
        if T % CHUNK != 0:
            _intracard_prefill_padded_dense(
                q,
                k,
                v,
                g,
                beta,
                scale,
                out,
                A_log,
                dt_bias,
                lower_bound,
                initial_state,
                final_state,
                state_transposed,
                s_split,
                allow_fallback,
            )
            return
    else:
        assert B == 1, "varlen requires packed B=1"
        assert cu_seqlens.dtype == torch.int32, f"cu_seqlens must be int32, got {cu_seqlens.dtype}"

    # Native varlen partial-tile handling (SM100-style): non-aligned seqs use ceil tile
    # counts + tile_starts so K1/pre_scan/K2 read packed v and mask the partial last tile,
    # instead of padding the whole input and scattering back.
    v_is_varlen = False
    v_tile_starts = None
    v_tile_actual_lens = None
    native_total_tiles = None
    if cu_seqlens is None:
        n_seqs = B
        seq_tiles = [T // CHUNK] * B
        T_total = B * T
    else:
        meta = _get_or_build_varlen_metadata(cu_seqlens)
        seq_lens = meta.seq_lens
        n_seqs = len(seq_lens)
        seq_tiles = [(sl + CHUNK - 1) // CHUNK for sl in seq_lens]  # ceil — last tile may be partial
        T_total = T
        if meta.needs_padding:
            v_is_varlen = True
            v_tile_starts = meta.tile_starts
            v_tile_actual_lens = meta.tile_actual_lens
            native_total_tiles = meta.total_tiles  # ceil tile sum

    min_seg_tiles = None
    if s_split is None:
        s_split = _auto_s_split(device, seq_tiles, H)
        min_seg_tiles = AUTO_MIN_SEG_TILES

    seg_cu, per_seq = _plan_segments(seq_tiles, s_split, min_seg_tiles)
    n_seg_total = len(seg_cu) - 1

    # Bypass: <= 2 segments per sequence => CP overhead outweighs parallelism.
    max_n_seg = max(n_seg for _, n_seg in per_seq)
    if n_seg_total == n_seqs or max_n_seg <= 2:
        if not allow_fallback:
            raise ValueError("SM90 intracard CP is not meaningfully splittable for this shape.")
        flash_kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            out=out,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
            state_transposed=state_transposed,
        )
        return

    seg_cu_tiles = _get_plan_tensor(tuple(seg_cu), torch.int32, device)
    total_tiles = native_total_tiles if v_is_varlen else T_total // CHUNK

    # ---- K1 once ----
    n_qk = total_tiles * H * CHUNK * D
    n_cc = total_tiles * H * CHUNK * CHUNK
    ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk, ws_beta = _get_or_alloc_workspaces(
        n_qk, n_cc, total_tiles * H * D, T_total * H, device, beta.dtype
    )
    # K1 emits raw beta into the compact ws_beta workspace (read by pre_scan + K2);
    # for the CHUNK-aligned data CP handles it is byte-identical to beta_flat.
    beta_flat = torch.empty(T_total * H, dtype=beta.dtype, device=device)
    _copy_beta_flat(beta, beta_flat, H, T_total)
    launch_k1(
        q, k, g, A_log, dt_bias, beta_flat, scale, lower_bound,
        ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk, ws_beta,
        tile_starts=v_tile_starts,
        tile_actual_lens=v_tile_actual_lens,
        total_tiles=native_total_tiles,
        is_varlen=v_is_varlen,
    )

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

    launch_pre_scan(
        v_flat, ws_beta, ws_kd, ws_kr, ws_gt, ws_inv, b_seg, m_seg, seg_cu_tiles,
        v_tile_starts=v_tile_starts,
        v_tile_actual_lens=v_tile_actual_lens,
        total_tiles=total_tiles,
    )

    # ---- stage 2: merge ----
    carries = _get_scratch("carries", (n_seg_total, H, D, D), torch.float32, device)
    launch_merge(carries, m_seg, b_seg, per_seq, init_bhvk)

    # ---- stage 3: rerun ----
    out_flat = out.view(1, T_total, H, D) if B > 1 else out
    seg_final = None
    if final_state is not None:
        seg_final = _get_scratch("seg_final", (n_seg_total, H, D, D), torch.float32, device)
    launch_k2(
        v_flat,
        ws_beta,
        ws_qd,
        ws_kd,
        ws_kr,
        ws_gt,
        ws_inv,
        ws_mqk,
        out_flat,
        seg_cu_tiles,
        initial_state=carries,
        final_state=seg_final,
        state_transposed=False,
        v_tile_starts=v_tile_starts,
        v_tile_actual_lens=v_tile_actual_lens,
    )

    if final_state is not None:
        last_idx = _get_plan_tensor(tuple(first + n_seg - 1 for first, n_seg in per_seq), torch.long, device)
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

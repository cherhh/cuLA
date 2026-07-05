# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Intracard-CP prefill driver: K1 once → pre_scan → merge → K2 rerun."""

from __future__ import annotations

import torch

from cula.ops.kda.sm90.cp.merge import launch_merge
from cula.ops.kda.sm90.cp.plan import CP_ENGAGE_MARGIN, CPPlan, estimate_cp_speedup, plan_cp
from cula.ops.kda.sm90.cp.pre_scan import launch_pre_scan
from cula.ops.kda.sm90.fwd import (
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


def _get_scratch(key_name: str, shape: tuple, dtype, device) -> torch.Tensor:
    key = (key_name, shape, dtype, str(device))
    cached = _SCRATCH_CACHE.get(key)
    if cached is None:
        if len(_SCRATCH_CACHE) >= _SCRATCH_CACHE_MAXSIZE:
            _SCRATCH_CACHE.pop(next(iter(_SCRATCH_CACHE)))
        cached = torch.empty(shape, dtype=dtype, device=device)
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
# Dense partial-tile support — pad to CHUNK multiple, run aligned CP, strip back.
# (Varlen partial-tile is handled natively via ceil tiles + tile_starts mask.)
# ---------------------------------------------------------------------------
def _pad_cp_inputs(pad, q, k, v, g, beta):
    """No-op sentinels: q/k/v=0, g=-1e6 (decay~0), beta=-80 (sigmoid~0)."""
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def _intracard_prefill_impl(
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
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    state_transposed=False,
    s_split=None,
    allow_fallback=True,
) -> None:
    assert q.is_cuda and q.dtype == torch.bfloat16
    B, T, H, K = q.shape
    assert K == D
    device = q.device

    # --- Step 1: handle non-aligned dense by padding ---
    if cu_seqlens is None and T % CHUNK != 0:
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

    # --- Step 2: compute tile counts and plan segments ---
    if cu_seqlens is None:
        n_seqs = B
        seq_tiles = [T // CHUNK] * B
        T_total = B * T
        varlen_meta = None
    else:
        assert B == 1
        assert cu_seqlens.dtype == torch.int32
        varlen_meta = _get_or_build_varlen_metadata(cu_seqlens)
        n_seqs = len(varlen_meta.seq_lens)
        seq_tiles = [(sl + CHUNK - 1) // CHUNK for sl in varlen_meta.seq_lens]
        T_total = T

    plan = plan_cp(device, n_seqs, seq_tiles, T_total, H, s_split, varlen_meta)

    # --- Step 3: bypass if CP isn't beneficial ---
    max_n_seg = max(n for _, n in plan.per_seq)
    bypass = plan.n_seg_total == n_seqs or max_n_seg <= 2
    if not bypass and allow_fallback:
        bypass = estimate_cp_speedup(device, seq_tiles, plan.seg_cu, plan.per_seq, H) < CP_ENGAGE_MARGIN
    if bypass:
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

    # --- Step 4: run CP pipeline (K1 → pre_scan → merge → K2) ---
    _run_cp_pipeline(
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
        cu_seqlens,
        state_transposed,
        plan,
        device,
        B,
        T_total,
        H,
    )


def _run_cp_pipeline(
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
    cu_seqlens,
    state_transposed,
    plan: CPPlan,
    device,
    B,
    T_total,
    H,
) -> None:
    n_seg = plan.n_seg_total
    seg_cu_tiles = _get_plan_tensor(tuple(plan.seg_cu), torch.int32, device)

    # ---- K1: prepare workspace tensors (run once) ----
    n_qk = plan.total_tiles * H * CHUNK * D
    n_cc = plan.total_tiles * H * CHUNK * CHUNK
    # ws_beta uses tile layout (total_tiles*CHUNK*H), not packed token layout (T_total*H).
    ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk, ws_beta = _get_or_alloc_workspaces(
        n_qk, n_cc, plan.total_tiles * H * D, plan.total_tiles * CHUNK * H, device, beta.dtype
    )
    launch_k1(
        q,
        k,
        g,
        A_log,
        dt_bias,
        beta.reshape(-1),
        scale,
        lower_bound,
        ws_qd,
        ws_kd,
        ws_kr,
        ws_gt,
        ws_inv,
        ws_mqk,
        ws_beta,
        tile_starts=plan.v_tile_starts,
        tile_actual_lens=plan.v_tile_actual_lens,
        total_tiles=plan.total_tiles if plan.v_is_varlen else None,
        is_varlen=plan.v_is_varlen,
    )

    # ---- pre_scan: compute per-segment B/M states ----
    b_seg = _get_scratch("b_seg", (n_seg, H, D, D), torch.float32, device)
    m_seg = _get_scratch("m_seg", (n_seg, H, D, D), torch.float32, device)
    v_flat = v.view(1, T_total, H, D) if B > 1 else v
    # Longest-first launch order: stable sort -> identity for uniform splits.
    seg_lens = [plan.seg_cu[i + 1] - plan.seg_cu[i] for i in range(n_seg)]
    seg_order = _get_plan_tensor(
        tuple(sorted(range(n_seg), key=lambda i: -seg_lens[i])), torch.int32, device
    )
    launch_pre_scan(
        v_flat,
        ws_beta,
        ws_kd,
        ws_kr,
        ws_gt,
        ws_inv,
        b_seg,
        m_seg,
        seg_cu_tiles,
        v_tile_starts=plan.v_tile_starts,
        v_tile_actual_lens=plan.v_tile_actual_lens,
        total_tiles=plan.total_tiles,
        seg_order=seg_order,
    )

    # ---- merge: propagate carries across segments within each sequence ----
    init_bhvk = None
    if initial_state is not None:
        assert initial_state.shape == (plan.n_seqs, H, D, D)
        init_bhvk = initial_state.to(torch.float32)
        if state_transposed:
            init_bhvk = init_bhvk.transpose(-1, -2)
        init_bhvk = init_bhvk.contiguous()
    carries = _get_scratch("carries", (n_seg, H, D, D), torch.float32, device)
    launch_merge(carries, m_seg, b_seg, plan.per_seq, init_bhvk)

    # ---- K2: rerun recurrence with merged carries as initial states ----
    out_flat = out.view(1, T_total, H, D) if B > 1 else out
    seg_final = None
    if final_state is not None:
        seg_final = _get_scratch("seg_final", (n_seg, H, D, D), torch.float32, device)
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
        v_tile_starts=plan.v_tile_starts,
        v_tile_actual_lens=plan.v_tile_actual_lens,
        seq_order=seg_order,
    )

    # ---- gather final states (one per original sequence) ----
    if final_state is not None:
        last_idx = _get_plan_tensor(tuple(first + n - 1 for first, n in plan.per_seq), torch.long, device)
        if (
            not state_transposed
            and final_state.dtype == torch.float32
            and final_state.is_contiguous()
            and final_state.shape == (plan.n_seqs, H, D, D)
        ):
            torch.index_select(seg_final, 0, last_idx, out=final_state)
        else:
            fin = seg_final.index_select(0, last_idx)
            if state_transposed:
                fin = fin.transpose(-1, -2)
            final_state.copy_(fin.to(final_state.dtype))

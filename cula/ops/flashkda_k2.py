# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
FlashKDA K2 (Recurrence) — CuteDSL port.

This is a from-scratch CuteDSL reimplementation of FlashKDA's
``fwd_kernel2.cuh``. The C++ kernel is highly optimized (warp specialization,
3-stage TMA pipeline, dual GMMA, register reuse via SM75_U32x1_MOVM_T) and
exceeds 800 lines.

Strategy here:
    * Phase A (THIS CODE): correctness baseline. Single block role, no warp
      specialization, plain shared-memory matmuls. All math validated end-to-
      end against a torch reference.
    * Phase B: introduce TMA loads + multi-stage software pipeline.
    * Phase C: replace plain matmuls with SM80 16x8x16 mma atoms.
    * Phase D: warp-specialize (LOAD / MMA / STORE warps) + GMMA on SM90+.

Math per (B, H) recurrence step over T_total / CHUNK tiles:
    Inputs (per tile, all from K1 workspace + user v/beta):
        v   bf16 [CHUNK, D]
        beta bf16 [CHUNK]                   (sigmoid applied per row)
        kd, qd, kr  bf16 [CHUNK, D]         (decay-applied k/q + restored k)
        gt  fp32 [D]                         (= exp(g_total) per d_k)
        INV bf16 [CHUNK, CHUNK]
        Mqk bf16 [CHUNK, CHUNK]
    State (persistent across tiles): state[D, D] bf16, init to zero.
    Per-tile:
        tmp_o = kd @ state          # [CHUNK, D]
        out0  = qd @ state          # [CHUNK, D]
        u     = (v - tmp_o) * sigmoid(beta)[:, None]   # [CHUNK, D]
        u     = INV @ u             # [CHUNK, D]
        out   = out0 + Mqk @ u      # [CHUNK, D]
        state = state * gt[:, None] + kr.T @ u

Output: out bf16 [B, T_total, H, D].

Layout assumption (matches K1 workspace / canonical FlashKDA C++ memory):
    v, beta, out: per (B*T_total, H, *) — stride matches user-facing tensors
    Workspace tensors are flat 1D (head_idx*total_tiles + tile_idx) keyed,
    same packing K1 produces.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import torch
from cutlass.cute.runtime import from_dlpack

# Geometry: must match K1.
CHUNK: int = 16
D: int = 128

# Phase A: 128 threads per CTA (single MMA-like role, no specialization yet).
THREADS_PER_CTA: int = 128


# ---------------------------------------------------------------------------
# Phase A: correctness-only K2. Plain SMEM matmuls, no TMA/warp-spec/MMA.
# ---------------------------------------------------------------------------
@cute.kernel
def k2_phaseA_kernel(
    v: cute.Tensor,  # bf16 [T_total, H, D]
    beta: cute.Tensor,  # bf16 [H * T_total]    (linear, head-major)
    ws_qd: cute.Tensor,  # bf16 [H * total_tiles * CHUNK * D]
    ws_kd: cute.Tensor,
    ws_kr: cute.Tensor,
    ws_gt: cute.Tensor,  # fp32 [H * total_tiles * D]
    ws_inv: cute.Tensor,  # bf16 [H * total_tiles * CHUNK * CHUNK]
    ws_mqk: cute.Tensor,
    out: cute.Tensor,  # bf16 [T_total, H, D]
    H: cutlass.Constexpr[int],
    total_tiles: cutlass.Constexpr[int],
    T_total: cutlass.Constexpr[int],
    seq_len: cutlass.Constexpr[int],
):
    seq_idx, head_idx, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    smem = cutlass.utils.SmemAllocator()
    qk_layout = cute.make_layout((CHUNK, D), stride=(D, 1))
    cc_layout = cute.make_layout((CHUNK, CHUNK), stride=(CHUNK, 1))
    state_layout = cute.make_layout((D, D), stride=(D, 1))

    sState = smem.allocate_tensor(cutlass.BFloat16, state_layout, 128)
    sV = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sKd = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sQd = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sKr = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sU = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)  # [CHUNK,D]
    sTmp = smem.allocate_tensor(cutlass.Float32, qk_layout, 128)  # [CHUNK,D] fp32 scratch
    sOut = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)  # [CHUNK,D] bf16 out tile
    sINV = smem.allocate_tensor(cutlass.BFloat16, cc_layout, 128)
    sMqk = smem.allocate_tensor(cutlass.BFloat16, cc_layout, 128)
    sGt = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D,)), 128)
    sBeta = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((CHUNK,)), 128)

    bos = seq_idx * seq_len
    t_tiles: cutlass.Constexpr[int] = (seq_len + CHUNK - 1) // CHUNK

    # ---- Initialize state to zero ----
    # 128 threads, D*D = 16384 elements -> 128 each
    for e in cutlass.range_constexpr(D):
        sState[tidx, e] = cutlass.BFloat16(0.0)
    cute.arch.barrier()

    # ---- Tile loop ----
    for t in cutlass.range(t_tiles, unroll=1):
        tile_off_qk = (head_idx * total_tiles + seq_idx * t_tiles + t) * (CHUNK * D)
        tile_off_cc = (head_idx * total_tiles + seq_idx * t_tiles + t) * (CHUNK * CHUNK)
        tile_off_gt = (head_idx * total_tiles + seq_idx * t_tiles + t) * D

        # Load v, kd, qd, kr [CHUNK, D] = 2048 elements / 128t = 16 each.
        # Access pattern: row = tidx//8, col_base = (tidx%8)*16
        row16 = tidx // 8
        col_base = (tidx % 8) * 16
        for e in cutlass.range_constexpr(16):
            ee: cutlass.Constexpr[int] = e
            v_row = bos + t * CHUNK + row16
            sV[row16, col_base + ee] = v[v_row, head_idx, col_base + ee]
            sKd[row16, col_base + ee] = ws_kd[tile_off_qk + row16 * D + col_base + ee]
            sQd[row16, col_base + ee] = ws_qd[tile_off_qk + row16 * D + col_base + ee]
            sKr[row16, col_base + ee] = ws_kr[tile_off_qk + row16 * D + col_base + ee]

        # Load gt [D] / 128t = 1 each
        sGt[tidx] = ws_gt[tile_off_gt + tidx]

        # Load INV, Mqk [CHUNK, CHUNK] = 256 / 128t = 2 each
        for e in cutlass.range_constexpr(2):
            ee: cutlass.Constexpr[int] = e
            idx = tidx + ee * 128
            r = idx // CHUNK
            c = idx % CHUNK
            sINV[r, c] = ws_inv[tile_off_cc + r * CHUNK + c]
            sMqk[r, c] = ws_mqk[tile_off_cc + r * CHUNK + c]

        # Load beta [CHUNK] / 128t — only first 16 threads do it
        if tidx < CHUNK:
            sBeta[tidx] = beta[head_idx * T_total + bos + t * CHUNK + tidx]

        cute.arch.barrier()

        # ---- 1) tmp = kd @ state into sTmp [CHUNK,D] (fp32) ----
        # 2048 outputs / 128t = 16 per thread; same row16/col_base mapping.
        for e in cutlass.range_constexpr(16):
            ee: cutlass.Constexpr[int] = e
            acc = cutlass.Float32(0.0)
            for kk in cutlass.range(D, unroll=8):
                acc = acc + cutlass.Float32(sKd[row16, kk]) * cutlass.Float32(sState[kk, col_base + ee])
            sTmp[row16, col_base + ee] = acc
        cute.arch.barrier()

        # u = (v - tmp) * sigmoid(beta[row]) into sU
        for e in cutlass.range_constexpr(16):
            ee: cutlass.Constexpr[int] = e
            bv = cutlass.Float32(sBeta[row16])
            sig_b = cutlass.Float32(0.5) * (cute.tanh(bv * cutlass.Float32(0.5), fastmath=True) + cutlass.Float32(1.0))
            diff = cutlass.Float32(sV[row16, col_base + ee]) - sTmp[row16, col_base + ee]
            sU[row16, col_base + ee] = cutlass.BFloat16(diff * sig_b)
        cute.arch.barrier()

        # ---- 2) u = INV @ u (write into sTmp_bf16 then back to sU) ----
        # We'll reuse sTmp as scratch [CHUNK,D] fp32.
        for e in cutlass.range_constexpr(16):
            ee: cutlass.Constexpr[int] = e
            acc = cutlass.Float32(0.0)
            for kk in cutlass.range_constexpr(CHUNK):
                acc = acc + cutlass.Float32(sINV[row16, kk]) * cutlass.Float32(sU[kk, col_base + ee])
            sTmp[row16, col_base + ee] = acc
        cute.arch.barrier()
        for e in cutlass.range_constexpr(16):
            ee: cutlass.Constexpr[int] = e
            sU[row16, col_base + ee] = cutlass.BFloat16(sTmp[row16, col_base + ee])
        cute.arch.barrier()

        # ---- 3) out0 = qd @ state, out = out0 + Mqk @ u, into sOut ----
        for e in cutlass.range_constexpr(16):
            ee: cutlass.Constexpr[int] = e
            acc = cutlass.Float32(0.0)
            for kk in cutlass.range(D, unroll=8):
                acc = acc + cutlass.Float32(sQd[row16, kk]) * cutlass.Float32(sState[kk, col_base + ee])
            mqk_acc = cutlass.Float32(0.0)
            for kk in cutlass.range_constexpr(CHUNK):
                mqk_acc = mqk_acc + cutlass.Float32(sMqk[row16, kk]) * cutlass.Float32(sU[kk, col_base + ee])
            sOut[row16, col_base + ee] = cutlass.BFloat16(acc + mqk_acc)
        cute.arch.barrier()

        # ---- 4) state = state * gt[d_k] + kr.T @ u ----
        # 16384 elements / 128t = 128 per thread.
        # Mapping: each thread owns row d=tidx, all cols 0..127.
        for c in cutlass.range(D, unroll=8):
            cc: cutlass.Constexpr[int] = c
            acc = cutlass.Float32(sState[tidx, cc]) * sGt[tidx]
            for kk in cutlass.range_constexpr(CHUNK):
                acc = acc + cutlass.Float32(sKr[kk, tidx]) * cutlass.Float32(sU[kk, cc])
            sState[tidx, cc] = cutlass.BFloat16(acc)
        cute.arch.barrier()

        # ---- 5) Store sOut to gmem ----
        for e in cutlass.range_constexpr(16):
            ee: cutlass.Constexpr[int] = e
            o_row = bos + t * CHUNK + row16
            out[o_row, head_idx, col_base + ee] = sOut[row16, col_base + ee]
        cute.arch.barrier()


@cute.jit
def run_k2_phaseA(
    v: cute.Tensor,
    beta: cute.Tensor,
    ws_qd: cute.Tensor,
    ws_kd: cute.Tensor,
    ws_kr: cute.Tensor,
    ws_gt: cute.Tensor,
    ws_inv: cute.Tensor,
    ws_mqk: cute.Tensor,
    out: cute.Tensor,
    H: cutlass.Constexpr[int],
    total_tiles: cutlass.Constexpr[int],
    T_total: cutlass.Constexpr[int],
    seq_len: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    stream: cuda_drv.CUstream,
):
    smem_bytes = (
        D * D * 2  # state
        + 5 * (CHUNK * D * 2)  # v, kd, qd, kr, U bf16
        + (CHUNK * D * 4)  # tmp fp32
        + (CHUNK * D * 2)  # out bf16
        + 2 * (CHUNK * CHUNK * 2)  # INV, Mqk
        + (D * 4)  # gt
        + (CHUNK * 2)  # beta
        + 256
    )
    k2_phaseA_kernel(
        v,
        beta,
        ws_qd,
        ws_kd,
        ws_kr,
        ws_gt,
        ws_inv,
        ws_mqk,
        out,
        H,
        total_tiles,
        T_total,
        seq_len,
    ).launch(
        grid=(N, H, 1),
        block=[THREADS_PER_CTA, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


_compiled_cache_k2A: dict = {}


def launch_k2_phaseA(
    v: torch.Tensor,
    beta: torch.Tensor,
    ws_qd: torch.Tensor,
    ws_kd: torch.Tensor,
    ws_kr: torch.Tensor,
    ws_gt: torch.Tensor,
    ws_inv: torch.Tensor,
    ws_mqk: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Run K2 Phase A (correctness-only). Inputs from K1 full workspace.
    v, beta, out follow user-facing layout: v[B,T,H,D], out[B,T,H,D],
    beta_flat is [H*T_total] (head-major)."""
    assert v.is_cuda and v.dtype == torch.bfloat16 and v.is_contiguous()
    assert out.is_cuda and out.dtype == torch.bfloat16 and out.is_contiguous()
    B, T, H, K = v.shape
    assert K == D and T % CHUNK == 0
    T_total = B * T
    seq_len = T
    total_tiles = T_total // CHUNK

    key = (B, T, H)
    if key not in _compiled_cache_k2A:
        stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
        v_flat = v.view(T_total, H, D)
        out_flat = out.view(T_total, H, D)
        _compiled_cache_k2A[key] = cute.compile(
            run_k2_phaseA,
            from_dlpack(v_flat.detach(), assumed_align=16),
            from_dlpack(beta.detach(), assumed_align=16),
            from_dlpack(ws_qd.detach(), assumed_align=16),
            from_dlpack(ws_kd.detach(), assumed_align=16),
            from_dlpack(ws_kr.detach(), assumed_align=16),
            from_dlpack(ws_gt.detach(), assumed_align=16),
            from_dlpack(ws_inv.detach(), assumed_align=16),
            from_dlpack(ws_mqk.detach(), assumed_align=16),
            from_dlpack(out_flat.detach(), assumed_align=16),
            H=H,
            total_tiles=total_tiles,
            T_total=T_total,
            seq_len=seq_len,
            N=B,
            stream=stream,
        )

    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    v_flat = v.view(T_total, H, D)
    out_flat = out.view(T_total, H, D)
    _compiled_cache_k2A[key](
        v_flat,
        beta,
        ws_qd,
        ws_kd,
        ws_kr,
        ws_gt,
        ws_inv,
        ws_mqk,
        out_flat,
        H,
        total_tiles,
        T_total,
        seq_len,
        B,
        stream,
    )


# ===========================================================================
# Phase B: TMA loads + TMA store. Same single-role / no warp-spec layout.
# Replaces all per-element gmem reads/writes for the heavy [CHUNK,D] and
# [CHUNK,CHUNK] tiles with TMA bulk transfers. Math kernel body identical to
# Phase A. Single mbarrier with phase-parity reuse across the tile loop.
# ===========================================================================
@cute.kernel
def k2_phaseB_kernel(
    tma_atom_v: cute.CopyAtom,
    tma_tensor_v: cute.Tensor,
    tma_atom_kd: cute.CopyAtom,
    tma_tensor_kd: cute.Tensor,
    tma_atom_qd: cute.CopyAtom,
    tma_tensor_qd: cute.Tensor,
    tma_atom_kr: cute.CopyAtom,
    tma_tensor_kr: cute.Tensor,
    tma_atom_inv: cute.CopyAtom,
    tma_tensor_inv: cute.Tensor,
    tma_atom_mqk: cute.CopyAtom,
    tma_tensor_mqk: cute.Tensor,
    tma_atom_out: cute.CopyAtom,
    tma_tensor_out: cute.Tensor,
    beta: cute.Tensor,
    ws_gt: cute.Tensor,
    H: cutlass.Constexpr[int],
    total_tiles: cutlass.Constexpr[int],
    T_total: cutlass.Constexpr[int],
    seq_len: cutlass.Constexpr[int],
):
    seq_idx, head_idx, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    smem = cutlass.utils.SmemAllocator()
    qk_layout = cute.make_layout((CHUNK, D), stride=(D, 1))
    cc_layout = cute.make_layout((CHUNK, CHUNK), stride=(CHUNK, 1))
    state_layout = cute.make_layout((D, D), stride=(D, 1))

    # NOTE: allocate TMA destination tiles BEFORE large state, so swizzle
    # offsets line up with TMA descriptor expectations.
    sV = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sKd = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sQd = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sKr = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sINV = smem.allocate_tensor(cutlass.BFloat16, cc_layout, 128)
    sMqk = smem.allocate_tensor(cutlass.BFloat16, cc_layout, 128)
    sOut = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sState = smem.allocate_tensor(cutlass.BFloat16, state_layout, 128)
    sU = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sTmp = smem.allocate_tensor(cutlass.Float32, qk_layout, 128)
    sGt = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D,)), 128)
    sBeta = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((CHUNK,)), 128)
    sMbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((1,)), 8)
    sMbar_ptr = sMbar.iterator

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(sMbar_ptr, cutlass.Int32(1))
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    gSrc_v = cute.local_tile(tma_tensor_v, (CHUNK, D), (None, None, None))
    tVs, tVg = cpasync.tma_partition(
        tma_atom_v,
        0,
        cute.make_layout(1),
        cute.group_modes(sV, 0, 2),
        cute.group_modes(gSrc_v, 0, 2),
    )
    gSrc_kd = cute.local_tile(tma_tensor_kd, (CHUNK, D), (None, None, None))
    tKDs, tKDg = cpasync.tma_partition(
        tma_atom_kd,
        0,
        cute.make_layout(1),
        cute.group_modes(sKd, 0, 2),
        cute.group_modes(gSrc_kd, 0, 2),
    )
    gSrc_qd = cute.local_tile(tma_tensor_qd, (CHUNK, D), (None, None, None))
    tQDs, tQDg = cpasync.tma_partition(
        tma_atom_qd,
        0,
        cute.make_layout(1),
        cute.group_modes(sQd, 0, 2),
        cute.group_modes(gSrc_qd, 0, 2),
    )
    gSrc_kr = cute.local_tile(tma_tensor_kr, (CHUNK, D), (None, None, None))
    tKRs, tKRg = cpasync.tma_partition(
        tma_atom_kr,
        0,
        cute.make_layout(1),
        cute.group_modes(sKr, 0, 2),
        cute.group_modes(gSrc_kr, 0, 2),
    )
    gSrc_inv = cute.local_tile(tma_tensor_inv, (CHUNK, CHUNK), (None, None, None))
    tIs, tIg = cpasync.tma_partition(
        tma_atom_inv,
        0,
        cute.make_layout(1),
        cute.group_modes(sINV, 0, 2),
        cute.group_modes(gSrc_inv, 0, 2),
    )
    gSrc_mqk = cute.local_tile(tma_tensor_mqk, (CHUNK, CHUNK), (None, None, None))
    tMs, tMg = cpasync.tma_partition(
        tma_atom_mqk,
        0,
        cute.make_layout(1),
        cute.group_modes(sMqk, 0, 2),
        cute.group_modes(gSrc_mqk, 0, 2),
    )
    gDst_o = cute.local_tile(tma_tensor_out, (CHUNK, D), (None, None, None))
    tOs, tOg = cpasync.tma_partition(
        tma_atom_out,
        0,
        cute.make_layout(1),
        cute.group_modes(sOut, 0, 2),
        cute.group_modes(gDst_o, 0, 2),
    )

    for e in cutlass.range_constexpr(D):
        sState[tidx, e] = cutlass.BFloat16(0.0)
    cute.arch.barrier()

    bos = seq_idx * seq_len
    t_tiles: cutlass.Constexpr[int] = (seq_len + CHUNK - 1) // CHUNK
    TMA_BYTES: cutlass.Constexpr[int] = 4 * CHUNK * D * 2 + 2 * CHUNK * CHUNK * 2

    row16 = tidx // 8
    col_base = (tidx % 8) * 16

    phase = cutlass.Int32(0)

    for t in cutlass.range(t_tiles, unroll=1):
        t_g = seq_idx * t_tiles + t
        ws_t = head_idx * total_tiles + t_g
        gt_off = ws_t * D

        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(sMbar_ptr, cutlass.Int32(TMA_BYTES))
            cute.copy(tma_atom_v, tVg[(None, t_g, 0, head_idx)], tVs[(None,)], tma_bar_ptr=sMbar_ptr)
            cute.copy(tma_atom_kd, tKDg[(None, 0, 0, ws_t)], tKDs[(None,)], tma_bar_ptr=sMbar_ptr)
            cute.copy(tma_atom_qd, tQDg[(None, 0, 0, ws_t)], tQDs[(None,)], tma_bar_ptr=sMbar_ptr)
            cute.copy(tma_atom_kr, tKRg[(None, 0, 0, ws_t)], tKRs[(None,)], tma_bar_ptr=sMbar_ptr)
            cute.copy(tma_atom_inv, tIg[(None, 0, 0, ws_t)], tIs[(None,)], tma_bar_ptr=sMbar_ptr)
            cute.copy(tma_atom_mqk, tMg[(None, 0, 0, ws_t)], tMs[(None,)], tma_bar_ptr=sMbar_ptr)

        sGt[tidx] = ws_gt[gt_off + tidx]
        if tidx < CHUNK:
            sBeta[tidx] = beta[head_idx * T_total + bos + t * CHUNK + tidx]

        cute.arch.mbarrier_wait(sMbar_ptr, phase)
        phase = phase ^ cutlass.Int32(1)

        # ----- Fused phase 1+2: U = sigmoid(beta) * (V - sKd @ sState) -----
        # Same-thread (row16, col_base+ee) → no need to round-trip through sTmp.
        bv = cutlass.Float32(sBeta[row16])
        sig_b = cutlass.Float32(0.5) * (cute.tanh(bv * cutlass.Float32(0.5), fastmath=True) + cutlass.Float32(1.0))
        for e in cutlass.range_constexpr(16):
            ee: cutlass.Constexpr[int] = e
            acc = cutlass.Float32(0.0)
            for kk in cutlass.range(D, unroll=8):
                acc = acc + cutlass.Float32(sKd[row16, kk]) * cutlass.Float32(sState[kk, col_base + ee])
            diff = cutlass.Float32(sV[row16, col_base + ee]) - acc
            sU[row16, col_base + ee] = cutlass.BFloat16(diff * sig_b)
        cute.arch.barrier()

        # ----- Fused phase 3 (INV @ U) and bf16 cast: hold U-row in regs -----
        # Need to read all 16 sU rows BEFORE we overwrite our own row.
        u_new = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), cutlass.Float32)
        for e in cutlass.range_constexpr(16):
            ee: cutlass.Constexpr[int] = e
            acc = cutlass.Float32(0.0)
            for kk in cutlass.range_constexpr(CHUNK):
                acc = acc + cutlass.Float32(sINV[row16, kk]) * cutlass.Float32(sU[kk, col_base + ee])
            u_new[ee] = acc
        cute.arch.barrier()
        for e in cutlass.range_constexpr(16):
            ee: cutlass.Constexpr[int] = e
            sU[row16, col_base + ee] = cutlass.BFloat16(u_new[ee])
        cute.arch.barrier()

        # ----- Phase 4: O = sQd @ sState + sMqk @ U -----
        for e in cutlass.range_constexpr(16):
            ee: cutlass.Constexpr[int] = e
            acc = cutlass.Float32(0.0)
            for kk in cutlass.range(D, unroll=8):
                acc = acc + cutlass.Float32(sQd[row16, kk]) * cutlass.Float32(sState[kk, col_base + ee])
            mqk_acc = cutlass.Float32(0.0)
            for kk in cutlass.range_constexpr(CHUNK):
                mqk_acc = mqk_acc + cutlass.Float32(sMqk[row16, kk]) * cutlass.Float32(sU[kk, col_base + ee])
            sOut[row16, col_base + ee] = cutlass.BFloat16(acc + mqk_acc)
        cute.arch.barrier()

        for c in cutlass.range(D, unroll=8):
            cc: cutlass.Constexpr[int] = c
            acc = cutlass.Float32(sState[tidx, cc]) * sGt[tidx]
            for kk in cutlass.range_constexpr(CHUNK):
                acc = acc + cutlass.Float32(sKr[kk, tidx]) * cutlass.Float32(sU[kk, cc])
            sState[tidx, cc] = cutlass.BFloat16(acc)
        cute.arch.barrier()

        # TMA store sOut
        if warp_idx == 0:
            cute.copy(tma_atom_out, tOs[(None,)], tOg[(None, t_g, 0, head_idx)])
            cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0, read=True)
        cute.arch.barrier()


@cute.jit
def run_k2_phaseB(
    v: cute.Tensor,
    beta: cute.Tensor,
    ws_qd: cute.Tensor,
    ws_kd: cute.Tensor,
    ws_kr: cute.Tensor,
    ws_gt: cute.Tensor,
    ws_inv: cute.Tensor,
    ws_mqk: cute.Tensor,
    out: cute.Tensor,
    H: cutlass.Constexpr[int],
    total_tiles: cutlass.Constexpr[int],
    T_total: cutlass.Constexpr[int],
    seq_len: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    stream: cuda_drv.CUstream,
):
    qk_smem = cute.make_layout((CHUNK, D), stride=(D, 1))
    cc_smem = cute.make_layout((CHUNK, CHUNK), stride=(CHUNK, 1))

    def make_thd_atom(t, op):
        view = cute.make_tensor(
            t.iterator,
            cute.make_layout((T_total, D, H), stride=(H * D, 1, D)),
        )
        return cpasync.make_tiled_tma_atom(op, view, qk_smem, (CHUNK, D))

    def make_ws_qkd_atom(t):
        view = cute.make_tensor(
            t.iterator,
            cute.make_layout((CHUNK, D, total_tiles * H), stride=(D, 1, CHUNK * D)),
        )
        return cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), view, qk_smem, (CHUNK, D))

    def make_ws_cc_atom(t):
        view = cute.make_tensor(
            t.iterator,
            cute.make_layout((CHUNK, CHUNK, total_tiles * H), stride=(CHUNK, 1, CHUNK * CHUNK)),
        )
        return cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), view, cc_smem, (CHUNK, CHUNK))

    tma_atom_v, tma_tensor_v = make_thd_atom(v, cpasync.CopyBulkTensorTileG2SOp())
    tma_atom_out, tma_tensor_out = make_thd_atom(out, cpasync.CopyBulkTensorTileS2GOp())
    tma_atom_kd, tma_tensor_kd = make_ws_qkd_atom(ws_kd)
    tma_atom_qd, tma_tensor_qd = make_ws_qkd_atom(ws_qd)
    tma_atom_kr, tma_tensor_kr = make_ws_qkd_atom(ws_kr)
    tma_atom_inv, tma_tensor_inv = make_ws_cc_atom(ws_inv)
    tma_atom_mqk, tma_tensor_mqk = make_ws_cc_atom(ws_mqk)

    smem_bytes = (
        D * D * 2
        + 5 * (CHUNK * D * 2)
        + (CHUNK * D * 4)
        + (CHUNK * D * 2)
        + 2 * (CHUNK * CHUNK * 2)
        + (D * 4)
        + (CHUNK * 2)
        + 16
        + 256
    )

    k2_phaseB_kernel(
        tma_atom_v,
        tma_tensor_v,
        tma_atom_kd,
        tma_tensor_kd,
        tma_atom_qd,
        tma_tensor_qd,
        tma_atom_kr,
        tma_tensor_kr,
        tma_atom_inv,
        tma_tensor_inv,
        tma_atom_mqk,
        tma_tensor_mqk,
        tma_atom_out,
        tma_tensor_out,
        beta,
        ws_gt,
        H,
        total_tiles,
        T_total,
        seq_len,
    ).launch(
        grid=(N, H, 1),
        block=[THREADS_PER_CTA, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


_compiled_cache_k2B: dict = {}


def launch_k2_phaseB(
    v: torch.Tensor,
    beta: torch.Tensor,
    ws_qd: torch.Tensor,
    ws_kd: torch.Tensor,
    ws_kr: torch.Tensor,
    ws_gt: torch.Tensor,
    ws_inv: torch.Tensor,
    ws_mqk: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Run K2 Phase B (TMA loads + TMA store, single-role)."""
    assert v.is_cuda and v.dtype == torch.bfloat16 and v.is_contiguous()
    assert out.is_cuda and out.dtype == torch.bfloat16 and out.is_contiguous()
    B, T, H, K = v.shape
    assert K == D and T % CHUNK == 0
    T_total = B * T
    seq_len = T
    total_tiles = T_total // CHUNK

    key = (B, T, H)
    if key not in _compiled_cache_k2B:
        stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
        v_flat = v.view(T_total, H, D)
        out_flat = out.view(T_total, H, D)
        _compiled_cache_k2B[key] = cute.compile(
            run_k2_phaseB,
            from_dlpack(v_flat.detach(), assumed_align=16),
            from_dlpack(beta.detach(), assumed_align=16),
            from_dlpack(ws_qd.detach(), assumed_align=16),
            from_dlpack(ws_kd.detach(), assumed_align=16),
            from_dlpack(ws_kr.detach(), assumed_align=16),
            from_dlpack(ws_gt.detach(), assumed_align=16),
            from_dlpack(ws_inv.detach(), assumed_align=16),
            from_dlpack(ws_mqk.detach(), assumed_align=16),
            from_dlpack(out_flat.detach(), assumed_align=16),
            H=H,
            total_tiles=total_tiles,
            T_total=T_total,
            seq_len=seq_len,
            N=B,
            stream=stream,
        )

    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    v_flat = v.view(T_total, H, D)
    out_flat = out.view(T_total, H, D)
    _compiled_cache_k2B[key](
        v_flat,
        beta,
        ws_qd,
        ws_kd,
        ws_kr,
        ws_gt,
        ws_inv,
        ws_mqk,
        out_flat,
        H,
        total_tiles,
        T_total,
        seq_len,
        B,
        stream,
    )

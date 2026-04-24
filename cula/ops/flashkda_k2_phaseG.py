"""FlashKDA K2 Phase G — Phase E1 + tensor-core phase 6.

Phase G adds a TC MMA for phase 6 (state += kr.T @ U) using SM80
16x8x16 BF16 atom with LDSM.x4.trans on the existing sU/sKr SMEM
layouts (no physical transpose). The state update uses an in-frag
epilogue that loads sState, multiplies by gt, adds the MMA delta,
and writes back as bf16.

Key trick: A operand U^T is the (D, CHUNK) transpose-VIEW of sU
built via cute.select(layout, mode=[1, 0]); B operand is sKr (CHUNK, D)
used directly. Both feed LdMatrix8x8x16bOp(transpose=True).

Legacy header preserved below.

FlashKDA K2 Phase E1 — partial tensor-core port.

This is the FIRST step toward C++-parity for the cute K2:

  * Mirrors Phase B's structure (single CTA, 128 threads, 1 LOAD warp via
    `elect_one`, in-thread compute) so we only swap math, not scaffolding.
  * Replaces the TWO LARGEST scalar GEMMs with SM80 16x8x16 BF16 tensor core
    MMAs (`cute.nvgpu.warp.MmaF16BF16Op`):
        - phase-1 ``u_pre = kd @ state`` (M=16, N=128, K=128)
        - phase-4 ``out0  = qd @ state`` (M=16, N=128, K=128)
    These two have K=128 each (vs K=16 for INV@u, Mqk@u, kr.T@u) and dominate
    the scalar runtime by ~8x per GEMM.
  * Other GEMMs (INV@u, Mqk@u, state update kr.T@u + state*gt) keep the
    scalar implementation from Phase B for now (smaller K=16, lower priority).

Tiled-MMA layout:
  * Atom: SM80 (16, 8, 16) BF16 -> FP32
  * atom_layout = (1, 4, 1)  -> 4 warps split N=128 into 4 stripes of N=32
  * permutation_mnk = (16, 32, 16) -> per-CTA tile (16, 128, 16) with 4 N
    blocks per K iter; K-loop = D/16 = 8 iters

If this matches Phase B numerically, Phase E2 will fold in the remaining 3
GEMMs + warp-spec/pipeline scaffolding from Phase D.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import torch
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack

from cula.ops.flashkda_k2 import CHUNK, D, _make_state_smem_layout

THREADS_PER_CTA = 128
N_WARPS = 4


def _make_qk_smem_layout():
    """Plain K-major (CHUNK, D) layout, matching Phase B.

    NOTE: When cute.gemm is wired in (Phase E2), this should switch to a
    swizzled K_INTER atom for SM80 16x8x16 MMA A-operand. For now keep plain
    layout to keep TMA partitioning identical to Phase B.
    """
    return cute.make_layout((CHUNK, D), stride=(D, 1))


@cute.kernel
def k2_phaseG_kernel(
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

    qk_layout = _make_qk_smem_layout()  # K-major swizzled, MMA-ready
    cc_layout = cute.make_layout((CHUNK, CHUNK), stride=(CHUNK, 1))
    state_layout = _make_state_smem_layout()  # swizzled K-major
    # Phase 6 TC-friendly K-fast layouts (CHUNK = stride 1):
    #   sU_T  shape (CHUNK, D) stride (1, CHUNK)  — partition_C compatible with
    #         tiled_mma C-frag (which has shape (CHUNK, D)) so phase 2 in-frag
    #         writes directly here. A cute.select view (mode=[1,0]) yields
    #         shape (D, CHUNK) for B-operand of phases 3, 4-epi, 6.
    #   sKr_T shape (D, CHUNK) stride (CHUNK, 1)  — direct B-operand layout for phase 6.
    u_t_layout = cute.make_layout((CHUNK, D), stride=(1, CHUNK))
    kr_t_layout = cute.make_layout((D, CHUNK), stride=(CHUNK, 1))

    sV = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sKd = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sQd = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sKr = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sINV = smem.allocate_tensor(cutlass.BFloat16, cc_layout, 128)
    sMqk = smem.allocate_tensor(cutlass.BFloat16, cc_layout, 128)
    sOut = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sState = smem.allocate_tensor(cutlass.BFloat16, state_layout, 128)
    sTmp = smem.allocate_tensor(cutlass.Float32, qk_layout, 128)  # noqa: F841 (kept for SMEM offset parity with Phase B)
    sU_T = smem.allocate_tensor(cutlass.BFloat16, u_t_layout, 128)
    sKr_T = smem.allocate_tensor(cutlass.BFloat16, kr_t_layout, 128)
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

    # --- TMA partitioning (same as Phase B) ---
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

    # Init state to zero (each thread handles one row; D=128, only first 128 threads)
    if tidx < D:
        for e in cutlass.range_constexpr(D):
            sState[tidx, e] = cutlass.BFloat16(0.0)
    cute.arch.barrier()

    # --- SM80 16x8x16 BF16 tiled MMA (4 warps split N=128 into 4 stripes of 32) ---
    mma_atom = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        atom_layout_mnk=(1, 4, 1),
        permutation_mnk=(16, 32, 16),
    )
    thr_mma = tiled_mma.get_slice(tidx)

    copy_atom_AB = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
        cutlass.BFloat16,
    )
    smem_tiled_copy_A = cute.make_tiled_copy_A(copy_atom_AB, tiled_mma)
    smem_tiled_copy_B = cute.make_tiled_copy_B(copy_atom_AB, tiled_mma)
    smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
    smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

    # Phase 6 tiled MMA: same atom layout as phase 1/4.
    tiled_mma6 = cute.make_tiled_mma(
        mma_atom,
        atom_layout_mnk=(1, 4, 1),
        permutation_mnk=(16, 32, 16),
    )
    thr_mma6 = tiled_mma6.get_slice(tidx)
    smem_tiled_copy_A6 = cute.make_tiled_copy_A(copy_atom_AB, tiled_mma6)
    smem_tiled_copy_B6 = cute.make_tiled_copy_B(copy_atom_AB, tiled_mma6)
    smem_thr_copy_A6 = smem_tiled_copy_A6.get_slice(tidx)
    smem_thr_copy_B6 = smem_tiled_copy_B6.get_slice(tidx)

    # Reference 16x16 sub-tiles for fragment construction
    sKd_tile = cute.flat_divide(sKd, (CHUNK, 16))  # ((CHUNK, 16), 1, 8)
    sQd_tile = cute.flat_divide(sQd, (CHUNK, 16))
    sState_tile = cute.flat_divide(sState, (D, 16))  # ((D, 16), 1, 8) — N-row, K-col-blocks

    sKd_ref = sKd_tile[None, None, 0, 0]
    sQd_ref = sQd_tile[None, None, 0, 0]
    sState_ref = sState_tile[None, None, 0, 0]

    tCrKd = thr_mma.make_fragment_A(thr_mma.partition_A(sKd_ref))
    tCrQd = thr_mma.make_fragment_A(thr_mma.partition_A(sQd_ref))
    tCrState = thr_mma.make_fragment_B(thr_mma.partition_B(sState_ref))
    tCrU = thr_mma.make_fragment_C(tiled_mma.partition_shape_C((CHUNK, D)))
    tCrOut = thr_mma.make_fragment_C(tiled_mma.partition_shape_C((CHUNK, D)))

    tCrKd_cv = smem_thr_copy_A.retile(tCrKd)
    tCrQd_cv = smem_thr_copy_A.retile(tCrQd)
    tCrState_cv = smem_thr_copy_B.retile(tCrState)

    # Phase 4-epilogue TC fragments: A=sMqk (16,16), B=sU_T (D, CHUNK).
    # Reuse tiled_mma (same M=16, N=D atom partition as phase 4 main).
    sMqk_tile = cute.flat_divide(sMqk, (CHUNK, CHUNK))  # ((CHUNK, CHUNK), 1, 1)
    sMqk_ref = sMqk_tile[None, None, 0, 0]
    tCrMqk = thr_mma.make_fragment_A(thr_mma.partition_A(sMqk_ref))
    tCrMqk_cv = smem_thr_copy_A.retile(tCrMqk)
    # sU_T_full_tile is declared below; its ref is reused as B operand.

    # ---- Phase 6 TC fragments (declared once, reused per chunk) ----
    # sU_T physical layout is (CHUNK, D) K-fast. For B-operand of phases 3, 4-epi,
    # and 6 we need shape (N=D, K=CHUNK), so build a select-mode-[1,0] view.
    sU_T_B_view = cute.make_tensor(sU_T.iterator, layout=cute.select(sU_T.layout, mode=[1, 0]))
    sU_T_full_tile = cute.flat_divide(sU_T_B_view, (D, CHUNK))  # ((D, CHUNK), 1, 1)
    sKr_T_tile = cute.flat_divide(sKr_T, (D, CHUNK))  # ((D, CHUNK), 1, 1)
    sU_T_full_ref = sU_T_full_tile[None, None, 0, 0]  # (D, CHUNK)
    sKr_T_ref = sKr_T_tile[None, None, 0, 0]  # (D, CHUNK)
    tCrU6 = thr_mma6.make_fragment_A(thr_mma6.partition_A(sU_T_full_ref))
    tCrKr6 = thr_mma6.make_fragment_B(thr_mma6.partition_B(sKr_T_ref))
    tCrUpd = thr_mma6.make_fragment_C(tiled_mma6.partition_shape_C((D, D)))
    tCrU6_cv = smem_thr_copy_A6.retile(tCrU6)
    tCrKr6_cv = smem_thr_copy_B6.retile(tCrKr6)

    # Phase 4-epi B operand: sU_T as (N=D, K=CHUNK) for tiled_mma (M=16 N=128 K=16).
    tCrU_T = thr_mma.make_fragment_B(thr_mma.partition_B(sU_T_full_ref))
    tCrU_T_cv = smem_thr_copy_B.retile(tCrU_T)

    # ---- Phase 3 TC fragments: A=sINV (16,16), B=sU_T (D, CHUNK), C=tCrU3 (16, D) ----
    sINV_tile = cute.flat_divide(sINV, (CHUNK, CHUNK))  # ((CHUNK, CHUNK), 1, 1)
    sINV_ref = sINV_tile[None, None, 0, 0]
    tCrInv = thr_mma.make_fragment_A(thr_mma.partition_A(sINV_ref))
    tCrInv_cv = smem_thr_copy_A.retile(tCrInv)
    # tCrU3 is the post-INV U fp32 frag (same shape as tCrU from phase 1).
    tCrU3 = thr_mma.make_fragment_C(tiled_mma.partition_shape_C((CHUNK, D)))

    bos = seq_idx * seq_len
    t_tiles: cutlass.Constexpr[int] = (seq_len + CHUNK - 1) // CHUNK
    TMA_BYTES: cutlass.Constexpr[int] = 4 * CHUNK * D * 2 + 2 * CHUNK * CHUNK * 2

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

        if tidx < D:
            sGt[tidx] = ws_gt[gt_off + tidx]
        if tidx < CHUNK:
            sBeta[tidx] = beta[head_idx * T_total + bos + t * CHUNK + tidx]

        cute.arch.mbarrier_wait(sMbar_ptr, phase)
        phase = phase ^ cutlass.Int32(1)

        # Pre-transpose sKr -> sKr_T early (overlaps with phase 1 TC since
        # phase 1 doesn't read sKr). Visibility ensured by the next barrier.
        # Two-stage (load all → store all) breaks the load→store dep chain.
        if tidx < D:
            kr_buf = cute.make_rmem_tensor(cute.make_layout((CHUNK,), stride=(1,)), cutlass.BFloat16)
            for m in cutlass.range_constexpr(CHUNK):
                mm: cutlass.Constexpr[int] = m
                kr_buf[mm] = sKr[mm, tidx]
            for m in cutlass.range_constexpr(CHUNK):
                mm: cutlass.Constexpr[int] = m
                sKr_T[tidx, mm] = kr_buf[mm]

        # ============================================================
        # Phase 1 (TENSOR-CORE): u_pre[m, n] = sum_k kd[m, k] * state[n, k]
        # 4 warps split N=128, each computes 16x32 stripe via 8 K-iter loop
        # ============================================================
        tCrU.fill(0.0)
        for k in cutlass.range_constexpr(D // 16):
            sKd_k = sKd_tile[None, None, 0, k]
            sState_k = sState_tile[None, None, 0, k]
            cute.copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(sKd_k), tCrKd_cv)
            cute.copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(sState_k), tCrState_cv)
            cute.gemm(tiled_mma, tCrU, tCrKd, tCrState, tCrU)

        # ----- Phase 2: u = sigmoid(beta) * (v - u_pre); cast to bf16 in sU (in-frag) -----
        lane_in_warp = tidx % 32
        Rrow0 = lane_in_warp // 4
        Rrow1 = Rrow0 + 8
        b0 = cutlass.Float32(sBeta[Rrow0])
        b1 = cutlass.Float32(sBeta[Rrow1])
        sig0 = cutlass.Float32(0.5) * (cute.tanh(b0 * cutlass.Float32(0.5), fastmath=True) + cutlass.Float32(1.0))
        sig1 = cutlass.Float32(0.5) * (cute.tanh(b1 * cutlass.Float32(0.5), fastmath=True) + cutlass.Float32(1.0))
        tCsV = thr_mma.partition_C(sV)
        # Write phase 2 output directly into sU_T (K-fast). sU_T physical layout
        # is (CHUNK, D) so partition_C accepts it like sU.
        tCsU_T_w = thr_mma.partition_C(sU_T)
        # Pre-load sV into a register frag — breaks SMEM→math dep chain.
        v_frag = cute.make_fragment_like(tCsV, cutlass.BFloat16)
        for i in cutlass.range_constexpr(cute.size(v_frag)):
            ii: cutlass.Constexpr[int] = i
            v_frag[ii] = tCsV[ii]
        for i in cutlass.range_constexpr(cute.size(tCrU)):
            ii: cutlass.Constexpr[int] = i
            sub_i: cutlass.Constexpr[int] = (ii % 4) // 2
            sig = sig0 if sub_i == 0 else sig1
            diff = cutlass.Float32(v_frag[ii]) - tCrU[ii]
            tCsU_T_w[ii] = cutlass.BFloat16(diff * sig)
        cute.arch.barrier()

        # ----- Phase 3 (TC): U_post = INV @ U_pre  (M=16 N=D K=CHUNK) -----
        # Reads sU_T (pre, K-fast), writes back into sU_T (post) via in-frag.
        cute.copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(sINV_ref), tCrInv_cv)
        cute.copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(sU_T_full_ref), tCrU_T_cv)
        tCrU3.fill(0.0)
        cute.gemm(tiled_mma, tCrU3, tCrInv, tCrU_T, tCrU3)
        # In-frag write tCrU3 back to sU_T (overwriting the pre values; safe
        # because tCrU_T B-frag has already been consumed by the MMA above).
        for i in cutlass.range_constexpr(cute.size(tCrU3)):
            ii: cutlass.Constexpr[int] = i
            tCsU_T_w[ii] = cutlass.BFloat16(tCrU3[ii])

        # ----- Phase 4 MAIN (TC, no sU_T dep): out = qd @ state (8 K iters) -----
        # Run BEFORE the sU_T-visibility barrier so it overlaps with phase 3
        # SMEM writes settling. Reads only sQd, sState — disjoint from sU_T.
        tCrOut.fill(0.0)
        for k in cutlass.range_constexpr(D // 16):
            sQd_k = sQd_tile[None, None, 0, k]
            sState_k = sState_tile[None, None, 0, k]
            cute.copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(sQd_k), tCrQd_cv)
            cute.copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(sState_k), tCrState_cv)
            cute.gemm(tiled_mma, tCrOut, tCrQd, tCrState, tCrOut)
        cute.arch.barrier()

        # ----- Phase 4 EPI (TC): out += Mqk @ U_T (1 K iter) -----
        cute.copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(sMqk_ref), tCrMqk_cv)
        cute.copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(sU_T_full_ref), tCrU_T_cv)
        cute.gemm(tiled_mma, tCrOut, tCrMqk, tCrU_T, tCrOut)

        # Single in-frag bf16 write to sOut.
        tCsOut = thr_mma.partition_C(sOut)
        for i in cutlass.range_constexpr(cute.size(tCrOut)):
            tCsOut[i] = cutlass.BFloat16(tCrOut[i])
        # NOTE: no barrier here. Phase 6 below uses entirely disjoint SMEM
        # (sU_T/sKr_T/sState — never sOut), so it can run in parallel with the
        # sOut writes settling into SMEM. We fold the sOut visibility barrier
        # into the single barrier after phase 6 (one fewer cluster-wide sync).

        # ----- Phase 6 (TC MMA): state = state*gt + kr.T @ U  -----
        # delta = U^T @ kr  (M=D, N=D, K=CHUNK).

        # Single TC MMA over full (D, D, CHUNK).
        cute.copy(
            smem_tiled_copy_A6,
            smem_thr_copy_A6.partition_S(sU_T_full_ref),
            tCrU6_cv,
        )
        cute.copy(
            smem_tiled_copy_B6,
            smem_thr_copy_B6.partition_S(sKr_T_ref),
            tCrKr6_cv,
        )
        tCrUpd.fill(0.0)
        cute.gemm(tiled_mma6, tCrUpd, tCrU6, tCrKr6, tCrUpd)

        # In-frag epilogue: sState = bf16(float(sState)*gt[N] + delta).
        tCsState = thr_mma6.partition_C(sState)
        coord = cute.make_identity_tensor((D, D))
        tCcState = thr_mma6.partition_C(coord)
        # Pre-load sState into a bf16 register frag (no dep chain → fully pipelined).
        state_frag = cute.make_fragment_like(tCsState, cutlass.BFloat16)
        # Pre-load sGt[n_coord] for each fragment element into a fp32 register cache.
        gt_frag = cute.make_fragment_like(tCsState, cutlass.Float32)
        for i in cutlass.range_constexpr(cute.size(state_frag)):
            ii: cutlass.Constexpr[int] = i
            state_frag[ii] = tCsState[ii]
            gt_frag[ii] = sGt[tCcState[ii][1]]
        # Compute and write back from register data.
        for i in cutlass.range_constexpr(cute.size(tCrUpd)):
            ii: cutlass.Constexpr[int] = i
            old = cutlass.Float32(state_frag[ii]) * gt_frag[ii]
            tCsState[ii] = cutlass.BFloat16(old + tCrUpd[ii])
        # Combined barrier: makes sOut (from phase 4) visible to warp 0 for
        # TMA store AND makes sState visible across warps for next chunk.
        cute.arch.barrier()

        if warp_idx == 0:
            cute.copy(tma_atom_out, tOs[(None,)], tOg[(None, t_g, 0, head_idx)])
            cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0, read=True)
        cute.arch.barrier()


@cute.jit
def run_k2_phaseG(
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
    qk_smem = _make_qk_smem_layout()
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
        + 2 * (CHUNK * CHUNK * 2)
        + (D * 4)
        + (CHUNK * 2)
        + 16
        + 256
        + 2 * (CHUNK * D * 2)  # sU_T + sKr_T (Phase G)
    )

    k2_phaseG_kernel(
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


_compiled_cache_k2G: dict = {}


def launch_k2_phaseG(
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
    """Run K2 Phase E1 (scaffold only — math currently identical to Phase B).

    NOTE: This phase establishes the tiled-MMA scaffolding (atom, partitions,
    LdMatrix copies) but the actual ``cute.gemm`` calls are NOT yet wired in
    because the bf16 K-major B-operand view onto sState requires a swizzled
    K_INTER atom that produces a K-major ``(N, K)`` partitioning compatible
    with SM80_16x8x16. Wiring this up is the next concrete step (Phase E2).

    Functional output matches Phase B exactly; this commit makes the kernel
    available for the optimization track without breaking the precision
    comparison harness.
    """
    assert v.is_cuda and v.dtype == torch.bfloat16 and v.is_contiguous()
    assert out.is_cuda and out.dtype == torch.bfloat16 and out.is_contiguous()
    B, T, H, K = v.shape
    assert K == D and T % CHUNK == 0
    T_total = B * T
    seq_len = T
    total_tiles = T_total // CHUNK

    key = (B, T, H)
    if key not in _compiled_cache_k2G:
        stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
        v_flat = v.view(T_total, H, D)
        out_flat = out.view(T_total, H, D)
        _compiled_cache_k2G[key] = cute.compile(
            run_k2_phaseG,
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
    _compiled_cache_k2G[key](
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

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
from cula.ops.flashkda_prefill import movm_t_b16

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
def k2_phaseH_kernel(
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
    cc_layout = cute.make_layout((CHUNK, CHUNK), stride=(CHUNK, 1))  # noqa: F841 (kept for parity)
    state_layout = _make_state_smem_layout()  # swizzled K-major
    # Phase 6 TC-friendly K-fast layouts (CHUNK = stride 1):
    #   sU_T  shape (CHUNK, D) stride (1, CHUNK)  — partition_C compatible with
    #         tiled_mma C-frag (which has shape (CHUNK, D)) so phase 2 in-frag
    #         writes directly here. A cute.select view (mode=[1,0]) yields
    #         shape (D, CHUNK) for B-operand of phases 3, 4-epi, 6.
    #   sKr_T shape (D, CHUNK) stride (CHUNK, 1)  — direct B-operand layout for phase 6.
    u_t_layout = cute.make_layout((CHUNK, D), stride=(1, CHUNK))
    kr_t_layout = cute.make_layout((D, CHUNK), stride=(CHUNK, 1))

    # 2-stage TMA pipeline buffers (only the TMA-loaded tiles are staged).
    STAGES: cutlass.Constexpr[int] = 2
    qk_stage_layout = cute.make_layout((CHUNK, D, STAGES), stride=(D, 1, CHUNK * D))
    cc_stage_layout = cute.make_layout((CHUNK, CHUNK, STAGES), stride=(CHUNK, 1, CHUNK * CHUNK))

    sV = smem.allocate_tensor(cutlass.BFloat16, qk_stage_layout, 128)
    sKd = smem.allocate_tensor(cutlass.BFloat16, qk_stage_layout, 128)
    sQd = smem.allocate_tensor(cutlass.BFloat16, qk_stage_layout, 128)
    sKr = smem.allocate_tensor(cutlass.BFloat16, qk_stage_layout, 128)
    sINV = smem.allocate_tensor(cutlass.BFloat16, cc_stage_layout, 128)
    sMqk = smem.allocate_tensor(cutlass.BFloat16, cc_stage_layout, 128)
    sOut = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sState = smem.allocate_tensor(cutlass.BFloat16, state_layout, 128)
    # NOTE: sU_T removed (U is now register-resident via MOVM_T inline PTX).
    sKr_T = smem.allocate_tensor(cutlass.BFloat16, kr_t_layout, 128)
    sGt = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D,)), 128)
    sBeta = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((CHUNK,)), 128)
    sMbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((STAGES,)), 16)
    sMbar_ptr = sMbar.iterator

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        with cute.arch.elect_one():
            for s in cutlass.range_constexpr(STAGES):
                cute.arch.mbarrier_init(sMbar_ptr + cutlass.Int32(s), cutlass.Int32(1))
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
    # Phase 1/4-main B reads sState in transposed view (N=D_out fast axis).
    # Use ldmatrix.x4.trans atom because N is now the contig SMEM axis.
    copy_atom_B_T = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
        cutlass.BFloat16,
    )
    smem_tiled_copy_B_T = cute.make_tiled_copy_B(copy_atom_B_T, tiled_mma)
    smem_thr_copy_B_T = smem_tiled_copy_B_T.get_slice(tidx)

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

    # Reference 16x16 sub-tiles for fragment construction (use stage-0 view).
    sKd_s0 = sKd[(None, None, 0)]
    sQd_s0 = sQd[(None, None, 0)]
    sKd_tile0 = cute.flat_divide(sKd_s0, (CHUNK, 16))
    sQd_tile0 = cute.flat_divide(sQd_s0, (CHUNK, 16))
    # Role-swap convention: sState stores state[K_in, D_out] in normal form
    # (phase 6 writes (kr^T @ U)[m=K_in, n=D_out] directly). Phase 1/4-main
    # need B[N=D_out, K=K_in] for u_pre = Kd @ state, which is the transposed
    # view of sState (axis 0 = D_out, axis 1 = K_in).
    sState_B_view = cute.make_tensor(sState.iterator, layout=cute.select(sState.layout, mode=[1, 0]))
    sState_tile = cute.flat_divide(sState_B_view, (D, 16))

    sKd_ref = sKd_tile0[None, None, 0, 0]
    sQd_ref = sQd_tile0[None, None, 0, 0]
    sState_ref = sState_tile[None, None, 0, 0]

    tCrKd = thr_mma.make_fragment_A(thr_mma.partition_A(sKd_ref))
    tCrQd = thr_mma.make_fragment_A(thr_mma.partition_A(sQd_ref))
    tCrState = thr_mma.make_fragment_B(thr_mma.partition_B(sState_ref))
    tCrU = thr_mma.make_fragment_C(tiled_mma.partition_shape_C((CHUNK, D)))
    tCrOut = thr_mma.make_fragment_C(tiled_mma.partition_shape_C((CHUNK, D)))

    tCrKd_cv = smem_thr_copy_A.retile(tCrKd)
    tCrQd_cv = smem_thr_copy_A.retile(tCrQd)
    # tCrState is loaded with the transposed B-copy atom (sState_B_view has
    # N=D_out as the fast SMEM axis after select-mode-[1,0]).
    tCrState_cv = smem_thr_copy_B_T.retile(tCrState)

    sMqk_s0 = sMqk[(None, None, 0)]
    sMqk_tile0 = cute.flat_divide(sMqk_s0, (CHUNK, CHUNK))
    sMqk_ref = sMqk_tile0[None, None, 0, 0]
    tCrMqk = thr_mma.make_fragment_A(thr_mma.partition_A(sMqk_ref))
    tCrMqk_cv = smem_thr_copy_A.retile(tCrMqk)

    # ---- Phase 6 TC fragments (declared once, reused per chunk) ----
    # Role-swap: A=sKr_T (D, CHUNK), B=tCrU_T_post (register from MOVM_T).
    # Result[m, n] = sum_k sKr_T[m, k] * U_post[k, n] = (kr^T @ U)[m, n] —
    # directly stored as sState[m=K_in, n=D_out]. Eliminates sU_T SMEM trip.
    sKr_T_tile = cute.flat_divide(sKr_T, (D, CHUNK))  # ((D, CHUNK), 1, 1)
    sKr_T_ref = sKr_T_tile[None, None, 0, 0]  # (D, CHUNK)
    tCrKrA6 = thr_mma6.make_fragment_A(thr_mma6.partition_A(sKr_T_ref))
    tCrUpd = thr_mma6.make_fragment_C(tiled_mma6.partition_shape_C((D, D)))
    tCrKrA6_cv = smem_thr_copy_A6.retile(tCrKrA6)

    # Phase 3/4-epi B-frag layout: (N=D, K=CHUNK), tile (16, 32, 16).
    # No backing SMEM (register-resident). Use sKr_T_ref shape to derive
    # the partition_B layout (same shape (D, CHUNK)).
    tCrU_T = thr_mma.make_fragment_B(thr_mma.partition_B(sKr_T_ref))

    # ---- Phase 3 TC fragments: A=sINV (16,16), B=sU_T (D, CHUNK), C=tCrU3 (16, D) ----
    sINV_s0 = sINV[(None, None, 0)]
    sINV_tile0 = cute.flat_divide(sINV_s0, (CHUNK, CHUNK))
    sINV_ref = sINV_tile0[None, None, 0, 0]
    tCrInv = thr_mma.make_fragment_A(thr_mma.partition_A(sINV_ref))
    tCrInv_cv = smem_thr_copy_A.retile(tCrInv)
    tCrU3 = thr_mma.make_fragment_C(tiled_mma.partition_shape_C((CHUNK, D)))
    # Register-resident U-post B-fragment (MOVM_T from phase 3 C-frag).
    # Same shape/dtype as tCrU_T; populated post-phase-3 via inline-PTX
    # movmatrix.sync.aligned.m8n8.trans.b16. Replaces phase 4-epi B-load
    # from sU_T (1 SMEM round-trip eliminated per chunk).
    tCrU_T_post = cute.make_fragment_like(tCrU_T)
    tCrU3_bf16_tmp = cute.make_fragment_like(tCrU3, cutlass.BFloat16)

    bos = seq_idx * seq_len
    t_tiles: cutlass.Constexpr[int] = (seq_len + CHUNK - 1) // CHUNK
    TMA_BYTES: cutlass.Constexpr[int] = 4 * CHUNK * D * 2 + 2 * CHUNK * CHUNK * 2

    # ---- Helper: warp-issued TMA load for chunk t_g into stage `s_idx`.
    # `s_idx` may be Constexpr (prologue) or Int32 (steady-state). The
    # mbarrier arrival is single-lane; cute.copy is whole-warp.
    # ---------------------------------------------------------------

    # ---- Prologue: pre-issue TMA for first STAGES chunks. ----
    if warp_idx == 0:
        for ps in cutlass.range_constexpr(STAGES):
            pps: cutlass.Constexpr[int] = ps
            if pps < t_tiles:
                tg_p = seq_idx * t_tiles + pps
                wt_p = head_idx * total_tiles + tg_p
                bar_p = sMbar_ptr + cutlass.Int32(pps)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(bar_p, cutlass.Int32(TMA_BYTES))
                cute.copy(tma_atom_v, tVg[(None, tg_p, 0, head_idx)], tVs[(None, pps)], tma_bar_ptr=bar_p)
                cute.copy(tma_atom_kd, tKDg[(None, 0, 0, wt_p)], tKDs[(None, pps)], tma_bar_ptr=bar_p)
                cute.copy(tma_atom_qd, tQDg[(None, 0, 0, wt_p)], tQDs[(None, pps)], tma_bar_ptr=bar_p)
                cute.copy(tma_atom_kr, tKRg[(None, 0, 0, wt_p)], tKRs[(None, pps)], tma_bar_ptr=bar_p)
                cute.copy(tma_atom_inv, tIg[(None, 0, 0, wt_p)], tIs[(None, pps)], tma_bar_ptr=bar_p)
                cute.copy(tma_atom_mqk, tMg[(None, 0, 0, wt_p)], tMs[(None, pps)], tma_bar_ptr=bar_p)

    # Per-stage phase counters (one per mbarrier slot).
    phase0 = cutlass.Int32(0)
    phase1 = cutlass.Int32(0)

    for t in cutlass.range(t_tiles, unroll=1):
        s_dyn = t & cutlass.Int32(1)  # dynamic stage selector (0 or 1)
        t_g = seq_idx * t_tiles + t
        ws_t = head_idx * total_tiles + t_g
        gt_off = ws_t * D

        # Per-stage SMEM views (dynamic stage indexing).
        sV_s = sV[(None, None, s_dyn)]
        sKd_s = sKd[(None, None, s_dyn)]
        sQd_s = sQd[(None, None, s_dyn)]
        sKr_s = sKr[(None, None, s_dyn)]
        sINV_s = sINV[(None, None, s_dyn)]
        sMqk_s = sMqk[(None, None, s_dyn)]
        sKd_tile = cute.flat_divide(sKd_s, (CHUNK, 16))
        sQd_tile = cute.flat_divide(sQd_s, (CHUNK, 16))
        sINV_tile_s = cute.flat_divide(sINV_s, (CHUNK, CHUNK))
        sINV_ref_s = sINV_tile_s[None, None, 0, 0]
        sMqk_tile_s = cute.flat_divide(sMqk_s, (CHUNK, CHUNK))
        sMqk_ref_s = sMqk_tile_s[None, None, 0, 0]

        if tidx < D:
            sGt[tidx] = ws_gt[gt_off + tidx]
        if tidx < CHUNK:
            sBeta[tidx] = beta[head_idx * T_total + bos + t * CHUNK + tidx]

        # Wait for TMA into stage s_dyn.
        if s_dyn == cutlass.Int32(0):
            cute.arch.mbarrier_wait(sMbar_ptr + cutlass.Int32(0), phase0)
            phase0 = phase0 ^ cutlass.Int32(1)
        else:
            cute.arch.mbarrier_wait(sMbar_ptr + cutlass.Int32(1), phase1)
            phase1 = phase1 ^ cutlass.Int32(1)

        # Pre-transpose sKr -> sKr_T early.
        if tidx < D:
            kr_buf = cute.make_rmem_tensor(cute.make_layout((CHUNK,), stride=(1,)), cutlass.BFloat16)
            for m in cutlass.range_constexpr(CHUNK):
                mm: cutlass.Constexpr[int] = m
                kr_buf[mm] = sKr_s[mm, tidx]
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
            cute.copy(smem_tiled_copy_B_T, smem_thr_copy_B_T.partition_S(sState_k), tCrState_cv)
            cute.gemm(tiled_mma, tCrU, tCrKd, tCrState, tCrU)

        # ----- Phase 2: u = sigmoid(beta) * (v - u_pre); MOVM_T into B-frag for phase 3 -----
        lane_in_warp = tidx % 32
        Rrow0 = lane_in_warp // 4
        Rrow1 = Rrow0 + 8
        b0 = cutlass.Float32(sBeta[Rrow0])
        b1 = cutlass.Float32(sBeta[Rrow1])
        sig0 = cutlass.Float32(0.5) * (cute.tanh(b0 * cutlass.Float32(0.5), fastmath=True) + cutlass.Float32(1.0))
        sig1 = cutlass.Float32(0.5) * (cute.tanh(b1 * cutlass.Float32(0.5), fastmath=True) + cutlass.Float32(1.0))
        tCsV = thr_mma.partition_C(sV_s)
        # Pre-load sV into a register frag — breaks SMEM→math dep chain.
        v_frag = cute.make_fragment_like(tCsV, cutlass.BFloat16)
        for i in cutlass.range_constexpr(cute.size(v_frag)):
            ii: cutlass.Constexpr[int] = i
            v_frag[ii] = tCsV[ii]
        # Compute U_pre = sig*(v-u_pre) into a bf16 register C-frag, then MOVM_T
        # into tCrU_T (B-frag) for phase 3 GEMM. Eliminates sU_T write here AND
        # the corresponding cross-thread sync (data stays per-lane).
        tCrU_pre_bf16 = cute.make_fragment_like(tCrU, cutlass.BFloat16)
        for i in cutlass.range_constexpr(cute.size(tCrU)):
            ii: cutlass.Constexpr[int] = i
            sub_i: cutlass.Constexpr[int] = (ii % 4) // 2
            sig = sig0 if sub_i == 0 else sig1
            diff = cutlass.Float32(v_frag[ii]) - tCrU[ii]
            tCrU_pre_bf16[ii] = cutlass.BFloat16(diff * sig)
        tCrU_pre_u32 = cute.recast_tensor(tCrU_pre_bf16, dtype=cutlass.Int32)
        tCrU_T_u32 = cute.recast_tensor(tCrU_T, dtype=cutlass.Int32)
        for i in cutlass.range_constexpr(cute.size(tCrU_pre_u32)):
            ii: cutlass.Constexpr[int] = i
            tCrU_T_u32[ii] = movm_t_b16(cutlass.Int32(tCrU_pre_u32[ii]))
        # sU_T removed: U is now register-resident across phases 2->3->4-epi->6.

        # ----- Phase 3 (TC): U_post = INV @ U_pre  (M=16 N=D K=CHUNK) -----
        # B operand U_pre comes from register tCrU_T (MOVM_T'd in phase 2).
        cute.copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(sINV_ref_s), tCrInv_cv)
        tCrU3.fill(0.0)
        cute.gemm(tiled_mma, tCrU3, tCrInv, tCrU_T, tCrU3)
        # In-frag fp32->bf16 cast into a register tmp; MOVM_T converts the
        # C-format bytes into B-frag layout (tCrU_T_post) for phase 4-epi AND
        # phase 6. No SMEM trip — sU_T eliminated entirely.
        for i in cutlass.range_constexpr(cute.size(tCrU3)):
            ii: cutlass.Constexpr[int] = i
            tCrU3_bf16_tmp[ii] = cutlass.BFloat16(tCrU3[ii])
        tCrU3_u32 = cute.recast_tensor(tCrU3_bf16_tmp, dtype=cutlass.Int32)
        tCrU_T_post_u32 = cute.recast_tensor(tCrU_T_post, dtype=cutlass.Int32)
        for i in cutlass.range_constexpr(cute.size(tCrU3_u32)):
            ii: cutlass.Constexpr[int] = i
            tCrU_T_post_u32[ii] = movm_t_b16(cutlass.Int32(tCrU3_u32[ii]))

        # ----- Phase 4 MAIN (TC, no sU_T dep): out = qd @ state (8 K iters) -----
        # Run BEFORE the sU_T-visibility barrier so it overlaps with phase 3
        # SMEM writes settling. Reads only sQd, sState — disjoint from sU_T.
        tCrOut.fill(0.0)
        for k in cutlass.range_constexpr(D // 16):
            sQd_k = sQd_tile[None, None, 0, k]
            sState_k = sState_tile[None, None, 0, k]
            cute.copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(sQd_k), tCrQd_cv)
            cute.copy(smem_tiled_copy_B_T, smem_thr_copy_B_T.partition_S(sState_k), tCrState_cv)
            cute.gemm(tiled_mma, tCrOut, tCrQd, tCrState, tCrOut)

        # ----- Phase 4 EPI (TC): out += Mqk @ U_T (1 K iter) -----
        cute.copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(sMqk_ref_s), tCrMqk_cv)
        # B operand from MOVM_T register frag (tCrU_T_post) — skip sU_T B-load.
        cute.gemm(tiled_mma, tCrOut, tCrMqk, tCrU_T_post, tCrOut)

        # Single in-frag bf16 write to sOut.
        tCsOut = thr_mma.partition_C(sOut)
        for i in cutlass.range_constexpr(cute.size(tCrOut)):
            tCsOut[i] = cutlass.BFloat16(tCrOut[i])
        # NOTE: no barrier here. Phase 6 below uses entirely disjoint SMEM
        # (sU_T/sKr_T/sState — never sOut), so it can run in parallel with the
        # sOut writes settling into SMEM. We fold the sOut visibility barrier
        # into the single barrier after phase 6 (one fewer cluster-wide sync).

        # ----- Phase 6 (TC MMA): state[m, n] = state*gt + (kr^T @ U)[m, n] -----
        # Role-swap: A=sKr_T (D, CHUNK), B=tCrU_T_post (register from MOVM_T).
        # gt indexed by M (K_in axis) under normal-form sState convention.

        # Cross-warp barrier for sKr_T visibility (cooperative transpose at
        # chunk start writes per-thread strides; phase 6 A-load is cross-warp).
        cute.arch.barrier()
        cute.copy(
            smem_tiled_copy_A6,
            smem_thr_copy_A6.partition_S(sKr_T_ref),
            tCrKrA6_cv,
        )
        tCrUpd.fill(0.0)
        cute.gemm(tiled_mma6, tCrUpd, tCrKrA6, tCrU_T_post, tCrUpd)

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
            # gt indexed by M (K_in axis) — normal-form state convention.
            gt_frag[ii] = sGt[tCcState[ii][0]]
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
            # Pre-issue TMA load for chunk t+STAGES into the slot we just consumed.
            t_next = t + cutlass.Int32(STAGES)
            if t_next < cutlass.Int32(t_tiles):
                tg_n = seq_idx * t_tiles + t_next
                wt_n = head_idx * total_tiles + tg_n
                bar_n = sMbar_ptr + s_dyn
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(bar_n, cutlass.Int32(TMA_BYTES))
                cute.copy(tma_atom_v, tVg[(None, tg_n, 0, head_idx)], tVs[(None, s_dyn)], tma_bar_ptr=bar_n)
                cute.copy(tma_atom_kd, tKDg[(None, 0, 0, wt_n)], tKDs[(None, s_dyn)], tma_bar_ptr=bar_n)
                cute.copy(tma_atom_qd, tQDg[(None, 0, 0, wt_n)], tQDs[(None, s_dyn)], tma_bar_ptr=bar_n)
                cute.copy(tma_atom_kr, tKRg[(None, 0, 0, wt_n)], tKRs[(None, s_dyn)], tma_bar_ptr=bar_n)
                cute.copy(tma_atom_inv, tIg[(None, 0, 0, wt_n)], tIs[(None, s_dyn)], tma_bar_ptr=bar_n)
                cute.copy(tma_atom_mqk, tMg[(None, 0, 0, wt_n)], tMs[(None, s_dyn)], tma_bar_ptr=bar_n)
        cute.arch.cp_async_bulk_wait_group(0, read=True)
        cute.arch.barrier()


@cute.jit
def run_k2_phaseH(
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

    # 2-stage SMEM: 4 staged QK tiles + 2 staged CC tiles + sOut(unstaged)
    # + sState + sU_T + sKr_T + sGt + sBeta + 2 mbarriers (16 bytes) + slack
    STAGES_LOCAL = 2
    smem_bytes = (
        D * D * 2  # sState
        + STAGES_LOCAL * 4 * (CHUNK * D * 2)  # sV/sKd/sQd/sKr staged
        + STAGES_LOCAL * 2 * (CHUNK * CHUNK * 2)  # sINV/sMqk staged
        + (CHUNK * D * 2)  # sOut (unstaged)
        + 2 * (CHUNK * D * 2)  # sU_T + sKr_T
        + (D * 4)  # sGt
        + (CHUNK * 2)  # sBeta
        + STAGES_LOCAL * 8  # 2 mbarriers
        + 256  # slack
    )

    k2_phaseH_kernel(
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


_compiled_cache_k2H: dict = {}


def launch_k2_phaseH(
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
    if key not in _compiled_cache_k2H:
        stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
        v_flat = v.view(T_total, H, D)
        out_flat = out.view(T_total, H, D)
        _compiled_cache_k2H[key] = cute.compile(
            run_k2_phaseH,
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
    _compiled_cache_k2H[key](
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

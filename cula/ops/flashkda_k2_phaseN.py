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
from cutlass.cute.nvgpu.warpgroup import SmemLayoutAtomKind, make_smem_layout_atom
from cutlass.cute.runtime import from_dlpack

from cula.ops.flashkda_k2 import CHUNK, D, _make_state_smem_layout
from cula.ops.flashkda_prefill import movm_t_b16


def _make_out_kinter_one_stage():
    """K_INTER swizzled (CHUNK, D) bf16 SMEM layout — matches cpp VOLayout.

    Uses 8x8 K-major atom (Swizzle<0,4,3>, no swizzle bits, just hierarchical
    8x8 atomic decomposition required for SM75/SM90 LDSM/STSM thread mapping).
    For bf16 the atom is (8, 64):(64, 1); tile_to_shape replicates it across
    the (CHUNK, D) tile.
    """
    atom = make_smem_layout_atom(SmemLayoutAtomKind.K_INTER, cutlass.BFloat16)
    return cute.tile_to_shape(atom, (CHUNK, D), order=(0, 1))


THREADS_PER_CTA = 192  # 128 compute (4 MMA warps) + 32 load + 32 store
N_WARPS = 4
LOAD_WARP_IDX = 4
STORE_WARP_IDX = 5


def _make_qk_smem_layout():
    """Plain K-major (CHUNK, D) layout, matching Phase B.

    NOTE: When cute.gemm is wired in (Phase E2), this should switch to a
    swizzled K_INTER atom for SM80 16x8x16 MMA A-operand. For now keep plain
    layout to keep TMA partitioning identical to Phase B.
    """
    return cute.make_layout((CHUNK, D), stride=(D, 1))


@cute.kernel
def k2_phaseN_kernel(
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

    qk_layout = _make_qk_smem_layout()  # K-major swizzled, MMA-ready  # noqa: F841 (kept for parity)
    cc_layout = cute.make_layout((CHUNK, CHUNK), stride=(CHUNK, 1))  # noqa: F841 (kept for parity)
    state_layout = _make_state_smem_layout()  # swizzled K-major
    # Phase 6 reads sKr's (D, CHUNK) transposed view directly via LdMatrix.x4.trans
    # (matches cpp baseline). No sKr_T SMEM, no manual transpose loop, no extra
    # cross-warp barrier.

    # Input pipeline (TMA load): InputStages slot ring.
    STAGES: cutlass.Constexpr[int] = 3
    # Output pipeline (TMA store): OutputStages slot ring of sOut tiles.
    OUT_STAGES: cutlass.Constexpr[int] = 2
    qk_stage_layout = cute.make_layout((CHUNK, D, STAGES), stride=(D, 1, CHUNK * D))
    cc_stage_layout = cute.make_layout((CHUNK, CHUNK, STAGES), stride=(CHUNK, 1, CHUNK * CHUNK))
    # sOut now uses K_INTER swizzled atom so STSM_N (cute.copy via
    # StMatrix8x8x16bOp) produces the correct SMEM thread-data mapping.
    out_kinter_atom = make_smem_layout_atom(SmemLayoutAtomKind.K_INTER, cutlass.BFloat16)
    out_stage_layout = cute.tile_to_shape(out_kinter_atom, (CHUNK, D, OUT_STAGES), order=(0, 1, 2))
    # sV uses K_INTER swizzled layout to reduce shared-load bank conflicts on the
    # ldmatrix path that feeds Phase-2 MOVM_T (consumed via partition_C, which is
    # transparent through swizzle).
    v_kinter_atom = make_smem_layout_atom(SmemLayoutAtomKind.K_INTER, cutlass.BFloat16)
    v_stage_layout = cute.tile_to_shape(v_kinter_atom, (CHUNK, D, STAGES), order=(0, 1, 2))
    # sKd K_INTER swizzled layout (consumed via cute.copy(LdMatrix...) only).
    kd_stage_layout = cute.tile_to_shape(v_kinter_atom, (CHUNK, D, STAGES), order=(0, 1, 2))
    # sQd K_INTER swizzled layout (consumed via cute.copy(LdMatrix...) only).
    qd_stage_layout = cute.tile_to_shape(v_kinter_atom, (CHUNK, D, STAGES), order=(0, 1, 2))
    # sKr K_INTER swizzled allocation. Same atom as sV/sKd/sQd. Phase 4-prelude
    # B-load uses LDSM_N on this swizzled (CHUNK, D) view directly. Phase 6
    # A-load uses LDSM_T on a MN_INTER-atom view of the SAME bytes (cpp
    # baseline pattern: SMEM written via K_INTER is read via MN_INTER+.trans).
    kr_stage_layout = cute.tile_to_shape(v_kinter_atom, (CHUNK, D, STAGES), order=(0, 1, 2))
    # MN_INTER atom for the transposed view of sKr used by Phase 6 (D, CHUNK).
    kr_mninter_atom = make_smem_layout_atom(SmemLayoutAtomKind.MN_INTER, cutlass.BFloat16)
    kr_t_stage_layout = cute.tile_to_shape(kr_mninter_atom, (D, CHUNK, STAGES), order=(0, 1, 2))

    sV = smem.allocate_tensor(cutlass.BFloat16, v_stage_layout, 128)
    sKd = smem.allocate_tensor(cutlass.BFloat16, kd_stage_layout, 128)
    sQd = smem.allocate_tensor(cutlass.BFloat16, qd_stage_layout, 128)
    sKr = smem.allocate_tensor(cutlass.BFloat16, kr_stage_layout, 128)
    sINV = smem.allocate_tensor(cutlass.BFloat16, cc_stage_layout, 128)
    sMqk = smem.allocate_tensor(cutlass.BFloat16, cc_stage_layout, 128)
    sOut = smem.allocate_tensor(cutlass.BFloat16, out_stage_layout, 128)
    sState = smem.allocate_tensor(cutlass.BFloat16, state_layout, 128)
    # NOTE: sU_T removed (U is now register-resident via MOVM_T inline PTX).
    # NOTE: sKr_T removed (Phase 6 uses LdMatrix.x4.trans on sKr directly).
    sGt = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D,)), 128)
    sBeta = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((CHUNK,)), 128)
    # ---- mbarriers (Int64 each) ----
    # Load pipeline: full (TMA producer arrival) / empty (compute consumer arrival).
    sMbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((STAGES,)), 16)
    sMbar_ptr = sMbar.iterator
    sMbarE = smem.allocate_tensor(cutlass.Int64, cute.make_layout((STAGES,)), 16)
    sMbarE_ptr = sMbarE.iterator
    # Store pipeline: full (compute producer arrival) / empty (store consumer arrival).
    sMbarSF = smem.allocate_tensor(cutlass.Int64, cute.make_layout((OUT_STAGES,)), 16)
    sMbarSF_ptr = sMbarSF.iterator
    sMbarSE = smem.allocate_tensor(cutlass.Int64, cute.make_layout((OUT_STAGES,)), 16)
    sMbarSE_ptr = sMbarSE.iterator

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        with cute.arch.elect_one():
            for s in cutlass.range_constexpr(STAGES):
                # Load-full: 1 producer arrival per stage (TMA completion).
                cute.arch.mbarrier_init(sMbar_ptr + cutlass.Int32(s), cutlass.Int32(1))
                # Load-empty: 1 consumer arrival per stage (compute lane 0 after
                # consuming the slot).
                cute.arch.mbarrier_init(sMbarE_ptr + cutlass.Int32(s), cutlass.Int32(1))
            for s in cutlass.range_constexpr(OUT_STAGES):
                # Store-full: 1 producer arrival per stage (compute lane 0 after
                # writing sOut[s_out] and fencing).
                cute.arch.mbarrier_init(sMbarSF_ptr + cutlass.Int32(s), cutlass.Int32(1))
                # Store-empty: 1 consumer arrival per stage (store lane 0 after
                # TMA store completion).
                cute.arch.mbarrier_init(sMbarSE_ptr + cutlass.Int32(s), cutlass.Int32(1))
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
    _ = smem_tiled_copy_B  # kept for symmetry; B-loads use the .trans variant below
    # Phase 1/4-main B reads sState in transposed view (N=D_out fast axis).
    # Use ldmatrix.x4.trans atom because N is now the contig SMEM axis.
    copy_atom_B_T = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
        cutlass.BFloat16,
    )
    smem_tiled_copy_B_T = cute.make_tiled_copy_B(copy_atom_B_T, tiled_mma)
    smem_thr_copy_B_T = smem_tiled_copy_B_T.get_slice(tidx)

    # STSM_N for sOut writes — mirrors cpp's SM90_U32x4_STSM_N. Requires the
    # K_INTER swizzled sOut layout above. Use make_tiled_copy_C_atom because
    # the STSM atom holds fewer values per warp than a full MMA C-frag stripe;
    # this helper produces a sub-tiled copy with explicit per-block iteration.
    copy_atom_stsm = cute.make_copy_atom(
        warp.StMatrix8x8x16bOp(transpose=False, num_matrices=2),
        cutlass.BFloat16,
    )
    smem_tiled_store_C = cute.make_tiled_copy_C_atom(copy_atom_stsm, tiled_mma)
    smem_thr_store_C = smem_tiled_store_C.get_slice(tidx)

    # Phase 6 tiled MMA: same atom layout as phase 1/4.
    tiled_mma6 = cute.make_tiled_mma(
        mma_atom,
        atom_layout_mnk=(1, 4, 1),
        permutation_mnk=(16, 32, 16),
    )
    thr_mma6 = tiled_mma6.get_slice(tidx)
    # Phase 6 A operand reads (D, CHUNK) transposed VIEW of sKr (CHUNK, D)
    # via LdMatrix.x4.trans — no physical sKr_T transpose needed.
    smem_tiled_copy_A6 = cute.make_tiled_copy_A(copy_atom_B_T, tiled_mma6)
    smem_tiled_copy_B6 = cute.make_tiled_copy_B(copy_atom_AB, tiled_mma6)
    smem_thr_copy_A6 = smem_tiled_copy_A6.get_slice(tidx)
    _ = smem_tiled_copy_B6  # B-operand for phase 6 is register-resident (tCrU_T_post)

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
    # A operand: (D, CHUNK) transposed VIEW of sKr (CHUNK, D, STAGES) — no
    # physical transpose; LdMatrix.x4.trans handles per-8x8 transpose in regs.
    # Result[m, n] = sum_k sKr_T[m, k] * U_post[k, n] = (kr^T @ U)[m, n] —
    # directly stored as sState[m=K_in, n=D_out]. Eliminates sU_T SMEM trip.
    #
    # CRITICAL: sKr is K_INTER swizzled-allocated. We construct sKr_T as an
    # MN_INTER-atom view of the SAME iterator (kr_t_stage_layout, shape
    # (D, CHUNK, STAGES)). Both atoms map onto the same 1024-byte cosize,
    # and ldmatrix.x4.trans + MN_INTER stride pattern yields the correct
    # per-8x8 transposed bf16 data — matching the cpp baseline's
    # MMALayout↔TransposedMMALayout dual-view pattern on `k_restored`.
    sKr_T_view = cute.make_tensor(sKr.iterator, kr_t_stage_layout)
    sKr_T_view_s0 = sKr_T_view[(None, None, 0)]
    sKr_T_tile = cute.flat_divide(sKr_T_view_s0, (D, CHUNK))  # ((D, CHUNK), 1, 1)
    sKr_T_ref = sKr_T_tile[None, None, 0, 0]  # (D, CHUNK)
    tCrKrA6 = thr_mma6.make_fragment_A(thr_mma6.partition_A(sKr_T_ref))
    tCrUpd = thr_mma6.make_fragment_C(tiled_mma6.partition_shape_C((D, D)))
    tCrKrA6_cv = smem_thr_copy_A6.retile(tCrKrA6)

    # ---- Phase 6 BLOCKED accumulator ref (CHUNK_M = CHUNK = 16) ----
    # Decompose the (D, D) Phase-6 GEMM into D/CHUNK = 4 M-block iterations
    # of (CHUNK, D, CHUNK) each. Per-warp accumulator per iteration is only
    # CHUNK / atom_M(=16) × D / (atom_N(=8) × 4 warps) × 8 fp32 = 1 × 2 × 8
    # = 8 fp32 regs (vs 64 for the full (D, D) accumulator). This matches
    # the cpp baseline pattern in fwd_kernel2.cuh Phase 6 (S_M_BLOCKS loop).
    # Build a (CHUNK, CHUNK) reference for partition_A frag-shape derivation.
    sKr_T_blk_for_frag = cute.flat_divide(sKr_T_view_s0, (CHUNK, CHUNK))[None, None, 0, 0]
    tCrKrA6_blk = thr_mma6.make_fragment_A(thr_mma6.partition_A(sKr_T_blk_for_frag))
    tCrKrA6_blk_cv = smem_thr_copy_A6.retile(tCrKrA6_blk)
    # C operand per block: (CHUNK, D) shape, same as tCrU/tCrOut/tCrU3.
    tCrUpd_blk = thr_mma6.make_fragment_C(tiled_mma6.partition_shape_C((CHUNK, D)))
    # State-block reference for partition_C frag-shape.
    sState_blk_tile = cute.flat_divide(sState, (CHUNK, D))  # ((CHUNK, D), M_BLOCKS_6, 1)
    sState_blk_for_frag = sState_blk_tile[None, None, 0, 0]
    coord_state_blk = cute.make_identity_tensor((CHUNK, D))
    tCcState_blk = thr_mma6.partition_C(coord_state_blk)
    # NOTE: per-iteration state_frag/gt_frag are allocated inside the loop so
    # their lifetime is tightly scoped (compiler can reuse storage across mi
    # iterations and with other transients). Shape: same as tCrUpd_blk.

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
    # Phase-2 scratch: bf16 view of phase-1 result for MOVM_T into tCrU_T.
    # Hoisted out of the t-loop so the allocation/lifetime is loop-uniform and
    # the compiler can fold its storage with other transient frags.
    tCrU_pre_bf16 = cute.make_fragment_like(tCrU, cutlass.BFloat16)

    # Identity coords for phase-6 epilogue gt indexing — loop-invariant.
    coord_state = cute.make_identity_tensor((D, D))
    tCcState = thr_mma6.partition_C(coord_state)
    tCsState = thr_mma6.partition_C(sState)

    bos = seq_idx * seq_len
    t_tiles: cutlass.Constexpr[int] = (seq_len + CHUNK - 1) // CHUNK
    TMA_BYTES: cutlass.Constexpr[int] = 4 * CHUNK * D * 2 + 2 * CHUNK * CHUNK * 2

    # ---- Helper: warp-issued TMA load for chunk t_g into stage `s_idx`.
    # `s_idx` may be Constexpr (prologue) or Int32 (steady-state). The
    # mbarrier arrival is single-lane; cute.copy is whole-warp.
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Warp specialization: warp 4 (tidx 128..159) is dedicated TMA-load
    # warp; warps 0..3 (tidx 0..127) are MMA / compute warps.
    # ---------------------------------------------------------------
    if warp_idx == LOAD_WARP_IDX:
        # ===== LOAD WARP =====
        # Producer-start phase = 1 so the first wait on each empty[s] returns
        # immediately (slots are conceptually "empty" at startup).
        s_dyn_l = cutlass.Int32(0)
        phase_emp = cutlass.Int32(1)
        for t in cutlass.range(t_tiles, unroll=1):
            cute.arch.mbarrier_wait(sMbarE_ptr + s_dyn_l, phase_emp)
            tg_l = seq_idx * t_tiles + t
            wt_l = head_idx * total_tiles + tg_l
            bar_l = sMbar_ptr + s_dyn_l
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar_l, cutlass.Int32(TMA_BYTES))
            cute.copy(tma_atom_v, tVg[(None, tg_l, 0, head_idx)], tVs[(None, s_dyn_l)], tma_bar_ptr=bar_l)
            cute.copy(tma_atom_kd, tKDg[(None, 0, 0, wt_l)], tKDs[(None, s_dyn_l)], tma_bar_ptr=bar_l)
            cute.copy(tma_atom_qd, tQDg[(None, 0, 0, wt_l)], tQDs[(None, s_dyn_l)], tma_bar_ptr=bar_l)
            cute.copy(tma_atom_kr, tKRg[(None, 0, 0, wt_l)], tKRs[(None, s_dyn_l)], tma_bar_ptr=bar_l)
            cute.copy(tma_atom_inv, tIg[(None, 0, 0, wt_l)], tIs[(None, s_dyn_l)], tma_bar_ptr=bar_l)
            cute.copy(tma_atom_mqk, tMg[(None, 0, 0, wt_l)], tMs[(None, s_dyn_l)], tma_bar_ptr=bar_l)
            s_dyn_l = s_dyn_l + cutlass.Int32(1)
            if s_dyn_l == cutlass.Int32(STAGES):
                s_dyn_l = cutlass.Int32(0)
                phase_emp = phase_emp ^ cutlass.Int32(1)
    elif warp_idx == STORE_WARP_IDX:
        # ===== STORE WARP =====
        # Consumer over store_full[s_out_s]; emits TMA store from sOut[s_out_s];
        # waits and arrives on store_empty[s_out_s] so MMA can recycle the slot.
        s_out_s = cutlass.Int32(0)
        phase_sf = cutlass.Int32(0)
        for t in cutlass.range(t_tiles, unroll=1):
            cute.arch.mbarrier_wait(sMbarSF_ptr + s_out_s, phase_sf)
            t_g_s = seq_idx * t_tiles + t
            cute.copy(tma_atom_out, tOs[(None, s_out_s)], tOg[(None, t_g_s, 0, head_idx)])
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(sMbarSE_ptr + s_out_s)
            s_out_s = s_out_s + cutlass.Int32(1)
            if s_out_s == cutlass.Int32(OUT_STAGES):
                s_out_s = cutlass.Int32(0)
                phase_sf = phase_sf ^ cutlass.Int32(1)
    else:
        # ===== COMPUTE WARPS (warps 0..3) =====
        # Single rotating consumer phase counter (rotates when s_dyn wraps).
        phase_full = cutlass.Int32(0)
        s_dyn = cutlass.Int32(0)
        # Output-pipeline producer state. Producer-start phase = 1 so the
        # first wait on store_empty[s_out] returns immediately.
        s_out = cutlass.Int32(0)
        phase_se = cutlass.Int32(1)

        for t in cutlass.range(t_tiles, unroll=2):
            t_g = seq_idx * t_tiles + t
            ws_t = head_idx * total_tiles + t_g
            gt_off = ws_t * D

            # Per-stage SMEM views (dynamic stage indexing).
            sV_s = sV[(None, None, s_dyn)]
            sKr_s = sKr[(None, None, s_dyn)]
            sKd_tile = cute.flat_divide(sKd[(None, None, s_dyn)], (CHUNK, 16))
            sQd_tile = cute.flat_divide(sQd[(None, None, s_dyn)], (CHUNK, 16))
            sINV_ref_s = cute.flat_divide(sINV[(None, None, s_dyn)], (CHUNK, CHUNK))[None, None, 0, 0]
            sMqk_ref_s = cute.flat_divide(sMqk[(None, None, s_dyn)], (CHUNK, CHUNK))[None, None, 0, 0]

            if tidx < D:
                sGt[tidx] = ws_gt[gt_off + tidx]
            if tidx < CHUNK:
                sBeta[tidx] = beta[head_idx * T_total + bos + t * CHUNK + tidx]

            # Wait for TMA into stage s_dyn (single rotating phase).
            cute.arch.mbarrier_wait(sMbar_ptr + s_dyn, phase_full)

            # Per-stage transposed view of sKr for Phase 6 (no physical transpose).
            sKr_T_s = sKr_T_view[(None, None, s_dyn)]
            sKr_T_ref_s = cute.flat_divide(sKr_T_s, (D, CHUNK))[None, None, 0, 0]
            # Per-stage M-blocked view: ((CHUNK, CHUNK), M_BLOCKS_6, 1).
            sKr_T_blk_tile_s = cute.flat_divide(sKr_T_s, (CHUNK, CHUNK))

            # ============================================================
            # Phase 1+4-MAIN FUSED (TENSOR-CORE): both share same State B-load.
            #   tCrU   += kd  @ state   (Phase 1: u_pre)
            #   tCrOut += qd  @ state   (Phase 4-main: out)
            # cpp does this fusion explicitly to halve the State LDSM traffic.
            # ============================================================
            tCrU.fill(0.0)
            tCrOut.fill(0.0)
            for k in cutlass.range_constexpr(D // 16):
                sKd_k = sKd_tile[None, None, 0, k]
                sQd_k = sQd_tile[None, None, 0, k]
                sState_k = sState_tile[None, None, 0, k]
                # Single State B-load shared by both GEMMs.
                cute.copy(smem_tiled_copy_B_T, smem_thr_copy_B_T.partition_S(sState_k), tCrState_cv)
                cute.copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(sKd_k), tCrKd_cv)
                cute.gemm(tiled_mma, tCrU, tCrKd, tCrState, tCrU)
                cute.copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(sQd_k), tCrQd_cv)
                cute.gemm(tiled_mma, tCrOut, tCrQd, tCrState, tCrOut)

            # ----- Phase 2: u = sigmoid(beta) * (v - u_pre); MOVM_T into B-frag for phase 3 -----
            lane_in_warp = tidx % 32
            Rrow0 = lane_in_warp // 4
            Rrow1 = Rrow0 + 8
            b0 = cutlass.Float32(sBeta[Rrow0])
            b1 = cutlass.Float32(sBeta[Rrow1])
            sig0 = cutlass.Float32(0.5) * (cute.tanh(b0 * cutlass.Float32(0.5), fastmath=True) + cutlass.Float32(1.0))
            sig1 = cutlass.Float32(0.5) * (cute.tanh(b1 * cutlass.Float32(0.5), fastmath=True) + cutlass.Float32(1.0))
            tCsV = thr_mma.partition_C(sV_s)
            # Compute U_pre = sig*(v-u_pre) into the hoisted bf16 frag, then MOVM_T
            # into tCrU_T (B-frag) for phase 3 GEMM. Eliminates sU_T write here AND
            # the corresponding cross-thread sync (data stays per-lane).
            for i in cutlass.range_constexpr(cute.size(tCrU)):
                ii: cutlass.Constexpr[int] = i
                sub_i: cutlass.Constexpr[int] = (ii % 4) // 2
                sig = sig0 if sub_i == 0 else sig1
                diff = cutlass.Float32(tCsV[ii]) - tCrU[ii]
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

            # ----- Phase 4 MAIN: FUSED above (tCrOut already accumulated qd@state) -----

            # ----- Phase 4 EPI (TC): out += Mqk @ U_T (1 K iter) -----
            cute.copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(sMqk_ref_s), tCrMqk_cv)
            # B operand from MOVM_T register frag (tCrU_T_post) — skip sU_T B-load.
            cute.gemm(tiled_mma, tCrOut, tCrMqk, tCrU_T_post, tCrOut)

            # Wait for sOut[s_out] to be free (consumed by STORE warp). Then
            # write the bf16 output frag via STSM (stmatrix.x4) into K_INTER
            # sOut — 1 warp instruction replaces 8 scalar bf16 stores per atom.
            cute.arch.mbarrier_wait(sMbarSE_ptr + s_out, phase_se)
            sOut_s = sOut[(None, None, s_out)]
            tCrOut_bf16 = cute.make_fragment_like(tCrOut, cutlass.BFloat16)
            for i in cutlass.range_constexpr(cute.size(tCrOut)):
                tCrOut_bf16[i] = cutlass.BFloat16(tCrOut[i])
            cute.copy(
                smem_tiled_store_C,
                smem_thr_store_C.retile(tCrOut_bf16),
                smem_thr_store_C.partition_D(sOut_s),
            )
            # NOTE: no barrier here. Phase 6 reads disjoint SMEM (sKr,
            # sState) so it can run in parallel with sOut writes settling.

            # ----- Phase 6 (TC MMA) BLOCKED: state[m, n] = state*gt + (kr^T @ U)[m, n] -----
            # Decomposed into D/CHUNK = 4 M-block iterations of (CHUNK, D, CHUNK)
            # GEMM each. Per-iteration accumulator (tCrUpd_blk) is only 8 fp32
            # regs/thread vs 64 for the full (D, D) accumulator. state_frag /
            # gt_frag are also block-sized (~8 bf16 + ~8 fp32 per iter) and
            # allocated inside the loop so the compiler can reuse storage
            # across iterations. Mirrors cpp baseline Phase 6 S_M_BLOCKS loop.
            M_BLOCKS_6: cutlass.Constexpr[int] = D // CHUNK  # = 4
            for mi in cutlass.range_constexpr(M_BLOCKS_6):
                # Slice this M-block from the per-stage transposed sKr view
                # (CHUNK rows of D-major K=CHUNK).
                sKr_T_blk_s = sKr_T_blk_tile_s[None, None, mi, 0]
                cute.copy(
                    smem_tiled_copy_A6,
                    smem_thr_copy_A6.partition_S(sKr_T_blk_s),
                    tCrKrA6_blk_cv,
                )
                # Compute (kr_blk^T @ U_post) -> (CHUNK, D) accumulator.
                tCrUpd_blk.fill(0.0)
                cute.gemm(tiled_mma6, tCrUpd_blk, tCrKrA6_blk, tCrU_T_post, tCrUpd_blk)

                # In-frag epilogue: state[mi-block] = bf16(float(state)*gt + delta).
                # Slice sState rows for this mi block and partition_C.
                sState_blk = sState_blk_tile[None, None, mi, 0]
                tCsState_blk = thr_mma6.partition_C(sState_blk)
                state_frag_blk = cute.make_fragment_like(tCsState_blk, cutlass.BFloat16)
                gt_frag_blk = cute.make_fragment_like(tCsState_blk, cutlass.Float32)
                m_off: cutlass.Constexpr[int] = mi * CHUNK
                for i in cutlass.range_constexpr(cute.size(state_frag_blk)):
                    ii: cutlass.Constexpr[int] = i
                    state_frag_blk[ii] = tCsState_blk[ii]
                    gt_frag_blk[ii] = sGt[m_off + tCcState_blk[ii][0]]
                for i in cutlass.range_constexpr(cute.size(tCrUpd_blk)):
                    ii: cutlass.Constexpr[int] = i
                    old = cutlass.Float32(state_frag_blk[ii]) * gt_frag_blk[ii]
                    tCsState_blk[ii] = cutlass.BFloat16(old + tCrUpd_blk[ii])
            # Combined barrier: syncs sOut (Phase 4) + sState (Phase 6) writes
            # across compute warps before signaling consumers. COMPUTE-ONLY.
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            # All compute threads fence prior generic-proxy stores to sOut
            # before the async-proxy TMA store reads them.
            cute.arch.fence_view_async_shared()
            if warp_idx == 0:
                with cute.arch.elect_one():
                    # Signal STORE warp the output slot is full and visible.
                    cute.arch.mbarrier_arrive(sMbarSF_ptr + s_out)
                    # Signal LOAD warp the input slot is now empty.
                    cute.arch.mbarrier_arrive(sMbarE_ptr + s_dyn)
            # Advance rotating input stage / phase.
            s_dyn = s_dyn + cutlass.Int32(1)
            if s_dyn == cutlass.Int32(STAGES):
                s_dyn = cutlass.Int32(0)
                phase_full = phase_full ^ cutlass.Int32(1)
            # Advance rotating output stage / phase.
            s_out = s_out + cutlass.Int32(1)
            if s_out == cutlass.Int32(OUT_STAGES):
                s_out = cutlass.Int32(0)
                phase_se = phase_se ^ cutlass.Int32(1)


@cute.jit
def run_k2_phaseN(
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
    out_smem = _make_out_kinter_one_stage()
    # K_INTER swizzled (CHUNK, D) layout for sV TMA atom (matches kernel-side
    # v_stage_layout for swizzled bank-conflict-free shared loads).
    v_smem = _make_out_kinter_one_stage()
    # K_INTER swizzled (CHUNK, D) layout for sKd TMA atom.
    kd_smem = _make_out_kinter_one_stage()
    # K_INTER swizzled (CHUNK, D) layout for sQd TMA atom.
    qd_smem = _make_out_kinter_one_stage()

    def make_thd_atom(t, op, smem_layout=qk_smem):
        view = cute.make_tensor(
            t.iterator,
            cute.make_layout((T_total, D, H), stride=(H * D, 1, D)),
        )
        return cpasync.make_tiled_tma_atom(op, view, smem_layout, (CHUNK, D))

    def make_ws_qkd_atom(t, smem_layout=qk_smem):
        view = cute.make_tensor(
            t.iterator,
            cute.make_layout((CHUNK, D, total_tiles * H), stride=(D, 1, CHUNK * D)),
        )
        return cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), view, smem_layout, (CHUNK, D))

    def make_ws_cc_atom(t):
        view = cute.make_tensor(
            t.iterator,
            cute.make_layout((CHUNK, CHUNK, total_tiles * H), stride=(CHUNK, 1, CHUNK * CHUNK)),
        )
        return cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), view, cc_smem, (CHUNK, CHUNK))

    tma_atom_v, tma_tensor_v = make_thd_atom(v, cpasync.CopyBulkTensorTileG2SOp(), smem_layout=v_smem)
    tma_atom_out, tma_tensor_out = make_thd_atom(out, cpasync.CopyBulkTensorTileS2GOp(), smem_layout=out_smem)
    tma_atom_kd, tma_tensor_kd = make_ws_qkd_atom(ws_kd, smem_layout=kd_smem)
    tma_atom_qd, tma_tensor_qd = make_ws_qkd_atom(ws_qd, smem_layout=qd_smem)
    tma_atom_kr, tma_tensor_kr = make_ws_qkd_atom(ws_kr)
    tma_atom_inv, tma_tensor_inv = make_ws_cc_atom(ws_inv)
    tma_atom_mqk, tma_tensor_mqk = make_ws_cc_atom(ws_mqk)

    # SMEM layout (warp-specialized, double-buffered output):
    #   sState (D*D bf16) + InputStages * (4 QK + 2 CC) + OutputStages * sOut
    #   + sKr_T (padded) + sGt + sBeta + 4 mbarrier rings + slack.
    STAGES_LOCAL = 3
    OUT_STAGES_LOCAL = 2
    smem_bytes = (
        D * D * 2  # sState
        + STAGES_LOCAL * 4 * (CHUNK * D * 2)  # sV/sKd/sQd/sKr staged
        + STAGES_LOCAL * 2 * (CHUNK * CHUNK * 2)  # sINV/sMqk staged
        + OUT_STAGES_LOCAL * (CHUNK * D * 2)  # sOut staged
        + (CHUNK * D * 2)  # sKr_T (padded conservatively to (D*CHUNK))
        + (D * 4)  # sGt
        + (CHUNK * 2)  # sBeta
        + STAGES_LOCAL * 8  # load-full mbarriers
        + STAGES_LOCAL * 8  # load-empty mbarriers
        + OUT_STAGES_LOCAL * 8  # store-full mbarriers
        + OUT_STAGES_LOCAL * 8  # store-empty mbarriers
        + 4096  # slack (alignment + KR_T_PAD overhead)
    )

    k2_phaseN_kernel(
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


_compiled_cache_k2N: dict = {}


def launch_k2_phaseN(
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
    if key not in _compiled_cache_k2N:
        stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
        v_flat = v.view(T_total, H, D)
        out_flat = out.view(T_total, H, D)
        _compiled_cache_k2N[key] = cute.compile(
            run_k2_phaseN,
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
    _compiled_cache_k2N[key](
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

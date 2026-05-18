# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SM90 (Hopper) Pre-Scan Kernel — warp-specialized, RS-WGMMA edition.

This is the Hopper counterpart of ``pre_scan.py`` (which targets SM100/Blackwell).
The high-level algorithm is identical to the SM100 version; the per-CTA
machinery is:

  * **1 producer warpgroup** (4 warps, 128 threads) — only warp 0 actually
    issues TMA loads for ``W``, ``K^T``, ``U`` and ``gk``.  The remaining
    warps in this warpgroup are idle and run with reduced register count.
  * **2 consumer (math) warpgroups** (2 × 128 = 256 threads) — together
    handle ``BV = 128`` rows of the V dimension via ``atom_layout = (2,1,1)``;
    each warpgroup MMA tile is ``M = 64``.
  * **RS-WGMMA** on both WH and KV — the FMHA trick:
      * State lives in registers in **KV accumulator** layout (fp32).
      * Before WH MMA, we cast state to bf16 via ``make_acc_into_op`` —
        producing an A-fragment view that matches WH's RMEM A operand.
      * After WH MMA, ``v_new = U - WH`` (he) / pre-negated WH (m) sits in
        registers in **WH accumulator** layout, again converted in-place to
        a KV A-fragment view.
      * Alignment: WH (M=BV, N=BT, K=BK) and KV (M=BV, N=BK, K=BT) share
        ``M`` and have ``WH.K == KV.N == BK`` and ``WH.N == KV.K == BT``,
        so the WGMMA accumulator-output / A-fragment-input layouts line up.
      * gk decay is a register-only element-wise multiply applied *before*
        the KV MMA; the KV MMA itself accumulates into ``state`` via
        ``ACCUMULATE=True`` so the entire ``state ± v_new @ K^T`` is one MMA
        (for m mode we pre-negate ``v_new`` so ``state - v_new @ K^T``
        becomes ``state + (-v_new) @ K^T``).

Tile sizes:
  * BV = 128 (CTA-level V tile, requires head_dim_v >= 128)
  * BV_per_wg = 64 (MMA M per consumer warpgroup)
  * BT = chunk_size (= 64 by default)
  * BK = head_dim_k (= 128)
  * BS = 64 (K-tile for m mode)
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warpgroup as warpgroup
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Float32, Int32, Int64

from cula.utils import USE_FAST_MATH, assert_hopper


def _coop_grp(size: int):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


# =====================================================================
# SM90 Fused CuTeDSL Kernel — warp-specialized, RS-WGMMA
# =====================================================================


class ChunkDeltaRulePreScanFusedSm90:
    """Hopper (sm_90a) warp-specialized pre-scan kernel."""

    def __init__(
        self,
        chunk_size: int = 64,
        head_dim_k: int = 128,
        head_dim_v: int = 128,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        io_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        use_fast_math: bool = True,
    ):
        assert head_dim_k == 128 and head_dim_v == 128, "SM90 pre_scan only supports K=V=128"
        assert_hopper()

        self.use_fast_math = use_fast_math
        self.chunk_size = chunk_size
        self.head_dim_k = head_dim_k
        self.head_dim_v = head_dim_v
        self.acc_dtype = acc_dtype
        self.io_dtype = io_dtype

        # ── Tile sizes ──
        self.BT = chunk_size  # 64
        self.BK = head_dim_k  # 128
        self.BV_per_wg = 64  # MMA M per consumer warpgroup
        self.num_mma_wgs = 2
        self.BV = self.BV_per_wg * self.num_mma_wgs  # 128 — CTA V tile
        self.BS = 64

        # ── Warp / thread layout ──
        self.threads_per_warp = 32
        self.threads_per_wg = 128
        self.num_load_wgs = 1
        self.threads_per_cta = (self.num_load_wgs + self.num_mma_wgs) * self.threads_per_wg

        self.load_wg_id = 0
        self.compute_wg_id_0 = 1
        self.compute_wg_id_1 = 2

        self.num_regs_load = 40
        self.num_regs_mma = 232

        # ── MMA tilers ──
        # WH MMA: state(BV, BK) @ W(BT, BK) → acc(BV, BT)
        # KV MMA: vnew (BV, BT) @ K^T(BK, BT) → acc(BV, BK)
        self.wh_mma_tiler_mn = (self.BV, self.BT)  # (128, 64)
        self.kv_mma_tiler_mn = (self.BV, self.BK)  # (128, 128)
        self.atom_layout_mnk = (self.num_mma_wgs, 1, 1)  # split M between 2 WGs
        self.wh_mma_tiler_mnk = (*self.wh_mma_tiler_mn, self.BK)  # K = 128
        self.kv_mma_tiler_mnk = (*self.kv_mma_tiler_mn, self.BT)  # K = 64

        # Pipeline stages
        self.w_stage = 3
        self.k_stage = 3
        self.u_stage = 2
        self.gk_stage = 2

        self.buffer_align_bytes = 1024

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------
    def _compute_grid(self, S_split, H, K, V):
        num_v_tiles = (V + self.BV - 1) // self.BV
        num_k_tiles = (K + self.BS - 1) // self.BS
        return (num_v_tiles + num_k_tiles, S_split * H, 1)

    @staticmethod
    @cute.jit
    def _convert_c_layout_to_a_layout(c_layout, a_inner_shape):
        """Re-interpret a WGMMA C-frag layout as a WGMMA A-frag layout.

        Mirrors HopperFusedMultiHeadAttentionForward.convert_c_layout_to_a_layout
        in cutlass/examples/python/CuTeDSL/hopper/fmha.py.
        """
        return cute.make_layout(
            (
                a_inner_shape,
                c_layout.shape[1],
                (c_layout.shape[2], cute.size(c_layout, mode=[0]) // cute.size(a_inner_shape)),
            ),
            stride=(
                c_layout.stride[0],
                c_layout.stride[1],
                (c_layout.stride[2], cute.size(a_inner_shape, mode=[2]) * c_layout.stride[0][2]),
            ),
        )

    @cute.jit
    def _make_acc_into_op(self, acc, operand_tv_layout_A, dtype):
        """Cast an accumulator register tensor in place to a bf16 RS-WGMMA A-frag.

        The data of ``acc`` (fp32, in WGMMA C-frag distribution) is element-wise
        cast to ``dtype`` and aliased as a new register tensor whose *layout*
        matches the consumer MMA's A-frag distribution.  Because of the Hopper
        WGMMA invariant ``C_layout(M) == A_layout(M)`` for the M dimension, the
        per-thread data ordering is correct — only the layout view changes.
        """
        operand = cute.make_rmem_tensor_like(
            self._convert_c_layout_to_a_layout(acc.layout, operand_tv_layout_A.shape[1]),
            dtype,
        )
        operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
        operand_as_acc.store(acc.load().to(dtype))
        return operand

    def _tma_partition_B(self, tma_atom, tma_tensor, smem, tile_shape_mnk, tiled_mma, hidx):
        """Partition B operand TMA tensors (mirror of SM100 helper)."""
        coord = (0, None, None)
        gX = cute.local_tile(
            tma_tensor,
            cute.slice_(tile_shape_mnk, coord),
            (None, None, (hidx, Int32(0))),
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgX = thr_mma.partition_B(gX)
        tXsX, tXgX = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 3),
            cute.group_modes(tCgX, 0, 3),
        )
        return tXsX, tXgX

    # ---------------------------------------------------------------
    # Host entry
    # ---------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        k_in: cute.Tensor,
        w_in: cute.Tensor,
        u_in: cute.Tensor,
        gk_in: cute.Tensor,
        hm_in: cute.Tensor,
        cu_seqlens_in: cute.Tensor,
        problem_size: tuple[Int32, Int32, Int32, Int32, Int32],
        use_gk: Int32,
        num_v_tiles: Int32,
        stream,
    ):
        k_ptr = k_in.iterator
        w_ptr = w_in.iterator
        u_ptr = u_in.iterator
        gk_ptr = gk_in.iterator
        hm_ptr = hm_in.iterator
        cu_seqlens_ptr = cu_seqlens_in.iterator

        S_split, T_total, H, K, V = problem_size

        # ===================== GMEM tensors =====================
        # K^T: (K, T, (H, 1)), K contiguous → B of KV MMA, MN-major
        kt = cute.make_tensor(
            k_ptr,
            cute.make_layout(
                (K, T_total, (H, Int32(1))),
                stride=(1, H * K, (K, T_total * H * K)),
            ),
        )
        # W: (T, K, (H, 1)), K contiguous → B of WH MMA, K-major
        w = cute.make_tensor(
            w_ptr,
            cute.make_layout(
                (T_total, K, (H, Int32(1))),
                stride=(H * K, 1, (K, T_total * H * K)),
            ),
        )
        # U^T: (V, T, (H, 1)), V contiguous (epilogue-style TMA)
        u_T = cute.make_tensor(
            u_ptr,
            cute.make_layout(
                (V, T_total, (H, Int32(1))),
                stride=(1, H * V, (V, T_total * H * V)),
            ),
        )
        # gk K-first: (K, T_gk, (H, 1)), K contiguous
        T_gk = gk_in.shape[0]
        gk_K = cute.make_tensor(
            gk_ptr,
            cute.make_layout(
                (K, T_gk, (H, Int32(1))),
                stride=(1, H * K, (K, T_gk * H * K)),
            ),
        )
        # Packed hm output: he in [:, :, :, :V], m in [:, :, :, V:]
        he = cute.make_tensor(
            hm_ptr,
            cute.make_layout(
                (K, V, (H, S_split)),
                stride=(V + K, 1, (K * (V + K), H * K * (V + K))),
            ),
        )
        m = cute.make_tensor(
            hm_ptr + V,
            cute.make_layout(
                (K, K, (H, S_split)),
                stride=(V + K, 1, (K * (V + K), H * K * (V + K))),
            ),
        )
        cu_seqlens = cute.make_tensor(cu_seqlens_ptr, cute.make_layout((S_split + 1,)))

        # ===================== MMA descriptors =====================
        # WH MMA: A in RMEM (state regs → bf16 frag), B = W (K-major).
        wh_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=self.wh_mma_tiler_mn,
            a_source=warpgroup.OperandSource.RMEM,
        )
        # KV MMA: A in RMEM (vnew regs → bf16 frag), B = K^T (MN-major).
        kv_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=self.kv_mma_tiler_mn,
            a_source=warpgroup.OperandSource.RMEM,
        )

        # ===================== SMEM layouts =====================
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()

        w_smem_staged = sm90_utils.make_smem_layout_b(
            utils.LayoutEnum.ROW_MAJOR,  # K-major B
            self.wh_mma_tiler_mnk,
            self.io_dtype,
            self.w_stage,
        )
        kt_smem_staged = sm90_utils.make_smem_layout_b(
            utils.LayoutEnum.COL_MAJOR,  # MN-major B
            self.kv_mma_tiler_mnk,
            self.io_dtype,
            self.k_stage,
        )
        u_epi_staged = sm90_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR,
            (self.BV, self.BT),
            self.u_stage,
        )

        # ===================== TMA atoms =====================
        cluster_shape = (1, 1, 1)
        w_smem_one = cute.select(w_smem_staged, mode=[0, 1, 2])
        tma_atom_w, tma_tensor_w = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            w,
            w_smem_one,
            self.wh_mma_tiler_mnk,
            wh_tiled_mma,
            cluster_shape,
        )
        kt_smem_one = cute.select(kt_smem_staged, mode=[0, 1, 2])
        tma_atom_kt, tma_tensor_kt = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            kt,
            kt_smem_one,
            self.kv_mma_tiler_mnk,
            kv_tiled_mma,
            cluster_shape,
        )
        u_smem_one = cute.select(u_epi_staged, mode=[0, 1])
        tma_atom_u, tma_tensor_u = cpasync.make_tiled_tma_atom(
            tma_load_op,
            u_T,
            u_smem_one,
            (self.BV, self.BT),
        )
        gk_smem_2d = cute.make_layout((self.BK, 1))
        tma_atom_gk, tma_tensor_gk = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gk_K,
            gk_smem_2d,
            (self.BK, 1),
        )

        self.tma_w_bytes = cute.size_in_bytes(self.io_dtype, w_smem_one)
        self.tma_kt_bytes = cute.size_in_bytes(self.io_dtype, kt_smem_one)
        self.tma_u_bytes = cute.size_in_bytes(self.io_dtype, u_smem_one)
        self.tma_gk_bytes = self.BK * 4  # fp32

        # ===================== SharedStorage =====================
        @cute.struct
        class SharedStorage:
            w_mbar: cute.struct.MemRange[Int64, self.w_stage * 2]
            kt_mbar: cute.struct.MemRange[Int64, self.k_stage * 2]
            u_mbar: cute.struct.MemRange[Int64, self.u_stage * 2]
            gk_mbar: cute.struct.MemRange[Int64, self.gk_stage * 2]

            sW: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(w_smem_staged)],
                self.buffer_align_bytes,
            ]
            sKt: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(kt_smem_staged)],
                self.buffer_align_bytes,
            ]
            sU: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(u_epi_staged)],
                self.buffer_align_bytes,
            ]
            sGK: cute.struct.Align[
                cute.struct.MemRange[Float32, self.BK * self.gk_stage],
                128,
            ]

        self.shared_storage = SharedStorage
        self.grid = self._compute_grid(S_split, H, K, V)

        self.kernel(
            wh_tiled_mma,
            kv_tiled_mma,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_kt,
            tma_tensor_kt,
            tma_atom_u,
            tma_tensor_u,
            tma_atom_gk,
            tma_tensor_gk,
            he,
            m,
            w_smem_staged,
            kt_smem_staged,
            u_epi_staged,
            cu_seqlens,
            problem_size,
            use_gk,
            num_v_tiles,
        ).launch(
            grid=self.grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    # ---------------------------------------------------------------
    # Device kernel
    # ---------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        wh_tiled_mma: cute.TiledMma,
        kv_tiled_mma: cute.TiledMma,
        tma_atom_w: cute.CopyAtom,
        tma_tensor_w: cute.Tensor,
        tma_atom_kt: cute.CopyAtom,
        tma_tensor_kt: cute.Tensor,
        tma_atom_u: cute.CopyAtom,
        tma_tensor_u: cute.Tensor,
        tma_atom_gk: cute.CopyAtom,
        tma_tensor_gk: cute.Tensor,
        he_tensor: cute.Tensor,
        m_tensor: cute.Tensor,
        w_smem_staged: cute.ComposedLayout,
        kt_smem_staged: cute.ComposedLayout,
        u_epi_staged: cute.ComposedLayout,
        cu_seqlens: cute.Tensor,
        problem_size: tuple[Int32, Int32, Int32, Int32, Int32],
        use_gk: Int32,
        num_v_tiles: Int32,
    ):
        S_split, T_total, H, K, V = problem_size

        tidx, _, _ = cute.arch.thread_idx()
        wg_idx = cute.arch.make_warp_uniform(tidx // self.threads_per_wg)
        warp_in_wg = cute.arch.make_warp_uniform((tidx // self.threads_per_warp) % 4)

        # ===================== SMEM allocation =====================
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sW = storage.sW.get_tensor(w_smem_staged.outer, swizzle=w_smem_staged.inner)
        sKt = storage.sKt.get_tensor(kt_smem_staged.outer, swizzle=kt_smem_staged.inner)
        sU_epi = storage.sU.get_tensor(u_epi_staged.outer, swizzle=u_epi_staged.inner)
        sGK_smem = storage.sGK.get_tensor(cute.make_layout((self.BK, self.gk_stage)))

        # ===================== Pipelines =====================
        load_prod_grp = _coop_grp(1)  # 1 effective producer thread
        compute_cons_grp = _coop_grp(self.num_mma_wgs * self.threads_per_wg)

        w_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.w_stage,
            producer_group=load_prod_grp,
            consumer_group=compute_cons_grp,
            tx_count=self.tma_w_bytes,
            barrier_storage=storage.w_mbar.data_ptr(),
        )
        kt_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.k_stage,
            producer_group=load_prod_grp,
            consumer_group=compute_cons_grp,
            tx_count=self.tma_kt_bytes,
            barrier_storage=storage.kt_mbar.data_ptr(),
        )
        u_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.u_stage,
            producer_group=load_prod_grp,
            consumer_group=compute_cons_grp,
            tx_count=self.tma_u_bytes,
            barrier_storage=storage.u_mbar.data_ptr(),
        )
        gk_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.gk_stage,
            producer_group=load_prod_grp,
            consumer_group=compute_cons_grp,
            tx_count=self.tma_gk_bytes,
            barrier_storage=storage.gk_mbar.data_ptr(),
        )

        # ===================== Work-unit decode =====================
        tile_idx = cute.arch.block_idx()[0]
        combined = cute.arch.block_idx()[1]
        i_subseq = combined // H
        i_h = combined % H
        bos = cu_seqlens[i_subseq]
        eos = cu_seqlens[i_subseq + 1]
        seq_len = eos - bos
        NT = (seq_len + self.BT - 1) // self.BT

        is_he_mode = tile_idx < num_v_tiles

        # =========================================================
        # ============== Producer warpgroup (load) ================
        # =========================================================
        if wg_idx == self.load_wg_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_load)

            if warp_in_wg == 0:
                # Per-CTA TMA tensor views with bos offset.
                tma_tensor_w_v = cute.domain_offset((bos, 0, (0, 0)), tma_tensor_w)
                tma_tensor_kt_v = cute.domain_offset((0, bos, (0, 0)), tma_tensor_kt)
                tma_tensor_u_v = cute.domain_offset((0, bos, (0, 0)), tma_tensor_u)
                tma_tensor_gk_v = cute.domain_offset((0, bos, (0, 0)), tma_tensor_gk)

                tWsW, tWgW = self._tma_partition_B(tma_atom_w, tma_tensor_w_v, sW, self.wh_mma_tiler_mnk, wh_tiled_mma, i_h)
                tKsK, tKgK = self._tma_partition_B(tma_atom_kt, tma_tensor_kt_v, sKt, self.kv_mma_tiler_mnk, kv_tiled_mma, i_h)

                # U epilogue-style partition.
                gU = tma_tensor_u_v[None, None, (i_h, Int32(0))]
                gU_epi = cute.flat_divide(gU, (self.BV, self.BT))
                sU_g = cute.group_modes(sU_epi, 0, 2)
                gU_g = cute.group_modes(gU_epi, 0, 2)
                bSG_sU, bSG_gU = cpasync.tma_partition(
                    tma_atom_u,
                    0,
                    cute.make_layout(1),
                    sU_g,
                    gU_g,
                )

                # gk: single (BK, 1) tile per chunk
                gGK = tma_tensor_gk_v[None, None, (i_h, Int32(0))]
                gGK_epi = cute.flat_divide(gGK, (self.BK, 1))
                sGK_view = cute.make_tensor(
                    sGK_smem.iterator,
                    cute.make_layout(
                        (self.BK, 1, self.gk_stage),
                        stride=(1, self.BK, self.BK),
                    ),
                )
                bSG_sGK, bSG_gGK = cpasync.tma_partition(
                    tma_atom_gk,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sGK_view, 0, 2),
                    cute.group_modes(gGK_epi, 0, 2),
                )

                w_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.w_stage)
                kt_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.k_stage)
                u_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.u_stage)
                gk_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.gk_stage)

                for chunk_idx in cutlass.range(0, NT, unroll=0):
                    # Load W chunk_idx
                    w_pipeline.producer_acquire(w_prod)
                    cute.copy(
                        tma_atom_w,
                        tWgW[None, chunk_idx, 0],
                        tWsW[None, w_prod.index],
                        tma_bar_ptr=w_pipeline.producer_get_barrier(w_prod),
                    )
                    w_pipeline.producer_commit(w_prod)
                    w_prod.advance()

                    # Load U[v, t=chunk*BT..(chunk+1)*BT] — only used in he-mode.
                    if is_he_mode:
                        u_pipeline.producer_acquire(u_prod)
                        cute.copy(
                            tma_atom_u,
                            bSG_gU[None, 0, chunk_idx],
                            bSG_sU[None, u_prod.index],
                            tma_bar_ptr=u_pipeline.producer_get_barrier(u_prod),
                        )
                        u_pipeline.producer_commit(u_prod)
                        u_prod.advance()

                    # Load K^T[chunk_idx]
                    kt_pipeline.producer_acquire(kt_prod)
                    cute.copy(
                        tma_atom_kt,
                        tKgK[None, 0, chunk_idx],
                        tKsK[None, kt_prod.index],
                        tma_bar_ptr=kt_pipeline.producer_get_barrier(kt_prod),
                    )
                    kt_pipeline.producer_commit(kt_prod)
                    kt_prod.advance()

                    # Load gk[t = last valid token in this chunk]
                    if use_gk != 0:
                        remaining = seq_len - chunk_idx * self.BT
                        if remaining < self.BT:
                            gk_t_idx = seq_len - 1
                        else:
                            gk_t_idx = chunk_idx * self.BT + self.BT - 1

                        gk_pipeline.producer_acquire(gk_prod)
                        cute.copy(
                            tma_atom_gk,
                            bSG_gGK[None, gk_t_idx, 0],
                            bSG_sGK[None, gk_prod.index],
                            tma_bar_ptr=gk_pipeline.producer_get_barrier(gk_prod),
                        )
                        gk_pipeline.producer_commit(gk_prod)
                        gk_prod.advance()

            # Other warps in the load WG are idle.
            return

        # =========================================================
        # ============= Consumer (math) warpgroups ================
        # =========================================================
        cute.arch.warpgroup_reg_alloc(self.num_regs_mma)

        # Each consumer WG identifies itself locally (0 or 1 of the 2 mma WGs).
        wg_local = wg_idx - self.compute_wg_id_0

        # Per-WG MMA slice — for atom_layout=(2,1,1) each slice covers
        # rows [wg_local*64, (wg_local+1)*64) of the BV=128 M tile.
        thr_mma_wh = wh_tiled_mma.get_slice(wg_local)
        thr_mma_kv = kv_tiled_mma.get_slice(wg_local)

        # B operands (SMEM → WGMMA fragment).
        tCsW = thr_mma_wh.partition_B(sW)
        tCrW = wh_tiled_mma.make_fragment_B(tCsW)
        tCsKt = thr_mma_kv.partition_B(sKt)
        tCrKt = kv_tiled_mma.make_fragment_B(tCsKt)

        # Accumulator register tensors (C-layout).
        wh_acc_shape = thr_mma_wh.partition_shape_C(self.wh_mma_tiler_mn)
        kv_acc_shape = thr_mma_kv.partition_shape_C(self.kv_mma_tiler_mn)
        acc_wh = thr_mma_wh.make_fragment_C(wh_acc_shape)
        state = thr_mma_kv.make_fragment_C(kv_acc_shape)

        # Identity tensors → per-element (M, N) global coords per thread.
        cM_kv = cute.make_identity_tensor(self.kv_mma_tiler_mn)
        tCcM_kv = thr_mma_kv.partition_C(cM_kv)
        cM_wh = cute.make_identity_tensor(self.wh_mma_tiler_mn)
        tCcM_wh = thr_mma_wh.partition_C(cM_wh)

        # ===================== Initialise state =====================
        if is_he_mode:
            for ei in cutlass.range(cute.size(state), unroll_full=True):
                state[ei] = Float32(0.0)
        else:
            k_col_tile = tile_idx - num_v_tiles
            for ei in cutlass.range(cute.size(state), unroll_full=True):
                v_coord, k_coord = tCcM_kv[ei]
                col_global = v_coord + k_col_tile * self.BS
                if k_coord == col_global:
                    state[ei] = Float32(1.0)
                else:
                    state[ei] = Float32(0.0)

        # Consumer pipeline states.
        w_cons_r = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.w_stage)
        w_cons_e = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.w_stage)
        kt_cons_r = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.k_stage)
        kt_cons_e = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.k_stage)
        u_cons_r = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.u_stage)
        u_cons_e = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.u_stage)
        gk_cons_r = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.gk_stage)
        gk_cons_e = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.gk_stage)

        # ===================== Main chunk loop =====================
        for chunk_idx in cutlass.range(0, NT, unroll=0):
            # ----- WH MMA: acc_wh = state_bf16 @ sW (ACC=False; RS-WGMMA) -----
            wh_a_frag = self._make_acc_into_op(state, wh_tiled_mma.tv_layout_A, self.io_dtype)

            w_pipeline.consumer_wait(w_cons_r)
            warpgroup.fence()
            num_k_blocks_wh = cute.size(tCrW, mode=[2])
            wh_tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
            for kp in cutlass.range(num_k_blocks_wh, unroll_full=True):
                cute.gemm(
                    wh_tiled_mma,
                    acc_wh,
                    wh_a_frag[None, None, kp],
                    tCrW[None, None, kp, w_cons_r.index],
                    acc_wh,
                )
                wh_tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            warpgroup.wait_group(0)
            w_pipeline.consumer_release(w_cons_e)
            w_cons_r.advance()
            w_cons_e.advance()

            # ----- CUDA: v_new = U - WH (he) or v_new = -WH (m); mask tail -----
            if is_he_mode:
                u_pipeline.consumer_wait(u_cons_r)
                for ei in cutlass.range(cute.size(acc_wh), unroll_full=True):
                    v_coord, t_coord = tCcM_wh[ei]
                    u_val = sU_epi[(v_coord, t_coord, u_cons_r.index)].to(self.acc_dtype)
                    acc_wh[ei] = u_val - acc_wh[ei]
                u_pipeline.consumer_release(u_cons_e)
                u_cons_r.advance()
                u_cons_e.advance()
            else:
                # m mode: state -= v_new @ K^T  →  pre-negate so we can re-use
                # the accumulating KV MMA: state += (-v_new) @ K^T.
                for ei in cutlass.range(cute.size(acc_wh), unroll_full=True):
                    acc_wh[ei] = -acc_wh[ei]

            # Varlen tail mask (zero out columns beyond seq_len).
            valid_len_chunk = seq_len - chunk_idx * self.BT
            if valid_len_chunk < self.BT:
                for ei in cutlass.range(cute.size(acc_wh), unroll_full=True):
                    _, t_coord = tCcM_wh[ei]
                    if t_coord >= valid_len_chunk:
                        acc_wh[ei] = Float32(0.0)

            # ----- gk decay (applied before KV MMA) -----
            if use_gk != 0:
                gk_pipeline.consumer_wait(gk_cons_r)
                # Apply exp2 cooperatively across all consumer threads.
                local_tid = tidx - self.compute_wg_id_0 * self.threads_per_wg
                if local_tid < self.BK:
                    sGK_smem[(local_tid, gk_cons_r.index)] = cute.exp2(
                        sGK_smem[(local_tid, gk_cons_r.index)],
                        fastmath=self.use_fast_math,
                    )
                cute.arch.barrier_arrive(barrier_id=1, number_of_threads=self.num_mma_wgs * self.threads_per_wg)
                cute.arch.barrier_wait(barrier_id=1, number_of_threads=self.num_mma_wgs * self.threads_per_wg)
                for ei in cutlass.range(cute.size(state), unroll_full=True):
                    _, k_coord = tCcM_kv[ei]
                    state[ei] = state[ei] * sGK_smem[(k_coord, gk_cons_r.index)]
                gk_pipeline.consumer_release(gk_cons_e)
                gk_cons_r.advance()
                gk_cons_e.advance()

            # ----- KV MMA: state += v_new_bf16 @ sKt (ACC=True; RS-WGMMA) -----
            kv_a_frag = self._make_acc_into_op(acc_wh, kv_tiled_mma.tv_layout_A, self.io_dtype)

            kt_pipeline.consumer_wait(kt_cons_r)
            warpgroup.fence()
            num_k_blocks_kv = cute.size(tCrKt, mode=[2])
            kv_tiled_mma.set(warpgroup.Field.ACCUMULATE, True)  # accumulate INTO state
            for kp in cutlass.range(num_k_blocks_kv, unroll_full=True):
                cute.gemm(
                    kv_tiled_mma,
                    state,
                    kv_a_frag[None, None, kp],
                    tCrKt[None, None, kp, kt_cons_r.index],
                    state,
                )
            warpgroup.commit_group()
            warpgroup.wait_group(0)
            kt_pipeline.consumer_release(kt_cons_e)
            kt_cons_r.advance()
            kt_cons_e.advance()

        # ===================== Write state to GMEM =====================
        if is_he_mode:
            for ei in cutlass.range(cute.size(state), unroll_full=True):
                v_coord, k_coord = tCcM_kv[ei]
                he_tensor[(k_coord, v_coord + tile_idx * self.BV, (i_h, i_subseq))] = state[ei]
        else:
            k_col_tile = tile_idx - num_v_tiles
            for ei in cutlass.range(cute.size(state), unroll_full=True):
                v_coord, k_coord = tCcM_kv[ei]
                col_global = v_coord + k_col_tile * self.BS
                m_tensor[(k_coord, col_global, (i_h, i_subseq))] = state[ei]


# =====================================================================
# Compile cache + Python entry point used by pre_scan.py dispatcher
# =====================================================================


_pre_scan_sm90_kernel_cache: dict = {}


def _compile_pre_scan_sm90_variant(H, K, V, chunk_size, use_fast_math):
    """Compile one SM90 ``ChunkDeltaRulePreScanFusedSm90`` kernel variant."""
    kernel_obj = ChunkDeltaRulePreScanFusedSm90(
        chunk_size=chunk_size,
        head_dim_k=K,
        head_dim_v=V,
        use_fast_math=use_fast_math,
    )

    sym_t = cute.sym_int()
    sym_s = cute.sym_int()
    sym_cu = cute.sym_int()
    sym_gk = cute.sym_int()

    k_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_t, H, K), stride_order=(2, 1, 0), assumed_align=128)
    w_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_t, H, K), stride_order=(2, 1, 0), assumed_align=128)
    u_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_t, H, V), stride_order=(2, 1, 0), assumed_align=128)
    gk_fake = make_fake_compact_tensor(cutlass.Float32, (sym_gk, H, K), stride_order=(2, 1, 0), assumed_align=128)
    hm_fake = make_fake_compact_tensor(
        cutlass.Float32,
        (sym_s, H, K, V + K),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    cu_fake = make_fake_compact_tensor(cutlass.Int32, (sym_cu,), assumed_align=128)
    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_fn = cute.compile(
        kernel_obj,
        k_fake,
        w_fake,
        u_fake,
        gk_fake,
        hm_fake,
        cu_fake,
        (Int32(1), Int32(1), Int32(H), Int32(K), Int32(V)),
        Int32(0),
        Int32(0),
        stream_fake,
        options="--enable-tvm-ffi",
    )
    return compiled_fn


def get_compiled_pre_scan_sm90(H, K, V, chunk_size):
    """Get cached compiled SM90 pre-scan kernel."""
    key = (H, K, V, chunk_size, USE_FAST_MATH)
    if key not in _pre_scan_sm90_kernel_cache:
        _pre_scan_sm90_kernel_cache[key] = _compile_pre_scan_sm90_variant(H, K, V, chunk_size, USE_FAST_MATH)
    return _pre_scan_sm90_kernel_cache[key]

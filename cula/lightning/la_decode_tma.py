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
Linear Attention Decode Kernel - TMA variant (Hopper / Blackwell, sm_90+)

Key difference vs la_decode.py:
  - Uses TMA (Tensor Memory Accelerator) bulk async copy instead of
    per-thread cp.async for loading state tiles GMEM → SMEM.
  - Only 1 elected thread per CTA issues the TMA copy; all other threads
    overlap compute with the async DMA transfer.
  - mbarrier (hardware barrier) replaces cp_async_wait_group for
    completion signalling.

Why TMA over cp.async here:
  1. Lower instruction pressure  — a single thread issues one
     cp.async.bulk.tensor instruction instead of 128 threads issuing 128
     individual 128-bit cp.async instructions.
  2. All 128 threads are free to overlap useful work (register loads,
     muls) with the state DMA transfer.
  3. TMA hardware automatically handles out-of-bounds predication for
     remainder tiles.

Requires sm_90+ (Hopper or Blackwell).

Core Formula:
    state_new = exp(decay) * state_old + k ⊗ v
    output    = q @ state_new
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects.arith import constant as arith_const
from cutlass._mlir import ir
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

from cula.utils import USE_FAST_MATH

# ============================================================================
# Global configuration (must match la_decode.py)
# ============================================================================
TILE_V = 8
TILE_K = 128
NUM_STAGES = 2
NUM_THREADS = 128  # 4 warps
NUM_BLOCKS_PER_STATE = 8

# Bytes transferred by one TMA copy (one TILE_V × TILE_K float32 tile)
TMA_TILE_BYTES = TILE_V * TILE_K * 4


# ============================================================================
# Big-batch kernel  (one CTA per batch×head, no further V-split)
# ============================================================================

@cute.kernel
def la_decode_kernel_big_batch_tma(
    tma_atom: cute.CopyAtom,
    tma_tensor: cute.Tensor,    # ArithTuple coordinate tensor  (V, K, B*H)
    h0_source: cute.Tensor,     # raw float32 tensor (B*H, V, K)  — for writeback
    smem_layout_staged: cute.Layout,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    decay_scales: cute.Tensor,  # [H]
    q: cute.Tensor,             # [B, H, K]
    k: cute.Tensor,             # [B, H, K]
    v: cute.Tensor,             # [B, HV, V]
    o: cute.Tensor,             # [B, HV, V]
    h0_indices: cute.Tensor,    # [B]  (unused, kept for API compat)
    scale: cutlass.Constexpr[float],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
):
    """TMA-based state-load kernel (big-batch path).

    Pipeline overview (per CTA):
      Prefetch stage  : tidx==0 issues TMA for tile 0  → sMbar[0]
      Main loop iter i:
        1.  mbarrier_wait(sMbar[stage], phase)    — wait for tile i
        2.  tidx==0 issues TMA for tile i+1      → sMbar[next_stage]  (async)
        3.  All 128 threads compute state update and partial q·state dot
      Final barrier + output write-back.
    """
    HV = H
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    batch_idx, _, _ = cute.arch.block_idx()
    i_n = batch_idx // HV
    i_hv = batch_idx % HV
    i_h = i_hv // (HV // H)

    # -----------------------------------------------------------------------
    # SMEM allocation
    # -----------------------------------------------------------------------
    smem = cutlass.utils.SmemAllocator()

    # State tile double-buffer  (TILE_V × TILE_K × NUM_STAGES)  float32
    sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)

    # Output accumulator  (V)  bfloat16
    sOutput = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((V,)), 16)

    # mbarrier array  (NUM_STAGES × 8 bytes)
    sMbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((NUM_STAGES,)), 8)
    sMbar_ptr = sMbar.iterator   # cute.Pointer to the first Int64 mbarrier slot

    # -----------------------------------------------------------------------
    # mbarrier initialisation  (elect_one — single thread per CTA)
    # -----------------------------------------------------------------------
    # elect_one() selects 1 thread per *warp*.  To touch the mbarrier from a
    # single CTA thread we guard with warp_idx == 0 first.
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    if warp_idx == 0:
        with cute.arch.elect_one():
            for s in cutlass.range_constexpr(NUM_STAGES):
                cute.arch.mbarrier_init(sMbar_ptr + s, cutlass.Int32(1))

    # fence.mbarrier.init must be visible to the ENTIRE CTA —
    # call from ALL threads (not just warp 0) before the barrier.
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    # -----------------------------------------------------------------------
    # Prepare GMEM coordinate and writeback tensors
    # -----------------------------------------------------------------------
    # Tile the TMA coordinate tensor over (V, K) dims; batch is the last dim.
    # tma_tensor shape: (V, K, B*H) — 2D tiler (TILE_V, TILE_K) hits V and K.
    # local_tile result: (TILE_V, TILE_K, num_v_tiles, 1, B*H)
    gSrc_tma = cute.local_tile(tma_tensor, (TILE_V, TILE_K), (None, None, None))

    # Write-back slice for updated state
    gDst = cute.local_tile(h0_source, (1, TILE_V, TILE_K), (batch_idx, None, 0))

    # -----------------------------------------------------------------------
    # TMA partition — done ONCE outside the loop on the full staged tensors.
    # Following the dense_gemm.py / fmha_bwd.py standard CUTLASS pattern:
    #   group_modes(sData,    0, 2) → (TILE_V*TILE_K, NUM_STAGES)
    #   group_modes(gSrc_tma, 0, 2) → (TILE_V*TILE_K, num_v_tiles)
    # tma_partition returns:
    # tS_part : (atom, NUM_STAGES)           — SMEM slot per stage
    #   tG_part : (atom, num_v_tiles, 1, B*H) — GMEM coord per (v-tile, batch)
    # In the loop we index: tG_part[(None, v, 0, batch_idx)] and tS_part[(None, stage)]
    # -----------------------------------------------------------------------
    tS_part, tG_part = cpasync.tma_partition(
        tma_atom, 0, cute.make_layout(1),
        cute.group_modes(sData,    0, 2),
        cute.group_modes(gSrc_tma, 0, 2),
    )

    # -----------------------------------------------------------------------
    # Register preload: q, k, v (all fit as num_v_tiles × 32 elements / warp)
    # -----------------------------------------------------------------------
    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_q = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_v = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_h = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)

    # -----------------------------------------------------------------------
    # Prefetch: issue TMA for tile 0 → stage 0 before entering the main loop.
    # -----------------------------------------------------------------------
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(
                sMbar_ptr, cutlass.Int32(TMA_TILE_BYTES)
            )
        cute.copy(tma_atom, tG_part[(None, 0, 0, batch_idx)], tS_part[(None, 0)],
                  tma_bar_ptr=sMbar_ptr)

    # Load q, k, v into registers (all threads, overlaps with TMA in flight)
    # range_constexpr unrolls the loop at compile time, making r_q[i] statically
    # indexed (i becomes 0,1,2,3 at compile time) — eliminates register spilling.
    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = cutlass.Float32(q[i_n, i_h, i * 32 + lane_id])
        r_k[i] = cutlass.Float32(k[i_n, i_h, i * 32 + lane_id])
        r_v[i] = cutlass.Float32(v[i_n, i_hv, i * 32 + lane_id])

    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = r_q[i] * scale

    r_g = cute.exp(-cutlass.Float32(decay_scales[i_h]), fastmath=USE_FAST_MATH)

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    for v_tiles in range(num_v_tiles):
        stage = v_tiles % NUM_STAGES
        phase = (v_tiles // NUM_STAGES) % 2

        # Step 1: wait for current tile's TMA to complete (all threads spin here)
        cute.arch.mbarrier_wait(sMbar_ptr + stage, phase)

        # Step 2: warp 0 issues TMA for next tile (1-stage lookahead).
        next_v = v_tiles + 1
        if next_v < num_v_tiles:
            next_stage = next_v % NUM_STAGES
            if warp_idx == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        sMbar_ptr + next_stage, cutlass.Int32(TMA_TILE_BYTES)
                    )
                cute.copy(tma_atom,
                          tG_part[(None, next_v, 0, batch_idx)],
                          tS_part[(None, next_stage)],
                          tma_bar_ptr=sMbar_ptr + next_stage)

        # Step 3: compute  state_new = decay * state_old + k ⊗ v
        #         readout  o[v] = q · state_new[v, :]
        # ----------------------------------------------------------------
        # Hoist v_src outside the row loop:
        #   v_idx // 32 == v_tiles // 4  (since row + row_offset < TILE_V = 8 < 32)
        # Use static if-elif to avoid dynamic r_v indexing, which would spill.
        # ----------------------------------------------------------------
        v_src = cutlass.Float32(0.0)
        v_grp = v_tiles // 4
        if v_grp == 0:
            v_src = r_v[0]
        elif v_grp == 1:
            v_src = r_v[1]
        elif v_grp == 2:
            v_src = r_v[2]
        else:
            v_src = r_v[3]

        for row in cutlass.range_constexpr(0, TILE_V, 4):
            row_offset = tidx // 32

            v_idx = v_tiles * TILE_V + row + row_offset
            v_row = cute.arch.shuffle_sync(
                v_src, v_idx % 32, mask=-1, mask_and_clamp=31
            )

            sum_hq = 0.0
            for i in cutlass.range_constexpr(vec_size):
                r_h[i] = sData[(row + row_offset, i * 32 + lane_id, stage)]
                r_h[i] = r_h[i] * r_g + r_k[i] * v_row
                gDst[(0, row + row_offset, i * 32 + lane_id, v_tiles)] = r_h[i]
                sum_hq += r_h[i] * r_q[i]

            for offset in [16, 8, 4, 2, 1]:
                sum_hq += cute.arch.shuffle_sync_bfly(
                    sum_hq, offset=offset, mask=-1, mask_and_clamp=31
                )

            o_idx = v_tiles * TILE_V + row + row_offset
            if lane_id == 0 and o_idx < V:
                sOutput[o_idx] = cutlass.BFloat16(sum_hq)

    # -----------------------------------------------------------------------
    # Output writeback
    # -----------------------------------------------------------------------
    cute.arch.barrier()
    if tidx < V:
        o[(i_n, i_hv, tidx)] = sOutput[tidx]


# ============================================================================
# Small-batch kernel  (NUM_BLOCKS_PER_STATE CTAs per batch×head)
# ============================================================================

@cute.kernel
def la_decode_kernel_small_batch_tma(
    tma_atom: cute.CopyAtom,
    tma_tensor: cute.Tensor,    # ArithTuple coordinate tensor  (V, K, B*H)
    h0_source: cute.Tensor,     # raw float32 tensor (B*H, V, K)
    smem_layout_staged: cute.Layout,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    decay_scales: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    scale: cutlass.Constexpr[float],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
):
    """TMA-based state-load kernel (small-batch path, V-dimension split)."""
    HV = H
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    block_idx, _, _ = cute.arch.block_idx()
    batch_idx = block_idx // NUM_BLOCKS_PER_STATE
    batch_inner = block_idx % NUM_BLOCKS_PER_STATE
    num_v_tiles_per_block = num_v_tiles // NUM_BLOCKS_PER_STATE
    i_n = batch_idx // HV
    i_hv = batch_idx % HV
    i_h = i_hv // (HV // H)

    smem = cutlass.utils.SmemAllocator()
    sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
    sOutput = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((V,)), 16)
    sMbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((NUM_STAGES,)), 8)
    sMbar_ptr = sMbar.iterator

    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    if warp_idx == 0:
        with cute.arch.elect_one():
            for s in cutlass.range_constexpr(NUM_STAGES):
                cute.arch.mbarrier_init(sMbar_ptr + s, cutlass.Int32(1))

    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    tma_batch = tma_tensor
    gSrc_tma = cute.local_tile(tma_batch, (TILE_V, TILE_K), (None, None, None))
    gDst = cute.local_tile(h0_source, (1, TILE_V, TILE_K), (batch_idx, None, 0))

    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_q = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_v = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_h = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)

    r_decay_scale = -cutlass.Float32(decay_scales[i_h])
    r_decay = cute.exp(r_decay_scale, fastmath=USE_FAST_MATH)

    start_v_tiles = batch_inner * num_v_tiles_per_block
    end_v_tiles = start_v_tiles + num_v_tiles_per_block

    # TMA partition — once outside the loop on the full staged tensors
    tS_part, tG_part = cpasync.tma_partition(
        tma_atom, 0, cute.make_layout(1),
        cute.group_modes(sData,    0, 2),
        cute.group_modes(gSrc_tma, 0, 2),
    )

    # Prefetch tile start_v_tiles → stage 0
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(
                sMbar_ptr, cutlass.Int32(TMA_TILE_BYTES)
            )
        cute.copy(tma_atom, tG_part[(None, start_v_tiles, 0, batch_idx)], tS_part[(None, 0)],
                  tma_bar_ptr=sMbar_ptr)

    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = cutlass.Float32(q[i_n, i_h, i * 32 + lane_id])
        r_k[i] = cutlass.Float32(k[i_n, i_h, i * 32 + lane_id])
        r_v[i] = cutlass.Float32(v[i_n, i_hv, i * 32 + lane_id])
    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = r_q[i] * scale

    for v_tiles in range(start_v_tiles, end_v_tiles):
        stage = (v_tiles - start_v_tiles) % NUM_STAGES
        phase = ((v_tiles - start_v_tiles) // NUM_STAGES) % 2

        cute.arch.mbarrier_wait(sMbar_ptr + stage, phase)

        next_v = v_tiles + 1   # 1-stage lookahead
        if next_v < end_v_tiles:
            next_stage = (next_v - start_v_tiles) % NUM_STAGES
            if warp_idx == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        sMbar_ptr + next_stage, cutlass.Int32(TMA_TILE_BYTES)
                    )
                cute.copy(tma_atom,
                          tG_part[(None, next_v, 0, batch_idx)],
                          tS_part[(None, next_stage)],
                          tma_bar_ptr=sMbar_ptr + next_stage)

        v_src = cutlass.Float32(0.0)
        v_grp = v_tiles // 4
        if v_grp == 0:
            v_src = r_v[0]
        elif v_grp == 1:
            v_src = r_v[1]
        elif v_grp == 2:
            v_src = r_v[2]
        else:
            v_src = r_v[3]

        for row in cutlass.range_constexpr(0, TILE_V, 4):
            row_offset = tidx // 32
            v_idx = v_tiles * TILE_V + row + row_offset
            v_row = cute.arch.shuffle_sync(
                v_src, v_idx % 32, mask=-1, mask_and_clamp=31
            )
            sum_hq = 0.0
            for i in cutlass.range_constexpr(vec_size):
                r_h[i] = sData[(row + row_offset, i * 32 + lane_id, stage)]
                r_h[i] = r_h[i] * r_decay + r_k[i] * v_row
                gDst[(0, row + row_offset, i * 32 + lane_id, v_tiles)] = r_h[i]
                sum_hq += r_h[i] * r_q[i]

            for offset in [16, 8, 4, 2, 1]:
                sum_hq += cute.arch.shuffle_sync_bfly(
                    sum_hq, offset=offset, mask=-1, mask_and_clamp=31
                )
            o_idx = v_tiles * TILE_V + row + row_offset
            if lane_id == 0 and o_idx < V:
                sOutput[o_idx] = cutlass.BFloat16(sum_hq)

    cute.arch.barrier()
    if tidx >= start_v_tiles * TILE_V and tidx < end_v_tiles * TILE_V:
        o[(i_n, i_hv, tidx)] = sOutput[tidx]


# ============================================================================
# JIT launcher functions
# ============================================================================

@cute.jit
def run_la_decode_kernel_big_batch_tma(
    h0_source: cute.Tensor,     # (B*H, V, K)
    decay_scales: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    softmax_scale: cutlass.Constexpr[float],
    H: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    batch_size, v_dim, _k_dim = (
        h0_source.layout.shape[0],
        h0_source.layout.shape[1],
        h0_source.layout.shape[2],
    )

    # SMEM layout for ONE stage (no stage dimension) — used by TMA descriptor
    smem_layout_one_stage = cute.make_layout((TILE_V, TILE_K), stride=(TILE_K, 1))

    # Build TMA atom + coordinate tensor.
    # Re-interpret h0_source as (V, K, B*H) so the 2D tiler (TILE_V, TILE_K)
    # correctly tiles the V and K dimensions.  The batch dim moves to last.
    h0_vkb = cute.make_tensor(
        h0_source.iterator,
        cute.make_layout((v_dim, _k_dim, batch_size), stride=(_k_dim, 1, v_dim * _k_dim))
    )
    tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        h0_vkb,
        smem_layout_one_stage,
        (TILE_V, TILE_K),
    )

    num_v_tiles = cute.ceil_div(v_dim, TILE_V)
    vec_size = TILE_K // 32

    # Staged SMEM layout (with stage dimension) — passed to kernel for allocation
    smem_layout_staged = cute.make_layout(
        (TILE_V, TILE_K, NUM_STAGES), stride=(TILE_K, 1, TILE_V * TILE_K)
    )

    # sData + sOutput + sMbar + alignment padding
    smem_bytes = (
        4 * TILE_V * TILE_K * NUM_STAGES   # sData  (float32)
        + 2 * v_dim                         # sOutput (bfloat16)
        + 8 * NUM_STAGES                    # sMbar  (int64)
        + 32                                # alignment headroom
    )

    la_decode_kernel_big_batch_tma(
        tma_atom,
        tma_tensor,
        h0_source,
        smem_layout_staged,
        vec_size,
        num_v_tiles,
        decay_scales,
        q,
        k,
        v,
        o,
        h0_indices,
        softmax_scale,
        B,
        T,
        H,
        K,
        V,
    ).launch(
        grid=(batch_size, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


@cute.jit
def run_la_decode_kernel_small_batch_tma(
    h0_source: cute.Tensor,     # (B*H, V, K)
    decay_scales: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    softmax_scale: cutlass.Constexpr[float],
    H: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    batch_size, v_dim, _k_dim = (
        h0_source.layout.shape[0],
        h0_source.layout.shape[1],
        h0_source.layout.shape[2],
    )

    smem_layout_one_stage = cute.make_layout((TILE_V, TILE_K), stride=(TILE_K, 1))

    # Re-interpret as (V, K, B*H) so the 2D tiler tiles V and K correctly
    h0_vkb = cute.make_tensor(
        h0_source.iterator,
        cute.make_layout((v_dim, _k_dim, batch_size), stride=(_k_dim, 1, v_dim * _k_dim))
    )
    tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        h0_vkb,
        smem_layout_one_stage,
        (TILE_V, TILE_K),
    )

    num_v_tiles = cute.ceil_div(v_dim, TILE_V)
    vec_size = TILE_K // 32

    smem_layout_staged = cute.make_layout(
        (TILE_V, TILE_K, NUM_STAGES), stride=(TILE_K, 1, TILE_V * TILE_K)
    )

    smem_bytes = (
        4 * TILE_V * TILE_K * NUM_STAGES
        + 2 * v_dim
        + 8 * NUM_STAGES
        + 32
    )

    la_decode_kernel_small_batch_tma(
        tma_atom,
        tma_tensor,
        h0_source,
        smem_layout_staged,
        vec_size,
        num_v_tiles,
        decay_scales,
        q,
        k,
        v,
        o,
        h0_indices,
        softmax_scale,
        B,
        T,
        H,
        K,
        V,
    ).launch(
        grid=(batch_size * NUM_BLOCKS_PER_STATE, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


# ============================================================================
# Public Python entry point
# ============================================================================

@functools.cache
def _get_compiled_kernel_tma(
    B: int, T: int, H: int, K: int, V: int, softmax_scale: float, use_fast_math: bool = True
):
    """Kernel compilation cache (TMA variant)."""
    return {}


def linear_attention_decode_tma(
    q: torch.Tensor,       # [B, H, K]
    k: torch.Tensor,       # [B, H, K]
    v: torch.Tensor,       # [B, HV, V]
    s: torch.Tensor,       # [pool_size, heads, V, K]
    out: torch.Tensor,     # [B, HV, V]
    softmax_scale: float,
    stride_q: int,
    stride_k: int,
    stride_v: int,
    stride_s: int,
    stride_o: int,
    s_offsets: torch.Tensor,
    decay_scales: torch.Tensor,
    HEAD_DIM: int,
    K_SPLIT_DIM: int,
    V_SPLIT_DIM: int,
) -> None:
    """Linear Attention Decode — TMA variant (sm_90+).

    Drop-in replacement for ``linear_attention_decode`` in la_decode.py.
    """
    B = q.shape[0]
    H = q.shape[1]

    k_dim_block = HEAD_DIM // K_SPLIT_DIM
    if k_dim_block > 1:
        raise NotImplementedError(
            f"TMA kernel does not support K splitting (k_dim_block={k_dim_block})"
        )

    cache_key = (B, 1, H, HEAD_DIM, HEAD_DIM, softmax_scale, USE_FAST_MATH)
    cache = _get_compiled_kernel_tma(*cache_key)

    if "compiled" not in cache:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        run_func = (
            run_la_decode_kernel_small_batch_tma if B <= 32
            else run_la_decode_kernel_big_batch_tma
        )

        h0_tensor = from_dlpack(s, assumed_align=16)
        decay_tensor = from_dlpack(decay_scales, assumed_align=16)
        q_tensor = from_dlpack(q, assumed_align=16)
        k_tensor = from_dlpack(k, assumed_align=16)
        v_tensor = from_dlpack(v, assumed_align=16)
        o_tensor = from_dlpack(out, assumed_align=16)
        h0_idx_tensor = from_dlpack(s_offsets, assumed_align=16)

        compiled = cute.compile(
            run_func,
            h0_tensor,
            decay_tensor,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            h0_idx_tensor,
            softmax_scale=softmax_scale,
            H=H,
            B=B,
            T=1,
            K=HEAD_DIM,
            V=HEAD_DIM,
            stream=stream,
            options="--enable-tvm-ffi",
        )
        cache["compiled"] = compiled

    compiled = cache["compiled"]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(s, decay_scales, q, k, v, out, s_offsets, stream)


def seg_la_d_kernel_cute_tma(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
    stride_q: int,
    stride_k: int,
    stride_v: int,
    stride_s: int,
    stride_o: int,
    s_offsets: torch.Tensor,
    decay_scales: torch.Tensor,
    HEAD_DIM: int,
    K_SPLIT_DIM: int,
    V_SPLIT_DIM: int,
) -> None:
    """Drop-in TMA replacement compatible with seg_la_d_kernel interface."""
    linear_attention_decode_tma(
        q, k, v, s, out,
        softmax_scale,
        stride_q, stride_k, stride_v, stride_s, stride_o,
        s_offsets, decay_scales,
        HEAD_DIM, K_SPLIT_DIM, V_SPLIT_DIM,
    )

# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
FlashKDA K1 (Prepare) — CuteDSL port.

This is a from-scratch CuteDSL reimplementation of FlashKDA's
``fwd_kernel1.cuh``. It is being built up phase-by-phase with per-phase
unit tests against a torch reference. The C++ kernel performs 9 phases per
(head, chunk) tile:

    1. TMA load q, k, beta, g_bf16, dt_bias               <- phase 1 (this file)
    2. L2 normalize q and k                                <- TODO
    3. Fused gate cumsum + k tail zero-fill                <- TODO
    4. exp_g_total                                         <- TODO
    5. decay_apply (q_decayed, k_decayed, k_inv, k_restored)<- TODO
    6. L_Mqk single-warp 16x16 GEMMs (fp16 / bf16 acc)     <- TODO
    7. Tril mask + INV = I - L                             <- TODO
    8. Neumann series 4-power inverse                      <- TODO
    9. TMA store 6 workspace tensors                       <- TODO

Reference implementation: ``/ossfs/workspace/FlashKDA/csrc/smxx/fwd_kernel1.cuh``.
SMEM byte budget per CTA: ~14 KB (uses unions, see C++).

Constraints for this port:
    * Fixed-len only (no varlen / no cu_seqlens) for the first cut.
    * head_dim_k = head_dim_v = 128.
    * CHUNK = 16.
    * Grid = (total_tiles, H), 256 threads per CTA.
    * sm_90+ (TMA, mbarrier).

Validation strategy: each phase appends its outputs to a debug workspace
buffer with a known stride. Unit tests dump the workspace and bit-compare
against a torch reference derived from the same C++ math.
"""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

# ---------------------------------------------------------------------------
# Constants — must match cula.ops.flashkda_prefill
# ---------------------------------------------------------------------------
CHUNK: int = 16
D: int = 128
THREADS_PER_CTA: int = 256

# Per-tile workspace bytes (must match C++ WorkspaceSizes).
# Layout: [k_decayed | q_decayed | k_restored | g_total | INV | Mqk]
_BYTES_KD = CHUNK * D * 2  # 4096
_BYTES_QD = CHUNK * D * 2  # 4096
_BYTES_KR = CHUNK * D * 2  # 4096
_BYTES_GT = D * 4  # 512
_BYTES_INV = CHUNK * CHUNK * 2  # 512
_BYTES_MQK = CHUNK * CHUNK * 2  # 512
WORKSPACE_BYTES_PER_TILE: int = _BYTES_KD + _BYTES_QD + _BYTES_KR + _BYTES_GT + _BYTES_INV + _BYTES_MQK
# Offsets into the per-tile workspace block (in bytes from the per-tile base).
_OFF_KD = 0
_OFF_QD = _OFF_KD + _BYTES_KD
_OFF_KR = _OFF_QD + _BYTES_QD
_OFF_GT = _OFF_KR + _BYTES_KR
_OFF_INV = _OFF_GT + _BYTES_GT
_OFF_MQK = _OFF_INV + _BYTES_INV


# ---------------------------------------------------------------------------
# Kernel — phase 1 only (TMA load q + workspace dump for verification)
# ---------------------------------------------------------------------------
# This is intentionally tiny: prove the K1 launch + TMA descriptor + per-tile
# workspace addressing all work end-to-end before adding compute phases.
@cute.kernel
def k1_phase1_tma_load_kernel(
    tma_atom_q: cute.CopyAtom,
    tma_tensor_q: cute.Tensor,  # ArithTuple coords: (T_total, H, D)
    workspace: cute.Tensor,  # raw bf16 view of the workspace, count = total_tiles*H*CHUNK*D
    H: cutlass.Constexpr[int],
    total_tiles: cutlass.Constexpr[int],
    T_total: cutlass.Constexpr[int],
):
    """K1 phase 1: TMA load q tile to SMEM, then dump to workspace KD slot.

    Grid: (total_tiles, H, 1), 256 threads/CTA.
    SMEM: q tile (CHUNK*D bf16) + 1 mbarrier (8 bytes).

    The dump uses simple vectorized stores (256 threads × 8 bf16 each = 2048
    elements = CHUNK*D). This is *not* representative of the final K1's TMA
    store; it exists to verify that the loaded q tile is correct.
    """
    tile_idx, head_idx, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    # SMEM allocation
    smem = cutlass.utils.SmemAllocator()
    q_smem = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((CHUNK, D), stride=(D, 1)),
        128,
    )
    sMbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((1,)), 8)
    sMbar_ptr = sMbar.iterator

    # mbarrier init (single thread per CTA)
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(sMbar_ptr, cutlass.Int32(1))
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    # Tile the TMA coordinate tensor: (CHUNK, D, n_tiles, 1, H).
    gSrc_q = cute.local_tile(tma_tensor_q, (CHUNK, D), (None, None, None))
    tQs, tQg = cpasync.tma_partition(
        tma_atom_q,
        0,
        cute.make_layout(1),
        cute.group_modes(q_smem, 0, 2),
        cute.group_modes(gSrc_q, 0, 2),
    )

    # Issue TMA load (single thread).
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(sMbar_ptr, cutlass.Int32(CHUNK * D * 2))
        cute.copy(
            tma_atom_q,
            tQg[(None, tile_idx, 0, head_idx)],
            tQs[(None,)],
            tma_bar_ptr=sMbar_ptr,
        )

    cute.arch.mbarrier_wait(sMbar_ptr, cutlass.Int32(0))

    # ========================================================================
    # Dump q_smem to workspace KD slot for validation.
    # workspace layout: [tile, head, kd|qd|kr|gt|inv|mqk] flattened to bf16.
    # We'll write CHUNK*D bf16 elements into the kd slot (offset 0 within tile).
    # workspace tensor is exposed as bf16 with total count
    # = total_tiles * H * CHUNK * D (so kd slot only).
    # Actually we model the workspace as bf16[total_tiles, H, CHUNK, D] where
    # only the "kd" portion is written here.
    # ========================================================================
    # Per-tile base index in the workspace (in bf16 elements).
    ws_base = (head_idx * total_tiles + tile_idx) * (CHUNK * D)

    # Each thread writes 8 bf16 = 1 row × 8 cols of the q_smem tile.
    # 256 threads × 8 elem = 2048 = CHUNK * D. Layout: thread i owns
    # row = i // 16, col = (i % 16) * 8.
    row = tidx // 16
    col = (tidx % 16) * 8
    for j in cutlass.range_constexpr(8):
        # Hold the static index in a constexpr local so the codegen sees a
        # compile-time int rather than a Python loop variable.
        jj: cutlass.Constexpr[int] = j
        workspace[ws_base + row * D + col + jj] = q_smem[row, col + jj]


# ---------------------------------------------------------------------------
# JIT launcher
# ---------------------------------------------------------------------------
@cute.jit
def run_k1_phase1(
    q_tensor: cute.Tensor,  # bf16 [B*T, H, D] flattened (contig)
    workspace: cute.Tensor,  # bf16 view, count = total_tiles*H*CHUNK*D
    H: cutlass.Constexpr[int],
    total_tiles: cutlass.Constexpr[int],
    T_total: cutlass.Constexpr[int],
    stream: cuda_drv.CUstream,
):
    # Reorder q logical shape from (T_total, H, D) to (T_total, D, H) so that
    # the TMA tile dims (T_total, D) come first; H becomes the outer mode.
    # Memory strides remain unchanged.
    q_view = cute.make_tensor(
        q_tensor.iterator,
        cute.make_layout((T_total, D, H), stride=(H * D, 1, D)),
    )
    smem_layout_q = cute.make_layout((CHUNK, D), stride=(D, 1))
    tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        q_view,
        smem_layout_q,
        (CHUNK, D),
    )

    # SMEM bytes: q (CHUNK*D*2) + mbarrier (8) + alignment.
    smem_bytes = CHUNK * D * 2 + 8 + 128

    k1_phase1_tma_load_kernel(
        tma_atom_q,
        tma_tensor_q,
        workspace,
        H,
        total_tiles,
        T_total,
    ).launch(
        grid=(total_tiles, H, 1),
        block=[THREADS_PER_CTA, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Python entry points
# ---------------------------------------------------------------------------
@dataclass
class K1Outputs:
    """Workspace dump as torch tensors for validation.

    Each field is a per-(tile, head) tensor reshape of the corresponding
    workspace slot. Currently only ``q_loaded`` (kd slot) is populated by
    phase 1; the rest are placeholders for future phases.
    """

    q_loaded: torch.Tensor  # [total_tiles, H, CHUNK, D] bf16 — TMA-loaded q


_compiled_cache: dict = {}


def launch_k1_phase1(
    q: torch.Tensor,  # [B, T, H, D] bf16
    workspace_bf16: torch.Tensor,  # bf16 [total_tiles*H*CHUNK*D] (or more)
) -> None:
    """Compile (cached) and launch K1 phase 1 (TMA load + workspace dump).

    The workspace must be at least ``total_tiles * H * CHUNK * D`` bf16
    elements (i.e. only the kd slot per tile). Larger workspaces are fine —
    only the prefix is touched.
    """
    assert q.dtype == torch.bfloat16 and q.is_cuda and q.is_contiguous()
    assert workspace_bf16.dtype == torch.bfloat16 and workspace_bf16.is_cuda
    B, T, H, K = q.shape
    assert K == D, f"K must be {D}, got {K}"
    assert T % CHUNK == 0, f"T must be a multiple of CHUNK={CHUNK}, got {T}"
    total_tiles = (B * T) // CHUNK
    T_total = B * T
    assert workspace_bf16.numel() >= total_tiles * H * CHUNK * D, (
        f"workspace too small: {workspace_bf16.numel()} < {total_tiles * H * CHUNK * D}"
    )

    # Cache key includes shapes for correct re-compile when shapes change.
    key = (T_total, H, total_tiles)
    if key not in _compiled_cache:
        stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
        # q is shape [B, T, H, D]; flatten B*T into a single leading dim.
        q_flat = q.view(T_total, H, D)
        q_cute = from_dlpack(q_flat.detach(), assumed_align=16)
        ws_cute = from_dlpack(workspace_bf16.detach(), assumed_align=16)
        compiled = cute.compile(
            run_k1_phase1,
            q_cute,
            ws_cute,
            H=H,
            total_tiles=total_tiles,
            T_total=T_total,
            stream=stream,
        )
        _compiled_cache[key] = compiled

    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    q_flat = q.view(T_total, H, D)
    _compiled_cache[key](q_flat, workspace_bf16, H, total_tiles, T_total, stream)


def launch_k1_workspace_only(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    workspace: torch.Tensor,  # uint8 buffer >= total_tiles*H*WORKSPACE_BYTES_PER_TILE
    problem,
) -> None:
    """End-to-end K1 launcher (currently only phase 1 implemented).

    Stub for the eventual full K1. View the uint8 workspace as bf16 and
    invoke the phase-1 kernel which only fills the kd slot of each tile.
    """
    # View workspace as bf16 over the kd slots (which are at offset 0 of each
    # per-tile block of WORKSPACE_BYTES_PER_TILE bytes). We allocate a
    # bf16-aligned view that is large enough to hold the full workspace.
    ws_bf16 = workspace.view(torch.bfloat16)
    launch_k1_phase1(q, ws_bf16)


# ===========================================================================
# Phases 1-5 kernel
# ---------------------------------------------------------------------------
# Adds: TMA load k and g_bf16, scalar A_log + dt_bias direct loads,
# L2 normalization of q and k (in-place SMEM), fused gate cumsum producing
# a per-row decay accumulator g[r,c] and a per-column g_total[c], then the
# exp_g_total step that overwrites g_total with exp(g_total).
#
# Validation outputs (one workspace per dump for clarity):
#   ws_q_l2 : bf16 [tot*H*CHUNK*D]   q after L2
#   ws_k_l2 : bf16 [tot*H*CHUNK*D]   k after L2
#   ws_gt   : fp32 [tot*H*D]         exp(cumsum_full) i.e. final exp_g_total
# ===========================================================================
@cute.kernel
def k1_phases_1to5_kernel(
    tma_atom_q: cute.CopyAtom,
    tma_tensor_q: cute.Tensor,
    tma_atom_k: cute.CopyAtom,
    tma_tensor_k: cute.Tensor,
    tma_atom_g: cute.CopyAtom,
    tma_tensor_g: cute.Tensor,
    a_log: cute.Tensor,  # [H] fp32 GMEM
    dt_bias: cute.Tensor,  # [H, D] fp32 GMEM
    ws_q_l2: cute.Tensor,  # bf16 flat
    ws_k_l2: cute.Tensor,  # bf16 flat
    ws_gt: cute.Tensor,  # fp32 flat
    H: cutlass.Constexpr[int],
    total_tiles: cutlass.Constexpr[int],
    T_total: cutlass.Constexpr[int],
    gate_scale: cutlass.Constexpr[float],
):
    tile_idx, head_idx, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    smem = cutlass.utils.SmemAllocator()
    qk_layout = cute.make_layout((CHUNK, D), stride=(D, 1))
    sQ = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sK = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sGbf = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sGtot = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D,)), 128)
    sMbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((1,)), 8)
    sMbar_ptr = sMbar.iterator

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(sMbar_ptr, cutlass.Int32(1))
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    # TMA partition (one descriptor per tensor; same shape).
    gSrc_q = cute.local_tile(tma_tensor_q, (CHUNK, D), (None, None, None))
    gSrc_k = cute.local_tile(tma_tensor_k, (CHUNK, D), (None, None, None))
    gSrc_g = cute.local_tile(tma_tensor_g, (CHUNK, D), (None, None, None))
    tQs, tQg = cpasync.tma_partition(
        tma_atom_q,
        0,
        cute.make_layout(1),
        cute.group_modes(sQ, 0, 2),
        cute.group_modes(gSrc_q, 0, 2),
    )
    tKs, tKg = cpasync.tma_partition(
        tma_atom_k,
        0,
        cute.make_layout(1),
        cute.group_modes(sK, 0, 2),
        cute.group_modes(gSrc_k, 0, 2),
    )
    tGs, tGg = cpasync.tma_partition(
        tma_atom_g,
        0,
        cute.make_layout(1),
        cute.group_modes(sGbf, 0, 2),
        cute.group_modes(gSrc_g, 0, 2),
    )

    # Issue all 3 TMA loads under the same mbar.
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(sMbar_ptr, cutlass.Int32(3 * CHUNK * D * 2))
        cute.copy(tma_atom_q, tQg[(None, tile_idx, 0, head_idx)], tQs[(None,)], tma_bar_ptr=sMbar_ptr)
        cute.copy(tma_atom_k, tKg[(None, tile_idx, 0, head_idx)], tKs[(None,)], tma_bar_ptr=sMbar_ptr)
        cute.copy(tma_atom_g, tGg[(None, tile_idx, 0, head_idx)], tGs[(None,)], tma_bar_ptr=sMbar_ptr)

    cute.arch.mbarrier_wait(sMbar_ptr, cutlass.Int32(0))

    # ------------------------------------------------------------------
    # Phase 3: L2 normalize q and k
    # 256 threads × 8 elements = 2048 = CHUNK*D. THREADS_PER_ROW = 16,
    # so warp_reduction with threads_in_group=16 yields the row sum for
    # the half-warp owning a row.
    # ------------------------------------------------------------------
    row = tidx // 16
    col = (tidx % 16) * 8
    base = row * D + col

    q_sq = cutlass.Float32(0.0)
    k_sq = cutlass.Float32(0.0)
    q_vals = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), cutlass.Float32)
    k_vals = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), cutlass.Float32)
    for j in cutlass.range_constexpr(8):
        qv = cutlass.Float32(sQ[row, col + j])
        kv = cutlass.Float32(sK[row, col + j])
        q_vals[j] = qv
        k_vals[j] = kv
        q_sq = q_sq + qv * qv
        k_sq = k_sq + kv * kv

    q_sq = cute.arch.warp_reduction(q_sq, lambda a, b: a + b, threads_in_group=16)
    k_sq = cute.arch.warp_reduction(k_sq, lambda a, b: a + b, threads_in_group=16)

    q_inv = cute.rsqrt(q_sq + cutlass.Float32(1.0e-6), fastmath=True)
    k_inv = cute.rsqrt(k_sq + cutlass.Float32(1.0e-6), fastmath=True)

    for j in cutlass.range_constexpr(8):
        sQ[row, col + j] = cutlass.BFloat16(q_vals[j] * q_inv)
        sK[row, col + j] = cutlass.BFloat16(k_vals[j] * k_inv)
    cute.arch.barrier()

    # ------------------------------------------------------------------
    # Phase 4 + 5: gate cumsum (one column per thread, only tidx<128)
    #              + exp_g_total stored in sGtot.
    # The C++ writes the per-row cumulative sums to a CHUNK*D smem `g` for
    # later decay_apply (phase 6); we don't need it yet so we skip it.
    # ------------------------------------------------------------------
    a_log_exp = cute.exp(cutlass.Float32(a_log[head_idx]), fastmath=True)
    if tidx < 128:
        col_c = tidx  # 0..127
        dt = cutlass.Float32(dt_bias[head_idx, col_c])
        s = cutlass.Float32(0.0)
        for r in cutlass.range_constexpr(CHUNK):
            x = cutlass.Float32(sGbf[r, col_c]) + dt
            x = a_log_exp * x
            # sigmoid via tanh: 0.5 * (tanh(x/2) + 1)
            sig = cutlass.Float32(0.5) * (cute.tanh(x * cutlass.Float32(0.5), fastmath=True) + cutlass.Float32(1.0))
            s = s + cutlass.Float32(gate_scale) * sig
        # exp_g_total
        sGtot[col_c] = cute.exp(s, fastmath=True)
    cute.arch.barrier()

    # ------------------------------------------------------------------
    # Workspace dumps (validation-only; not the final phase-9 layout).
    # 256 threads × 8 elem = 2048 = CHUNK*D.
    # ------------------------------------------------------------------
    ws_base = (head_idx * total_tiles + tile_idx) * (CHUNK * D)
    for j in cutlass.range_constexpr(8):
        jj: cutlass.Constexpr[int] = j
        ws_q_l2[ws_base + base + jj] = sQ[row, col + jj]
        ws_k_l2[ws_base + base + jj] = sK[row, col + jj]

    # exp_g_total: D=128 fp32 values per (head,tile). 128 threads write 1 each.
    if tidx < 128:
        gt_base = (head_idx * total_tiles + tile_idx) * D
        ws_gt[gt_base + tidx] = sGtot[tidx]


@cute.jit
def run_k1_phases_1to5(
    q: cute.Tensor,
    k: cute.Tensor,
    g: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    ws_q_l2: cute.Tensor,
    ws_k_l2: cute.Tensor,
    ws_gt: cute.Tensor,
    H: cutlass.Constexpr[int],
    total_tiles: cutlass.Constexpr[int],
    T_total: cutlass.Constexpr[int],
    gate_scale: cutlass.Constexpr[float],
    stream: cuda_drv.CUstream,
):
    smem_layout_qk = cute.make_layout((CHUNK, D), stride=(D, 1))

    def make_qkg_atom(t):
        view = cute.make_tensor(
            t.iterator,
            cute.make_layout((T_total, D, H), stride=(H * D, 1, D)),
        )
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            view,
            smem_layout_qk,
            (CHUNK, D),
        )

    tma_atom_q, tma_tensor_q = make_qkg_atom(q)
    tma_atom_k, tma_tensor_k = make_qkg_atom(k)
    tma_atom_g, tma_tensor_g = make_qkg_atom(g)

    # SMEM bytes: 3 × bf16 tile + 1 × fp32 D vec + 1 mbar + alignment headroom
    smem_bytes = 3 * (CHUNK * D * 2) + (D * 4) + 8 + 256

    k1_phases_1to5_kernel(
        tma_atom_q,
        tma_tensor_q,
        tma_atom_k,
        tma_tensor_k,
        tma_atom_g,
        tma_tensor_g,
        a_log,
        dt_bias,
        ws_q_l2,
        ws_k_l2,
        ws_gt,
        H,
        total_tiles,
        T_total,
        gate_scale,
    ).launch(
        grid=(total_tiles, H, 1),
        block=[THREADS_PER_CTA, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


_compiled_cache_phases5: dict = {}


def launch_k1_phases_1to5(
    q: torch.Tensor,  # [B, T, H, D] bf16
    k: torch.Tensor,
    g: torch.Tensor,
    A_log: torch.Tensor,  # [H] fp32
    dt_bias: torch.Tensor,  # [H, D] fp32
    gate_scale: float,
    ws_q_l2: torch.Tensor,  # bf16 [tot*H*CHUNK*D]
    ws_k_l2: torch.Tensor,  # bf16 [tot*H*CHUNK*D]
    ws_gt: torch.Tensor,  # fp32 [tot*H*D]
) -> None:
    for t in (q, k, g):
        assert t.dtype == torch.bfloat16 and t.is_cuda and t.is_contiguous()
    assert A_log.dtype == torch.float32 and A_log.is_contiguous()
    assert dt_bias.dtype == torch.float32 and dt_bias.is_contiguous()
    B, T, H, K = q.shape
    assert K == D
    assert T % CHUNK == 0
    total_tiles = (B * T) // CHUNK
    T_total = B * T

    key = (T_total, H, total_tiles, gate_scale)
    if key not in _compiled_cache_phases5:
        stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
        q_flat = q.view(T_total, H, D)
        k_flat = k.view(T_total, H, D)
        g_flat = g.view(T_total, H, D)
        _compiled_cache_phases5[key] = cute.compile(
            run_k1_phases_1to5,
            from_dlpack(q_flat.detach(), assumed_align=16),
            from_dlpack(k_flat.detach(), assumed_align=16),
            from_dlpack(g_flat.detach(), assumed_align=16),
            from_dlpack(A_log.detach(), assumed_align=16),
            from_dlpack(dt_bias.detach(), assumed_align=16),
            from_dlpack(ws_q_l2.detach(), assumed_align=16),
            from_dlpack(ws_k_l2.detach(), assumed_align=16),
            from_dlpack(ws_gt.detach(), assumed_align=16),
            H=H,
            total_tiles=total_tiles,
            T_total=T_total,
            gate_scale=gate_scale,
            stream=stream,
        )

    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    q_flat = q.view(T_total, H, D)
    k_flat = k.view(T_total, H, D)
    g_flat = g.view(T_total, H, D)
    _compiled_cache_phases5[key](
        q_flat,
        k_flat,
        g_flat,
        A_log,
        dt_bias,
        ws_q_l2,
        ws_k_l2,
        ws_gt,
        H,
        total_tiles,
        T_total,
        gate_scale,
        stream,
    )


__all__ = [
    "CHUNK",
    "D",
    "WORKSPACE_BYTES_PER_TILE",
    "K1Outputs",
    "launch_k1_phase1",
    "launch_k1_phases_1to5",
    "launch_k1_phases_1to6",
    "launch_k1_workspace_only",
]


# ===========================================================================
# Phases 1-6 kernel
# ---------------------------------------------------------------------------
# Adds Phase 6 (decay_apply) on top of phases 1-5:
#   q_decayed[r,c] = q_l2[r,c] * scale * exp(g_cumsum[r,c])
#   k_decayed[r,c] = k_l2[r,c] *         exp(g_cumsum[r,c])
#   k_inv    [r,c] = k_l2[r,c] *         exp(-g_cumsum[r,c])
#   k_restored[r,c]= k_inv[r,c] * exp_g_total[c]   (= k_l2 * exp(g_total[c] - g_cumsum[r,c]))
# All four outputs are bf16 [CHUNK, D] row-major.
# To compute these we need the per-row cumsum kept in SMEM (sG_cs fp32
# [CHUNK, D]).
# ===========================================================================
@cute.kernel
def k1_phases_1to6_kernel(
    tma_atom_q: cute.CopyAtom,
    tma_tensor_q: cute.Tensor,
    tma_atom_k: cute.CopyAtom,
    tma_tensor_k: cute.Tensor,
    tma_atom_g: cute.CopyAtom,
    tma_tensor_g: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    ws_qd: cute.Tensor,  # bf16 flat
    ws_kd: cute.Tensor,
    ws_ki: cute.Tensor,
    ws_kr: cute.Tensor,
    ws_gt: cute.Tensor,  # fp32 flat
    H: cutlass.Constexpr[int],
    total_tiles: cutlass.Constexpr[int],
    T_total: cutlass.Constexpr[int],
    scale: cutlass.Constexpr[float],
    gate_scale: cutlass.Constexpr[float],
):
    tile_idx, head_idx, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    smem = cutlass.utils.SmemAllocator()
    qk_layout = cute.make_layout((CHUNK, D), stride=(D, 1))
    sQ = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sK = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sGbf = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sGcs = smem.allocate_tensor(cutlass.Float32, qk_layout, 128)  # cumsum fp32
    sGtot = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D,)), 128)
    sMbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((1,)), 8)
    sMbar_ptr = sMbar.iterator

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(sMbar_ptr, cutlass.Int32(1))
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    gSrc_q = cute.local_tile(tma_tensor_q, (CHUNK, D), (None, None, None))
    gSrc_k = cute.local_tile(tma_tensor_k, (CHUNK, D), (None, None, None))
    gSrc_g = cute.local_tile(tma_tensor_g, (CHUNK, D), (None, None, None))
    tQs, tQg = cpasync.tma_partition(
        tma_atom_q,
        0,
        cute.make_layout(1),
        cute.group_modes(sQ, 0, 2),
        cute.group_modes(gSrc_q, 0, 2),
    )
    tKs, tKg = cpasync.tma_partition(
        tma_atom_k,
        0,
        cute.make_layout(1),
        cute.group_modes(sK, 0, 2),
        cute.group_modes(gSrc_k, 0, 2),
    )
    tGs, tGg = cpasync.tma_partition(
        tma_atom_g,
        0,
        cute.make_layout(1),
        cute.group_modes(sGbf, 0, 2),
        cute.group_modes(gSrc_g, 0, 2),
    )

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(sMbar_ptr, cutlass.Int32(3 * CHUNK * D * 2))
        cute.copy(tma_atom_q, tQg[(None, tile_idx, 0, head_idx)], tQs[(None,)], tma_bar_ptr=sMbar_ptr)
        cute.copy(tma_atom_k, tKg[(None, tile_idx, 0, head_idx)], tKs[(None,)], tma_bar_ptr=sMbar_ptr)
        cute.copy(tma_atom_g, tGg[(None, tile_idx, 0, head_idx)], tGs[(None,)], tma_bar_ptr=sMbar_ptr)

    cute.arch.mbarrier_wait(sMbar_ptr, cutlass.Int32(0))

    # -------- L2 normalize q, k --------
    row = tidx // 16
    col = (tidx % 16) * 8

    q_sq = cutlass.Float32(0.0)
    k_sq = cutlass.Float32(0.0)
    q_vals = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), cutlass.Float32)
    k_vals = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), cutlass.Float32)
    for j in cutlass.range_constexpr(8):
        qv = cutlass.Float32(sQ[row, col + j])
        kv = cutlass.Float32(sK[row, col + j])
        q_vals[j] = qv
        k_vals[j] = kv
        q_sq = q_sq + qv * qv
        k_sq = k_sq + kv * kv
    q_sq = cute.arch.warp_reduction(q_sq, lambda a, b: a + b, threads_in_group=16)
    k_sq = cute.arch.warp_reduction(k_sq, lambda a, b: a + b, threads_in_group=16)
    q_inv = cute.rsqrt(q_sq + cutlass.Float32(1.0e-6), fastmath=True)
    k_inv = cute.rsqrt(k_sq + cutlass.Float32(1.0e-6), fastmath=True)
    for j in cutlass.range_constexpr(8):
        sQ[row, col + j] = cutlass.BFloat16(q_vals[j] * q_inv)
        sK[row, col + j] = cutlass.BFloat16(k_vals[j] * k_inv)
    cute.arch.barrier()

    # -------- Gate cumsum + g_total + exp_g_total --------
    a_log_exp = cute.exp(cutlass.Float32(a_log[head_idx]), fastmath=True)
    if tidx < 128:
        col_c = tidx
        dt = cutlass.Float32(dt_bias[head_idx, col_c])
        s = cutlass.Float32(0.0)
        for r in cutlass.range_constexpr(CHUNK):
            x = cutlass.Float32(sGbf[r, col_c]) + dt
            x = a_log_exp * x
            sig = cutlass.Float32(0.5) * (cute.tanh(x * cutlass.Float32(0.5), fastmath=True) + cutlass.Float32(1.0))
            s = s + cutlass.Float32(gate_scale) * sig
            sGcs[r, col_c] = s  # per-row cumsum (fp32) — needed by phase 6
        sGtot[col_c] = cute.exp(s, fastmath=True)
    cute.arch.barrier()

    # -------- Phase 6: decay_apply --------
    # Reuse the (row, col) decomposition: 256 threads × 8 elems = 2048 = CHUNK*D.
    # For each thread, compute the 4 bf16 outputs and store back into 4 SMEM
    # (or directly to workspace; we go direct since the phase-7 GEMM is not
    # in this kernel).
    ws_base = (head_idx * total_tiles + tile_idx) * (CHUNK * D)
    base = row * D + col
    for j in cutlass.range_constexpr(8):
        jj: cutlass.Constexpr[int] = j
        g_cs = sGcs[row, col + jj]
        gt = sGtot[col + jj]
        exp_pos = cute.exp(g_cs, fastmath=True)
        # exp(-g_cs) = 1 / exp_pos: use reciprocal for one fewer transcendental.
        inv_pos = cutlass.Float32(1.0) / exp_pos
        # exp(g_total - g_cs) = gt / exp_pos = gt * inv_pos.
        rest = gt * inv_pos
        q_v = cutlass.Float32(sQ[row, col + jj])
        k_v = cutlass.Float32(sK[row, col + jj])
        ws_qd[ws_base + base + jj] = cutlass.BFloat16(q_v * exp_pos * cutlass.Float32(scale))
        ws_kd[ws_base + base + jj] = cutlass.BFloat16(k_v * exp_pos)
        ws_ki[ws_base + base + jj] = cutlass.BFloat16(k_v * inv_pos)
        ws_kr[ws_base + base + jj] = cutlass.BFloat16(k_v * rest)

    if tidx < 128:
        gt_base = (head_idx * total_tiles + tile_idx) * D
        ws_gt[gt_base + tidx] = sGtot[tidx]


@cute.jit
def run_k1_phases_1to6(
    q: cute.Tensor,
    k: cute.Tensor,
    g: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    ws_qd: cute.Tensor,
    ws_kd: cute.Tensor,
    ws_ki: cute.Tensor,
    ws_kr: cute.Tensor,
    ws_gt: cute.Tensor,
    H: cutlass.Constexpr[int],
    total_tiles: cutlass.Constexpr[int],
    T_total: cutlass.Constexpr[int],
    scale: cutlass.Constexpr[float],
    gate_scale: cutlass.Constexpr[float],
    stream: cuda_drv.CUstream,
):
    smem_layout_qk = cute.make_layout((CHUNK, D), stride=(D, 1))

    def make_atom(t):
        view = cute.make_tensor(
            t.iterator,
            cute.make_layout((T_total, D, H), stride=(H * D, 1, D)),
        )
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            view,
            smem_layout_qk,
            (CHUNK, D),
        )

    tma_atom_q, tma_tensor_q = make_atom(q)
    tma_atom_k, tma_tensor_k = make_atom(k)
    tma_atom_g, tma_tensor_g = make_atom(g)

    # SMEM: 3*bf16(CHUNK*D) + fp32(CHUNK*D) + fp32(D) + mbar + alignment
    smem_bytes = 3 * (CHUNK * D * 2) + (CHUNK * D * 4) + (D * 4) + 8 + 256

    k1_phases_1to6_kernel(
        tma_atom_q,
        tma_tensor_q,
        tma_atom_k,
        tma_tensor_k,
        tma_atom_g,
        tma_tensor_g,
        a_log,
        dt_bias,
        ws_qd,
        ws_kd,
        ws_ki,
        ws_kr,
        ws_gt,
        H,
        total_tiles,
        T_total,
        scale,
        gate_scale,
    ).launch(
        grid=(total_tiles, H, 1),
        block=[THREADS_PER_CTA, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


_compiled_cache_phases6: dict = {}


def launch_k1_phases_1to6(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    gate_scale: float,
    ws_qd: torch.Tensor,
    ws_kd: torch.Tensor,
    ws_ki: torch.Tensor,
    ws_kr: torch.Tensor,
    ws_gt: torch.Tensor,
) -> None:
    for t in (q, k, g):
        assert t.dtype == torch.bfloat16 and t.is_cuda and t.is_contiguous()
    assert A_log.dtype == torch.float32 and A_log.is_contiguous()
    assert dt_bias.dtype == torch.float32 and dt_bias.is_contiguous()
    B, T, H, K = q.shape
    assert K == D and T % CHUNK == 0
    total_tiles = (B * T) // CHUNK
    T_total = B * T

    key = (T_total, H, total_tiles, scale, gate_scale)
    if key not in _compiled_cache_phases6:
        stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
        q_flat = q.view(T_total, H, D)
        k_flat = k.view(T_total, H, D)
        g_flat = g.view(T_total, H, D)
        _compiled_cache_phases6[key] = cute.compile(
            run_k1_phases_1to6,
            from_dlpack(q_flat.detach(), assumed_align=16),
            from_dlpack(k_flat.detach(), assumed_align=16),
            from_dlpack(g_flat.detach(), assumed_align=16),
            from_dlpack(A_log.detach(), assumed_align=16),
            from_dlpack(dt_bias.detach(), assumed_align=16),
            from_dlpack(ws_qd.detach(), assumed_align=16),
            from_dlpack(ws_kd.detach(), assumed_align=16),
            from_dlpack(ws_ki.detach(), assumed_align=16),
            from_dlpack(ws_kr.detach(), assumed_align=16),
            from_dlpack(ws_gt.detach(), assumed_align=16),
            H=H,
            total_tiles=total_tiles,
            T_total=T_total,
            scale=scale,
            gate_scale=gate_scale,
            stream=stream,
        )

    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    q_flat = q.view(T_total, H, D)
    k_flat = k.view(T_total, H, D)
    g_flat = g.view(T_total, H, D)
    _compiled_cache_phases6[key](
        q_flat,
        k_flat,
        g_flat,
        A_log,
        dt_bias,
        ws_qd,
        ws_kd,
        ws_ki,
        ws_kr,
        ws_gt,
        H,
        total_tiles,
        T_total,
        scale,
        gate_scale,
        stream,
    )


# ===========================================================================
# Phases 1-8 kernel ??? adds L_Mqk + tril/beta + Neumann inverse
# ---------------------------------------------------------------------------
# Phase 7:
#   L  [16,16] = k_decayed @ k_inv^T   (fp32 acc, then masked)
#   Mqk[16,16] = q_decayed @ k_inv^T   (fp32 acc, then masked)
# Mask: L is strict lower-tri; for i>j, L[i,j] *= sigmoid(beta[bos+i]).
# Mqk is upper-incl-diagonal-zero: Mqk[i,j]=0 for i<j.
# Phase 8:
#   INV = (I + L_lower)^(-1) via Neumann series exact for strictly lower 16x16:
#   INV_init = I - L  (note sign flip because (I+L)^(-1) = (I-L)*(I+L^2)*(I+L^4)*(I+L^8))
#   then INV ??= (I + L^2k) for k=1,2,3.
# Implementation uses CUDA-core compute with 256 threads ?? 1 element each
# (16x16 = 256 outputs). All matmuls staged via SMEM. Tensor cores will be a
# follow-up perf pass ??? for these tiny 16x16x{16,128} GEMMs the bottleneck is
# the recurrent K2 kernel anyway.
# ===========================================================================
@cute.kernel
def k1_phases_1to8_kernel(
    tma_atom_q: cute.CopyAtom,
    tma_tensor_q: cute.Tensor,
    tma_atom_k: cute.CopyAtom,
    tma_tensor_k: cute.Tensor,
    tma_atom_g: cute.CopyAtom,
    tma_tensor_g: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,  # [B*T*H] bf16 flat (linear: head*T + t)
    ws_l: cute.Tensor,  # bf16 flat [tot*H*16*16]
    ws_mqk: cute.Tensor,  # bf16 flat [tot*H*16*16]
    ws_inv: cute.Tensor,  # bf16 flat [tot*H*16*16]
    H: cutlass.Constexpr[int],
    total_tiles: cutlass.Constexpr[int],
    T_total: cutlass.Constexpr[int],
    scale: cutlass.Constexpr[float],
    gate_scale: cutlass.Constexpr[float],
):
    tile_idx, head_idx, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    smem = cutlass.utils.SmemAllocator()
    qk_layout = cute.make_layout((CHUNK, D), stride=(D, 1))
    cc_layout = cute.make_layout((CHUNK, CHUNK), stride=(CHUNK, 1))  # 16x16
    sQ = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sK = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sGbf = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sGcs = smem.allocate_tensor(cutlass.Float32, qk_layout, 128)
    sGtot = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D,)), 128)
    # Phase 6 output tensors (kept in SMEM, then dumped at phase 7+8 end).
    sQD = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sKD = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sKI = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    # Phase 7+8 output tensors (16x16 bf16 / fp32 work).
    sL = smem.allocate_tensor(cutlass.Float32, cc_layout, 128)  # fp32 for precision
    sMqk = smem.allocate_tensor(cutlass.Float32, cc_layout, 128)
    sINV = smem.allocate_tensor(cutlass.Float32, cc_layout, 128)
    sLp = smem.allocate_tensor(cutlass.Float32, cc_layout, 128)  # L^2, L^4, L^8 buffer
    sTmp = smem.allocate_tensor(cutlass.Float32, cc_layout, 128)  # scratch
    sMbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((1,)), 8)
    sMbar_ptr = sMbar.iterator

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(sMbar_ptr, cutlass.Int32(1))
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    gSrc_q = cute.local_tile(tma_tensor_q, (CHUNK, D), (None, None, None))
    gSrc_k = cute.local_tile(tma_tensor_k, (CHUNK, D), (None, None, None))
    gSrc_g = cute.local_tile(tma_tensor_g, (CHUNK, D), (None, None, None))
    tQs, tQg = cpasync.tma_partition(
        tma_atom_q,
        0,
        cute.make_layout(1),
        cute.group_modes(sQ, 0, 2),
        cute.group_modes(gSrc_q, 0, 2),
    )
    tKs, tKg = cpasync.tma_partition(
        tma_atom_k,
        0,
        cute.make_layout(1),
        cute.group_modes(sK, 0, 2),
        cute.group_modes(gSrc_k, 0, 2),
    )
    tGs, tGg = cpasync.tma_partition(
        tma_atom_g,
        0,
        cute.make_layout(1),
        cute.group_modes(sGbf, 0, 2),
        cute.group_modes(gSrc_g, 0, 2),
    )

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(sMbar_ptr, cutlass.Int32(3 * CHUNK * D * 2))
        cute.copy(tma_atom_q, tQg[(None, tile_idx, 0, head_idx)], tQs[(None,)], tma_bar_ptr=sMbar_ptr)
        cute.copy(tma_atom_k, tKg[(None, tile_idx, 0, head_idx)], tKs[(None,)], tma_bar_ptr=sMbar_ptr)
        cute.copy(tma_atom_g, tGg[(None, tile_idx, 0, head_idx)], tGs[(None,)], tma_bar_ptr=sMbar_ptr)

    cute.arch.mbarrier_wait(sMbar_ptr, cutlass.Int32(0))

    # -------- L2 normalize q, k --------
    row = tidx // 16
    col = (tidx % 16) * 8

    q_sq = cutlass.Float32(0.0)
    k_sq = cutlass.Float32(0.0)
    q_vals = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), cutlass.Float32)
    k_vals = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), cutlass.Float32)
    for j in cutlass.range_constexpr(8):
        qv = cutlass.Float32(sQ[row, col + j])
        kv = cutlass.Float32(sK[row, col + j])
        q_vals[j] = qv
        k_vals[j] = kv
        q_sq = q_sq + qv * qv
        k_sq = k_sq + kv * kv
    q_sq = cute.arch.warp_reduction(q_sq, lambda a, b: a + b, threads_in_group=16)
    k_sq = cute.arch.warp_reduction(k_sq, lambda a, b: a + b, threads_in_group=16)
    q_inv = cute.rsqrt(q_sq + cutlass.Float32(1.0e-6), fastmath=True)
    k_inv = cute.rsqrt(k_sq + cutlass.Float32(1.0e-6), fastmath=True)
    for j in cutlass.range_constexpr(8):
        sQ[row, col + j] = cutlass.BFloat16(q_vals[j] * q_inv)
        sK[row, col + j] = cutlass.BFloat16(k_vals[j] * k_inv)
    cute.arch.barrier()

    # -------- Gate cumsum + g_total + exp_g_total --------
    a_log_exp = cute.exp(cutlass.Float32(a_log[head_idx]), fastmath=True)
    if tidx < 128:
        col_c = tidx
        dt = cutlass.Float32(dt_bias[head_idx, col_c])
        s = cutlass.Float32(0.0)
        for r in cutlass.range_constexpr(CHUNK):
            x = cutlass.Float32(sGbf[r, col_c]) + dt
            x = a_log_exp * x
            sig = cutlass.Float32(0.5) * (cute.tanh(x * cutlass.Float32(0.5), fastmath=True) + cutlass.Float32(1.0))
            s = s + cutlass.Float32(gate_scale) * sig
            sGcs[r, col_c] = s
        sGtot[col_c] = cute.exp(s, fastmath=True)
    cute.arch.barrier()

    # -------- Phase 6: decay_apply ??? SMEM --------
    for j in cutlass.range_constexpr(8):
        jj: cutlass.Constexpr[int] = j
        g_cs = sGcs[row, col + jj]
        exp_pos = cute.exp(g_cs, fastmath=True)
        inv_pos = cutlass.Float32(1.0) / exp_pos
        q_v = cutlass.Float32(sQ[row, col + jj])
        k_v = cutlass.Float32(sK[row, col + jj])
        sQD[row, col + jj] = cutlass.BFloat16(q_v * exp_pos * cutlass.Float32(scale))
        sKD[row, col + jj] = cutlass.BFloat16(k_v * exp_pos)
        sKI[row, col + jj] = cutlass.BFloat16(k_v * inv_pos)
    cute.arch.barrier()

    # -------- Phase 7: L = sKD @ sKI^T, Mqk = sQD @ sKI^T --------
    # 256 threads ??? one (i,j) of the 16x16 outputs each.
    i = tidx // CHUNK
    j = tidx % CHUNK
    sum_l = cutlass.Float32(0.0)
    sum_m = cutlass.Float32(0.0)
    for kk in cutlass.range(D, unroll=8):
        ki_v = cutlass.Float32(sKI[j, kk])  # second operand transposed: KI[j, kk]
        sum_l = sum_l + cutlass.Float32(sKD[i, kk]) * ki_v
        sum_m = sum_m + cutlass.Float32(sQD[i, kk]) * ki_v
    # Apply masks + beta sigmoid.
    # beta linear index: head_idx * T_total + tile_idx * CHUNK + i
    beta_lin = head_idx * T_total + tile_idx * CHUNK + i
    if i > j:
        bv = cutlass.Float32(beta[beta_lin])
        sig_b = cutlass.Float32(0.5) * (cute.tanh(bv * cutlass.Float32(0.5), fastmath=True) + cutlass.Float32(1.0))
        sL[i, j] = sum_l * sig_b
    else:
        sL[i, j] = cutlass.Float32(0.0)
    if i >= j:
        sMqk[i, j] = sum_m
    else:
        sMqk[i, j] = cutlass.Float32(0.0)
    cute.arch.barrier()

    # -------- Phase 8: Neumann inverse (I + L)^(-1) --------
    # Identity Init: INV = I - L. Then iterate 3 powers (L^2, L^4, L^8).
    # Layout: each (i,j) thread holds 1 element of each 16x16 matrix.
    inv_v = cutlass.Float32(1.0 if i == j else 0.0) - sL[i, j]
    sINV[i, j] = inv_v
    sLp[i, j] = sL[i, j]
    cute.arch.barrier()

    for _p in cutlass.range_constexpr(3):
        # 1) sTmp = sLp @ sLp  (compute L^2k ??? temp)
        s = cutlass.Float32(0.0)
        for kk in cutlass.range_constexpr(CHUNK):
            s = s + sLp[i, kk] * sLp[kk, j]
        # use sMqk as scratch ??? actually we need a separate scratch to avoid races.
        # We have sTmp.
        cute.arch.barrier()
        sTmp[i, j] = s
        cute.arch.barrier()

        # 2) sLp_new = sTmp; compute INV += INV @ sLp_new
        s2 = cutlass.Float32(0.0)
        for kk in cutlass.range_constexpr(CHUNK):
            s2 = s2 + sINV[i, kk] * sTmp[kk, j]
        cute.arch.barrier()
        sINV[i, j] = sINV[i, j] + s2
        sLp[i, j] = sTmp[i, j]  # promote scratch to current power
        cute.arch.barrier()

    # -------- Workspace dumps --------
    # Each (head, tile) has a 16x16 = 256-element output region.
    ws_base_cc = (head_idx * total_tiles + tile_idx) * (CHUNK * CHUNK)
    ws_l[ws_base_cc + i * CHUNK + j] = cutlass.BFloat16(sL[i, j])
    ws_mqk[ws_base_cc + i * CHUNK + j] = cutlass.BFloat16(sMqk[i, j])
    ws_inv[ws_base_cc + i * CHUNK + j] = cutlass.BFloat16(sINV[i, j])


@cute.jit
def run_k1_phases_1to8(
    q: cute.Tensor,
    k: cute.Tensor,
    g: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    ws_l: cute.Tensor,
    ws_mqk: cute.Tensor,
    ws_inv: cute.Tensor,
    H: cutlass.Constexpr[int],
    total_tiles: cutlass.Constexpr[int],
    T_total: cutlass.Constexpr[int],
    scale: cutlass.Constexpr[float],
    gate_scale: cutlass.Constexpr[float],
    stream: cuda_drv.CUstream,
):
    smem_layout_qk = cute.make_layout((CHUNK, D), stride=(D, 1))

    def make_atom(t):
        view = cute.make_tensor(
            t.iterator,
            cute.make_layout((T_total, D, H), stride=(H * D, 1, D)),
        )
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            view,
            smem_layout_qk,
            (CHUNK, D),
        )

    tma_atom_q, tma_tensor_q = make_atom(q)
    tma_atom_k, tma_tensor_k = make_atom(k)
    tma_atom_g, tma_tensor_g = make_atom(g)

    # SMEM bytes: 6 ?? bf16(CHUNK*D) [q,k,gbf,qd,kd,ki] + fp32(CHUNK*D) [g_cs] +
    # fp32(D) [g_tot] + 5 ?? fp32(CHUNK*CHUNK) [L,Mqk,INV,Lp,Tmp] + mbar + slack
    smem_bytes = 6 * (CHUNK * D * 2) + (CHUNK * D * 4) + (D * 4) + 5 * (CHUNK * CHUNK * 4) + 8 + 512

    k1_phases_1to8_kernel(
        tma_atom_q,
        tma_tensor_q,
        tma_atom_k,
        tma_tensor_k,
        tma_atom_g,
        tma_tensor_g,
        a_log,
        dt_bias,
        beta,
        ws_l,
        ws_mqk,
        ws_inv,
        H,
        total_tiles,
        T_total,
        scale,
        gate_scale,
    ).launch(
        grid=(total_tiles, H, 1),
        block=[THREADS_PER_CTA, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


_compiled_cache_phases8: dict = {}


def launch_k1_phases_1to8(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    gate_scale: float,
    ws_l: torch.Tensor,
    ws_mqk: torch.Tensor,
    ws_inv: torch.Tensor,
) -> None:
    for t in (q, k, g, beta):
        assert t.dtype == torch.bfloat16 and t.is_cuda and t.is_contiguous()
    assert A_log.dtype == torch.float32 and A_log.is_contiguous()
    assert dt_bias.dtype == torch.float32 and dt_bias.is_contiguous()
    B, T, H, K = q.shape
    assert K == D and T % CHUNK == 0
    total_tiles = (B * T) // CHUNK
    T_total = B * T
    # beta layout in C++: linear (head_idx * T_total + t).
    # Caller expectation: a [B*T*H] bf16 flat tensor in that order.
    assert beta.numel() == B * T * H

    key = (T_total, H, total_tiles, scale, gate_scale)
    if key not in _compiled_cache_phases8:
        stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
        q_flat = q.view(T_total, H, D)
        k_flat = k.view(T_total, H, D)
        g_flat = g.view(T_total, H, D)
        _compiled_cache_phases8[key] = cute.compile(
            run_k1_phases_1to8,
            from_dlpack(q_flat.detach(), assumed_align=16),
            from_dlpack(k_flat.detach(), assumed_align=16),
            from_dlpack(g_flat.detach(), assumed_align=16),
            from_dlpack(A_log.detach(), assumed_align=16),
            from_dlpack(dt_bias.detach(), assumed_align=16),
            from_dlpack(beta.detach(), assumed_align=16),
            from_dlpack(ws_l.detach(), assumed_align=16),
            from_dlpack(ws_mqk.detach(), assumed_align=16),
            from_dlpack(ws_inv.detach(), assumed_align=16),
            H=H,
            total_tiles=total_tiles,
            T_total=T_total,
            scale=scale,
            gate_scale=gate_scale,
            stream=stream,
        )

    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    q_flat = q.view(T_total, H, D)
    k_flat = k.view(T_total, H, D)
    g_flat = g.view(T_total, H, D)
    _compiled_cache_phases8[key](
        q_flat,
        k_flat,
        g_flat,
        A_log,
        dt_bias,
        beta,
        ws_l,
        ws_mqk,
        ws_inv,
        H,
        total_tiles,
        T_total,
        scale,
        gate_scale,
        stream,
    )


# ===========================================================================
# K1 FULL ??? phases 1-9: produces ALL 6 tensors needed by K2.
# ---------------------------------------------------------------------------
# Outputs (all per-(head, tile)):
#   ws_qd  bf16 [H, total_tiles, CHUNK, D]   q_decayed
#   ws_kd  bf16 [H, total_tiles, CHUNK, D]   k_decayed
#   ws_kr  bf16 [H, total_tiles, CHUNK, D]   k_restored
#   ws_gt  fp32 [H, total_tiles, D]          exp_g_total
#   ws_inv bf16 [H, total_tiles, CHUNK, CHUNK]  (I+L)^(-1)
#   ws_mqk bf16 [H, total_tiles, CHUNK, CHUNK]  Mqk
# ===========================================================================
@cute.kernel
def k1_full_kernel(
    tma_atom_q: cute.CopyAtom,
    tma_tensor_q: cute.Tensor,
    tma_atom_k: cute.CopyAtom,
    tma_tensor_k: cute.Tensor,
    tma_atom_g: cute.CopyAtom,
    tma_tensor_g: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    ws_qd: cute.Tensor,
    ws_kd: cute.Tensor,
    ws_kr: cute.Tensor,
    ws_gt: cute.Tensor,
    ws_inv: cute.Tensor,
    ws_mqk: cute.Tensor,
    H: cutlass.Constexpr[int],
    total_tiles: cutlass.Constexpr[int],
    T_total: cutlass.Constexpr[int],
    scale: cutlass.Constexpr[float],
    gate_scale: cutlass.Constexpr[float],
):
    tile_idx, head_idx, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    smem = cutlass.utils.SmemAllocator()
    qk_layout = cute.make_layout((CHUNK, D), stride=(D, 1))
    cc_layout = cute.make_layout((CHUNK, CHUNK), stride=(CHUNK, 1))
    sQ = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sK = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sGbf = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sGcs = smem.allocate_tensor(cutlass.Float32, qk_layout, 128)
    sGtot = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D,)), 128)
    sQD = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sKD = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sKI = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sL = smem.allocate_tensor(cutlass.Float32, cc_layout, 128)
    sMqk = smem.allocate_tensor(cutlass.Float32, cc_layout, 128)
    sINV = smem.allocate_tensor(cutlass.Float32, cc_layout, 128)
    sLp = smem.allocate_tensor(cutlass.Float32, cc_layout, 128)
    sTmp = smem.allocate_tensor(cutlass.Float32, cc_layout, 128)
    sMbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((1,)), 8)
    sMbar_ptr = sMbar.iterator

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(sMbar_ptr, cutlass.Int32(1))
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    gSrc_q = cute.local_tile(tma_tensor_q, (CHUNK, D), (None, None, None))
    gSrc_k = cute.local_tile(tma_tensor_k, (CHUNK, D), (None, None, None))
    gSrc_g = cute.local_tile(tma_tensor_g, (CHUNK, D), (None, None, None))
    tQs, tQg = cpasync.tma_partition(
        tma_atom_q,
        0,
        cute.make_layout(1),
        cute.group_modes(sQ, 0, 2),
        cute.group_modes(gSrc_q, 0, 2),
    )
    tKs, tKg = cpasync.tma_partition(
        tma_atom_k,
        0,
        cute.make_layout(1),
        cute.group_modes(sK, 0, 2),
        cute.group_modes(gSrc_k, 0, 2),
    )
    tGs, tGg = cpasync.tma_partition(
        tma_atom_g,
        0,
        cute.make_layout(1),
        cute.group_modes(sGbf, 0, 2),
        cute.group_modes(gSrc_g, 0, 2),
    )

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(sMbar_ptr, cutlass.Int32(3 * CHUNK * D * 2))
        cute.copy(tma_atom_q, tQg[(None, tile_idx, 0, head_idx)], tQs[(None,)], tma_bar_ptr=sMbar_ptr)
        cute.copy(tma_atom_k, tKg[(None, tile_idx, 0, head_idx)], tKs[(None,)], tma_bar_ptr=sMbar_ptr)
        cute.copy(tma_atom_g, tGg[(None, tile_idx, 0, head_idx)], tGs[(None,)], tma_bar_ptr=sMbar_ptr)

    cute.arch.mbarrier_wait(sMbar_ptr, cutlass.Int32(0))

    # L2 normalize
    row = tidx // 16
    col = (tidx % 16) * 8
    q_sq = cutlass.Float32(0.0)
    k_sq = cutlass.Float32(0.0)
    q_vals = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), cutlass.Float32)
    k_vals = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), cutlass.Float32)
    for j in cutlass.range_constexpr(8):
        qv = cutlass.Float32(sQ[row, col + j])
        kv = cutlass.Float32(sK[row, col + j])
        q_vals[j] = qv
        k_vals[j] = kv
        q_sq = q_sq + qv * qv
        k_sq = k_sq + kv * kv
    q_sq = cute.arch.warp_reduction(q_sq, lambda a, b: a + b, threads_in_group=16)
    k_sq = cute.arch.warp_reduction(k_sq, lambda a, b: a + b, threads_in_group=16)
    q_inv = cute.rsqrt(q_sq + cutlass.Float32(1.0e-6), fastmath=True)
    k_inv = cute.rsqrt(k_sq + cutlass.Float32(1.0e-6), fastmath=True)
    for j in cutlass.range_constexpr(8):
        sQ[row, col + j] = cutlass.BFloat16(q_vals[j] * q_inv)
        sK[row, col + j] = cutlass.BFloat16(k_vals[j] * k_inv)
    cute.arch.barrier()

    # Gate cumsum
    a_log_exp = cute.exp(cutlass.Float32(a_log[head_idx]), fastmath=True)
    if tidx < 128:
        col_c = tidx
        dt = cutlass.Float32(dt_bias[head_idx, col_c])
        s = cutlass.Float32(0.0)
        for r in cutlass.range_constexpr(CHUNK):
            x = cutlass.Float32(sGbf[r, col_c]) + dt
            x = a_log_exp * x
            sig = cutlass.Float32(0.5) * (cute.tanh(x * cutlass.Float32(0.5), fastmath=True) + cutlass.Float32(1.0))
            s = s + cutlass.Float32(gate_scale) * sig
            sGcs[r, col_c] = s
        sGtot[col_c] = cute.exp(s, fastmath=True)
    cute.arch.barrier()

    # decay_apply: into SMEM (sQD, sKD, sKI) AND dump to workspace
    ws_base = (head_idx * total_tiles + tile_idx) * (CHUNK * D)
    base = row * D + col
    for j in cutlass.range_constexpr(8):
        jj: cutlass.Constexpr[int] = j
        g_cs = sGcs[row, col + jj]
        gt = sGtot[col + jj]
        exp_pos = cute.exp(g_cs, fastmath=True)
        inv_pos = cutlass.Float32(1.0) / exp_pos
        rest = gt * inv_pos
        q_v = cutlass.Float32(sQ[row, col + jj])
        k_v = cutlass.Float32(sK[row, col + jj])
        qd_bf = cutlass.BFloat16(q_v * exp_pos * cutlass.Float32(scale))
        kd_bf = cutlass.BFloat16(k_v * exp_pos)
        ki_bf = cutlass.BFloat16(k_v * inv_pos)
        kr_bf = cutlass.BFloat16(k_v * rest)
        sQD[row, col + jj] = qd_bf
        sKD[row, col + jj] = kd_bf
        sKI[row, col + jj] = ki_bf
        ws_qd[ws_base + base + jj] = qd_bf
        ws_kd[ws_base + base + jj] = kd_bf
        ws_kr[ws_base + base + jj] = kr_bf

    if tidx < 128:
        gt_base = (head_idx * total_tiles + tile_idx) * D
        ws_gt[gt_base + tidx] = sGtot[tidx]
    cute.arch.barrier()

    # L = sKD @ sKI^T, Mqk = sQD @ sKI^T
    i = tidx // CHUNK
    j2 = tidx % CHUNK
    sum_l = cutlass.Float32(0.0)
    sum_m = cutlass.Float32(0.0)
    for kk in cutlass.range(D, unroll=8):
        ki_v = cutlass.Float32(sKI[j2, kk])
        sum_l = sum_l + cutlass.Float32(sKD[i, kk]) * ki_v
        sum_m = sum_m + cutlass.Float32(sQD[i, kk]) * ki_v
    beta_lin = head_idx * T_total + tile_idx * CHUNK + i
    if i > j2:
        bv = cutlass.Float32(beta[beta_lin])
        sig_b = cutlass.Float32(0.5) * (cute.tanh(bv * cutlass.Float32(0.5), fastmath=True) + cutlass.Float32(1.0))
        sL[i, j2] = sum_l * sig_b
    else:
        sL[i, j2] = cutlass.Float32(0.0)
    if i >= j2:
        sMqk[i, j2] = sum_m
    else:
        sMqk[i, j2] = cutlass.Float32(0.0)
    cute.arch.barrier()

    # Neumann inverse
    inv_v = cutlass.Float32(1.0 if i == j2 else 0.0) - sL[i, j2]
    sINV[i, j2] = inv_v
    sLp[i, j2] = sL[i, j2]
    cute.arch.barrier()

    for _p in cutlass.range_constexpr(3):
        s = cutlass.Float32(0.0)
        for kk in cutlass.range_constexpr(CHUNK):
            s = s + sLp[i, kk] * sLp[kk, j2]
        cute.arch.barrier()
        sTmp[i, j2] = s
        cute.arch.barrier()
        s2 = cutlass.Float32(0.0)
        for kk in cutlass.range_constexpr(CHUNK):
            s2 = s2 + sINV[i, kk] * sTmp[kk, j2]
        cute.arch.barrier()
        sINV[i, j2] = sINV[i, j2] + s2
        sLp[i, j2] = sTmp[i, j2]
        cute.arch.barrier()

    # Dump INV, Mqk
    ws_base_cc = (head_idx * total_tiles + tile_idx) * (CHUNK * CHUNK)
    ws_inv[ws_base_cc + i * CHUNK + j2] = cutlass.BFloat16(sINV[i, j2])
    ws_mqk[ws_base_cc + i * CHUNK + j2] = cutlass.BFloat16(sMqk[i, j2])


@cute.jit
def run_k1_full(
    q: cute.Tensor,
    k: cute.Tensor,
    g: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    ws_qd: cute.Tensor,
    ws_kd: cute.Tensor,
    ws_kr: cute.Tensor,
    ws_gt: cute.Tensor,
    ws_inv: cute.Tensor,
    ws_mqk: cute.Tensor,
    H: cutlass.Constexpr[int],
    total_tiles: cutlass.Constexpr[int],
    T_total: cutlass.Constexpr[int],
    scale: cutlass.Constexpr[float],
    gate_scale: cutlass.Constexpr[float],
    stream: cuda_drv.CUstream,
):
    smem_layout_qk = cute.make_layout((CHUNK, D), stride=(D, 1))

    def make_atom(t):
        view = cute.make_tensor(
            t.iterator,
            cute.make_layout((T_total, D, H), stride=(H * D, 1, D)),
        )
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            view,
            smem_layout_qk,
            (CHUNK, D),
        )

    tma_atom_q, tma_tensor_q = make_atom(q)
    tma_atom_k, tma_tensor_k = make_atom(k)
    tma_atom_g, tma_tensor_g = make_atom(g)

    smem_bytes = 6 * (CHUNK * D * 2) + (CHUNK * D * 4) + (D * 4) + 5 * (CHUNK * CHUNK * 4) + 8 + 512

    k1_full_kernel(
        tma_atom_q,
        tma_tensor_q,
        tma_atom_k,
        tma_tensor_k,
        tma_atom_g,
        tma_tensor_g,
        a_log,
        dt_bias,
        beta,
        ws_qd,
        ws_kd,
        ws_kr,
        ws_gt,
        ws_inv,
        ws_mqk,
        H,
        total_tiles,
        T_total,
        scale,
        gate_scale,
    ).launch(
        grid=(total_tiles, H, 1),
        block=[THREADS_PER_CTA, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


_compiled_cache_full: dict = {}


def launch_k1_full(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    gate_scale: float,
    ws_qd: torch.Tensor,
    ws_kd: torch.Tensor,
    ws_kr: torch.Tensor,
    ws_gt: torch.Tensor,
    ws_inv: torch.Tensor,
    ws_mqk: torch.Tensor,
) -> None:
    """Run K1 full pipeline; produces all 6 K2-ready workspace tensors."""
    for t in (q, k, g, beta):
        assert t.dtype == torch.bfloat16 and t.is_cuda and t.is_contiguous()
    assert A_log.dtype == torch.float32 and A_log.is_contiguous()
    assert dt_bias.dtype == torch.float32 and dt_bias.is_contiguous()
    B, T, H, K = q.shape
    assert K == D and T % CHUNK == 0
    total_tiles = (B * T) // CHUNK
    T_total = B * T

    key = (T_total, H, total_tiles, scale, gate_scale)
    if key not in _compiled_cache_full:
        stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
        q_flat = q.view(T_total, H, D)
        k_flat = k.view(T_total, H, D)
        g_flat = g.view(T_total, H, D)
        _compiled_cache_full[key] = cute.compile(
            run_k1_full,
            from_dlpack(q_flat.detach(), assumed_align=16),
            from_dlpack(k_flat.detach(), assumed_align=16),
            from_dlpack(g_flat.detach(), assumed_align=16),
            from_dlpack(A_log.detach(), assumed_align=16),
            from_dlpack(dt_bias.detach(), assumed_align=16),
            from_dlpack(beta.detach(), assumed_align=16),
            from_dlpack(ws_qd.detach(), assumed_align=16),
            from_dlpack(ws_kd.detach(), assumed_align=16),
            from_dlpack(ws_kr.detach(), assumed_align=16),
            from_dlpack(ws_gt.detach(), assumed_align=16),
            from_dlpack(ws_inv.detach(), assumed_align=16),
            from_dlpack(ws_mqk.detach(), assumed_align=16),
            H=H,
            total_tiles=total_tiles,
            T_total=T_total,
            scale=scale,
            gate_scale=gate_scale,
            stream=stream,
        )

    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    q_flat = q.view(T_total, H, D)
    k_flat = k.view(T_total, H, D)
    g_flat = g.view(T_total, H, D)
    _compiled_cache_full[key](
        q_flat,
        k_flat,
        g_flat,
        A_log,
        dt_bias,
        beta,
        ws_qd,
        ws_kd,
        ws_kr,
        ws_gt,
        ws_inv,
        ws_mqk,
        H,
        total_tiles,
        T_total,
        scale,
        gate_scale,
        stream,
    )

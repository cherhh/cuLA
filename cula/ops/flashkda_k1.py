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


__all__ = [
    "CHUNK",
    "D",
    "WORKSPACE_BYTES_PER_TILE",
    "K1Outputs",
    "launch_k1_phase1",
    "launch_k1_workspace_only",
]

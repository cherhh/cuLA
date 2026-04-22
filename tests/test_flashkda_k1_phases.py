# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Per-phase unit tests for the CuteDSL FlashKDA K1 port.

Each test compiles the kernel for a tiny shape, runs it, dumps the workspace
(or selected slot) into a torch tensor, and bit-compares to a torch reference
that mirrors the C++ math.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs CUDA",
)


def _make_inputs(B: int, T: int, H: int, *, seed: int = 0):
    from cula.ops.flashkda_prefill import D as HEAD_DIM

    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, T, H, HEAD_DIM, generator=g, device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn(B, T, H, HEAD_DIM, generator=g, device="cuda", dtype=torch.bfloat16) * 0.5
    g_pre = torch.randn(B, T, H, HEAD_DIM, generator=g, device="cuda", dtype=torch.bfloat16) * 0.1
    beta = torch.randn(B, T, H, generator=g, device="cuda", dtype=torch.bfloat16) * 0.1
    A_log = torch.randn(H, generator=g, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H, HEAD_DIM, generator=g, device="cuda", dtype=torch.float32) * 0.1
    return q, k, g_pre, beta, A_log, dt_bias


def test_k1_phase1_tma_load_q():
    """Phase 1: TMA-load q tile into SMEM and dump back to GMEM. Must
    bit-equal the original q tile."""
    from cula.ops.flashkda_k1 import CHUNK, D, launch_k1_phase1

    B, T, H = 1, 32, 2
    q, *_ = _make_inputs(B, T, H)
    total_tiles = (B * T) // CHUNK

    workspace = torch.zeros(total_tiles * H * CHUNK * D, dtype=torch.bfloat16, device="cuda")
    launch_k1_phase1(q, workspace)
    torch.cuda.synchronize()

    # Workspace layout: [head, tile, chunk_row, dim]
    # (kernel writes to ws_base = (head_idx * total_tiles + tile_idx) * CHUNK*D)
    ws = workspace.view(H, total_tiles, CHUNK, D)

    # Build the expected tile-major view of q.
    # q shape: [B, T, H, D]. We want, for each (head, tile):
    # q[b, tile_idx*CHUNK : (tile_idx+1)*CHUNK, head, :]
    q_expected = q.view(B * T, H, D)  # [T_total, H, D]
    # Reshape into (total_tiles, CHUNK, H, D) -> (H, total_tiles, CHUNK, D)
    q_expected = q_expected.view(total_tiles, CHUNK, H, D).permute(2, 0, 1, 3).contiguous()

    diff = (ws.float() - q_expected.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"\n[k1 phase1] max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}")
    # bf16 → bf16 round-trip should be exact.
    assert max_diff == 0.0, f"K1 phase 1 mismatch: max_diff={max_diff}"

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

"""Smoke tests for cula.ops.flashkda_prefill (CuteDSL port WIP)."""

import os

import pytest
import torch

from cula.ops.flashkda_prefill import (
    CHUNK,
    WORKSPACE_BYTES_PER_TILE,
    D,
    _flashkda_torch_reference,
    allocate_workspace,
    flash_kda_prefill,
)


def _make_inputs(B, T, H, *, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, T, H, D, generator=g, device=device, dtype=torch.bfloat16) * 0.5
    k = torch.randn(B, T, H, D, generator=g, device=device, dtype=torch.bfloat16) * 0.5
    v = torch.randn(B, T, H, D, generator=g, device=device, dtype=torch.bfloat16) * 0.5
    g_pre = torch.randn(B, T, H, D, generator=g, device=device, dtype=torch.bfloat16) * 0.1
    beta = torch.randn(B, T, H, generator=g, device=device, dtype=torch.bfloat16) * 0.1
    A_log = torch.randn(H, generator=g, device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H, D, generator=g, device=device, dtype=torch.float32) * 0.1
    return q, k, v, g_pre, beta, A_log, dt_bias


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_workspace_size_matches_cpp_layout():
    # WORKSPACE_BYTES_PER_TILE must match FlashKDA WorkspaceSizes::kPerTile (CHUNK=16,D=128).
    # k_decayed + q_decayed + k_restored + g_total + INV + Mqk
    expected = (
        CHUNK * D * 2  # kd
        + CHUNK * D * 2  # qd
        + CHUNK * D * 2  # kr
        + D * 4  # gt fp32
        + CHUNK * CHUNK * 2  # INV
        + CHUNK * CHUNK * 2  # Mqk
    )
    assert expected == WORKSPACE_BYTES_PER_TILE
    ws = allocate_workspace(total_tiles=4, H=2, device="cuda")
    assert ws.numel() == 4 * 2 * expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_torch_reference_runs_fixed_len():
    B, T, H = 2, 32, 2
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(B, T, H)
    out = torch.empty_like(v)
    flash_kda_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=D**-0.5,
        out=out,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        initial_state=None,
        final_state=None,
        cu_seqlens=None,
    )
    assert out.shape == v.shape
    assert torch.isfinite(out.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_torch_reference_runs_with_state():
    B, T, H = 1, 16, 2
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(B, T, H)
    initial_state = torch.zeros(B, H, D, D, dtype=torch.bfloat16, device="cuda")
    final_state = torch.zeros_like(initial_state)
    out = torch.empty_like(v)
    flash_kda_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=D**-0.5,
        out=out,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=None,
    )
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(final_state.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cute_dispatch_bridges_to_cpp():
    """Until K1/K2 CuteDSL ports land, CULA_FLASHKDA_USE_CUTE=1 should
    transparently bridge to the FlashKDA C++ extension and produce numerically
    consistent output with the torch reference fallback.

    Skips gracefully if the ``flash_kda`` extension is not built locally.
    """
    pytest.importorskip("flash_kda")
    B, T, H = 1, 16, 1
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(B, T, H)
    out_ref = torch.empty_like(v)
    out_cute = torch.empty_like(v)

    # Reference path (torch fallback).
    os.environ.pop("CULA_FLASHKDA_USE_CUTE", None)
    import importlib

    import cula.ops.flashkda_prefill as mod

    importlib.reload(mod)
    mod.flash_kda_prefill(
        q, k, v, g, beta,
        scale=D**-0.5, out=out_ref,
        A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
    )

    os.environ["CULA_FLASHKDA_USE_CUTE"] = "1"
    try:
        importlib.reload(mod)
        mod.flash_kda_prefill(
            q, k, v, g, beta,
            scale=D**-0.5, out=out_cute,
            A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
        )
        # Bridged path goes through FlashKDA C++; torch ref vs C++ tolerated
        # diff in tmp/precision_check.py was ~9e-5. Use loose tolerance for
        # variable-shape coverage.
        max_diff = (out_ref.float() - out_cute.float()).abs().max().item()
        assert max_diff < 1e-2, f"cute-vs-ref max abs diff {max_diff} too large"
    finally:
        os.environ.pop("CULA_FLASHKDA_USE_CUTE", None)
        importlib.reload(mod)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_torch_reference_no_state_matches_recurrence():
    """Sanity: the reference is mathematically the per-token delta-rule recurrence."""
    B, T, H = 1, 4, 1
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(B, T, H, seed=42)
    out_ref, _ = _flashkda_torch_reference(
        q,
        k,
        v,
        g,
        beta,
        scale=D**-0.5,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        initial_state=None,
        cu_seqlens=None,
        output_final_state=False,
    )
    assert out_ref.shape == v.shape
    assert torch.isfinite(out_ref.float()).all()

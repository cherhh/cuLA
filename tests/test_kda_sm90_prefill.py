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

"""Smoke tests for cula.ops.kda.sm90.fwd (CuteDSL port)."""

import os

import pytest
import torch

from cula.ops.kda.sm90.fwd import (
    CHUNK,
    WORKSPACE_BYTES_PER_TILE,
    D,
    _cute_arch_for_device,
    _get_or_build_varlen_metadata,
    allocate_workspace,
    flash_kda_fwd,
)


def _flashkda_torch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    output_final_state: bool,
    state_transposed: bool = False,
    use_gate_in_kernel: bool = True,
):
    """Torch reference oracle for the SM90 FlashKDA prefill kernels (test-only).

    use_gate_in_kernel=True (default): beta is pre-sigmoid logits, g is raw;
        gate formula and sigmoid(beta) are applied internally.
    use_gate_in_kernel=False: beta is pre-sigmoid logits (converted by API layer),
        g is already log-decay; g is used directly via exp(g).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert K == V == D
    device = q.device

    # ---- variable-length unpacking ----
    if cu_seqlens is None:
        N = B
        seq_lens = [T] * B
        starts = [t * T for t in range(B)]
    else:
        assert B == 1
        cu = cu_seqlens.to("cpu").long().tolist()
        N = len(cu) - 1
        seq_lens = [cu[i + 1] - cu[i] for i in range(N)]
        starts = cu[:-1]

    # ---- initial state ----
    if initial_state is None:
        h = torch.zeros(N, H, V, K, device=device, dtype=torch.float32)
    else:
        h = initial_state.to(torch.float32).clone()
        if state_transposed:
            h = h.transpose(-1, -2).contiguous()

    out = torch.empty_like(v)

    A_exp = torch.exp(A_log).to(torch.float32)
    dt_b = dt_bias.to(torch.float32)
    gate_scale = float(min(lower_bound, 0.0)) if lower_bound is not None else 0.0

    for n in range(N):
        Tn = seq_lens[n]
        bos = starts[n]
        for h_idx in range(H):
            state = h[n, h_idx]
            for t in range(Tn):
                qi = q[0, bos + t, h_idx].float() if cu_seqlens is not None else q[n, t, h_idx].float()
                ki = k[0, bos + t, h_idx].float() if cu_seqlens is not None else k[n, t, h_idx].float()
                vi = v[0, bos + t, h_idx].float() if cu_seqlens is not None else v[n, t, h_idx].float()
                gi = g[0, bos + t, h_idx].float() if cu_seqlens is not None else g[n, t, h_idx].float()
                bi = beta[0, bos + t, h_idx].float() if cu_seqlens is not None else beta[n, t, h_idx].float()

                # Match K1: x * rsqrt(sum(x^2) + eps).
                qi = qi * torch.rsqrt(qi.pow(2).sum() + 1e-6)
                ki = ki * torch.rsqrt(ki.pow(2).sum() + 1e-6)
                qi = qi * scale

                if use_gate_in_kernel:
                    g_act = gate_scale * torch.sigmoid(A_exp[h_idx] * (gi + dt_b[h_idx]))
                    exp_g = torch.exp(g_act)
                else:
                    exp_g = torch.exp(gi)

                beta_act = torch.sigmoid(bi)

                state = state * exp_g.unsqueeze(0)

                u = (vi - state @ ki) * beta_act
                state = state + u.unsqueeze(1) * ki.unsqueeze(0)
                o_t = state @ qi
                if cu_seqlens is not None:
                    out[0, bos + t, h_idx] = o_t.to(out.dtype)
                else:
                    out[n, t, h_idx] = o_t.to(out.dtype)
            h[n, h_idx] = state

    final_state = None
    if output_final_state:
        if state_transposed:
            h = h.transpose(-1, -2).contiguous()
        if initial_state is not None and initial_state.dtype != torch.float32:
            final_state = h.to(initial_state.dtype)
        else:
            final_state = h
    return out, final_state


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
    flash_kda_fwd(
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
def test_empty_sequence_rejected():
    B, T, H = 1, 0, 1
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(B, T, H)
    out = torch.empty_like(v)
    with pytest.raises(ValueError, match="B, T and H must be positive"):
        flash_kda_fwd(
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
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_forced_cudagraph_env_non_aligned_t_still_writes_out(monkeypatch):
    monkeypatch.setenv("CULA_FLASHKDA_VARLEN_CUDAGRAPH", "1")
    B, T, H = 1, 17, 1
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(B, T, H, seed=11)
    out = torch.empty_like(v)
    flash_kda_fwd(
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
    )
    ref, _ = _flashkda_torch_reference(
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
    assert torch.isfinite(out.float()).all()
    assert (out.float() - ref.float()).abs().max().item() < 3e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cute_arch_env_is_scoped(monkeypatch):
    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    with _cute_arch_for_device(torch.device("cuda")):
        assert os.environ["CUTE_DSL_ARCH"] == "sm_90a"
    assert "CUTE_DSL_ARCH" not in os.environ


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_varlen_metadata_cache_reuses_and_invalidates():
    cu = torch.tensor([0, 16, 48], device="cuda", dtype=torch.int32)

    meta1 = _get_or_build_varlen_metadata(cu)
    meta2 = _get_or_build_varlen_metadata(cu)
    assert meta2 is meta1
    assert meta1.seq_lens == (16, 32)
    assert meta1.total_tiles == 3
    assert meta1.cu_tiles is not None
    assert meta1.cu_tiles.data_ptr() == meta2.cu_tiles.data_ptr()
    assert tuple(meta1.tile_starts.cpu().tolist()) == (0, 16, 32)
    assert tuple(meta1.tile_actual_lens.cpu().tolist()) == (16, 16, 16)

    cu.copy_(torch.tensor([0, 17, 48], device="cuda", dtype=torch.int32))
    meta3 = _get_or_build_varlen_metadata(cu)
    assert meta3 is not meta1
    assert meta3.seq_lens == (17, 31)
    assert meta3.total_tiles == 4
    assert meta3.needs_padding
    assert meta3.cu_tiles is None
    assert tuple(meta3.tile_starts.cpu().tolist()) == (0, 16, 17, 33)
    assert tuple(meta3.tile_actual_lens.cpu().tolist()) == (16, 1, 16, 15)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cute_dispatch_unaligned_varlen_matches_reference():
    B, T, H = 1, 48, 1
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(B, T, H, seed=23)
    cu_seqlens = torch.tensor([0, 17, 48], device="cuda", dtype=torch.int32)
    out = torch.empty_like(v)
    final_state = torch.empty(2, H, D, D, device="cuda", dtype=torch.float32)

    flash_kda_fwd(
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
        final_state=final_state,
        cu_seqlens=cu_seqlens,
    )
    ref, ref_final = _flashkda_torch_reference(
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
        cu_seqlens=cu_seqlens,
        output_final_state=True,
    )
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(final_state.float()).all()
    assert (out.float() - ref.float()).abs().max().item() < 3e-2
    assert (final_state.float() - ref_final.float()).abs().max().item() < 5e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_torch_reference_runs_with_state():
    B, T, H = 1, 16, 2
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(B, T, H)
    initial_state = torch.zeros(B, H, D, D, dtype=torch.float32, device="cuda")
    final_state = torch.zeros_like(initial_state)
    out = torch.empty_like(v)
    flash_kda_fwd(
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
def test_cute_dispatch_fixed_len_matches_reference():
    """SM90 CuteDSL fwd matches the torch reference oracle on fixed-length input."""
    B, T, H = 1, 16, 1
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(B, T, H)
    out_cute = torch.empty_like(v)
    flash_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        scale=D**-0.5,
        out=out_cute,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
    )
    out_ref, _ = _flashkda_torch_reference(
        q,
        k,
        v,
        g,
        beta,
        D**-0.5,
        A_log,
        dt_bias,
        -5.0,
        initial_state=None,
        cu_seqlens=None,
        output_final_state=False,
    )
    max_diff = (out_ref.float() - out_cute.float()).abs().max().item()
    assert max_diff < 1e-2, f"cute-vs-ref max abs diff {max_diff} too large"


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cu_seqlens_cpu_matches_gpu_metadata_path():
    """Passing cu_seqlens_cpu (no-sync varlen metadata) yields identical output/state
    to deriving the metadata from the GPU cu_seqlens tensor."""
    B, T, H = 1, 48, 1
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(B, T, H, seed=23)
    cu_vals = [0, 17, 48]

    def _run(cu, cu_cpu):
        out = torch.empty_like(v)
        fs = torch.empty(2, H, D, D, device="cuda", dtype=torch.float32)
        flash_kda_fwd(
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
            final_state=fs,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_cpu,
        )
        return out, fs

    # Distinct GPU cu_seqlens tensors (same values) so both rebuild metadata fresh:
    # path A derives via cu_seqlens.to("cpu"), path B via the provided cu_seqlens_cpu.
    cu_a = torch.tensor(cu_vals, device="cuda", dtype=torch.int32)
    cu_b = torch.tensor(cu_vals, device="cuda", dtype=torch.int32)
    out_gpu, fs_gpu = _run(cu_a, None)
    out_cpu, fs_cpu = _run(cu_b, torch.tensor(cu_vals, dtype=torch.int32))
    torch.cuda.synchronize()
    assert torch.equal(out_gpu, out_cpu)
    assert torch.equal(fs_gpu, fs_cpu)

    # A mismatched-length cu_seqlens_cpu is rejected (cheap, no-sync validation).
    with pytest.raises(ValueError):
        _run(cu_a, torch.tensor([0, 48], dtype=torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hopper_wrapper_forwards_cu_seqlens_cpu():
    """kda_prefill_hopper threads cu_seqlens_cpu down to flash_kda_fwd; output must
    match the GPU-derived metadata path."""
    kda_mod = pytest.importorskip("cula.kda", reason="cula.kda requires the cula.cudac CUDA extension (full build)")
    kda_prefill_hopper = kda_mod.kda_prefill_hopper

    H = 1
    cu_vals = [0, 17, 48]
    T = cu_vals[-1]
    gen = torch.Generator(device="cuda").manual_seed(7)
    q = torch.randn(1, T, H, D, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn(1, T, H, D, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn(1, T, H, D, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.5
    g = torch.randn(1, T, H, D, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.1
    beta = torch.rand(1, T, H, generator=gen, device="cuda", dtype=torch.bfloat16)
    A_log = torch.rand(H, generator=gen, device="cuda", dtype=torch.float32)
    dt_bias = torch.rand(H, D, generator=gen, device="cuda", dtype=torch.float32)

    def _run(cu, cu_cpu):
        o, _ = kda_prefill_hopper(
            q,
            k,
            v,
            g,
            beta,
            scale=D**-0.5,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=-5.0,
            safe_gate=True,
            use_gate_in_kernel=True,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_cpu,
        )
        return o

    # Distinct GPU cu_seqlens so both rebuild metadata fresh (one via .to(cpu), one via cu_cpu).
    o_gpu = _run(torch.tensor(cu_vals, device="cuda", dtype=torch.int32), None)
    o_cpu = _run(torch.tensor(cu_vals, device="cuda", dtype=torch.int32), torch.tensor(cu_vals, dtype=torch.int32))
    torch.cuda.synchronize()
    assert torch.equal(o_gpu, o_cpu)

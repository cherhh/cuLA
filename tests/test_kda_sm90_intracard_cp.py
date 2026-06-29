# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Intracard-CP end-to-end correctness: CP vs serial, CP vs FLA, determinism."""

import pytest
import torch
from fla.ops.kda.chunk import chunk_kda as fla_chunk_kda
from fla.utils import assert_close

from cula.kda import kda_prefill_hopper as cula_kda_prefill
from cula.ops.kda.sm90.cp import intracard_prefill
from cula.ops.kda.sm90.fwd import D, flash_kda_fwd

H = 8
SCALE = D**-0.5
LB = -5.0
TOL_MAX = 1e-2
TOL_RMSE = 4e-3
TOL_FLA = 5e-3

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _make_inputs(B, T, seed=0):
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=dev)
    g = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=dev)
    beta = torch.randn(B, T, H, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev)
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=dev)
    return q, k, v, g, beta, A_log, dt_bias


def _make_fla_inputs(T, *, with_state, n_state, seed):
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    q = torch.rand(1, T, H, D, dtype=torch.bfloat16, device=dev)
    k = torch.rand(1, T, H, D, dtype=torch.bfloat16, device=dev)
    v = torch.rand(1, T, H, D, dtype=torch.bfloat16, device=dev)
    g = torch.randn(1, T, H, D, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev)
    dt_bias = torch.randn(H * D, dtype=torch.float32, device=dev)
    beta = torch.randn(1, T, H, dtype=torch.float32, device=dev).sigmoid().to(torch.bfloat16)
    h0 = torch.randn(n_state, H, D, D, dtype=torch.float32, device=dev) if with_state else None
    return q, k, v, g, beta, A_log, dt_bias, h0


def _rel_max(a, b):
    return (a.float() - b.float()).abs().max().item() / max(b.float().abs().max().item(), 1e-6)


def _rel_rmse(a, b):
    a, b = a.float(), b.float()
    return (a - b).pow(2).mean().sqrt().item() / max(b.pow(2).mean().sqrt().item(), 1e-6)


def _assert_cp_matches(actual, ref, name):
    rrmse = _rel_rmse(actual, ref)
    assert rrmse < TOL_RMSE, f"{name}: rel_rmse {rrmse:.2e} >= {TOL_RMSE}"
    rmax = _rel_max(actual, ref)
    assert rmax < TOL_MAX, f"{name}: rel_max {rmax:.2e} >= {TOL_MAX}"


def _alloc_final(n_seqs, device):
    return torch.empty(n_seqs, H, D, D, dtype=torch.float32, device=device)


def _run_serial(q, k, v, g, beta, A_log, dt_bias, init=None, want_final=True, cu=None, transposed=False):
    n = (cu.numel() - 1 if cu is not None else q.shape[0]) if want_final else 0
    out = torch.empty_like(v)
    fin = _alloc_final(n, q.device) if want_final else None
    flash_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        scale=SCALE,
        out=out,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=LB,
        initial_state=init,
        final_state=fin,
        cu_seqlens=cu,
        state_transposed=transposed,
    )
    return out, fin


def _run_cp(q, k, v, g, beta, A_log, dt_bias, init=None, want_final=True, cu=None, transposed=False, s_split=4):
    n = (cu.numel() - 1 if cu is not None else q.shape[0]) if want_final else 0
    out = torch.empty_like(v)
    fin = _alloc_final(n, q.device) if want_final else None
    intracard_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=SCALE,
        out=out,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=LB,
        initial_state=init,
        final_state=fin,
        cu_seqlens=cu,
        state_transposed=transposed,
        s_split=s_split,
    )
    return out, fin


# ---------------------------------------------------------------------------
# CP vs serial (same K1/K2, isolates CP-specific logic)
# ---------------------------------------------------------------------------
@needs_cuda
@pytest.mark.parametrize("s_split", [1, 2, 4, 7])
def test_cp_matches_serial_fixed(s_split):
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(1, 2048)
    out_ref, fin_ref = _run_serial(q, k, v, g, beta, A_log, dt_bias, None, True)
    out_cp, fin_cp = _run_cp(q, k, v, g, beta, A_log, dt_bias, None, True, s_split=s_split)
    _assert_cp_matches(out_cp, out_ref, "o")
    _assert_cp_matches(fin_cp, fin_ref, "ht")


@needs_cuda
def test_cp_matches_serial_fixed_b2_with_state():
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(2, 1024, seed=3)
    init = torch.randn(2, H, D, D, dtype=torch.float32, device="cuda")
    out_ref, fin_ref = _run_serial(q, k, v, g, beta, A_log, dt_bias, init, True)
    out_cp, fin_cp = _run_cp(q, k, v, g, beta, A_log, dt_bias, init, True, s_split=4)
    _assert_cp_matches(out_cp, out_ref, "o")
    _assert_cp_matches(fin_cp, fin_ref, "ht")


@needs_cuda
def test_cp_matches_serial_varlen():
    lens = [1024, 512, 2048, 256]
    T = sum(lens)
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(1, T, seed=7)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
    init = torch.randn(len(lens), H, D, D, dtype=torch.float32, device="cuda")
    out_ref, fin_ref = _run_serial(q, k, v, g, beta, A_log, dt_bias, init, True, cu=cu)
    out_cp, fin_cp = _run_cp(q, k, v, g, beta, A_log, dt_bias, init, True, cu=cu, s_split=4)
    _assert_cp_matches(out_cp, out_ref, "o")
    _assert_cp_matches(fin_cp, fin_ref, "ht")


@needs_cuda
def test_cp_matches_serial_state_transposed():
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(1, 1024, seed=11)
    init = torch.randn(1, H, D, D, dtype=torch.float32, device="cuda")
    out_ref, fin_ref = _run_serial(q, k, v, g, beta, A_log, dt_bias, init, True, transposed=True)
    out_cp, fin_cp = _run_cp(q, k, v, g, beta, A_log, dt_bias, init, True, transposed=True, s_split=4)
    _assert_cp_matches(out_cp, out_ref, "o")
    _assert_cp_matches(fin_cp, fin_ref, "ht")


@needs_cuda
def test_cp_no_final_state():
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(1, 512, seed=13)
    out_ref, _ = _run_serial(q, k, v, g, beta, A_log, dt_bias, None, False)
    out_cp, _ = _run_cp(q, k, v, g, beta, A_log, dt_bias, None, False, s_split=2)
    _assert_cp_matches(out_cp, out_ref, "o")


# ---------------------------------------------------------------------------
# Non-CHUNK-aligned inputs
# ---------------------------------------------------------------------------
@needs_cuda
@pytest.mark.parametrize(
    "lens",
    [
        [1024, 1, 63, 65, 129],
        [28679, 4096],
        [40007],
        [32768, 100],
    ],
)
def test_cp_matches_serial_varlen_nonaligned(lens):
    T = sum(lens)
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(1, T, seed=5)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
    out_ref, fin_ref = _run_serial(q, k, v, g, beta, A_log, dt_bias, None, True, cu=cu)
    out_cp, fin_cp = _run_cp(q, k, v, g, beta, A_log, dt_bias, None, True, cu=cu, s_split=8)
    _assert_cp_matches(out_cp, out_ref, "o")
    _assert_cp_matches(fin_cp, fin_ref, "ht")


@needs_cuda
@pytest.mark.parametrize("T", [100, 4100, 8197])
def test_cp_matches_serial_dense_nonaligned(T):
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(1, T, seed=9)
    out_ref, fin_ref = _run_serial(q, k, v, g, beta, A_log, dt_bias, None, True)
    out_cp, fin_cp = _run_cp(q, k, v, g, beta, A_log, dt_bias, None, True, s_split=8)
    _assert_cp_matches(out_cp, out_ref, "o")
    _assert_cp_matches(fin_cp, fin_ref, "ht")


# ---------------------------------------------------------------------------
# CP vs FLA (ground truth) — forced CP via public API
# ---------------------------------------------------------------------------
def _check_cp_vs_fla(T, *, with_state, cu, seed):
    n_state = (cu.numel() - 1) if cu is not None else 1
    q, k, v, g, beta, A_log, dt_bias, h0 = _make_fla_inputs(T, with_state=with_state, n_state=n_state, seed=seed)
    cu_cpu = cu.cpu() if cu is not None else None
    with torch.no_grad():
        ref_o, ref_ht = fla_chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=h0,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_cpu,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=LB,
        )
        h0_vk = h0.transpose(-2, -1).contiguous() if h0 is not None else None
        cp_o, cp_ht_vk = cula_kda_prefill(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=h0_vk,
            cu_seqlens=cu,
            output_final_state=True,
            safe_gate=True,
            lower_bound=LB,
            use_intracard_cp=True,
        )
    assert_close("o", ref_o, cp_o, TOL_FLA)
    assert_close("ht", ref_ht, cp_ht_vk.transpose(-2, -1), TOL_FLA)


@needs_cuda
@pytest.mark.parametrize("T", [8192, 16384])
def test_cp_vs_fla_dense(T):
    _check_cp_vs_fla(T, with_state=False, cu=None, seed=42)


@needs_cuda
def test_cp_vs_fla_dense_with_state():
    _check_cp_vs_fla(16384, with_state=True, cu=None, seed=42)


@needs_cuda
def test_cp_vs_fla_varlen_with_state():
    lens = [16384, 8192, 4096]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
    _check_cp_vs_fla(sum(lens), with_state=True, cu=cu, seed=7)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
_DETERMINISM_ITERS = 10000


@needs_cuda
def test_cp_determinism():
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(1, 4096, seed=17)
    out0, fin0 = _run_cp(q, k, v, g, beta, A_log, dt_bias, None, True, s_split=8)
    for i in range(_DETERMINISM_ITERS):
        out, fin = _run_cp(q, k, v, g, beta, A_log, dt_bias, None, True, s_split=8)
        assert torch.equal(out, out0), f"non-deterministic out at iter {i}"
        assert torch.equal(fin, fin0), f"non-deterministic ht at iter {i}"


@needs_cuda
def test_cp_determinism_varlen():
    lens = [1024, 1, 63, 65, 129]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(1, sum(lens), seed=5)
    out0, fin0 = _run_cp(q, k, v, g, beta, A_log, dt_bias, None, True, cu=cu, s_split=8)
    for i in range(_DETERMINISM_ITERS):
        out, fin = _run_cp(q, k, v, g, beta, A_log, dt_bias, None, True, cu=cu, s_split=8)
        assert torch.equal(out, out0), f"non-deterministic varlen out at iter {i}"
        assert torch.equal(fin, fin0), f"non-deterministic varlen ht at iter {i}"

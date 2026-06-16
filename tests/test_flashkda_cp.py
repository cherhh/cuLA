# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Intracard-CP correctness: flash_kda_prefill_cp vs serial CuTeDSL prefill."""

import os

import pytest
import torch

os.environ["CULA_FLASHKDA_USE_CUTE"] = "1"
os.environ.setdefault("CULA_FLASHKDA_STRICT_CUTE", "1")

from cula.ops.flashkda.sm90.cp import _merge_carries_, _plan_segments, flash_kda_prefill_cp  # noqa: E402
from cula.ops.flashkda.sm90.prefill import D, flash_kda_prefill  # noqa: E402

H = 8
SCALE = D**-0.5
LB = -5.0
TOL = 2e-2  # bf16 chain tolerance class (serial tests use 1e-2 abs vs torch ref)

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


def _rel_max(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.float() - b.float()).abs().max().item()
    return d / max(b.float().abs().max().item(), 1e-6)


def _run_serial(q, k, v, g, beta, init, want_final, cu=None, transposed=False):
    out = torch.empty_like(v)
    fin = (
        torch.empty(
            (cu.numel() - 1 if cu is not None else q.shape[0], H, D, D),
            dtype=torch.float32, device=q.device,
        )
        if want_final
        else None
    )
    flash_kda_prefill(
        q, k, v, g, beta, scale=SCALE, out=out,
        A_log=_run_serial.A_log, dt_bias=_run_serial.dt_bias, lower_bound=LB,
        initial_state=init, final_state=fin, cu_seqlens=cu,
        state_transposed=transposed,
    )
    return out, fin


def _run_cp(q, k, v, g, beta, init, want_final, cu=None, transposed=False, s_split=4):
    out = torch.empty_like(v)
    fin = (
        torch.empty(
            (cu.numel() - 1 if cu is not None else q.shape[0], H, D, D),
            dtype=torch.float32, device=q.device,
        )
        if want_final
        else None
    )
    flash_kda_prefill_cp(
        q, k, v, g, beta, scale=SCALE, out=out,
        A_log=_run_cp.A_log, dt_bias=_run_cp.dt_bias, lower_bound=LB,
        initial_state=init, final_state=fin, cu_seqlens=cu,
        state_transposed=transposed, s_split=s_split,
    )
    return out, fin


# ---------------------------------------------------------------------------
# Merge unit test (no kernels): right-multiply bhvk convention
# ---------------------------------------------------------------------------
def test_merge_unit():
    torch.manual_seed(1)
    S, Hh = 6, 3
    per_seq = [(0, 4), (4, 2)]  # two sequences: 4 + 2 segments
    m_seg = torch.randn(S, Hh, 16, 16, dtype=torch.float64)
    b_seg = torch.randn(S, Hh, 16, 16, dtype=torch.float64)
    init = torch.randn(2, Hh, 16, 16, dtype=torch.float64)

    carries = _merge_carries_(torch.empty_like(b_seg), m_seg, b_seg, per_seq, init)

    for s, (first, n_seg) in enumerate(per_seq):
        carry = init[s].clone()
        for i in range(first, first + n_seg):
            assert torch.allclose(carries[i], carry, atol=1e-12), f"seg {i}"
            carry = torch.baddbmm(b_seg[i], carry, m_seg[i])


def test_plan_segments():
    seg_cu, per_seq = _plan_segments([64, 7, 4], s_split=4)
    # seq0: 64 tiles -> 4 segs of 16; seq1: 7 tiles -> min(4, 7//4=1)=1 seg;
    # seq2: 4 tiles -> 1 seg.
    assert per_seq == [(0, 4), (4, 1), (5, 1)]
    assert seg_cu == [0, 16, 32, 48, 64, 71, 75]


# ---------------------------------------------------------------------------
# CP vs serial (kernel path)
# ---------------------------------------------------------------------------
@needs_cuda
@pytest.mark.parametrize("s_split", [1, 2, 4, 7])
def test_cp_matches_serial_fixed(s_split):
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(1, 2048)
    _run_serial.A_log = _run_cp.A_log = A_log
    _run_serial.dt_bias = _run_cp.dt_bias = dt_bias

    out_ref, fin_ref = _run_serial(q, k, v, g, beta, None, True)
    out_cp, fin_cp = _run_cp(q, k, v, g, beta, None, True, s_split=s_split)

    assert _rel_max(out_cp, out_ref) < TOL
    assert _rel_max(fin_cp, fin_ref) < TOL


@needs_cuda
def test_cp_matches_serial_fixed_b2_with_state():
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(2, 1024, seed=3)
    _run_serial.A_log = _run_cp.A_log = A_log
    _run_serial.dt_bias = _run_cp.dt_bias = dt_bias
    init = torch.randn(2, H, D, D, dtype=torch.float32, device="cuda")

    out_ref, fin_ref = _run_serial(q, k, v, g, beta, init, True)
    out_cp, fin_cp = _run_cp(q, k, v, g, beta, init, True, s_split=4)

    assert _rel_max(out_cp, out_ref) < TOL
    assert _rel_max(fin_cp, fin_ref) < TOL


@needs_cuda
def test_cp_matches_serial_varlen():
    torch.manual_seed(7)
    lens = [1024, 512, 2048, 256]
    T = sum(lens)
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(1, T, seed=7)
    _run_serial.A_log = _run_cp.A_log = A_log
    _run_serial.dt_bias = _run_cp.dt_bias = dt_bias
    cu = torch.tensor(
        [0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.long, device="cuda"
    )
    init = torch.randn(len(lens), H, D, D, dtype=torch.float32, device="cuda")

    out_ref, fin_ref = _run_serial(q, k, v, g, beta, init, True, cu=cu)
    out_cp, fin_cp = _run_cp(q, k, v, g, beta, init, True, cu=cu, s_split=4)

    assert _rel_max(out_cp, out_ref) < TOL
    assert _rel_max(fin_cp, fin_ref) < TOL


@needs_cuda
def test_cp_matches_serial_state_transposed():
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(1, 1024, seed=11)
    _run_serial.A_log = _run_cp.A_log = A_log
    _run_serial.dt_bias = _run_cp.dt_bias = dt_bias
    init = torch.randn(1, H, D, D, dtype=torch.float32, device="cuda")

    out_ref, fin_ref = _run_serial(q, k, v, g, beta, init, True, transposed=True)
    out_cp, fin_cp = _run_cp(q, k, v, g, beta, init, True, transposed=True, s_split=4)

    assert _rel_max(out_cp, out_ref) < TOL
    assert _rel_max(fin_cp, fin_ref) < TOL


@needs_cuda
def test_cp_no_final_state():
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(1, 512, seed=13)
    _run_serial.A_log = _run_cp.A_log = A_log
    _run_serial.dt_bias = _run_cp.dt_bias = dt_bias

    out_ref, _ = _run_serial(q, k, v, g, beta, None, False)
    out_cp, _ = _run_cp(q, k, v, g, beta, None, False, s_split=2)

    assert _rel_max(out_cp, out_ref) < TOL

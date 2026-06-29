# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Intracard-CP correctness.

Primary oracle: FLA Triton (ground truth, per cula-kernel-wiki) — forced-CP
``cula_kda_prefill`` vs ``fla_chunk_kda`` on long sequences (the regime CP runs
in, which the short-sequence serial-vs-FLA suite does not cover; comparing CP to
cuLA's own serial cannot catch a bug the two share via K1/K2).
Secondary: CP vs serial CuTeDSL prefill — isolates CP-specific logic (pre_scan/
merge/segment) from the shared K1/K2 math when the FLA check fails.
Plus a determinism guard (cula-kernel-wiki §1.2) on the warp-specialized CP path.
"""

import pytest
import torch
from fla.ops.kda.chunk import chunk_kda as fla_chunk_kda

from cula.kda import kda_prefill_hopper as cula_kda_prefill
from cula.ops.kda.sm90.cp import intracard_prefill
from cula.ops.kda.sm90.cp.plan import _plan_segments
from cula.ops.kda.sm90.fwd import D, flash_kda_fwd

H = 8
SCALE = D**-0.5
LB = -5.0
# CP vs serial bounds (cuLA multi-metric convention). Measured worst across the suite:
# rel_max 5.1e-3, rel_rmse 1.6e-3 (both on the +init cases — initial_state propagates
# through CP's TF32 merge); most aligned / long-varlen cases are bit-exact. Bounds are
# ~2x the measured worst.
TOL_MAX = 1e-2  # worst-element relative error
TOL_RMSE = 4e-3  # mean guard: catches systematic divergence a single loose rel_max can't

# CP vs FLA uses the SAME bar as the serial-vs-FLA suite: FLA's assert_close == relative
# L2 error (rel_rmse) < 0.005. Measured cuLA-SM90 vs FLA is a CONSTANT ~3.3e-3 rel_rmse at
# every length (512..16384) — not CP (CP===serial vs FLA), not accumulation, but the SM90
# bf16 port's intrinsic gap to FLA. The wiki's tighter rel_rmse<4e-4 / rel_max<5e-3 are the
# SM100 standard this bf16 port does not meet (its worst element vs FLA is ~1.3e-2, noise).
TOL_FLA_RMSE = 5e-3  # == serial-vs-FLA assert_close(0.005); measured 3.3e-3

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


def _rel_rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return (a - b).pow(2).mean().sqrt().item() / max(b.pow(2).mean().sqrt().item(), 1e-6)


def _err_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return (a - b).abs().mean().item() / max(b.abs().mean().item(), 1e-6)


def _assert_close(actual: torch.Tensor, ref: torch.Tensor, name: str, *, rm, mx=None, er=None) -> None:
    """rel_rmse (relative L2, == FLA's assert_close metric) is the primary bound. rel_max
    (mx) and err_ratio (er) are optional extra guards — used for the CP-vs-serial diagnostic,
    not the FLA check (the SM90 bf16 worst element vs FLA is intrinsic noise, not a bar)."""
    rrmse = _rel_rmse(actual, ref)
    assert rrmse < rm, f"{name}: rel_rmse {rrmse:.2e} >= {rm}"
    if mx is not None:
        rmax = _rel_max(actual, ref)
        assert rmax < mx, f"{name}: rel_max {rmax:.2e} >= {mx}"
    if er is not None:
        rerr = _err_ratio(actual, ref)
        assert rerr < er, f"{name}: err_ratio {rerr:.2e} >= {er}"


def _assert_cp_matches(actual: torch.Tensor, ref: torch.Tensor, name: str) -> None:
    """CP vs serial diagnostic (2-metric; FLA is the primary ground-truth oracle below)."""
    _assert_close(actual, ref, name, mx=TOL_MAX, rm=TOL_RMSE)


def _run_serial(q, k, v, g, beta, init, want_final, cu=None, transposed=False):
    out = torch.empty_like(v)
    fin = (
        torch.empty(
            (cu.numel() - 1 if cu is not None else q.shape[0], H, D, D),
            dtype=torch.float32,
            device=q.device,
        )
        if want_final
        else None
    )
    flash_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        scale=SCALE,
        out=out,
        A_log=_run_serial.A_log,
        dt_bias=_run_serial.dt_bias,
        lower_bound=LB,
        initial_state=init,
        final_state=fin,
        cu_seqlens=cu,
        state_transposed=transposed,
    )
    return out, fin


def _run_cp(q, k, v, g, beta, init, want_final, cu=None, transposed=False, s_split=4):
    out = torch.empty_like(v)
    fin = (
        torch.empty(
            (cu.numel() - 1 if cu is not None else q.shape[0], H, D, D),
            dtype=torch.float32,
            device=q.device,
        )
        if want_final
        else None
    )
    intracard_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=SCALE,
        out=out,
        A_log=_run_cp.A_log,
        dt_bias=_run_cp.dt_bias,
        lower_bound=LB,
        initial_state=init,
        final_state=fin,
        cu_seqlens=cu,
        state_transposed=transposed,
        s_split=s_split,
    )
    return out, fin


# ---------------------------------------------------------------------------
# Merge unit test (no kernels): right-multiply bhvk convention
# ---------------------------------------------------------------------------
def _merge_carries_ref(out, m_seg, b_seg, per_seq, init):
    """Pure-PyTorch reference: sequential prefix scan over carry recurrence."""
    for s, (first, n_seg) in enumerate(per_seq):
        carry = init[s].clone()
        for i in range(first, first + n_seg):
            out[i] = carry
            carry = torch.baddbmm(b_seg[i], carry, m_seg[i])
    return out


def test_merge_unit():
    torch.manual_seed(1)
    S, Hh = 6, 3
    per_seq = [(0, 4), (4, 2)]  # two sequences: 4 + 2 segments
    m_seg = torch.randn(S, Hh, 16, 16, dtype=torch.float64)
    b_seg = torch.randn(S, Hh, 16, 16, dtype=torch.float64)
    init = torch.randn(2, Hh, 16, 16, dtype=torch.float64)

    carries = _merge_carries_ref(torch.empty_like(b_seg), m_seg, b_seg, per_seq, init)

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

    _assert_cp_matches(out_cp, out_ref, "o")
    _assert_cp_matches(fin_cp, fin_ref, "ht")


@needs_cuda
def test_cp_matches_serial_fixed_b2_with_state():
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(2, 1024, seed=3)
    _run_serial.A_log = _run_cp.A_log = A_log
    _run_serial.dt_bias = _run_cp.dt_bias = dt_bias
    init = torch.randn(2, H, D, D, dtype=torch.float32, device="cuda")

    out_ref, fin_ref = _run_serial(q, k, v, g, beta, init, True)
    out_cp, fin_cp = _run_cp(q, k, v, g, beta, init, True, s_split=4)

    _assert_cp_matches(out_cp, out_ref, "o")
    _assert_cp_matches(fin_cp, fin_ref, "ht")


@needs_cuda
def test_cp_matches_serial_varlen():
    torch.manual_seed(7)
    lens = [1024, 512, 2048, 256]
    T = sum(lens)
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(1, T, seed=7)
    _run_serial.A_log = _run_cp.A_log = A_log
    _run_serial.dt_bias = _run_cp.dt_bias = dt_bias
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
    init = torch.randn(len(lens), H, D, D, dtype=torch.float32, device="cuda")

    out_ref, fin_ref = _run_serial(q, k, v, g, beta, init, True, cu=cu)
    out_cp, fin_cp = _run_cp(q, k, v, g, beta, init, True, cu=cu, s_split=4)

    _assert_cp_matches(out_cp, out_ref, "o")
    _assert_cp_matches(fin_cp, fin_ref, "ht")


@needs_cuda
def test_cp_matches_serial_state_transposed():
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(1, 1024, seed=11)
    _run_serial.A_log = _run_cp.A_log = A_log
    _run_serial.dt_bias = _run_cp.dt_bias = dt_bias
    init = torch.randn(1, H, D, D, dtype=torch.float32, device="cuda")

    out_ref, fin_ref = _run_serial(q, k, v, g, beta, init, True, transposed=True)
    out_cp, fin_cp = _run_cp(q, k, v, g, beta, init, True, transposed=True, s_split=4)

    _assert_cp_matches(out_cp, out_ref, "o")
    _assert_cp_matches(fin_cp, fin_ref, "ht")


@needs_cuda
def test_cp_no_final_state():
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(1, 512, seed=13)
    _run_serial.A_log = _run_cp.A_log = A_log
    _run_serial.dt_bias = _run_cp.dt_bias = dt_bias

    out_ref, _ = _run_serial(q, k, v, g, beta, None, False)
    out_cp, _ = _run_cp(q, k, v, g, beta, None, False, s_split=2)

    _assert_cp_matches(out_cp, out_ref, "o")


# ---------------------------------------------------------------------------
# Partial-tile (non-CHUNK-aligned) inputs — must run through CP, not error.
# Oracle = serial flash_kda_fwd, which handles non-aligned via native varlen
# masking (a different mechanism than CP's pad-before-segment, so agreement is
# a real cross-check).
# ---------------------------------------------------------------------------
@needs_cuda
@pytest.mark.parametrize(
    "lens",
    [
        [1024, 1, 63, 65, 129],  # several non-aligned seqs (partial tiles)
        [28679, 4096],  # long non-aligned head -> splits, partial tile
        [40007],  # single long non-aligned -> splits
        [32768, 100],  # aligned head + short non-aligned tail
    ],
)
def test_cp_matches_serial_varlen_nonaligned(lens):
    T = sum(lens)
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(1, T, seed=5)
    _run_serial.A_log = _run_cp.A_log = A_log
    _run_serial.dt_bias = _run_cp.dt_bias = dt_bias
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")

    out_ref, fin_ref = _run_serial(q, k, v, g, beta, None, True, cu=cu)
    out_cp, fin_cp = _run_cp(q, k, v, g, beta, None, True, cu=cu, s_split=8)

    _assert_cp_matches(out_cp, out_ref, "o")
    _assert_cp_matches(fin_cp, fin_ref, "ht")


@needs_cuda
@pytest.mark.parametrize("T", [100, 4100, 8197])
def test_cp_matches_serial_dense_nonaligned(T):
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(1, T, seed=9)
    _run_serial.A_log = _run_cp.A_log = A_log
    _run_serial.dt_bias = _run_cp.dt_bias = dt_bias

    out_ref, fin_ref = _run_serial(q, k, v, g, beta, None, True)
    out_cp, fin_cp = _run_cp(q, k, v, g, beta, None, True, s_split=8)

    _assert_cp_matches(out_cp, out_ref, "o")
    _assert_cp_matches(fin_cp, fin_ref, "ht")


@needs_cuda
def test_auto_router_bypasses_few_segments():
    """Too few segments/seq is not worth CP: auto disables, force still runs."""
    from cula.ops.kda.policy import sm90_intracard_cp_decision
    from cula.ops.kda.sm90.cp.plan import CHUNK as CP_CHUNK
    from cula.ops.kda.sm90.cp.plan import auto_plan_segments

    q = torch.empty(1, 8192, H, D, device="cuda", dtype=torch.bfloat16)
    _, _, per = auto_plan_segments(q.device, [8192 // CP_CHUNK], H)
    max_seg = max(ns for _, ns in per)
    if not (2 < max_seg <= 4):
        pytest.skip(f"planned into {max_seg} segments; the 3-4 'not beneficial' band is not hit here")
    assert sm90_intracard_cp_decision(q, None, None, "auto").enabled is False  # auto bypasses
    assert sm90_intracard_cp_decision(q, None, None, True).enabled is True  # force still runs


# ---------------------------------------------------------------------------
# CP vs FLA (ground truth) — forced CP via the public entry on long sequences.
# Mirrors test_kda_sm90_prefill_vs_fla.py's convention (beta post-sigmoid, VK state
# transpose, FLA flags); use_intracard_cp=True forces the CP path.
# ---------------------------------------------------------------------------
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
    _assert_close(cp_o, ref_o, "o", rm=TOL_FLA_RMSE)
    _assert_close(cp_ht_vk.transpose(-2, -1), ref_ht, "ht", rm=TOL_FLA_RMSE)


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
# Determinism (cula-kernel-wiki §1.2): the CP path is warp-specialized
# (pre_scan/merge) with mbarrier + SMEM reuse — exactly the class that needs a
# bit-exact guard against probabilistic races. (Wiki targets 10K+ for deep
# race-hunting; ITERS here is a CI-cost compromise — raise it to chase a
# suspected timing-sensitive bug.)
# ---------------------------------------------------------------------------
_DETERMINISM_ITERS = 10000


@needs_cuda
def test_cp_determinism():
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(1, 4096, seed=17)
    _run_cp.A_log, _run_cp.dt_bias = A_log, dt_bias
    out0, fin0 = _run_cp(q, k, v, g, beta, None, True, s_split=8)
    assert not (out0.isnan().any() or out0.isinf().any()), "CP output has NaN/Inf"
    for i in range(_DETERMINISM_ITERS):
        out, fin = _run_cp(q, k, v, g, beta, None, True, s_split=8)
        assert torch.equal(out, out0), f"non-deterministic out at iter {i}"
        assert torch.equal(fin, fin0), f"non-deterministic ht at iter {i}"


@needs_cuda
def test_cp_determinism_varlen():
    # Regression guard for the varlen ws_beta under-allocation bug (docs/
    # kda_sm90_cp_varlen_race.md): ws_beta must be sized total_tiles*CHUNK*H, not
    # T_total*H — for varlen total_tiles*CHUNK > T_total, so the smaller size let
    # K1's ws_beta stores run out of bounds into adjacent allocator memory,
    # nondeterministically corrupting downstream ws_gt/ws_inv -> o. test_cp_determinism
    # above only covers dense (where total_tiles*CHUNK == T_total), which is why this
    # went unnoticed. Many short varlen segments (s_split=8 on a 64-tile head)
    # maximize the OOB tile indices; revert the size fix and this fails fast.
    lens = [1024, 1, 63, 65, 129]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(1, sum(lens), seed=5)
    _run_cp.A_log, _run_cp.dt_bias = A_log, dt_bias
    out0, fin0 = _run_cp(q, k, v, g, beta, None, True, cu=cu, s_split=8)
    assert not (out0.isnan().any() or out0.isinf().any()), "CP varlen output has NaN/Inf"
    for i in range(_DETERMINISM_ITERS):
        out, fin = _run_cp(q, k, v, g, beta, None, True, cu=cu, s_split=8)
        assert torch.equal(out, out0), f"non-deterministic varlen out at iter {i}"
        assert torch.equal(fin, fin0), f"non-deterministic varlen ht at iter {i}"

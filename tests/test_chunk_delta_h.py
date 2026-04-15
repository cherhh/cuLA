#!/usr/bin/env python3
# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for ChunkDeltaRuleFwdH CuTe DSL kernel.
Tests correctness against FLA's Triton reference (chunk_gated_delta_rule_fwd_h).
"""

import argparse
import os
import sys

import pytest
import torch

# ─── FLA reference ───
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as fla_fwd_h

# ─── CuTe DSL kernel (via importlib to avoid cula __init__ requiring cudac) ───
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "chunk_delta_h", os.path.join(os.path.dirname(__file__), "..", "cula", "ops", "chunk_delta_h.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
chunk_gated_delta_rule_fwd_h = _mod.chunk_gated_delta_rule_fwd_h

# ─── Intracard CP ───
from cula.ops.cp.chunk_delta_h import (
    compute_subseq_len,
    intracard_fwd_h,
    prepare_subseq_cu_seqlens,
    _precompute_intracard_indices,
    SplitSeqInfo,
)
from cula.ops.cp.merge import merge_fwd


BT = 64
device = "cuda"


def run_fla_ref(k, w, u, g=None, gk=None, initial_state=None, output_final_state=False, save_new_value=True, cu_seqlens=None):
    """Call FLA's Triton kernel as reference."""
    return fla_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=BT,
        save_new_value=save_new_value,
        cu_seqlens=cu_seqlens,
    )


def run_cute_dsl(k, w, u, g=None, gk=None, initial_state=None, output_final_state=False, save_new_value=True, cu_seqlens=None):
    """Call CuTe DSL kernel wrapper (FLA-compatible API) and return (h_out, v_new, ht)."""
    return chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=BT,
        save_new_value=save_new_value,
        cu_seqlens=cu_seqlens,
    )


# ===================== Pytest parametrized tests =====================


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("H", [1, 4])
@pytest.mark.parametrize("T", [64, 128, 256])
@pytest.mark.parametrize("K", [128])
@pytest.mark.parametrize("V", [128])
@pytest.mark.parametrize("use_gk", [False, True])
@pytest.mark.parametrize("use_h0", [False, True])
def test_h_against_fla(B, H, T, K, V, use_gk, use_h0):
    """Test CuTe DSL h_out matches FLA's Triton kernel."""
    torch.manual_seed(42)

    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    w = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    u = torch.randn(B, T, H, V, dtype=torch.bfloat16, device=device) * 0.1

    h0 = None
    if use_h0:
        h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device) * 0.01

    gk_val = None
    if use_gk:
        gk_val = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
        gk_val = -torch.abs(gk_val).cumsum(dim=1)

    # FLA reference
    ref_h, ref_vnew, ref_ht = run_fla_ref(
        k,
        w,
        u,
        gk=gk_val,
        initial_state=h0,
        output_final_state=use_h0,
        save_new_value=True,
    )

    # CuTe DSL kernel
    our_h, our_vnew, our_ht = run_cute_dsl(
        k,
        w,
        u,
        gk=gk_val,
        initial_state=h0,
        output_final_state=use_h0,
        save_new_value=True,
    )

    # Compare h_out: FLA returns [B, NT, H, K, V], ours is [B, NT, H, K, V]
    torch.testing.assert_close(
        our_h.float(),
        ref_h.float(),
        atol=1e-2,
        rtol=1e-2,
        msg=f"h_out mismatch B={B} H={H} T={T} gk={use_gk} h0={use_h0}",
    )

    # Compare v_new
    if ref_vnew is not None and our_vnew is not None:
        torch.testing.assert_close(
            our_vnew.float(),
            ref_vnew.float(),
            atol=1e-2,
            rtol=1e-2,
            msg=f"v_new mismatch B={B} H={H} T={T} gk={use_gk} h0={use_h0}",
        )

    # Compare ht (final state)
    if use_h0 and ref_ht is not None and our_ht is not None:
        torch.testing.assert_close(
            our_ht.float(),
            ref_ht.float(),
            atol=1e-2,
            rtol=1e-2,
            msg=f"ht mismatch B={B} H={H} T={T} gk={use_gk} h0={use_h0}",
        )


@pytest.mark.parametrize(
    "B,T,H,K,V",
    [
        (1, 64, 1, 128, 128),
        (2, 128, 4, 128, 128),
        (4, 512, 4, 128, 128),
    ],
)
def test_vnew_no_gating(B, T, H, K, V):
    """Test v_new output without gating matches FLA."""
    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    w = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    u = torch.randn(B, T, H, V, dtype=torch.bfloat16, device=device) * 0.1

    ref_h, ref_vnew, _ = run_fla_ref(k, w, u, save_new_value=True)
    our_h, our_vnew, _ = run_cute_dsl(k, w, u, save_new_value=True)

    torch.testing.assert_close(
        our_vnew.float(),
        ref_vnew.float(),
        atol=1e-2,
        rtol=1e-2,
        msg=f"v_new no-gating mismatch B={B} T={T} H={H}",
    )


# ===================== Varlen pytest tests =====================


def _make_varlen_inputs(seq_lens, H, K, V, use_gk=False, use_h0=False, seed=42):
    """Create varlen-packed tensors in FLA convention: [1, T_total, H, D]."""
    T_total = sum(seq_lens)
    num_seqs = len(seq_lens)
    cu_seqlens_list = [0]
    for sl in seq_lens:
        cu_seqlens_list.append(cu_seqlens_list[-1] + sl)

    torch.manual_seed(seed)
    k = torch.randn(1, T_total, H, K, dtype=torch.bfloat16, device=device) * 0.1
    w = torch.randn(1, T_total, H, K, dtype=torch.bfloat16, device=device) * 0.1
    u = torch.randn(1, T_total, H, V, dtype=torch.bfloat16, device=device) * 0.1

    gk_val = None
    if use_gk:
        # Per-sequence cumsum (reset at sequence boundaries)
        gk_val = torch.zeros(1, T_total, H, K, dtype=torch.float32, device=device)
        for i in range(num_seqs):
            bos, eos = cu_seqlens_list[i], cu_seqlens_list[i + 1]
            seg = torch.randn(1, eos - bos, H, K, dtype=torch.float32, device=device) * 0.1
            gk_val[:, bos:eos] = -torch.abs(seg).cumsum(dim=1)

    h0 = None
    if use_h0:
        h0 = torch.randn(num_seqs, H, K, V, dtype=torch.float32, device=device) * 0.01

    cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)
    return k, w, u, gk_val, h0, cu_seqlens


@pytest.mark.parametrize(
    "seq_lens",
    [
        [128, 128],
        [50, 192, 100],
        [33, 128, 200, 95],
    ],
)
@pytest.mark.parametrize("H", [1, 4])
@pytest.mark.parametrize("use_gk", [False, True])
@pytest.mark.parametrize("use_h0", [False, True])
def test_varlen_against_fla(seq_lens, H, use_gk, use_h0):
    """Test varlen CuTe DSL h_out/v_new/ht matches FLA's Triton kernel."""
    K, V = 128, 128
    k, w, u, gk_val, h0, cu_seqlens = _make_varlen_inputs(
        seq_lens,
        H,
        K,
        V,
        use_gk=use_gk,
        use_h0=use_h0,
    )

    ref_h, ref_vnew, ref_ht = run_fla_ref(
        k,
        w,
        u,
        gk=gk_val,
        initial_state=h0,
        output_final_state=use_h0,
        save_new_value=True,
        cu_seqlens=cu_seqlens,
    )
    our_h, our_vnew, our_ht = run_cute_dsl(
        k,
        w,
        u,
        gk=gk_val,
        initial_state=h0,
        output_final_state=use_h0,
        save_new_value=True,
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(
        our_h.float(),
        ref_h.float(),
        atol=1e-2,
        rtol=1e-2,
        msg=f"varlen h_out mismatch seqs={seq_lens} H={H} gk={use_gk} h0={use_h0}",
    )
    if ref_vnew is not None and our_vnew is not None:
        torch.testing.assert_close(
            our_vnew.float(),
            ref_vnew.float(),
            atol=1e-2,
            rtol=1e-2,
            msg=f"varlen v_new mismatch seqs={seq_lens} H={H} gk={use_gk} h0={use_h0}",
        )
    if use_h0 and ref_ht is not None and our_ht is not None:
        torch.testing.assert_close(
            our_ht.float(),
            ref_ht.float(),
            atol=1e-2,
            rtol=1e-2,
            msg=f"varlen ht mismatch seqs={seq_lens} H={H} gk={use_gk} h0={use_h0}",
        )


def test_varlen_vs_nonvarlen():
    """Test that varlen with a single sequence matches non-varlen output."""
    H, K, V = 2, 128, 128
    T = 256

    torch.manual_seed(42)
    k = torch.randn(1, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    w = torch.randn(1, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    u = torch.randn(1, T, H, V, dtype=torch.bfloat16, device=device) * 0.1

    # Non-varlen
    h_nv, vnew_nv, _ = run_cute_dsl(k, w, u, save_new_value=True)

    # Varlen with single sequence (should be identical)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)
    h_vl, vnew_vl, _ = run_cute_dsl(k, w, u, save_new_value=True, cu_seqlens=cu_seqlens)

    torch.testing.assert_close(
        h_nv.float(),
        h_vl.float(),
        atol=1e-6,
        rtol=1e-6,
        msg="varlen vs non-varlen h_out mismatch for single sequence",
    )
    torch.testing.assert_close(
        vnew_nv.float(),
        vnew_vl.float(),
        atol=1e-6,
        rtol=1e-6,
        msg="varlen vs non-varlen v_new mismatch for single sequence",
    )


# ===================== Intracard CP unit tests =====================


def test_cp_stage0_compute_subseq_len():
    """Test compute_subseq_len produces sane values."""
    # Short sequence: no splitting
    ssl = compute_subseq_len(256, 208, 64, 64)
    assert ssl >= 256, f"Short seq should not split: {ssl}"

    # Long sequence: should split
    ssl = compute_subseq_len(131072, 208, 64, 64)
    assert ssl < 131072, f"Long seq should split: {ssl}"
    assert ssl % 64 == 0, f"Must be chunk_size aligned: {ssl}"


def test_cp_stage0_prepare_subseq_cu_seqlens():
    """Test sequence splitting."""
    # Single long sequence
    T = 65536
    cu = torch.tensor([0, T], dtype=torch.int64)
    ssl = compute_subseq_len(T, 208, 4, 64)

    boundaries, split_info, total_subseqs = prepare_subseq_cu_seqlens(cu, ssl, 64)
    assert split_info, "Long seq should be split"
    assert len(boundaries) == total_subseqs + 1
    assert boundaries[0] == 0
    assert boundaries[-1] == T

    # Short sequence: no split
    cu_short = torch.tensor([0, 128], dtype=torch.int64)
    boundaries, split_info, total = prepare_subseq_cu_seqlens(cu_short, ssl, 64)
    assert not split_info, "Short seq should not be split"


def test_cp_stage0_precompute_indices():
    """Test index precomputation."""
    split_info = SplitSeqInfo(
        split_seq_ids=[0],
        start_subseq_idx=[0],
        num_subseqs=[4],
    )
    cu_values = [0, 1024, 2048, 3072, 4096]

    result = _precompute_intracard_indices(split_info, cu_values, N_orig=1)
    (non_first, first_idx, last_idx,
     num_nf, seq_starts, seq_counts, init_off) = result

    assert non_first == [1, 2, 3]
    assert first_idx == [0]
    assert last_idx == [3]
    assert num_nf == 3
    assert seq_starts == [0]
    assert seq_counts == [4]
    assert init_off == [0, 3]


def test_cp_merge_kernel():
    """Test merge standalone with a known case."""
    torch.manual_seed(42)
    S = 4  # 4 sub-seqs from 1 original seq
    H = 2
    K, V = 128, 128

    # Random hm
    hm = torch.randn(S, H, K, V + K, device=device, dtype=torch.float32) * 0.01
    he = hm[:, :, :, :V]
    m_mat = hm[:, :, :, V:]

    # Reference: sequential prefix scan
    h_ref_list = []
    b_h = torch.zeros(H, K, V, device=device, dtype=torch.float32)
    for s in range(S):
        b_h = torch.bmm(m_mat[s], b_h) + he[s]
        if s < S - 1:
            h_ref_list.append(b_h.clone())
    h_ref = torch.stack(h_ref_list)  # [3, H, K, V]

    # merge_fwd
    num_non_first = S - 1
    seq_starts = [0]
    seq_counts = [S]
    init_offsets = [0, num_non_first]
    split_seq_ids = [0]

    h_out = merge_fwd(
        hm=hm,
        seq_starts=seq_starts,
        seq_counts=seq_counts,
        init_offsets=init_offsets,
        split_seq_ids=split_seq_ids,
        h0=None,
        num_non_first=num_non_first,
    )

    rel = (h_out - h_ref).abs().max().item() / (h_ref.abs().max().item() + 1e-8)
    assert rel < 5e-3, f"merge relative error too large: {rel}"


# ===================== Intracard CP E2E tests =====================


@pytest.mark.parametrize("T", [32768, 65536])
@pytest.mark.parametrize("H", [4, 8])
@pytest.mark.parametrize("use_gk", [True, False])
def test_cp_e2e_single_seq(T, H, use_gk):
    """E2E: intracard_fwd_h vs baseline for a single long sequence."""
    torch.manual_seed(42)
    K, V = 128, 128

    k = torch.randn(1, T, H, K, device=device, dtype=torch.bfloat16) * 0.02
    w = torch.randn(1, T, H, K, device=device, dtype=torch.bfloat16) * 0.02
    u = torch.randn(1, T, H, V, device=device, dtype=torch.bfloat16) * 0.02
    gk = (torch.randn(1, T, H, K, device=device, dtype=torch.float32) * 0.01) if use_gk else None

    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int64)

    h_base, v_base, _ = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, gk=gk,
        initial_state=None, output_final_state=False,
        chunk_size=BT, save_new_value=True,
        cu_seqlens=cu_seqlens,
    )
    h_cp, v_cp, _ = intracard_fwd_h(
        k=k, w=w, u=u, gk=gk,
        initial_state=None, output_final_state=False,
        chunk_size=BT, save_new_value=True,
        cu_seqlens=cu_seqlens,
    )

    h_rel = (h_cp.float() - h_base.float()).abs().max().item() / (h_base.float().abs().max().item() + 1e-8)
    v_rel = (v_cp.float() - v_base.float()).abs().max().item() / (v_base.float().abs().max().item() + 1e-8) if v_cp is not None else 0
    assert h_rel < 0.05, f"h relative error too large: {h_rel}"
    assert v_rel < 0.05, f"v_new relative error too large: {v_rel}"


def test_cp_e2e_multi_seq():
    """E2E: mixed-length batch with long + short sequences."""
    torch.manual_seed(123)
    K, V, H = 128, 128, 4
    seq_lens = [32768, 256, 32768, 128]
    T = sum(seq_lens)

    k = torch.randn(1, T, H, K, device=device, dtype=torch.bfloat16) * 0.02
    w = torch.randn(1, T, H, K, device=device, dtype=torch.bfloat16) * 0.02
    u = torch.randn(1, T, H, V, device=device, dtype=torch.bfloat16) * 0.02

    cu = [0]
    for sl in seq_lens:
        cu.append(cu[-1] + sl)
    cu_seqlens = torch.tensor(cu, device=device, dtype=torch.int64)

    h_base, _, _ = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u,
        initial_state=None, output_final_state=False,
        chunk_size=BT, save_new_value=True,
        cu_seqlens=cu_seqlens,
    )
    h_cp, _, _ = intracard_fwd_h(
        k=k, w=w, u=u,
        initial_state=None, output_final_state=False,
        chunk_size=BT, save_new_value=True,
        cu_seqlens=cu_seqlens,
    )

    h_rel = (h_cp.float() - h_base.float()).abs().max().item() / (h_base.float().abs().max().item() + 1e-8)
    assert h_rel < 0.05, f"h relative error too large: {h_rel}"


def test_cp_e2e_with_h0():
    """E2E: with initial state h0 and output_final_state."""
    torch.manual_seed(77)
    K, V, H = 128, 128, 4
    T = 32768
    N = 1

    k = torch.randn(1, T, H, K, device=device, dtype=torch.bfloat16) * 0.02
    w = torch.randn(1, T, H, K, device=device, dtype=torch.bfloat16) * 0.02
    u = torch.randn(1, T, H, V, device=device, dtype=torch.bfloat16) * 0.02
    h0 = torch.randn(N, H, K, V, device=device, dtype=torch.float32) * 0.01

    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int64)

    h_base, _, ht_base = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u,
        initial_state=h0, output_final_state=True,
        chunk_size=BT, save_new_value=False,
        cu_seqlens=cu_seqlens,
    )
    h_cp, _, ht_cp = intracard_fwd_h(
        k=k, w=w, u=u,
        initial_state=h0, output_final_state=True,
        chunk_size=BT, save_new_value=False,
        cu_seqlens=cu_seqlens,
    )

    h_rel = (h_cp.float() - h_base.float()).abs().max().item() / (h_base.float().abs().max().item() + 1e-8)
    ht_rel = (ht_cp.float() - ht_base.float()).abs().max().item() / (ht_base.float().abs().max().item() + 1e-8) if ht_cp is not None else 0
    assert h_rel < 0.05, f"h relative error too large: {h_rel}"
    assert ht_rel < 0.05, f"ht relative error too large: {ht_rel}"


# ===================== Manual test runner =====================


def run_correctness_tests():
    """Run correctness tests manually (not pytest)."""
    configs = [
        (1, 64, 1, 128, 128, False, False, "minimal"),
        (1, 128, 1, 128, 128, True, False, "gk only"),
        (1, 128, 1, 128, 128, False, True, "h0 only"),
        (1, 128, 1, 128, 128, True, True, "gk + h0"),
        (2, 256, 4, 128, 128, True, True, "multi-batch multi-head"),
        (4, 512, 4, 128, 128, False, False, "larger no gating"),
        (4, 1024, 8, 128, 128, True, True, "large with gk + h0"),
    ]

    all_passed = True
    for B, T, H, K, V, use_gk, use_h0, desc in configs:
        print(f"\n--- {desc} (B={B} T={T} H={H} K={K} V={V} gk={use_gk} h0={use_h0}) ---")
        torch.manual_seed(42)

        k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
        w = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
        u = torch.randn(B, T, H, V, dtype=torch.bfloat16, device=device) * 0.1

        h0_val = torch.randn(B, H, K, V, dtype=torch.float32, device=device) * 0.01 if use_h0 else None
        gk_val = None
        if use_gk:
            gk_val = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
            gk_val = -torch.abs(gk_val).cumsum(dim=1)

        try:
            ref_h, ref_vnew, ref_ht = run_fla_ref(
                k,
                w,
                u,
                gk=gk_val,
                initial_state=h0_val,
                output_final_state=use_h0,
                save_new_value=True,
            )
            our_h, our_vnew, our_ht = run_cute_dsl(
                k,
                w,
                u,
                gk=gk_val,
                initial_state=h0_val,
                output_final_state=use_h0,
                save_new_value=True,
            )

            h_diff = (our_h.float() - ref_h.float()).abs().max().item()
            vnew_diff = (our_vnew.float() - ref_vnew.float()).abs().max().item() if ref_vnew is not None else 0
            ht_diff = 0
            if use_h0 and ref_ht is not None and our_ht is not None:
                ht_diff = (our_ht.float() - ref_ht.float()).abs().max().item()

            passed = h_diff < 0.02 and vnew_diff < 0.02 and ht_diff < 0.02
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] h_diff={h_diff:.6f} vnew_diff={vnew_diff:.6f} ht_diff={ht_diff:.6f}")
            all_passed = all_passed and passed
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False

    print(f"\n{'=' * 50}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


def run_benchmark(B=4, T=4096, H=64, K=128, V=128, num_iters=20):
    """Benchmark CuTe DSL vs FLA Triton."""
    print(f"\n=== Benchmark: B={B}, T={T}, H={H}, K={K}, V={V} ===")

    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    w = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    u = torch.randn(B, T, H, V, dtype=torch.bfloat16, device=device) * 0.1
    gk = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
    gk = -torch.abs(gk).cumsum(dim=1)
    h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device) * 0.01

    # --- CuTe DSL ---
    # Warmup (triggers compilation)
    chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        gk=gk,
        initial_state=h0,
        output_final_state=True,
        chunk_size=BT,
        save_new_value=True,
    )
    torch.cuda.synchronize()

    for _ in range(5):
        chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            gk=gk,
            initial_state=h0,
            output_final_state=True,
            chunk_size=BT,
            save_new_value=True,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            gk=gk,
            initial_state=h0,
            output_final_state=True,
            chunk_size=BT,
            save_new_value=True,
        )
    end.record()
    torch.cuda.synchronize()
    cute_ms = start.elapsed_time(end) / num_iters

    # --- FLA Triton ---
    for _ in range(5):
        fla_fwd_h(k=k, w=w, u=u, gk=gk, initial_state=h0, output_final_state=True, chunk_size=BT, save_new_value=True)
    torch.cuda.synchronize()

    start.record()
    for _ in range(num_iters):
        fla_fwd_h(k=k, w=w, u=u, gk=gk, initial_state=h0, output_final_state=True, chunk_size=BT, save_new_value=True)
    end.record()
    torch.cuda.synchronize()
    fla_ms = start.elapsed_time(end) / num_iters

    print(f"  CuTe DSL: {cute_ms:.3f} ms")
    print(f"  FLA:      {fla_ms:.3f} ms")
    print(f"  Speedup:  {fla_ms / cute_ms:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="correctness", choices=["correctness", "benchmark", "both"])
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--V", type=int, default=128)
    args = parser.parse_args()

    if args.test in ("correctness", "both"):
        run_correctness_tests()

    if args.test in ("benchmark", "both"):
        run_benchmark(B=args.B, T=args.T, H=args.H, K=args.K, V=args.V)

# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for the CuteDSL FlashKDA K1 full kernel."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs CUDA",
)


def _make_inputs(B: int, T: int, H: int, *, seed: int = 0):
    from cula.ops.flashkda.prefill import D as HEAD_DIM

    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, T, H, HEAD_DIM, generator=g, device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn(B, T, H, HEAD_DIM, generator=g, device="cuda", dtype=torch.bfloat16) * 0.5
    g_pre = torch.randn(B, T, H, HEAD_DIM, generator=g, device="cuda", dtype=torch.bfloat16) * 0.1
    beta = torch.randn(B, T, H, generator=g, device="cuda", dtype=torch.bfloat16) * 0.1
    A_log = torch.randn(H, generator=g, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H, HEAD_DIM, generator=g, device="cuda", dtype=torch.float32) * 0.1
    return q, k, g_pre, beta, A_log, dt_bias


def test_k1():
    """K1 pipeline producing all 6 K2-ready workspace tensors."""
    from cula.ops.flashkda.k1 import CHUNK, D, launch_k1

    B, T, H = 1, 32, 2
    scale = 0.125
    gate_scale = -1.0
    q, k, g_pre, beta, A_log, dt_bias = _make_inputs(B, T, H, seed=4)
    total_tiles = (B * T) // CHUNK
    beta_flat = beta.view(B * T, H).permute(1, 0).contiguous().view(-1)

    n_qk = total_tiles * H * CHUNK * D
    n_cc = total_tiles * H * CHUNK * CHUNK
    ws_qd = torch.zeros(n_qk, dtype=torch.bfloat16, device="cuda")
    ws_kd = torch.zeros_like(ws_qd)
    ws_kr = torch.zeros_like(ws_qd)
    ws_gt = torch.zeros(total_tiles * H * D, dtype=torch.float32, device="cuda")
    ws_inv = torch.zeros(n_cc, dtype=torch.bfloat16, device="cuda")
    ws_mqk = torch.zeros_like(ws_inv)

    launch_k1(
        q,
        k,
        g_pre,
        A_log,
        dt_bias,
        beta_flat,
        scale,
        gate_scale,
        ws_qd,
        ws_kd,
        ws_kr,
        ws_gt,
        ws_inv,
        ws_mqk,
    )
    torch.cuda.synchronize()

    # references
    q_tm = q.view(B * T, H, D).view(total_tiles, CHUNK, H, D).permute(2, 0, 1, 3).contiguous().float()
    k_tm = k.view(B * T, H, D).view(total_tiles, CHUNK, H, D).permute(2, 0, 1, 3).contiguous().float()
    g_tm = g_pre.view(B * T, H, D).view(total_tiles, CHUNK, H, D).permute(2, 0, 1, 3).contiguous().float()
    beta_tm = beta.view(B * T, H).permute(1, 0).contiguous().view(H, total_tiles, CHUNK).float()

    q_l2 = (q_tm * (q_tm.pow(2).sum(-1, keepdim=True) + 1.0e-6).rsqrt()).to(torch.bfloat16).float()
    k_l2 = (k_tm * (k_tm.pow(2).sum(-1, keepdim=True) + 1.0e-6).rsqrt()).to(torch.bfloat16).float()
    A_exp = torch.exp(A_log.float())
    g_act = gate_scale * torch.sigmoid(A_exp.view(H, 1, 1, 1) * (g_tm + dt_bias.view(H, 1, 1, D)))
    g_cs = torch.cumsum(g_act, dim=2)
    exp_pos = torch.exp(g_cs)
    inv_pos = 1.0 / exp_pos
    g_total = g_cs[:, :, -1, :]
    rest = torch.exp(g_total).unsqueeze(2) * inv_pos

    qd_ref = (q_l2 * exp_pos * scale).to(torch.bfloat16)
    kd_ref = (k_l2 * exp_pos).to(torch.bfloat16)
    ki_ref = (k_l2 * inv_pos).to(torch.bfloat16)
    kr_ref = (k_l2 * rest).to(torch.bfloat16)
    gt_ref = torch.exp(g_total)

    L_full = torch.einsum("htmk,htnk->htmn", kd_ref.float(), ki_ref.float())
    Mqk_full = torch.einsum("htmk,htnk->htmn", qd_ref.float(), ki_ref.float())
    tril_mask = torch.tril(torch.ones(CHUNK, CHUNK, device="cuda"), diagonal=-1)
    L_masked = L_full * tril_mask * torch.sigmoid(beta_tm).view(H, total_tiles, CHUNK, 1)
    keep_mqk = torch.tril(torch.ones(CHUNK, CHUNK, device="cuda"), diagonal=0)
    Mqk_masked = Mqk_full * keep_mqk
    eye = torch.eye(CHUNK, device="cuda").view(1, 1, CHUNK, CHUNK)
    INV_ref = torch.linalg.inv(eye + L_masked)

    def rel(name, got_flat, ref, view_shape):
        got = got_flat.view(*view_shape).float()
        d = (got - ref.float()).abs()
        s = ref.float().abs().max().clamp_min(1.0)
        return (d.max() / s).item(), name

    qk_shape = (H, total_tiles, CHUNK, D)
    cc_shape = (H, total_tiles, CHUNK, CHUNK)
    gt_shape = (H, total_tiles, D)
    diffs = [
        rel("qd", ws_qd, qd_ref, qk_shape),
        rel("kd", ws_kd, kd_ref, qk_shape),
        rel("kr", ws_kr, kr_ref, qk_shape),
        rel("gt", ws_gt, gt_ref, gt_shape),
        rel("inv", ws_inv, INV_ref, cc_shape),
        rel("mqk", ws_mqk, Mqk_masked, cc_shape),
    ]
    msg = "  ".join(f"{n}={d:.3e}" for d, n in diffs)
    print(f"\n[k1 full] {msg}")
    for d, n in diffs:
        thresh = 5e-2 if n == "inv" else 1e-2
        assert d < thresh, f"{n}: {d}"

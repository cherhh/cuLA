# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end correctness test for CuteDSL K1 (full) + K2 (Phase A).

Pipes K1's six workspace tensors into K2 and compares the output against a
pure-torch reference that re-implements the FlashKDA recurrence math.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _torch_ref(q, k, g_pre, beta, A_log, dt_bias, scale, gate_scale):
    """Pure torch reference implementing the same math as K1 + K2."""
    B, T, H, Dh = q.shape
    CHUNK = 16
    total_tiles = (B * T) // CHUNK
    qf = q.view(B * T, H, Dh).view(total_tiles, CHUNK, H, Dh).permute(2, 0, 1, 3).contiguous().float()
    kf = k.view(B * T, H, Dh).view(total_tiles, CHUNK, H, Dh).permute(2, 0, 1, 3).contiguous().float()
    gf = g_pre.view(B * T, H, Dh).view(total_tiles, CHUNK, H, Dh).permute(2, 0, 1, 3).contiguous().float()
    vf = torch.randn  # placeholder
    # actually we get v passed in below

    # L2 norm + decay (same as K1)
    q_l2 = (qf * (qf.pow(2).sum(-1, keepdim=True) + 1e-6).rsqrt()).to(torch.bfloat16).float()
    k_l2 = (kf * (kf.pow(2).sum(-1, keepdim=True) + 1e-6).rsqrt()).to(torch.bfloat16).float()
    A_exp = torch.exp(A_log.float())
    g_act = gate_scale * torch.sigmoid(A_exp.view(H, 1, 1, 1) * (gf + dt_bias.view(H, 1, 1, Dh)))
    g_cs = torch.cumsum(g_act, dim=2)
    exp_pos = torch.exp(g_cs)
    inv_pos = 1.0 / exp_pos
    g_total = g_cs[:, :, -1, :]
    rest = torch.exp(g_total).unsqueeze(2) * inv_pos
    qd = (q_l2 * exp_pos * scale).to(torch.bfloat16).float()
    kd = (k_l2 * exp_pos).to(torch.bfloat16).float()
    ki = (k_l2 * inv_pos).to(torch.bfloat16).float()
    kr = (k_l2 * rest).to(torch.bfloat16).float()
    gt = torch.exp(g_total)  # [H, total_tiles, D]

    L = torch.einsum("htmk,htnk->htmn", kd, ki)
    Mqk = torch.einsum("htmk,htnk->htmn", qd, ki)
    tril = torch.tril(torch.ones(CHUNK, CHUNK, device=q.device), diagonal=-1)
    keep = torch.tril(torch.ones(CHUNK, CHUNK, device=q.device), diagonal=0)
    beta_tm = beta.view(B * T, H).permute(1, 0).contiguous().view(H, total_tiles, CHUNK).float()
    L = L * tril * torch.sigmoid(beta_tm).view(H, total_tiles, CHUNK, 1)
    Mqk = Mqk * keep
    eye = torch.eye(CHUNK, device=q.device).view(1, 1, CHUNK, CHUNK)
    INV = torch.linalg.inv(eye + L)
    return qd, kd, kr, gt, INV, Mqk


def _k2_torch_ref(v, beta_flat, qd, kd, kr, gt, INV, Mqk, B, T, H, Dh, seq_len):
    """Run torch K2 recurrence; return out [B,T,H,D]."""
    CHUNK = 16
    t_tiles = seq_len // CHUNK
    total_tiles = (B * T) // CHUNK
    v_tm = v.view(B * T, H, Dh).view(total_tiles, CHUNK, H, Dh).permute(2, 0, 1, 3).contiguous().float()
    beta_tm = beta_flat.view(H, B, t_tiles, CHUNK).float()
    out = torch.zeros(H, B, t_tiles, CHUNK, Dh, device=v.device)
    for b in range(B):
        for h in range(H):
            state = torch.zeros(Dh, Dh, device=v.device, dtype=torch.float32)
            for t in range(t_tiles):
                ttile = b * t_tiles + t
                v_b = v_tm[h, ttile].float()  # [CHUNK, D]
                kd_b = kd[h, ttile]
                qd_b = qd[h, ttile]
                kr_b = kr[h, ttile]
                gt_b = gt[h, ttile]  # [D]
                INV_b = INV[h, ttile]
                Mqk_b = Mqk[h, ttile]
                beta_b = beta_tm[h, b, t]  # [CHUNK]
                sig_b = torch.sigmoid(beta_b).view(CHUNK, 1)
                # use bf16 round-trip for fairness (matches kernel intermediate dtype)
                tmp_o = kd_b @ state
                u = ((v_b - tmp_o) * sig_b).to(torch.bfloat16).float()
                u = (INV_b @ u).to(torch.bfloat16).float()
                out0 = qd_b @ state
                out_t = out0 + Mqk_b @ u
                out[h, b, t] = out_t.to(torch.bfloat16).float()
                state = state * gt_b.view(Dh, 1) + kr_b.transpose(0, 1) @ u
                state = state.to(torch.bfloat16).float()
    out_user = out.permute(1, 2, 3, 0, 4).contiguous().view(B, T, H, Dh)
    return out_user


def test_k1full_k2A_e2e():
    from cula.ops.flashkda_k1 import CHUNK, D, launch_k1_full
    from cula.ops.flashkda_k2 import launch_k2_phaseA

    B, T, H = 1, 32, 2
    scale = 0.125
    gate_scale = -1.0
    g = torch.Generator(device="cuda").manual_seed(7)
    q = torch.randn(B, T, H, D, generator=g, device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn(B, T, H, D, generator=g, device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn(B, T, H, D, generator=g, device="cuda", dtype=torch.bfloat16) * 0.5
    g_pre = torch.randn(B, T, H, D, generator=g, device="cuda", dtype=torch.bfloat16) * 0.1
    beta = torch.randn(B, T, H, generator=g, device="cuda", dtype=torch.bfloat16) * 0.1
    A_log = torch.randn(H, generator=g, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H, D, generator=g, device="cuda", dtype=torch.float32) * 0.1

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

    launch_k1_full(
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
    out = torch.zeros_like(v)
    launch_k2_phaseA(v, beta_flat, ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk, out)
    torch.cuda.synchronize()

    # torch reference
    qd_ref, kd_ref, kr_ref, gt_ref, INV_ref, Mqk_ref = _torch_ref(q, k, g_pre, beta, A_log, dt_bias, scale, gate_scale)
    out_ref = _k2_torch_ref(v, beta_flat, qd_ref, kd_ref, kr_ref, gt_ref, INV_ref, Mqk_ref, B, T, H, D, T)

    diff = (out.float() - out_ref).abs()
    rel = (diff.max() / out_ref.abs().max().clamp_min(1.0)).item()
    print(f"\n[k1full+k2A] out max_abs={diff.max().item():.4e}  rel={rel:.4e}  ref_absmax={out_ref.abs().max().item():.4e}")
    assert rel < 5e-2, f"K2 phaseA mismatch: rel={rel}"


def test_k1full_k2A_longer():
    """Longer recurrence (T=128 = 8 tiles per seq) to stress state evolution."""
    from cula.ops.flashkda_k1 import CHUNK, D, launch_k1_full
    from cula.ops.flashkda_k2 import launch_k2_phaseA

    B, T, H = 1, 128, 4
    scale = 0.125
    gate_scale = -1.0
    g = torch.Generator(device="cuda").manual_seed(11)
    q = torch.randn(B, T, H, D, generator=g, device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn(B, T, H, D, generator=g, device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn(B, T, H, D, generator=g, device="cuda", dtype=torch.bfloat16) * 0.5
    g_pre = torch.randn(B, T, H, D, generator=g, device="cuda", dtype=torch.bfloat16) * 0.1
    beta = torch.randn(B, T, H, generator=g, device="cuda", dtype=torch.bfloat16) * 0.1
    A_log = torch.randn(H, generator=g, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H, D, generator=g, device="cuda", dtype=torch.float32) * 0.1

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

    launch_k1_full(q, k, g_pre, A_log, dt_bias, beta_flat, scale, gate_scale, ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk)
    out = torch.zeros_like(v)
    launch_k2_phaseA(v, beta_flat, ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk, out)
    torch.cuda.synchronize()

    qd_ref, kd_ref, kr_ref, gt_ref, INV_ref, Mqk_ref = _torch_ref(q, k, g_pre, beta, A_log, dt_bias, scale, gate_scale)
    out_ref = _k2_torch_ref(v, beta_flat, qd_ref, kd_ref, kr_ref, gt_ref, INV_ref, Mqk_ref, B, T, H, D, T)
    diff = (out.float() - out_ref).abs()
    am = out_ref.abs().max().clamp_min(1e-3).item()
    rel = (diff.max() / am).item()
    print(f"\n[k1full+k2A T=128] out max_abs={diff.max().item():.4e}  ref_absmax={am:.4e}  rel={rel:.4e}")
    assert rel < 0.10, f"K2 phaseA T=128 mismatch: rel={rel}"

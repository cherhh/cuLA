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


def test_k1_phases_1to5():
    """Phases 1-5: TMA load q,k,g + L2 norm + gate cumsum + exp_g_total.
    Validates dumped workspace tensors against torch reference."""
    from cula.ops.flashkda_k1 import CHUNK, D, launch_k1_phases_1to5

    B, T, H = 1, 32, 2
    gate_scale = -5.0  # min(lower_bound=-5, 0)
    q, k, g_pre, _beta, A_log, dt_bias = _make_inputs(B, T, H, seed=1)
    total_tiles = (B * T) // CHUNK

    ws_q = torch.zeros(total_tiles * H * CHUNK * D, dtype=torch.bfloat16, device="cuda")
    ws_k = torch.zeros(total_tiles * H * CHUNK * D, dtype=torch.bfloat16, device="cuda")
    ws_gt = torch.zeros(total_tiles * H * D, dtype=torch.float32, device="cuda")

    launch_k1_phases_1to5(q, k, g_pre, A_log, dt_bias, gate_scale, ws_q, ws_k, ws_gt)
    torch.cuda.synchronize()

    ws_q = ws_q.view(H, total_tiles, CHUNK, D)
    ws_k = ws_k.view(H, total_tiles, CHUNK, D)
    ws_gt = ws_gt.view(H, total_tiles, D)

    # ---------------- torch reference ----------------
    # tile-major view of inputs: [H, total_tiles, CHUNK, D]
    q_tm = q.view(B * T, H, D).view(total_tiles, CHUNK, H, D).permute(2, 0, 1, 3).contiguous()
    k_tm = k.view(B * T, H, D).view(total_tiles, CHUNK, H, D).permute(2, 0, 1, 3).contiguous()
    g_tm = g_pre.view(B * T, H, D).view(total_tiles, CHUNK, H, D).permute(2, 0, 1, 3).contiguous()

    qf = q_tm.float()
    kf = k_tm.float()
    gf = g_tm.float()  # [H, tot, CHUNK, D]

    # L2 norm along D
    q_inv = (qf.pow(2).sum(-1, keepdim=True) + 1.0e-6).rsqrt()
    k_inv = (kf.pow(2).sum(-1, keepdim=True) + 1.0e-6).rsqrt()
    q_l2_ref = (qf * q_inv).to(torch.bfloat16)
    k_l2_ref = (kf * k_inv).to(torch.bfloat16)

    # Gate cumsum  → exp_g_total
    A_exp = torch.exp(A_log.float())  # [H]
    # broadcast: g[h,tot,r,c] + dt_bias[h,c]
    g_act = gate_scale * torch.sigmoid(A_exp.view(H, 1, 1, 1) * (gf + dt_bias.view(H, 1, 1, D)))  # [H, tot, CHUNK, D]
    cumsum_full = g_act.sum(dim=2)  # [H, tot, D]
    gt_ref = torch.exp(cumsum_full)

    def report(name, got, ref):
        diff = (got.float() - ref.float()).abs()
        return diff.max().item(), diff.mean().item()

    mq, _ = report("q_l2", ws_q, q_l2_ref)
    mk, _ = report("k_l2", ws_k, k_l2_ref)
    mgt, _ = report("g_total", ws_gt, gt_ref)
    print(f"\n[k1 phases1-5] max diffs: q_l2={mq:.4e}  k_l2={mk:.4e}  exp_g_total={mgt:.4e}")

    # bf16 quantization tolerance for L2 norm; float math precision for exp_g_total.
    assert mq < 5e-3, f"q L2 mismatch: {mq}"
    assert mk < 5e-3, f"k L2 mismatch: {mk}"
    assert mgt < 1e-2, f"exp_g_total mismatch: {mgt}"


def test_k1_phases_1to6():
    """Phases 1-6: also validates decay_apply outputs (q_decayed, k_decayed,
    k_inv, k_restored)."""
    from cula.ops.flashkda_k1 import CHUNK, D, launch_k1_phases_1to6

    B, T, H = 1, 32, 2
    scale = 0.125  # typical 1/sqrt(D/2) for KDA-style attention scale
    gate_scale = -1.0  # mild — keeps exp(g_cs) and exp(-g_cs) in bf16 range
    q, k, g_pre, _beta, A_log, dt_bias = _make_inputs(B, T, H, seed=2)
    total_tiles = (B * T) // CHUNK

    ws_qd = torch.zeros(total_tiles * H * CHUNK * D, dtype=torch.bfloat16, device="cuda")
    ws_kd = torch.zeros_like(ws_qd)
    ws_ki = torch.zeros_like(ws_qd)
    ws_kr = torch.zeros_like(ws_qd)
    ws_gt = torch.zeros(total_tiles * H * D, dtype=torch.float32, device="cuda")

    launch_k1_phases_1to6(
        q,
        k,
        g_pre,
        A_log,
        dt_bias,
        scale,
        gate_scale,
        ws_qd,
        ws_kd,
        ws_ki,
        ws_kr,
        ws_gt,
    )
    torch.cuda.synchronize()

    ws_qd = ws_qd.view(H, total_tiles, CHUNK, D)
    ws_kd = ws_kd.view(H, total_tiles, CHUNK, D)
    ws_ki = ws_ki.view(H, total_tiles, CHUNK, D)
    ws_kr = ws_kr.view(H, total_tiles, CHUNK, D)

    # ---- torch reference ----
    q_tm = q.view(B * T, H, D).view(total_tiles, CHUNK, H, D).permute(2, 0, 1, 3).contiguous().float()
    k_tm = k.view(B * T, H, D).view(total_tiles, CHUNK, H, D).permute(2, 0, 1, 3).contiguous().float()
    g_tm = g_pre.view(B * T, H, D).view(total_tiles, CHUNK, H, D).permute(2, 0, 1, 3).contiguous().float()

    q_inv = (q_tm.pow(2).sum(-1, keepdim=True) + 1.0e-6).rsqrt()
    k_inv_l = (k_tm.pow(2).sum(-1, keepdim=True) + 1.0e-6).rsqrt()
    q_l2 = (q_tm * q_inv).to(torch.bfloat16).float()  # round-trip bf16 to match kernel
    k_l2 = (k_tm * k_inv_l).to(torch.bfloat16).float()

    A_exp = torch.exp(A_log.float())
    g_act = gate_scale * torch.sigmoid(A_exp.view(H, 1, 1, 1) * (g_tm + dt_bias.view(H, 1, 1, D)))  # [H, tot, CHUNK, D]
    g_cs = torch.cumsum(g_act, dim=2)  # [H, tot, CHUNK, D]
    g_total = g_cs[:, :, -1, :]  # [H, tot, D]

    exp_pos = torch.exp(g_cs)
    inv_pos = 1.0 / exp_pos
    rest = torch.exp(g_total).unsqueeze(2) * inv_pos

    qd_ref = (q_l2 * exp_pos * scale).to(torch.bfloat16)
    kd_ref = (k_l2 * exp_pos).to(torch.bfloat16)
    ki_ref = (k_l2 * inv_pos).to(torch.bfloat16)
    kr_ref = (k_l2 * rest).to(torch.bfloat16)

    def diff(name, got, ref):
        d = (got.float() - ref.float()).abs()
        scale_d = ref.float().abs().max().clamp_min(1.0)
        return (d.max() / scale_d).item()  # relative to dynamic range

    mq = diff("qd", ws_qd, qd_ref)
    mk = diff("kd", ws_kd, kd_ref)
    mi = diff("ki", ws_ki, ki_ref)
    mr = diff("kr", ws_kr, kr_ref)
    print(f"\n[k1 phases1-6] rel diffs: qd={mq:.4e}  kd={mk:.4e}  ki={mi:.4e}  kr={mr:.4e}")
    # bf16 mantissa ≈ 0.4% relative; allow 1% for compounded round-trips.
    for n, m in (("qd", mq), ("kd", mk), ("ki", mi), ("kr", mr)):
        assert m < 1e-2, f"{n} relative mismatch: {m}"

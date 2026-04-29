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

"""
FlashKDA Prefill — CuteDSL port of MoonshotAI/FlashKDA.

Two-kernel design (CHUNK=16, D=128, SM90):

  K1 (Prepare, grid=(total_tiles, H)):
      g activation -> L2 norm -> decay apply -> L/Mqk GEMM
      -> 16x16 Neumann inverse -> write workspace (gmem)

  K2 (Recurrence, grid=(N, H), warp-specialized):
      load workspace -> dual GEMM (k@s, q@s) -> u = (v - k_s) * beta
      -> u = INV @ u -> out = q_s + Mqk @ u -> state += k_restored^T @ u

See cula/ops/flashkda_prefill_design.md for the full design rationale.

Status:
    - Public API + torch reference: implemented.
    - K1/K2 CuteDSL kernels: WIP. The wrapper currently dispatches to the
      torch reference unless ``CULA_FLASHKDA_USE_CUTE=1`` is set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import T as _T

# ---------------------------------------------------------------------------
# Constants (mirror FlashKDA C++ defaults)
# ---------------------------------------------------------------------------
CHUNK: int = 16  # token-block size
D: int = 128  # head_dim_k == head_dim_v (only 128 supported)
LOG2E: float = 1.4426950408889634  # log2(e), folds change-of-base into ex2

# Per-tile workspace byte sizes (must match C++ WorkspaceSizes).
# Layout: [k_decayed | q_decayed | k_restored | g_total | INV | Mqk] x H*total_tiles
_BYTES_KD = CHUNK * D * 2  # bf16
_BYTES_QD = CHUNK * D * 2
_BYTES_KR = CHUNK * D * 2
_BYTES_GT = D * 4  # fp32
_BYTES_INV = CHUNK * CHUNK * 2  # bf16
_BYTES_MQK = CHUNK * CHUNK * 2  # bf16
WORKSPACE_BYTES_PER_TILE: int = _BYTES_KD + _BYTES_QD + _BYTES_KR + _BYTES_GT + _BYTES_INV + _BYTES_MQK


# ============================================================================
# NVVM helpers
# ============================================================================
# CuteDSL already exposes:
#   cute.math.exp2(x, fastmath=True)  -> ex2.approx.ftz.f32
#   cute.math.tanh(x, fastmath=True)  -> tanh.approx.f32
#   cute.math.rsqrt(x, fastmath=True) -> rsqrt.approx
#
# What we still need to inline manually:
#   movmatrix.sync.aligned.m8n8.trans.b16  (no high-level CuteDSL wrapper)
#   This is the SM75 register-file matrix transpose used to convert MMA C-format
#   bf16 fragments into B-format operands without an SMEM round-trip.


@cutlass.dsl_user_op
def movm_t_b16(src_u32: Int32, *, loc=None, ip=None) -> Int32:
    """SM75 ``movmatrix.sync.aligned.m8n8.trans.b16``.

    Transposes an 8x8 b16 matrix that lives across the 32 lanes of a warp
    (1 ``u32`` register per lane, packing two bf16/fp16 elements). The
    instruction operates entirely in the register file - no SMEM trip.

    This is the cornerstone of the K2 inner loop: it converts the bf16
    fragment produced by an MMA accumulator (C-format) into a fragment laid
    out as a B operand for the next MMA, avoiding an SMEM write/read pair
    for every iteration.
    """
    result = _llvm.inline_asm(
        _T.i32(),
        [Int32(src_u32).ir_value(loc=loc, ip=ip)],
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@cutlass.dsl_user_op
def add_f16x2_u32(a_u32: Int32, b_u32: Int32, *, loc=None, ip=None) -> Int32:
    """Packed ``add.f16x2`` — adds two pairs of fp16 packed in u32.

    Used by the K1 Neumann register-resident accumulator path: when both
    A-frag (current INV) and C-frag (delta from MMA) hold fp16 values with
    the same per-thread u32 layout (a coincidence of m16n8k16 fp16-acc on a
    16x16 square tile), this PTX folds the +=update into a single 2-way
    SIMD packed add per u32 register.
    """
    result = _llvm.inline_asm(
        _T.i32(),
        [
            Int32(a_u32).ir_value(loc=loc, ip=ip),
            Int32(b_u32).ir_value(loc=loc, ip=ip),
        ],
        "add.f16x2 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@cute.jit
def sigmoid_fast(x: Float32) -> Float32:
    """``sigmoid(x)`` via ``tanh.approx.f32`` (single PTX instruction)."""
    return cute.math.tanh(x * Float32(0.5), fastmath=True) * Float32(0.5) + Float32(0.5)


# ============================================================================
# Torch reference (used for unit tests and as initial fallback)
# ============================================================================
def _flashkda_torch_reference(
    q: torch.Tensor,  # [B, T, H, K] bf16
    k: torch.Tensor,  # [B, T, H, K] bf16
    v: torch.Tensor,  # [B, T, H, V] bf16
    g: torch.Tensor,  # [B, T, H, K] bf16 (pre-activation)
    beta: torch.Tensor,  # [B, T, H] bf16 (pre-sigmoid)
    scale: float,
    A_log: torch.Tensor,  # [H] fp32
    dt_bias: torch.Tensor,  # [H, K] fp32
    lower_bound: float,  # gate floor (negative; e.g. -5.0)
    initial_state: torch.Tensor | None,  # [N, H, V, K] bf16/fp32 or None
    cu_seqlens: torch.Tensor | None,  # [N+1] int64 or None
    output_final_state: bool,
):
    """Bit-equivalent torch reference (chunk-free, all fp32 intermediates).

    Mirrors the math in flash_kda/__init__.py + FlashKDA C++ code.
    Returns (out_bf16, final_state) where final_state is None if not requested.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert K == V == D
    device = q.device

    # ---- variable-length unpacking ------------------------------------------------
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

    # ---- initial state ------------------------------------------------------------
    if initial_state is None:
        h = torch.zeros(N, H, V, K, device=device, dtype=torch.float32)
    else:
        h = initial_state.to(torch.float32).clone()

    out = torch.empty_like(v)

    A_exp = torch.exp(A_log).to(torch.float32)  # [H]
    dt_b = dt_bias.to(torch.float32)  # [H, K]
    # gate_scale = clamp(lower_bound, max=0); same as C++ launcher (negative scaling).
    gate_scale = float(min(lower_bound, 0.0))

    for n in range(N):
        Tn = seq_lens[n]
        bos = starts[n]
        for h_idx in range(H):
            state = h[n, h_idx]  # [V, K] fp32  (V-major like FlashKDA)
            for t in range(Tn):
                qi = q[0, bos + t, h_idx].float() if cu_seqlens is not None else q[n, t, h_idx].float()
                ki = k[0, bos + t, h_idx].float() if cu_seqlens is not None else k[n, t, h_idx].float()
                vi = v[0, bos + t, h_idx].float() if cu_seqlens is not None else v[n, t, h_idx].float()
                gi = g[0, bos + t, h_idx].float() if cu_seqlens is not None else g[n, t, h_idx].float()
                bi = beta[0, bos + t, h_idx].float() if cu_seqlens is not None else beta[n, t, h_idx].float()

                # L2 norm
                qi = qi / (qi.pow(2).sum().sqrt() + 1e-6).clamp(min=1e-6)
                ki = ki / (ki.pow(2).sum().sqrt() + 1e-6).clamp(min=1e-6)
                qi = qi * scale

                # Gate: g = gate_scale * sigmoid(A_exp * (g_raw + dt_bias))
                g_act = gate_scale * torch.sigmoid(A_exp[h_idx] * (gi + dt_b[h_idx]))  # [K]
                exp_g = torch.exp(g_act)  # decay per-key

                # Beta
                beta_act = torch.sigmoid(bi)

                # Decay state along K dimension: state[v,k] *= exp_g[k]
                state = state * exp_g.unsqueeze(0)

                # Delta-rule update: u = (v - state @ k) * beta
                u = (vi - state @ ki) * beta_act  # [V]
                # state += u outer k
                state = state + u.unsqueeze(1) * ki.unsqueeze(0)
                # output: o = state @ q
                o_t = state @ qi  # [V]
                if cu_seqlens is not None:
                    out[0, bos + t, h_idx] = o_t.to(out.dtype)
                else:
                    out[n, t, h_idx] = o_t.to(out.dtype)
            h[n, h_idx] = state

    final_state = None
    if output_final_state:
        if initial_state is not None and initial_state.dtype != torch.float32:
            final_state = h.to(initial_state.dtype)
        else:
            final_state = h
    return out, final_state


# ============================================================================
# Workspace helpers
# ============================================================================
def _compute_total_tiles(seq_lens: list[int]) -> int:
    return sum((sl + CHUNK - 1) // CHUNK for sl in seq_lens)


def allocate_workspace(
    total_tiles: int,
    H: int,
    *,
    device: torch.device | str | int = "cuda",
) -> torch.Tensor:
    """Allocate the inter-kernel workspace consumed by K1 (write) and K2 (read)."""
    n_bytes = total_tiles * H * WORKSPACE_BYTES_PER_TILE
    return torch.empty(n_bytes, dtype=torch.uint8, device=device)


# ============================================================================
# Public API
# ============================================================================
@dataclass
class _PrefillProblem:
    B: int
    T: int
    H: int
    N: int  # number of sequences (=B for fixed-len, =len(cu_seqlens)-1 for varlen)
    total_tiles: int
    is_varlen: bool
    has_state_in: bool
    has_state_out: bool
    state_fp32: bool


def _validate_inputs(q, k, v, g, beta, A_log, dt_bias, initial_state, final_state, cu_seqlens) -> _PrefillProblem:
    assert q.is_cuda and q.dtype == torch.bfloat16, f"q must be bf16 cuda, got {q.dtype}"
    assert k.dtype == v.dtype == g.dtype == beta.dtype == torch.bfloat16, "q/k/v/g/beta must all be bf16"
    assert q.shape == k.shape == g.shape, "q/k/g shapes must match"
    assert v.shape == q.shape, f"v shape {v.shape} != q shape {q.shape}"
    B, T, H, K = q.shape
    assert K == D and v.shape[-1] == D, f"only K=V={D} supported, got K={K} V={v.shape[-1]}"
    assert beta.shape == (B, T, H), f"beta shape mismatch: {beta.shape} vs ({B},{T},{H})"
    assert A_log.shape == (H,) and A_log.dtype == torch.float32
    assert dt_bias.shape == (H, K) and dt_bias.dtype == torch.float32

    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert B == 1, f"varlen requires B=1, got B={B}"
        assert cu_seqlens.dtype in (torch.int32, torch.int64)
        N = cu_seqlens.numel() - 1
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        total_tiles = _compute_total_tiles(seq_lens)
    else:
        N = B
        total_tiles = B * ((T + CHUNK - 1) // CHUNK)

    has_state_in = initial_state is not None
    has_state_out = final_state is not None
    state_fp32 = False
    if has_state_in:
        assert initial_state.shape == (N, H, D, D), f"initial_state shape mismatch: {initial_state.shape}"
        assert initial_state.dtype in (torch.bfloat16, torch.float32)
        state_fp32 = initial_state.dtype == torch.float32
    if has_state_out:
        assert final_state.shape == (N, H, D, D)
        if has_state_in:
            assert final_state.dtype == initial_state.dtype, "initial_state and final_state dtype must match"
        else:
            state_fp32 = final_state.dtype == torch.float32

    return _PrefillProblem(
        B=B,
        T=T,
        H=H,
        N=N,
        total_tiles=total_tiles,
        is_varlen=is_varlen,
        has_state_in=has_state_in,
        has_state_out=has_state_out,
        state_fp32=state_fp32,
    )


_USE_CUTE = os.environ.get("CULA_FLASHKDA_USE_CUTE", "0") == "1"

# ---- Cached scratch workspaces for K1+K2 ----
# Reused across calls when shape/device match; avoids allocator + zero-fill.
_WS_CACHE: dict = {}


def _get_or_alloc_workspaces(n_qk: int, n_cc: int, n_gt: int, n_beta: int, device, dtype):
    key = (n_qk, n_cc, n_gt, n_beta, str(device), dtype)
    cached = _WS_CACHE.get(key)
    if cached is not None:
        return cached
    ws_qd = torch.empty(n_qk, dtype=torch.bfloat16, device=device)
    ws_kd = torch.empty_like(ws_qd)
    ws_kr = torch.empty_like(ws_qd)
    ws_gt = torch.empty(n_gt, dtype=torch.float32, device=device)
    ws_inv = torch.empty(n_cc, dtype=torch.bfloat16, device=device)
    ws_mqk = torch.empty_like(ws_inv)
    # Persistent beta-flat scratch [H, B*T] reused across calls (same shape).
    beta_flat = torch.empty(n_beta, dtype=dtype, device=device)
    cached = (ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk, beta_flat)
    _WS_CACHE[key] = cached
    return cached


def flash_kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> None:
    """FlashKDA prefill (CuteDSL port of MoonshotAI/FlashKDA).

    Args mirror ``flash_kda.fwd``. ``out`` and ``final_state`` are written
    in-place. Currently only ``head_dim_k = head_dim_v = 128`` is supported.
    """
    problem = _validate_inputs(q, k, v, g, beta, A_log, dt_bias, initial_state, final_state, cu_seqlens)

    if _USE_CUTE:
        _dispatch_cute(
            q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, initial_state, final_state, cu_seqlens, problem
        )
        return

    # Reference fallback
    ref_out, ref_final = _flashkda_torch_reference(
        q,
        k,
        v,
        g,
        beta,
        scale,
        A_log,
        dt_bias,
        lower_bound,
        initial_state,
        cu_seqlens,
        output_final_state=problem.has_state_out,
    )
    out.copy_(ref_out)
    if problem.has_state_out:
        final_state.copy_(ref_final)


# ============================================================================
# CuteDSL kernel dispatch
# ============================================================================
def _dispatch_cute(
    q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, initial_state, final_state, cu_seqlens, problem: _PrefillProblem
):
    """Launch K1 + K2 (CuteDSL ports of FlashKDA C++).

    K1 is being built up phase-by-phase. K2 (warp-specialized recurrence)
    follows once K1 produces the correct workspace.

    Until K2 lands, the dispatcher refuses to run end-to-end (raises). To
    drive K1 development, use the unit tests in
    ``tests/test_flashkda_k1_phases.py`` which validate each K1 phase against
    a torch reference by reading the per-tile workspace dump.
    """
    from cula.ops.flashkda_k1 import CHUNK as K1_CHUNK
    from cula.ops.flashkda_k1 import D as K1_D
    from cula.ops.flashkda_k1 import launch_k1_full

    # K2 variant selector (env CULA_FLASHKDA_K2_VARIANT).
    # Default: phaseN (latest, matches/beats cpp baseline at T>=4096).
    _k2_variant = os.environ.get("CULA_FLASHKDA_K2_VARIANT", "N").upper()
    if _k2_variant == "N":
        from cula.ops.flashkda_k2_phaseN import launch_k2_phaseN as _launch_k2
    elif _k2_variant == "B":
        from cula.ops.flashkda_k2 import launch_k2_phaseB as _launch_k2
    else:
        raise ValueError(f"unknown CULA_FLASHKDA_K2_VARIANT={_k2_variant!r}")

    if problem.has_state_in or problem.has_state_out:
        if _k2_variant != "N":
            raise NotImplementedError("State inputs/outputs require K2 variant N (CULA_FLASHKDA_K2_VARIANT=N).")

    B, T, H = problem.B, problem.T, problem.H

    # Determine T_total and cu_seqlens_tiles for K2.
    if problem.is_varlen:
        # Varlen: B=1, q.shape=(1, T_total, H, D); cu_seqlens is provided.
        # Require all sequence lengths to be multiples of K1_CHUNK (CHUNK=16).
        assert cu_seqlens is not None
        assert B == 1
        T_total = T  # T is already T_total for B=1
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        assert (seq_lens % K1_CHUNK == 0).all(), f"All varlen sequence lengths must be multiples of CHUNK={K1_CHUNK}"
        # cu_seqlens_tiles: prefix sum of per-sequence tile counts (int32).
        cu_seqlens_tiles = (cu_seqlens // K1_CHUNK).to(torch.int32).contiguous()
        k2_cu_seqlens_tiles = cu_seqlens_tiles
    else:
        T_total = B * T
        k2_cu_seqlens_tiles = None  # launch_k2_phaseN builds uniform tiles internally

    total_tiles = T_total // K1_CHUNK

    # Allocate K2-shaped workspaces (separate buffers per tensor).
    n_qk = total_tiles * H * K1_CHUNK * K1_D
    n_cc = total_tiles * H * K1_CHUNK * K1_CHUNK
    # K1 writes every element of these workspaces before K2 reads them, so
    # ``torch.empty`` is sufficient and avoids the 5 zero-fill kernels per call
    # (these buffers total ~200 MB at H=64,T=8192). Workspaces are cached per
    # (n_qk,n_cc,total_tiles*H,device) key so repeated calls with the same
    # shape skip the cudaMalloc as well as the zero-fill.
    ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk, beta_flat = _get_or_alloc_workspaces(
        n_qk, n_cc, total_tiles * H * K1_D, T_total * H, q.device, beta.dtype
    )

    # Beta arrives as [B, T, H]; K1/K2 expect head-major [H, B*T] flat.
    # Reuse cached destination buffer (same shape across calls) and emit a
    # single transpose kernel into it instead of allocating a fresh tensor.
    beta_flat.view(H, T_total).copy_(beta.view(T_total, H).transpose(0, 1))

    launch_k1_full(
        q,
        k,
        g,
        A_log,
        dt_bias,
        beta_flat,
        scale,
        lower_bound,
        ws_qd,
        ws_kd,
        ws_kr,
        ws_gt,
        ws_inv,
        ws_mqk,
    )
    if _k2_variant == "N":
        # Prepare fp32 state tensors for K2 (bhvk layout: [N, H, V, K]).
        # initial_state may be bf16; convert to fp32 (K2 kernel uses fp32 for state).
        k2_initial_state = None
        k2_final_state = None
        if problem.has_state_in:
            k2_initial_state = (
                initial_state.to(torch.float32).contiguous() if initial_state.dtype != torch.float32 else initial_state
            )
        if problem.has_state_out:
            # final_state is pre-allocated by the caller; use fp32 scratch if bf16.
            if final_state.dtype == torch.float32:
                k2_final_state = final_state
            else:
                k2_final_state = torch.empty_like(final_state, dtype=torch.float32)
        _launch_k2(
            v,
            beta_flat,
            ws_qd,
            ws_kd,
            ws_kr,
            ws_gt,
            ws_inv,
            ws_mqk,
            out,
            k2_cu_seqlens_tiles,
            initial_state=k2_initial_state,
            final_state=k2_final_state,
        )
        # Copy fp32 scratch back to caller's bf16 final_state tensor.
        if problem.has_state_out and final_state.dtype != torch.float32:
            final_state.copy_(k2_final_state.to(final_state.dtype))
    else:
        if problem.is_varlen:
            raise NotImplementedError("Varlen requires K2 variant N (CULA_FLASHKDA_K2_VARIANT=N)")
        _launch_k2(v, beta_flat, ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk, out)


# ============================================================================
# K1 Prepare kernel (skeleton — to be filled in next commits)
# ============================================================================
class FlashKDAPrepare:
    """K1 Prepare kernel.

    Grid: (total_tiles, H)
    Threads/CTA: 256 (8 warps), `__launch_bounds__(256, 8)`.

    For each (head, chunk) tile:
        1. TMA load q, k, beta, g_bf16, dt_bias (single-shot).
        2. L2-normalize q and k along K dim.
        3. Compute fused gate cumsum:
              g_val = gate_scale * sigmoid(A_exp[h] * (g_raw + dt_bias))
              cumsum along chunk; g_total[k] = sum of all rows.
        4. decay_apply (vectorized 8 elem/thread):
              q_decayed  = q * exp(g_cumsum) * scale
              k_decayed  = k * exp(g_cumsum)
              k_inv      = k * exp(-g_cumsum)
              k_restored = k_inv * exp(g_total)
        5. L_Mqk = single-warp 16x16 GEMMs (fp16 acc for L, bf16 acc for Mqk).
        6. Apply tril mask, beta scaling; INV = I - L.
        7. Neumann series (4 powers in fp16): INV = (I - L)^(-1).
        8. TMA store workspace (kd, qd, kr, gt, INV, Mqk).

    NOTE: This is a skeleton. The full implementation is the next deliverable.
    """

    def __init__(self):
        self.chunk = CHUNK
        self.head_dim = D

    @cute.jit
    def __call__(self, *args, **kwargs):  # pragma: no cover - WIP
        raise NotImplementedError("FlashKDAPrepare.__call__ not yet implemented")


# ============================================================================
# K2 Recurrence kernel (skeleton — to be filled in next commits)
# ============================================================================
class FlashKDARecurrence:
    """K2 Recurrence kernel.

    Grid: (N, H). Warp-specialized:
        warps 0-3 (128 threads): MMA compute
        warp 4: TMA LOAD producer
        warp 5: TMA STORE consumer

    Per-chunk inner loop (Phase 1-6) follows the C++ reference closely. Most
    inner-loop perf depends on the MOVM_T transpose (movm_t_b16 above) keeping
    intermediate U fragments in registers across the four MMA stages.

    NOTE: This is a skeleton. The full implementation is the next deliverable.
    """

    def __init__(self):
        self.chunk = CHUNK
        self.head_dim = D

    @cute.jit
    def __call__(self, *args, **kwargs):  # pragma: no cover - WIP
        raise NotImplementedError("FlashKDARecurrence.__call__ not yet implemented")

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

# Copyright (c) 2026 MoonshotAI
# Licensed under the MIT License.
# Based on MoonshotAI/FlashKDA (https://github.com/MoonshotAI/FlashKDA)

"""
FlashKDA Prefill

two-kernel (K1 Prepare + K2 Recurrence), CHUNK=16, D=128.
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import T as _T

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK: int = 16
D: int = 128  # only 128 supported
LOG2E: float = 1.4426950408889634

# Per-tile workspace byte sizes.
_BYTES_KD = CHUNK * D * 2
_BYTES_QD = CHUNK * D * 2
_BYTES_KR = CHUNK * D * 2
_BYTES_GT = D * 4
_BYTES_INV = CHUNK * CHUNK * 2
_BYTES_MQK = CHUNK * CHUNK * 2
WORKSPACE_BYTES_PER_TILE: int = _BYTES_KD + _BYTES_QD + _BYTES_KR + _BYTES_GT + _BYTES_INV + _BYTES_MQK
_SUPPORTED_CUTE_ARCH = "sm_90a"
_VARLEN_LAYOUT_CACHE_MAXSIZE = 64


# ============================================================================
# NVVM helpers
# ============================================================================


@cutlass.dsl_user_op
def movm_t_b16(src_u32: Int32, *, loc=None, ip=None) -> Int32:
    """``movmatrix.sync.aligned.m8n8.trans.b16`` -- register-file 8x8 b16 transpose."""
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
    """Packed ``add.f16x2`` on two u32 registers."""
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
    """``sigmoid(x)`` via ``tanh.approx.f32``."""
    return Float32(cute.math.tanh(x * Float32(0.5), fastmath=True) * Float32(0.5) + Float32(0.5))


# ============================================================================
# Torch reference
# ============================================================================
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
    """Torch reference for unit tests and fallback.

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


# ============================================================================
# Workspace helpers
# ============================================================================
def _compute_total_tiles(seq_lens: list[int] | tuple[int, ...]) -> int:
    return sum((sl + CHUNK - 1) // CHUNK for sl in seq_lens)


def allocate_workspace(
    total_tiles: int,
    H: int,
    *,
    device: torch.device | str | int = "cuda",
) -> torch.Tensor:
    """Allocate inter-kernel workspace for K1/K2."""
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
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B, T, H, D], got {tuple(q.shape)}")
    if not q.is_cuda or q.dtype != torch.bfloat16:
        raise TypeError(f"q must be a CUDA bfloat16 tensor, got dtype={q.dtype}, device={q.device}")
    for name, tensor in (("k", k), ("v", v), ("g", g), ("beta", beta)):
        if not tensor.is_cuda or tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be a CUDA bfloat16 tensor, got dtype={tensor.dtype}, device={tensor.device}")
    if q.shape != k.shape or q.shape != g.shape:
        raise ValueError(f"q/k/g shapes must match, got q={tuple(q.shape)}, k={tuple(k.shape)}, g={tuple(g.shape)}")
    if v.shape != q.shape:
        raise ValueError(f"v shape {tuple(v.shape)} must match q shape {tuple(q.shape)}")

    B, T, H, K = q.shape
    if B <= 0 or T <= 0 or H <= 0:
        raise ValueError(f"B, T and H must be positive, got B={B}, T={T}, H={H}")
    if K != D or v.shape[-1] != D:
        raise ValueError(f"only K=V={D} supported, got K={K} V={v.shape[-1]}")
    if beta.shape != (B, T, H):
        raise ValueError(f"beta shape mismatch: {tuple(beta.shape)} vs ({B},{T},{H})")
    if A_log is None or not A_log.is_cuda or not A_log.is_contiguous() or A_log.shape != (H,) or A_log.dtype != torch.float32:
        raise ValueError(f"A_log must be float32 with shape ({H},), got {None if A_log is None else (A_log.dtype, tuple(A_log.shape))}")
    if dt_bias is None or not dt_bias.is_cuda or not dt_bias.is_contiguous() or dt_bias.shape != (H, K) or dt_bias.dtype != torch.float32:
        raise ValueError(
            f"dt_bias must be float32 with shape ({H}, {K}), "
            f"got {None if dt_bias is None else (dt_bias.dtype, tuple(dt_bias.shape))}"
        )

    is_varlen = cu_seqlens is not None
    if is_varlen:
        if B != 1:
            raise ValueError(f"varlen requires B=1, got B={B}")
        if not cu_seqlens.is_cuda or cu_seqlens.ndim != 1:
            raise ValueError("cu_seqlens must be a 1D CUDA tensor")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"cu_seqlens must be int32 or int64, got {cu_seqlens.dtype}")
        if cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must contain at least two entries")
        N = cu_seqlens.numel() - 1
        seq_lens = _get_or_build_seq_lens(cu_seqlens)
        if int(cu_seqlens[0].item()) != 0:
            raise ValueError("cu_seqlens must start at 0")
        if int(cu_seqlens[-1].item()) != T:
            raise ValueError(f"cu_seqlens[-1] must equal packed T={T}, got {int(cu_seqlens[-1].item())}")
        if any(sl <= 0 for sl in seq_lens):
            raise ValueError(f"all variable-length sequences must be non-empty, got seq_lens={seq_lens}")
        total_tiles = _compute_total_tiles(seq_lens)
    else:
        N = B
        total_tiles = B * ((T + CHUNK - 1) // CHUNK)

    has_state_in = initial_state is not None
    has_state_out = final_state is not None
    state_fp32 = False
    if has_state_in:
        if initial_state.shape != (N, H, D, D):
            raise ValueError(f"initial_state shape must be ({N}, {H}, {D}, {D}), got {tuple(initial_state.shape)}")
        if not initial_state.is_cuda or initial_state.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError("initial_state must be a CUDA bf16 or fp32 tensor")
        state_fp32 = initial_state.dtype == torch.float32
    if has_state_out:
        if final_state.shape != (N, H, D, D):
            raise ValueError(f"final_state shape must be ({N}, {H}, {D}, {D}), got {tuple(final_state.shape)}")
        if not final_state.is_cuda or final_state.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError("final_state must be a CUDA bf16 or fp32 tensor")
        if has_state_in:
            if final_state.dtype != initial_state.dtype:
                raise TypeError("initial_state and final_state dtype must match")
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


_USE_CUTE = os.environ.get("CULA_FLASHKDA_USE_CUTE", "1") != "0"
_STRICT_CUTE = os.environ.get("CULA_FLASHKDA_STRICT_CUTE", "1") != "0"
_WARNED_CUTE_FALLBACK = False
_WARNED_CUDAGRAPH_DISABLED = False


@contextmanager
def _cute_arch_for_device(device: torch.device):
    """Temporarily provide the SM90 CuTeDSL arch without leaking process env."""
    if not torch.cuda.is_available() or device.type != "cuda":
        yield
        return

    major, _minor = torch.cuda.get_device_capability(device)
    if major != 9:
        raise RuntimeError(f"SM90 FlashKDA prefill requires a Hopper device, got compute capability {major}.x")

    old_arch = os.environ.get("CUTE_DSL_ARCH")
    if old_arch is not None and old_arch != _SUPPORTED_CUTE_ARCH:
        raise RuntimeError(
            f"SM90 FlashKDA prefill requires CUTE_DSL_ARCH={_SUPPORTED_CUTE_ARCH}, "
            f"but the process has CUTE_DSL_ARCH={old_arch!r}."
        )

    if old_arch is None:
        os.environ["CUTE_DSL_ARCH"] = _SUPPORTED_CUTE_ARCH
    try:
        yield
    finally:
        if old_arch is None:
            os.environ.pop("CUTE_DSL_ARCH", None)


def _is_cute_runtime_compat_error(exc: Exception) -> bool:
    """True if exception is a CuteDSL runtime/env mismatch (not a kernel bug)."""
    msg = repr(exc)
    markers = (
        "DSLCudaRuntimeError",
        "cudaErrorInsufficientDriver",
        "Target SM ARCH",
        "CUTE_DSL_ARCH",
    )
    return any(m in msg for m in markers)

# ---- Cached scratch workspaces ----
_VARLEN_LAYOUT_CACHE: dict = {}
_K1_SYMBOLS = None
_K2_LAUNCHER = None


def _get_or_alloc_workspaces(n_qk: int, n_cc: int, n_gt: int, n_beta: int, device, dtype):
    ws_qd = torch.empty(n_qk, dtype=torch.bfloat16, device=device)
    ws_kd = torch.empty_like(ws_qd)
    ws_kr = torch.empty_like(ws_qd)
    ws_gt = torch.empty(n_gt, dtype=torch.float32, device=device)
    ws_inv = torch.empty(n_cc, dtype=torch.bfloat16, device=device)
    ws_mqk = torch.empty_like(ws_inv)
    beta_flat = torch.empty(n_beta, dtype=dtype, device=device)
    return ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk, beta_flat


def _copy_beta_flat(beta: torch.Tensor, beta_flat: torch.Tensor, H: int, T_total: int) -> None:
    """Transpose beta [.., T, H] -> beta_flat [H, T_total]."""
    beta_flat.view(H, T_total).copy_(beta.view(T_total, H).transpose(0, 1))


def _get_or_alloc_varlen_pack_buffers(total_aligned: int, H: int, N: int, device, q_dtype, beta_dtype):
    q_pad = torch.empty((1, total_aligned, H, D), dtype=q_dtype, device=device)
    k_pad = torch.empty_like(q_pad)
    v_pad = torch.empty_like(q_pad)
    g_pad = torch.empty_like(q_pad)
    beta_pad = torch.empty((1, total_aligned, H), dtype=beta_dtype, device=device)
    out_pad = torch.empty_like(q_pad)
    return q_pad, k_pad, v_pad, g_pad, beta_pad, out_pad


def _get_or_build_varlen_layout(seq_lens: tuple[int, ...], device, cu_dtype):
    key = (seq_lens, str(device), cu_dtype)
    cached = _VARLEN_LAYOUT_CACHE.get(key)
    if cached is not None:
        return cached

    idx_list: list[int] = []
    valid_dst_list: list[int] = []
    pad_idx_list: list[int] = []
    out_offsets = [0]

    src_cursor = 0
    dst_cursor = 0
    for sl in seq_lens:
        aligned = ((sl + CHUNK - 1) // CHUNK) * CHUNK
        idx_list.extend(range(src_cursor, src_cursor + sl))
        valid_dst_list.extend(range(dst_cursor, dst_cursor + sl))
        if aligned > sl:
            idx_list.extend([src_cursor] * (aligned - sl))
            pad_idx_list.extend(range(dst_cursor + sl, dst_cursor + aligned))
        src_cursor += sl
        dst_cursor += aligned
        out_offsets.append(dst_cursor)

    idx = torch.tensor(idx_list, dtype=torch.int32, device=device)
    valid_dst = torch.tensor(valid_dst_list, dtype=torch.int32, device=device)
    pad_idx = torch.tensor(pad_idx_list, dtype=torch.int64, device=device)
    cu_pad = torch.tensor(out_offsets, dtype=cu_dtype, device=device)
    cu_tiles = torch.tensor([off // CHUNK for off in out_offsets], dtype=torch.int32, device=device)
    cached = (idx, valid_dst, pad_idx, cu_pad, cu_tiles, tuple(out_offsets))
    if len(_VARLEN_LAYOUT_CACHE) >= _VARLEN_LAYOUT_CACHE_MAXSIZE:
        _VARLEN_LAYOUT_CACHE.pop(next(iter(_VARLEN_LAYOUT_CACHE)))
    _VARLEN_LAYOUT_CACHE[key] = cached
    return cached


def _get_or_build_seq_lens(cu_seqlens: torch.Tensor) -> tuple[int, ...]:
    return tuple((cu_seqlens[1:] - cu_seqlens[:-1]).to("cpu").tolist())


def _get_or_build_cu_tiles(cu_seqlens: torch.Tensor, chunk: int) -> torch.Tensor:
    return (cu_seqlens // chunk).to(torch.int32).contiguous()


def _get_k1_symbols():
    global _K1_SYMBOLS
    if _K1_SYMBOLS is None:
        from cula.ops.sm90.k1 import CHUNK as k1_chunk
        from cula.ops.sm90.k1 import D as k1_d
        from cula.ops.sm90.k1 import launch_k1 as k1_launch

        _K1_SYMBOLS = (k1_chunk, k1_d, k1_launch)
    return _K1_SYMBOLS


def _get_k2_launcher():
    global _K2_LAUNCHER
    if _K2_LAUNCHER is not None:
        return _K2_LAUNCHER
    from cula.ops.sm90.k2 import launch_k2
    _K2_LAUNCHER = launch_k2
    return launch_k2


def flash_kda_fwd(
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
    state_transposed: bool = False,
    use_gate_in_kernel: bool = True,
) -> None:
    """FlashKDA fwd. ``out`` and ``final_state`` are written in-place.

    Args:
        q, k, v, g: [B, T, H, D] bf16.
        beta: [B, T, H] bf16 (pre-sigmoid).
        scale: attention scale.
        out: [B, T, H, D] bf16 output (written in-place).
        A_log: [H] fp32.
        dt_bias: [H, D] fp32.
        lower_bound: gate floor (negative).
        initial_state: [N, H, D, D] bf16/fp32 or None.
        final_state: [N, H, D, D] bf16/fp32 or None (written in-place).
        cu_seqlens: [N+1] int32/int64 for variable-length, or None.
        state_transposed: False -> [N,H,V,K] (default), True -> [N,H,K,V].
    """
    global _WARNED_CUTE_FALLBACK

    problem = _validate_inputs(q, k, v, g, beta, A_log, dt_bias, initial_state, final_state, cu_seqlens)
    if out.shape != q.shape or not out.is_cuda or out.dtype != torch.bfloat16:
        raise ValueError(f"out must be CUDA bfloat16 with shape {tuple(q.shape)}, got dtype={out.dtype}, shape={tuple(out.shape)}")
    if use_gate_in_kernel:
        if lower_bound is None:
            raise ValueError("lower_bound must be specified when use_gate_in_kernel=True.")
        if not (-5 <= lower_bound < 0):
            raise ValueError(f"lower_bound must be in the safe range [-5, 0), got {lower_bound}.")

    if _USE_CUTE and use_gate_in_kernel:
        with _cute_arch_for_device(q.device):
            try:
                _dispatch_cute(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    scale,
                    out,
                    A_log,
                    dt_bias,
                    lower_bound,
                    initial_state,
                    final_state,
                    cu_seqlens,
                    problem,
                    state_transposed=state_transposed,
                )
                return
            except Exception as exc:
                if _STRICT_CUTE:
                    raise
                if not _is_cute_runtime_compat_error(exc):
                    raise
                if not _WARNED_CUTE_FALLBACK:
                    warnings.warn(
                        "CuteDSL fwd dispatch failed due to runtime compatibility; "
                        "falling back to torch reference. "
                        f"Error: {exc!r}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    _WARNED_CUTE_FALLBACK = True

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
        state_transposed=state_transposed,
        use_gate_in_kernel=use_gate_in_kernel,
    )
    out.copy_(ref_out)
    if problem.has_state_out:
        final_state.copy_(ref_final)


# ============================================================================
# CuteDSL kernel dispatch
# ============================================================================
def _dispatch_cute(
    q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, initial_state, final_state, cu_seqlens, problem: _PrefillProblem,
    *, state_transposed: bool = False,
):
    """Launch K1 + K2."""
    K1_CHUNK, K1_D, launch_k1 = _get_k1_symbols()

    # Non-varlen: pad T to chunk boundary if needed.
    T_orig = problem.T
    need_t_pad = (not problem.is_varlen) and (T_orig % K1_CHUNK != 0)
    if need_t_pad:
        T_pad = ((T_orig + K1_CHUNK - 1) // K1_CHUNK) * K1_CHUNK
        B, H = problem.B, problem.H
        pad_len = T_pad - T_orig
        q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad_len))
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad_len))
        g = torch.nn.functional.pad(g, (0, 0, 0, 0, 0, pad_len), value=-1e6)
        beta = torch.nn.functional.pad(beta, (0, 0, 0, pad_len), value=-80.0)
        out_orig = out
        out = torch.empty_like(q)
        problem = _PrefillProblem(
            B=B, T=T_pad, H=H, N=problem.N,
            total_tiles=B * (T_pad // K1_CHUNK),
            is_varlen=False,
            has_state_in=problem.has_state_in,
            has_state_out=problem.has_state_out,
            state_fp32=problem.state_fp32,
        )

    # Varlen: pad unaligned sequences to CHUNK boundary, scatter back after.
    scatter_back_target = None
    scatter_back_idx = None
    k2_cu_seqlens_tiles_cached = None
    if problem.is_varlen:
        assert cu_seqlens is not None
        seq_lens_list = _get_or_build_seq_lens(cu_seqlens)
        if any((sl % K1_CHUNK) != 0 for sl in seq_lens_list):
            assert problem.B == 1, "varlen path expects packed B=1"

            aligned_lens = [((sl + K1_CHUNK - 1) // K1_CHUNK) * K1_CHUNK for sl in seq_lens_list]
            total_aligned = sum(aligned_lens)

            q_pad, k_pad, v_pad, g_pad, beta_pad, out_pad = _get_or_alloc_varlen_pack_buffers(
                total_aligned,
                problem.H,
                problem.N,
                q.device,
                q.dtype,
                beta.dtype,
            )
            gather_idx, valid_dst_idx, pad_idx, cu_pad_cached, cu_tiles_cached, _out_offsets = _get_or_build_varlen_layout(
                tuple(seq_lens_list),
                q.device,
                cu_seqlens.dtype,
            )
            cu_pad = cu_pad_cached
            k2_cu_seqlens_tiles_cached = cu_tiles_cached

            torch.index_select(q, 1, gather_idx, out=q_pad)
            torch.index_select(k, 1, gather_idx, out=k_pad)
            torch.index_select(v, 1, gather_idx, out=v_pad)
            torch.index_select(g, 1, gather_idx, out=g_pad)
            torch.index_select(beta, 1, gather_idx, out=beta_pad)

            if pad_idx.numel() > 0:
                beta_pad.index_fill_(1, pad_idx, -80.0)
                g_pad.index_fill_(1, pad_idx, -1e6)

            problem_pad = _PrefillProblem(
                B=1,
                T=total_aligned,
                H=problem.H,
                N=problem.N,
                total_tiles=total_aligned // K1_CHUNK,
                is_varlen=True,
                has_state_in=problem.has_state_in,
                has_state_out=problem.has_state_out,
                state_fp32=problem.state_fp32,
            )
            scatter_back_target = out
            scatter_back_idx = valid_dst_idx
            q, k, v, g, beta, out, cu_seqlens, problem = (
                q_pad,
                k_pad,
                v_pad,
                g_pad,
                beta_pad,
                out_pad,
                cu_pad,
                problem_pad,
            )

    _launch_k2 = _get_k2_launcher()

    B, T, H = problem.B, problem.T, problem.H

    if problem.is_varlen:
        assert cu_seqlens is not None
        assert B == 1
        T_total = T
        if k2_cu_seqlens_tiles_cached is not None:
            k2_cu_seqlens_tiles = k2_cu_seqlens_tiles_cached
        else:
            k2_cu_seqlens_tiles = _get_or_build_cu_tiles(cu_seqlens, K1_CHUNK)
    else:
        T_total = B * T
        k2_cu_seqlens_tiles = None

    total_tiles = T_total // K1_CHUNK

    n_qk = total_tiles * H * K1_CHUNK * K1_D
    n_cc = total_tiles * H * K1_CHUNK * K1_CHUNK
    ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk, beta_flat = _get_or_alloc_workspaces(
        n_qk, n_cc, total_tiles * H * K1_D, T_total * H, q.device, beta.dtype
    )

    _copy_beta_flat(beta, beta_flat, H, T_total)

    k2_initial_state = None
    if problem.has_state_in:
        k2_initial_state = initial_state.to(torch.float32).contiguous() if initial_state.dtype != torch.float32 else initial_state

    def _run_k1k2(out_tensor: torch.Tensor, k2_final_state_tensor: torch.Tensor | None) -> None:
        launch_k1(
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
        _launch_k2(
            v,
            beta_flat,
            ws_qd,
            ws_kd,
            ws_kr,
            ws_gt,
            ws_inv,
            ws_mqk,
            out_tensor,
            k2_cu_seqlens_tiles,
            initial_state=k2_initial_state,
            final_state=k2_final_state_tensor,
            state_transposed=state_transposed,
        )

    # CUDA Graph is disabled until graph-owned buffers are made stream-owned.
    # The old module-level graph/output/state cache was unsafe for overlapping
    # requests and skipped the non-varlen T-padding copy-back on forced graph.
    graph_mode = os.environ.get("CULA_FLASHKDA_VARLEN_CUDAGRAPH", "auto").lower()
    if graph_mode not in ("0", "auto"):
        global _WARNED_CUDAGRAPH_DISABLED
        if not _WARNED_CUDAGRAPH_DISABLED:
            warnings.warn(
                "CULA_FLASHKDA_VARLEN_CUDAGRAPH is ignored for SM90 CuTeDSL prefill "
                "until graph-owned scratch buffers are stream-owned.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_CUDAGRAPH_DISABLED = True

    k2_final_state = None
    if problem.has_state_out:
        if final_state.dtype == torch.float32:
            k2_final_state = final_state
        else:
            k2_final_state = torch.empty_like(final_state, dtype=torch.float32)
    _run_k1k2(out, k2_final_state)
    if problem.has_state_out and final_state.dtype != torch.float32:
        final_state.copy_(k2_final_state.to(final_state.dtype))

    if scatter_back_target is not None:
        torch.index_select(out, 1, scatter_back_idx, out=scatter_back_target)

    if need_t_pad:
        out_orig.copy_(out[:, :T_orig])

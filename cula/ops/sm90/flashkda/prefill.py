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

"""FlashKDA Prefill — two-kernel (K1 Prepare + K2 Recurrence), CHUNK=16, D=128, SM90."""

from __future__ import annotations

import os
import weakref
import warnings
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
# Torch reference (used for unit tests and as initial fallback)
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
):
    """Torch reference for unit tests and fallback."""
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
    gate_scale = float(min(lower_bound, 0.0))

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

                # L2 norm
                qi = qi / (qi.pow(2).sum().sqrt() + 1e-6).clamp(min=1e-6)
                ki = ki / (ki.pow(2).sum().sqrt() + 1e-6).clamp(min=1e-6)
                qi = qi * scale

                g_act = gate_scale * torch.sigmoid(A_exp[h_idx] * (gi + dt_b[h_idx]))
                exp_g = torch.exp(g_act)

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
def _compute_total_tiles(seq_lens: list[int]) -> int:
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
        seq_lens = _get_or_build_seq_lens(cu_seqlens)
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
_STRICT_CUTE = os.environ.get("CULA_FLASHKDA_STRICT_CUTE", "0") == "1"
_WARNED_CUTE_FALLBACK = False


def _ensure_cute_arch_for_device(device: torch.device) -> None:
    """Set CUTE_DSL_ARCH from device capability if unset."""
    if os.environ.get("CUTE_DSL_ARCH"):
        return
    if not torch.cuda.is_available():
        return
    if device.type != "cuda":
        return
    major, _minor = torch.cuda.get_device_capability(device)
    if major == 9:
        os.environ["CUTE_DSL_ARCH"] = "sm_90a"
    if major == 10:
        os.environ["CUTE_DSL_ARCH"] = "sm_100"
    


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
_WS_CACHE: dict = {}
_VARLEN_PACK_CACHE: dict = {}
_VARLEN_LAYOUT_CACHE: dict = {}
_LAST_VARLEN_REPACK_REFS = None
_LAST_BETA_FLAT_COPY = None
_LAST_VARLEN_GRAPH_KEY = None
_LAST_VARLEN_GRAPH = None
_LAST_VARLEN_GRAPH_OUT = None
_LAST_VARLEN_GRAPH_STATE = None
_LAST_PROBLEM_KEY = None
_LAST_PROBLEM: _PrefillProblem | None = None
_K1_SYMBOLS = None
_K2_LAUNCHER = None
_SEQ_LENS_OBJ_CACHE: dict[int, tuple[weakref.ReferenceType, int, tuple[int, ...]]] = {}
_CU_TILES_OBJ_CACHE: dict[int, tuple[weakref.ReferenceType, int, torch.Tensor]] = {}


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
    beta_flat = torch.empty(n_beta, dtype=dtype, device=device)
    cached = (ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk, beta_flat)
    _WS_CACHE[key] = cached
    return cached


def _make_tensor_refs(tensors) -> tuple:
    return tuple((weakref.ref(t), int(t._version)) for t in tensors)


def _same_tensor_refs(cached, tensors) -> bool:
    return (
        cached is not None
        and len(cached) == len(tensors)
        and all(r() is t and ver == int(t._version) for (r, ver), t in zip(cached, tensors))
    )


def _copy_beta_flat(beta: torch.Tensor, beta_flat: torch.Tensor, H: int, T_total: int) -> None:
    """Transpose beta [.., T, H] -> beta_flat [H, T_total], with caching."""
    global _LAST_BETA_FLAT_COPY
    c = _LAST_BETA_FLAT_COPY
    if c is not None and c[1] == (H, T_total) and _same_tensor_refs(c[0], (beta, beta_flat)):
        return
    beta_flat.view(H, T_total).copy_(beta.view(T_total, H).transpose(0, 1))
    _LAST_BETA_FLAT_COPY = (_make_tensor_refs((beta, beta_flat)), (H, T_total))


def _get_or_alloc_varlen_pack_buffers(total_aligned: int, H: int, N: int, device, q_dtype, beta_dtype):
    key = (total_aligned, H, N, str(device), q_dtype, beta_dtype)
    cached = _VARLEN_PACK_CACHE.get(key)
    if cached is not None:
        return cached

    q_pad = torch.empty((1, total_aligned, H, D), dtype=q_dtype, device=device)
    k_pad = torch.empty_like(q_pad)
    v_pad = torch.empty_like(q_pad)
    g_pad = torch.empty_like(q_pad)
    beta_pad = torch.empty((1, total_aligned, H), dtype=beta_dtype, device=device)
    out_pad = torch.empty_like(q_pad)
    cached = (q_pad, k_pad, v_pad, g_pad, beta_pad, out_pad)
    _VARLEN_PACK_CACHE[key] = cached
    return cached


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
    _VARLEN_LAYOUT_CACHE[key] = cached
    return cached


def _get_or_build_seq_lens(cu_seqlens: torch.Tensor) -> tuple[int, ...]:
    obj_id = id(cu_seqlens)
    ver = int(cu_seqlens._version)
    cached = _SEQ_LENS_OBJ_CACHE.get(obj_id)
    if cached is not None:
        ref_obj, cached_ver, cached_seq = cached
        if ref_obj() is cu_seqlens and cached_ver == ver:
            return cached_seq
    seq_lens = tuple((cu_seqlens[1:] - cu_seqlens[:-1]).to("cpu").tolist())
    _SEQ_LENS_OBJ_CACHE[obj_id] = (weakref.ref(cu_seqlens), ver, seq_lens)
    return seq_lens


def _get_or_build_cu_tiles(cu_seqlens: torch.Tensor, chunk: int) -> torch.Tensor:
    obj_id = id(cu_seqlens)
    ver = int(cu_seqlens._version)
    cached = _CU_TILES_OBJ_CACHE.get(obj_id)
    if cached is not None:
        ref_obj, cached_ver, cached_tiles = cached
        if ref_obj() is cu_seqlens and cached_ver == ver:
            return cached_tiles
    cu_tiles = (cu_seqlens // chunk).to(torch.int32).contiguous()
    _CU_TILES_OBJ_CACHE[obj_id] = (weakref.ref(cu_seqlens), ver, cu_tiles)
    return cu_tiles


def _get_k1_symbols():
    global _K1_SYMBOLS
    if _K1_SYMBOLS is None:
        from cula.ops.sm90.flashkda.k1 import CHUNK as k1_chunk
        from cula.ops.sm90.flashkda.k1 import D as k1_d
        from cula.ops.sm90.flashkda.k1 import launch_k1 as k1_launch

        _K1_SYMBOLS = (k1_chunk, k1_d, k1_launch)
    return _K1_SYMBOLS


def _get_k2_launcher():
    global _K2_LAUNCHER
    if _K2_LAUNCHER is not None:
        return _K2_LAUNCHER
    from cula.ops.sm90.flashkda.k2 import launch_k2
    _K2_LAUNCHER = launch_k2
    return launch_k2


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
    state_transposed: bool = False,
) -> None:
    """FlashKDA prefill. ``out`` and ``final_state`` are written in-place.

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
    global _WARNED_CUTE_FALLBACK, _LAST_PROBLEM_KEY, _LAST_PROBLEM

    problem_key = (
        id(q),
        id(k),
        id(v),
        id(g),
        id(beta),
        id(A_log),
        id(dt_bias),
        q.shape,
        q.dtype,
        q.device,
        beta.shape,
        A_log.shape,
        A_log.dtype,
        dt_bias.shape,
        dt_bias.dtype,
        None if cu_seqlens is None else (id(cu_seqlens), int(cu_seqlens._version), cu_seqlens.dtype, cu_seqlens.device),
        None if initial_state is None else (initial_state.shape, initial_state.dtype, initial_state.device),
        None if final_state is None else (final_state.shape, final_state.dtype, final_state.device),
    )
    if problem_key == _LAST_PROBLEM_KEY and _LAST_PROBLEM is not None:
        problem = _LAST_PROBLEM
    else:
        problem = _validate_inputs(q, k, v, g, beta, A_log, dt_bias, initial_state, final_state, cu_seqlens)
        _LAST_PROBLEM_KEY = problem_key
        _LAST_PROBLEM = problem

    if _USE_CUTE:
        _ensure_cute_arch_for_device(q.device)
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
                    "CuteDSL prefill dispatch failed due to runtime compatibility; "
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

            global _LAST_VARLEN_REPACK_REFS
            repack_srcs = (q, k, v, g, beta, gather_idx)
            if not _same_tensor_refs(_LAST_VARLEN_REPACK_REFS, repack_srcs):
                torch.index_select(q, 1, gather_idx, out=q_pad)
                torch.index_select(k, 1, gather_idx, out=k_pad)
                torch.index_select(v, 1, gather_idx, out=v_pad)
                torch.index_select(g, 1, gather_idx, out=g_pad)
                torch.index_select(beta, 1, gather_idx, out=beta_pad)

                if pad_idx.numel() > 0:
                    beta_pad.index_fill_(1, pad_idx, -80.0)
                _LAST_VARLEN_REPACK_REFS = _make_tensor_refs(repack_srcs)

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

    # CUDA Graph path (skipped when initial state is provided).
    use_cuda_graph = (
        (not problem.has_state_in)
        and os.environ.get("CULA_FLASHKDA_VARLEN_CUDAGRAPH", "1") != "0"
    )
    if use_cuda_graph:
        global _LAST_VARLEN_GRAPH_KEY, _LAST_VARLEN_GRAPH, _LAST_VARLEN_GRAPH_OUT, _LAST_VARLEN_GRAPH_STATE
        bind_user_out = scatter_back_target is None
        bind_user_state = (
            problem.has_state_out
            and final_state is not None
            and final_state.dtype == torch.float32
        )
        graph_key = (
            q.data_ptr(), int(q._version),
            k.data_ptr(), int(k._version),
            v.data_ptr(), int(v._version),
            g.data_ptr(), int(g._version),
            beta.data_ptr(), int(beta._version),
            A_log.data_ptr(), int(A_log._version),
            dt_bias.data_ptr(), int(dt_bias._version),
            cu_seqlens.data_ptr() if cu_seqlens is not None else 0,
            int(cu_seqlens._version) if cu_seqlens is not None else -1,
            out.data_ptr() if bind_user_out else 0,
            out.shape,
            out.dtype,
            out.device,
            problem.has_state_out,
            final_state.data_ptr() if bind_user_state else 0,
            problem.N,
            scale,
            lower_bound,
            state_transposed,
        )
        if graph_key != _LAST_VARLEN_GRAPH_KEY:
            graph_out = out if bind_user_out else torch.empty_like(out)
            if problem.has_state_out:
                graph_state = (
                    final_state
                    if bind_user_state
                    else torch.empty((problem.N, H, D, D), dtype=torch.float32, device=out.device)
                )
            else:
                graph_state = None
            _run_k1k2(graph_out, graph_state)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                _run_k1k2(graph_out, graph_state)
            _LAST_VARLEN_GRAPH_KEY = graph_key
            _LAST_VARLEN_GRAPH = graph
            _LAST_VARLEN_GRAPH_OUT = graph_out
            _LAST_VARLEN_GRAPH_STATE = graph_state

        _LAST_VARLEN_GRAPH.replay()
        if not bind_user_out:
            if scatter_back_target is not None:
                torch.index_select(_LAST_VARLEN_GRAPH_OUT, 1, scatter_back_idx, out=scatter_back_target)
            else:
                out.copy_(_LAST_VARLEN_GRAPH_OUT)
        if problem.has_state_out and not bind_user_state:
            if final_state.dtype == torch.float32:
                final_state.copy_(_LAST_VARLEN_GRAPH_STATE)
            else:
                final_state.copy_(_LAST_VARLEN_GRAPH_STATE.to(final_state.dtype))
        return

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


# ============================================================================
# K1 Prepare kernel (skeleton)
# ============================================================================
class FlashKDAPrepare:
    """K1 Prepare kernel. Grid: (total_tiles, H), 256 threads."""

    def __init__(self):
        self.chunk = CHUNK
        self.head_dim = D

    @cute.jit
    def __call__(self, *args, **kwargs):  # pragma: no cover - WIP
        raise NotImplementedError("FlashKDAPrepare.__call__ not yet implemented")


# ============================================================================
# K2 Recurrence kernel (skeleton)
# ============================================================================
class FlashKDARecurrence:
    """K2 Recurrence kernel. Grid: (N, H), warp-specialized."""

    def __init__(self):
        self.chunk = CHUNK
        self.head_dim = D

    @cute.jit
    def __call__(self, *args, **kwargs):  # pragma: no cover - WIP
        raise NotImplementedError("FlashKDARecurrence.__call__ not yet implemented")

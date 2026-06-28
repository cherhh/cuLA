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
import weakref
from contextlib import contextmanager
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK: int = 16
D: int = 128  # only 128 supported

# Per-tile workspace byte sizes.
_BYTES_KD = CHUNK * D * 2
_BYTES_QD = CHUNK * D * 2
_BYTES_KR = CHUNK * D * 2
_BYTES_GT = D * 4
_BYTES_INV = CHUNK * CHUNK * 2
_BYTES_MQK = CHUNK * CHUNK * 2
WORKSPACE_BYTES_PER_TILE: int = _BYTES_KD + _BYTES_QD + _BYTES_KR + _BYTES_GT + _BYTES_INV + _BYTES_MQK

_CUTE_ARCH_BY_CC = {(9, 0): "sm_90a", (10, 0): "sm_100a", (10, 3): "sm_103a"}
_VARLEN_LAYOUT_CACHE_MAXSIZE = 64


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
class _VarlenMetadata:
    cu_values: tuple[int, ...]
    seq_lens: tuple[int, ...]
    total_tiles: int
    needs_padding: bool
    total_aligned: int
    cu_tiles: torch.Tensor | None
    tile_starts: torch.Tensor
    tile_actual_lens: torch.Tensor


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
    varlen_meta: _VarlenMetadata | None = None


def _validate_inputs(
    q, k, v, g, beta, A_log, dt_bias, initial_state, final_state, cu_seqlens, cu_seqlens_cpu=None
) -> _PrefillProblem:
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
        raise ValueError(
            f"A_log must be float32 with shape ({H},), got {None if A_log is None else (A_log.dtype, tuple(A_log.shape))}"
        )
    if (
        dt_bias is None
        or not dt_bias.is_cuda
        or not dt_bias.is_contiguous()
        or dt_bias.shape != (H, K)
        or dt_bias.dtype != torch.float32
    ):
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
        if cu_seqlens.dtype != torch.int32:
            raise TypeError(f"cu_seqlens must be int32, got {cu_seqlens.dtype}")
        if cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must contain at least two entries")
        if cu_seqlens_cpu is not None and (
            cu_seqlens_cpu.device.type != "cpu" or cu_seqlens_cpu.ndim != 1 or cu_seqlens_cpu.numel() != cu_seqlens.numel()
        ):
            raise ValueError(
                "cu_seqlens_cpu must be a 1D CPU tensor with the same numel as "
                f"cu_seqlens ({cu_seqlens.numel()}), got device={cu_seqlens_cpu.device}, "
                f"shape={tuple(cu_seqlens_cpu.shape)}"
            )
        varlen_meta = _get_or_build_varlen_metadata(cu_seqlens, cu_seqlens_cpu)
        N = len(varlen_meta.seq_lens)
        if varlen_meta.cu_values[0] != 0:
            raise ValueError("cu_seqlens must start at 0")
        if varlen_meta.cu_values[-1] != T:
            raise ValueError(f"cu_seqlens[-1] must equal packed T={T}, got {varlen_meta.cu_values[-1]}")
        seq_lens = varlen_meta.seq_lens
        if any(sl <= 0 for sl in seq_lens):
            raise ValueError(f"all variable-length sequences must be non-empty, got seq_lens={seq_lens}")
        total_tiles = varlen_meta.total_tiles
    else:
        N = B
        total_tiles = B * ((T + CHUNK - 1) // CHUNK)
        varlen_meta = None

    has_state_in = initial_state is not None
    has_state_out = final_state is not None
    if has_state_in:
        if initial_state.shape != (N, H, D, D):
            raise ValueError(f"initial_state shape must be ({N}, {H}, {D}, {D}), got {tuple(initial_state.shape)}")
        if not initial_state.is_cuda or initial_state.dtype != torch.float32 or not initial_state.is_contiguous():
            raise TypeError("initial_state must be a contiguous CUDA float32 tensor")
    if has_state_out:
        if final_state.shape != (N, H, D, D):
            raise ValueError(f"final_state shape must be ({N}, {H}, {D}, {D}), got {tuple(final_state.shape)}")
        if not final_state.is_cuda or final_state.dtype != torch.float32 or not final_state.is_contiguous():
            raise TypeError("final_state must be a contiguous CUDA float32 tensor")

    return _PrefillProblem(
        B=B,
        T=T,
        H=H,
        N=N,
        total_tiles=total_tiles,
        is_varlen=is_varlen,
        has_state_in=has_state_in,
        has_state_out=has_state_out,
        varlen_meta=varlen_meta,
    )


@contextmanager
def _cute_arch_for_device(device: torch.device):
    """Temporarily provide the CuTeDSL arch for ``device`` (sm_90a / sm_100a) without leaking process env."""
    if not torch.cuda.is_available() or device.type != "cuda":
        yield
        return

    major, minor = torch.cuda.get_device_capability(device)
    arch = _CUTE_ARCH_BY_CC.get((major, minor))
    if arch is None:
        raise RuntimeError(
            f"FlashKDA prefill supports Hopper (sm_90a) or Blackwell (sm_100a/sm_103a); "
            f"got compute capability sm_{major}{minor}."
        )

    old_arch = os.environ.get("CUTE_DSL_ARCH")
    if old_arch is not None and old_arch != arch:
        raise RuntimeError(
            f"FlashKDA prefill requires CUTE_DSL_ARCH={arch} for this device, but the process has CUTE_DSL_ARCH={old_arch!r}."
        )

    if old_arch is None:
        os.environ["CUTE_DSL_ARCH"] = arch
    try:
        yield
    finally:
        if old_arch is None:
            os.environ.pop("CUTE_DSL_ARCH", None)


# ---- Cached scratch workspaces ----
_VARLEN_LAYOUT_CACHE: dict = {}
_VARLEN_METADATA_CACHE: dict[int, tuple[weakref.ReferenceType[torch.Tensor], tuple, _VarlenMetadata]] = {}
_K1_SYMBOLS = None
_K2_LAUNCHER = None


def _get_or_alloc_workspaces(n_qk: int, n_cc: int, n_gt: int, n_beta: int, device, dtype):
    """Allocate K1/K2 scratch tensors (ws_qd/kd/kr/gt/inv/mqk, beta_flat)."""
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


def _varlen_metadata_attrs(cu_seqlens: torch.Tensor) -> tuple:
    return (
        cu_seqlens.data_ptr(),
        tuple(cu_seqlens.shape),
        str(cu_seqlens.device),
        cu_seqlens.dtype,
        int(cu_seqlens._version),
    )


def _prune_varlen_metadata_cache() -> None:
    for key, (tensor_ref, _attrs, _meta) in list(_VARLEN_METADATA_CACHE.items()):
        if tensor_ref() is None:
            _VARLEN_METADATA_CACHE.pop(key, None)


def _store_varlen_metadata(cu_seqlens: torch.Tensor, attrs: tuple, meta: _VarlenMetadata) -> None:
    if len(_VARLEN_METADATA_CACHE) >= _VARLEN_LAYOUT_CACHE_MAXSIZE:
        _prune_varlen_metadata_cache()
    if len(_VARLEN_METADATA_CACHE) >= _VARLEN_LAYOUT_CACHE_MAXSIZE:
        _VARLEN_METADATA_CACHE.pop(next(iter(_VARLEN_METADATA_CACHE)))
    _VARLEN_METADATA_CACHE[id(cu_seqlens)] = (weakref.ref(cu_seqlens), attrs, meta)


def _get_or_build_varlen_metadata(cu_seqlens: torch.Tensor, cu_seqlens_cpu: torch.Tensor | None = None) -> _VarlenMetadata:
    """Cache CPU varlen metadata (seq_lens, tile offsets, padding flags) for cu_seqlens."""
    cache_key = id(cu_seqlens)
    attrs = _varlen_metadata_attrs(cu_seqlens)
    cached = _VARLEN_METADATA_CACHE.get(cache_key)
    if cached is not None:
        tensor_ref, cached_attrs, meta = cached
        if tensor_ref() is cu_seqlens and cached_attrs == attrs:
            return meta
        _VARLEN_METADATA_CACHE.pop(cache_key, None)

    src_cpu = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.detach().to("cpu")
    cu_values = tuple(int(v) for v in src_cpu.tolist())
    seq_lens = tuple(cu_values[i + 1] - cu_values[i] for i in range(len(cu_values) - 1))
    total_tiles = _compute_total_tiles(seq_lens)
    needs_padding = any((sl % CHUNK) != 0 for sl in seq_lens)
    aligned_lens = tuple(((sl + CHUNK - 1) // CHUNK) * CHUNK for sl in seq_lens)
    total_aligned = sum(aligned_lens)
    tile_starts_list: list[int] = []
    tile_actual_lens_list: list[int] = []
    for bos, sl in zip(cu_values[:-1], seq_lens):
        for offset in range(0, sl, CHUNK):
            tile_starts_list.append(bos + offset)
            tile_actual_lens_list.append(min(CHUNK, sl - offset))
    tile_starts = torch.tensor(tile_starts_list, dtype=torch.int32, device=cu_seqlens.device)
    tile_actual_lens = torch.tensor(tile_actual_lens_list, dtype=torch.int32, device=cu_seqlens.device)
    cu_tiles = None
    if not needs_padding:
        cu_tiles = torch.tensor(
            [v // CHUNK for v in cu_values],
            dtype=torch.int32,
            device=cu_seqlens.device,
        )

    meta = _VarlenMetadata(
        cu_values=cu_values,
        seq_lens=seq_lens,
        total_tiles=total_tiles,
        needs_padding=needs_padding,
        total_aligned=total_aligned,
        cu_tiles=cu_tiles,
        tile_starts=tile_starts,
        tile_actual_lens=tile_actual_lens,
    )
    _store_varlen_metadata(cu_seqlens, attrs, meta)
    return meta


def _get_or_build_seq_lens(cu_seqlens: torch.Tensor) -> tuple[int, ...]:
    return _get_or_build_varlen_metadata(cu_seqlens).seq_lens


def _get_or_build_cu_tiles(cu_seqlens: torch.Tensor, chunk: int) -> torch.Tensor:
    if chunk == CHUNK:
        meta = _get_or_build_varlen_metadata(cu_seqlens)
        if meta.cu_tiles is not None:
            return meta.cu_tiles
    return (cu_seqlens // chunk).to(torch.int32).contiguous()


def _get_k1_symbols():
    global _K1_SYMBOLS
    if _K1_SYMBOLS is None:
        from cula.ops.kda.sm90.k1 import CHUNK as k1_chunk
        from cula.ops.kda.sm90.k1 import D as k1_d
        from cula.ops.kda.sm90.k1 import launch_k1 as k1_launch

        _K1_SYMBOLS = (k1_chunk, k1_d, k1_launch)
    return _K1_SYMBOLS


def _get_k2_launcher():
    global _K2_LAUNCHER
    if _K2_LAUNCHER is not None:
        return _K2_LAUNCHER
    from cula.ops.kda.sm90.k2 import launch_k2

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
    cu_seqlens_cpu: torch.Tensor | None = None,
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
        cu_seqlens_cpu: optional CPU copy of cu_seqlens (same values) to skip the
            GPU->host sync when first building varlen metadata. Trusted, not verified.
        state_transposed: False -> [N,H,V,K] (default), True -> [N,H,K,V].
    """
    problem = _validate_inputs(q, k, v, g, beta, A_log, dt_bias, initial_state, final_state, cu_seqlens, cu_seqlens_cpu)
    if out.shape != q.shape or not out.is_cuda or out.dtype != torch.bfloat16:
        raise ValueError(
            f"out must be CUDA bfloat16 with shape {tuple(q.shape)}, got dtype={out.dtype}, shape={tuple(out.shape)}"
        )
    if not use_gate_in_kernel:
        raise NotImplementedError(
            "CuTeDSL FlashKDA prefill only supports use_gate_in_kernel=True. "
            "Pre-gated inputs would require the torch reference, which is test-only."
        )
    if lower_bound is None:
        raise ValueError("lower_bound must be specified.")
    if not (-5 <= lower_bound < 0):
        raise ValueError(f"lower_bound must be in the safe range [-5, 0), got {lower_bound}.")

    with _cute_arch_for_device(q.device):
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


# ============================================================================
# CuteDSL kernel dispatch
# ============================================================================
def _dispatch_cute(
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
    problem: _PrefillProblem,
    *,
    state_transposed: bool = False,
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
            B=B,
            T=T_pad,
            H=H,
            N=problem.N,
            total_tiles=B * (T_pad // K1_CHUNK),
            is_varlen=False,
            has_state_in=problem.has_state_in,
            has_state_out=problem.has_state_out,
        )

    k1_q, k1_k, k1_g, k1_beta = q, k, g, beta
    k1_T_total = problem.B * problem.T
    k1_total_tiles = problem.total_tiles
    k1_tile_starts = None
    k1_tile_actual_lens = None
    k1_is_varlen = False

    # Varlen: K1/K2 read original q/k/g/v; beta remains padded for the
    # existing compact workspace layout.
    k2_cu_seqlens_tiles_cached = None
    k2_v_tile_starts = None
    k2_v_tile_actual_lens = None
    if problem.is_varlen:
        varlen_meta = problem.varlen_meta
        seq_lens_list = varlen_meta.seq_lens
        if varlen_meta.needs_padding:
            total_aligned = varlen_meta.total_aligned

            k1_q = q.contiguous()
            k1_k = k.contiguous()
            k1_g = g.contiguous()
            k1_beta = beta.contiguous()
            k1_T_total = problem.T
            k1_total_tiles = varlen_meta.total_tiles
            k1_tile_starts = varlen_meta.tile_starts
            k1_tile_actual_lens = varlen_meta.tile_actual_lens
            k1_is_varlen = True
            k2_v_tile_starts = varlen_meta.tile_starts
            k2_v_tile_actual_lens = varlen_meta.tile_actual_lens

            beta_pad = torch.empty((1, total_aligned, problem.H), dtype=beta.dtype, device=q.device)
            gather_idx, _valid_dst_idx, pad_idx, cu_pad, k2_cu_seqlens_tiles_cached, _out_offsets = (
                _get_or_build_varlen_layout(
                    tuple(seq_lens_list),
                    q.device,
                    cu_seqlens.dtype,
                )
            )

            torch.index_select(beta, 1, gather_idx, out=beta_pad)

            if pad_idx.numel() > 0:
                beta_pad.index_fill_(1, pad_idx, -80.0)

            problem_pad = _PrefillProblem(
                B=1,
                T=total_aligned,
                H=problem.H,
                N=problem.N,
                total_tiles=total_aligned // K1_CHUNK,
                is_varlen=True,
                has_state_in=problem.has_state_in,
                has_state_out=problem.has_state_out,
            )
            beta, cu_seqlens, problem = beta_pad, cu_pad, problem_pad
        else:
            k2_cu_seqlens_tiles_cached = varlen_meta.cu_tiles

    _launch_k2 = _get_k2_launcher()

    B, T, H = problem.B, problem.T, problem.H

    if problem.is_varlen:
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
    k2_beta_flat = beta_flat
    if k1_is_varlen:
        k1_beta_flat = torch.empty(k1_T_total * H, dtype=k1_beta.dtype, device=k1_beta.device)
        _copy_beta_flat(k1_beta, k1_beta_flat, H, k1_T_total)
    else:
        k1_beta_flat = k2_beta_flat

    k2_initial_state = None
    if problem.has_state_in:
        k2_initial_state = initial_state.contiguous()

    k2_final_state = None
    if problem.has_state_out:
        k2_final_state = final_state

    launch_k1(
        k1_q,
        k1_k,
        k1_g,
        A_log,
        dt_bias,
        k1_beta_flat,
        scale,
        lower_bound,
        ws_qd,
        ws_kd,
        ws_kr,
        ws_gt,
        ws_inv,
        ws_mqk,
        tile_starts=k1_tile_starts,
        tile_actual_lens=k1_tile_actual_lens,
        total_tiles=k1_total_tiles,
        is_varlen=k1_is_varlen,
    )
    _launch_k2(
        v,
        k2_beta_flat,
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
        state_transposed=state_transposed,
        v_tile_starts=k2_v_tile_starts,
        v_tile_actual_lens=k2_v_tile_actual_lens,
    )

    if need_t_pad:
        out_orig.copy_(out[:, :T_orig])

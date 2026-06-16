# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Hopper (SM90) KDA prefill — CuTeDSL implementation.

Replaces the former CUTLASS C++ ``kda_fwd_prefill`` kernel with the CuTeDSL
two-kernel pipeline (K1 prepare + K2 recurrence, CHUNK=16, D=128).
Gate preprocessing, L2-norm, and sigmoid(beta) are all handled inside K1;
callers pass raw inputs.
"""

import warnings

import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from cula.ops.flashkda.prefill import flash_kda_prefill
from cula.utils import assert_hopper


class HopperChunkKDAFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool = False,
        lower_bound: float | None = None,
        cu_seqlens: torch.IntTensor | None = None,
    ):
        batch_size, seq_len, num_heads, head_dim = q.shape

        out = torch.empty_like(v)

        n_seqs = batch_size
        if cu_seqlens is not None:
            n_seqs = cu_seqlens.numel() - 1

        final_state = None
        if output_final_state:
            final_state = torch.empty(
                n_seqs, num_heads, head_dim, head_dim,
                dtype=torch.float32, device=q.device,
            )

        g = g.to(torch.bfloat16)
        beta = beta.to(torch.bfloat16)

        flash_kda_prefill(
            q, k, v, g, beta,
            scale=scale,
            out=out,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound if lower_bound is not None else -5.0,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
            state_transposed=True,
        )

        return out.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        raise NotImplementedError("Backward pass is not implemented yet.")


@torch.compiler.disable
def cula_kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.IntTensor | None = None,
    chunk_indices: torch.IntTensor | None = None,
    **kwargs,
):
    r"""
    Hopper (SM90) KDA forward prefill using CuTeDSL two-kernel pipeline.

    Gate preprocessing (A_log, dt_bias, lower_bound) and L2-norm are handled
    internally by the K1 kernel. The ``use_qk_l2norm_in_kernel`` and
    ``use_gate_in_kernel`` parameters are accepted for API compatibility but
    ignored (the CuTeDSL path always processes raw inputs).

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the KDA attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Accepted for API compatibility; CuTeDSL always applies L2-norm
            internally. Default: `False`.
        use_gate_in_kernel (bool):
            Accepted for API compatibility; CuTeDSL always computes the gate
            internally. Default: `False`.
        safe_gate (bool):
            Whether the kernel can assume the input gate values `g` are in a safe range.
            Default: `False`.
        lower_bound (Optional[float]):
            Lower bound for the forget gate activation function. Default: `None`.
        cu_seqlens (torch.IntTensor):
            Cumulative sequence lengths of shape `[N+1]`, int32.
        chunk_indices (torch.IntTensor):
            Accepted for API compatibility; unused by CuTeDSL.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
    assert_hopper()
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."

    num_heads, head_dim = q.shape[2], q.shape[3]
    A_log = kwargs.pop("A_log", None)
    dt_bias = kwargs.pop("dt_bias", None)
    kwargs.pop("cu_seqlens_cpu", None)
    if kwargs:
        raise TypeError(f"cula_kda_prefill got unexpected keyword arguments: {set(kwargs)}")
    if A_log is None:
        A_log = torch.zeros(num_heads, dtype=torch.float32, device=q.device)
    if dt_bias is None:
        dt_bias = torch.zeros(num_heads, head_dim, dtype=torch.float32, device=q.device)
    elif dt_bias.ndim == 1:
        dt_bias = dt_bias.view(num_heads, head_dim)

    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert beta.shape == q.shape[:3], "beta must be of shape (batch size, seq len, num of head)."
    assert v.shape == (*q.shape[:3], v.shape[-1]), "v must be of shape (batch size, seq len, num of head, head dim)."
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16, "q, k, v must be in bfloat16."
    assert beta.dtype == torch.bfloat16 or beta.dtype == torch.float32, "beta must be in bfloat16 or float32."
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == 128, "Currently we only support head dim of 128 for KDA"
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = HopperChunkKDAFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        initial_state,
        output_final_state,
        lower_bound,
        cu_seqlens,
    )
    return o, final_state

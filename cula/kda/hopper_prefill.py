# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""SM90 KDA prefill wrapper for the two-kernel K1+K2 CuTeDSL path"""

from typing import Literal

import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from cula.ops.kda.policy import (
    IntracardCPMode,
    resolve_intracard_cp_mode,
    sm90_intracard_cp_decision,
)
from cula.ops.kda.sm90.fwd import flash_kda_fwd
from cula.utils import assert_hopper


def _cast_g_bf16(g: torch.Tensor) -> torch.Tensor:
    return g if g.dtype == torch.bfloat16 else g.to(torch.bfloat16)


def _beta_logits_bf16(beta: torch.Tensor) -> torch.Tensor:
    return torch.logit(beta.float(), eps=1e-6).to(torch.bfloat16)


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
        cu_seqlens_cpu: torch.IntTensor | None = None,
        use_intracard_cp: IntracardCPMode | None = None,
    ):
        batch_size, seq_len, num_heads, head_dim = q.shape

        out = torch.empty_like(v)

        n_seqs = batch_size
        if cu_seqlens is not None:
            n_seqs = cu_seqlens.numel() - 1

        final_state = None
        if output_final_state:
            final_state = torch.empty(
                n_seqs,
                num_heads,
                head_dim,
                head_dim,
                dtype=torch.float32,
                device=q.device,
            )

        g = _cast_g_bf16(g)
        # FLA convention: beta is post-sigmoid [0,1].
        # cuLA kernel applies sigmoid internally, so convert to pre-sigmoid logits.
        beta = _beta_logits_bf16(beta)

        cp_decision = sm90_intracard_cp_decision(q, cu_seqlens, cu_seqlens_cpu, use_intracard_cp)
        if cp_decision.enabled:
            from cula.ops.kda.sm90.cp.driver import intracard_prefill

            intracard_prefill(
                q,
                k,
                v,
                g,
                beta,
                scale=scale,
                out=out,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                initial_state=initial_state,
                final_state=final_state,
                cu_seqlens=cu_seqlens,
                state_transposed=False,
                allow_fallback=False,
            )
        else:
            flash_kda_fwd(
                q,
                k,
                v,
                g,
                beta,
                scale=scale,
                out=out,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                initial_state=initial_state,
                final_state=final_state,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                state_transposed=False,
                use_gate_in_kernel=True,
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
    use_gate_in_kernel: bool = True,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.IntTensor | None = None,
    chunk_indices: torch.IntTensor | None = None,
    use_intracard_cp: Literal["auto"] | bool | None = None,
    **kwargs,
):
    r"""
    Hopper (SM90) KDA forward prefill using CuTeDSL two-kernel pipeline.

    Gate preprocessing (A_log, dt_bias, lower_bound) and L2-norm are handled
    internally by the K1 kernel. This SM90 CuTeDSL path supports only the safe
    in-kernel gate mode: ``use_gate_in_kernel=True`` and ``safe_gate=True``.
    ``use_qk_l2norm_in_kernel`` is accepted for API compatibility; CuTeDSL
    always applies L2-norm internally.

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
            Must be `True`; CuTeDSL computes the gate internally. Default: `True`.
        safe_gate (bool):
            Must be `True`; unsupported unsafe gate inputs are rejected. Default: `False`.
        lower_bound (Optional[float]):
            Lower bound for the forget gate activation function. Required when
            `safe_gate=True`; must be in `[-5, 0)`. Default: `None`.
        cu_seqlens (torch.IntTensor):
            Cumulative sequence lengths of shape `[N+1]`, int32.
        cu_seqlens_cpu (Optional[torch.IntTensor]):
            Optional CPU copy of `cu_seqlens` (same values), passed via kwargs, to
            skip the GPU->host sync when first building varlen metadata. Trusted,
            not verified (FLA convention). Default: `None`.
        chunk_indices (torch.IntTensor):
            Accepted for API compatibility; unused by CuTeDSL.
        use_intracard_cp (Literal["auto"] | bool):
            Whether to use the SM90 intracard-CP path when profitable. ``True``
            requires CP support and raises on rejection, ``"auto"`` falls back
            to the serial K1+K2 path, and ``False`` disables CP. ``use_cp`` is
            accepted as a compatibility alias.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
    assert_hopper()
    if not use_gate_in_kernel:
        raise NotImplementedError(
            "SM90 CuTeDSL KDA prefill only supports use_gate_in_kernel=True. "
            "Passing preprocessed gates would otherwise fall back to the slow reference path."
        )
    if not safe_gate:
        raise NotImplementedError("SM90 CuTeDSL KDA prefill only supports safe_gate=True.")
    if lower_bound is None:
        raise ValueError("`lower_bound` must be specified when `safe_gate=True` and `use_gate_in_kernel=True`.")
    if not (-5 <= lower_bound < 0):
        raise ValueError(f"`lower_bound` must be in the safe range [-5, 0), got {lower_bound}.")
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
        if initial_state.dtype != torch.float32:
            raise TypeError("initial_state must be in float32.")

    num_qk_heads, head_dim = q.shape[2], q.shape[3]
    num_kv_heads = v.shape[2]
    A_log = kwargs.pop("A_log", None)
    dt_bias = kwargs.pop("dt_bias", None)
    cu_seqlens_cpu = kwargs.pop("cu_seqlens_cpu", None)
    use_cp_alias = kwargs.pop("use_cp", None)
    use_intracard_cp = resolve_intracard_cp_mode(use_intracard_cp, use_cp_alias)
    if kwargs:
        raise TypeError(f"cula_kda_prefill got unexpected keyword arguments: {set(kwargs)}")
    if A_log is None:
        raise ValueError("A_log must be provided when use_gate_in_kernel=True.")
    if dt_bias is None:
        raise ValueError("dt_bias must be provided when use_gate_in_kernel=True.")
    elif dt_bias.ndim == 1:
        dt_bias = dt_bias.view(num_kv_heads, head_dim)

    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got q={tuple(q.shape)}, k={tuple(k.shape)}")
    if g.shape != v.shape:
        raise ValueError(f"g and v must have the same shape, got g={tuple(g.shape)}, v={tuple(v.shape)}")
    if beta.shape != v.shape[:3]:
        raise ValueError(f"beta must have shape {tuple(v.shape[:3])}, got {tuple(beta.shape)}")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("q, k, v must be in bfloat16.")
    if beta.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("beta must be in bfloat16 or float32.")
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        raise ValueError("Currently we only support head dim of 128 for KDA.")
    if num_kv_heads != num_qk_heads:
        raise NotImplementedError(
            "SM90 CuTeDSL KDA prefill does not support grouped-value attention yet "
            f"(num_kv_heads={num_kv_heads} != num_qk_heads={num_qk_heads}); native GVA is a follow-up change."
        )

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
        cu_seqlens_cpu,
        use_intracard_cp,
    )
    return o, final_state

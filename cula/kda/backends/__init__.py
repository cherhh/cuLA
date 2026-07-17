# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib

import torch

from cula.backends import BackendRegistry
from cula.kda.backends.flashkda import FlashKDABackend
from cula.kda.backends.fully_fused import FullyFusedBackend

kda_prefill_registry = BackendRegistry("kda_prefill")
kda_prefill_registry.register(FlashKDABackend())
kda_prefill_registry.register(FullyFusedBackend())


@torch.compiler.disable
def kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    safe_gate: bool = True,
    lower_bound: float | None = -5.0,
    cu_seqlens: torch.IntTensor | None = None,
    chunk_indices: torch.IntTensor | None = None,
    **kwargs,
):
    """[cherhh] Select the first SM90 prefill backend that accepts the call.

    This forward-only API requires grad-disabled execution. FlashKDA has
    priority over the fully-fused CUDA path; disabling either backend with its
    ``CULA_BACKEND_*`` variable makes dispatch try the remaining implementation.
    """
    device_ctx = torch.cuda.device(q.device) if q.device.type == "cuda" else contextlib.nullcontext()
    with device_ctx:
        return kda_prefill_registry.dispatch(
            "kda_prefill",
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            **kwargs,
        )


__all__ = ["kda_prefill", "kda_prefill_registry"]

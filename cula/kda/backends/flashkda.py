# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util

import torch

from cula.backends import BaseBackend
from cula.kda.backends._common import verify_common_inputs
from cula.ops.kda.cp_mode import CPMode

_ACCEPTED_KWARGS = {
    "A_log",
    "dt_bias",
    "cu_seqlens_cpu",
    "use_beta_sigmoid_in_kernel",
    "use_intracard_cp",
    "out",
    "final_state",
}


def _is_sm90(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (9, 0)


class FlashKDABackend(BaseBackend):
    backend_type = "flashkda"
    env_var = "CULA_BACKEND_FLASHKDA"
    priority = 1

    def probe(self):
        if importlib.util.find_spec("cutlass") is None:
            return False, "nvidia-cutlass-dsl not installed"
        return True, None

    def kda_prefill_verifier(
        self,
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        cu_seqlens=None,
        chunk_indices=None,
        **kwargs,
    ):
        ok, why = verify_common_inputs(q, k, v, g, beta)
        if not ok:
            return ok, why
        if not _is_sm90(q.device):
            return False, "requires an SM90 (Hopper) device"
        unknown = set(kwargs) - _ACCEPTED_KWARGS
        if unknown:
            return False, f"unsupported kwargs: {sorted(unknown)}"
        if v.shape[2] != q.shape[2]:
            return False, f"no GVA support (HV={v.shape[2]} != H={q.shape[2]})"
        if any(not t.is_contiguous() for t in (q, k, v, g)):
            return False, "requires contiguous q/k/v/g (no implicit copies)"
        if g.dtype != torch.bfloat16:
            return False, f"requires bfloat16 g (no implicit cast), got {g.dtype}"
        if not use_gate_in_kernel:
            return False, "requires use_gate_in_kernel=True"
        if not safe_gate:
            return False, "requires safe_gate=True"
        if lower_bound is None or not isinstance(lower_bound, (int, float)) or not -5 <= lower_bound < 0:
            return False, f"requires lower_bound in [-5, 0), got {lower_bound!r}"
        if not use_qk_l2norm_in_kernel:
            return False, "always applies qk l2norm in kernel"
        A_log, dt_bias = kwargs.get("A_log"), kwargs.get("dt_bias")
        if A_log is None or dt_bias is None:
            return False, "requires A_log and dt_bias"
        if not isinstance(A_log, torch.Tensor) or not isinstance(dt_bias, torch.Tensor):
            return False, "requires tensor A_log and dt_bias"
        if A_log.device != q.device or dt_bias.device != q.device:
            return False, "requires A_log and dt_bias on the q device"
        if A_log.dtype != torch.float32 or dt_bias.dtype != torch.float32:
            return False, "requires float32 A_log and dt_bias"
        if not A_log.is_contiguous() or not dt_bias.is_contiguous():
            return False, "requires contiguous A_log and dt_bias"
        H, D = q.shape[2], q.shape[3]
        if A_log.shape != (H,) or dt_bias.numel() != H * D:
            return False, f"requires A_log shape ({H},) and {H * D} dt_bias elements"
        if cu_seqlens is not None:
            if not isinstance(cu_seqlens, torch.Tensor):
                return False, "requires tensor cu_seqlens"
            if (
                q.shape[0] != 1
                or cu_seqlens.ndim != 1
                or cu_seqlens.device != q.device
                or cu_seqlens.dtype != torch.int32
                or not cu_seqlens.is_contiguous()
            ):
                return False, "requires packed B=1 and contiguous int32 cu_seqlens on the q device"
        if initial_state is not None:
            if not isinstance(initial_state, torch.Tensor):
                return False, "requires tensor initial_state"
            N = cu_seqlens.numel() - 1 if cu_seqlens is not None else q.shape[0]
            expected = (N, H, D, D)
            if initial_state.device != q.device or initial_state.dtype != torch.float32 or not initial_state.is_contiguous():
                return False, "requires contiguous float32 initial_state on the q device"
            if initial_state.shape != expected:
                return False, f"requires initial_state shape {expected}, got {tuple(initial_state.shape)}"
        cu_seqlens_cpu = kwargs.get("cu_seqlens_cpu")
        if cu_seqlens_cpu is not None:
            if not isinstance(cu_seqlens_cpu, torch.Tensor):
                return False, "requires tensor cu_seqlens_cpu"
            if cu_seqlens_cpu.device.type != "cpu" or cu_seqlens_cpu.dtype != torch.int32:
                return False, "requires int32 CPU cu_seqlens_cpu"
        use_beta_sigmoid = kwargs.get("use_beta_sigmoid_in_kernel", False)
        if not isinstance(use_beta_sigmoid, bool):
            return False, "requires boolean use_beta_sigmoid_in_kernel"
        try:
            CPMode.parse(kwargs.get("use_intracard_cp"))
        except (TypeError, ValueError) as exc:
            return False, str(exc)
        out = kwargs.get("out")
        if out is not None:
            if not isinstance(out, torch.Tensor):
                return False, "requires tensor out"
            if out.device != q.device or out.dtype != torch.bfloat16 or out.shape != v.shape or not out.is_contiguous():
                return False, "requires contiguous bfloat16 out matching v on the q device"
        final_state = kwargs.get("final_state")
        if final_state is not None:
            if not isinstance(final_state, torch.Tensor):
                return False, "requires tensor final_state"
            expected = (cu_seqlens.numel() - 1 if cu_seqlens is not None else q.shape[0], H, D, D)
            if not output_final_state:
                return False, "final_state requires output_final_state=True"
            if (
                final_state.device != q.device
                or final_state.dtype != torch.float32
                or final_state.shape != expected
                or not final_state.is_contiguous()
            ):
                return False, f"requires contiguous float32 final_state shape {expected} on the q device"
        return True, None

    def kda_prefill(self, q, k, v, g, beta, **kwargs):
        from cula.kda.flashkda import cula_kda_prefill

        kwargs.setdefault("use_intracard_cp", "auto")
        return cula_kda_prefill(q, k, v, g, beta, **kwargs)

# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util

import torch

from cula.backends import BaseBackend
from cula.kda.backends._common import verify_common_inputs

_ACCEPTED_KWARGS = {
    "A_log",
    "dt_bias",
    "cu_seqlens_cpu",
    "use_beta_sigmoid_in_kernel",
    "use_intracard_cp",
    "use_cp",
    "out",
    "final_state",
}


def _is_sm90(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (9, 0)


class FullyFusedBackend(BaseBackend):
    backend_type = "fully_fused"
    env_var = "CULA_BACKEND_FULLY_FUSED"
    priority = 2

    def probe(self):
        if importlib.util.find_spec("cula._cudac_sm90") is None:
            return False, "cula._cudac_sm90 extension not built (rebuild with CULA_DISABLE_SM90=0)"
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
        H, HV, D = q.shape[2], v.shape[2], q.shape[3]
        if HV % H != 0:
            return False, f"requires HV to be a multiple of H, got HV={HV} H={H}"
        if not safe_gate:
            return False, "requires safe_gate=True"
        if use_gate_in_kernel:
            if lower_bound is None or not isinstance(lower_bound, (int, float)) or not -5 <= lower_bound < 0:
                return False, f"requires lower_bound in [-5, 0), got {lower_bound!r}"
            A_log = kwargs.get("A_log")
            if A_log is None:
                return False, "requires A_log when use_gate_in_kernel=True"
            if not isinstance(A_log, torch.Tensor):
                return False, "requires tensor A_log"
            if A_log.device != q.device or A_log.dtype != torch.float32 or A_log.shape != (HV,):
                return False, f"requires float32 A_log shape ({HV},) on the q device"
            dt_bias = kwargs.get("dt_bias")
            if dt_bias is not None:
                if not isinstance(dt_bias, torch.Tensor):
                    return False, "requires tensor dt_bias"
                if dt_bias.device != q.device or dt_bias.dtype != torch.float32 or dt_bias.numel() != HV * D:
                    return False, f"requires float32 dt_bias with {HV * D} elements on the q device"
        if cu_seqlens is not None:
            if not isinstance(cu_seqlens, torch.Tensor):
                return False, "requires tensor cu_seqlens"
            if q.shape[0] != 1 or cu_seqlens.ndim != 1 or cu_seqlens.device != q.device or cu_seqlens.dtype != torch.int32:
                return False, "requires packed B=1 and int32 cu_seqlens on the q device"
        if initial_state is not None:
            if not isinstance(initial_state, torch.Tensor):
                return False, "requires tensor initial_state"
            N = cu_seqlens.numel() - 1 if cu_seqlens is not None else q.shape[0]
            expected = (N, HV, D, D)
            if initial_state.device != q.device or initial_state.dtype != torch.float32:
                return False, "requires float32 initial_state on the q device"
            if initial_state.shape != expected:
                return False, f"requires initial_state shape {expected}, got {tuple(initial_state.shape)}"
        if chunk_indices is not None:
            if not isinstance(chunk_indices, torch.Tensor):
                return False, "requires tensor chunk_indices"
            if chunk_indices.device != q.device or chunk_indices.dtype != torch.int32:
                return False, "requires int32 chunk_indices on the q device"
        cu_seqlens_cpu = kwargs.get("cu_seqlens_cpu")
        if cu_seqlens_cpu is not None:
            if not isinstance(cu_seqlens_cpu, torch.Tensor):
                return False, "requires tensor cu_seqlens_cpu"
            if cu_seqlens_cpu.device.type != "cpu" or cu_seqlens_cpu.dtype != torch.int32:
                return False, "requires int32 CPU cu_seqlens_cpu"
        if kwargs.get("use_beta_sigmoid_in_kernel", False) is not False:
            return False, "does not support use_beta_sigmoid_in_kernel=True"
        use_intracard_cp, use_cp = kwargs.get("use_intracard_cp"), kwargs.get("use_cp")
        if use_intracard_cp is not None and use_cp is not None:
            return False, "pass only one of use_intracard_cp or use_cp"
        cp_mode = use_intracard_cp if use_intracard_cp is not None else use_cp
        if cp_mode not in (None, "auto"):
            return False, "supports only automatic intracard CP selection"
        if kwargs.get("out") is not None or kwargs.get("final_state") is not None:
            return False, "does not support preallocated out or final_state buffers"
        return True, None

    def kda_prefill(self, q, k, v, g, beta, **kwargs):
        from cula.kda.auto_route import cula_kda_prefill_auto

        for name in ("use_beta_sigmoid_in_kernel", "use_intracard_cp", "use_cp", "out", "final_state"):
            kwargs.pop(name, None)
        return cula_kda_prefill_auto(q, k, v, g, beta, **kwargs)

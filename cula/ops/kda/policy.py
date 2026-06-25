# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel dispatch policy for KDA wrappers."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

IntracardCPMode = Literal["auto"] | bool


@dataclass(frozen=True)
class IntracardCPDecision:
    enabled: bool
    reason: str | None = None
    force: bool = False


def normalize_intracard_cp_mode(mode: IntracardCPMode) -> IntracardCPMode:
    # Identity checks (not `in`): `1 == True` / `0 == False` would otherwise let stray
    # ints slip past validation and be mishandled downstream (silently treated like "auto").
    if mode != "auto" and mode is not True and mode is not False:
        raise ValueError(f'use_intracard_cp must be "auto", True, or False, got {mode!r}')
    return mode


def resolve_intracard_cp_mode(
    use_intracard_cp: IntracardCPMode | None,
    use_cp_alias: IntracardCPMode | None,
) -> IntracardCPMode | None:
    if use_intracard_cp is not None and use_cp_alias is not None:
        raise TypeError("Pass only one of use_intracard_cp or use_cp.")
    mode = use_intracard_cp if use_intracard_cp is not None else use_cp_alias
    if mode is None:
        # Unspecified: defer to the arch default (SM100 = env-gated CULA_INTRACARD_CP,
        # SM90 = off). Returning None lets each backend keep its legacy default instead
        # of silently forcing CP on.
        return None
    return normalize_intracard_cp_mode(mode)


def _reject_or_disable(mode: IntracardCPMode, reason: str) -> IntracardCPDecision:
    if mode is True:
        raise ValueError(reason)
    return IntracardCPDecision(False, reason)


def _sm100_env_cp_enabled() -> bool:
    """Legacy gate: any CULA_INTRACARD_CP value other than "0" enables CP when the
    caller passes no explicit mode. Matches FLA truthiness."""
    return os.environ.get("CULA_INTRACARD_CP", "0") != "0"


def sm100_intracard_cp_decision(
    *,
    mode: IntracardCPMode | None,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    g: torch.Tensor | None,
    num_qk_heads: int,
    chunk_size: int,
    is_inference: bool,
    sm_count_provider: Callable[[], int],
    no_cp: bool = False,
) -> IntracardCPDecision:
    """Decide whether SM100 intracard CP should run for this call.

    Pure dispatch policy — imports no CuTeDSL kernels. ``mode is None`` defers to
    the legacy env gate (``CULA_INTRACARD_CP``). Hard support constraints (varlen /
    ``g is None`` / inference-only) raise under ``mode is True`` and merely disable
    under ``"auto"``. The perf heuristic (``should_use_intracard_cp``, CPU-only) is
    consulted only on the ``"auto"`` path and imported lazily; ``mode is True``
    bypasses it (the kernel still raises on a truly unsplittable shape).
    """
    if mode is None:
        mode = "auto" if _sm100_env_cp_enabled() else False
    mode = normalize_intracard_cp_mode(mode)
    # no_cp is the intracard-CP recursion guard: intracard_fwd_h splits the sequence and
    # re-invokes fwd_h on the sub-sequences with _no_cp=True so they do not recursively
    # re-trigger CP. It therefore wins over force (mode is True) — a forced top-level
    # request is honored on the outer call (no_cp=False) while the inner recursive calls
    # disable CP. Do NOT raise here on forced+no_cp: that would break the recursion guard
    # if force were ever threaded into the recursive call.
    if mode is False or no_cp:
        return IntracardCPDecision(False, "disabled")

    if cu_seqlens is None:
        return _reject_or_disable(mode, "SM100 intracard CP requires varlen cu_seqlens.")
    if g is not None:
        return _reject_or_disable(mode, "SM100 intracard CP requires g is None; pass gate through gk.")
    if not is_inference:
        return _reject_or_disable(mode, "SM100 intracard CP is inference-only.")

    if mode is True:
        return IntracardCPDecision(True, force=True)

    # auto: consult the CPU-only perf heuristic (lazy import keeps this module free
    # of CuTeDSL imports).
    from cula.ops.kda.sm100.cp.chunk_delta_h import should_use_intracard_cp

    cpu = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.cpu()
    if should_use_intracard_cp(cpu, sm_count_provider(), num_qk_heads, chunk_size):
        return IntracardCPDecision(True)
    return IntracardCPDecision(False, "SM100 intracard CP heuristic declined for this shape.")

# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


def verify_common_inputs(q, k, v, g, beta) -> tuple[bool, str | None]:
    if torch.is_grad_enabled():
        return False, "requires grad-disabled execution"
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or g.ndim != 4 or beta.ndim != 3:
        return False, "requires q/k/v/g rank 4 and beta rank 3"
    if any(t.device != q.device for t in (k, v, g, beta)):
        return False, "requires q/k/v/g/beta on the same device"
    if q.shape != k.shape:
        return False, f"requires matching q/k shapes, got {tuple(q.shape)} and {tuple(k.shape)}"
    if v.shape != g.shape:
        return False, f"requires matching v/g shapes, got {tuple(v.shape)} and {tuple(g.shape)}"
    if q.shape[:2] != v.shape[:2]:
        return False, "requires q/k/v/g to share batch and sequence dimensions"
    if beta.shape != v.shape[:3]:
        return False, f"requires beta shape {tuple(v.shape[:3])}, got {tuple(beta.shape)}"
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return False, f"requires bfloat16 q/k/v, got {q.dtype}/{k.dtype}/{v.dtype}"
    if beta.dtype not in (torch.bfloat16, torch.float32):
        return False, f"requires bfloat16 or float32 beta, got {beta.dtype}"
    if q.shape[-1] != 128 or v.shape[-1] != 128:
        return False, f"requires K=V=128, got K={q.shape[-1]} V={v.shape[-1]}"
    if q.shape[2] <= 0 or v.shape[2] <= 0:
        return False, "requires positive q/k and v/g head counts"
    return True, None

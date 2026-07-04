# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Shared low-level helpers for the SM90 FlashKDA kernels.

"""


import weakref

import cutlass
import torch
from cutlass import Int32
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T as _T


# ============================================================================
# NVVM / inline-PTX helpers
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


# ============================================================================
# Input wrapping
# ============================================================================
_WRAP_CACHE: dict = {}
_WRAP_CACHE_MAXSIZE = 512


def _wrap_input(t: torch.Tensor, align: int, *, view_shape=None, cache: bool = False):
    """Wrap a tensor as a CuTe tensor via from_dlpack.

    ``cache=True``: reuse across launches, keyed by (id, _version, align, view_shape)
    and verified by weakref. 
    
    Use ``cache=False`` for per-call buffers (workspaces, states).
    """
    if not cache:
        src = t if view_shape is None else t.view(view_shape)
        return from_dlpack(src.detach(), assumed_align=align)
    ckey = (id(t), t._version, align, view_shape)
    entry = _WRAP_CACHE.get(ckey)
    if entry is not None and entry[0]() is t:
        return entry[1]
    src = t if view_shape is None else t.view(view_shape)
    w = from_dlpack(src.detach(), assumed_align=align)
    if len(_WRAP_CACHE) >= _WRAP_CACHE_MAXSIZE:
        _WRAP_CACHE.pop(next(iter(_WRAP_CACHE)))
    _WRAP_CACHE[ckey] = (weakref.ref(t), w)
    return w

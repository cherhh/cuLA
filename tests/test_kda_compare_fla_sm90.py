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

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Precision tests for the SM90 (Hopper) CuTeDSL FlashKDA prefill vs FLA baseline.

Forward-only: the SM90 path has no backward, so only the output ``o`` and the
final state ``ht`` are compared.

NOTE: the SM90 kernel keeps its KV recurrence state in **VK-transposed** layout
(shape ``[B, H, V, K]``) while FLA uses the standard KV layout (``[B, H, K, V]``).
The test transposes ``initial_state`` when passing it to cuLA, and transposes
cuLA's ``ht`` back to KV layout before comparison.
"""

import pytest
import torch
from fla.ops.kda.chunk import chunk_kda as fla_chunk_kda
from fla.utils import assert_close, device

from cula.kda import kda_prefill_hopper as cula_kda_prefill

pytestmark = [
    pytest.mark.sm90_only,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

D = 128
_LOWER_BOUND = -5.0
_RTOL = 0.005
_KWARGS = dict(
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    safe_gate=True,
    lower_bound=_LOWER_BOUND,
)


def _make_inputs(B, T, H, *, with_state, n_state=None, seed=42):
    torch.manual_seed(seed)
    q = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    k = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    v = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    g = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(H, dtype=torch.float32, device=device)
    dt_bias = torch.randn(H * D, dtype=torch.float32, device=device)
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid().to(torch.bfloat16)
    # One initial state per sequence: dense -> B, packed varlen -> len(cu_seqlens) - 1.
    n_state = B if n_state is None else n_state
    h0 = torch.randn(n_state, H, D, D, dtype=torch.float32, device=device) if with_state else None
    return q, k, v, g, beta, A_log, dt_bias, h0


# (B, T, H, with_state) — non-CHUNK-aligned T exercises tail-chunk handling (CHUNK=16).
_DENSE = [
    pytest.param(1, 256, 2, False, id="B1-T256-H2-aligned"),
    pytest.param(1, 500, 2, False, id="B1-T500-H2-tail"),
    pytest.param(1, 512, 2, True, id="B1-T512-H2-init_state"),
    pytest.param(1, 1000, 4, False, marks=pytest.mark.kda_slow, id="B1-T1000-H4-tail"),
    pytest.param(2, 512, 2, False, marks=pytest.mark.kda_slow, id="B2-T512-H2-stride"),
    pytest.param(2, 1024, 4, False, marks=pytest.mark.kda_slow, id="B2-T1024-H4"),
    pytest.param(1, 1024, 2, True, marks=pytest.mark.kda_slow, id="B1-T1024-H2-init_state"),
]


@pytest.mark.parametrize(("B", "T", "H", "with_state"), _DENSE)
def test_sm90_prefill_dense_matches_fla(B, T, H, with_state):
    q, k, v, g, beta, A_log, dt_bias, h0 = _make_inputs(B, T, H, with_state=with_state)

    with torch.no_grad():
        ref_o, ref_ht = fla_chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=h0,
            cu_seqlens=None,
            **_KWARGS,
        )

        # SM90 uses VK-transposed state layout: swap last two dims.
        h0_vk = h0.transpose(-2, -1).contiguous() if h0 is not None else None
        tri_o, tri_ht_vk = cula_kda_prefill(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=h0_vk,
            cu_seqlens=None,
            **_KWARGS,
        )
    tri_ht = tri_ht_vk.transpose(-2, -1)

    assert_close("o", ref_o, tri_o, _RTOL)
    assert_close("ht", ref_ht, tri_ht, _RTOL)


# Packed varlen (B=1), cu_seqlens includes non-CHUNK-aligned segment lengths.
_VARLEN = [
    pytest.param([0, 17, 48], 2, False, id="cu[0,17,48]-H2"),
    pytest.param([0, 256, 500, 1000], 4, False, marks=pytest.mark.kda_slow, id="cu[0,256,500,1000]-H4"),
    pytest.param([0, 15, 100, 300, 1200], 4, True, marks=pytest.mark.kda_slow, id="cu[0,15,100,300,1200]-H4-init"),
]


@pytest.mark.parametrize(("cu_seqlens", "H", "with_state"), _VARLEN)
def test_sm90_prefill_varlen_matches_fla(cu_seqlens, H, with_state):
    total_t = cu_seqlens[-1]
    q, k, v, g, beta, A_log, dt_bias, h0 = _make_inputs(1, total_t, H, with_state=with_state, n_state=len(cu_seqlens) - 1)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    with torch.no_grad():
        ref_o, ref_ht = fla_chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=h0,
            cu_seqlens=cu,
            **_KWARGS,
        )

        h0_vk = h0.transpose(-2, -1).contiguous() if h0 is not None else None
        tri_o, tri_ht_vk = cula_kda_prefill(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=h0_vk,
            cu_seqlens=cu,
            **_KWARGS,
        )
    tri_ht = tri_ht_vk.transpose(-2, -1)

    assert_close("o", ref_o, tri_o, _RTOL)
    assert_close("ht", ref_ht, tri_ht, _RTOL)

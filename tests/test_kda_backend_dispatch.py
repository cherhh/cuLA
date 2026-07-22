# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch

from cula.backends import BackendRegistry
from cula.kda import backends as dispatch
from cula.kda.backends import flashkda as flashkda_backend
from cula.kda.backends import fully_fused as fully_fused_backend

D = 128


def _inputs(H=2, HV=None):
    HV = H if HV is None else HV
    q = torch.empty((1, 8, H, D), dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty((1, 8, HV, D), dtype=torch.bfloat16)
    g = torch.empty_like(v)
    beta = torch.empty((1, 8, HV), dtype=torch.bfloat16)
    return q, k, v, g, beta


def _gate_args(H):
    return {
        "A_log": torch.empty(H, dtype=torch.float32),
        "dt_bias": torch.empty((H, D), dtype=torch.float32),
    }


@pytest.fixture
def registered_backends(monkeypatch):
    flash = flashkda_backend.FlashKDABackend()
    fully_fused = fully_fused_backend.FullyFusedBackend()
    flash._available = (True, None)
    fully_fused._available = (True, None)
    registry = BackendRegistry("kda_prefill")
    registry.register(flash)
    registry.register(fully_fused)
    monkeypatch.setattr(dispatch, "kda_prefill_registry", registry)
    monkeypatch.setattr(flashkda_backend, "_is_sm90", lambda device: True)
    monkeypatch.setattr(fully_fused_backend, "_is_sm90", lambda device: True)
    monkeypatch.delenv("CULA_BACKEND_FLASHKDA", raising=False)
    monkeypatch.delenv("CULA_BACKEND_FULLY_FUSED", raising=False)
    return flash, fully_fused


def test_grad_enabled_inputs_without_grad_dispatch(registered_backends, monkeypatch):
    flash, _ = registered_backends
    monkeypatch.setattr(flash, "kda_prefill", lambda *args, **kwargs: "flashkda")
    q, k, v, g, beta = _inputs()

    assert torch.is_grad_enabled()
    assert not any(t.requires_grad for t in (q, k, v, g, beta))
    assert dispatch.kda_prefill(q, k, v, g, beta, **_gate_args(2)) == "flashkda"


def test_real_numeric_lower_bound_is_accepted(registered_backends, monkeypatch):
    np = pytest.importorskip("numpy")
    flash, _ = registered_backends
    monkeypatch.setattr(flash, "kda_prefill", lambda *args, **kwargs: "flashkda")
    q, k, v, g, beta = _inputs()

    result = dispatch.kda_prefill(q, k, v, g, beta, lower_bound=np.float32(-2.5), **_gate_args(2))

    assert result == "flashkda"


def test_cp_off_falls_back_without_enabling_fully_fused_cp(registered_backends, monkeypatch):
    auto_route = importlib.import_module("cula.kda.auto_route")
    captured = {}

    def fake_opt(**kwargs):
        captured.update(kwargs)
        return "fully_fused"

    monkeypatch.setattr(auto_route, "_should_use_opt", lambda q, cu_seqlens: True)
    monkeypatch.setattr(auto_route, "_opt", fake_opt)
    q, k, v, g, beta = _inputs(H=2, HV=4)

    result = dispatch.kda_prefill(q, k, v, g, beta, use_intracard_cp=False, **_gate_args(4))

    assert result == "fully_fused"
    assert captured["auto_cp"] is False


def test_unknown_kwargs_fail_at_public_boundary(registered_backends, monkeypatch):
    def fail_dispatch(*args, **kwargs):
        raise AssertionError("registry should not receive unknown kwargs")

    monkeypatch.setattr(dispatch.kda_prefill_registry, "dispatch", fail_dispatch)
    q, k, v, g, beta = _inputs()

    with pytest.raises(TypeError, match="unexpected keyword arguments.*typo"):
        dispatch.kda_prefill(q, k, v, g, beta, typo=True, **_gate_args(2))


@pytest.mark.sm90_only
@pytest.mark.kda_fast
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_public_dispatch_matches_direct_flashkda(monkeypatch):
    from cula.kda import flashkda_prefill, kda_prefill

    monkeypatch.delenv("CULA_BACKEND_FLASHKDA", raising=False)
    torch.manual_seed(3)
    H = 4
    q = torch.randn((1, 512, H, D), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q)
    beta = torch.rand((1, 512, H), device="cuda", dtype=torch.bfloat16)
    A_log = torch.rand(H, device="cuda", dtype=torch.float32)
    dt_bias = torch.rand((H, D), device="cuda", dtype=torch.float32)
    common = {
        "A_log": A_log,
        "dt_bias": dt_bias,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "safe_gate": True,
        "lower_bound": -5.0,
    }

    assert torch.is_grad_enabled()
    out, final_state = kda_prefill(q, k, v, g, beta, **common)
    expected_out, expected_final_state = flashkda_prefill(
        q,
        k,
        v,
        g,
        beta,
        use_intracard_cp="auto",
        **common,
    )

    assert torch.equal(out, expected_out)
    assert torch.equal(final_state, expected_final_state)

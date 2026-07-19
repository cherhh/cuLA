# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from cula.backends import BackendRegistry, BaseBackend
from cula.kda.backends import flashkda as flashkda_backend
from cula.kda.backends import fully_fused as fully_fused_backend
from cula.kda.backends import kda_prefill_registry

D = 128


class _Dummy(BaseBackend):
    def __init__(self, name, priority, ok=True, reason=None, available=True):
        self.backend_type = name
        self.priority = priority
        self._ok, self._reason, self._avail = ok, reason, available
        self.calls = 0

    def probe(self):
        return (True, None) if self._avail else (False, "not built")

    def op_verifier(self, *a, **k):
        return self._ok, self._reason

    def op(self, *a, **k):
        self.calls += 1
        return self.backend_type


def test_priority_order():
    reg = BackendRegistry("t")
    lo, hi = _Dummy("lo", 5), _Dummy("hi", 1)
    reg.register(lo)
    reg.register(hi)
    assert reg.dispatch("op") == "hi"
    assert hi.calls == 1 and lo.calls == 0


def test_falls_through_rejection_and_unavailability():
    reg = BackendRegistry("t")
    reg.register(_Dummy("a", 1, ok=False, reason="wrong dtype"))
    reg.register(_Dummy("b", 2, available=False))
    reg.register(_Dummy("c", 3))
    assert reg.dispatch("op") == "c"


def test_env_disable(monkeypatch):
    reg = BackendRegistry("t")
    first = _Dummy("first", 1)
    first.env_var = "TEST_CULA_FIRST"
    reg.register(first)
    reg.register(_Dummy("second", 2))
    monkeypatch.setenv("TEST_CULA_FIRST", "0")
    assert reg.dispatch("op") == "second"
    monkeypatch.setenv("TEST_CULA_FIRST", "1")
    assert reg.dispatch("op") == "first"


def test_all_rejected_raises_with_reasons():
    reg = BackendRegistry("t")
    reg.register(_Dummy("a", 1, ok=False, reason="wrong dtype"))
    reg.register(_Dummy("b", 2, available=False))
    with pytest.raises(RuntimeError, match="wrong dtype") as ei:
        reg.dispatch("op")
    assert "not built" in str(ei.value)


def test_verifier_error_propagates():
    class _Broken(_Dummy):
        def op_verifier(self, *a, **k):
            raise ValueError("broken verifier")

    reg = BackendRegistry("t")
    reg.register(_Broken("broken", 1))
    with pytest.raises(ValueError, match="broken verifier"):
        reg.dispatch("op")


@pytest.fixture
def sm90(monkeypatch):
    monkeypatch.setattr(flashkda_backend, "_is_sm90", lambda device: True)
    return flashkda_backend.FlashKDABackend()


@pytest.fixture
def fully_fused(monkeypatch):
    monkeypatch.setattr(fully_fused_backend, "_is_sm90", lambda device: True)
    return fully_fused_backend.FullyFusedBackend()


def _fake(dtype=torch.bfloat16, H=2, HV=None, K=D, beta_dtype=None):
    HV = H if HV is None else HV
    q = torch.empty(1, 1, 1, 1, dtype=dtype).expand(1, 64, H, K)
    v = torch.empty(1, 1, 1, 1, dtype=dtype).expand(1, 64, HV, K)
    beta = torch.empty(1, 64, HV, dtype=beta_dtype or dtype)
    return q, q, v, v, beta


def _good(H=2):
    return {
        "A_log": torch.empty(H, dtype=torch.float32),
        "dt_bias": torch.empty(H, D, dtype=torch.float32),
        "lower_bound": -5.0,
    }


def _verify(backend, *args, **kwargs):
    with torch.inference_mode():
        return backend.kda_prefill_verifier(*args, **kwargs)


def test_flashkda_accepts_good_call(sm90):
    ok, why = _verify(sm90, *_fake(), **_good())
    assert ok, why


def test_flashkda_rejects_dtype(sm90):
    ok, why = _verify(sm90, *_fake(dtype=torch.float32), **_good())
    assert not ok and "bfloat16" in why


def test_flashkda_rejects_gva(sm90):
    ok, why = _verify(sm90, *_fake(HV=4), **_good())
    assert not ok and "GVA" in why


def test_flashkda_rejects_unsafe_gate_and_unknown_kwargs(sm90):
    ok, why = _verify(sm90, *_fake(), safe_gate=False, **_good())
    assert not ok and "safe_gate" in why
    ok, why = _verify(sm90, *_fake(), transpose_state_layout=True, **_good())
    assert not ok and "transpose_state_layout" in why


def test_flashkda_rejects_missing_gate_params(sm90):
    ok, why = _verify(sm90, *_fake(), lower_bound=-5.0)
    assert not ok and "A_log" in why


def test_flashkda_rejects_grad_and_invalid_lower_bound(sm90):
    ok, why = sm90.kda_prefill_verifier(*_fake(), **_good())
    assert not ok and "grad-disabled" in why
    ok, why = _verify(sm90, *_fake(), **(_good() | {"lower_bound": 0.0}))
    assert not ok and "lower_bound" in why


def test_flashkda_preserves_cp_alias(monkeypatch, sm90):
    monkeypatch.setattr("cula.kda.flashkda.cula_kda_prefill", lambda *args, **kwargs: kwargs)
    kwargs = sm90.kda_prefill(*_fake(), use_cp=True)
    assert kwargs["use_cp"] is True
    assert "use_intracard_cp" not in kwargs


def test_fully_fused_accepts_gva(fully_fused):
    ok, why = _verify(fully_fused, *_fake(HV=4), **_good(H=4))
    assert ok, why


def test_fully_fused_rejects_grad(fully_fused):
    ok, why = fully_fused.kda_prefill_verifier(*_fake(), **_good())
    assert not ok and "grad-disabled" in why


@pytest.mark.parametrize(
    "unsupported",
    [
        {"use_beta_sigmoid_in_kernel": True},
        {"use_intracard_cp": True},
        {"out": object()},
        {"final_state": object()},
        {"unknown_option": True},
    ],
)
def test_fully_fused_rejects_unsupported_options(fully_fused, unsupported):
    ok, why = _verify(fully_fused, *_fake(), **(_good() | unsupported))
    assert not ok, why


def test_real_verifiers_fall_through_to_fully_fused(monkeypatch):
    monkeypatch.setattr(flashkda_backend, "_is_sm90", lambda device: True)
    monkeypatch.setattr(fully_fused_backend, "_is_sm90", lambda device: True)
    flashkda = flashkda_backend.FlashKDABackend()
    fully_fused = fully_fused_backend.FullyFusedBackend()
    flashkda._available = (True, None)
    fully_fused._available = (True, None)
    monkeypatch.setattr(flashkda, "kda_prefill", lambda *args, **kwargs: "flashkda")
    monkeypatch.setattr(fully_fused, "kda_prefill", lambda *args, **kwargs: "fully_fused")
    reg = BackendRegistry("kda_prefill")
    reg.register(flashkda)
    reg.register(fully_fused)
    with torch.inference_mode():
        assert reg.dispatch("kda_prefill", *_fake(HV=4), **_good(H=4)) == "fully_fused"


def test_registered_backends():
    names = [b.backend_type for b in kda_prefill_registry.backends()]
    assert names == ["flashkda", "fully_fused"]


@pytest.mark.sm90_only
@pytest.mark.kda_fast
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_dispatch_picks_flashkda_and_matches_direct():
    from cula.kda import flashkda_prefill, kda_prefill

    torch.manual_seed(3)
    q = torch.randn(1, 512, 4, D, dtype=torch.bfloat16, device="cuda")
    k, v, g = torch.randn_like(q), torch.randn_like(q), torch.randn_like(q)
    beta = torch.randn(1, 512, 4, dtype=torch.bfloat16, device="cuda")
    A_log = torch.rand(4, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(4, D, dtype=torch.float32, device="cuda")
    common = dict(A_log=A_log, dt_bias=dt_bias, output_final_state=True, safe_gate=True, lower_bound=-5.0)

    with torch.inference_mode():
        o, ht = kda_prefill(q, k, v, g, beta, **common)
        o_ref, ht_ref = flashkda_prefill(
            q,
            k,
            v,
            g,
            beta,
            use_gate_in_kernel=True,
            use_intracard_cp="auto",
            **common,
        )
    assert torch.equal(o, o_ref) and torch.equal(ht, ht_ref)


@pytest.mark.sm90_only
@pytest.mark.kda_fast
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_dispatch_picks_fully_fused_and_matches_direct(monkeypatch):
    from cula.kda import kda_prefill, kda_prefill_hopper_auto

    monkeypatch.setenv("CULA_BACKEND_FLASHKDA", "0")
    torch.manual_seed(5)
    q = torch.randn(1, 512, 4, D, dtype=torch.bfloat16, device="cuda")
    k, v, g = torch.randn_like(q), torch.randn_like(q), torch.randn_like(q)
    beta = torch.randn(1, 512, 4, dtype=torch.bfloat16, device="cuda")
    A_log = torch.rand(4, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(4, D, dtype=torch.float32, device="cuda")
    common = dict(A_log=A_log, dt_bias=dt_bias, output_final_state=True, safe_gate=True, lower_bound=-5.0)

    with torch.inference_mode():
        o, ht = kda_prefill(q, k, v, g, beta, **common)
        o_ref, ht_ref = kda_prefill_hopper_auto(q, k, v, g, beta, **common)
    assert torch.equal(o, o_ref) and torch.equal(ht, ht_ref)

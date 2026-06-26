import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from cula.ops.kda.policy import (
    resolve_intracard_cp_mode,
    sm90_intracard_cp_decision,
    sm100_intracard_cp_decision,
)


def test_intracard_cp_mode_resolution():
    assert resolve_intracard_cp_mode(None, None) is None  # unspecified → arch default (SM100 env / SM90 off)
    assert resolve_intracard_cp_mode("auto", None) == "auto"
    assert resolve_intracard_cp_mode(False, None) is False
    assert resolve_intracard_cp_mode(None, True) is True

    with pytest.raises(TypeError):
        resolve_intracard_cp_mode("auto", False)
    with pytest.raises(ValueError):
        resolve_intracard_cp_mode("invalid", None)


def test_sm90_policy_accepts_unaligned_without_cuda_probe():
    # Non-CHUNK-aligned lengths are now supported by the backend (pad-before-CP), so
    # the policy no longer rejects them: _sm90_seq_tiles returns ceil tile counts on
    # CPU without a CUDA probe, and force (True) does NOT raise on alignment.
    from cula.ops.kda.policy import _sm90_seq_tiles

    q_dense = torch.empty(1, 17, 1, 128)  # T=17 -> ceil(17/16)=2 tiles
    assert _sm90_seq_tiles(q_dense, None, None, True) == [2]

    cu = _cu(17, 63)  # varlen non-aligned -> ceil per seq: 2, 4
    q_packed = torch.empty(1, 80, 1, 128)
    assert _sm90_seq_tiles(q_packed, cu, cu, True) == [2, 4]

    # The one remaining CPU-only rejection (packed varlen must be B=1) still force-raises.
    with pytest.raises(ValueError, match="packed B=1"):
        _sm90_seq_tiles(torch.empty(2, 80, 1, 128), cu, cu, True)


def test_sm90_policy_none_mode_disabled_without_cuda_probe():
    # Unspecified (None) must default to OFF for SM90 and short-circuit before any
    # CUDA probe — even for a CHUNK-aligned shape that would otherwise be planned.
    q = torch.empty(1, 2048, 1, 128)
    decision = sm90_intracard_cp_decision(q, None, None, None)
    assert decision.enabled is False
    assert decision.reason == "disabled"


def test_policy_import_does_not_import_sm90_cp_kernel():
    env = dict(os.environ)
    repo_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in [repo_root, env.get("PYTHONPATH", "")] if p)
    code = (
        "import sys; "
        "from cula.ops.kda.policy import resolve_intracard_cp_mode; "
        "assert resolve_intracard_cp_mode(None, False) is False; "
        "raise SystemExit('cula.ops.kda.sm90.cp.driver' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env)
    assert result.returncode == 0


def _cu(*lens):
    vals = [0]
    for n in lens:
        vals.append(vals[-1] + n)
    return torch.tensor(vals, dtype=torch.int32)


_SM100_COMMON = dict(num_qk_heads=4, chunk_size=64, sm_count_provider=lambda: 132)


def test_sm100_decision_hard_constraints_force_vs_auto():
    cu = _cu(64)
    # non-varlen: auto disables, force raises (rejected before any kernel import)
    assert (
        sm100_intracard_cp_decision(
            mode="auto", cu_seqlens=None, cu_seqlens_cpu=None, g=None, is_inference=True, **_SM100_COMMON
        ).enabled
        is False
    )
    with pytest.raises(ValueError, match="varlen"):
        sm100_intracard_cp_decision(
            mode=True, cu_seqlens=None, cu_seqlens_cpu=None, g=None, is_inference=True, **_SM100_COMMON
        )
    # inference-only
    with pytest.raises(ValueError, match="inference"):
        sm100_intracard_cp_decision(mode=True, cu_seqlens=cu, cu_seqlens_cpu=cu, g=None, is_inference=False, **_SM100_COMMON)
    # g must be None (gate goes through gk)
    with pytest.raises(ValueError, match="g is None"):
        sm100_intracard_cp_decision(
            mode=True, cu_seqlens=cu, cu_seqlens_cpu=cu, g=torch.zeros(1), is_inference=True, **_SM100_COMMON
        )


def test_sm100_decision_mode_and_env(monkeypatch):
    cu = _cu(64)
    kw = dict(cu_seqlens=cu, cu_seqlens_cpu=cu, g=None, is_inference=True, **_SM100_COMMON)
    # explicit False / no_cp → disabled (no env consulted)
    assert sm100_intracard_cp_decision(mode=False, **kw).reason == "disabled"
    assert sm100_intracard_cp_decision(mode="auto", no_cp=True, **kw).reason == "disabled"
    # unspecified (None) defers to env: off → disabled
    monkeypatch.delenv("CULA_INTRACARD_CP", raising=False)
    assert sm100_intracard_cp_decision(mode=None, **kw).reason == "disabled"
    # None + env on → "auto" (proven via varlen rejection — no kernel import reached)
    monkeypatch.setenv("CULA_INTRACARD_CP", "1")
    decision = sm100_intracard_cp_decision(
        mode=None, cu_seqlens=None, cu_seqlens_cpu=None, g=None, is_inference=True, **_SM100_COMMON
    )
    assert decision.enabled is False and "varlen" in decision.reason

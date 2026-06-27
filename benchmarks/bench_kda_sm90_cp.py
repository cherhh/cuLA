#!/usr/bin/env python3
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

"""
bench_kda_sm90_cp.py — Benchmark: SM90 intracard context-parallel (CP) prefill.

Intracard CP only engages at LOW occupancy — FEW heads x SMALL batch x LONG
sequence(s) — where the serial K1+K2 path leaves the SM array idle. This bench
targets exactly that regime and compares MATCHED paths:

  - non-CP : cuLA serial            vs FLA Triton (no CP)
  - CP     : cuLA intracard CP (auto) vs FLA intracard CP

FLA's intracard CP is gated by FLA_INTRACARD_CP and only runs under
torch.inference_mode(); cuLA's auto policy decides engage vs fall back.
At high head counts both fall back to serial (covered by bench_kda_sm90_prefill).

Usage:
  python benchmarks/bench_kda_sm90_cp.py [--ncu] [--sanitizer]
"""

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("CULA_INTRACARD_CP", "1")
os.environ["FLA_INTRACARD_CP"] = "1"  # enable FLA's intracard-CP backend (before fla import)
os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))

import torch
from fla.ops.kda import chunk_kda as fla_chunk_kda

from benchmarks.utils import (
    SEED,
    benchmark_cuda_mode_fn,
    exclusive_cumsum,
    relative_rms_error_rel_max_mean_abs,
    set_seed,
)
from cula.kda import kda_prefill_hopper as cula_kda_prefill
from cula.ops.kda.policy import sm90_intracard_cp_decision
from cula.utils import assert_hopper, get_device_sm_version

_device = torch.device("cuda")
assert_hopper(_device)
_major, _minor = get_device_sm_version(_device)
_SM_TAG = f"sm{_major}{_minor}"

D = 128
WARMUP = 20
N_ITERS = 50
NCU_MODE = False
SANITIZER_MODE = False

# CP regime: few heads (H>=4, the realistic minimum after tensor parallelism),
# small batch (B=1), long sequence(s). Last row is a high-H control where both
# fall back to serial.
CONFIGS = [
    ("H=4  T=16384", 4, [16384]),
    ("H=8  T=16384", 8, [16384]),
    ("H=4  T=32768", 4, [32768]),
    ("H=8  T=32768", 8, [32768]),
    ("H=4  T=65536", 4, [65536]),
    ("H=8  T=65536", 8, [65536]),
    ("H=4  2-seq T=32768", 4, [16384, 16384]),
    ("CTRL H=64 T=16384", 64, [16384]),
]


def _make(H, seq_lens):
    set_seed(SEED)
    T = sum(seq_lens)
    rnd = lambda: torch.rand(1, T, H, D, dtype=torch.bfloat16, device=_device)
    raw_beta = torch.randn(1, T, H, dtype=torch.bfloat16, device=_device)
    return dict(
        q=rnd(),
        k=rnd(),
        v=rnd(),
        g=torch.randn(1, T, H, D, dtype=torch.bfloat16, device=_device) * 0.1,
        sig_beta=raw_beta.float().sigmoid().to(torch.bfloat16),
        raw_beta=raw_beta,
        A_log=torch.randn(H, dtype=torch.float32, device=_device) * 0.01,
        dt_bias=torch.zeros(H * D, dtype=torch.float32, device=_device),
        cu=torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=_device),
        scale=D**-0.5,
    )


def _cula(d, mode):  # public wrapper takes post-sigmoid beta
    o, _ = cula_kda_prefill(
        q=d["q"],
        k=d["k"],
        v=d["v"],
        g=d["g"],
        beta=d["sig_beta"],
        scale=d["scale"],
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        initial_state=None,
        output_final_state=True,
        cu_seqlens=d["cu"],
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        use_intracard_cp=mode,
    )
    return o


def _fla(d):  # FLA's intracard CP engages only under inference_mode (+ FLA_INTRACARD_CP)
    o, _ = fla_chunk_kda(
        q=d["q"],
        k=d["k"],
        v=d["v"],
        g=d["g"],
        beta=d["raw_beta"],
        scale=d["scale"],
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        initial_state=None,
        output_final_state=True,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        cu_seqlens=d["cu"],
        safe_gate=True,
        lower_bound=-5.0,
        transpose_state_layout=True,
    )
    return o


def _time(fn):
    return benchmark_cuda_mode_fn(
        fn,
        default_warmup=WARMUP,
        default_rep=N_ITERS,
        ncu_mode=NCU_MODE,
        sanitizer_mode=SANITIZER_MODE,
        aggregate="iqr_mean",
    )


def main():
    parser = argparse.ArgumentParser(description="bench_kda_sm90_cp: SM90 intracard CP, matched cuLA-vs-FLA")
    parser.add_argument("--ncu", action="store_true", help="NCU mode: warmup=1, iters=1")
    parser.add_argument("--sanitizer", action="store_true", help="Sanitizer mode: warmup=1, iters=1")
    args = parser.parse_args()
    global NCU_MODE, SANITIZER_MODE
    NCU_MODE = args.ncu
    SANITIZER_MODE = args.sanitizer

    print(f"[Device] {torch.cuda.get_device_name(0)}  {_SM_TAG}  D={D}  dtype=bf16")
    print("  SM90 intracard CP — matched comparison (non-CP vs non-CP, CP vs CP)\n")
    print(
        f"  {'config':20s} │ {'cuLA_ser':>8s} {'cuLA_CP':>8s} {'FLA_ser':>8s} {'FLA_CP':>8s} │ "
        f"{'ser c/f':>8s} {'CP c/f':>8s} │ {'cuLA_dec':>8s} {'rrmse':>8s}"
    )
    print("  " + "─" * 104)

    for label, H, sl in CONFIGS:
        d = _make(H, sl)
        dec = sm90_intracard_cp_decision(d["q"], d["cu"], None, "auto")
        with torch.no_grad():
            o_cs = _cula(d, False)
            o_cc = _cula(d, "auto")
            torch.cuda.synchronize()
            rr, _, _ = relative_rms_error_rel_max_mean_abs(o_cs, o_cc)
            ms_cs = _time(lambda: _cula(d, False))
            ms_cc = _time(lambda: _cula(d, "auto"))
            ms_fs = _time(lambda: _fla(d))  # FLA non-CP (not inference_mode -> backend declines)
        with torch.inference_mode():
            ms_fc = _time(lambda: _fla(d))  # FLA intracard CP (inference_mode + FLA_INTRACARD_CP)
        print(
            f"  {label:20s} │ {ms_cs:8.3f} {ms_cc:8.3f} {ms_fs:8.3f} {ms_fc:8.3f} │ "
            f"{ms_fs / ms_cs:7.2f}x {ms_fc / ms_cc:7.2f}x │ {'ENGAGE' if dec.enabled else 'fallbk':>8s} {rr:8.5f}"
        )
        del d
        torch.cuda.empty_cache()

    print("  " + "─" * 104)
    print("  ser c/f = FLA_ser/cuLA_ser, CP c/f = FLA_CP/cuLA_CP  (>1 = cuLA faster)")
    print("  CP engages only at low H/batch + long seq; high-H control falls back to serial.\n")


if __name__ == "__main__":
    main()

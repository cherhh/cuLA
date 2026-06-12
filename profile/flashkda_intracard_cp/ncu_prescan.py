# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""S7 NCU target: one warmed-up CP call so ncu can profile k2_prescan.

Run:
  ncu --kernel-name regex:k2_prescan --launch-count 1 \
      --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis \
      --section ComputeWorkloadAnalysis --section WarpStateStats \
      python profile/flashkda_intracard_cp/ncu_prescan.py
"""

import os
import sys

os.environ.setdefault("CULA_FLASHKDA_USE_CUTE", "1")
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import torch  # noqa: E402

from cula.ops.flashkda.cp import flash_kda_prefill_cp  # noqa: E402
from cula.ops.flashkda.k2 import D  # noqa: E402

T, H, S_SPLIT = 32768, 8, 16  # TP8 deployment headline config

torch.manual_seed(0)
dev = torch.device("cuda")
mk = lambda *shape: torch.randn(*shape, dtype=torch.bfloat16, device=dev)
q, k, v, g = (mk(1, T, H, D) for _ in range(4))
beta = mk(1, T, H)
A_log = torch.randn(H, dtype=torch.float32, device=dev)
dt_bias = torch.randn(H, D, dtype=torch.float32, device=dev)
out = torch.empty_like(v)
fin = torch.empty(1, H, D, D, dtype=torch.float32, device=dev)

run = lambda: flash_kda_prefill_cp(
    q, k, v, g, beta, scale=D**-0.5, out=out, A_log=A_log, dt_bias=dt_bias,
    lower_bound=-5.0, final_state=fin, s_split=S_SPLIT)

run()  # compile + warm caches outside profiling
torch.cuda.synchronize()
run()  # profiled iteration
torch.cuda.synchronize()
print("ok", out.float().abs().mean().item())

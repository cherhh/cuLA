"""Profiling harness for K2 — single kernel launch for ncu capture.

Usage:
  ncu [flags] -k "regex:k2_kernel" -c 1 python profile_k2.py <T> <H>
"""
import sys
import torch

sys.path.insert(0, "/ossfs/workspace/chenhao/cula")

from cula.ops.sm90.flashkda.k1 import CHUNK, D, launch_k1
from cula.ops.sm90.flashkda.k2 import launch_k2

T = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
H = int(sys.argv[2]) if len(sys.argv) > 2 else 64

g = torch.Generator(device="cuda").manual_seed(0)
q = torch.randn(1, T, H, D, generator=g, device="cuda", dtype=torch.bfloat16) * 0.5
k = torch.randn(1, T, H, D, generator=g, device="cuda", dtype=torch.bfloat16) * 0.5
v = torch.randn(1, T, H, D, generator=g, device="cuda", dtype=torch.bfloat16) * 0.5
g_pre = torch.randn(1, T, H, D, generator=g, device="cuda", dtype=torch.bfloat16) * 0.1
beta_raw = torch.randn(1, T, H, generator=g, device="cuda", dtype=torch.bfloat16) * 0.1
A_log = torch.rand(H, generator=g, device="cuda", dtype=torch.float32)
dt_bias = torch.rand(H, D, generator=g, device="cuda", dtype=torch.float32)

beta_flat = beta_raw.view(T, H).permute(1, 0).contiguous().view(-1)
total_tiles = T // CHUNK
n_qk = total_tiles * H * CHUNK * D
n_cc = total_tiles * H * CHUNK * CHUNK
ws_qd = torch.empty(n_qk, dtype=torch.bfloat16, device="cuda")
ws_kd = torch.empty_like(ws_qd)
ws_kr = torch.empty_like(ws_qd)
ws_gt = torch.empty(total_tiles * H * D, dtype=torch.float32, device="cuda")
ws_inv = torch.empty(n_cc, dtype=torch.bfloat16, device="cuda")
ws_mqk = torch.empty_like(ws_inv)

launch_k1(q, k, g_pre, A_log, dt_bias, beta_flat, D ** -0.5, -1.0,
               ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk)
torch.cuda.synchronize()

out = torch.empty_like(v)

# Warmup to trigger JIT compilation
launch_k2(v, beta_flat, ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk, out)
torch.cuda.synchronize()

# The profiled launch
launch_k2(v, beta_flat, ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk, out)
torch.cuda.synchronize()

print(f"[harness] K2 done: T={T}, H={H}, out shape={out.shape}")

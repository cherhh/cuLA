# `cula.kda` — KDA (Kimi Delta Attention) operators

## API

| Symbol | Description | Arch | Direction |
|--------|-------------|------|-----------|
| `chunk_kda` | Chunked KDA prefill | SM100 (Blackwell) | Forward + backward |
| `kda_prefill_hopper` / `_opt` / `_auto` | KDA prefill (CUDA C++ fused) | SM90 (Hopper) | Forward only |
| `flashkda_prefill` | KDA prefill (CuTeDSL K1+K2, intracard-CP capable) | SM90 (Hopper) | Forward only |
| `kda_prefill` | KDA prefill, auto backend dispatch (flashkda -> fully_fused) | SM90 (Hopper) | Forward only |
| `kda_decode` | Single-token decode | SM100 | Forward |
| `fused_sigmoid_gating_delta_rule_update` | Decode state update | SM100 | Forward |

## Quick start

### Prefill (Hopper)

```python
import torch
from cula.kda import kda_prefill

B, T, H, D = 1, 1024, 4, 128
q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
g = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
beta = torch.randn(B, T, H, dtype=torch.bfloat16, device="cuda").sigmoid()
A_log = torch.randn(H, dtype=torch.float32, device="cuda")
dt_bias = torch.randn(H * D, dtype=torch.float32, device="cuda")

with torch.inference_mode():
    o, ht = kda_prefill(
        q, k, v, g, beta,
        A_log=A_log, dt_bias=dt_bias,
        scale=D**-0.5, lower_bound=-5.0,
        safe_gate=True, use_gate_in_kernel=True,
        output_final_state=True,
    )
```

The dispatcher tries FlashKDA first, then the fully-fused CUDA C++ path. Set `CULA_BACKEND_FLASHKDA=0` or `CULA_BACKEND_FULLY_FUSED=0` to disable one backend.

### Prefill (Blackwell, with backward)

```python
from cula.kda import chunk_kda

o, ht = chunk_kda(
    q, k, v, g, beta,
    A_log=A_log, dt_bias=dt_bias,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    safe_gate=True, lower_bound=-5.0,
    output_final_state=True,
)
```

### Variable-length (packed)

```python
cu_seqlens = torch.tensor([0, 256, 500, 1000], dtype=torch.int32, device="cuda")
q = torch.randn(1, 1000, H, D, dtype=torch.bfloat16, device="cuda")
# ... k, v, g, beta shaped [1, 1000, H, D] / [1, 1000, H]

o, ht = flashkda_prefill(
    q, k, v, g, beta,
    A_log=A_log, dt_bias=dt_bias,
    scale=D**-0.5, lower_bound=-5.0,
    safe_gate=True, use_gate_in_kernel=True,
    cu_seqlens=cu_seqlens,
    output_final_state=True,
)
```

### Intracard context-parallel

```python
# "auto": use CP only when beneficial for the given sequence lengths
o, ht = flashkda_prefill(
    q, k, v, g, beta,
    A_log=A_log, dt_bias=dt_bias,
    scale=D**-0.5, lower_bound=-5.0,
    safe_gate=True, use_gate_in_kernel=True,
    cu_seqlens=cu_seqlens,
    use_intracard_cp="auto",
)
```

## Requirements

- **D = 128** (head dimension, currently the only supported value)
- **bf16** for q/k/v/g/beta, **fp32** for A_log/dt_bias
- All tensors must be CUDA and contiguous
- SM90 prefill initial/final states use `[N, H, V, K]` (VK) layout
- `safe_gate=True` + `lower_bound` in `[-5, 0)` required for Hopper prefill
- `use_gate_in_kernel=True` required for Hopper prefill

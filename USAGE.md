# Usage Guide

This document provides usage notes and examples for cuLA kernels.

---

## KDA

cuLA provides two KDA kernel implementations targeting different GPU architectures:

| Kernel | GPU | Import |
|---|---|---|
| Modular Forward | Blackwell (SM100) | `from cula.kda import chunk_kda` |
| Two-Kernel Prefill | Hopper (SM90) | `from cula.kda import kda_prefill_hopper` |

Both are drop-in replacements for [FLA](https://github.com/fla-org/flash-linear-attention)'s `chunk_kda` — just change the import.

**General Notes**

- **`safe_gate=True`** is required to leverage TensorCore (M=16) acceleration.
- **`beta`** must be **`float32`** or **`bfloat16`**; **`initial_state`** must be **`float32`**.
- **`cu_seqlens`** (for variable-length sequences) must be **`int32`**.

---

### Modular Forward (SM100 — Blackwell)

The modular forward kernel replaces sub-kernels of KDA in FLA (chunk_intra, chunk_delta_h, fwd_o, etc.) for easy integration with [Kimi CP](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/cp/README.md).

#### Example

```python
import torch
from cula.kda import chunk_kda

B, T, H, K, V = 2, 2048, 32, 128, 128
device = 'cuda'

q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16, requires_grad=True)
g = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1
beta = torch.randn(B, T, H, device=device, dtype=torch.bfloat16).sigmoid()
A_log = torch.randn(H, device=device, dtype=torch.float32) * 0.01
dt_bias = torch.zeros(H * K, device=device, dtype=torch.float32)
init_state = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)

o, final_state = chunk_kda(
    q=q, k=k, v=v, g=g, beta=beta,
    A_log=A_log, dt_bias=dt_bias,
    initial_state=init_state,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    safe_gate=True,
    lower_bound=-5.0,
)

# Backward is supported
o.backward(torch.randn_like(o))

print(f'Output shape: {o.shape}')             # [2, 2048, 32, 128]
print(f'Final state shape: {final_state.shape}')  # [2, 32, 128, 128]
```

**Notes**

- The backward pass is currently supported via FLA's implementation; further optimizations are on the roadmap.
- Compatible with [Kimi CP](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/cp/README.md) via the `cp_context` parameter, same as in FLA.

---

### Two-Kernel Prefill (SM90 — Hopper)

The SM90 prefill is a **two-kernel pipeline** — K1 (Prepare) → K2 (Recurrence) — *not* a single fused kernel. Intra-chunk attention, inter-chunk state propagation, and output are computed across the two kernels. **Forward-only; backward is not yet implemented.**

#### Example

```python
import torch
from cula.kda import kda_prefill_hopper

B, T, H, K, V = 2, 2048, 32, 128, 128
device = 'cuda'

q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
g = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1
beta = torch.randn(B, T, H, device=device, dtype=torch.bfloat16).sigmoid()
A_log = torch.randn(H, device=device, dtype=torch.float32) * 0.01
dt_bias = torch.zeros(H * K, device=device, dtype=torch.float32)
init_state = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)

o, final_state = kda_prefill_hopper(
    q=q, k=k, v=v, g=g, beta=beta,
    A_log=A_log, dt_bias=dt_bias,
    initial_state=init_state,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    safe_gate=True,
    lower_bound=-5.0,
)

print(f'Output shape: {o.shape}')             # [2, 2048, 32, 128]
print(f'Final state shape: {final_state.shape}')  # [2, 32, 128, 128]
```

**Notes**

- Mainly **suitable for large-batch inference**. When both batch size and head count are small, throughput on long sequences is recovered by enabling **intra-card CP** (`use_intracard_cp`), which parallelizes the sequence-length dimension on a single GPU — see [Intra-Card Context Parallel](#intra-card-context-parallel).
- **Matrix inversion uses fp16 precision**, which is faster and occupies less shared memory but introduces minor numerical differences compared to tf32 inversion.
- **Intra-subchunk attention uses g-first as anchor**, which causes some numerical differences compared with the FLA Triton implementation (FLA uses g-half as anchor in the diagonal).

---

## Intra-Card Context Parallel

Long sequences can be split into sub-sequences, processed in parallel on one GPU, and merged via a prefix scan — unlocking sequence-dimension parallelism (≈3–7× on long single sequences where `batch × heads` under-utilises the SMs). **Default off; inference-only.** Two surfaces:

### SM90 — via `kda_prefill_hopper(use_intracard_cp=...)`

Pass `use_intracard_cp` (alias `use_cp`) to the Hopper prefill:

- **`"auto"`** — the auto-router enables CP only when it predicts a speedup, otherwise runs serial (no error, no regression).
- **`True`** — force CP; raises if the shape cannot be meaningfully split.
- **`False`** (default) — CP off.

Works with **any sequence length** (non-CHUNK-aligned is handled internally) and **dense or varlen** input. Tunable via `CULA_KDA_CP_MIN_SEG` (the auto-router skips CP below this many segments per sequence).

```python
o, final_state = kda_prefill_hopper(
    q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias,
    cu_seqlens=cu_seqlens,              # varlen packed (int32); omit for dense
    output_final_state=True, use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0,
    use_intracard_cp="auto",           # "auto" | True | False  (default False)
)
```

### SM100 — low-level `chunk_gated_delta_rule_fwd_h`

cuLA also exposes intra-card CP directly on the SM100 recurrence `chunk_gated_delta_rule_fwd_h`, gated by an environment variable.

**Requirements**

| Condition | Detail |
|---|---|
| Environment variable | `CULA_INTRACARD_CP=1` |
| Execution context | Inside `torch.inference_mode()` |
| Input mode | Varlen only (`cu_seqlens` must be provided) |
| Global gate | `g=None` (scalar gate `g` not supported; key-dim gate `gk` is supported) |

If the heuristic decides CP would not help (e.g. batch already saturates SMs, or sequences are too short), it silently falls back to the standard single-pass kernel.

**Example**

```python
import os
os.environ["CULA_INTRACARD_CP"] = "1"

import torch
from cula.ops.kda.sm100.delta_h import chunk_gated_delta_rule_fwd_h

B, T, H, K, V = 1, 65536, 8, 128, 128
device = 'cuda'

k  = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
w  = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
u  = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

with torch.inference_mode():
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
    )

print(f'h shape: {h.shape}')              # [1, NT, H, K, V]
print(f'final_state shape: {final_state.shape}')  # [1, H, K, V]
```

**Notes**

- CP is only beneficial when a small number of long sequences under-utilise the SM array. The built-in heuristic checks SM saturation, minimum sequence length (≥ 256 chunks), and effective batch size before enabling CP.
- Currently **inference-only**; the backward pass is not supported through the CP path.
- `cu_seqlens` must be **`int32`**.

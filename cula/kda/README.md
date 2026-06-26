# `cula.kda` — KDA (Kimi Delta Attention) operators

This package is the **public API + autograd + dispatch** layer for KDA. The actual GPU
kernels live under `cula/ops/kda/` (CuTe DSL / TVM-FFI) and `csrc/kda/sm100/` (CUDA C++,
exposed as `cula.cudac`). Dependency direction is one-way: `cula.kda` → `cula.ops.kda`
→ `cula.cudac`; backends never import `cula.kda`. `import cula` / `import cula.kda` are
lazy (PEP 562) and pull no CuTeDSL/CUDA at import time.

> Repo-wide tree: [`../../REPO_LAYOUT.md`](../../REPO_LAYOUT.md).

## Public API (`cula.kda.__init__`)

| Symbol | API wrapper | Backend (arch) | Notes |
|--------|-------------|--------|-------|
| `chunk_kda` | `chunk.py` | modular chunk (`ops/kda/sm100/`) | Full fwd **+ bwd** autograd. Default training & Blackwell prefill path. |
| `kda_prefill_hopper` | `hopper_prefill.py` (`= cula_kda_prefill`) | two-kernel prefill (`ops/kda/sm90/`, K1+K2) | Forward-only. Hopper. Two-kernel pipeline, *not* a fused kernel. |
| `kda_decode` | wraps `cula/ops/kda/decode/cute.py` | **decode** | Single-token decode. |
| `fused_sigmoid_gating_delta_rule_update` | wraps `cula/ops/kda/decode/cute.py` | **decode** | Decode state update. |

**Not exported:**
- `flash_kda_prefill` (`cula/ops/kda/experimental/sm100_fused/wrapper.py`) — `[exp]` **unwired / dead path**. No live caller: `get_kda_fused_fwd()` raises `NotImplementedError` on SM100/SM103. Backend: `…/experimental/sm100_fused/kda_fully_fused_wip.py` (~6k lines).

## The pipelines

### 1. Modular chunked — `chunk_kda` (SM100, train + Blackwell prefill)
```
chunk_kda                         chunk.py            (autograd: ChunkKDAFunction)
└ fwd  chunk_kda_fwd              chunk_fwd.py        (kernels imported lazily)
   ├ gate     FLA kda_gate_chunk_cumsum / chunk_local_cumsum
   ├ intra    chunk_kda_fwd_intra            chunk_intra.py
   │            └ C++  cula.cudac.chunk_kda_fwd_intra_cuda + recompute_w_u_cuda
   ├ [CP pre] FLA chunk_gated_delta_rule_fwd_h_pre_process     (only if cp_context)
   ├ recur    chunk_gated_delta_rule_fwd_h   ops/kda/sm100/delta_h.py  (CuTeDSL)
   │            └ may dispatch SM100 intracard-CP (see §5)
   └ out      chunk_gla_fwd_o                ops/kda/sm100/fwd_o.py    (CuTeDSL)
└ bwd  chunk_kda_bwd              chunk_bwd.py        ← 4 runtimes in one function:
        C++ recompute_w_u_cuda · FLA chunk_gated_delta_rule_bwd_dhu ·
        CuTeDSL bwd_wy_dqkg (ops/kda/sm100/bwd_wy_dqkg.py) ·
        Triton dAv/wy_dqkg (in chunk_bwd.py) + Triton bwd-intra (in chunk_intra.py) ·
        FLA gate bwd
```

### 2. Two-kernel (K1+K2) prefill (Hopper) — `kda_prefill_hopper` (SM90, fwd-only)
```
kda_prefill_hopper = cula_kda_prefill     hopper_prefill.py  (HopperChunkKDAFunction)
└ flash_kda_fwd                           ops/kda/sm90/fwd.py
   └ _dispatch_cute → launch_k1 (…/sm90/k1.py) + launch_k2 (…/sm90/k2.py)
      CuTe DSL, CHUNK=16, D=128. Handles varlen padding/repack. CUDA graph disabled.
```
> **Note:** this is a **two-kernel pipeline** (K1 prepare → 6 workspace tensors →
> K2 recurrence), *not* a single fused kernel.

### 3. Fused prefill (Blackwell) — `flash_kda_prefill` `[exp]`, not exported
```
flash_kda_prefill   ops/kda/experimental/sm100_fused/wrapper.py
                      → KDAChunkwise  ops/kda/experimental/sm100_fused/kda_fully_fused_wip.py (~6k lines)
```
**Unwired / dead** — no live caller (`get_kda_fused_fwd()` raises `NotImplementedError`).
The **production** Blackwell prefill is the modular `chunk_kda` (§1), not this.

### 4. Decode — `kda_decode` / `fused_sigmoid_gating_delta_rule_update`
```
ops/kda/decode/cute.py   — small / large / varlen kernel variants + a fast dense path.
   Independent of the sm90/sm100 prefill paths. (FLA reference: ops/kda/decode/reference_fla.py.)
```

### 5. Context Parallel (intracard, SM100) — `use_intracard_cp` / `use_cp`
Surfaced via an explicit `use_intracard_cp: "auto" | bool` (alias `use_cp`) on `chunk_kda`;
**default off**. Decision logic is centralized in `cula/ops/kda/policy.py`
(`sm100_intracard_cp_decision`): force (`True`) raises on unsupported/unsplittable, `"auto"`
runs only when supported + heuristically beneficial else falls back, `False` disables.
`cp_context` (FLA *cross-rank* CP) is orthogonal: when a `cp_context` is passed, forcing
`use_intracard_cp=True` **raises** (the two cannot be combined), while `"auto"`/`False`/default
force intracard CP **off** and let `cp_context` proceed.

| | SM100 CP |
|--|----------|
| Entry | `chunk_kda(use_cp=...)` → inside `chunk_gated_delta_rule_fwd_h` |
| Pipeline | `intracard_fwd_h`: `pre_scan` → `merge` → `fwd_h` on sub-seqs (recurses with `_no_cp=True`) |
| Default | off (`None`→env `CULA_INTRACARD_CP`) |
| Backend | `ops/kda/sm100/cp/{chunk_delta_h,pre_scan,merge}.py` |

> The Hopper two-kernel prefill (§2) is forward-only and **serial** here; its single-card
> CP variant is a separate change.

## Gotchas / known rough edges

- **`cp_context` (FLA cross-rank CP) ≠ `use_cp` (cuLA single-card intracard CP).** Both
  live on `chunk_kda`; they are orthogonal. `cp_context` comes from `fla.ops.cp` (FLA ≥ 0.5.0).
- **SM100 paths not CI-runtime-verified here:** this box is Hopper (no SM100 GPU, `cula.cudac`
  not built). SM100 (`chunk_kda`, decode, intracard-CP) is import/compile-verified; SM90 is
  kernel-test verified.

## Runtime cheat-sheet

| Runtime | Where |
|---------|-------|
| CUDA C++ (`cula.cudac`) | chunk intra fwd + recompute_w_u (`csrc/kda/sm100/`) |
| CuTe DSL / TVM-FFI | SM90 prefill (k1/k2), SM100 recurrence/output/bwd-fused, decode, SM100 CP backend |
| Triton | bwd intra (`chunk_intra.py`), bwd dAv/wy_dqkg (`chunk_bwd.py`) |
| FLA (`third_party/`) | gate cumsum/bwd, `chunk_gated_delta_rule_bwd_dhu`, cross-rank CP pre/post-process |

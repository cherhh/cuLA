# Repository Layout

> **Status:** updated after the KDA backend reorg. The KDA **Python** backends now live
> under `cula/ops/kda/` (by arch — currently `sm100/`); non-KDA operators moved to
> `cula/ops/lightning/` + `cula/ops/experimental/`; the old `cula/ops/*_sm100.py` /
> `cula/ops/cp/*` flat modules and compatibility shims have been removed. The SM90
> (Hopper) prefill is **still the CUDA C++ kernel** under `csrc/kda/sm90/` (exposed as
> `cula.cudac`); porting it to CuTeDSL is a subsequent change.

Legend: `[exp]` experimental / unwired · `[non-KDA]` other operator.

```
cuLA/
├── cula/                         # Python package (pip install -e .)
│   ├── __init__.py               # lazy (PEP 562): LinearAttentionChunkwiseDecay
│   ├── utils.py                  # arch asserts, get_pre_scan, cu_seqlens helpers, ...
│   ├── cudac.py                  # re-export shim over the compiled C++ extension(s)
│   │
│   ├── kda/                      # KDA PUBLIC API + autograd + dispatch (NO kernels)
│   │   ├── __init__.py           # lazy PUBLIC API: chunk_kda, kda_prefill_hopper,
│   │   │                         #                  kda_decode, fused_sigmoid_gating_delta_rule_update
│   │   ├── chunk.py              # chunk_kda + autograd — SM100 modular path (train + Blackwell prefill)
│   │   ├── chunk_fwd.py          # chunk_kda_fwd — fwd orchestration (lazy-imports kernels)
│   │   ├── chunk_intra.py        # fwd intra (C++ ext) + bwd intra (Triton)
│   │   ├── chunk_bwd.py          # chunk_kda_bwd — Triton + FLA + CuTeDSL + C++ mix
│   │   └── hopper_fused_fwd.py   # cula_kda_prefill (=kda_prefill_hopper) — SM90 prefill via the C++ kernel (cula.cudac)
│   │
│   ├── lightning/                # [non-KDA] Lightning Attention operator (package stub)
│   │   └── __init__.py
│   │
│   └── ops/                      # backend kernels (CuTe DSL / TVM-FFI) + shared helpers
│       ├── __init__.py           # exports kda_decode, fused_sigmoid_..., linear_attention_decode
│       ├── inv.py / ptx.py       # shared low-level helpers
│       ├── sm100/                # SM100 shared helper only
│       │   └── ptx.py            #   shared PTX helpers (used by KDA + lightning kernels)
│       │
│       ├── kda/                  # ★ KDA Python backends — by arch (sm100 today)
│       │   ├── policy.py         # SM100 CP dispatch policy: use_cp:"auto"|bool
│       │   ├── sm100/            # SM100 (Blackwell) modular-chunk kernels
│       │   │   ├── delta_h.py    #   recurrence (chunk_gated_delta_rule_fwd_h)
│       │   │   ├── fwd_o.py      #   output (chunk_gla_fwd_o)
│       │   │   ├── bwd_wy_dqkg.py#   backward wy/dqkg fused (used by chunk_bwd)
│       │   │   └── cp/           #   SM100 intracard-CP: chunk_delta_h, pre_scan, merge
│       │   ├── decode/           #   single-token decode
│       │   │   ├── cute.py       #     kda_decode / fused_sigmoid_gating_delta_rule_update (CuTe DSL)
│       │   │   └── reference_fla.py
│       │   └── experimental/sm100_fused/   # [exp] unwired fully-fused
│       │       ├── kda_fully_fused_wip.py   #   KDAChunkwise (~6k lines)
│       │       └── wrapper.py                #   flash_kda_prefill (dead path; raises on SM100 dispatch)
│       │
│       ├── lightning/            # [non-KDA] moved out of the KDA neighborhood
│       │   ├── prefill_sm100.py  #   Lightning Attn prefill (LinearAttentionChunkwiseDecay, lightning_attn_fwd[_varlen])
│       │   └── decode.py         #   linear_attention_decode
│       └── experimental/
│           └── linear_attn_prototype.py     # [non-KDA] unwired normalized-linear-attn prototype
│
├── csrc/                         # CUDA C++ / CUTLASS
│   ├── api/{kda_sm90.cu, kda_sm100.cu}        # PyBind11 (cula.cudac): SM90 prefill + SM100 chunk intra/recompute_w_u
│   ├── kda/sm90/                 # SM90 (Hopper) KDA C++ kernels (CUTLASS 3.x, TMA/wgmma)
│   ├── kda/sm100/                # Blackwell KDA C++ kernels (CUTLASS 3.x + UMMA)
│   └── kerutils/include/         # shared C++ headers (generic device helpers sm80/sm90/sm100, host)
│
├── benchmarks/  tests/  docs/    # flat, not yet grouped by operator
├── profile/                      # scratch profiling/experiment dirs (working area)
├── scripts/build_wheel.sh
├── third_party/flash-linear-attention/      # FLA submodule (baseline + reused gate/CP ops)
├── README.md  USAGE.md  REPO_LAYOUT.md  RECOMMENDED_CODING_STYLE.md
└── setup.py  pyproject.toml  LICENSE
```

## Key Directories

| Directory | Language | Description |
|-----------|----------|-------------|
| `cula/kda/` | Python | KDA **public API only** — autograd + dispatch, no kernels. Two prefill entries: modular chunk `chunk_kda` (SM100) and `kda_prefill_hopper` (SM90, driving the C++ kernel). Lazy `__init__` (no CuTeDSL pulled at import). |
| `cula/ops/kda/` | Python (CuTe DSL) | **KDA Python backends**, by arch: `sm100/` (+cp), `decode/`, `experimental/`, plus `policy.py` (CP dispatch). |
| `cula/ops/lightning/` · `cula/ops/experimental/` | Python (CuTe DSL) | `[non-KDA]` Lightning/linear attention kernels, moved out of the KDA neighborhood. |
| `cula/ops/{inv,ptx}.py`, `cula/ops/sm100/ptx.py` | Python | Shared low-level helpers (kept in place; not KDA-specific). |
| `csrc/kda/{sm90,sm100}/` · `csrc/api/` | CUDA C++ | Hopper SM90 prefill + Blackwell SM100 (chunk intra + recompute_w_u), exposed as `cula.cudac`. |

## State notes

- **SM90 prefill is the CUDA C++ kernel** (`csrc/kda/sm90/`, `csrc/api/kda_sm90.cu`),
  reached through `cula/kda/hopper_fused_fwd.py` → `cula.cudac`. The CuTeDSL port is a
  subsequent change.
- **Import-light:** `import cula` / `import cula.kda` no longer pull CuTeDSL/cutlass or the
  `cula.cudac` extension (lazy `__getattr__`); decode / SM100 Python backends are importable
  without building the C++ extension.
- **CP dispatch:** surfaced via `use_intracard_cp: "auto"|bool` (alias `use_cp`) on
  `chunk_kda` (SM100); **default off** — unspecified defers to the legacy `CULA_INTRACARD_CP`
  env. Decision logic centralized in `cula/ops/kda/policy.py`. The SM90 prefill is serial.

## Still open (not yet addressed)

- **SM90 CuTeDSL port** — the Hopper prefill is still C++; a CuTeDSL two-kernel
  implementation under `cula/ops/kda/sm90/` is a subsequent change.
- **`flash_kda_prefill` (Blackwell fully-fused) is an unwired dead path** —
  `get_kda_fused_fwd()` raises `NotImplementedError` on SM100/SM103 (`cula/utils.py`). Now
  quarantined under `cula/ops/kda/experimental/sm100_fused/`.
- **Three confusingly-named "linear attention" symbols — only two are wired:**
  - `linear_attention_decode` (`cula/ops/lightning/decode.py`) — decode; exported via `cula.ops` + `cula.lightning`.
  - `LinearAttentionChunkwiseDecay` + `lightning_attn_fwd[_varlen]` (`cula/ops/lightning/prefill_sm100.py`) — Lightning Attn prefill; exported via `cula` + `cula.lightning`.
  - `LinearAttentionChunkwise` (`cula/ops/experimental/linear_attn_prototype.py`) — *normalized* linear-attn **prototype**, unwired, benchmark-only.
- **Layering / grouping:** `cula/utils.py → cula.kda` import inversion; `tests/` ·
  `benchmarks/` · `docs/` not yet grouped under `kda/`; `profile/` scratch still at repo root.

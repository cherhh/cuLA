# FlashKDA Intracard-CP (卡内序列并行) — Plan Draft

## Task Contract

- **Task name**: `flashkda-intracard-cp`
- **Objective**: Implement intra-card sequence parallelism (pre_scan / merge / K2-rerun) for the
  cuLA CuTeDSL FlashKDA prefill, so that single-sequence long-context prefill (grid=(1,H)=64 CTAs,
  ~0.24 waves on L20Y) can use the whole GPU. Target use case: B=1 (or few seqs), T ≥ 8K.
- **Correctness requirements**:
  - CP output `out` and `final_state` must match the serial cute path
    (`flash_kda_prefill`) within bf16 chain tolerance: `max|Δ|/max|ref| < 2e-2`
    (same tolerance class as existing tests, which use 1e-2 abs vs torch ref).
  - Degenerate case `s_split=1` must reduce to exactly the serial pipeline shape
    (one segment per sequence, carry = user initial_state).
  - Supports: fixed (B,T) and varlen (cu_seqlens, all lens % 16 == 0); with/without
    initial_state/final_state; `state_transposed` both values (handled at driver edge).
  - Out of scope v0/v1: unaligned varlen repack (prefill.py's pad path) — assert aligned.
- **Allowed approaches**: CuTeDSL kernels only (no C++); torch host glue for merge;
  existing `launch_k1`/`launch_k2` reuse encouraged. No changes to serial-path behavior.
- **Validation command**: `python -m pytest tests/test_flashkda_cp.py tests/test_flashkda_prefill.py -v`
- **Evaluation command**: `python profile/flashkda_intracard_cp/bench_cp.py`
  (deployment settings, see table below; CP s_split sweep vs serial cute vs cpp)
- **Promotion criteria**: candidate promoted iff (a) all tests pass, (b) CP at auto
  s_split beats serial cute by ≥1.3× on the B=1 T≥16K rows of the deployment
  settings table, (c) no candidate-over-candidate regression on any bench config,
  (d) evidence in benchmark.csv + candidates.jsonl.

## Evaluation settings (2026-06-10, user-provided deployment table)

Real TP-sharded head counts (FlashQLA comparison table; reference timings on the
same seqlens: FlashQLA 0.18–0.88 ms, FLA 0.46–1.95 ms, FlashInfer 0.83–1.65 ms):

| setting | h_qk | h_v | seqlens |
|---|---|---|---|
| 397B/122B TP8 | 2 | 8 | 1×32768, 1×16384, varlen 24576+8192 |
| 397B/122B TP4 | 4 | 16 | 1×32768, 1×16384 |
| 27B TP2 | 8 | 24 | 1×32768 |
| 2B/0.8B TP1 | 16 | 16 | 1×32768 |
| Sym h32 | 32 | 32 | 1×32768 |

Mapping: cuLA FlashKDA kernels share one H across q/k/v/g (no GQA), so bench
uses **H := h_v** (state/recurrence count is per v-head — compute-equivalent
proxy; overestimates q/k traffic where h_qk < h_v). TP1 2B row coincides with
TP4 1×32768 at H=16. The earlier H=64 exploratory sweep (serial nearly
saturated, CP capped at 0.93×) is archived in `benchmark_h64.csv` — at the real
settings serial K2 occupies only (1,H)= 8–32 CTAs of 132 SMs, the regime CP
was designed for.

## Math (verified against k1.py / k2.py in review session 2026-06-10)

Per chunk c, K2 computes `S' = M_c·S + B_c` with
- `M_c = diag(gt) − kr^T·W`, `W = INV·diag(σβ)·kd`, `INV = (I+L)^{-1}` (k1 Neumann starts I−L,
  L row-scaled by σβ at k1.py:389), `kr = gt⊙ki` (k1.py:306-307).
- **V:=0 duality** (exact, incl. signs): running the K2 pipeline with V=0 and state:=M yields
  `M' = M_c·M` (Phase2 → −σβ⊙(kd@M), Phase3 → −W·M, Phase6 → diag(gt)M − kr^T·W·M).
  Left-multiply accumulation matches composition order M_[a,b) = M_{b−1}···M_a. M₀=I exact in bf16.
- **GMEM bhvk trick**: internal layout False stores G = S^T (G[v,k]=S[k,v]); therefore the merge
  is right-multiplication with NO transposes: `G_S' = G_S @ G_M + G_B` (torch.baddbmm per [H,D,D]).
  All internal CP state buffers fixed to layout False; user `state_transposed=True` converted
  by a [D,D] transpose at driver entry/exit.

Three stages:
1. **pre_scan**: per segment (contiguous tile range), compute `B_seg` (S-chain, S₀=0) and
   `M_seg` (M-chain, M₀=I) — no output O written.
2. **merge** (host, tiny): per original sequence, fold carry: `carry₀ = user_init or 0;
   carry_{i+1} = carry_i @ G_M_i + G_B_i`. ≤ (S_total−N) bmm's of [H,128,128].
3. **K2-rerun**: existing `launch_k2` with segment-level `cu_seqlens_tiles`,
   `initial_state = carries`, writes the real `out` (global tile indexing makes segment
   outputs land at correct positions); user final_state = last segment's kernel final.

Key structural facts that make this cheap:
- K1 is purely per-tile (no cross-chunk recurrence) → **run K1 once**, share ws_* across
  pre_scan and rerun (k2 reads ws at `head_idx*total_tiles + global_tile`).
- K2 already supports pseudo-sequences via `cu_seqlens_tiles` + per-seq state slots →
  segments-as-pseudo-sequences requires zero kernel changes for v0 and for the rerun.

## Baseline

- Serial cute: K1 459µs + K2 417µs at T=8192/H=64-class workloads; K2 grid=(N,H), 2 CTA/SM,
  L20Y sm_90 (132? SMs — query at runtime), 1.94 waves at grid=512. For B=1: grid=64 → ~0.24
  waves → expected CP headroom ≈ min(s_split, SM·2/H) on the K2 portion. K1 unchanged.
- cpp comparison available via `flash_kda.fwd` (serial only; no intracard CP upstream).

## Candidates (ranked)

| id | description | risk | expected value |
|----|-------------|------|----------------|
| `cp-v0-reuse` | he-chain = existing launch_k2 (S₀=0, final=B_seg, scratch out); m-chain = existing launch_k2 (V=zeros tensor, init=I, final=M_seg, scratch out); torch merge; rerun = existing launch_k2. Zero kernel changes. | low — pure glue | correctness baseline; perf ≈ 3 K2 passes / s_split (zeros-V + double out writes wasted) |
| `cp-v1-prescan-fused-seq` | new `k2_prescan.py`: 160 thr (drop STORE warp), drop sOut/sQd/sMqk (+TMA), add sM 32KB (init I); per tile run S-chain phases then M-chain phases **reusing the same fragment variables** (M-chain: diff = −u_pre, no sV read); epilogue writes B_seg+M_seg. SMEM ≈ 94.5KB → 2 CTA/SM; regs ≈ single-chain (~130-150) — safe under 204 ceiling (160thr×2CTA). | medium — new kernel, but all phases copied from validated k2.py | pre_scan = 1 launch ≈ 1.3-1.6× single-K2-tile-time; total ≈ 2.3-2.6 passes/s_split; removes zeros-V (2×T×H×D bytes) and scratch-out traffic |
| `cp-v2-prescan-interleaved` | interleave S/M chains per phase for latency hiding (+~64 regs for dual accumulators ≈ 190; fits 204 ceiling at 160thr). | high — register cliff, CuTeDSL fragments are function-scope (no liveness coalescing; 5 dead-end attempts documented in flashkda_kda_optim/docs/draft.md) | only if v1 NCU shows TC active well below serial K2's 27% with occupancy intact |
| `cp-v3-split-he-m` | separate he/m CTA flavors (3 CTA/SM SMEM-feasible at ≤61.8KB but needs ≤136 regs/thr @160thr — unproven) | medium | only relevant for SM-unsaturated regime; bench will show whether that regime matters |

## Main risks / unknowns

1. **bf16 M-chain precision at long segments**: M is a product chain rounded to bf16 per chunk
   (sState is bf16, k2.py:112). fp64 identity checks ≠ e2e guarantee. Mitigation: test compares
   CP vs serial cute at segment lengths up to 2K tiles; tolerance gate 2e-2.
2. **v0 zeros-V memory**: m-chain needs a zeros [1,T,H,D] bf16 (T=64K → 1GB). Cap v0 validation
   at T ≤ 32K; v1 removes it.
3. **Rerun goes through the no-CUDA-Graph path** (has_state_in disables graph in prefill.py) —
   accepted; CP targets long sequences where launch overhead is amortized. CP driver does its
   own direct launches (no graph at v0/v1).
4. **Register estimate for v1** is an estimate; CuTeDSL gives no mid-compile feedback. Gate:
   compile, check `ncu` regs/thread and local LD (spills must be 0).
5. **Merge layout bug class**: right-vs-left multiply convention. Gate: dedicated unit test of
   merge against torch serial chunk recurrence before any kernel work trusts it.

## First implementation steps

1. `cula/ops/flashkda/cp.py`: segment planner (tiles, per-seq near-equal split, MIN_SEG_TILES=4,
   auto s_split from SM count: ceil(2·SM/H/N) clamped), driver `flash_kda_prefill_cp(...)`,
   torch merge, v0 pre_scan via launch_k2 ×2.
2. `tests/test_flashkda_cp.py`: merge unit test + CP-vs-serial (fixed/varlen × state/no-state ×
   transposed × s_split ∈ {1,2,4,7}).
3. Validate → record `cp-v0-reuse` in candidates.jsonl + benchmark.csv.
4. `cula/ops/flashkda/k2_prescan.py` (v1) → wire into cp.py (env `CULA_FLASHKDA_CP_V0=1` keeps
   v0 path) → validate → bench → NCU spot-check (regs, spills, SMEM, occupancy) → promotion call.

## Evidence to record

- candidates.jsonl: id, parent, status (validated/promoted/rejected), evidence pointer.
- benchmark.csv: config, serial_cute_ms, cpp_ms, cp_ms per s_split, speedup.
- profile/: ncu summary for k2_prescan (regs/thread, local LD, SMEM, achieved occupancy).
- plan.md: iteration log + promotion decision.

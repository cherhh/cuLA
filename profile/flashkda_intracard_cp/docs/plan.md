# FlashKDA Intracard-CP — Executable Plan & Iteration Log

Contract: see `draft.md`. This file tracks concrete steps and evidence per candidate.

## Step plan

### Candidate `cp-v0-reuse` (gate: correctness)
- [x] S1. `cula/ops/flashkda/cp.py`
  - `_plan_segments(seq_lens_tiles, s_split, min_seg_tiles)` → `seg_cu_tiles (int32 [S+1])`,
    `seg_to_seq`, per-seq `(first_seg, n_seg)`. Near-equal contiguous tile ranges, each ≥1 tile.
  - `_auto_s_split(R_tiles, N, H)` = clamp(ceil(2·SM_count/H/N), 1, ·) then per-seq clamp by
    `R_s // MIN_SEG_TILES` (MIN_SEG_TILES=4, env `CULA_FLASHKDA_CP_MIN_SEG_TILES`).
  - `_merge_carries(M_seg, B_seg, seg_plan, user_init_bhvk)` → carries [S,H,D,D] fp32 (bhvk):
    `carry_{i+1} = baddbmm(B_i, carry_i, M_i)` per original sequence.
  - `flash_kda_prefill_cp(q,k,v,g,beta,scale,out,A_log,dt_bias,lower_bound,
     initial_state=None, final_state=None, cu_seqlens=None, state_transposed=False, s_split=None)`
    1. validate (reuse prefill asserts; require lens%16==0), build beta_flat [H,T_total],
       alloc ws via `_get_or_alloc_workspaces`, `launch_k1` ONCE.
    2. pre_scan v0: he = `launch_k2(v, …, out=scratch, cu=seg_cu_tiles, init=None, final=B_seg)`;
       m = `launch_k2(zeros_v, …, out=scratch, cu=seg_cu_tiles, init=eye, final=M_seg)`.
    3. merge (fold user initial_state, transposed → pre-transpose to bhvk).
    4. rerun: `launch_k2(v, …, out=user_out, cu=seg_cu_tiles, init=carries, final=seg_final?)`;
       user final_state[s] = seg_final[last_seg(s)] (transpose back + dtype if needed).
  - All internal state buffers layout False (bhvk). s_split==1 short-circuit → carries==inits,
    still 3-stage (keeps code path single) — acceptable for v0.
- [x] S2. `tests/test_flashkda_cp.py`
  - `test_merge_unit`: random per-chunk (M_c, B_c) in fp32, compose serially vs `_merge_carries`
    on segment products — exact (fp32) match. Guards the right-multiply convention.
  - `test_cp_matches_serial_{fixed,varlen}`: T∈{1024, 4096}, H=8 (small-H to keep runtime down,
    grid logic identical), s_split∈{1,2,4,7}; with/without initial/final state; state_transposed
    True case included. Gate `max|Δ|/max|ref| < 2e-2` on out and final_state vs serial cute.
- [x] S3. Run validation; record candidate row + bench quick row.

### Candidate `cp-v1-prescan-fused-seq` (gate: correctness + perf)
- [x] S4. `cula/ops/flashkda/k2_prescan.py` — copy k2.py and edit:
  - drop: tma qd/mqk/out, sQd/sMqk/sOut allocs, STORE warp branch, OUT pipeline mbarriers,
    has_initial/has_final flags, state_transposed param (False hardcoded).
  - add: `sM` (32KB, state_layout, init I), `b_state_g`/`m_state_g` fp32 outs (always written),
    M-chain phase block after S-chain phases per tile, reusing the SAME fragments
    (tCrU/tCrU_T/tCrU3/tCrU_T_post/tCrU_pre_bf16/tCrU3_bf16_tmp/tCrUpd_blk and operand frags);
    M-chain Phase2': `diff = −tCrU[ii]` (no sV); Phase6' targets sM views.
  - THREADS=160 (5 warps: 4 MMA + LOAD); TMA_BYTES = 3·CHUNK·D·2 + CHUNK²·2 + D·4 + CHUNK·2 = 13344.
  - smem_bytes: 2·32768 (sS+sM) + 2·3·4096 (sV/sKd/sKr) + 2·512 (sINV) + 2·512 (sGt) + 2·128
    (sBeta) + 4·8 mbar + slack ≈ 92.5KB. Single end-of-tile barrier(barrier_id=1, 128) as in k2.
  - launch grid=(S_total, H); compile cache key (S_total? → N_seg, H, total_tiles).
- [x] S5. Wire into cp.py (default; env `CULA_FLASHKDA_CP_V0=1` → v0 path). Re-run S2 tests.
- [x] S6. bench_cp.py — settings updated 2026-06-10 to the deployment table (draft.md):
  H:=h_v ∈ {8,16,24,32}, T ∈ {16K,32K}, varlen 24576+8192; CP s_split ∈ {4,8,16,32,auto}
  vs serial cute vs cpp → benchmark.csv (H=64 exploratory sweep → benchmark_h64.csv).
- [x] S7. NCU spot-check k2_prescan (1 iter, TP8 h8 T32K s=16): regs/thread, local LD == 0,
  SMEM, occupancy, TC active. Save summary to profile/.
- [x] S8. Promotion decision: cp-v1 PROMOTED (see 2026-06-10 S6 deployment entry);
  candidates.jsonl updated.

### Candidate `cp-v2-hostslim` (gate: strict improvement, driver-only)
Host-path slimming of the CP driver; kernels untouched, so correctness risk ≈ 0 and
any wall-clock change is pure dispatch/alloc overhead removal:
1. varlen `cu_seqlens` host lengths via prefill's weakref-cached `_get_or_build_seq_lens`
   (kills the per-call DtoH sync), 2. `seg_cu_tiles` / `last_idx` device tensors cached
   per plan (`_get_plan_tensor`), 3. merge fold in-place: `out=` baddbmm into a cached
   `carries` scratch (was: alloc + slice-copy per step ⇒ ~32 dispatches + 15 allocs at
   S=16, now 15 dispatches + 0 allocs), 4. `final_state` gather via
   `index_select(out=user_buffer)` fast path, 5. SM-count cache for the auto planner.
Gate: all 25 tests green; cp[auto] strictly faster than cp-v1 on every deployment row.

### Deferred (only with evidence)
- `cp-v2-graph` (demoted by hostslim, 2026-06-11): CUDA-graph the 3-stage pipeline.
  Hostslim recovered ~0.25 of the ~0.32 ms host gap; residual ≈ 0.07 ms (TP8 32K wall
  0.982 vs GPU-sum ~0.91 ms). Only worth it as an opt-in for ptr-stable callers,
  keyed on (shapes, s_split, ptr-set) like the serial graph path.
- `cp-v2-prescan-interleaved` if S7 NCU shows prescan TC-active ≪ serial K2's with
  occupancy intact.
- `cp-v3-split-he-m` for the H≥32 tail (Sym h32 only 1.36×): split state V-axis across
  CTA flavors to raise parallelism without scan work inflation.
- Proportional varlen planner beyond the 128-tile clamp (current clamp already balances
  the 24576+8192 row; revisit only if more skewed mixes appear).

## Iteration log

### 2026-06-10 — cp-v0-reuse: VALIDATED (correctness baseline)

- Tests: 15 passed — `tests/test_flashkda_cp.py` (10: merge unit, planner, fixed s_split∈{1,2,4,7},
  B=2+state, varlen+state, state_transposed, no-final) + `tests/test_flashkda_prefill.py` (5).
- Quick bench (L20Y 132 SM, B=1, H=64, auto s_split=5):

  | T | serial cute | cp[2] | cp[4] | cp[8] | cp[auto=5] |
  |---|---|---|---|---|---|
  | 8192 | 1.280 ms | 2.057 (0.62×) | 1.950 (0.66×) | 2.061 (0.62×) | 2.462 (0.52×) |
  | 16384 | 2.541 ms | 3.794 (0.67×) | 3.538 (0.72×) | 3.634 (0.70×) | 4.491 (0.57×) |

  Serial with `CULA_FLASHKDA_VARLEN_CUDAGRAPH=0` ≈ identical (1.302/2.561 ms) ⇒ GPU-bound;
  CuTeDSL host launch overhead is NOT the bottleneck at T≥8K. v0 deficit = 2 extra K2-class
  passes + zeros-V read + 2× scratch-out writes, exactly the v1 target. auto=5 worse than 4
  ⇒ wave quantization: rerun grid (5,64)=320 CTAs on 132 SM @2 CTA/SM = 264 slots → 2 waves
  same as s_split=4, but prescan passes pay more tail. Revisit `_auto_s_split` in v1 bench.
- **Bug found & fixed (pre-existing, prefill.py)**: `_LAST_BETA_FLAT_COPY_KEY` keyed the
  beta→beta_flat copy-skip on `(data_ptr, _version)`. After a test freed its tensors, the next
  test's `beta` reused the SAME address with `_version=0` → serial prefill skipped the copy and
  K1 ran with stale beta (rel err 0.39). Pure-serial sequences can hit it too (no CP needed).
  Fix: weakref-identity cache (`_make_tensor_refs`/`_same_tensor_refs`), shared helper
  `_copy_beta_flat` now the sole writer of beta_flat (cp.py uses it too); same treatment for
  `_LAST_VARLEN_REPACK_KEY` → `_LAST_VARLEN_REPACK_REFS` (same bug class on the repack copy).
  Note (not fixed, low priority): varlen graph_key bakes the derived cu_tiles pointer; a
  freed-and-collided `cu_seqlens` could replay against a stale cu_tiles buffer.
- Decision: keep v0 as fallback path (`CULA_FLASHKDA_CP_V0=1`); proceed to
  `cp-v1-prescan-fused-seq` (S4) — fuses prescan to 1 pass, removes zeros-V + scratch-out.

### 2026-06-10 — cp-v1-prescan-fused-seq: VALIDATED (correctness); perf gate NOT met at T≤16K

- Kernel: `k2_prescan.py` — single pass computes B_seg (S-chain) + M_seg (M-chain, V:=0
  duality) per tile, reusing the same fragments; 160 threads (no STORE warp), no out
  pipeline; TMA_BYTES=13344; smem 94 496 B → 2 CTA/SM. Always writes both states fp32 bhvk.
- Tests: 10/10 `tests/test_flashkda_cp.py` on v1 path; 10/10 with `CULA_FLASHKDA_CP_V0=1`
  (fallback intact); 5/5 `tests/test_flashkda_prefill.py`. Smoke rel: out 8.3e-4, fin 0.0.
- Quick bench (L20Y, B=1, H=64; ms, speedup vs serial cute):

  | T | serial | cp[2] | cp[4] | cp[8] | cp[auto=5, ceil] |
  |---|---|---|---|---|---|
  | 8192 | 1.278 | 1.864 (0.69×) | 1.663 (0.77×) | 1.805 (0.71×) | 2.165 (0.59×) |
  | 16384 | 2.538 | 3.363 (0.75×) | 2.991 (0.85×) | 3.102 (0.82×) | 3.928 (0.65×) |

- Profiler decomposition (T=8192, s_split=4, 5-iter avg, GPU time):
  - serial: k2 818.3 µs + k1 455.7 µs = 1275 µs.
  - cp4: prescan 509.3 + k1 459.3 + rerun k2 393.7 + merge baddbmm 36.6 + DtoD 26.1
    + misc ≈ 1436 µs GPU; wall 1663 µs ⇒ ~227 µs host gaps (un-graphed 3-stage launches).
- Analysis (why 0.77–0.85×):
  1. K1 is an unparallelized floor: ~456 µs ≈ 36% of serial total at T=8K (scales with T,
     so the *fraction* stays ~36% — Amdahl caps any rerun-only speedup at ~2.8×).
  2. Saturation penalty: serial k2 grid (1,64)=64 CTAs is latency-bound (0.24 waves,
     fast per-CTA); at s_split=4 the grid is 256 CTAs ≈ 2 CTA/SM saturated and per-CTA
     tile time degrades ~1.9× → rerun = 394 µs = 0.48× serial k2, not 0.25×. Prescan pays
     the same factor (509 µs vs ~268 ideal: 1.3× MMA work of k2/tile, 34 vs 26 blocks).
  3. Net: K2-class work = prescan(1.3×) + rerun(1×) = 2.3× serial FLOPs at ~1.9× worse
     per-CTA throughput / s_split parallelism — at s=4 this roughly cancels.
- Driver tweaks from this evidence: `_auto_s_split` ceil→floor (auto=5→320 CTAs ragged
  wave measurably worse than 4→256); degenerate plans (1 seg/seq) now delegate straight
  to `flash_kda_prefill`.
- Decision: status **validated** (correctness + fallback). Perf gate (≥1.3× at T≥16K)
  not met at T≤16K — S6 full bench must cover T∈{32K,64K} + varlen where serial stays
  latency-bound and the saturation argument predicts CP's best case. Promotion call at S8.

### 2026-06-10 — S6 @ H=64 (exploratory): NEGATIVE — then settings corrected to deployment table

- Full sweep at H=64 (archived `benchmark_h64.csv`): best CP ratio rises with T but
  asymptotes BELOW serial: cp[4] = 0.77× (8K) → 0.85× (16K) → 0.90× (32K) → 0.93× (64K).
  varlen N∈{2,4} strictly worse (0.53–0.63×; serial already N·64 CTAs); auto=1 delegation
  at N=4 passes through at 0.99×. cpp ≈ 0.97–0.98× of serial cute everywhere.
- Ceiling argument (closes the H=64 question): saturated-aggregate K2 throughput is only
  ~2.15× serial's (serial 64 CTAs already run ~47% of saturated per-SM rate; at 2 CTA/SM
  per-CTA tile time degrades 1.93×). Any prescan+rerun design pays ≥2.3× K2-class work ⇒
  hard cap ≈ 2.15/2.3 ≈ 0.93× — exactly what's measured. H=64 CP is structurally dead on
  132-SM parts; no kernel tuning escapes a work-multiple > throughput-headroom.
- Driver tweaks validated by re-run: auto ceil→floor and 1-seg/seq delegation —
  10/10 v1 + 10/10 v0-fallback after the change.
- Bench fix: cpp ext requires int64 cu_seqlens (cute path uses int32) — `cu64 = cu.long()`
  in bench; backfilled cpp varlen rows: N2 3.515 ms, N4 7.006 ms (≈ serial cute ±2%).
- **Settings correction (user, 2026-06-10)**: real deployment uses TP-sharded small head
  counts — H:=h_v ∈ {8,16,24,32}, T ∈ {16K,32K} + varlen 24576+8192 (table now in
  draft.md). There serial K2 = (1,H) = 8–32 CTAs on 132 SMs (6–24% occupancy) — the
  SM-unsaturated regime CP targets. Model forecast at H=8 T=32K: serial ≈ k1/8 + same
  chain-bound k2 ≈ 3.5 ms; CP[auto=33] ≈ 0.23 (k1) + 2.3·(2048·8)/85.8 inst/µs ≈ 0.44 (K2)
  + merge/gaps ≈ 0.8–0.9 ms ⇒ ~4×. Re-running S6 on the table settings.

### 2026-06-10 — S6 @ deployment settings: cp-v1 PROMOTED

- Full sweep (`benchmark.csv`; serial cute baseline; cpp = 0.98–1.04× of serial everywhere,
  i.e. both serial impls equivalent). Best-of-sweep and the auto row per config:

  | config | serial ms | best swept | cp[auto] (rule v2) |
  |---|---|---|---|
  | TP8 h8 1×32K | 3.463 | 1.219 @ s=16 (2.84×) | 1.231, 16 segs (**2.81×**) |
  | TP8 h8 1×16K | 1.743 | 0.989 @ s=8 (1.76×) | 0.995, 8 segs (**1.75×**) |
  | TP8 h8 24576+8192 | 2.656 | 1.513 @ s=8 (1.76×) | 1.303, 16 segs (**2.04×**) |
  | TP4 h16 1×32K | 3.694 | 1.816 @ s=16 (2.03×) | 1.880, 16 segs (**1.96×**) |
  | TP4 h16 1×16K | 1.855 | 1.130 @ s=8 (1.64×) | 1.145, 8 segs (**1.62×**) |
  | TP2 h24 1×32K | 3.941 | 2.427 @ auto=11 (1.62×) | 2.427, 11 segs (**1.62×**) |
  | Sym h32 1×32K | 4.141 | 3.039 @ s=8 (1.36×) | 3.051, 8 segs (**1.36×**) |

- **Auto rule v2** (from this sweep): `s = floor(2·SM / (H·N))` then per-sequence clamp
  `n_seg_s = min(s, R_s // AUTO_MIN_SEG_TILES)` with `AUTO_MIN_SEG_TILES = 128`
  (env `CULA_FLASHKDA_CP_AUTO_MIN_SEG_TILES`). One rule hits the measured optimum on
  every config: large H saturates to 2 CTA/SM; small H/T stops at ≥128-tile (2K-token)
  chains where prescan inflation + merge overhead would otherwise dominate. The same
  clamp auto-balances varlen: 24576+8192 → 12+4 segs of ~128 tiles each, 2.04× — better
  than ANY uniform s_split (max 1.76× with 192/64-tile imbalance). Explicit s_split
  keeps the old MIN_SEG_TILES=4 clamp (tests unchanged).
- Why h32 is the weakest row (1.36×): serial already runs 32 CTAs (24% of SMs) and the
  K1 floor + 2.3× prescan work eat the rest. Consistent with the H=64 ceiling analysis —
  speedup decays toward 0.93× as H→64. CP path auto-delegates when the plan degenerates,
  so enabling CP unconditionally is safe across H.
- Reference frame: FLA at TP8 1×32K = 0.913 ms, FlashQLA = 0.310 ms (user table; different
  stack). cp-v1 = 1.231 ms closes ~3.6× of serial cute's 3.8× gap to FLA-class; remaining
  delta is K1 floor (~230 µs) + un-graphed 3-stage host gaps (~0.2–0.35 ms) + prescan
  inflation — see deferred cp-v2-graph.
- Gate check (draft.md): (a) tests 10/10 v1 + 10/10 v0-fallback ✓ (re-run after planner
  change); (b) auto ≥1.3× on all B=1 T≥16K deployment rows: 1.36–2.81× ✓; (c) no
  candidate-over-candidate regression (v0 was never promoted; serial path untouched —
  serial numbers stable across runs ±0.3%) ✓; (d) benchmark.csv + candidates.jsonl ✓.
- **Decision: cp-v1-prescan-fused-seq PROMOTED** (default CP path; v0 kept behind
  `CULA_FLASHKDA_CP_V0=1`). S7 NCU spot-check recorded separately below.

### 2026-06-10 — S7 NCU spot-check, k2_prescan @ TP8 h8 T32K s=16 (`profile/ncu_prescan_tp8.txt`)

- Grid (16,8)=128 CTAs × 160 thr; **56 regs/thread; local LD/ST sectors = 0 (no spills)**;
  dynamic SMEM 94.50 KB → Block Limit Shared Mem = 2 (as designed); theoretical occupancy
  15.6% (2 CTA/SM × 5 warps), achieved 7.8% ≈ 1 CTA/SM at this grid (waves 0.48).
- Duration 422.9 µs (= 3.3 µs/tile at 128 tiles/CTA ⇒ prescan/tile ≈ 2.1× serial-k2/tile,
  above the 1.3× MMA-count model — extra cost is sM SMEM traffic, see next point).
- Throughputs: Compute (SM) 35.4%, **Memory (L1TEX/SMEM pipe) 62.0%**, DRAM 15.6%
  (522 GB/s). Prescan is SMEM-pipe-bound, not DRAM- or TC-bound — also explains the
  1.93× per-CTA degradation when 2 CTAs/SM co-reside (pipe saturates) and why
  `cp-v2-prescan-interleaved` (more MMA overlap) is NOT the right lever; reducing
  SMEM round-trips in the M-chain phases would be.
- GPU-sum reconstruction at this config: prescan 423 + rerun ≈205 + k1 ≈230 + merge ≈50
  ≈ 0.91 ms vs 1.231 ms wall ⇒ ~0.32 ms host gaps — quantifies the cp-v2-graph headroom
  (TP8 32K ~2.8× → ~3.8× if graphed).

### 2026-06-11 — cp-v2-hostslim: PROMOTED (driver-only, strict win on all rows)

- Change set (cp.py only; kernels + plans identical to cp-v1): cached varlen seq_lens
  (no per-call DtoH sync), cached `seg_cu_tiles`/`last_idx` plan tensors (no per-call
  `torch.tensor`+HtoD), in-place merge (`out=` baddbmm into cached carries scratch —
  was 15 allocs + 15 extra slice copies at S=16), `index_select(out=final_state)`
  gather, SM-count cache. Test API change: `_merge_carries` → in-place `_merge_carries_`.
- Tests: 10 passed tests/test_flashkda_cp.py + 10 passed under `CULA_FLASHKDA_CP_V0=1`
  + 5 passed tests/test_flashkda_prefill.py.
- Bench (`benchmark_hostslim.csv`, auto rows, serial baselines reproduced ±0.001 ms —
  clean A/B vs `benchmark.csv` cp-v1 rows). Uniform −0.22…−0.33 ms on every config:
  | config | cp-v1 auto | hostslim | speedup vs serial |
  |---|---|---|---|
  | TP8_h8_T32768 | 1.231 | **0.982** | 2.81× → **3.53×** |
  | TP8_h8_T16384 | 0.995 | **0.771** | 1.75× → **2.26×** |
  | TP8_h8_varlen_24576_8192 | 1.303 | **0.973** | 2.04× → **2.73×** (DtoH-sync kill ⇒ biggest Δ, −0.33) |
  | TP4_h16_T32768 | 1.880 | **1.580** | 1.96× → **2.33×** |
  | TP4_h16_T16384 | 1.145 | **0.913** | 1.62× → **2.03×** |
  | TP2_h24_T32768 | 2.427 | **2.168** | 1.62× → **1.82×** |
  | Sym_h32_T32768 | 3.051 | **2.811** | 1.36× → **1.47×** |
- Reference frame: TP8 32K now 0.982 ms vs FLA 0.913 ms (was 1.231) — near parity;
  varlen 0.973 vs FLA 0.767. Residual host gap ≈ 0.07 ms (GPU-sum ~0.91) ⇒ cp-v2-graph
  demoted to opt-in deferred; next GPU-side lever remains prescan SMEM-pipe work
  (S7) and the h32 tail (cp-v3-split-he-m).
- Decision: PROMOTE cp-v2-hostslim (strictly better on all 7 rows, driver-only,
  delegation + v0 fallback intact).

### 2026-06-11 — Same-card FLA cross-check (`bench_fla_cross.py` → `benchmark_fla_cross.csv`)

- Trigger: user challenged "flashkda 本身比 FLA 快 2×, CP 怎么还比 FLA 慢?" — my earlier
  framing compared L20Y wall times against the deployment table's FLA numbers
  (cross-hardware, methodologically wrong; FlashKDA's own evidence base is
  BENCHMARK_H20.md @ H∈{64,96}, T=8192, on H20). Resolution: measure FLA locally,
  using the exact chunk_kda invocation from FlashKDA/benchmarks/bench_fwd.py
  (use_gate_in_kernel / use_qk_l2norm / use_beta_sigmoid / transpose_state_layout).
- L20Y, prefill-from-zero + final_state, iters=20 (CP rows reproduce hostslim ±0.01):
  | config | fla_chunk_kda | flash_kda_ext | cuLA serial | cuLA CP[auto] | CP vs FLA |
  |---|---|---|---|---|---|
  | TP8_h8_T32768 | 1.849 | 3.478 (0.53×) | 3.460 | **0.971** | **1.91×** |
  | TP8_h8_T16384 | 0.957 | 1.750 (0.55×) | 1.742 | **0.765** | **1.25×** |
  | TP8_h8_varlen | 1.683 | 2.553 (0.66×) | 2.656 | **1.171** | **1.44×** |
  | TP4_h16_T32768 | 2.922 | 3.695 (0.79×) | 3.689 | **1.572** | **1.86×** |
  | TP4_h16_T16384 | 1.494 | 1.883 (0.79×) | 1.857 | **0.905** | **1.65×** |
  | TP2_h24_T32768 | 4.065 | 3.964 (1.03×) | 3.935 | **2.169** | **1.87×** |
  | Sym_h32_T32768 | 5.190 | 4.185 (1.24×) | 4.145 | **2.807** | **1.85×** |
- Reading: flash_kda's known ~2× edge over FLA (H=64/96) inverts at TP-sharded small H —
  at h8 the fused serial scan has 8 CTAs and FLA's tile-parallel chunk form wins 1.9×;
  flash_kda recovers with H (1.03× @h24, 1.24× @h32, →2× @64/96 per BENCHMARK_H20.md).
  CP repairs exactly this regime: ≥1.25× over FLA and 1.5–3.6× over flash_kda_ext on
  every deployment row. Weakest row = TP8 16K (1.25× vs FLA): 8 segs × 8 heads = 64 CTAs,
  machine half idle — revisit AUTO_MIN_SEG_TILES post-hostslim (sweep was pre-hostslim).
- Standing rule: cross-hardware table numbers (H20/GB200 docs, deployment table) are
  context only; promotion gates use same-card measurements.

### 2026-06-11 — Cross-impl A/B: cuLA flashkda-CP vs Hyaloid cuLA intcd-cp (SM90)

- Trigger: user request — compare our CuTeDSL CP against the upstream
  `Hyaloid/cuLA` `intcd-cp` PR (CUTLASS C++ SM90 KDA prefill + CP wrapper)
  on the same hardware.
- Setup: H800 (cap 9.0, 132 SMs). Hyaloid bench is `benchmarks/bench_intracard_cp_sm90.py`
  unmodified (its 18 configs × H∈{4,8}, warmup=25, iters=100, `safe_gate=True`,
  `use_qk_l2norm_in_kernel=True`, BT=64). cuLA bench (`bench_vs_hyaloid.py`)
  mirrors shapes/warmup/iters; beta pre-sigmoided + cast to bf16 to keep K1 work
  parity (cuLA K1 has no l2norm-in-kernel, so q/k traffic identical; per-tile
  MMA counts equal). Two processes (cwd-isolated `cula` imports), merge by
  (H, config) tag.
- Result (`benchmark_vs_hyaloid_merged.csv`, 36 rows): CP A/B geo-mean cuLA/hyaloid
  = **2.00×**, best 6.34×, worst 0.80×; cuLA wins on **32/36** configs. The 4
  losses are all tiny single-seq T∈{4K,8K} where CP overhead dominates (cuLA
  0.80-0.84×). Serial cuLA CuTeDSL is uniformly **1.7-1.8× faster than hyaloid
  CUTLASS C++ serial** — even before CP, the baseline kernel already wins.
- Apples-to-apples (hyaloid pred=Y, 25 rows, both CPs actually fire):
  - large single-seq T∈{32K,64K,128K} H∈{4,8}: cuLA **1.6-2.2×** faster
  - balanced varlen (2x16K, 16K+16K, 24K+8K, 28K+4K, 32K+256+256 H=4): **1.7-1.9×**
  - small single-seq T∈{4K,8K}: hyaloid **1.2× faster** (CUTLASS mainloop has lower
    per-chunk dispatch overhead than CuTeDSL → 2 CTAs of 64-tile chunks beat 8
    segs of 128-tile clamp at very short T)
- Apples-not-equal (hyaloid pred=N, 11 rows, hyaloid bypasses CP): cuLA fires CP
  on everything where ≥2 segments make sense ⇒ wins **3.1-6.3×**, e.g.
  128K+5x1K @H=4 hyaloid 23.68ms → cuLA 3.74ms. This is partly a planner-policy
  difference, not a kernel difference: hyaloid's CP heuristic refuses
  tail-padded shapes (`128K + small_seqs`) that our 128-tile-floor planner happily
  splits along the long seq.
- Standing rule (kept): cross-hardware tables = context only; promotion gates on
  same-card measurements. This A/B counts as a same-card cross-impl read-out.
- Caveats (recorded so the user doesn't have to ask):
  - Two CPs target slightly different semantics: hyaloid does l2norm in-kernel,
    safe_gate, sigmoid(beta) in-kernel. cuLA K1 does scalar gate + dt_bias only;
    we feed pre-sigmoided beta to match work shape. Number of MMAs is equal but
    per-chunk traffic differs marginally (≤5% of total wall on T≥32K).
  - BT (chunk size) differs: hyaloid 64, cuLA 16. This favors hyaloid at very
    small T (fewer chunk boundaries) but inflates its scan-chain length at large
    T (more state-passing iterations per CTA). The cross-over is at ~T=16K.
- Action: nothing structural needed — cuLA CP is the faster path everywhere
  except T≤8K single-seq. AUTO_MIN_SEG_TILES=128 leaves T=4K/8K underutilized
  (1 segment); optionally relax floor on H≤4 small-T to compete at the bottom
  end. Defer until needed.

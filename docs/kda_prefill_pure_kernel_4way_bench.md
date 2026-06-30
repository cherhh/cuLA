# KDA prefill — PURE-KERNEL 4-way benchmark (Hopper sm_90)

FlashKDA (C++ CUTLASS) vs FLA/Triton vs cuLA-noncp (K1+K2) vs cuLA-cp (intracard CP, auto).
**D=128, bf16, H ∈ {8,16,32,64}.** Settings replicate cuLA's `bench_kda_sm90_prefill`
(Fixed B×T + `build_varlen_configs`).

> **Device:** torch capability **(9,0) = Hopper sm_90** (nvidia-smi labels it "L20Y / cc 8.9"
> — a relabel; the SM90 TMA/wgmma kernels run, `assert_hopper` passes; identified as H200-class).
> CUDA 12.9, PyTorch 2.9.1.

## ⚠️ Methodology: PURE KERNEL via CUDA graph (not per-iter events)

Both cuLA's and FlashKDA's own bench scripts use **per-iter CUDA events**
(`start.record(); fn(); end.record()` per iteration). That does **NOT** give pure
kernel time: a multi-launch op's host-dispatch gaps fall *inside* `start…end`, so each
impl carries its own dispatch floor — **FLA/Triton ≈ 0.9ms** (many triton launches),
**cuLA ≈ 0.45ms** (2–4 CuTeDSL launches + Python), **FlashKDA ≈ 0.08ms** (one C++ launch).
On small shapes the per-iter comparison is dominated by *dispatch*, not kernel speed
(e.g. per-iter shows "FlashKDA 11× faster than FLA" at T=512 — almost entirely FlashKDA's
single-C++-launch advantage).

To measure **only kernel time** we use **CUDA-graph replay**, which removes ALL host
dispatch. Under graphs FLA scales cleanly with length (no floor), and the comparison is a
true kernel-vs-kernel one. (Caveat: in *eager* serving without CUDA graphs, FlashKDA's
low dispatch is a real end-to-end advantage that pure-kernel numbers hide.)

Timing: warmup 8, then min of 4×50 graph replays. speedup = `fla_ms / x_ms` (>1 = x faster).

---

## Results (pure kernel, ms)

### H=8
| config | fla | noncp | cp | FlashKDA | nc/fla | cp/fla | fk/fla |
|---|--:|--:|--:|--:|--:|--:|--:|
| B1 T512 | 0.0588 | 0.0712 | 0.0711 | 0.0650 | 0.83x | 0.83x | 0.90x |
| B1 T1024 | 0.0813 | 0.1254 | 0.1255 | 0.1158 | 0.65x | 0.65x | 0.70x |
| B1 T4096 | 0.2693 | 0.4564 | 0.4562 | 0.4342 | 0.59x | 0.59x | 0.62x |
| B1 T8192 | 0.5146 | 0.8872 | 0.8886 | 0.8474 | 0.58x | 0.58x | 0.61x |
| B1 T16384 | 0.9901 | 1.7379 | _gfail_ | 1.6494 | 0.57x | – | 0.60x |
| B2 T512 | 0.0716 | 0.0752 | 0.0752 | 0.0695 | 0.95x | 0.95x | 1.03x |
| B2 T1024 | 0.1152 | 0.1351 | 0.1348 | 0.1259 | 0.85x | 0.85x | 0.91x |
| B2 T4096 | 0.4208 | 0.4821 | 0.4808 | 0.4612 | 0.87x | 0.88x | 0.91x |
| B2 T8192 | 0.8074 | 0.9390 | 0.9384 | 0.9017 | 0.86x | 0.86x | 0.90x |
| B2 T16384 | 1.5725 | 1.8574 | _gfail_ | 1.7857 | 0.85x | – | 0.88x |
| uniform 10seq T=4096 | 0.2180 | 0.1002 | 0.1001 | 0.0967 | 2.18x | 2.18x | 2.25x |
| random 10seq T=4096 | 0.2193 | 0.1854 | 0.1851 | 0.1646 | 1.18x | 1.18x | 1.33x |
| skewed 10seq T=4096 | 0.2456 | 0.2781 | 0.2776 | 0.2436 | 0.88x | 0.88x | 1.01x |
| uniform 20seq T=4096 | 0.2201 | 0.0930 | 0.0926 | 0.0952 | 2.37x | 2.38x | 2.31x |
| random 20seq T=4096 | 0.2235 | 0.1532 | 0.1513 | 0.1369 | 1.46x | 1.48x | 1.63x |
| skewed 20seq T=4096 | 0.2488 | 0.2852 | 0.2856 | 0.2593 | 0.87x | 0.87x | 0.96x |
| uniform 10seq T=8192 | 0.3849 | 0.1744 | 0.1740 | 0.1604 | 2.21x | 2.21x | 2.40x |
| random 10seq T=8192 | 0.4010 | 0.3436 | 0.3440 | 0.3048 | 1.17x | 1.17x | 1.32x |
| skewed 10seq T=8192 | 0.4389 | 0.5280 | 0.5286 | 0.4619 | 0.83x | 0.83x | 0.95x |
| uniform 20seq T=8192 | 0.3915 | 0.1558 | 0.1558 | 0.1588 | 2.51x | 2.51x | 2.47x |
| random 20seq T=8192 | 0.3994 | 0.2742 | 0.2747 | 0.2506 | 1.46x | 1.45x | 1.59x |
| skewed 20seq T=8192 | 0.4464 | 0.5430 | 0.5423 | 0.4819 | 0.82x | 0.82x | 0.93x |
| uniform 10seq T=16384 | 0.7324 | 0.3189 | 0.3193 | 0.2966 | 2.30x | 2.29x | 2.47x |
| random 10seq T=16384 | 0.7565 | 0.6611 | 0.6592 | 0.5844 | 1.14x | 1.15x | 1.29x |
| skewed 10seq T=16384 | 0.8295 | 1.0265 | 1.0271 | 0.8972 | 0.81x | 0.81x | 0.92x |
| uniform 20seq T=16384 | 0.7169 | 0.2777 | 0.2778 | 0.2815 | 2.58x | 2.58x | 2.55x |
| random 20seq T=16384 | 0.7421 | 0.5102 | 0.5130 | 0.4732 | 1.45x | 1.45x | 1.57x |
| skewed 20seq T=16384 | 0.8314 | 1.0506 | 1.0506 | 0.9312 | 0.79x | 0.79x | 0.89x |

### H=16
| config | fla | noncp | cp | FlashKDA | nc/fla | cp/fla | fk/fla |
|---|--:|--:|--:|--:|--:|--:|--:|
| B1 T512 | 0.0712 | 0.0755 | 0.0755 | 0.0703 | 0.94x | 0.94x | 1.01x |
| B1 T1024 | 0.1061 | 0.1354 | 0.1355 | 0.1279 | 0.78x | 0.78x | 0.83x |
| B1 T4096 | 0.3845 | 0.4811 | 0.4816 | 0.4630 | 0.80x | 0.80x | 0.83x |
| B1 T8192 | 0.7251 | 0.9385 | 0.9385 | 0.8995 | 0.77x | 0.77x | 0.81x |
| B1 T16384 | 1.4044 | 1.8541 | _gfail_ | 1.7878 | 0.76x | – | 0.79x |
| B2 T512 | 0.0960 | 0.0871 | 0.0868 | 0.0812 | 1.10x | 1.11x | 1.18x |
| B2 T1024 | 0.1804 | 0.1547 | 0.1548 | 0.1475 | 1.17x | 1.17x | 1.22x |
| B2 T4096 | 0.6403 | 0.5448 | 0.5448 | 0.5217 | 1.18x | 1.18x | 1.23x |
| B2 T8192 | 1.2393 | 1.0648 | 1.0629 | 1.0264 | 1.16x | 1.17x | 1.21x |
| B2 T16384 | 2.4447 | 2.1029 | 2.1027 | 2.0308 | 1.16x | 1.16x | 1.20x |
| uniform 10seq T=4096 | 0.3546 | 0.1550 | 0.1543 | 0.1577 | 2.29x | 2.30x | 2.25x |
| random 10seq T=4096 | 0.3550 | 0.2383 | 0.2380 | 0.2164 | 1.49x | 1.49x | 1.64x |
| skewed 10seq T=4096 | 0.3691 | 0.3544 | 0.3530 | 0.3155 | 1.04x | 1.05x | 1.17x |
| uniform 20seq T=4096 | 0.3703 | 0.1655 | 0.1653 | 0.1679 | 2.24x | 2.24x | 2.21x |
| random 20seq T=4096 | 0.3733 | 0.2067 | 0.2061 | 0.1952 | 1.81x | 1.81x | 1.91x |
| skewed 20seq T=4096 | 0.3777 | 0.3486 | 0.3521 | 0.3281 | 1.08x | 1.07x | 1.15x |
| uniform 10seq T=8192 | 0.6336 | 0.2773 | 0.2775 | 0.2799 | 2.28x | 2.28x | 2.26x |
| random 10seq T=8192 | 0.6372 | 0.4419 | 0.4420 | 0.4030 | 1.44x | 1.44x | 1.58x |
| skewed 10seq T=8192 | 0.6726 | 0.6757 | 0.6560 | 0.6047 | 1.00x | 1.03x | 1.11x |
| uniform 20seq T=8192 | 0.6507 | 0.2895 | 0.2893 | 0.2924 | 2.25x | 2.25x | 2.23x |
| random 20seq T=8192 | 0.6581 | 0.3712 | 0.3722 | 0.3532 | 1.77x | 1.77x | 1.86x |
| skewed 20seq T=8192 | 0.6825 | 0.6744 | 0.6699 | 0.6153 | 1.01x | 1.02x | 1.11x |
| uniform 10seq T=16384 | 1.2129 | 0.5217 | 0.5204 | 0.5375 | 2.32x | 2.33x | 2.26x |
| random 10seq T=16384 | 1.2085 | 0.8531 | 0.8518 | 0.7768 | 1.42x | 1.42x | 1.56x |
| skewed 10seq T=16384 | 1.2695 | 1.2699 | 1.2763 | 1.1492 | 1.00x | 0.99x | 1.10x |
| uniform 20seq T=16384 | 1.2077 | 0.5410 | 0.5383 | 0.5403 | 2.23x | 2.24x | 2.24x |
| random 20seq T=16384 | 1.2250 | 0.7052 | 0.7065 | 0.6742 | 1.74x | 1.73x | 1.82x |
| skewed 20seq T=16384 | 1.2760 | 1.2733 | 1.2489 | 1.1623 | 1.00x | 1.02x | 1.10x |

### H=32
| config | fla | noncp | cp | FlashKDA | nc/fla | cp/fla | fk/fla |
|---|--:|--:|--:|--:|--:|--:|--:|
| B1 T512 | 0.0977 | 0.0881 | 0.0880 | 0.0819 | 1.11x | 1.11x | 1.19x |
| B1 T1024 | 0.1813 | 0.1550 | 0.1552 | 0.1487 | 1.17x | 1.17x | 1.22x |
| B1 T4096 | 0.6454 | 0.5444 | 0.5444 | 0.5223 | 1.19x | 1.19x | 1.24x |
| B1 T8192 | 1.2369 | 1.0635 | 1.0629 | 1.0247 | 1.16x | 1.16x | 1.21x |
| B1 T16384 | 2.4407 | 2.0999 | 2.1005 | 2.0296 | 1.16x | 1.16x | 1.20x |
| B2 T512 | 0.1749 | 0.1042 | 0.1041 | 0.1006 | 1.68x | 1.68x | 1.74x |
| B2 T1024 | 0.3301 | 0.1837 | 0.1833 | 0.1784 | 1.80x | 1.80x | 1.85x |
| B2 T4096 | 1.1846 | 0.6613 | 0.6611 | 0.6439 | 1.79x | 1.79x | 1.84x |
| B2 T8192 | 2.3351 | 1.2922 | 1.2909 | 1.2633 | 1.81x | 1.81x | 1.85x |
| B2 T16384 | 4.6475 | 2.5508 | 2.5499 | 2.5157 | 1.82x | 1.82x | 1.85x |
| uniform 10seq T=4096 | 0.6522 | 0.2894 | 0.2899 | 0.2913 | 2.25x | 2.25x | 2.24x |
| random 10seq T=4096 | 0.6556 | 0.3322 | 0.3323 | 0.3207 | 1.97x | 1.97x | 2.04x |
| skewed 10seq T=4096 | 0.6528 | 0.4555 | 0.4658 | 0.4407 | 1.43x | 1.40x | 1.48x |
| uniform 20seq T=4096 | 0.7012 | 0.2678 | 0.2677 | 0.2926 | 2.62x | 2.62x | 2.40x |
| random 20seq T=4096 | 0.6869 | 0.3273 | 0.3291 | 0.3161 | 2.10x | 2.09x | 2.17x |
| skewed 20seq T=4096 | 0.6729 | 0.4714 | 0.4726 | 0.4635 | 1.43x | 1.42x | 1.45x |
| uniform 10seq T=8192 | 1.2071 | 0.5415 | 0.5411 | 0.5397 | 2.23x | 2.23x | 2.24x |
| random 10seq T=8192 | 1.2136 | 0.6278 | 0.6245 | 0.6086 | 1.93x | 1.94x | 1.99x |
| skewed 10seq T=8192 | 1.2230 | 0.8937 | 0.8883 | 0.8265 | 1.37x | 1.38x | 1.48x |
| uniform 20seq T=8192 | 1.2516 | 0.4918 | 0.4921 | 0.5125 | 2.55x | 2.54x | 2.44x |
| random 20seq T=8192 | 1.2488 | 0.5994 | 0.6045 | 0.5832 | 2.08x | 2.07x | 2.14x |
| skewed 20seq T=8192 | 1.2544 | 0.8943 | 0.9096 | 0.8675 | 1.40x | 1.38x | 1.45x |
| uniform 10seq T=16384 | 2.3532 | 1.0311 | 1.0302 | 1.0436 | 2.28x | 2.28x | 2.25x |
| random 10seq T=16384 | 2.3520 | 1.2140 | 1.2150 | 1.1707 | 1.94x | 1.94x | 2.01x |
| skewed 10seq T=16384 | 2.3662 | 1.7019 | 1.7130 | 1.5922 | 1.39x | 1.38x | 1.49x |
| uniform 20seq T=16384 | 2.3718 | 0.9386 | 0.9387 | 0.9660 | 2.53x | 2.53x | 2.46x |
| random 20seq T=16384 | 2.3736 | 1.1491 | 1.1438 | 1.1069 | 2.07x | 2.08x | 2.14x |
| skewed 20seq T=16384 | 2.3735 | 1.7261 | 1.7462 | 1.6624 | 1.38x | 1.36x | 1.43x |

### H=64
| config | fla | noncp | cp | FlashKDA | nc/fla | cp/fla | fk/fla |
|---|--:|--:|--:|--:|--:|--:|--:|
| B1 T512 | 0.1742 | 0.1042 | 0.1045 | 0.1013 | 1.67x | 1.67x | 1.72x |
| B1 T1024 | 0.3293 | 0.1845 | 0.1839 | 0.1783 | 1.79x | 1.79x | 1.85x |
| B1 T4096 | 1.1813 | 0.6587 | 0.6583 | 0.6457 | 1.79x | 1.79x | 1.83x |
| B1 T8192 | 2.3163 | 1.2908 | 1.2907 | 1.2633 | 1.79x | 1.79x | 1.83x |
| B1 T16384 | 4.6165 | 2.5553 | 2.5548 | 2.5198 | 1.81x | 1.81x | 1.83x |
| B2 T512 | 0.3341 | 0.1380 | 0.1373 | 0.1375 | 2.42x | 2.43x | 2.43x |
| B2 T1024 | 0.6193 | 0.2482 | 0.2482 | 0.2418 | 2.50x | 2.50x | 2.56x |
| B2 T4096 | 2.3203 | 0.9181 | 0.9153 | 0.8946 | 2.53x | 2.54x | 2.59x |
| B2 T8192 | 4.6194 | 1.8043 | 1.8053 | 1.7578 | 2.56x | 2.56x | 2.63x |
| B2 T16384 | 9.2254 | 3.6025 | 3.6052 | 3.5072 | 2.56x | 2.56x | 2.63x |
| uniform 10seq T=4096 | 1.2447 | 0.4933 | 0.4938 | 0.5133 | 2.52x | 2.52x | 2.42x |
| random 10seq T=4096 | 1.2395 | 0.5606 | 0.5635 | 0.5535 | 2.21x | 2.20x | 2.24x |
| skewed 10seq T=4096 | 1.2424 | 0.6700 | 0.6702 | 0.6508 | 1.85x | 1.85x | 1.91x |
| uniform 20seq T=4096 | 1.3277 | 0.4839 | 0.4838 | 0.5334 | 2.74x | 2.74x | 2.49x |
| random 20seq T=4096 | 1.3045 | 0.5636 | 0.5613 | 0.5593 | 2.31x | 2.32x | 2.33x |
| skewed 20seq T=4096 | 1.2814 | 0.6807 | 0.6783 | 0.6960 | 1.88x | 1.89x | 1.84x |
| uniform 10seq T=8192 | 2.3561 | 0.9364 | 0.9362 | 0.9664 | 2.52x | 2.52x | 2.44x |
| random 10seq T=8192 | 2.3696 | 1.0690 | 1.0678 | 1.0470 | 2.22x | 2.22x | 2.26x |
| skewed 10seq T=8192 | 2.3927 | 1.2736 | 1.2793 | 1.2335 | 1.88x | 1.87x | 1.94x |
| uniform 20seq T=8192 | 2.4395 | 0.9029 | 0.9027 | 0.9563 | 2.70x | 2.70x | 2.55x |
| random 20seq T=8192 | 2.4320 | 1.0367 | 1.0366 | 1.0372 | 2.35x | 2.35x | 2.34x |
| skewed 20seq T=8192 | 2.4285 | 1.2837 | 1.2970 | 1.3001 | 1.89x | 1.87x | 1.87x |
| uniform 10seq T=16384 | 4.6492 | 1.8130 | 1.8123 | 1.8765 | 2.56x | 2.57x | 2.48x |
| random 10seq T=16384 | 4.6492 | 2.0834 | 2.0853 | 2.0476 | 2.23x | 2.23x | 2.27x |
| skewed 10seq T=16384 | 4.6607 | 2.4659 | 2.4913 | 2.4186 | 1.89x | 1.87x | 1.93x |
| uniform 20seq T=16384 | 4.7016 | 1.7423 | 1.7437 | 1.8233 | 2.70x | 2.70x | 2.58x |
| random 20seq T=16384 | 4.7064 | 1.9984 | 1.9975 | 2.0013 | 2.36x | 2.36x | 2.35x |
| skewed 20seq T=16384 | 4.6870 | 2.4866 | 2.4975 | 2.5010 | 1.88x | 1.88x | 1.87x |

---

## Conclusions (pure kernel)

1. **FlashKDA ≈ cuLA-noncp.** Across all H and configs the two are within **~5–10%**
   (FlashKDA usually a hair faster on Fixed; cuLA a hair faster on some uniform-varlen).
   cuLA's CuteDSL K1+K2 kernel is **competitive with FlashKDA's C++ CUTLASS** in pure-kernel terms.

2. **Strong H-dependence (the cuLA occupancy story).** cuLA's K2 recurrence kernel has
   `grid = (N_seqs, H)`, so a single sequence yields only `H` CTAs:
   - **H=8** single sequence → 8 CTAs (~6% of SMs) → **FLA wins ~1.7×** (its chunk-parallel
     output fills the GPU; cuLA/FlashKDA serialize the recurrence+output per (seq,head)).
   - **H=16** → FLA still wins single-seq ~1.25×.
   - **H=32** → cuLA/FlashKDA flip ahead (~1.16×).
   - **H=64** → cuLA/FlashKDA win **everything**, single-seq ~1.79×, uniform-varlen ~2.7×.

3. **Per-config pattern (any H):** **uniform** multi-seq → cuLA/FK ~2.2–2.7× over FLA (best
   occupancy). **random** → ~1.4–2.3×. **skewed** (one long seq dominates → low occupancy)
   tracks the single-sequence behavior (FLA wins at low H, cuLA wins at high H).

4. **cuLA-cp ≈ cuLA-noncp** for all these configs — intracard CP does not meaningfully engage
   at D=128 with these (H, shape) combinations. cuLA-cp also **fails CUDA-graph capture** on
   single-long-sequence configs (B1 T16384) because its segment plan is computed dynamically
   on the host — a limitation for CUDA-graph serving.

5. **Eager vs pure-kernel caveat:** the per-iter-event numbers in cuLA's/FlashKDA's own bench
   tables include each impl's host-dispatch floor, where **FlashKDA's single-C++-launch design
   wins big on small shapes** (e.g. 11× at T=512). That advantage is real for *eager* serving
   but disappears here under CUDA graphs — choose the measure that matches your serving mode.

> Repro: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /tmp/bench_4impl_graph.py <H>`
> (per-H process; CUDA-graph replay, min of 4×50).

"""Profile cuLA intracard CP pipeline on GB200 to identify critical path.

Profiles each step individually with torch.cuda.synchronize() timing.

Usage:
    python benchmarks/profile_intracard_cp.py
    nsys profile -o intracard_cp python benchmarks/profile_intracard_cp.py --nsys
"""

from __future__ import annotations

import argparse
import sys
import time

# Remove system dist-packages with broken torchvision (incompatible with torch 2.9+).
sys.path = [p for p in sys.path if '/usr/local/lib/python3.12/dist-packages' not in p
            and '/usr/lib/python3/dist-packages' not in p]

import torch  # noqa: E402
import triton  # noqa: E402

from cula.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from cula.ops.cp.chunk_delta_h import (
    _get_num_sms,
    _precompute_intracard_indices,
    compute_subseq_len,
    intracard_merge,
    intracard_pre_scan,
    prepare_subseq_cu_seqlens,
)
from fla.ops.utils.index import prepare_chunk_indices


def make_inputs(seq_len: int, H: int, K: int, V: int, dtype=torch.bfloat16, device="cuda"):
    """Create synthetic inputs for profiling (B=1, varlen)."""
    B = 1
    T = seq_len
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    w = torch.randn(B, T, H, K, dtype=dtype, device=device)
    u = torch.randn(B, T, H, V, dtype=dtype, device=device)
    gk = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
    cu_seqlens = torch.tensor([0, T], dtype=torch.long, device=device)
    cu_seqlens_cpu = cu_seqlens.cpu()
    N = 1
    initial_state = torch.randn(N, H, K, V, dtype=torch.float32, device=device)
    return k, w, u, gk, cu_seqlens, cu_seqlens_cpu, initial_state


def bench_fn(fn, warmup=3, repeat=10):
    """Benchmark a callable with cuda sync, return list of times in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def median(times):
    ts = sorted(times)
    return ts[len(ts) // 2]


def run_config(seq_len: int, H: int, K: int, V: int, warmup=3, repeat=10):
    print(f"\n{'='*70}")
    print(f"Config: seq_len={seq_len}, H={H}, K={K}, V={V}")
    print(f"{'='*70}")

    k, w, u, gk, cu_seqlens, cu_seqlens_cpu, initial_state = make_inputs(seq_len, H, K, V)
    B, T = k.shape[:2]
    device = k.device
    chunk_size = 64
    N_orig = len(cu_seqlens_cpu) - 1

    num_sms = _get_num_sms(device)
    seq_lens = torch.diff(cu_seqlens_cpu)
    max_seq_len = int(seq_lens.max().item())
    subseq_len = compute_subseq_len(max_seq_len, num_sms, H, chunk_size)
    num_splits = (seq_len + subseq_len - 1) // subseq_len
    threshold = 3 * subseq_len
    activated = seq_len >= threshold

    print(f"  num_sms={num_sms}, subseq_len={subseq_len}, num_splits={num_splits}")
    print(f"  threshold={threshold}, activated={'YES' if activated else 'NO'}")

    if not activated:
        print("  Intracard CP not activated for this config. Skipping.")
        return

    # ===== Pre-compute indices (needed for all steps) =====
    boundaries, split_info, total_subseqs = prepare_subseq_cu_seqlens(
        cu_seqlens_cpu, subseq_len, chunk_size)
    if not split_info:
        print("  No sequences need splitting. Skipping.")
        return

    (non_first_indices,
     first_subseq_indices, last_subseq_indices, num_non_first,
     merge_seq_starts, merge_seq_counts, merge_init_offsets,
     ) = _precompute_intracard_indices(split_info, boundaries, N_orig)

    # Build cu_seqlens for split sub-sequences (extract from boundaries)
    starts = split_info.start_subseq_idx
    num_ss = split_info.num_subseqs
    cu_seqlens_split_values = []
    S_split_total = 0
    for s, n in zip(starts, num_ss):
        cu_seqlens_split_values.extend(boundaries[s:s + n + 1])
        S_split_total += n

    cu_seqlens_subseq_gpu = torch.tensor(boundaries, dtype=cu_seqlens_cpu.dtype, device=device)
    cu_seqlens_split_flat = torch.tensor(cu_seqlens_split_values, dtype=cu_seqlens_cpu.dtype, device=device)

    print(f"  total_subseqs={total_subseqs}, S_split_total={S_split_total}, "
          f"num_non_first={num_non_first}")

    # ===== 1. Baseline: sequential fwd_h (no split) =====
    print("\n  --- Baseline (no split, sequential fwd_h) ---")
    baseline_times = bench_fn(
        lambda: chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=None, gk=gk,
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=chunk_size,
            save_new_value=True,
            cu_seqlens=cu_seqlens,
        ),
        warmup=warmup, repeat=repeat,
    )
    baseline_ms = median(baseline_times)
    print(f"    Baseline fwd_h:  {baseline_ms:.3f} ms")

    # ===== 2. Step 0: CPU index compute =====
    print("\n  --- Step 0: CPU index compute ---")
    cpu_times = bench_fn(
        lambda: (
            compute_subseq_len(max_seq_len, num_sms, H, chunk_size),
            prepare_subseq_cu_seqlens(cu_seqlens_cpu, subseq_len, chunk_size),
            _precompute_intracard_indices(split_info, boundaries, N_orig),
        ),
        warmup=warmup, repeat=repeat,
    )
    cpu_ms = median(cpu_times)
    print(f"    CPU index:       {cpu_ms:.3f} ms")

    # ===== 3. Step 1: CPU→GPU transfer =====
    print("\n  --- Step 1: CPU->GPU transfer ---")
    transfer_times = bench_fn(
        lambda: (
            torch.tensor(boundaries, dtype=cu_seqlens_cpu.dtype, device=device),
            torch.tensor(cu_seqlens_split_values, dtype=cu_seqlens_cpu.dtype, device=device),
        ),
        warmup=warmup, repeat=repeat,
    )
    transfer_ms = median(transfer_times)
    print(f"    CPU->GPU:        {transfer_ms:.3f} ms")

    # ===== 4. Step 2: pre_scan (KEY METRIC) =====
    print("\n  --- Step 2: pre_scan (pre_process_fwd_kernel_merged) ---")
    pre_scan_times = bench_fn(
        lambda: intracard_pre_scan(
            k=k, w=w, u=u, gk=gk,
            cu_seqlens_subseq_split=cu_seqlens_split_flat,
            S_split=S_split_total,
            chunk_size=chunk_size,
        ),
        warmup=warmup, repeat=repeat,
    )
    pre_scan_ms = median(pre_scan_times)

    # Also get grid info for pre_scan
    BK = triton.next_power_of_2(K)
    BLOCK_SIZE = 32 if K <= 64 else 64
    grid_v = triton.cdiv(V, BLOCK_SIZE)
    grid_k = triton.cdiv(K, BLOCK_SIZE)
    grid_total = (grid_v + grid_k) * H * S_split_total
    print(f"    pre_scan:        {pre_scan_ms:.3f} ms  <-- KEY")
    print(f"      grid=({grid_v + grid_k}, {H}, {S_split_total}), total_blocks={grid_total}")

    # ===== 5. Step 3: merge =====
    print("\n  --- Step 3: merge (Triton tf32 kernel) ---")
    # Pre-compute hm for merge timing
    hm_for_merge = intracard_pre_scan(
        k=k, w=w, u=u, gk=gk,
        cu_seqlens_subseq_split=cu_seqlens_split_flat,
        S_split=S_split_total,
        chunk_size=chunk_size,
    )
    merge_times = bench_fn(
        lambda: intracard_merge(
            hm=hm_for_merge,
            split_info=split_info,
            num_non_first=num_non_first,
            merge_seq_starts=merge_seq_starts,
            merge_seq_counts=merge_seq_counts,
            merge_init_offsets=merge_init_offsets,
            device=device,
            initial_state=initial_state,
        ),
        warmup=warmup, repeat=repeat,
    )
    merge_ms = median(merge_times)
    print(f"    merge:           {merge_ms:.3f} ms  ({num_non_first} chain multiplies)")

    # ===== 6. Step 4: scatter =====
    print("\n  --- Step 4: scatter initial_state_expanded ---")
    # Pre-compute a fake initial_states_merge for scatter timing
    initial_states_merge = torch.randn(num_non_first, H, K, V, dtype=torch.float32, device=device)

    def do_scatter():
        ise = k.new_zeros(total_subseqs, H, K, V, dtype=torch.float32)
        ise[first_subseq_indices] = initial_state
        if num_non_first > 0:
            ise[non_first_indices] = initial_states_merge
        return ise

    scatter_times = bench_fn(do_scatter, warmup=warmup, repeat=repeat)
    scatter_ms = median(scatter_times)
    print(f"    scatter:         {scatter_ms:.3f} ms")

    # ===== 7. Step 5: prepare_chunk_indices =====
    print("\n  --- Step 5: prepare_chunk_indices ---")
    chunk_idx_times = bench_fn(
        lambda: prepare_chunk_indices(cu_seqlens_subseq_gpu, chunk_size),
        warmup=warmup, repeat=repeat,
    )
    chunk_idx_ms = median(chunk_idx_times)
    print(f"    chunk_indices:   {chunk_idx_ms:.3f} ms")

    # ===== 8. Step 6: main fwd_h with subseq cu_seqlens =====
    print("\n  --- Step 6: main fwd_h (with split cu_seqlens, parallel sub-seqs) ---")
    initial_state_expanded = do_scatter()
    chunk_indices_subseq = prepare_chunk_indices(cu_seqlens_subseq_gpu, chunk_size)

    fwd_h_split_times = bench_fn(
        lambda: chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=None, gk=gk,
            initial_state=initial_state_expanded,
            output_final_state=True,
            chunk_size=chunk_size,
            save_new_value=True,
            cu_seqlens=cu_seqlens_subseq_gpu,
            chunk_indices=chunk_indices_subseq,
        ),
        warmup=warmup, repeat=repeat,
    )
    fwd_h_split_ms = median(fwd_h_split_times)

    # Grid info for split fwd_h
    NT_split = len(chunk_indices_subseq)
    fwd_h_grid = (triton.cdiv(V, 64), total_subseqs * H)  # approximate
    print(f"    fwd_h (split):   {fwd_h_split_ms:.3f} ms")
    print(f"      NT_split={NT_split}, grid≈({triton.cdiv(V,64)}, {total_subseqs}*{H}={total_subseqs*H})")

    # ===== Summary =====
    e2e_estimated = cpu_ms + transfer_ms + pre_scan_ms + merge_ms + scatter_ms + chunk_idx_ms + fwd_h_split_ms
    overhead = pre_scan_ms + merge_ms + scatter_ms + chunk_idx_ms + transfer_ms + cpu_ms

    print(f"\n  {'='*60}")
    print(f"  SUMMARY")
    print(f"  {'='*60}")
    print(f"  Baseline (sequential):      {baseline_ms:8.3f} ms")
    print(f"  fwd_h (split, parallel):    {fwd_h_split_ms:8.3f} ms  "
          f"(speedup over baseline: {baseline_ms/fwd_h_split_ms:.2f}x)")
    print(f"  Intracard overhead:         {overhead:8.3f} ms")
    print(f"  Estimated E2E (sum):        {e2e_estimated:8.3f} ms  "
          f"(net speedup: {baseline_ms/e2e_estimated:.2f}x)")
    print()
    print(f"  Per-step breakdown:")
    steps = [
        ("CPU index compute",   cpu_ms),
        ("CPU->GPU transfer",   transfer_ms),
        ("pre_scan",            pre_scan_ms),
        ("merge",               merge_ms),
        ("scatter",             scatter_ms),
        ("chunk_indices",       chunk_idx_ms),
        ("main fwd_h (split)",  fwd_h_split_ms),
    ]
    for name, ms in steps:
        pct = ms / e2e_estimated * 100 if e2e_estimated > 0 else 0
        marker = "  <-- KEY" if name == "pre_scan" else ""
        print(f"    {name:25s} {ms:8.3f} ms  ({pct:5.1f}%){marker}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsys", action="store_true", help="Minimal output for nsys tracing")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"SMs: {_get_num_sms(torch.device('cuda'))}")

    configs = [
        # (seq_len, H, K, V)
        (131072, 8, 128, 128),   # 128K, 8 heads
        (65536,  8, 128, 128),   # 64K, 8 heads
        (131072, 16, 128, 128),  # 128K, 16 heads
    ]

    with torch.inference_mode():
        for seq_len, H, K, V in configs:
            run_config(seq_len, H, K, V, warmup=args.warmup, repeat=args.repeat)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Benchmark cuLA CuTeDSL K1+K2 vs FlashKDA C++ vs FLA Triton chunk_kda.

Uses the same settings as FlashKDA/benchmarks/bench_fwd.py:
  - H=96/64, D=128
  - T=8192 fixed, varlen cases
  - warmup=30, iters=200, repeats=5
  - scale=1/sqrt(D), lower_bound=-5.0
  - With fp32 initial/final state
"""

import argparse
import math
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ["CULA_FLASHKDA_USE_CUTE"] = "1"

import torch
import torch.nn.functional as F


def bench_fn(fn, warmup, iters, repeats):
    for _ in range(max(warmup, 1)):
        fn()
    torch.cuda.synchronize()

    all_ms = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        all_ms.extend([s.elapsed_time(e) for s, e in zip(starts, ends)])

    xs = sorted(float(x) for x in all_ms)
    n = len(xs)
    mean = sum(xs) / n if n else float("nan")
    mn = xs[0] if xs else float("nan")
    mx = xs[-1] if xs else float("nan")
    median = xs[n // 2] if xs else float("nan")
    return mean, mn, mx, median


def _setup_cula_k1k2(q, k, v, g, beta, scale_float, LOWER_BOUND, A_log, dt_bias,
                      initial_state_fp32, cu_seqlens_int32, seq_lens, K1_CHUNK):
    """Prepare cuLA K1/K2 workspaces once, return closures for K1-only and K2-only."""
    from cula.ops.flashkda.k1 import launch_k1
    from cula.ops.flashkda.k2 import launch_k2

    H = q.shape[2]
    D = q.shape[3]
    device = q.device

    is_varlen = cu_seqlens_int32 is not None
    N = len(seq_lens)

    if is_varlen:
        aligned_lens = [((sl + K1_CHUNK - 1) // K1_CHUNK) * K1_CHUNK for sl in seq_lens]
        total_aligned = sum(aligned_lens)
        T_total = total_aligned
        total_tiles = total_aligned // K1_CHUNK

        cu_aligned = [0]
        cu_tiles = [0]
        for al in aligned_lens:
            cu_aligned.append(cu_aligned[-1] + al)
            cu_tiles.append(cu_tiles[-1] + al // K1_CHUNK)
        cu_seqlens_aligned = torch.tensor(cu_aligned, dtype=torch.int32, device=device)
        cu_seqlens_tiles = torch.tensor(cu_tiles, dtype=torch.int32, device=device)

        q_pad = torch.zeros((1, total_aligned, H, D), dtype=q.dtype, device=device)
        k_pad = torch.zeros_like(q_pad)
        v_pad = torch.zeros_like(q_pad)
        g_pad = torch.zeros_like(q_pad)
        beta_pad = torch.full((1, total_aligned, H), -80.0, dtype=beta.dtype, device=device)

        orig_seqlens = [0]
        for sl in seq_lens:
            orig_seqlens.append(orig_seqlens[-1] + sl)

        for i, sl in enumerate(seq_lens):
            src_s, src_e = orig_seqlens[i], orig_seqlens[i] + sl
            dst_s = cu_aligned[i]
            dst_e = dst_s + sl
            q_pad[0, dst_s:dst_e] = q[0, src_s:src_e]
            k_pad[0, dst_s:dst_e] = k[0, src_s:src_e]
            v_pad[0, dst_s:dst_e] = v[0, src_s:src_e]
            g_pad[0, dst_s:dst_e] = g[0, src_s:src_e]
            beta_pad[0, dst_s:dst_e] = beta[0, src_s:src_e]

        q_use, k_use, v_use, g_use, beta_use = q_pad, k_pad, v_pad, g_pad, beta_pad
    else:
        T_total = q.shape[0] * q.shape[1]
        total_tiles = T_total // K1_CHUNK
        cu_seqlens_tiles = None
        q_use, k_use, v_use, g_use, beta_use = q, k, v, g, beta

    beta_flat = torch.empty(H, T_total, dtype=beta_use.dtype, device=device)
    beta_flat.copy_(beta_use.view(T_total, H).transpose(0, 1))

    K1_D = 128
    n_qk = total_tiles * H * K1_CHUNK * K1_D
    n_cc = total_tiles * H * K1_CHUNK * K1_CHUNK
    ws_qd = torch.empty(n_qk, dtype=torch.bfloat16, device=device)
    ws_kd = torch.empty(n_qk, dtype=torch.bfloat16, device=device)
    ws_kr = torch.empty(n_qk, dtype=torch.bfloat16, device=device)
    ws_gt = torch.empty(total_tiles * H * K1_D, dtype=torch.float32, device=device)
    ws_inv = torch.empty(n_cc, dtype=torch.bfloat16, device=device)
    ws_mqk = torch.empty(n_cc, dtype=torch.bfloat16, device=device)

    out_k2 = torch.zeros_like(q_use)

    final_state = None
    if initial_state_fp32 is not None:
        final_state = torch.zeros_like(initial_state_fp32)

    def run_k1():
        launch_k1(
            q_use, k_use, g_use, A_log, dt_bias, beta_flat,
            scale_float, LOWER_BOUND,
            ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk,
        )

    def run_k2():
        launch_k2(
            v_use, beta_flat,
            ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk,
            out_k2, cu_seqlens_tiles,
            initial_state=initial_state_fp32,
            final_state=final_state,
        )

    def run_k1k2():
        launch_k1(
            q_use, k_use, g_use, A_log, dt_bias, beta_flat,
            scale_float, LOWER_BOUND,
            ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk,
        )
        launch_k2(
            v_use, beta_flat,
            ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk,
            out_k2, cu_seqlens_tiles,
            initial_state=initial_state_fp32,
            final_state=final_state,
        )

    # Warmup K1 once so K2 has valid workspace data
    run_k1()
    torch.cuda.synchronize()

    return run_k1, run_k2, run_k1k2


def run_case(seq_lens, H, D, warmup, iters, repeats, implementations):
    device = torch.device("cuda")
    LOWER_BOUND = -5.0
    scale_float = 1.0 / math.sqrt(D)

    varlen = len(seq_lens) > 1
    T_total = sum(seq_lens)
    N = len(seq_lens)

    tag = f"varlen seqs={seq_lens}" if varlen else f"fixed T={T_total}"
    print(f"\n{'='*70}")
    print(f"  {tag}, H={H}, D={D}")
    print(f"  warmup={warmup}, iters={iters}, repeats={repeats}")
    print(f"{'='*70}")

    q = F.normalize(torch.randn((1, T_total, H, D), dtype=torch.float32, device=device), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn((1, T_total, H, D), dtype=torch.float32, device=device), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn((1, T_total, H, D), dtype=torch.bfloat16, device=device)
    g = torch.randn((1, T_total, H, D), dtype=torch.bfloat16, device=device)
    beta = torch.randn((1, T_total, H), dtype=torch.bfloat16, device=device)
    A_log = torch.rand(H, dtype=torch.float32, device=device)
    dt_bias = torch.rand(H, D, dtype=torch.float32, device=device)

    initial_state_fp32 = torch.arange(N * H * D * D, dtype=torch.float32, device=device).reshape(N, H, D, D)
    final_state_fp32 = torch.zeros_like(initial_state_fp32)
    out = torch.zeros_like(q)

    if varlen:
        cu_seqlens = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
            dtype=torch.long, device=device,
        )
        cu_seqlens_int32 = cu_seqlens.int()
    else:
        cu_seqlens = None
        cu_seqlens_int32 = None

    results = {}

    # --- flash_kda C++ (fp32 state) ---
    if "flash_kda" in implementations:
        try:
            import flash_kda
            extra = {"cu_seqlens": cu_seqlens} if varlen else {}

            def run_flash_kda():
                flash_kda.fwd(q, k, v, g, beta, scale_float, out,
                              A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
                              initial_state=initial_state_fp32,
                              final_state=final_state_fp32, **extra)

            mean, mn, mx, med = bench_fn(run_flash_kda, warmup, iters, repeats)
            results["flash_kda_cpp"] = (mean, mn, mx, med)
            print(f"  flash_kda C++ (fp32 st): mean={mean:.4f} min={mn:.4f} max={mx:.4f} median={med:.4f} ms")
        except ImportError:
            print(f"  flash_kda C++: SKIPPED (not installed)")
        except Exception as e:
            print(f"  flash_kda C++: ERROR ({e})")

    # --- flash_kda C++ (no state) ---
    if "flash_kda_nostate" in implementations:
        try:
            import flash_kda
            extra = {"cu_seqlens": cu_seqlens} if varlen else {}

            def run_flash_kda_nostate():
                flash_kda.fwd(q, k, v, g, beta, scale_float, out,
                              A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
                              **extra)

            mean, mn, mx, med = bench_fn(run_flash_kda_nostate, warmup, iters, repeats)
            results["flash_kda_nostate"] = (mean, mn, mx, med)
            print(f"  flash_kda C++ (no st)  : mean={mean:.4f} min={mn:.4f} max={mx:.4f} median={med:.4f} ms")
        except ImportError:
            print(f"  flash_kda C++ (no st): SKIPPED (not installed)")
        except Exception as e:
            print(f"  flash_kda C++ (no st): ERROR ({e})")

    # --- cuLA CuTeDSL K1 only, K2 only, K1+K2 ---
    if any(x in implementations for x in ("cula", "cula_k1", "cula_k2", "cula_nostate")):
        try:
            from cula.ops.flashkda.k1 import CHUNK as K1_CHUNK

            # With state
            if any(x in implementations for x in ("cula", "cula_k1", "cula_k2")):
                run_k1, run_k2, run_k1k2 = _setup_cula_k1k2(
                    q, k, v, g, beta, scale_float, LOWER_BOUND,
                    A_log, dt_bias, initial_state_fp32, cu_seqlens_int32, seq_lens, K1_CHUNK,
                )

                if "cula_k1" in implementations or "cula" in implementations:
                    mean, mn, mx, med = bench_fn(run_k1, warmup, iters, repeats)
                    results["cula_k1"] = (mean, mn, mx, med)
                    print(f"  cuLA K1 only          : mean={mean:.4f} min={mn:.4f} max={mx:.4f} median={med:.4f} ms")

                if "cula_k2" in implementations or "cula" in implementations:
                    mean, mn, mx, med = bench_fn(run_k2, warmup, iters, repeats)
                    results["cula_k2"] = (mean, mn, mx, med)
                    print(f"  cuLA K2 only          : mean={mean:.4f} min={mn:.4f} max={mx:.4f} median={med:.4f} ms")

                if "cula" in implementations:
                    mean, mn, mx, med = bench_fn(run_k1k2, warmup, iters, repeats)
                    results["cula_k1k2"] = (mean, mn, mx, med)
                    print(f"  cuLA K1+K2 (fp32 st)  : mean={mean:.4f} min={mn:.4f} max={mx:.4f} median={med:.4f} ms")

            # No state
            if "cula_nostate" in implementations:
                run_k1_ns, run_k2_ns, run_k1k2_ns = _setup_cula_k1k2(
                    q, k, v, g, beta, scale_float, LOWER_BOUND,
                    A_log, dt_bias, None, cu_seqlens_int32, seq_lens, K1_CHUNK,
                )

                mean, mn, mx, med = bench_fn(run_k1k2_ns, warmup, iters, repeats)
                results["cula_nostate"] = (mean, mn, mx, med)
                print(f"  cuLA K1+K2 (no st)    : mean={mean:.4f} min={mn:.4f} max={mx:.4f} median={med:.4f} ms")

        except Exception as e:
            import traceback
            print(f"  cuLA CuTeDSL: ERROR ({e})")
            traceback.print_exc()

    # --- FLA Triton chunk_kda ---
    if "chunk_kda" in implementations:
        try:
            from fla.ops.kda import chunk_kda
            h0_ck = initial_state_fp32.clone()
            extra_ck = {"cu_seqlens": cu_seqlens} if varlen else {}

            def run_chunk_kda():
                chunk_kda(
                    q=q, k=k, v=v, g=g, beta=beta,
                    scale=scale_float,
                    initial_state=h0_ck,
                    output_final_state=True,
                    use_gate_in_kernel=True,
                    use_qk_l2norm_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True,
                    A_log=A_log, dt_bias=dt_bias,
                    lower_bound=LOWER_BOUND,
                    transpose_state_layout=True,
                    **extra_ck,
                )

            mean, mn, mx, med = bench_fn(run_chunk_kda, warmup, iters, repeats)
            results["chunk_kda"] = (mean, mn, mx, med)
            print(f"  FLA chunk_kda (Triton) : mean={mean:.4f} min={mn:.4f} max={mx:.4f} median={med:.4f} ms")
        except ImportError:
            print(f"  FLA chunk_kda: SKIPPED (not installed)")
        except Exception as e:
            print(f"  FLA chunk_kda: ERROR ({e})")

    # --- Speedup summary ---
    if len(results) >= 2:
        print(f"\n  Speedups (vs flash_kda C++ fp32 state):")
        ref_key = "flash_kda_cpp" if "flash_kda_cpp" in results else list(results.keys())[0]
        ref_mean = results[ref_key][0]
        ref_med = results[ref_key][3]
        for name, (mean, mn, mx, med) in results.items():
            if name != ref_key:
                speedup_mean = ref_mean / mean if mean > 0 else float("inf")
                speedup_med = ref_med / med if med > 0 else float("inf")
                print(f"    {name}: {speedup_mean:.2f}x (mean), {speedup_med:.2f}x (median)")

    return results


FIXED_CASES = [
    [8192],
]

VARLEN_CASES = [
    [1300, 547, 2048, 963, 271, 3063],
    [1024] * 8,
]


def main():
    p = argparse.ArgumentParser(description="Benchmark cuLA vs FlashKDA vs FLA")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--mode", choices=["fixed", "varlen", "all"], default="all")
    p.add_argument("--H", type=int, nargs="+", default=[96, 64])
    p.add_argument("--D", type=int, default=128)
    p.add_argument("--impl", nargs="+",
                   default=["flash_kda", "flash_kda_nostate", "cula", "cula_nostate", "chunk_kda"],
                   help="Implementations to benchmark")
    args = p.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    cases = []
    if args.mode in ("fixed", "all"):
        cases.extend(FIXED_CASES)
    if args.mode in ("varlen", "all"):
        cases.extend(VARLEN_CASES)

    for H in args.H:
        for seq_lens in cases:
            run_case(seq_lens, H, args.D, args.warmup, args.iters, args.repeats, args.impl)


if __name__ == "__main__":
    main()

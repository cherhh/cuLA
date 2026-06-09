#!/usr/bin/env python3
"""
K1 vs K2 split timing: cuLA CuTeDSL vs FlashKDA C++.

cuLA: timed directly with CUDA events (separate K1/K2 calls).
FlashKDA C++: timed via nsys profiling (single fwd call, parse kernel durations).
"""

import argparse
import math
import os
import sys
import pathlib
import subprocess
import tempfile
import sqlite3

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


def _nsys_profile_flashkda(seq_lens, H, D, scale, lower_bound, has_state, niters):
    """Launch a subprocess under nsys to get per-kernel durations for FlashKDA C++."""
    repo_root = str(pathlib.Path(__file__).resolve().parent.parent)
    script = f"""
import sys, pathlib
sys.path.insert(0, "{repo_root}")
import torch, torch.nn.functional as F, flash_kda

device = torch.device("cuda")
seq_lens = {seq_lens}
H, D = {H}, {D}
T_total = sum(seq_lens)
N = len(seq_lens)
varlen = N > 1

q = F.normalize(torch.randn((1,T_total,H,D),dtype=torch.float32,device=device),p=2,dim=-1).to(torch.bfloat16)
k = F.normalize(torch.randn((1,T_total,H,D),dtype=torch.float32,device=device),p=2,dim=-1).to(torch.bfloat16)
v = torch.randn((1,T_total,H,D),dtype=torch.bfloat16,device=device)
g = torch.randn((1,T_total,H,D),dtype=torch.bfloat16,device=device)
beta = torch.randn((1,T_total,H),dtype=torch.bfloat16,device=device)
A_log = torch.rand(H,dtype=torch.float32,device=device)
dt_bias = torch.rand(H,D,dtype=torch.float32,device=device)
out = torch.zeros_like(q)

kw = {{}}
if {has_state}:
    kw["initial_state"] = torch.zeros(N,H,D,D,dtype=torch.float32,device=device)
    kw["final_state"] = torch.zeros(N,H,D,D,dtype=torch.float32,device=device)
if varlen:
    kw["cu_seqlens"] = torch.tensor([0]+list(torch.cumsum(torch.tensor(seq_lens),dim=0).tolist()),dtype=torch.long,device=device)

for _ in range(5):
    flash_kda.fwd(q,k,v,g,beta,{scale},out,A_log=A_log,dt_bias=dt_bias,lower_bound={lower_bound},**kw)
torch.cuda.synchronize()

torch.cuda.cudart().cudaProfilerStart()
for _ in range({niters}):
    flash_kda.fwd(q,k,v,g,beta,{scale},out,A_log=A_log,dt_bias=dt_bias,lower_bound={lower_bound},**kw)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    base = tempfile.mktemp()
    nsys_rep = base + ".nsys-rep"
    sqlite_path = base + ".sqlite"

    try:
        r = subprocess.run(
            ["nsys", "profile", "--capture-range=cudaProfilerApi",
             "--capture-range-end=stop", "-t", "cuda", "--stats=false",
             "-o", base, "-f", "true", sys.executable, script_path],
            capture_output=True, text=True, timeout=180,
        )
        if r.returncode != 0:
            print(f"    nsys profile failed (rc={r.returncode})")
            return None

        r2 = subprocess.run(
            ["nsys", "export", "--type=sqlite", "-o", sqlite_path, nsys_rep],
            capture_output=True, text=True, timeout=60,
        )
        if r2.returncode != 0:
            print(f"    nsys export failed")
            return None

        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT s.value AS name, (k.end - k.start) / 1e6 AS dur_ms
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.shortName
            ORDER BY k.start
        """)
        rows = cur.fetchall()
        conn.close()

        result = {}
        for name, dur_ms in rows:
            if name not in result:
                result[name] = []
            result[name].append(dur_ms)
        return result

    except Exception as e:
        print(f"    nsys error: {e}")
        return None
    finally:
        for p in [script_path, nsys_rep, sqlite_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def _setup_cula(q, k, v, g, beta, scale_float, LOWER_BOUND, A_log, dt_bias,
                initial_state_fp32, cu_seqlens_int32, seq_lens, K1_CHUNK):
    from cula.ops.flashkda.k1 import launch_k1
    from cula.ops.flashkda.k2 import launch_k2

    H, D = q.shape[2], q.shape[3]
    device = q.device
    is_varlen = cu_seqlens_int32 is not None
    N = len(seq_lens)

    if is_varlen:
        aligned_lens = [((sl + K1_CHUNK - 1) // K1_CHUNK) * K1_CHUNK for sl in seq_lens]
        total_aligned = sum(aligned_lens)
        T_total = total_aligned

        cu_aligned = [0]
        cu_tiles = [0]
        for al in aligned_lens:
            cu_aligned.append(cu_aligned[-1] + al)
            cu_tiles.append(cu_tiles[-1] + al // K1_CHUNK)
        cu_seqlens_tiles = torch.tensor(cu_tiles, dtype=torch.int32, device=device)

        q_pad = torch.zeros((1, total_aligned, H, D), dtype=q.dtype, device=device)
        k_pad = torch.zeros_like(q_pad)
        v_pad = torch.zeros_like(q_pad)
        g_pad = torch.zeros_like(q_pad)
        beta_pad = torch.full((1, total_aligned, H), -80.0, dtype=beta.dtype, device=device)

        orig = [0]
        for sl in seq_lens:
            orig.append(orig[-1] + sl)
        for i, sl in enumerate(seq_lens):
            s, e = orig[i], orig[i] + sl
            d = cu_aligned[i]
            q_pad[0, d:d+sl] = q[0, s:e]
            k_pad[0, d:d+sl] = k[0, s:e]
            v_pad[0, d:d+sl] = v[0, s:e]
            g_pad[0, d:d+sl] = g[0, s:e]
            beta_pad[0, d:d+sl] = beta[0, s:e]

        q_use, k_use, v_use, g_use, beta_use = q_pad, k_pad, v_pad, g_pad, beta_pad
    else:
        T_total = q.shape[0] * q.shape[1]
        cu_seqlens_tiles = None
        q_use, k_use, v_use, g_use, beta_use = q, k, v, g, beta

    total_tiles = T_total // K1_CHUNK
    beta_flat = torch.empty(H, T_total, dtype=beta_use.dtype, device=device)
    beta_flat.copy_(beta_use.view(T_total, H).transpose(0, 1))

    n_qk = total_tiles * H * K1_CHUNK * 128
    n_cc = total_tiles * H * K1_CHUNK * K1_CHUNK
    ws_qd = torch.empty(n_qk, dtype=torch.bfloat16, device=device)
    ws_kd = torch.empty(n_qk, dtype=torch.bfloat16, device=device)
    ws_kr = torch.empty(n_qk, dtype=torch.bfloat16, device=device)
    ws_gt = torch.empty(total_tiles * H * 128, dtype=torch.float32, device=device)
    ws_inv = torch.empty(n_cc, dtype=torch.bfloat16, device=device)
    ws_mqk = torch.empty(n_cc, dtype=torch.bfloat16, device=device)
    out_k2 = torch.zeros_like(q_use)
    final_state = torch.zeros_like(initial_state_fp32) if initial_state_fp32 is not None else None

    def run_k1():
        launch_k1(q_use, k_use, g_use, A_log, dt_bias, beta_flat,
                       scale_float, LOWER_BOUND, ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk)

    def run_k2():
        launch_k2(v_use, beta_flat, ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk,
                         out_k2, cu_seqlens_tiles,
                         initial_state=initial_state_fp32, final_state=final_state)

    def run_k1k2():
        run_k1()
        run_k2()

    run_k1()
    torch.cuda.synchronize()
    return run_k1, run_k2, run_k1k2


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def run_case(seq_lens, H, D, warmup, iters, repeats):
    device = torch.device("cuda")
    LOWER_BOUND = -5.0
    scale_float = 1.0 / math.sqrt(D)
    varlen = len(seq_lens) > 1
    T_total = sum(seq_lens)
    N = len(seq_lens)

    tag = f"varlen seqs={seq_lens}" if varlen else f"fixed T={T_total}"
    print(f"\n{'='*78}")
    print(f"  {tag}, H={H}, D={D}")
    print(f"{'='*78}")

    q = F.normalize(torch.randn((1, T_total, H, D), dtype=torch.float32, device=device), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn((1, T_total, H, D), dtype=torch.float32, device=device), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn((1, T_total, H, D), dtype=torch.bfloat16, device=device)
    g = torch.randn((1, T_total, H, D), dtype=torch.bfloat16, device=device)
    beta = torch.randn((1, T_total, H), dtype=torch.bfloat16, device=device)
    A_log = torch.rand(H, dtype=torch.float32, device=device)
    dt_bias = torch.rand(H, D, dtype=torch.float32, device=device)
    initial_state = torch.arange(N * H * D * D, dtype=torch.float32, device=device).reshape(N, H, D, D)

    cu_seqlens_int32 = None
    if varlen:
        cu_seqlens_int32 = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
            dtype=torch.int32, device=device,
        )

    # ── FlashKDA C++ (nsys) ──
    print(f"\n  Profiling FlashKDA C++ via nsys ({iters} iters)...")
    nsys_data = _nsys_profile_flashkda(seq_lens, H, D, scale_float, LOWER_BOUND, True, iters)
    fkda_k1 = fkda_k2 = None
    if nsys_data:
        for name, times in sorted(nsys_data.items(), key=lambda x: -len(x[1])):
            med = _median(times)
            cnt = len(times)
            label = "K1" if "prepare" in name.lower() else ("K2" if "recurrence" in name.lower() else "??")
            if label == "K1":
                fkda_k1 = med
            elif label == "K2":
                fkda_k2 = med
            print(f"  FlashKDA {label:2s} ({name[:40]:40s}): median={med:.4f} ms  (n={cnt})")
        if fkda_k1 and fkda_k2:
            print(f"  FlashKDA K1+K2 sum                             : {fkda_k1+fkda_k2:.4f} ms")

    # ── cuLA CuTeDSL (CUDA events) ──
    print()
    from cula.ops.flashkda.k1 import CHUNK as K1_CHUNK
    run_k1, run_k2, run_k1k2 = _setup_cula(
        q, k, v, g, beta, scale_float, LOWER_BOUND,
        A_log, dt_bias, initial_state, cu_seqlens_int32, seq_lens, K1_CHUNK,
    )

    m1, n1, x1, med1 = bench_fn(run_k1, warmup, iters, repeats)
    m2, n2, x2, med2 = bench_fn(run_k2, warmup, iters, repeats)
    mb, nb, xb, medb = bench_fn(run_k1k2, warmup, iters, repeats)

    print(f"  cuLA K1 (prepare)    : mean={m1:.4f} min={n1:.4f} max={x1:.4f} median={med1:.4f} ms")
    print(f"  cuLA K2 (recurrence) : mean={m2:.4f} min={n2:.4f} max={x2:.4f} median={med2:.4f} ms")
    print(f"  cuLA K1+K2           : mean={mb:.4f} min={nb:.4f} max={xb:.4f} median={medb:.4f} ms")

    # ── Comparison ──
    if fkda_k1 is not None and fkda_k2 is not None:
        print(f"\n  {'─'*60}")
        print(f"  {'Kernel':20s} {'FlashKDA C++':>14s} {'cuLA':>14s} {'cuLA speedup':>14s}")
        print(f"  {'─'*60}")
        print(f"  {'K1 (prepare)':20s} {fkda_k1:13.4f}  {med1:13.4f}  {fkda_k1/med1:13.2f}x")
        print(f"  {'K2 (recurrence)':20s} {fkda_k2:13.4f}  {med2:13.4f}  {fkda_k2/med2:13.2f}x")
        print(f"  {'K1+K2 total':20s} {fkda_k1+fkda_k2:13.4f}  {medb:13.4f}  {(fkda_k1+fkda_k2)/medb:13.2f}x")


def main():
    p = argparse.ArgumentParser(description="K1/K2 split benchmark: cuLA vs FlashKDA C++")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--mode", choices=["fixed", "varlen", "all"], default="all")
    p.add_argument("--H", type=int, nargs="+", default=[96, 64])
    p.add_argument("--D", type=int, default=128)
    args = p.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    FIXED = [[8192]]
    VARLEN = [[1300, 547, 2048, 963, 271, 3063], [1024] * 8]

    cases = []
    if args.mode in ("fixed", "all"):
        cases.extend(FIXED)
    if args.mode in ("varlen", "all"):
        cases.extend(VARLEN)

    for H in args.H:
        for seq_lens in cases:
            run_case(seq_lens, H, args.D, args.warmup, args.iters, args.repeats)


if __name__ == "__main__":
    main()

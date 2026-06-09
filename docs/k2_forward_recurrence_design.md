# K2 Forward Recurrence — 技术拆解

---

## 技术点总结

1. **Warp Specialization（6 warps / 192 threads）**
   将 CTA 划分为三种角色：Warp 0–3（4 个 MMA warp，128 threads，负责 Phase 1–6 的所有 GEMM + CUDA Core epilogue）、Warp 4（TMA Load）、Warp 5（TMA Store）。通过 mbarrier ring 实现 load/compute/store 三级流水。

2. **Phase 1+4 Dual GEMM Fusion**
   `u_pre = kd @ state` 和 `out = qd @ state` 共享同一个 State B-operand load（K-loop 8 次 `ldmatrix.x4.trans`），单次 B-load 同时喂两个 GEMM，减半 sState LDSM 流量（32 KB state × 2 → ×1）。

3. **MOVM_T 寄存器内 C→B 转置（零 SMEM round-trip）**
   Phase 2 产生 u（C-frag fp32）→ cast bf16 → `movmatrix.sync.aligned.m8n8.trans.b16` → B-frag（寄存器内）。Phase 3 结果同样 MOVM_T → `tCrU_T_post`，该 B-frag 一路存活到 Phase 6 结束。彻底消除中间结果 sU_T 的 SMEM 分配（省 4 KB × STAGES）+ 跨 warp barrier。

4. **sKr 双视图 Zero-Copy Transpose**
   sKr 物理上只分配一份 K_INTER swizzled `(CHUNK, D)`。Phase 1/4 的 B-load 走 `LDSM_N` 读 `(CHUNK, D)` 视图；Phase 6 的 A-load 走 `LDSM_T` 读 MN_INTER `(D, CHUNK)` 转置视图。同一块 SMEM 字节，两种 atom 解释，无需额外 transpose buffer。

5. **Phase 6 BLOCKED State Update**
   `state[D,D] = state*gt + kr^T @ u_post` 是 (128, 128, 16) GEMM。全量 acc 需要 64 fp32 regs/thread，拆成 D/CHUNK = 4 次 M-block 迭代，每次 (16, 128, 16)，acc 仅 16 fp32 regs + in-frag read-modify-write epilogue。

6. **sState 持久驻留 SMEM（32 KB）**
   recurrence 的核心——D×D bf16 state matrix 在整个序列遍历期间常驻 SMEM，每个 chunk 的 Phase 1/4 读、Phase 6 原地 RMW 更新。不参与 pipeline stage ring。

---

## 常量与线程配置

| 项目 | 值 |
|------|-----|
| CHUNK | 16 |
| D | 128 |
| Grid | `(N_sequences, H_heads, 1)` |
| Block | 192 threads (6 warps) |
| MMA atom | SM80 `m16n8k16` bf16→fp32 |
| tiled_mma | `atom_layout=(1,4,1)`, `perm=(16,32,16)` → per-CTA (16, 128, 16) |
| InputStages | 2 (double buffer) |
| OutputStages | 2 (double buffer) |
| SMEM | ~85 KB |
| Regs/thread | 156 (NVVM) vs 78 (ptxas C++) |

---

## 算法公式（每个 chunk t）

```
输入（K1 预计算，从 GMEM 加载）：
  kd[16,128]  = k_decayed    (k × exp(cumsum(g)))
  qd[16,128]  = q_decayed    (q × exp(cumsum(g)) × scale)
  kr[16,128]  = k_restored   (k × exp(-g) × exp(g_total))
  gt[128]     = g_total      (exp(sum(g)), 衰减因子)
  INV[16,16]  = (I - L)^{-1} (Neumann 逆)
  Mqk[16,16]  = qd @ kr^T    (query-key 矩阵)
  v[16,128]   = value
  β[16]       = beta          (gating scalar)

持久状态（SMEM 常驻, D×D = 128×128 bf16）：
  state[128,128]

Phase 1: u_pre   = kd @ state          (16,128) × (128,128) → (16,128), K=128
Phase 2: u       = sigmoid(β) * (v - u_pre)     逐元素 (16,128)
Phase 3: u_post  = INV @ u             (16,16) × (16,128) → (16,128), K=16
Phase 4: out     = qd @ state + Mqk @ u_post
                   ↑ Phase 1 fused     ↑ (16,16) × (16,128), K=16
Phase 5: write out → GMEM (TMA store)
Phase 6: state   = state * gt + kr^T @ u_post
                   (128,128)*scalar + (128,16)×(16,128) → (128,128)
```

---

## 计算逻辑拆解 + Pipeline 设计

### LOAD Warp (Warp 4)

每个 chunk t，TMA async load 8 个 tensor 到 input ring `stage[s]`：

```
wait   sMbarE[s]          # 等 COMPUTE 释放 slot
expect sMbar[s], TMA_BYTES
copy   v       [16, 128] bf16  → sV[s]       4096 B
copy   kd      [16, 128] bf16  → sKd[s]      4096 B
copy   qd      [16, 128] bf16  → sQd[s]      4096 B
copy   kr      [16, 128] bf16  → sKr[s]      4096 B
copy   INV     [16, 16]  bf16  → sINV[s]      512 B
copy   Mqk     [16, 16]  bf16  → sMqk[s]      512 B
copy   gt      [128]     fp32  → sGt[s]       512 B
copy   beta    [16]      bf16  → sBeta[s]      32 B (padded 128B)
arrive sMbar[s]                                total = 13,824 B
```

### COMPUTE Warps (Warp 0–3, 128 threads)

```
for each chunk t:
  wait sMbar[s]                    # 等 LOAD 填好 slot

  ┌─ Phase 1+4-main (FUSED DUAL GEMM) ─────────────────────
  │  clear tCrU[16 fp32], tCrOut[16 fp32]
  │  for k in range(D//16 = 8):        # K-loop
  │    B = LDSM_T(sState[:, k*16:(k+1)*16])     ← 共享 B-load
  │    A_kd = LDSM_N(sKd[:, k*16:(k+1)*16])
  │    tCrU   += mma(A_kd, B)                    ← Phase 1
  │    A_qd = LDSM_N(sQd[:, k*16:(k+1)*16])
  │    tCrOut += mma(A_qd, B)                    ← Phase 4-main
  │
  │  输入: sKd, sQd, sState (SMEM)
  │  输出: tCrU[16 fp32], tCrOut[16 fp32] (register)
  │  MMA次数: 8×2 = 16 mma.sync
  └──────────────────────────────────────────────

  ┌─ Phase 2 (CUDA Core + MOVM_T) ─────────────────────────
  │  beta_row0 = sigmoid(sBeta[row0])
  │  beta_row1 = sigmoid(sBeta[row1])
  │  for i in range(16):  # per C-frag element
  │    u_bf16[i] = bf16((sV[i] - tCrU[i]) * beta)
  │  for i in range(8):   # per u32 pair
  │    tCrU_T[i] = MOVM_T(u_bf16_u32[i])        ← C→B transpose
  │
  │  输入: tCrU[16 fp32], sV (SMEM), sBeta (SMEM)
  │  输出: tCrU_T[8 u32] (register, B-frag for Phase 3)
  │  tCrU 生命周期结束
  └──────────────────────────────────────────────

  ┌─ Phase 3 (TC GEMM + MOVM_T) ───────────────────────────
  │  A_inv = LDSM_N(sINV[s])                     1 K-iter
  │  clear tCrU3[16 fp32]
  │  tCrU3 = mma(A_inv, tCrU_T)                  ← INV @ u
  │  for i in range(16):
  │    u3_bf16[i] = bf16(tCrU3[i])
  │  for i in range(8):
  │    tCrU_T_post[i] = MOVM_T(u3_bf16_u32[i])  ← C→B transpose
  │
  │  输入: sINV (SMEM), tCrU_T[8 u32] (register)
  │  输出: tCrU_T_post[8 u32] (register, 存活到 Phase 6 结束!)
  │  tCrU_T, tCrU3 生命周期结束
  │  MMA次数: 1 mma.sync
  └──────────────────────────────────────────────

  ┌─ Phase 4-epi (TC GEMM) ────────────────────────────────
  │  A_mqk = LDSM_N(sMqk[s])                     1 K-iter
  │  tCrOut += mma(A_mqk, tCrU_T_post)           ← Mqk @ u_post
  │
  │  输入: sMqk (SMEM), tCrU_T_post[8 u32], tCrOut[16 fp32]
  │  输出: tCrOut[16 fp32] (register, 累加完毕)
  │  MMA次数: 1 mma.sync
  └──────────────────────────────────────────────

  ┌─ Phase 5 (Store to sOut) ──────────────────────────────
  │  wait sMbarSE[s_out]                          # 等 STORE warp 释放
  │  for i in range(16):
  │    out_bf16[i] = bf16(tCrOut[i])
  │  STSM_N(out_bf16 → sOut[s_out])              ← stmatrix.x4
  │
  │  输入: tCrOut[16 fp32] (register)
  │  输出: sOut[s_out] (SMEM)
  │  tCrOut 生命周期结束
  └──────────────────────────────────────────────

  ┌─ Phase 6 (TC BLOCKED State Update) ────────────────────
  │  M_BLOCKS = D/CHUNK = 4
  │  for mi in range(4):
  │    A_kr_blk = LDSM_T(sKr_T_view[mi*16:(mi+1)*16, :])
  │    clear tCrUpd_blk[16 fp32]
  │    tCrUpd_blk = mma(A_kr_blk, tCrU_T_post)  ← kr_blk^T @ u_post
  │    # In-frag epilogue (CUDA Core):
  │    state_blk = load sState[mi*16:(mi+1)*16, :] via partition_C
  │    gt_blk = load sGt[mi*16:(mi+1)*16]
  │    for i in range(16):
  │      sState[..] = bf16(fp32(state_blk[i]) * gt_blk[i] + tCrUpd_blk[i])
  │
  │  输入: sKr (SMEM, MN_INTER view), tCrU_T_post[8 u32], sState, sGt
  │  输出: sState (SMEM, in-place RMW)
  │  tCrU_T_post 生命周期结束
  │  MMA次数: 4 mma.sync
  └──────────────────────────────────────────────

  barrier(compute 128 threads)
  fence_view_async_shared()
  arrive sMbarSF[s_out]                           # 通知 STORE warp
  arrive sMbarE[s]                                # 释放 input slot

  advance s, s_out, phase counters
```

### STORE Warp (Warp 5)

```
for each chunk t:
  wait   sMbarSF[s_out]       # 等 COMPUTE 填好 sOut
  TMA store sOut[s_out] → GMEM out[t]
  cp_async_bulk_commit + wait
  arrive sMbarSE[s_out]       # 释放 output slot
```

---

## SMEM 布局

```
BT=16, D=128, STAGES=2, OUT_STAGES=2

持久 (不参与 ring):
  sState    [128, 128] bf16  K_INTER swizzled    32768 B    32.0 KB

Input ring (×2 stages):
  sV        [16, 128]  bf16  K_INTER swizzled    4096 × 2 =  8.0 KB
  sKd       [16, 128]  bf16  K_INTER swizzled    4096 × 2 =  8.0 KB
  sQd       [16, 128]  bf16  K_INTER swizzled    4096 × 2 =  8.0 KB
  sKr       [16, 128]  bf16  K_INTER swizzled    4096 × 2 =  8.0 KB
  sINV      [16, 16]   bf16  row-major            512 × 2 =  1.0 KB
  sMqk      [16, 16]   bf16  row-major            512 × 2 =  1.0 KB
  sGt       [128]      fp32  contiguous            512 × 2 =  1.0 KB
  sBeta     [16]       bf16  padded 128B           128 × 2 =  0.25 KB

Output ring (×2 stages):
  sOut      [16, 128]  bf16  K_INTER swizzled    4096 × 2 =  8.0 KB

Barriers:
  sMbar     [2] Int64  load full                            16 B
  sMbarE    [2] Int64  load empty                           16 B
  sMbarSF   [2] Int64  store full                           16 B
  sMbarSE   [2] Int64  store empty                          16 B

总计 ≈ 85.3 KB

Block Limit Shared Mem = floor(228 / 85.3) = 2 blocks/SM
需 ≤76 KB 才能达到 3 blocks/SM
```

### SMEM 复用分析

- C++ 用 `union { pipeline_bufs; state_fp32_buf; }` 在 pipeline loop 前后复用空间做 fp32 state load/store。CuTeDSL 的 `SmemAllocator` 不支持 union。
- sKr 做 zero-copy transpose（K_INTER ↔ MN_INTER 双视图），省了 8 KB 的 sKr_T。
- MOVM_T 省了 sU_T（否则需要 `[16,128] bf16 × STAGES = 8 KB`）。

### 不可压缩的大头

| 缓冲区 | 大小 | 是否可省 |
|--------|------|---------|
| sState | 32 KB | 不可（算法核心，D×D persistent） |
| sV+sKd+sQd+sKr ×2 | 32 KB | 不可（TMA pipeline 最小 2 stage） |
| sOut ×2 | 8 KB | 不可（store pipeline 最小 2 stage） |
| sINV+sMqk+sGt+sBeta ×2 | 3.25 KB | 不可（已经很小） |

---

## Register 布局 (per compute thread)

### Fragment 生命周期图

```
Phase:    |--1+4main--|--2--|--3--|--4epi--|--5--|------6 (×4 mi)------|
          K-loop(×8)

tCrU      ■■■■■■■■■■■■■■■.                          16 fp32  Phase 1 acc
tCrOut    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■.           16 fp32  Phase 4 acc
tCrKd     ■■■■■■■■■■.                                4 u32   A-frag (K-loop)
tCrQd     ■■■■■■■■■■.                                4 u32   A-frag (K-loop)
tCrState  ■■■■■■■■■■.                                8 u32   B-frag (K-loop)
tCrU_pre  .............■■■.                           8 u32   Phase 2 scratch
tCrU_T    .............■■■■■.                         8 u32   B-frag (Phase 2→3)
tCrInv    .................■■.                        4 u32   A-frag Phase 3
tCrU3     .................■■■.                      16 fp32  Phase 3 acc
tCrU3_tmp .................■■■.                       8 u32   Phase 3 cast scratch
tCrU_T_p  ....................■■■■■■■■■■■■■■■■■■■■   8 u32   B-frag (Phase 3→6!)
tCrMqk    ........................■■.                 4 u32   A-frag Phase 4-epi
tCrOut_bf ...............................■■.          8 u32   Phase 5 cast
tCrKrA6   ...................................■.■.■.■  4 u32   A-frag Phase 6
tCrUpd    ...................................■.■.■.■ 16 fp32  Phase 6 acc (per mi)
st_frag   ...................................■.■.■.■  4 u32   Phase 6 state load
gt_frag   ...................................■.■.■.■  8 fp32  Phase 6 gt load
scalars   ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ ~20      misc
```

### 峰值分析

| 冲突点 | 同时 live 的 fragment | 理论 regs | 说明 |
|--------|----------------------|-----------|------|
| Phase 1+4 K-loop | tCrU + tCrOut + A/B operands | 68 | Dual GEMM 必须同时持有两个 acc |
| Phase 3 | tCrU_T + tCrU3 + tCrOut (still live) | 72 | tCrOut 跨 Phase 3 存活 |
| Phase 6 mi=0 | tCrU_T_post + tCrUpd + state + gt | 60 | tCrU_T_post 从 Phase 3 存活至此 |

```
理论峰值 ≈ 72 regs (Phase 3 区间)
实际 NVVM 分配: 156 regs (SSA lifetime 膨胀 ~2.2x)
C++ ptxas 分配: 78 regs (接近理论值)

Block Limit Registers = floor(65536 / (156 × 192)) = 2 blocks/SM
需 ≤113 regs/thread 才能达到 3 blocks/SM
```

---

## CUDA Core 逻辑拆解

```
Phase 2 (全部是 CUDA Core):
  ├─ beta load: sig0 = sigmoid(sBeta[row0]), sig1 = sigmoid(sBeta[row1])
  │   └─ sigmoid = 0.5 * (tanh(x*0.5) + 1.0)，使用 tanh.approx
  ├─ element-wise: diff = fp32(sV[i]) - tCrU[i]
  ├─ element-wise: u_bf16[i] = bf16(diff * sig)
  └─ MOVM_T: u32 = movmatrix.sync.aligned.m8n8.trans.b16(u32)  × 8 次

Phase 3 epilogue (CUDA Core):
  ├─ fp32→bf16 cast: tCrU3_bf16[i] = bf16(tCrU3[i])  × 16
  └─ MOVM_T: tCrU_T_post[i] = movm_t(u3_u32[i])  × 8 次

Phase 5 (CUDA Core):
  └─ fp32→bf16 cast: tCrOut_bf16[i] = bf16(tCrOut[i])  × 16

Phase 6 in-frag epilogue (CUDA Core, per mi-block):
  ├─ state load: state_frag[i] = sState[partition_C coords]
  ├─ gt load: gt_frag[i] = sGt[m_off + coord_row]
  ├─ RMW: sState[..] = bf16(fp32(state_frag[i]) * gt_frag[i] + tCrUpd_blk[i])
  └─ × 4 mi-blocks × 16 elements/block = 64 RMW operations
```

---

## Pipeline 依赖图

```
Time →
Chunk:  t-1              t                  t+1

LOAD:   ─────────────── load[t,s] ────────── load[t+1,s'] ──
                         arrive sMbar[s]

COMPUTE: ── Phase1-6[t-1] ─┤ wait sMbar[s]
                            ├─ Phase 1+4-main (16 mma)
                            ├─ Phase 2 (CUDA Core)
                            ├─ Phase 3 (1 mma + CUDA Core)
                            ├─ Phase 4-epi (1 mma)
                            │  wait sMbarSE[s_out]
                            ├─ Phase 5 (STSM → sOut)
                            ├─ Phase 6 (4 mma + CUDA Core RMW)
                            ├─ barrier(128t)
                            ├─ fence_view_async_shared
                            ├─ arrive sMbarSF[s_out]  → STORE
                            └─ arrive sMbarE[s]       → LOAD

STORE:   ── store[t-2] ──┤ wait sMbarSF[s_out]
                          ├─ TMA store sOut → GMEM
                          ├─ cp_async_bulk_wait
                          └─ arrive sMbarSE[s_out]   → COMPUTE

关键依赖链 (recurrence):
  Phase 6[t] 写 sState → Phase 1[t+1] 读 sState
  通过 barrier(128 compute threads) + mbarrier[s] 保证顺序
```

---

## 与 C++ 实现的结构差异

| 方面 | C++ (fwd_kernel2.cuh) | CuTeDSL (flashkda_k2_phaseN.py) |
|------|----------------------|----------------------------------|
| Pipeline | `PipelineTmaAsync<3>` | 手写 mbarrier, STAGES=2 |
| Per-warp N-blocks | 显式 `[2]` array 循环 | tiled_mma 自动 32 列/warp |
| Phase 6 M-blocks | 8 (per 16×16 block, 2 per warp) | 4 (per 16×128 block) |
| State access | `s_acc_T` 转置视图 + LDSM_T/STSM_T | `select(mode=[1,0])` + `partition_C` |
| A/B operand prefetch | `ring_A_kr[PREFETCH]` | 无 (直接 load-use) |
| SMEM union | pipeline buf ↔ fp32 state buf | 无 (SmemAllocator 不支持) |
| MOVM_T | `SM75_U32x1_MOVM_T::copy` 内联 | `@dsl_user_op` + `llvm.inline_asm` |
| FP32 state | TMA load fp32 → cvt → sState bf16 | 逐元素 load gmem fp32 → cast → sState bf16 |
| Regs/thread | 78 | 156 |

---

## CuTeDSL 编译器限制对 K2 的影响

### 编译管线

```
CuTeDSL Python → MLIR → NVVM IR → libNVVM → cubin
                                    ↑
                              不经过 ptxas
```

### 具体限制

| 手段 | C++ (ptxas) | CuTeDSL (NVVM) |
|------|------------|-----------------|
| `--maxrregcount=N` | 有效，强制 spill 到 local mem | 无效 |
| `__launch_bounds__` | 间接限制 regs | CuTeDSL 有等价物但 NVVM 不一定 honor |
| `setmaxnregister` (SM90 动态) | 可用于 warpgroup(128t) | 需要 128t 对齐；192t = 1.5 warpgroup，不可用 |
| 跨 phase 寄存器复用 | ptxas 激进折叠 dead value | NVVM SSA 保守分配 |

### 结果

```
C++ (ptxas):     78 regs × 192 threads = 14,976 → 4 blocks/SM → 37.5% occupancy
CuTeDSL (NVVM): 156 regs × 192 threads = 29,952 → 2 blocks/SM → 18.75% occupancy
```

2x 寄存器膨胀直接导致 occupancy 减半。156 vs 78 的差距 100% 归因于 NVVM backend 的寄存器分配器不如 ptxas 激进——代码结构完全一致。

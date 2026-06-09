# FlashKDA Prefill — CuteDSL 移植设计文档

> 目标文件：`cula/ops/flashkda_prefill.py`  
> 参考实现：`/ossfs/workspace/FlashKDA/csrc/smxx/`  
> 目标架构：SM90（Hopper）

---

## 1. 算法回顾

FlashKDA prefill 将完整计算切成两个独立的 CUDA kernel，沿"并行维度"自然分割：

```
K1 (Prepare)   grid=(total_tiles, H)   — token-parallel
  └─ g 激活 → L2 归一化 → decay 应用
     → L / Mqk 构造 → 16×16 矩阵求逆
     → 结果写入 workspace (gmem)

K2 (Recurrence)  grid=(N, H)           — seq-parallel (每 CTA 负责一条序列、一个 head)
  └─ 逐 chunk 的 delta-rule 循环递推
     → 读 workspace → MMA 融合 → 写 out / final_state
```

**CHUNK=16, D=128**（当前仅支持这一配置，与 C++ 参考保持一致）。

---

## 2. 总体数据流

```
输入 gmem: q[B,T,H,K], k[B,T,H,K], v[B,T,H,K], g[B,T,H,K],
           beta[B,T,H], A_log[H], dt_bias[H,K], initial_state[N,H,D,D]

Workspace (gmem, K1 生产 → K2 消费):
  ┌─ ws_kd   [H·n_tiles, 16, 128]  bf16   k_decayed
  ├─ ws_qd   [H·n_tiles, 16, 128]  bf16   q_decayed  
  ├─ ws_kr   [H·n_tiles, 16, 128]  bf16   k_restored
  ├─ ws_gt   [H·n_tiles, 128]      fp32   g_total (exp)
  ├─ ws_inv  [H·n_tiles, 16, 16]   bf16   INV
  └─ ws_mqk  [H·n_tiles, 16, 16]   bf16   Mqk

输出 gmem: out[B,T,H,D], final_state[N,H,D,D]
```

---

## 3. Kernel 1 (Prepare) 设计

### 3.1 线程配置

| 项目 | 取值 |
|------|------|
| Grid | `(total_tiles, H)` |
| Block | 256 threads (8 warps) |
| SMEM  | ≈44 KB (union 复用，Phase A/B 不重叠) |
| `__launch_bounds__` | `(256, 8)` |

### 3.2 SMEM 布局（union 复用）

```python
# Phase A (q/k/g 存活)
smem.q      : [16, 128] bf16  (QKLayout = RowMajor)
smem.k      : [16, 128] bf16
smem.g_fp32 : [16, 128] fp32  (累积 cumsum)
smem.g_bf16 : [16, 128] bf16  (TMA 目标，后与 k_restored union)

# Phase B (q/k/g smem 可复用)
smem.k_decayed   : [16, 128] bf16  MMA K_INTER layout
smem.q_decayed   : [16, 128] bf16
smem.k_inv       : [16, 128] bf16
smem.k_restored  : [16, 128] bf16
smem.L           : [16, 16]  bf16/fp16 (L 矩阵)
smem.INV         : [16, 16]  bf16      (Neumann 输出)
smem.Mqk         : [16, 16]  bf16      (q_decayed @ k_inv)

# 始终存活
smem.beta    : [32] bf16  (对齐到 8-element 边界，TMA 1D)
smem.dt_bias : [128] fp32 or smem.g_total : [128] fp32  (union)
smem.tma_barrier : ClusterTransactionBarrier
```

### 3.3 计算流程

```
Step 0: (thread 0) TMA 单射 — q, k, beta, g_bf16, dt_bias (单阶段，无流水)
        ↕ overlap: 其它线程预计算 a_log_exp = exp(A_log[head])
Step 1: 等待 TMA barrier
Step 2: QK L2 归一化 (CUDA core, warp reduction)
Step 3: 门激活 + cumsum (128 线程负责 D=128 列)
        g_val = a_log_exp * (g_bf16[row,col] + dt_bias[col])
        g_val = gate_scale * sigmoid(g_val)    ← sigmoid via tanh.approx
        累积 sum → g_smem[row,col] = cumsum
        g_total[col] = final sum
        另 128 线程: 对 tail token 补零 k_smem[actual_len:]
Step 4: exp2(g_total) in-place   ← ex2.approx.ftz.f32
Step 5: decay_apply (256 线程, 8 elem/thread vectorized load)
        exp_cumsum = ex2(g)
        q_decayed = q * exp_cumsum * scale
        k_decayed = k * exp_cumsum
        k_inv     = k * ex2(-g)
        k_restored= k_inv * g_total_exp
Step 6: L_Mqk GEMM (单 warp 调用 mma_m16n16)
        L   = k_decayed[16,128] @ k_inv^T[128,16]  (fp16 accum)
        Mqk = q_decayed[16,128] @ k_inv^T[128,16]  (bf16 accum)
Step 7: 下三角掩码 + beta 缩放 + INV = I - L (256 线程, 1 elem/thread)
Step 8: Neumann 矩阵求逆 (fp16, 单 warp)
        INV = (I-L)^{-1} = I + L + L^2 + L^4 + L^8
Step 9: TMA store workspace — k_decayed, q_decayed, k_restored, g_total, INV, Mqk
```

---

## 4. Kernel 2 (Recurrence) 设计

### 4.1 线程配置（Warp Specialization）

| Warp ID | 角色 | 数量 |
|---------|------|------|
| 0–3 (128 线程) | MMA (compute) | 4 |
| 4 (32 线程)    | LOAD_QKG (TMA producer) | 1 |
| 5 (32 线程)    | STORE (TMA consumer) | 1 |
| — | NonParticipant | 0 |

总计 192 线程 (6 warps)。

Pipeline：`PipelineTmaAsync<InputStages=3>` + `PipelineAsync<OutputStages=2>`

### 4.2 SMEM 布局

```python
smem.state_acc : [128, 128] bf16, MMA K_INTER layout  (贯穿整个 K2 生命周期)

# 流水线 double/triple buffer
smem.input[InputStages]:
    .v          : [16, 128] bf16
    .beta       : [32] bf16
    .k_decayed  : [16, 128] bf16
    .q_decayed  : [16, 128] bf16
    .k_restored : [16, 128] bf16
    .g_total    : [128] fp32
    .INV        : [16, 16] bf16
    .Mqk        : [16, 16] bf16

smem.output[OutputStages]:
    .out : [16, 128] bf16

# fp32 state 加载/存储时复用 input/output 的 union
smem.state_fp32_buf : [128×128] fp32  (与 pipeline buf union)
smem.load_pipeline  : PipelineTmaAsync SharedStorage
smem.store_pipeline : PipelineAsync SharedStorage
smem.state_tma_barrier : ClusterTransactionBarrier
```

### 4.3 计算流程（每个 chunk t）

```
LOAD warp:
  ├─ producer_acquire(load_write)
  ├─ TMA load v          → input[stage].v
  ├─ TMA load beta       → input[stage].beta
  ├─ TMA load ws_kd      → input[stage].k_decayed
  ├─ TMA load ws_qd      → input[stage].q_decayed
  ├─ TMA load ws_kr      → input[stage].k_restored
  ├─ TMA load ws_gt      → input[stage].g_total
  ├─ TMA load ws_inv     → input[stage].INV
  ├─ TMA load ws_mqk     → input[stage].Mqk
  └─ load_write++

MMA warps (4 warps, 每 warp 负责 2×16 列 state):
  Phase 1: Dual GEMM (k@state, q@state)
           k_s = k_decayed[16,128] @ state[128,128]  → u_acc (fp32 regs)
           q_s = q_decayed[16,128] @ state[128,128]  → out_acc (fp32 regs)

  Phase 2: cast u_acc/out_acc → bf16 regs
           load v, INV, beta from smem

  Phase 3: u = (v - k_s) * beta               (寄存器内逐元素)
           u = INV @ u                          (GEMM，u 作为 B operand)
           ← 需要 MOVM_T 将 u 从 C-format 转为 B-format，避免 smem round-trip

  Phase 4: out = q_s + Mqk @ u                 (GEMM + 加法)
           ← 同样需要 MOVM_T 将 u 转为 B-format

  Phase 5: store out_bf16 → output[stage].out

  Phase 6: state_acc update:
           state[D,D] = state * g_total + k_restored^T @ u
           ← k_restored 按 LDSM_T layout 读入寄存器 (MN_INTER)
           ← u (已在 B-format 寄存器, tCrB_u_arr[]) 直接复用 Phase 4 的结果

STORE warp:
  ├─ consumer_wait(out_read)
  ├─ TMA store output[stage].out → gmem out (full tile)
  │   或: 逐元素写 tail tile
  └─ consumer_release + out_read++
```

---

## 5. CuteDSL API 可用性分析

### 5.1 直接可用的 API

| C++ 对应 | CuteDSL API | 备注 |
|----------|-------------|------|
| `make_tma_copy(SM90_TMA_LOAD)` | `cute.make_tma_copy_desc` / `cute.TMA` | K1/K2 TMA 加载 |
| `make_tma_copy(SM90_TMA_STORE)` | 同上 | workspace / out TMA 存储 |
| `SM80_16x8x16_F32BF16BF16F32_TN` | `cute.nvgpu.sm80.MmaAtom` (bf16 mma) | Phase 1/3/4/6 GEMM |
| `SM80_16x8x16_F16F16F16F16_TN` | `cute.nvgpu.sm80.MmaAtom` (fp16 mma) | Neumann 逆 (K1 Step 8) |
| `SM75_U32x4_LDSM_N` | `cute.arch.LdsmCopyAtom` N 变体 | A/B smem→reg copy |
| `SM75_U16x8_LDSM_T` | `cute.arch.LdsmCopyAtom` T 变体 | k_restored_T 读取 |
| `SM90_U32x4_STSM_N` | `cute.arch.StsmCopyAtom` N 变体 | C smem 写回 |
| `SM90_U16x8_STSM_T` | `cute.arch.StsmCopyAtom` T 变体 | state_T 写回 |
| `cutlass::PipelineTmaAsync` | `cutlass.pipeline.PipelineTmaAsync` | K2 LOAD pipeline |
| `cutlass::PipelineAsync` | `cutlass.pipeline.PipelineAsync` | K2 STORE pipeline |
| `cutlass::arch::NamedBarrier` | `cutlass.pipeline.NamedBarrier` | MMA warp 间同步 |
| `ClusterTransactionBarrier` | `cutlass.pipeline.ClusterBarrier` | TMA mbarrier |
| `elect_one_sync` | `cute.arch.elect_one_sync()` | TMA 单线程 issue |
| `cute::transform` | `cute.transform` | 寄存器级别类型转换 |
| `local_tile` | `cute.local_tile` | tile 分块 |

### 5.2 需要 NVVM 内联 PTX 的操作

以下三个操作在 CuteDSL 中**没有现成高层 API**，必须通过
`cutlass._mlir.dialects.llvm.inline_asm` 注入 PTX：

---

#### 5.2.1 `ex2.approx.ftz.f32` — 快速以 2 为底指数

**用途**：K1 Step 3/5 的 gate cumsum 指数化（`log2(e)` 变基已在 gate activation 中完成，直接用 ex2 即可）。

**PTX**：
```ptx
ex2.approx.ftz.f32 dst, src;
```

**CuteDSL 实现**：
```python
@cutlass.dsl_user_op
def ex2_approx_ftz(x: Float32, *, loc=None, ip=None) -> Float32:
    result = _llvm.inline_asm(
        _T.f32(), [x],
        "ex2.approx.ftz.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return Float32(result)
```

**影响**：不用则退回 `exp2f` 标准库，吞吐量明显下降（ex2.approx 比 exp2f 快约 4×）。**强烈建议引入**。

---

#### 5.2.2 `tanh.approx.f32` — sigmoid 近似 via tanh

**用途**：K1 Step 3 的门激活 `sigmoid(x) = tanh(x/2)/2 + 0.5`，以及 K1 Step 7 的 `sigmoid(beta)` 计算。

**PTX**：
```ptx
tanh.approx.f32 dst, src;
```

**CuteDSL 实现**：
```python
@cutlass.dsl_user_op
def tanh_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    result = _llvm.inline_asm(
        _T.f32(), [x],
        "tanh.approx.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return Float32(result)

def sigmoid_approx(x: Float32) -> Float32:
    th = tanh_approx(x * Float32(0.5))
    return th * Float32(0.5) + Float32(0.5)
```

**影响**：不用则退回 `tanhf` + 额外计算，性能约下降 2-3×。**强烈建议引入**。

---

#### 5.2.3 `movmatrix.sync.aligned.m8n8.trans.b16` — 寄存器内矩阵转置

**用途**（K2 最关键）：K2 Phase 3 和 Phase 4 中，MMA 的 C-format 输出（u_bf16，行主序 fragment）需要转成 B-format（列主序 fragment），才能作为下一次 MMA 的 B 操作数。C++ 实现用 `SM75_U32x1_MOVM_T::copy()` 完成这一转换**完全在寄存器内**，零 smem round-trip。

**PTX**（每次处理 1 个 u32 = 2 个 bf16）：
```ptx
movmatrix.sync.aligned.m8n8.trans.b16 dst_reg, src_reg;
```

**CuteDSL 实现**：
```python
@cutlass.dsl_user_op
def movm_t_u32(src: cute.Int32, *, loc=None, ip=None) -> cute.Int32:
    """SM75_U32x1_MOVM_T: register-file matrix transpose (1×u32 / 2×bf16)."""
    result = _llvm.inline_asm(
        _T.i32(), [src],
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return cute.Int32(result)
```

对 fragment 中 4 个 u32 各调用一次，即可完成整个 16×16 C→B 格式转换：

```python
def convert_c_to_b_format(u_frag):
    """将 MMA C-format bf16 fragment 转换为 B-format，全程寄存器内。"""
    u32_regs = cute.recast(u_frag, cute.Int32)  # reinterpret as i32 x4
    for i in cutlass.range_constexpr(4):
        u32_regs[i] = movm_t_u32(u32_regs[i])
    return cute.recast(u32_regs, cute.BFloat16)
```

**影响**：  
- 若**不引入**此指令，必须通过 smem 做转置（写 C-layout smem → 换 layout 读回），每个 chunk 增加 ~2 次 smem 读写，lat 大幅增加。  
- 在 K2 Phase 6 state update 中，state[128,128] 的 g_total scaling 也可以利用 MOVM_T 保持 B operand 的复用，进一步减少 smem 访问。  
- **强烈建议引入**；这是 K2 性能最关键的 NVVM 操作。

---

### 5.3 可选的 NVVM 操作（次优先级）

| 操作 | 用途 | 备注 |
|------|------|------|
| `cvt.f32.bf16` inline PTX | bf16→f32 转换 | CuteDSL `cute.cast` 应可自动生成，优先用高层 API |
| `__hadd2` / `__hmul2` PTX | Neumann 级数 fp16x2 加法 | 可用 CuteDSL `cute.transform` 替代 |

---

## 6. 实现路线图

### Phase 0：辅助 NVVM 函数（`flashkda_nvvm.py` 或内嵌）
- [ ] `ex2_approx_ftz`
- [ ] `tanh_approx` + `sigmoid_approx`
- [ ] `movm_t_u32` + `convert_c_to_b_format`

### Phase 1：K1 Prepare kernel
- [ ] SMEM layout 设计（union）
- [ ] TMA descriptor 构建（q, k, beta, g, dt_bias）
- [ ] L2 归一化（CUDA core，warp shuffle reduce）
- [ ] 门激活 + cumsum（CUDA core，列并行）
- [ ] decay_apply（vectorized load, CUDA core）
- [ ] `mma_m16n16` helpers（bf16/fp16 两种 accum）
- [ ] L_Mqk GEMM（单 warp）
- [ ] 下三角掩码 + beta + `INV = I - L`
- [ ] Neumann 矩阵求逆（fp16，单 warp，4 次幂次 GEMM）
- [ ] TMA store workspace

### Phase 2：K2 Recurrence kernel
- [ ] Warp role 分配 + pipeline 初始化
- [ ] Initial state TMA load（bf16 / fp32 两路）
- [ ] LOAD warp 流水（所有 workspace TMA load）
- [ ] MMA warp Phase 1：Dual GEMM k@state + q@state
- [ ] MMA warp Phase 2–4：v-sub + beta + INV GEMM + Mqk GEMM（含 MOVM_T）
- [ ] MMA warp Phase 5：store output
- [ ] MMA warp Phase 6：state update（g_total scaling + k_restored^T @ U）
- [ ] STORE warp：TMA store out + final state

### Phase 3：Python wrapper + autotuning
- [ ] `allocate_workspace(N, H, T, CHUNK, D)` 工具函数
- [ ] varlen / fixed-len 分支
- [ ] `flash_kda_prefill(q, k, v, g, beta, ...)` 公共 API

---

## 7. 精度设计决策（与 C++ 对齐）

| 运算 | 精度 | 原因 |
|------|------|------|
| g cumsum accumulator | fp32 | 避免误差累积 |
| g_total, dt_bias | fp32 | 宽动态范围 |
| q/k L2 norm | fp32 | rsqrt 精度 |
| L 矩阵 | **fp16** | 元素在 [-1,1]，fp16 足够，避免 fp32→bf16 cast |
| INV (Neumann) | **fp16** | 同上 |
| state_acc | **bf16** | 减少 smem 占用；FMA 用 fp32 |
| 门激活 sigmoid | fp32（via tanh.approx） | 精度/速度平衡 |

---

## 8. 关键性能注意事项

1. **K1 SMEM union 复用**：Phase A（q/k/g）与 Phase B（k_decayed 等）生命周期不重叠 → 节省约 14 KB smem → 更高 occupancy。
2. **K2 MOVM_T**：避免每个 chunk 约 4 次 smem round-trip（Phase 3+4 各 2 次）是 K2 性能的核心。
3. **ex2 代替 exp**：K1 Gate activation 先乘 `log2(e)`（folded into a_log），再用 ex2，消除 mul-then-exp 中的变基 FMA。
4. **K2 双 GEMM 共享 B operand**：Phase 1 中 k@state 和 q@state 共享同一个 B (state) prefetch buffer，warp 间分工列块而非行块，减少 smem 读次数。
5. **STORE warp tail tile 处理**：tail tile（token 数 < 16）必须逐元素写，避免越界覆盖相邻序列的 gmem。

---

## 9. 文件结构规划

```
cula/ops/
├── flashkda_prefill.py      ← 主实现（K1 + K2 + wrapper）
└── flashkda_prefill_design.md  ← 本文档
```

内部模块划分（均在同一 .py 文件中）：

```python
# ── NVVM helpers ──────────────────────────────────────
def ex2_approx_ftz(x): ...
def tanh_approx(x): ...
def sigmoid_approx(x): ...
def movm_t_u32(src): ...

# ── K1: Prepare ───────────────────────────────────────
class FlashKDAPrepare:
    def __call__(self, ...): ...  # @cute.jit kernel

# ── K2: Recurrence ────────────────────────────────────
class FlashKDARecurrence:
    def __call__(self, ...): ...  # @cute.jit kernel

# ── Public API ────────────────────────────────────────
def flash_kda_prefill(q, k, v, g, beta, scale, out,
                      A_log, dt_bias, lower_bound,
                      initial_state=None, final_state=None,
                      cu_seqlens=None): ...
```

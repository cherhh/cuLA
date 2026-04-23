# CuteDSL练习笔记：在 CuteDSL 中通过 Inline PTX / NVVM 实现 UMMA Masked MMA

## 背景

SM100 (Blackwell) 引入了 `tcgen05.mma` 指令，其中包含一个 128-bit(对应 TMEM 128 lanes) 的 **disable-output-lane mask** 操作数。这个 mask 允许在一次 MMA 指令中选择性地禁用部分输出行的累加，直接在 Tensor Core 执行时跳过指定行的计算。

然而，CuteDSL 的高层 API（`cute.gemm()` / `make_tiled_mma()`）并未暴露这个 mask 操作数。为了探索这一硬件特性，我们需要绕过高层抽象，直接操作底层指令。

根据池帅的建议需要使用到 cutedsl inline ptx 能力，还贴心地给了 tilelang 的相关代码地址, https://github.com/tile-ai/tilelang/tree/main/tilelang/contrib/cutedsl


## 实现方案

我们实现了两条路径来发射带 mask 的 `tcgen05.mma`：

### 路径 1：NVVM 原生 Op（SS / TS 形式）

CuteDSL 底层的 MLIR 基础设施包含 `nvvm.tcgen05_mma` op，它直接对应 PTX `tcgen05.mma` 指令，并且接受 `write_disable_mask` 参数（`vector<4xi32>` 类型）。通过 `@dsl_user_op` 装饰器，我们可以在 `@cute.jit` 函数内部直接构造 MLIR IR：

```python
@dsl_user_op
def _do(c_val, da_val, db_val, dv_val, sc_val,
        m0_val, m1_val, m2_val, m3_val, *, loc=None, ip=None):
    # 构造 TMEM 指针 (address space 6)
    d_ptr = llvm.inttoptr(ptr6_ty, _ir(c_val, loc, ip), loc=loc, ip=ip)

    # 构造 vector<4xi32> mask
    undef = llvm.mlir_undef(vec4i32_ty, loc=loc, ip=ip)
    v = llvm.InsertElementOp(undef, _ir(m0_val, loc, ip), idx0, ...)
    v = llvm.InsertElementOp(v,     _ir(m1_val, loc, ip), idx1, ...)
    v = llvm.InsertElementOp(v,     _ir(m2_val, loc, ip), idx2, ...)
    mask = llvm.InsertElementOp(v,  _ir(m3_val, loc, ip), idx3, ...)

    # 发射 NVVM op → 编译为 PTX tcgen05.mma 指令
    _nvvm.tcgen05_mma(
        mma_kind=_nvvm.Tcgen05MMAKind.TF32,
        cta_group=_nvvm.Tcgen05GroupKind.CTA_1,
        d=d_ptr, a=da_ir, b=db_ir, idesc=dv_ir,
        enable_input_d=enable_d,
        write_disable_mask=mask,    # ← 关键：128-bit 输出行屏蔽
    )
```

这条路径覆盖了 **SS 形式**（SMEM A + SMEM B）和 **TS 形式**（TMEM A + SMEM B）。

### 路径 2：Inline ASM（WS 形式）

Weight-stationary 变体有对应的 `nvvm.tcgen05_mma_ws` op，但该 op 不接受 `write_disable_mask` 参数，无法生成带 mask 的 `.ws.` 变体。因此当前 WS 形式使用 `llvm.inline_asm`：

```python
asm_str = (
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, $4, 0;\n"
    "tcgen05.mma.ws.cta_group::1.kind::tf32 "
    "[$0], $1, $2, $3, p, 0;\n"
    "}"
)
llvm.inline_asm(None, [c, a, b, desc, scale], asm_str, "r,l,l,r,r", ...)
```

> **Aside**：A 神建议能用 `nvvm.*` 原生 op 就优先用 `nvvm.*`，而非 `llvm.inline_asm`。因为 `nvvm`  基于 MLIR， MLIR pass 可以对其进行分析和优化。

## Disable-Output-Lane Mask 布局

128-bit mask 由 4 个 uint32 组成，每个 word 控制 M 维度的一组行

```
mask[0]:  rows  0-15   (group 0)    0x00000000 = active, 0xFFFFFFFF = disabled
mask[1]:  rows 16-31   (group 1)
mask[2]:  rows 32-47   (group 2)
mask[3]:  rows 48-63   (group 3)
```

##  inline ptx / nvvm 有什么用

### 1. CuteDSL 的"逃生舱" 机制

CuteDSL 提供了 两个层次的底层访问能力：

- **`llvm.inline_asm`**：嵌入任意 PTX 汇编字符串，用于 NVVM op 尚未覆盖的指令变体
- **`nvvm.*` 原生 op**：直接映射到 PTX 指令的 MLIR op，语义最精确，编译器可以做更好的优化

这意味着 CuteDSL **不是一个封闭系统**。任何 PTX ISA 中定义的指令，即使高层 API 尚未封装，开发者都可以通过这些机制直接使用。

### 2. AI + 充分的硬件文档 = 更强的 kernel 开发能力

本项目的实现过程其中有大量部分由 AI coding agent 完成. 当 AI 被喂入足够丰富的 CUDA 底层文档时，它可以：

1. 从 PTX ISA 规范中提取指令编码细节（如 descriptor 位域、mask 布局）
2. 定位 CuteDSL/MLIR 中对应的底层 op 和类型系统
3. 完成你需要的非标 API 功能

这表明 **CuteDSL + AI + 硬件文档** 的组合具有很强的扩展性，理论上可以覆盖任何新硬件特性——只要 PTX ISA 有定义、NVVM 有对应 op 或支持 inline asm，就能在 CuteDSL 中使用。

## 相关代码

逻辑封转：
https://github.com/inclusionAI/cuLA/blob/cavan.inline_ptx/cula/kda/ptx_umma_masked.py 主要代码基本是仿照 tilelang cutedsl backend 示例构造出来的

测试代码：
https://github.com/inclusionAI/cuLA/blob/cavan.inline_ptx/tests/test_ptx_umma_masked.py


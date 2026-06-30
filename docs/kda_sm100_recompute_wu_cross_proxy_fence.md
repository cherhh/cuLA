# SM100 `recompute_wu` — cross-proxy fence 缺失导致 `qg` 非确定性

**状态:已修复(PR [#77](https://github.com/inclusionAI/cuLA/pull/77))。** 本文是 post-mortem + 排查方法记录。
代码:`csrc/kda/sm100/kda_fwd_recomp_w_u_mainloop_sm100.hpp`,`if constexpr (StoreQG)` 分支(约 L503–615)。

## TLDR

`recompute_wu` kernel 在 `StoreQG=true`(对应上层 `disable_recompute=True`)时,输出 `qg`
**非确定性**。根因:Q 的 TMA-load pipeline 在 **consumer 侧** 做了 `consumer_wait → S2R 读 sQ →
consumer_release`,但 **release 之前没有 `fence.proxy.async.shared::cta`**。CUDA Core(generic
proxy)对 SMEM 的读取对 TMA(async proxy)**不可见**,于是 TMA 在 `mbar.arrive(bar_empty)` 之后
就开始复写 sQ,而此时 S2R 还没真正读完 → sQ 被覆写 → `q_reg` 拿到错的数据。

修复:在 `consumer_release` 之前加 `fence_view_async_shared()`(= PTX `fence.proxy.async.shared::cta`)。

## 现象

- 触发条件:`StoreQG=true`,`Q` pipeline `stage=1`,sQ 由 TMA load 填充。
- **稳定的非确定性**:同输入多次跑,`qg` 结果 bit 不一致,**diff 较大**(不是 1-ULP 级别)。
- 复现强度:需要 **~10 万次** 重复才能稳定复现(timing-sensitive,见 cula-kernel-wiki 坑1:
  "某些 timing-sensitive 的 bug 需要 100K+ 才能复现")。
- `compute-sanitizer racecheck` **查不出**:racecheck 只追 generic proxy 的 `ld/st.shared`,
  对 TMA(async proxy)读写 SMEM 是盲的。

## 背景:两个 memory proxy

GPU 有两套独立的内存访问 proxy,各自有独立的一致性域:

| proxy | 谁用 | 指令 |
|---|---|---|
| **Generic proxy** | CUDA Core | `ld.shared`/`st.shared`(含 S2R/LDS) |
| **Async proxy** | TMA / UMMA | `cp.async.bulk` 等 |

关键:`mbarrier.arrive`(`consumer_release` 内部)**只保证 generic proxy 内的内存序**,
**不跨 proxy**。所以当 Core 用 `ld.shared`(S2R)读完 sQ 后调用 `consumer_release()`,
TMA(async proxy)**看不到** Core 的读取是否完成,会以为 buffer 已空、立刻复写。

参考:
- cula-kernel-wiki §坑1(本仓库这个 bug 的 canonical 条目)
- memory model 详解:<https://yang-yifan.github.io/blogs/memory_model/memory_model.html>
- 文章:<https://zhuanlan.zhihu.com/p/2041185737639518940>

## 根因:consumer_release 之前缺 cross-proxy fence

**Bug 版本(PR #77 之前)** —— S2R 读完 sQ 后直接 release,没有 fence:

```cpp
if constexpr (StoreQG) {
    q_pipeline.consumer_wait(q_pipe_state_read);          // 等 TMA 把 sQ 填好
    Tensor sQ = make_tensor(make_smem_ptr(...), SmemLayoutInputBF16{});

    // S2R: 把 sQ 读进寄存器 q_reg(异步完成,约 30 cycle)
    bf16x8 q_reg[TileT/16][TileK/64];
    for (...) q_reg[ti][k_yi] = *reinterpret_cast<bf16x8*>(&sQ(t, y));

    // ❌ 危险:S2R 还没真正读完,就把 sQ 还给 TMA;TMA 看不到 Core 的读
    q_pipeline.consumer_release(q_pipe_state_read);
    ++q_pipe_state_read;

    // ... 后面才用 q_reg 算 QG ...
}
```

### timing 分析(为什么会复写)

**Before(buggy)** —— `mbar.arrive` 大概率早于 S2R 真正完成,TMA 抢先复写:

```
CUDA Core 侧                          TMA 侧
  mbar.wait  bar_full
  S2R 发射                              mbar.wait  bar_empty   ← 等到 release
  mbar.arrive bar_empty  ── 危险 ──▶    TMA load(复写 sQ)     ← 此刻 S2R 还没读完
  S2R 真正完成(~30 cyc)                mbar.arrive bar_full
        ↑ 读到的是 TMA 复写后的新数据 → q_reg 被污染 → 非确定
```

**After(fixed)** —— fence 保证 S2R 已完成且对 async proxy 可见,再 release:

```
CUDA Core 侧                          TMA 侧
  mbar.wait  bar_full
  S2R 发射
  S2R 真正完成(~30 cyc)
  fence.proxy.async.shared::cta        mbar.wait  bar_empty
  mbar.arrive bar_empty  ──────────▶   TMA load(安全,sQ 已读完)
                                       mbar.arrive bar_full
```

## 修复

在 `consumer_release` 前插入 cross-proxy fence(当前代码 `kda_fwd_recomp_w_u_mainloop_sm100.hpp` L581–584):

```cpp
    // NOTE: must make smem visible from CUDA Core (general proxy) to TMA (async proxy)
    fence_view_async_shared();          // = fence.proxy.async.shared::cta
    // Release Q SMEM back to Load warp
    q_pipeline.consumer_release(q_pipe_state_read);
    ++q_pipe_state_read;
```

`fence_view_async_shared()` 把当前线程在 generic proxy 上对 SMEM 的访问(这里是 S2R 读 sQ)
排到后续 async proxy 访问(TMA 复写)之前,从而关闭复写窗口。本文件里**凡是 Core 读/写过、
且要归还给 TMA 的 SMEM buffer,release 前都配了这道 fence**(k@L400→405、q@L582→584、
v@L735→740、prologue_ready@L839→865;另有 consumer_wait 后紧跟 Core 写 SMEM 的也加,如
beta@L277→278、L670→671)。纯信号量式的 release(g/beta/acc_done 等,Core 未触碰其 SMEM、
或不归还给 TMA)则不需要 —— 见下方规则。

## 通用规则

> ⚠️ **凡是 CUDA Core 读取或写入了 SMEM,且该 buffer 即将归还给 TMA(consumer_release),
> 必须在 release 前插入 `fence_view_async_shared()`。`ld.shared`(S2R/LDS,读)和 `st.shared`
> (写)都需要 —— 读也会被复写污染。**

方向上只有一个方向需要显式 fence:

| 方向 | 需要 fence? | 原因 |
|---|---|---|
| Async → Generic(TMA → Core) | ❌ | TMA completion 隐式含 proxy fence |
| **Generic → Async(Core → TMA)** | ✅ | mbarrier release 不扩展到 async proxy |

## 排查方法(可复用)

1. 现象画像:**输出非确定 + racecheck 干净 + 某个 stage / TMA buffer 相关**。racecheck 干净
   不代表没竞争 —— 它看不见 async proxy(TMA)对 SMEM 的访问。
2. 复现要够狠:cross-proxy 这类 timing race 要 **100K+ 次** 才稳定复现;迭代不够会误判为"确定"。
3. 定位:盯住每一个 `consumer_wait … (Core 读/写 SMEM) … consumer_release`,检查 release 前是否
   有 `fence_view_async_shared()`;以及 `consumer_wait` 后若紧跟 Core 写 SMEM,也要 fence。
4. 修复后用 10K+(必要时 100K)determinism 回归守住。

## 关联

同类但**完全不同**的一个坑(供对照,别混淆):SM90 CP 的 `ws_beta` 越界 bug
(`docs/kda_sm90_cp_varlen_race.md`)—— 那个是 host 侧 workspace **欠分配导致 OOB**,
表象也是"输出非确定、racecheck 干净",但根因在 allocator 而非 proxy fence。两者都提醒:
**"输出非确定 + racecheck 干净"有两类常见来源 —— cross-proxy fence 缺失,或 OOB 进了 allocator slab。**

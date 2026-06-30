# SM90 KDA intracard-CP —— varlen 下 `ws_beta` 欠分配导致输出非确定

**状态:已定位并修复。** 一行 allocation bug。本文是 post-mortem。

## Bug

`ws_beta`(K1 产出的紧凑 raw-beta workspace,被 pre_scan + K2 读取)是按 **tile layout**
`ws_beta[wt_l*CHUNK + tidx]` 存储的,`wt_l = head*total_tiles + tile`,所以需要
`total_tiles*CHUNK*H` 个槽位。但它被按 `T_total*H` 分配(`cula/ops/kda/sm90/cp/driver.py`)。

- **Dense / 对齐:** `total_tiles*CHUNK == T_total` → 正确,无 bug。
- **Varlen(native,`v_is_varlen`):** `total_tiles` 是 **ceil** tile 和,所以
  `total_tiles*CHUNK > T_total`(partial-tile padding)。例如 lens0 `[1024,1,63,65,129]`:
  `total_tiles=83`,`total_tiles*CHUNK = 1328 > T_total = 1282`。高位 `wt_l`(`head=7`、
  tiles ~60–82)会越过 buffer 末尾 `(1328−1282)*H` 个元素。

K1 越界写 `ws_beta` 落到 **相邻的 torch-allocator 内存** 上,而 caching allocator 每次迭代
分配的相邻块是非确定的。于是它悄悄污染了邻近的 workspace(`ws_gt`、`ws_inv`),再沿
`pre_scan → b_seg → merge → carries → K2 → o` 一路传下去。`compute-sanitizer memcheck`
**查不出**:这个 OOB 落在 torch allocator 的 slab 之内,并没有越过所有 device 分配。

## 现象

`cula_kda_prefill(use_intracard_cp=True)` 在非对齐 varlen batch 上,输出 `o`
**非确定**(约 13–18% 的跑 bit 不一致,最差 ~3.5e-2),而递推状态 `fin` 始终 bit 稳定;
`o` 也过不了 vs-serial 的精度检查(rel_rmse 4.3e-2 ≫ tol)。修复后:`o` vs serial 的
rel_rmse **6e-5**,determinism **0/400**(即便是最小的 8-tile 段)。

## 修复

`driver.py`:把 `ws_beta` 的大小从 `T_total * H` 改成 `total_tiles * CHUNK * H`。一行。

## 怎么定位的(以及为什么绕了这么久)

误导点在于:`fin` 是 bit 确定的而 `o` 不是,且 `compute-sanitizer racecheck` 是干净的
—— 这把第一轮排查直接带进了 **K2 的 TMA 输出流水线**(store warp 的 SASS 审计、cross-proxy
fence、ncu)。那是个 **red herring**:K2 只是忠实地把被污染的输入产出的结果写出去而已。
甚至一次多 agent 的 SASS 审计还(正确地)认定 K2 输出同步是对的 —— 只是 bug 根本不在那。

真正破局的步骤,按顺序:
1. **固定输入时 K2 是确定的**(用捕获的 K1/pre_scan/merge 输出回放 K2 → 0/300)。
   所以竞争在 K2 上游。
2. **逐 kernel 分歧捕获** —— 把每个 kernel 的输出跨多次跑 clone 下来,找**第一个**分歧的 buffer:

   ```
   ws_qd/kd/kr/mqk: 0/200   ws_gt/ws_inv/ws_beta: 198/200   b_seg/carries/o: 198/200   m_seg/fin: 0/200
   ```

   源头明确是 **K1** 的 `ws_gt`/`ws_inv`/`ws_beta`。一看它们的 sizing 就发现了 `ws_beta` 越界。

**教训:** 遇到"输出非确定但状态确定、racecheck 干净、只在 varlen"这种画像,先去
**跨 kernel 流水线逐 buffer 捕获并二分**中间结果 —— 一个落进 allocator slab 的 OOB
对 racecheck/memcheck 都是隐形的,却会伪装成一个很深的 kernel 同步 bug。之前那一大套
红鲱鱼(cross-proxy fence、TMA store ordering、planner 端 `>=32-tile` 段长钳制)全都不必要,
已经回退。

## 回归守护

`tests/test_kda_sm90_intracard_cp.py::test_cp_determinism_varlen` 用 lens0 varlen 配置、
`s_split=8`(把 OOB tile 索引顶到最大)跑 determinism loop。原有的 `test_cp_determinism`
只覆盖了 dense(`total_tiles*CHUNK == T_total`),这正是这个 bug 逃过 CI 的原因。

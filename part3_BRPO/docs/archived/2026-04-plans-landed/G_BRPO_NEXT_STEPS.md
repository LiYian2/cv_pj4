# G-BRPO_NEXT_STEPS.md - 下一步优化方向

> 更新时间：2026-04-21 02:00 (Asia/Shanghai)

## 1. G-BRPO 当前状态

G-BRPO 已完成 BRPO 论文的 **完整对齐**（universe/score/action/timing）：
- Universe: `population_active` ✅
- Score: `brpo_unified_v1` ✅
- Action: `stochastic_bernoulli_opacity` ✅
- Timing: `current_step_probe_loss` ✅

**Replay eval 结果**：
- Direct BRPO current_step: **PSNR=24.019** (+0.0114 vs baseline 24.007)
- 改善幅度较小，但方向正确

## 2. 当前上游信号模块

上游 A1 使用的是 **exact BRPO-style** 配置：
- `confidence_mask_source = brpo`
- `rgb_mask_mode_effective = brpo_v2_raw`
- `depth_mask_mode_effective = brpo_v2_depth`

这与 STATUS.md 里的 `exact_brpo_full_target_v1` 一致（该配置 PSNR=24.174，仍 -0.01325 vs old A1）。

## 3. 下一步优化方向

根据 `part3_BRPO_A1_B3_vs_BRPO_detailed_analysis.md` 和 `GAUSSIAN_MANAGEMENT_DESIGN.md` 分析，当前瓶颈不在 G-BRPO（B3），而在 **A1 target-side contract**：

### 3.1 A1 核心问题

BRPO 的 A1 与当前实现的根本差异：

| 维度 | BRPO 论文 | 当前实现 |
|------|-----------|---------|
| Confidence 来源 | **独立 verifier**（pseudo-frame 与真实 reference 的几何验证） | **candidate score 同源**（target 和 confidence 都从 score 推出） |
| Confidence 类型 | 三值硬门控（1.0/0.5/0.0） | 连续平滑值 |
| Target 构造 | 先有 `I_t^{fix}`，再问能不能被验证 | candidate competition + softmax fusion |

**关键问题**：当前 A1 的 confidence 和 target 来自同一套 candidate score，缺少独立 verifier。这会导致：
- 错的 depth target → 但又有偏高 confidence → "平滑但自洽的错误"

### 3.2 具体优化路径

#### Path A：加强 Verifier Backend（推荐）

**目标**：引入独立 verifier，与 target 构造解耦。

**方案**：
1. 保持当前 candidate competition + softmax fusion 的 target 构造方式
2. 新增一个 **独立 verifier 模块**：
   - 输入：pseudo render result + real reference views
   - 输出：independent confidence map（基于几何一致性 / mutual-NN）
   - 不依赖 candidate score

3. 最终 confidence = `smooth_target_confidence × hard_verifier_mask`

**优点**：
- 保留 smooth target 的优点（更稳定的 depth fusion）
- 加入 BRPO-style hard verifier 的可靠性
- Confidence 和 target 来源解耦

#### Path B：改进 `I_t^{fix}` Proxy

**目标**：改进 pseudo-frame 的生成质量。

**当前问题**：
- `I_t^{fix}` 使用 `projected_depth_left/right` 作为 depth proxy
- Projected depth backend 精度有限，导致 depth target 质量差

**方案**：
1. 引入 multi-view depth refinement（类似于 BRPO 的 diffusion restoration）
2. 或者用更好的 depth estimation model 替代 projected depth

#### Path C：改进 Depth Target Backend

**目标**：改进 depth target 的构造方式。

**当前问题**：
- `exact_brpo_full_target_v1` 的 depth target 使用 shared-`C_m` depth loss
- 在当前 `I_t^{fix}` / projected-depth backend 下，纯 exact BRPO target-side 也未赢过 old A1

**方案**：
1. 不再用 `projected_depth` 作为唯一 depth source
2. 引入 render_prior + multi-source fusion 的更稳健 depth target

### 3.3 G-BRPO (B3) 本身的优化

当前 G-BRPO 的 PSNR 改善较小（+0.0114），可能原因：

| 原因 | 证据 | 下一步 |
|------|------|--------|
| Drop rate 太保守 | `drop_rate_global=0.05`，实际 sampled_keep_ratio ≈ 0.97 | 测试更 aggressive drop rate（0.10-0.20）|
| Score 分数太集中 | `corr(state, participation)=0.83`，candidate 重合度高 | 在 C0 层拉开分数（修改 score formula）|
| Probe render 开销大 | 每个 iter 做 probe + formal render，相当于两倍开销 | 优化 probe 精度（用更少 Gaussians？）|

## 4. 优先级排序

根据当前状态，下一步优先级：

1. **A1 Verifier Backend（最高优先级）**：引入独立 verifier，解耦 confidence 与 target
2. **A1 Depth Target Backend**：改进 depth target 质量（不再依赖 projected_depth）
3. **G-BRPO 超参调优**：测试更 aggressive drop rate + score拉开

## 5. 结论

G-BRPO 已完成 BRPO 论文完整对齐，PSNR 有轻微改善（+0.0114）。但当前整体 pipeline 的瓶颈不在 G-BRPO（B3），而在 **A1 target-side contract**：
- Confidence 与 target 同源
- 缺少独立 verifier
- Projected-depth backend 精度有限

下一步应优先加强 **A1 Verifier Backend**，再考虑 G-BRPO 超参调优。
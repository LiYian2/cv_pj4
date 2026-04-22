# G_BRPO_CLEAN_COMPARE_20260421.md

> 更新时间：2026-04-21 16:50 (Asia/Shanghai)
> 角色：G~ clean compare 的长文证据记录；STATUS/DESIGN/CHANGELOG 只保留结论。

---

## 1. 背景

这轮不是继续扩 G~，而是先把已有 compare 清干净，回答两个问题：
1. 之前的 `+0.0114 PSNR` 是否来自干净 control？
2. 在去掉 compare 污染后，G~ direct BRPO 还能否保持正向？

对应三步：
- 修 clean compare
- 跑最小三臂 compare
- 用 clean compare 决定 G~ 的定位

---

## 2. 本轮代码清理内容

### 2.1 summary_only control 修正
- `spgm_action_semantics` 默认值从会覆盖 legacy manager 的隐式 action，改为 `inherit_manager_mode`
- clean baseline 显式指定：`manager_mode=summary_only` + `action_semantics=summary_only`
- `summary_only` 不再偷偷走 legacy grad-mask / state action

### 2.2 direct BRPO path 去混杂
- direct BRPO path 现在走 `neutral_passthrough` update-policy
- `grad_keep_ratio_xyz = 1.0`
- `grad_weight_mean_xyz = 1.0`
- `spgm_weight_mean = 1.0`

也就是说，direct BRPO compare 现在主要测的是：
- `population_active`
- `brpo_unified_v1`
- `stochastic_bernoulli_opacity`
- `current_step_probe_loss`

而不是再混 legacy B2 grad-weight policy。

### 2.3 current-step history 修正
- current-step mode 不再 probe append 一次、post-backward 再 append 一次
- 每个 iter 只保留一条 gating history
- `timing_mode_effective` 在 current-step arm 中稳定为 `current_step_probe_loss`

---

## 3. 最小 clean compare 配置

### 3.1 Run root
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260421_g_brpo_clean_compare_v1`

### 3.2 固定输入
- PLY: `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_gated_rgb0192_80/refined_gaussians.ply`
- Pseudo cache: `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/pseudo_cache_baseline`
- Signal v2 root: `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/signal_v2`
- Internal replay root: `/data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache`
- Protocol: StageB 20 iters, `joint_topology_mode=brpo_joint_v1`

### 3.3 三个 arms
1. `baseline_summary_only`
   - true summary_only control
   - no grad-mask
   - no state action
2. `legacy_delayed_opacity`
   - old delayed deterministic opacity path
   - 仍保留 legacy dense_keep / grad-weight 语义
3. `direct_brpo_current_step`
   - `population_active`
   - `brpo_unified_v1`
   - `stochastic_bernoulli_opacity`
   - `current_step_probe_loss`
   - neutral grad policy

---

## 4. Clean compare 结果

| arm | replay PSNR | SSIM | LPIPS | vs clean baseline |
|---|---:|---:|---:|---:|
| baseline_summary_only | 24.021169 | 0.873495 | 0.080122 | 0.000000 |
| legacy_delayed_opacity | 24.004704 | 0.873153 | 0.080177 | -0.016466 |
| direct_brpo_current_step | 24.026548 | 0.873549 | 0.080146 | +0.005379 |

---

## 5. 机制核对

### 5.1 baseline_summary_only
- `spgm_manager_mode_effective = summary_only`
- `grad_weight_mean_xyz = 1.0 -> 1.0`
- `spgm_weight_mean = 1.0 -> 1.0`
- `spgm_state_participation_ratio = 1.0 -> 1.0`

结论：它现在是干净的 no-action control。

### 5.2 direct_brpo_current_step
- `timing_mode_effective = current_step_probe_loss`
- `spgm_manager_mode_effective = stochastic_bernoulli_opacity`
- `spgm_control_universe = population_active`
- `spgm_score_semantics = brpo_unified_v1`
- `grad_weight_mean_xyz = 1.0 -> 1.0`
- `spgm_weight_mean = 1.0 -> 1.0`
- `spgm_state_participation_ratio ≈ 0.9754 -> 0.9727`
- `spgm_drop_prob_mean ≈ 0.0258 -> 0.0275`

结论：这条 arm 现在可以被解释为“较干净的 direct BRPO action/timing compare”。

### 5.3 legacy_delayed_opacity
- `spgm_manager_mode_effective = deterministic_opacity_participation`
- `grad_weight_mean_xyz ≈ 0.298 -> 0.293`
- `spgm_weight_mean ≈ 0.359 -> 0.356`

结论：legacy delayed opacity 仍然混着旧 grad-weight 语义，所以它更适合作为历史 control，而不是 clean direct-BRPO 对照基线。

---

## 6. 对旧结论的修正

之前 handoff 里写的：
- baseline `24.007`
- direct current-step `24.019`
- gain `+0.0114`

这轮 clean compare 说明：那个 baseline 不是干净的 summary-only control，导致之前的增益被高估。

修正后应使用的新口径是：
- clean baseline `24.021169`
- clean direct BRPO current-step `24.026548`
- clean gain `+0.005379`

因此：
- **G~ direct BRPO 仍是小幅正向**
- 但它不是此前看上去的 `+0.0114` 级别收益
- 它更像一个“语义对齐后的小幅增益侧模块”，不是当前主瓶颈突破口

---

## 7. 结论

1. G~ clean compare 现在已经成立，旧 compare 污染点已清掉：
   - summary_only control 干净
   - direct BRPO 去掉 legacy grad-weight 混杂
   - current-step history 一 iter 一条记录
2. direct BRPO current-step 在 clean compare 下仍然正向，但只有 `+0.00538 PSNR`
3. legacy delayed opacity 在 clean compare 中明确负向，不应继续作为 landing 路线
4. 因此 G~ 的当前定位应是：
   - **语义对齐已完成**
   - **收益有限，可冻结为次级模块 / side branch**
   - **主瓶颈仍在 T~ upstream backend，而不是 G~**

---

## 8. 下一步

推荐顺序：
1. 冻结本轮 G~ clean compare 口径
2. 不再优先做 G~ aggressive 调参 / O2a/b 扩展
3. 正式转入 T~ direct BRPO upstream 路线：
   - branch-native verifier input
   - exact verifier backend
   - exact projected-depth / target field
   - exact consumer/loss contract

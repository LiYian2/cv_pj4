# PIPELINE.md

> Purpose: a compact source-of-truth for drawing the current Part3 BRPO system.
> Scope: all pipeline branches as of 2026-04-19, including current mainline, alternate branches, and deprecated/landed branches.

## 1. One-sentence view

当前默认候选主线：**old A1 + new T1**，即 joint_confidence_v2 + joint_depth_v2 observation + brpo_joint_v1 topology + repair A dense_keep + bounded StageB。

主数据流：part2 S3PO full rerun - internal_eval_cache - internal prepare (select / Difix / fusion / verify / pack) - signal branch - refine - replay eval

---

## 2. 全链路总览图

```
+---------------------------------------------------------------------+
|               Part3 BRPO 全链路总览 (2026-04-19)                     |
+---------------------------------------------------------------------+

                       [ Dataset split ]
                              |
                              v
                  [ Part2 S3PO full rerun ]
                              |
                              v
              [ internal_eval_cache: before/after_opt PLY ]
                              |
                              v
+----------------------------------------------------------+
|                  Internal Prepare                        |
|  select -> Difix(left/right) -> fusion -> verify -> pack |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                    Signal Branch                         |
|                                                          |
|  [ Legacy ]      [ signal_v2 ]     [ A1 joint_v2 ]       |
|  depth mask      RGB-only         joint_confidence_v2    |
|  [废弃]         [对照]           [当前主线old A1]        |
|                                                          |
|                  [ new A1 joint_v1 ]                     |
|                  pseudo_*_joint_v1                       |
|                  [已实现暂不landing]                     |
|                                                          |
|                  [ A2 geometry expand ]                  |
|                  [废弃 widening失败]                     |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                    Refine Stage                          |
|                                                          |
|  [ StageA ] pose/exposure only, 不更新Gaussian [闭环]   |
|       |                                                  |
|       v                                                  |
|  [ StageA.5 ] local micro [降级为optional warmup]       |
|       |                                                  |
|       v                                                  |
|  [ StageB ]                                              |
|    |-- old topology (两阶段串联) [对照]                  |
|    |-- new topology (T1: brpo_joint_v1) [当前拓扑主线]   |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|            Gradient Management / SPGM                    |
|                                                          |
|  [ No extra manage ] baseline [对照]                    |
|  [ Local gating ] view-conditioned gate [可用]          |
|  [ SPGM repair A ] dense_keep [当前anchor]              |
|  [ selector-first ] support_blend far=0.90 [near-parity]|
|  [ B1/B2 shell ] manager shell + diagnostics [已接通]   |
|  [ B3旧版 ] xyz_lr_scale [降级为probe]                  |
|  [ B3新版 ] deterministic_participation [已接入weak负]  |
+----------------------------------------------------------+
                              |
                              v
              [ Replay Eval: PSNR/SSIM/LPIPS ]
```

---

## 3. 上游信号链路详解

### 3.1 Link-1: Legacy depth-first branch [废弃]
状态: 废弃，仅作历史对照

方法: depth 做 mask 和 depth target，RGB supervision 来自 propagated train-mask semantics

废弃原因: 被 signal_v2 取代，depth verified 过窄(约2%)

---

### 3.2 Link-2: signal_v2 RGB-only branch [对照]
状态: 对照臂，不再作为主线

方法:
- raw_rgb_confidence_v2: RGB correspondence confidence
- target_depth_for_refine_v2_brpo: depth target(sidecar)
- depth verified coverage 约 2%，过窄

角色: canonical control arm，用于 compare

---

### 3.3 Link-3: old A1 (joint_confidence_v2) [当前observation主线]
状态: 当前 observation 主线

方法: unified RGB-D joint support filter
- joint_confidence_v2 = min(raw_rgb_confidence, geometry_tier)
- joint_depth_target_v2 = target_depth_for_refine_v2_brpo(直接复用)
- mask coverage 约 1.96%

设计定位: joint support filter / unified consumer semantics

关键发现: A1 是 Re10k 上第一条明确优于 sidecar control 的 observation 正信号

---

### 3.4 Link-4: new A1 (brpo_joint_v1) [已实现暂不landing]
状态: builder + consumer lock 已完成，但首轮 compare 不占优

方法: BRPO-style joint observation rewrite
- pseudo_depth_target_joint_v1: 由 joint candidate competition/fusion 重新生成
- pseudo_confidence_joint_v1: joint builder 内部定义
- pseudo_source_map_joint_v1 / pseudo_uncertainty_joint_v1

与 old A1 的本质差异:
- old A1: 统一 support 语义，depth target 直接复用旧 target
- new A1: 统一 observation object，depth target 重新生成

首轮 compare 结果: new A1 + new T1 = 24.135512 vs old A1 + new T1 = 24.185846
结论: old A1 仍优于 new A1

---

### 3.5 Link-5: A2 geometry-constrained expand [废弃]
状态: widening 方案失败，不作为主线

方法: support_expand.py，双侧 geometry-supported expansion

compare 结果: A1+A2 相比 A1 下降 -0.286 PSNR(coverage 从 1.96% 扩到 6.05%)

废弃原因: 扩张策略带来过多低质量像素，导致负收益

---

## 4. Refine Stage 链路详解

### 4.1 StageA [闭环可用]
职责: pose/exposure only，不更新 Gaussian

关键参数: lambda_abs_t = 3.0 / lambda_abs_r = 0.1，pose residual fold-back 已补回

设计判断: StageA-only replay 不是判优指标

---

### 4.2 StageA.5 [已降级]
状态: 降级为 optional warmup / control，不再是主线必经阶段

旧职责: local Gaussian micro-tune，pseudo-only backward

降级原因: T1 joint topology 已证明 pseudo + real 在同一 loop 内更稳定

当前角色: orchestration anchor / 对照臂

---

### 4.3 StageB old topology [对照]
状态: 对照臂，不再作为主线

结构: StageA - StageA.5 - StageB 两阶段串联

问题: 弱 pseudo signal 在 StageA.5 难以积累，StageB 继承后仍易被稀释

---

### 4.4 StageB new topology (T1: brpo_joint_v1) [当前拓扑主线]
状态: 当前 topology 主线

方法: joint loop，pseudo + real 在同一 iteration 内共同作用
- joint_topology_mode=brpo_joint_v1
- 每轮: sample real/pseudo - forward - apply gating/SPGM - assemble joint loss - backward once - step once

compare 结果: new topology = 24.149837 vs old topology = 24.116956 (+0.032881 PSNR)

关键发现: T1 的 topology 收益是跨 observation 稳定存在的

当前最强候选: old A1 + new T1 = 24.185846

---

## 5. Gradient Management / SPGM 链路详解

### 5.1 No extra Gaussian management [对照]
方法: plain refine baseline

---

### 5.2 Local gating [可用]
方法: view-conditioned gate，pseudo-view signal quality 决定哪些 pseudo views 应贡献

modes: off / hard_visible_union_signal / soft_visible_union_signal

角色: pseudo branch scope controller

---

### 5.3 SPGM repair A (dense_keep) [当前anchor]
状态: 当前最稳 SPGM policy

参数: keep = (1,1,1) / eta = 0 / weight_floor = 0.25

方法: keep the active set, weaken over-suppression via soft weighting

---

### 5.4 SPGM selector-first [near-parity]
状态: near-parity 参考臂，不升级为 anchor

方法: selector_quantile + ranking_mode=support_blend + far_keep 约 0.90

问题: ranking 质量不足导致误删有用 Gaussian，伤 replay

保留原因: 极窄窗口(0.90~0.92)接近 parity，待 ranking 改进后可再探索

---

### 5.5 B1/B2 [已接通]
状态: manager shell + diagnostics 已接通，不直接改 Gaussian state

B1: SPGM 结构拆成 update policy + state management 两层
- 新增 manager.py
- history 字段: spgm_manager_mode_effective / spgm_state_*

B2: scene-aware density proxy + decoupled scores
- ranking_score: selector ordering
- weight_score: soft weighting
- state_score: manager lr scale / opacity attenuation

---

### 5.6 B3 旧版 (xyz_lr_scale) [降级为diagnostic probe]
状态: 降级，不再当主线

方法: post-backward deterministic grad scaling
- 对 _xyz.grad 再乘一层 state-dependent scale
- action 发生在 backward 之后，不改变 render participation

问题: 方法对象没换，仍是 post-backward grad modulator，不是 population controller

compare 结果: 40iter compare 为 weak-negative

---

### 5.7 B3 新版 (deterministic_participation) [已接入首轮weak-negative]
状态: 已接入 pseudo render 前参与控制，但首轮 compare weak-negative

方法: pre-render participation control
- manager_mode=deterministic_participation
- 统计 low-score candidate subset
- 生成 participation_render_mask 供下一轮 pseudo render 消费
- action 先于 pseudo render，不是 backward 后救火

首轮参数: near=1.0 / mid=0.9 / far=0.75

compare 结果: b3_det_participation = 24.182511 vs summary_only = 24.185744 (-0.00323 PSNR)

当前判断: 方法对象已正确切对，但 action 强度需收缩

下一步: 更保守的 participation keep

---

## 6. 当前推荐主线配置

### 6.1 固定参照线
RGB-only v2 + gated_rgb0192 + post40_lr03_120

### 6.2 当前默认候选主线
old A1 (joint_confidence_v2 + joint_depth_v2)
+ new T1 (joint_topology_mode=brpo_joint_v1)
+ repair A dense_keep
+ bounded StageB (post40_lr03_120)

Re10k 最佳结果: PSNR 24.185846 / SSIM 0.875423 / LPIPS 0.080379

---

## 7. 废弃/降级链路汇总

| 链路 | 状态 | 废弃/降级原因 |
|------|------|---------------|
| Legacy depth-first | 废弃 | 被 signal_v2 取代 |
| full v2 depth branch | 废弃 | verified 约 2%，过窄 |
| A2 geometry expand | 废弃 | widening 策略失败，-0.286 PSNR |
| StageA.5 as mainline | 降级 | T1 joint topology 更稳 |
| old topology | 对照 | new topology 有稳定正增益 |
| new A1 (joint v1) | 暂不landing | old A1 + new T1 更优 |
| B3 xyz_lr_scale | 降级 | diagnostic probe，weak-negative |
| selector-first | near-parity | 极窄窗口接近 parity，不升级 |

---

## 8. 数据集 case 状态

| 数据集 | 当前状态 | 当前主线 |
|--------|----------|----------|
| Re10k-1 | A1/T1 收敛，B3 第一版 compare 已完成 | old A1 + new T1 |
| DL3DV-2 | canonical baseline + repair A 已打通 | 暂未平移新 topology |

---

## 9. 实验历史关键节点

| 日期 | 里程碑 | 结果 |
|------|--------|------|
| 2026-04-16 | P2-J bounded StageB schedule | post40_lr03_120 最佳 |
| 2026-04-17 | P2-S support_blend far-keep | far=0.90 接近 parity |
| 2026-04-17 | A1 joint confidence compare | A1 优于 control |
| 2026-04-17 | A1+A2 geometry expand | widening 失败 |
| 2026-04-18 | B1/B2 manager shell | 已接通 |
| 2026-04-18 | T1 topology compare | new topology +0.033 PSNR |
| 2026-04-18 | T1-R3 2x2 factor experiment | old A1 + new T1 最优 |
| 2026-04-19 | B3-R1 deterministic participation | 已接入，weak-negative |

---

## 10. 参考文档

- 当前状态: docs/STATUS.md
- 设计原则: docs/DESIGN.md
- 过程记录: docs/CHANGELOG.md
- 总规划: docs/BRPO_alignment_unified_RGBD_and_scene_SPGM_plan.md
- A1 计划: docs/A1_unified_rgbd_joint_confidence_engineering_plan.md
- T1 计划: docs/T1_brpo_joint_optimization_topology_engineering_plan.md
- B3 计划: docs/B3_deterministic_state_management_engineering_plan.md
- archived: docs/archived/
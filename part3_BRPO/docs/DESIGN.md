# DESIGN.md - Part3 BRPO 设计文档

> **书写规范**：
> 1. 只记录"设计原则、架构决策、接口定义"，不记录实验数据（数值在 STATUS 或实验文档）
> 2. 覆盖式更新，直接修改对应版块，不追加
> 3. 设计判断用一句话固化，不展开过程
> 4. 引用格式：`[参见 STATUS §X]` 或 `[参见 archived/P2X_*.md]`
> 5. 更新后修改文档顶部时间戳

---

## 1. 系统边界与当前主线

### 1.1 当前主线

**RGB-only v2 + gated_rgb0192 + post40_lr03_120** 是 canonical StageB baseline。

### 1.2 当前阶段定义

| 阶段 | 设计角色 | 说明 |
|------|----------|------|
| StageA | pose/exposure 调整 | 不更新 Gaussian，replay 不是判优指标 |
| StageA.5 | local Gaussian 微调 | xyz 或 xyz+opacity，pseudo-only backward |
| StageB | joint refine | bounded schedule，real anchor + pseudo RGBD |
| SPGM repair A | 当前 anchor | 最稳配置，待 selector-first 替代 |

---

## 2. 设计判断（固化）

### 2.1 StageA 性质

- **只更新 pseudo pose/exposure，不更新 Gaussian**
- replay-on-PLY 在 StageA-only compare 中只是 sanity check
- 真正依赖 replay 的 compare 必须移到 StageA.5/StageB

### 2.2 StageB 性质

- 短程（40iter）有效，长程需 bounded schedule
- post40_lr03_120 是当前最佳 bounded baseline
- 不继续 schedule sweep

### 2.3 Signal v2 性质

- RGB-only v2：窄但干净的 fused-RGB correspondence mask，可保留
- Full v2 depth：过窄（verified≈2%），不直接接管 full depth
- 若需扩张，优先几何约束的 support/depth expand

### 2.4 SPGM 性质

- 当前更像 suppressor，不是 selector
- repair A 是最稳 anchor（suppress 最温和）
- selector-first 在 far≈0.90 已出现可行区间

### 2.5 Abs prior 性质

- 固定背景：λ_abs_t=3.0, λ_abs_r=0.1
- drift 抑制有效，但 depth 改善不明显
- 不继续大范围扫描

---

## 3. Pipeline 架构

```text
part2 internal cache
    ↓
internal prepare: select → difix → fusion → verify → pack
    ↓
pseudo_cache + signal_v2
    ↓
StageA (pose/exposure only)
    ↓
StageA.5 (local Gaussian + gating)
    ↓
StageB (joint refine + bounded schedule)
    ↓
refined_gaussians.ply + replay eval
```

---

## 4. 接口定义

### 4.1 Pseudo cache schema

核心产物：
- `target_rgb_fused.png`：fusion 输出
- `train_confidence_mask_brpo_*.npy`：训练 mask
- `target_depth_for_refine.npy`：M3 blended target
- `target_depth_for_refine_v2.npy`：M5 densified target
- `signal_v2/*`：BRPO-style 隔离路径

### 4.2 Refine consumer 接口

关键 CLI：
- `signal_pipeline`: legacy / brpo_v2
- `stageA_target_depth_mode`: blended_depth_m5
- `stageA_depth_loss_mode`: source_aware
- `pseudo_local_gating`: off / hard_visible_union_signal / spgm_keep / spgm_soft

### 4.3 SPGM 接口

关键 CLI：
- `spgm_policy_mode`: dense_keep / selector_quantile
- `spgm_ranking_mode`: v1 / support_blend
- `spgm_cluster_keep`: (near, mid, far)
- `spgm_weight_floor / spgm_support_eta`

---

## 5. 设计原则（长期有效）

### 5.1 Stage 顺序语义

- StageA/A.5/B 必须顺序 handoff（相机状态传递）
- 每阶段必须消费前一阶段导出的 pseudo_camera_states_final.json

### 5.2 Replay 评估口径

- Replay 必须消费真实变化的 artifact（PLY / camera states）
- StageA-only replay 无信息量，不能作为判优指标

### 5.3 Supervision scope 与 optimization scope 匹配

- 弱 supervision 不应驱动全局 Gaussian 更新
- Local gating / SPGM 用于限制 optimization scope

### 5.4 实验验收阶梯

1. Smoke：wiring 通，不是 no-op
2. Short compare：有可观察差异
3. Formal compare：apples-to-apples，protocol 对齐
4. Replay：下游指标变化

---

## 6. 当前不做的事

- 继续无边界 StageB schedule sweep
- 继续增加 selector keep ratio（已证明伤 replay）
- 直接推进 stochastic drop
- 对 raw RGB mask 直接 densify
- 继续在 StageA-only 上用 replay 判优

---

## 7. 参考

- 当前状态：[STATUS.md]
- 过程记录：[CHANGELOG.md]
- 实验细节：[archived/2026-04-experiments/P2X_*.md]
- SPGM 落地计划：[SPGM_landing_plan_for_part3_BRPO.md]

---

## 8. 已废弃链路（保留历史决策）

| 链路 | 废弃原因 |
|------|----------|
| M3 depth target v1 | 被 v2 densified target 取代 |
| midpoint selection | 被 signal-aware-8 取代 |
| legacy StageA-only replay | 确认不是判优指标 |
| StageB 300iter | 确认长程退化，改用 bounded schedule |
| SPGM v1 raw-120 | protocol 与 baseline 漂离 |
# STATUS.md - Part3 Stage1 当前状态

> **书写规范**：
> 1. 只记录"现在"，不记录历史过程（过程在 CHANGELOG）
> 2. 覆盖式更新，直接修改对应版块，不追加
> 3. 不写实验数据（数值在 DESIGN 或实验文档），只写结构/状态/判断
> 4. 状态用 ✅ ⚠️ ❌ 标记，"已完成"不重复列出
> 5. 更新后修改文档顶部时间戳

---

## 1. 概览

当前主线（Re10k 参考线）：**RGB-only v2 + gated_rgb0192 + post40_lr03_120** 为 canonical StageB baseline。

一句话判断：Re10k 上 repair A 仍是当前最稳 SPGM anchor，但 selector-first 可行区间已出现在 `far≈0.90` 的极保守一侧；DL3DV 上 canonical chain 已推进到 Phase G，首轮结果是 bounded StageB baseline 略优于 repair A，selector-first 还未开始。

### 1.1 数据集 case 状态

| 数据集 | 当前状态 | 当前判断 |
|------|------|------|
| Re10k-1 | ✅ 主参考线完整 | repair A 稳定；selector-first 进入 `far≈0.90` confirmation 区 |
| DL3DV-2 | ✅ 已完成 Part2→Prepare→signal_v2→StageA/A.5/StageB baseline/SPGM repair A | 首轮 bounded baseline 略优于 repair A；先不直接开 selector-first |

---

## 2. 数据结构

### 2.1 Internal cache（源数据）

```text
<run_root>/internal_eval_cache/
├── manifest.json
├── camera_states.json
├── before_opt/
│   ├── point_cloud/point_cloud.ply
│   ├── render_rgb/ / render_depth_npy/
├── after_opt/
│   ├── point_cloud/point_cloud.ply
│   ├── render_rgb/ / render_depth_npy/
```

### 2.2 Pseudo cache（训练输入）

```text
internal_prepare/<prepare_key>/pseudo_cache/
├── manifest.json
└── samples/<frame_id>/
    ├── render_rgb.png / render_depth.npy
    ├── target_rgb_fused.png
    ├── train_confidence_mask_brpo_*.npy
    ├── target_depth_for_refine.npy
    ├── target_depth_for_refine_v2.npy
    ├── projected_depth_left/right.npy
    └── signal_v2/（独立路径）
```

### 2.3 Signal v2（BRPO-style 隔离路径）

```text
signal_v2/frame_<frame_id>/
├── raw_rgb_confidence_v2.npy
├── target_depth_for_refine_v2_brpo.npy
├── depth_supervision_mask_v2_brpo.npy
└── rgb_mask_meta_v2.json / depth_meta_v2_brpo.json
```

---

## 3. Pipeline 状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| Internal cache 导出 | ✅ | before/after PLY + render cache |
| Same-ply replay | ✅ | 可用于 baseline/refined 比较 |
| Prepare: select→difix→fusion→verify→pack | ✅ | canonical schema 打通 |
| M5 densify target | ✅ | coverage 从 ~1.5% 提到 ~14% |
| StageA: source-aware depth loss | ✅ | 弱有效，闭环已修 |
| StageA: abs prior | ✅ | 固定背景 λ_t=3.0, λ_r=0.1 |
| StageA.5: local gating | ✅ | hard gating + SPGM v1 |
| StageB: bounded baseline | ✅ | post40_lr03_120 |
| SPGM: repair A anchor | ✅ | 当前最稳配置 |
| SPGM: selector-first far=0.90 | ⚠️ | 可行区间已出现，待确认 |

---

## 4. 当前锁定配置

| 类别 | 参数 | 当前值 |
|------|------|--------|
| Abs prior | lambda_abs_t | 3.0 |
| Abs prior | lambda_abs_r | 0.1 |
| StageB baseline | schedule | post40_lr03_120 |
| StageB baseline | upstream gate | gated_rgb0192 |
| StageB baseline | signal pipeline | RGB-only v2 |
| SPGM repair A | keep | (1,1,1) |
| SPGM repair A | eta | 0.0 |
| SPGM repair A | weight_floor | 0.25 |
| SPGM ranking | mode | support_blend |
| SPGM ranking | lambda | 0.5 |
| SPGM selector | far keep | 0.90（候选） |

---

## 5. 当前判断

### 5.1 StageA
- 闭环已修，pose 真实可更新，但 depth 只"弱有效"
- StageA-only replay 不是判优指标（PLY 与输入 hash 一致）

### 5.2 StageB
- 短程（40iter）有效，长程（120iter）需 bounded schedule
- post40_lr03_120 是当前最佳 bounded baseline
- 不再继续 schedule sweep

### 5.3 SPGM
- repair A 是最稳 anchor，但不是最终解
- selector-first 在 far=0.90 已逼近 parity
- 下一步：围绕 0.90 做窄范围确认
- 暂不推进 stochastic、xyz+opacity、更长 iter

### 5.4 Signal v2
- RGB-only v2 可保留，full v2 depth 过窄（verified≈2%）
- 不直接接管 full depth，待后续几何扩张

### 5.5 DL3DV case
- canonical DL3DV 链已完成：internal cache → replay parity → signal-aware selection → canonical prepare root → signal_v2 → StageA / StageA.5 / StageB baseline / SPGM repair A
- 当前 DL3DV 第一轮 refine 已证明链路可跑通，bounded baseline 与 repair A 都已完成统一 replay
- 第一轮结果上，`canonical_baseline_post40_lr03_120` 略优于 `spgm_repair_a_keep111_eta0_wf025`，所以 repair A 还不是 DL3DV 当前 winner
- 当前 DL3DV refine 使用 `signal_pipeline=brpo_v2 + brpo_v2_raw`，但 depth target 仍取 canonical `target_depth_for_refine`；该 case 的 pseudo_cache 目前没有 `target_depth_for_refine_v2`
- 在解释上要把它视为“DL3DV case bring-up 已完成、baseline 与 repair A 已落地”，而不是“SPGM 已在 DL3DV 上站住”

---

## 6. 待办

### 当前进行中 ⚠️
- Re10k-1：固定 support_blend ranking，围绕 far≈0.90 做 selector confirmation / precision sweep
- DL3DV-2：已完成 baseline + repair A 首轮 bring-up，下一步先判断是补一个小的 repair-A sanity/threshold debug，还是直接把 bounded baseline 当作 DL3DV 当前 anchor

### 下一阶段 ⏳
- Re10k-1：若 far≈0.90 仍不能稳定站住，再考虑补 ranking score（更强 support-aware / two-stage）
- DL3DV-2：在 repair A 没有明确站住前，不直接把 Re10k 的 selector-first follow-up 平移过去

### 明确不做 🚫
- 继续 StageB schedule 无边界深调
- 继续增加 selector keep ratio（已证明会伤 replay）
- 直接推进 stochastic drop
- 对 raw RGB mask 直接 densify（应优先几何约束的 support/depth expand）

---

## 7. 最近版本摘要（追溯）

| 时间 | 主要进展 |
|------|----------|
| 2026-04-17 23:59 | DL3DV Phase G 完成，canonical baseline 与 repair A 首轮都已跑通并统一 replay |
| 2026-04-17 20:44 | P2-S 完成，far=0.90 逼近 repair A parity |
| 2026-04-17 16:06 | P2-R 完成，support_blend ranking 有效但 partial |
| 2026-04-17 14:57 | P2-O 完成，selector-first 会伤 replay |
| 2026-04-17 13:26 | P2-N 完成，selector plumbing 接通 |
| 2026-04-17 01:33 | P2-M 完成，conservative repair 部分回升 |
| 2026-04-17 01:11 | P2-L 完成，protocol 对齐后 SPGM 仍负 |

---

## 8. 参考

- 设计原则：[DESIGN.md]
- 过程记录：[CHANGELOG.md]
- 实验细节：[archived/2026-04-experiments/P2X_*.md]
- 落地计划：[SPGM_landing_plan_for_part3_BRPO.md]
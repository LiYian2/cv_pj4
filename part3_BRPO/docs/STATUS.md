# STATUS.md - Part3 Stage1 当前状态

> 更新时间：2026-04-19 05:16 (Asia/Shanghai)

> **书写规范**：
> 1. 只记录"现在"，不记录历史过程（过程在 CHANGELOG）
> 2. 覆盖式更新，直接修改对应版块，不追加
> 3. 不写实验数据（数值在 CHANGELOG / compare json），只写结构/状态/判断
> 4. 状态用 ✅ ⚠️ ❌ 标记，"已完成"不重复列出
> 5. 更新后修改文档顶部时间戳

---

## 1. 概览

当前固定参照线仍是：**RGB-only v2 + gated_rgb0192 + post40_lr03_120**（canonical StageB protocol）。

A/T 主线已经收敛到：
- **old A1** 继续作为当前 observation 主线
- **new T1**（`joint_topology_mode=brpo_joint_v1`）继续作为当前 topology 主线
- **StageA.5** 已降级为 optional warmup / control

B3 已正式开始执行，但状态也已经明确：
- **第一版 deterministic participation controller 已接入主 loop 并完成首轮 formal compare**
- **当前 compare 结论为 weak-negative / no-go**
- **因此 B3 还不能接管默认主线，只能进入更保守的第二轮调整**

一句话判断：**当前 Re10k 默认候选主线仍是 `old A1 + new T1`；B3 第一版已经从“旧 grad scaler”推进成“pre-render participation controller”，但首轮结果略差，暂不 landing。**

### 1.1 数据集 case 状态

| 数据集 | 当前状态 | 当前判断 |
|------|------|------|
| Re10k-1 | ✅ A1/T1 收敛，✅ B3 第一版 compare 已完成 | 主线仍是 `old A1 + new T1`；B3 v1 当前 weak-negative |
| DL3DV-2 | ✅ canonical baseline + repair A 已打通 | 暂未平移新 topology / B3 |

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

### 2.3 Signal v2（当前并存两条 A1 语义）

```text
signal_v2/frame_<frame_id>/
├── raw_rgb_confidence_v2.npy
├── target_depth_for_refine_v2_brpo.npy
├── depth_supervision_mask_v2_brpo.npy
├── joint_confidence_v2.npy
├── joint_confidence_cont_v2.npy
├── joint_depth_target_v2.npy
├── pseudo_depth_target_joint_v1.npy
├── pseudo_confidence_joint_v1.npy
├── pseudo_confidence_rgb_joint_v1.npy
├── pseudo_confidence_depth_joint_v1.npy
├── pseudo_uncertainty_joint_v1.npy
├── pseudo_source_map_joint_v1.npy
├── pseudo_valid_mask_joint_v1.npy
└── rgb_mask_meta_v2.json / depth_meta_v2_brpo.json / joint_meta_v2.json / joint_observation_meta_v1.json
```

---

## 3. Pipeline 状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| Internal cache 导出 | ✅ | before/after PLY + render cache |
| Same-ply replay | ✅ | 可用于 baseline/refined 比较 |
| Prepare: select→difix→fusion→verify→pack | ✅ | canonical schema 打通 |
| StageA: source-aware depth loss | ✅ | 闭环可用 |
| StageA: abs prior | ✅ | 固定背景 λ_t=3.0, λ_r=0.1 |
| StageA.5: local gating micro-tune | ⚠️ | 保留，但已降级为可选 warmup / 对照 |
| StageB: bounded baseline | ✅ | post40_lr03_120 |
| SPGM: repair A anchor | ✅ | 当前最稳 policy 配置 |
| SPGM: B1/B2 | ✅ | manager shell + decoupled score + diagnostics 已接通 |
| B3 旧版：xyz deterministic lr scaling | ❌ | 已降级为 diagnostic probe，不再当主线 |
| B3 新版：deterministic participation controller | ⚠️ | 已接入 pseudo render 前参与控制并完成首轮 compare，但当前 weak-negative |
| A1 旧语义：joint support filter | ✅ | 当前 observation 主线 |
| A1 新语义：joint observation rewrite | ✅ | builder + consumer lock 已完成，但暂不 landing |
| T1：`joint_topology_mode=brpo_joint_v1` | ✅ | topology mode + compare + confirmation 已完成 |
| T1 orchestration：StageA.5 optional warmup/control | ✅ | 角色已经固化 |
| A2 geometry-constrained expand | ❌ | 当前 widening 方案不作为主线 |
| B3 stochastic masking | ❌ | 暂未启动；需等待 deterministic participation 更稳的版本 |

---

## 4. 当前锁定配置

| 类别 | 参数 | 当前值 |
|------|------|--------|
| Abs prior | lambda_abs_t / lambda_abs_r | 3.0 / 0.1 |
| StageB baseline | schedule | post40_lr03_120 |
| StageB baseline | upstream gate | gated_rgb0192 |
| StageB baseline | signal pipeline | RGB-only v2 + depth-sidecar（control） |
| A1 参考主线 | observation semantics | `joint_confidence_v2 + joint_depth_v2` |
| T1 主线 | topology | `joint_topology_mode=brpo_joint_v1` |
| 当前默认候选主线 | observation + topology | `old A1 + new T1` |
| StageA.5 role | orchestration | optional warmup / control |
| SPGM repair A | keep / eta / weight_floor | (1,1,1) / 0.0 / 0.25 |
| B3 v1 | manager mode | `deterministic_participation` |
| B3 v1 | first tested keep | near=1.0 / mid=0.9 / far=0.75 |

---

## 5. 当前判断

### 5.1 StageA
- 闭环已修，pose 可更新；StageA-only replay 不是判优指标。

### 5.2 StageB / T1
- `post40_lr03_120` 仍是固定 protocol。
- `joint_topology_mode=brpo_joint_v1` 仍是当前稳定 topology 主线。

### 5.3 StageA.5 角色
- StageA.5 不再是默认主线必经阶段。
- 当前工程定位仍是：**可选 warmup / 对照 / orchestration anchor**。

### 5.4 Signal v2 / A1 / T1
- new A1 已证明不是旧 A1 换壳，但当前 observation 主线仍保留 old A1。
- 当前默认候选主线仍是：**`old A1 + new T1`**。

### 5.5 SPGM / B3
- B3 第一版已经真正把动作位置从 **post-backward grad scaling** 推到 **pre-render participation control**。
- 这说明方法对象已经换了，不再只是旧 B3 的 mild grad probe。
- 但首轮 deterministic participation compare 是 **weak-negative**，所以当前结论不是 landing，而是：**需要更保守的 participation schedule / candidate-action 设计再做下一轮。**

### 5.6 DL3DV case
- 暂维持 canonical baseline + repair A 现状；不在 Re10k 的 B3 还未稳定前贸然平移。

---

## 6. 待办

### 当前进行中 ⚠️
- 基于 `old A1 + new T1` 主线，设计并执行 **更保守的 B3 deterministic participation** 第二轮 compare。

### 下一阶段 ⏳
- 优先尝试更保守 keep（尤其 far / mid），避免第一版 drop 过重。
- 如果 deterministic participation 能转正，再讨论 stochastic masking。

### 明确不做 🚫
- 不把当前 `deterministic_participation` v1 直接推进为默认主线。
- 不回到旧 B3 `xyz_lr_scale` 当主线。
- 不在 deterministic participation 还未站稳前直接跳 stochastic masking。
- 不重新打开 A1/T1 主线归属争论。

---

## 7. 参考

- 设计原则：[DESIGN.md]
- 过程记录：[CHANGELOG.md]
- 总规划：[BRPO_alignment_unified_RGBD_and_scene_SPGM_plan.md]
- B3 重写计划：[B3_deterministic_state_management_engineering_plan.md]

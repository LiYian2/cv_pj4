# DESIGN.md - Part3 BRPO 设计文档

> 更新时间：2026-04-20 16:35 (Asia/Shanghai)

> **书写规范**：
> 1. 只记录"设计原则、架构决策、接口定义"，不记录实验数据（数值在 CHANGELOG 或 compare json）
> 2. 覆盖式更新，直接修改对应版块，不追加
> 3. 设计判断用一句话固化，不展开过程
> 4. 引用格式：`[参见 STATUS §X]` 或 `[参见 archived/P2X_*.md]`
> 5. 更新后修改文档顶部时间戳

---

## 1. 系统边界与当前主线

### 1.1 当前主线

固定参照线仍是 **RGB-only v2 + gated_rgb0192 + post40_lr03_120**。

当前主线设计已经收敛到更稳的版本：
1. **old A1**（`joint_confidence_v2 + joint_depth_v2`）继续作为当前 observation 主线；
2. **T1**（`joint_topology_mode=brpo_joint_v1`）继续作为当前 topology 主线；
3. **A1 verify proxy** 已完成 negative proof，不再是当前 A1 主执行方向；
4. **A1 BRPO-style v1 / v2** 已完成 compare，但都应被视为过渡 probe，而不是当前默认候选；
5. **A1 direct BRPO v1**（`brpo_direct_v1`）已成为当前最值得继续的 A1 研究线，但还不构成新的默认主线；
6. **B3** 已完成 boolean `deterministic_participation` 与 delayed `deterministic_opacity_participation` 两条 C1 前置路径，但两者都还不构成新的默认主线；
7. 因此当前默认候选主线仍是 **old A1 + new T1**。 [参见 STATUS §5.4]

### 1.2 当前阶段定义

| 阶段 | 设计角色 | 说明 |
|------|----------|------|
| StageA | pose/exposure 调整 | 不更新 Gaussian，replay 不是判优指标 |
| StageA.5 | optional warmup / control | 可保留，但不再是默认主线必经阶段 |
| StageB new topology | `brpo_joint_v1` joint loop | 当前 topology 主线 |
| SPGM repair A | 当前 policy anchor | B3 未稳定前继续作为主对照 |
| B3 deterministic participation | first real boolean population-control actor | 已接入，但当前实现尚未站稳 |
| B3 deterministic opacity participation | current action-law branch | O0/O1 + delayed C1 已完成，但当前仍未 landing |
| A1 direct BRPO v1 | current A1 research branch | 已接线并完成 compare，是当前最强的新 A1 分支 |

---

## 2. 设计判断（固化）

### 2.1 StageA 性质
- **只更新 pseudo pose/exposure，不更新 Gaussian**。
- replay-on-PLY 在 StageA-only compare 中只是 sanity check。

### 2.2 StageB / T1 性质
- `post40_lr03_120` 继续作为固定 protocol。
- `joint_topology_mode=brpo_joint_v1` 的关键是：**同一 iteration 内 assemble joint loss、backward once、再让 local gating / SPGM 扮演 pseudo scope controller**。
- 当前 T1 已经是稳定成立的 topology rewrite，不再是待验证 patch。

### 2.3 StageA.5 性质
- StageA.5 的设计定位已固定成：**optional warmup / control / orchestration anchor**。
- 它不再定义主线。

### 2.4 A1 六种语义的当前定位
- **old A1** 的正确定位是：`joint support filter / unified consumer semantics`。
- **current new A1**（`brpo_joint_v1`）的正确定位是：`candidate-built joint observation object`。
- **verify proxy**（`brpo_verify_v1`）的正确定位是：`只改 confidence 来源的失败 probe`。
- **BRPO-style v1**（`brpo_style_v1`）的正确定位是：`shared C_m + verified projected depth target 的第一版工程 builder`。
- **BRPO-style v2**（`brpo_style_v2`）的正确定位是：`continuous verifier quality + stable target blend 的过渡版 builder`。
- **direct BRPO v1**（`brpo_direct_v1`）的正确定位是：`直接用 fused-frame reciprocal correspondence valid set 与 overlap-confidence weighted projected depth 逼近 BRPO A1 的当前主 probe`。

### 2.5 当前 direct BRPO v1 不是完整 BRPO A1
- 当前 `brpo_direct_v1` 已经抓住了更直接的 BRPO A1 主干：**shared `C_m`、同源 verifier set、target/confidence 共用同一 direct support contract**。
- 但它还不是完整 BRPO implementation，至少还有三处差异：
  1. pseudo-frame 当前仍用 `target_rgb_fused.png` 作为工程代理；
  2. `valid_left / valid_right` 当前仍依赖现有 matcher support + overlap validity，而不是更独立的 BRPO-style direct verifier backend；
  3. `depth_target_brpo_direct_v1` 当前是 direct projected-depth composition，没有接住 old A1 / 更完整 BRPO builder 的稳定化与 fallback contract。

### 2.6 当前 A1 的下一步设计判断
- A1 不应回头继续打磨 `verify proxy v1`。
- A1 也不应继续做 conf-only / depth-only 单侧 toggle 来猜剩余 gap。
- 更合理的下一步是：**先把 `old A1` 与 `brpo_direct_v1` 的 residual gap 诊断清楚，再决定 direct verifier backend 或 direct target builder 的下一小步 patch。**

### 2.7 为什么 direct BRPO v1 仍差于 old A1
- 当前差距已经不是“方向错了”，而是“direct builder 还没把 old A1 的稳定 contract 接住”。
- 已有事实是：
  1. `brpo_direct_v1` 与 `brpo_style_v2` 的平均 valid coverage 基本相同；
  2. `brpo_direct_v1` 的正样本 confidence 均值已接近 old A1；
  3. 但 old A1 仍然更好。
- 因此当前设计判断应是：**剩余 gap 更可能在 target builder / exact valid set / fallback contract，而不是简单的 confidence 强弱。**

### 2.8 SPGM / B3 性质
- 旧 B3 的正确定位是 **post-backward deterministic grad/state probe**，不能再误称为 population manager。
- 新 B3 第一版已经真正改成：**pre-render deterministic participation controller**。
- 本轮又把动作变量从 boolean selector 推到 **delayed deterministic opacity participation**，即 pseudo render 前的 continuous opacity attenuation。
- 这意味着 current B3 已经不仅在改“谁参与”，也开始改“参与到什么程度”，方法对象比 boolean selector 更接近 BRPO-style participation manager。
- 但 delayed opacity C1 compare 仍轻微弱负，因此当前设计判断应是：**action variable 已更对，但 current score/candidate/action law 还不够成熟，不能直接推进到 O2a/b。**

### 2.9 当前 B3 调整方向
- 下一轮优先方向不是 stochastic，也不是继续回退到 xyz grad scale。
- 下一轮也**不应直接推进 O2a/b**，因为 delayed opacity path 还没有证明自己至少不伤 replay。
- 更合理的下一步应是：
  1. 先补 C0-2：确认 `participation_score` 与 `state_score / ranking_score` 的差异、candidate disagreement 与 cluster 统计；
  2. 只改小步的 candidate / attenuation 规则，然后**只重跑新 opacity 臂**；
  3. 只有当 delayed opacity 不再弱负时，才进入 O2a/b；
  4. 若 action law 已明显更对但仍弱负，再考虑 timing/current-step integration，而不是先切 population universe。

### 2.10 Abs prior 性质
- 固定背景：λ_abs_t=3.0, λ_abs_r=0.1。
- 当前主收益解释仍在 topology，不在 abs prior 扫描。

---

## 3. Pipeline 架构

### 3.1 当前候选主线

```text
StageA output / optional StageA.5 handoff
→ StageB with old A1 observation + new T1 topology
→ replay
```

### 3.2 当前 A1 分支关系

```text
signal_v2 builder
├── old A1: joint_confidence_v2 + joint_depth_target_v2
├── current new A1: pseudo_*_joint_v1
├── verify proxy: pseudo_depth_target_joint_v1 + pseudo_confidence_verify_v1
├── brpo_style_v1: pseudo_confidence_brpo_style_v1 + pseudo_depth_target_brpo_style_v1
├── brpo_style_v2: pseudo_confidence_brpo_style_v2 + pseudo_depth_target_brpo_style_v2
└── brpo_direct_v1: pseudo_confidence_brpo_direct_v1 + pseudo_depth_target_brpo_direct_v1
```

关键语义：
- old A1 = downstream support filter
- current new A1 = candidate-built observation object
- verify proxy = negative probe，已退场
- brpo_style_v1 / v2 = 过渡 builder probes
- brpo_direct_v1 = 当前 A1 的真正主研究线

### 3.3 B3 第一版接入位置

```text
iter t:
  render pseudo / real
  assemble joint loss
  backward once
  local gating / SPGM summary
  build participation action for iter t+1
    - boolean path: participation mask
    - opacity path: participation opacity scale
  optimizer.step

iter t+1:
  pseudo render consumes participation mask or opacity scale
```

---

## 4. 接口定义

### 4.1 Refine / observation / topology 接口

关键 CLI：
- `pseudo_observation_mode`: `off` / `brpo_joint_v1` / `brpo_verify_v1` / `brpo_style_v1` / `brpo_style_v2` / `brpo_direct_v1`
- `joint_topology_mode`: `off` / `brpo_joint_v1`
- `pseudo_local_gating_spgm_manager_mode`: `summary_only` / `xyz_lr_scale` / `deterministic_participation` / `deterministic_opacity_participation`

### 4.2 BRPO-style / direct A1 接口

当前 `brpo_direct_v1` contract：
- `confidence`: `pseudo_confidence_brpo_direct_v1.npy`
- `depth_target`: `pseudo_depth_target_brpo_direct_v1.npy`
- `source_map`: `pseudo_source_map_brpo_direct_v1.npy`
- `valid_mask`: `pseudo_valid_mask_brpo_direct_v1.npy`
- meta: `brpo_direct_observation_meta_v1.json`

它的核心语义是：
- `C_m` 来自 fused pseudo-frame reciprocal correspondence support，并受 projected-depth overlap validity 约束
- `depth_target` 来自同一 verified support set 下、按 `overlap_conf_left/right` 加权的 direct projected-depth composition
- RGB / depth 共用同一个 direct valid set

`brpo_style_v2` 保留为对照接口：
- `confidence`: `pseudo_confidence_brpo_style_v2.npy`
- `depth_target`: `pseudo_depth_target_brpo_style_v2.npy`
- meta: `brpo_style_observation_meta_v2.json`
- 角色：continuous verifier quality + stable-target blend 的过渡 builder，对照 direct BRPO v1 使用

### 4.3 B3 participation control 接口

当前保留两条 manager action 路径：
- boolean path：
  - `state_candidate_quantile`
  - `state_participation_keep_near`
  - `state_participation_keep_mid`
  - `state_participation_keep_far`
  - `participation_render_mask`
- opacity path：
  - `participation_score`
  - `state_opacity_floor_near`
  - `state_opacity_floor_mid`
  - `state_opacity_floor_far`
  - `participation_opacity_scale`

### 4.4 Renderer contract

为了让 B3 的 participation control 真正进入主 loop，renderer 的 masked / opacity-scaled 分支必须满足：
- 返回值签名与 unmasked 分支一致；
- `visibility_filter` 必须回到 **full-length Gaussian mask** 口径，而不是子集口径；
- `opacity_scale` 只能作用于 pseudo branch 的临时 render，不得永久写坏参数本体。

---

## 5. 设计原则（长期有效）

### 5.1 Replay 评估口径
- replay 必须消费真实变化 artifact（PLY / camera states）。
- A1 compare 必须在 **fixed new T1 + summary_only** 下做 apples-to-apples；不要再把 topology / B3 混进 observation compare。

### 5.2 A1 实验验收逻辑
- 当前 A1 不再问“是不是比 control 好一点”，而要问：
  1. 是否超过 `old A1 + new T1`；
  2. 是否更接近 BRPO-style semantics；
  3. 剩余 gap 是否能明确归因到 verifier backend / target builder / exact supervision contract 哪一层。

### 5.3 Supervision scope 与 population control
- 弱 supervision 不应直接驱动激进 population shrinking。
- population manager 的第一步应是**轻度 participation attenuation**，不是过早 drop 过多 low-score subset。

### 5.4 实验验收阶梯
1. Wiring / smoke 成立；
2. Formal compare 不再伤 replay；
3. 再讨论更强 action（更低 keep 或 stochastic）。

---

## 6. 当前不做的事
- 不回头继续打磨 `brpo_verify_v1`。
- 不把当前 `brpo_direct_v1` 误写成完整 BRPO implementation。
- 不把当前 residual gap 简化成 confidence-only 或 depth-only 单侧问题。
- 不在 observation compare 里同时改 topology 或 B3。
- 不把当前 deterministic participation v1 直接 landing。
- 不在 delayed deterministic_opacity_participation 仍弱负时直接推进 O2a/b。

---

## 7. 参考
- 当前状态：[STATUS.md]
- 过程记录：[CHANGELOG.md]
- A1 细化分析：[MASK_DESIGN.md]
- B3 / refine 设计：[REFINE_DESIGN.md]
- 总规划：[BRPO_A1_B3_reexploration_master_plan.md]
- A1 落地文档：[A1_verifier_decoupled_pseudo_observation_engineering_plan.md]

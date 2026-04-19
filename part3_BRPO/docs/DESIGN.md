# DESIGN.md - Part3 BRPO 设计文档

> 更新时间：2026-04-19 05:16 (Asia/Shanghai)

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
3. **B3** 已经开始进入 rewrite 执行，但只完成了 deterministic participation 的第一版，不构成新的默认主线；
4. 因此当前默认候选主线仍是 **old A1 + new T1**，B3 仍处在“方法对象已切换、结果尚未转正”的阶段。 [参见 STATUS §5.5]

### 1.2 当前阶段定义

| 阶段 | 设计角色 | 说明 |
|------|----------|------|
| StageA | pose/exposure 调整 | 不更新 Gaussian，replay 不是判优指标 |
| StageA.5 | optional warmup / control | 可保留，但不再是默认主线必经阶段 |
| StageB new topology | `brpo_joint_v1` joint loop | 当前 topology 主线 |
| SPGM repair A | 当前 policy anchor | B3 未稳定前继续作为主对照 |
| B3 deterministic participation | first real population-control actor | 已接入，但当前实现尚未站稳 |

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

### 2.4 Signal v2 / A1 性质
- old A1 的正确定位仍是 **joint support filter / unified consumer semantics**。
- new A1 的 rewrite 已成立，但当前不构成默认主线。
- 当前最稳 observation 仍是 old A1。

### 2.5 SPGM / B3 性质
- 旧 B3 的正确定位是 **post-backward deterministic grad/state probe**，不能再误称为 population manager。
- 新 B3 第一版已经真正改成：**pre-render deterministic participation controller**。
- 这意味着它开始回答“谁更少参与 pseudo render”，而不是“render 完之后梯度怎么乘”。
- 但当前第一版 keep 策略过于激进，首轮 compare weak-negative，因此设计判断应是：**方法对象已正确，action schedule 仍需收缩。**

### 2.6 当前 B3 调整方向
- 下一轮优先方向不是 stochastic，也不是继续回退到 xyz grad scale。
- 下一轮应做：
  1. **更保守的 participation keep**（尤其 far / mid）
  2. 必要时让 candidate 仍按 low-score subset 选，但 action 更温和
  3. 先验证 deterministic participation 不再伤 replay，再讨论 stochastic masking
- 因而当前 B3 的策略重点是：**减小 action 强度，而不是再改 action 位置。**

### 2.7 Abs prior 性质
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

### 3.2 B3 第一版接入位置

```text
iter t:
  render pseudo / real
  assemble joint loss
  backward once
  local gating / SPGM summary
  build participation mask for iter t+1
  optimizer.step

iter t+1:
  pseudo render consumes participation mask
```

关键语义：
- B3 第一版还不是 stochastic manager；
- 但它已经进入 **pseudo render 前的 participation control**，不再只是 step 前 grad scaler。

---

## 4. 接口定义

### 4.1 Refine / topology 接口

关键 CLI：
- `pseudo_observation_mode`: off / brpo_joint_v1
- `joint_topology_mode`: off / brpo_joint_v1
- `pseudo_local_gating_spgm_manager_mode`: `summary_only` / `xyz_lr_scale` / `deterministic_participation`

### 4.2 B3 deterministic participation 接口

新增/固化的 manager 语义：
- `state_candidate_quantile`
- `state_participation_keep_near`
- `state_participation_keep_mid`
- `state_participation_keep_far`
- `participation_render_mask`（供下一轮 pseudo render 消费）

history / diagnostics 重点字段：
- `spgm_state_participation_ratio`
- `spgm_state_participation_ratio_near/mid/far`
- `spgm_state_participation_drop_count_near/mid/far`
- `spgm_state_candidate_count_near/mid/far`

### 4.3 Renderer contract

为了让 B3 的 participation control 真正进入主 loop，renderer 的 masked 分支必须满足：
- 返回值签名与 unmasked 分支一致；
- `visibility_filter` 必须回到 **full-length Gaussian mask** 口径，而不是子集口径。

这是本轮 B3 执行中实际暴露并修复的必要 contract。

---

## 5. 设计原则（长期有效）

### 5.1 Replay 评估口径
- replay 必须消费真实变化 artifact（PLY / camera states）。
- B3 compare 仍必须在 **old A1 + new T1** 固定主线下做 apples-to-apples。

### 5.2 Supervision scope 与 population control
- 弱 supervision 不应直接驱动激进 population shrinking。
- population manager 的第一步应是**轻度 participation attenuation**，不是过早 drop 过多 low-score subset。

### 5.3 实验验收阶梯
1. Wiring / smoke 成立；
2. Formal compare 不再伤 replay；
3. 再讨论更强 action（更低 keep 或 stochastic）。

---

## 6. 当前不做的事
- 不把当前 deterministic participation v1 直接 landing。
- 不重新把旧 `xyz_lr_scale` 当成 B3 主线。
- 不在 deterministic participation 仍弱负向时直接跳 stochastic masking。
- 不重新打开 old A1 / new A1 主线归属争论。

---

## 7. 参考
- 当前状态：[STATUS.md]
- 过程记录：[CHANGELOG.md]
- B3 主规划：[B3_deterministic_state_management_engineering_plan.md]

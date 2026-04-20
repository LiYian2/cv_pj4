# BRPO A1 / B3 新一轮重构探索总规划

> 记录时间：2026-04-20 00:40
> 约束来源：`part3_BRPO_A1_B3_vs_BRPO_detailed_analysis.md`
> 文档角色：这轮探索的总控文档。后续 A1 / B3 / timing 的工程落地都以本文件为上位约束。

---

## 0. 先说结论

这轮工作不是继续在旧 pipeline 上做小修，而是**按照 detailed analysis 里拆出来的真实差异，重新组织 A1 / B3 / optimization timing 三条主线**。

同时要明确三件事：

1. **新一轮结果不预设优于旧主线。** 所有新实现都要回到 formal compare，和已归档基线做对照。
2. **旧模块默认保留。** 除非确认 wiring / semantics 有问题，否则保留脚本、接口、历史产物，只在 pipeline / 文档层降级为 archived or non-mainline。
3. **后续所有工程动作必须贴着 analysis 文档。** 不能再沿用“先在现有主线补一点、看能不能碰巧变好”的节奏。

---

## 1. 当前基线与归档地位

### 1.1 当前需要保留的已知可运行基线

当前 formal compare 的主参考仍然是：

- observation baseline: `old A1 = joint_confidence_v2 + joint_depth_v2`
- topology baseline: `new T1 = joint_topology_mode=brpo_joint_v1`
- StageB control: `summary_only`

也就是：

> **`old A1 + new T1 + summary_only`**

这条线不是“真理”，但它是当前最稳的 compare anchor。

### 1.2 已归档但必须保留可比性的历史方向

以下文档已经归档，只作为历史实现 / compare baseline，不再作为主线执行文档：

- `archived/2026-04-superseded-plans/A1_unified_rgbd_joint_confidence_engineering_plan.md`
- `archived/2026-04-superseded-plans/A2_geometry_constrained_support_depth_expand_engineering_plan.md`
- `archived/2026-04-superseded-plans/B1_spgm_manager_shell_engineering_plan.md`
- `archived/2026-04-superseded-plans/B2_scene_aware_density_proxy_and_state_score_engineering_plan.md`
- `archived/2026-04-superseded-plans/B3_deterministic_state_management_engineering_plan.md`
- `archived/2026-04-superseded-plans/T1_brpo_joint_optimization_topology_engineering_plan.md`
- `archived/2026-04-experiments/A1_joint_confidence_stageb_compare_20260418.md`
- `archived/2026-04-experiments/P2T_selector_confirmation_precision_compare_20260418.md`

这些文件**不删、不改内容**，后续只作为 compare object / historical reference。

---

## 2. detailed analysis 给出的三条真正主线

analysis 文档已经把差异讲清楚了，这轮探索只围绕下面三件事展开。

### 2.1 A1：从 candidate-scoring confidence 走向 verifier-style confidence

当前 `pseudo_branch/brpo_v2_signal/joint_observation.py` 的 `brpo_joint_v1` 已经是完整 joint builder，
但它的核心问题是：

- `target_depth` 和 `confidence_joint` 来自同一套 candidate score；
- 也就是 `score_k -> best_score / fused_depth -> confidence_joint / valid_mask`；
- 这会形成 **target / confidence 同源**。

analysis 的判断很明确：

> BRPO 的关键不只是 “joint”，而是 **pseudo-frame 生成后再做独立 verifier**。

所以 A1 后续不再围绕“怎么继续改 candidate score”，而是围绕：

- 怎么引入**独立 verifier**；
- 怎么把 verifier 结果变成 RGB/depth 共用的 confidence；
- 怎么在**不打断现有接口**的前提下，把 verifier-style observation 接入当前 consumer。

### 2.2 B3：从 next-step boolean render selection 走向 opacity participation manager

当前 B3 的关键代码链是：

- `collect_spgm_stats(...)`
- `build_spgm_importance_score(...)`
- `apply_spgm_state_management(...)`
- `run_pseudo_refinement_v2.py` 里 `stageB_spgm_participation_render_mask`

它已经不再是旧 `_xyz.grad` scaler，但仍然有三个核心 gap：

1. 动作变量还是 **boolean render mask**，不是 opacity attenuation；
2. 主 score 还是工程化 `state_score`，不是更 BRPO-like 的 importance score；
3. control universe 仍主要锚在 `active_mask = support_count > 0`，而不是 scene-level `population_active_mask`。

所以 B3 后续也不再围绕 keep-ratio 微扫本身，而是围绕：

- action variable 改造；
- score routing 改造；
- universe 改造。

### 2.3 Optimization timing：从 delayed controller 走向 current-step effective controller

当前 StageB 时序是：

1. 当前 iter 先拿 `current_pseudo_render_mask` render pseudo；
2. 组 loss / backward；
3. `maybe_apply_pseudo_local_gating(...)` 统计本轮 SPGM；
4. 生成下一轮 `_spgm_participation_render_mask`；
5. iter `t+1` 再消费。

也就是：

> **iter t 统计 / 决策 -> iter t+1 生效**

analysis 判断这是 delayed operator，不是 BRPO-style current-step operator。

所以 timing 这条线要单独做，不与 A1/B3 混在一起讲。

---

## 3. 这轮探索的硬约束

### 3.1 保留脚本和接口，默认做“加法式接入”

除非确认某段逻辑本身错误，否则：

- 不删除现有 producer / consumer；
- 不重命名旧接口；
- 不覆盖历史结果目录；
- 新方法优先通过 `new mode / new bundle / new flag` 接入。

### 3.2 新方向允许失败，但失败必须可解释

任何新实现最终都要落回 compare。判断标准不是“写出来像 BRPO”，而是：

- 是否更贴近 analysis 中识别出的真实差异；
- 即便结果负，也能清楚解释是 **verifier / action law / timing** 的哪一层导致的。

### 3.3 不再用“局部补丁变好”当主线推进方式

禁止的推进方式：

- 在 `brpo_joint_v1` 现有 score 上继续局部调几个权重，然后宣称更接近 BRPO；
- 在 `deterministic_participation` 上继续只扫 keep-ratio，然后把结果解释成 B3 主线进展；
- 不拆 timing，只用 delayed controller 继续近似 current-step action。

---

## 4. 四份工程落地文档的角色分工

### 4.1 `BRPO_A1_B3_reexploration_master_plan.md`（本文件）

回答：

- 这轮到底做什么；
- 哪些方向继续，哪些方向归档；
- compare anchor 是什么；
- 实施顺序如何排。

### 4.2 `A1_verifier_decoupled_pseudo_observation_engineering_plan.md`

回答：

- 如何在当前 signal pipeline 上引入 verifier-style confidence；
- 如何让 target 与 confidence 解耦；
- producer / consumer / compare 怎么落地。

### 4.3 `B3_opacity_participation_population_manager_engineering_plan.md`

回答：

- 如何把 current B3 从 boolean selector 推向 opacity participation；
- 如何把 action score 从 `state_score` 收回到更 BRPO-like 的 importance score；
- 如何逐步把 universe 从 `active_mask` 推到 `population_active_mask`。

### 4.4 `current_step_joint_integration_timing_engineering_plan.md`

回答：

- 如何让 A1 verifier / B3 participation 在 current step 生效；
- 如何拆出 probe / decide / loss render 时序；
- 如何避免继续使用 delayed controller 近似 paper semantics。

---

## 5. 这轮建议执行顺序

执行顺序必须固定，不要来回跳：

### Phase 0：先把执行文档和历史线整理干净

- 本文件落地；
- 四份新文档建立；
- `charles.md` 更新成新读法；
- 旧执行文档继续停留在 archive，不再回主线。

### Phase 1：A1 先行，先做 verifier decoupling

原因：

- A1 直接决定 pseudo observation semantics；
- B3 和 timing 的后续设计，都需要清楚 observation 到底是什么；
- verifier-style confidence 是 analysis 里最明确、最“不是旧 pipeline 小修”的部分。

### Phase 2：B3 action law 改造

在 A1 新 observation 定义明确后，再推进：

- opacity attenuation；
- importance-score-based participation；
- population universe。

### Phase 3：timing / current-step integration

最后专门处理 current-step effective wiring：

- 先拆 clear API；
- 再改 StageB 迭代时序；
- 最后再考虑是否需要 stochastic branch。

### Phase 4：formal compare

所有新线最终都要做 compare，最少包含：

- archived baseline：`old A1 + new T1 + summary_only`
- new A1 only
- new B3 only
- new A1 + new B3 + new timing（如果已完成）

---

## 6. compare 原则

### 6.1 compare 的最小单位

每次只回答一个问题：

- verifier confidence 本身是否比 current score-derived confidence 更稳？
- opacity attenuation 是否优于 boolean selector？
- current-step 生效是否优于 delayed 生效？

不要一轮 compare 同时改 observation / manager / timing 三层。

### 6.2 compare 输出要求

每次 compare 都必须至少有：

- replay eval summary
- 主参数清单
- 和 archived baseline 的 delta
- 和当前主实验目标相关的诊断摘要

### 6.3 compare 解释要求

结果即使为负，也必须能说清：

- 是 verifier 太硬 / 太稀；
- 是 participation 太强 / 太弱；
- 是 timing 增加了不稳定性；
- 还是 wiring 本身没真正让 current step 受控。

---

## 7. 本轮不做的事情

1. 不回退去重新讨论 old A1 / new A1 / T1 的旧叙事；
2. 不把 keep-ratio sweep 当成 B3 主线本身；
3. 不先上 stochastic Bernoulli masking；
4. 不删除 archived 脚本 / 文档；
5. 不把 PIPELINE 里旧模块 physically remove，只做地位更新。

---

## 8. 给后续执行者的一句话

> **这轮工作的目标不是把旧 pipeline 再补平一点，而是按 detailed analysis 重新建立三条真正的主线：A1 verifier、B3 opacity participation、以及 current-step integration。所有新实现都允许输给旧线，但必须和旧线做正式对照，并且输赢都能解释。**

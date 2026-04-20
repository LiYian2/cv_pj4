# REFINE_DESIGN.md - B3 细化机制与诊断文档

> 更新时间：2026-04-20 05:40 (Asia/Shanghai)
>
> 目标：参照 `MASK_DESIGN.md` 的写法，把当前 B3（含不同实现）与 BRPO B3 的信息流、信号构建、下游消费路径讲清楚，并给出基于当前实测结果的诊断结论。

---

## 0. 先说结论

当前 B3 已经从旧 `xyz` 梯度缩放推进到 pre-render participation 控制，并进一步接通了 delayed opacity attenuation（O0/O1 + C1）。

但在当前配置下，`deterministic_opacity_participation` 仍然是 weak-negative，且略差于旧 boolean conservative arm。**结论不是方向错，而是当前 action law 还不够“有差异地作用到对的点”。**

核心原因（按影响顺序）是：
1. `ranking/state/participation` 三条分数并没有真正拉开到可显著改变 candidate 的程度；
2. 当前 quantile=0.5 让每个 cluster 都接近“固定 50% candidate”，再叠加高相关分数，导致 boolean 与 opacity 命中的群体高度重合；
3. opacity floor 设定偏保守（near=1.0, mid=1.0, far=0.9），再乘上当前 participation 分布后，实际 far 仅衰减到约 `0.981`，作用太轻；
4. 当前依然是 delayed controller（iter t 统计，iter t+1 生效）。

因此当前不应进入 O2a/b；先做 C0 级别的 score/candidate/action 诊断与小步改动更合理。

---

## 1. 证据来源（本文件所有结论的依据）

### 1.1 设计与上位约束
- `docs/BRPO_A1_B3_reexploration_master_plan.md`
- `docs/B3_opacity_participation_population_manager_engineering_plan.md`
- `docs/current_step_joint_integration_timing_engineering_plan.md`
- `docs/part3_BRPO_A1_B3_vs_BRPO_detailed_analysis.md`（B3 section）

### 1.2 当前实现代码
- `pseudo_branch/spgm/stats.py`
- `pseudo_branch/spgm/score.py`
- `pseudo_branch/spgm/policy.py`
- `pseudo_branch/spgm/manager.py`
- `scripts/run_pseudo_refinement_v2.py`
- `third_party/S3PO-GS/gaussian_splatting/gaussian_renderer/__init__.py`

### 1.3 当前实验与诊断数据
- C1 compare（复用 controls + 新跑 opacity）：
  - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_b3_opacity_participation_compare_e1/compare_summary.json`
- C0 诊断汇总：
  - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_b3_c0_diagnosis/diagnosis_summary.json`
- C0 四分组 probe：
  - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_b3_c0_diagnosis/partition_probe.json`
  - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_b3_c0_diagnosis/C0_DIAGNOSIS_REPORT.md`

---

## 2. 统一符号与对象

设 Gaussian 索引为 $i$，cluster 索引为 $c \in \{near,mid,far\}$。

- $s_i$：`support_count`（accepted pseudo support）
- $\tilde s_i$：`population_support_count`（current train window support）
- $z_i$：`depth_value`
- $\rho_i$：`density_proxy` 或 `struct_density_proxy`
- $\mathcal A$：`active_mask = [s_i > 0]`
- $\mathcal P$：`population_active_mask = [\tilde s_i > 0]`

本实现当前主控制域仍是 $\mathcal A$，而不是 $\mathcal P$。

---

## 3. BRPO 的 B3（参考语义）

按 analysis 文档，BRPO B3 的主链路是：

1. **Depth partition**：按深度排序分层（near/mid/far）。
2. **Density entropy**：评估 density 信号在本轮是否有辨别力。
3. **Unified score**：
$$
S_i = \alpha \hat s_i^{(z)} + (1-\alpha)\hat s_i^{(\rho)}
$$
4. **Stochastic opacity participation**：
$$
p_i^{drop} = r \cdot w_{c(i)} \cdot S_i, \quad m_i \sim Bernoulli(1-p_i^{drop}), \quad \alpha_i^{eff}=\alpha_i m_i
$$

关键点：BRPO 的动作对象是 **opacity**，动作规律是 **per-Gaussian stochastic masking**，并且语义上更接近 current-step 生效。

---

## 4. 我们当前 B3 的几种实现（按信息流拆解）

## 4.1 B3-0（旧）：`xyz_lr_scale`（已降级）

### 信息来源
- `stats.py` / `score.py` 的 scene统计与分数。

### 信号构建
- manager 生成 per-Gaussian xyz grad scale。

### 下游消费
- backward 后对 `_xyz.grad` 乘缩放，再 `optimizer.step`。

### 数学形态（近似）
$$
g_{xyz,i} \leftarrow g_{xyz,i} \cdot \kappa_i
$$

### 本质
- post-backward probe，不是 population participation manager。

---

## 4.2 B3-1（当前 boolean 路径）：`deterministic_participation`

### 信息来源
- `collect_spgm_stats(...)`：$s_i,\tilde s_i,z_i,\rho_i$ 及 $\mathcal A,\mathcal P$
- `build_spgm_importance_score(...)`：`ranking_score` / `state_score`

### 信号构建
1. `state_score` cluster 内 quantile 取 candidate；
2. candidate 内按 `state_score` keep-top-k；
3. 生成 boolean `participation_render_mask`。

### 下游消费
- `run_pseudo_refinement_v2.py` 在下一轮 pseudo render 里：
  - `render(..., mask=current_pseudo_render_mask)`

### 数学形态（近似）
$$
\mathcal C_c = \{i\in\mathcal A\cap c : state_i \le Q_q(state|c)\}, \quad
\mathcal K_c = TopK(state,\mathcal C_c,keep_c), \quad
m_i = \mathbb 1[i\in \mathcal K_c \cup (\mathcal A\setminus \mathcal C_c)]
$$

---

## 4.3 B3-2（当前 opacity 路径）：`deterministic_opacity_participation`

### 信息来源
- 同上游统计，但动作分数切到 `participation_score`。

### 信号构建
1. `participation_score` cluster 内 quantile 取 candidate；
2. 对 candidate 连续衰减：
$$
scale_i = floor_c + (1-floor_c)\cdot participation_i
$$
3. 非 candidate 维持 1.0。

### 下游消费
- `render(..., opacity_scale=current_pseudo_opacity_scale)`
- 仍为 delayed 生效：iter t 决策，iter t+1 消费。

### 当前测试配置
- `floor_near=1.0, floor_mid=1.0, floor_far=0.9`

---

## 4.4 `summary_only`（control）

### 信息来源
- stats/score照常统计。

### 信号构建
- 不输出任何状态动作。

### 下游消费
- pseudo render 不受 B3 manager 额外控制。

---

## 4.5 从信息到动作的完整消费链（逐层讲“每个 part 是做什么的”）

当前 B3 真正的链路不是一个函数，而是六层：

1. **stats 层**（`stats.py`）
   - 回答：**当前 scene / current window 里每个 Gaussian 被谁看见、看见多少、深度大概在哪、密度代理如何**。
   - 产物：`support_count / population_support_count / depth_value / density_proxy / struct_density_proxy / active_mask / population_active_mask`。

2. **score 层**（`score.py`）
   - 回答：**这些统计量在“排序”“状态诊断”“参与控制”三种语义下分别该怎么压缩成分数**。
   - 产物：`ranking_score / state_score / participation_score`。

3. **policy 层**（`policy.py`）
   - 回答：**当前 active set 里哪些点在 update 侧要被保守对待，梯度权重怎么给**。
   - 产物：`weights / selected_ratio / selected_count_*`。
   - 注意：这层主要还是 update / grad 权重语义，不是 render participation 主语义。

4. **manager 层**（`manager.py`）
   - 回答：**score 怎样变成真正的 participation action**。
   - boolean 路径输出 `participation_render_mask`；opacity 路径输出 `participation_opacity_scale`。

5. **renderer 层**（`gaussian_renderer/__init__.py`）
   - 回答：**action 到底怎么进入 forward**。
   - boolean：通过 `mask=...` 直接决定哪些 Gaussian 进入 rasterizer。
   - opacity：通过 `opacity_scale=...` 在 rasterization 前改 `opacity`。

6. **loop 层**（`run_pseudo_refinement_v2.py`）
   - 回答：**这些 action 在哪一轮生效**。
   - 当前答案仍是 delayed：iter `t` 统计 / 决策，iter `t+1` 才消费。

这六层里，当前真正的瓶颈主要在 2/4/6：
- score 没拉开 enough；
- manager 的动作律还太保守；
- loop 仍是 delayed。

---

## 5. 三条分数（ranking / state / participation）的机制差异

### 5.1 当前代码定义

在 `score.py`：

1. $importance\_raw$
$$
importance_i = \alpha \cdot depth_i + (1-\alpha)\cdot density_i
$$

2. `ranking_score`（当前 `v1` 下与 `importance_score`同源）
$$
ranking_i \approx importance_i \cdot support_i^{\eta}
$$

3. `state_score`
$$
state_i = 0.45\cdot structDensity_i + 0.35\cdot popSupport_i + 0.20\cdot depth_i
$$

4. `participation_score`
$$
part_i = 0.80\cdot importance_i + 0.20\cdot popSupport_i
$$

### 5.2 语义定位
- `ranking_score`：更偏“排序与更新权限”；
- `state_score`：更偏“状态/可见性/支撑强度诊断”；
- `participation_score`：尝试作为“参与控制”分数。

---

## 6. 现状诊断：三者差异拆解（你要的 diagnosis）

## 6.1 C1 结果（行为层）

来自 `20260420_b3_opacity_participation_compare_e1/compare_summary.json`：

- summary_only：`24.187304 / 0.875587 / 0.080364`
- old boolean far090_mid100：`24.186847 / 0.875591 / 0.080387`
- new opacity floor090_mid100：`24.186731 / 0.875586 / 0.080398`

delta：
- boolean vs summary：`-0.000457 PSNR`
- opacity vs summary：`-0.000573 PSNR`
- opacity vs boolean：`-0.000116 PSNR`

=> 当前 opacity path 仍是 weak-negative，且略差于 boolean conservative arm。

---

## 6.2 C0 结果（机制层）

来自 `20260420_b3_c0_diagnosis/diagnosis_summary.json`（10 batches, 无训练，仅统计）：

### A) 分数相关性
- corr(`ranking`,`participation`) = **0.8807**（高）
- corr(`state`,`participation`) = **0.8268**（高）
- corr(`ranking`,`state`) = **0.5261**（中）

### B) 与 population_support / depth 的耦合
- corr(`state`,pop_support_norm) = **0.9219**（极高）
- corr(`participation`,pop_support_norm) = **0.8047**（高）
- corr(`ranking`,pop_support_norm) = **0.4323**（中）
- corr(`ranking`,depth) = **+0.2344**
- corr(`state`,depth) = **-0.1255**
- corr(`participation`,depth) = **-0.0474**

解释：
- `state` 比较像“population-support controller”；
- `ranking` 比较像“importance排序”；
- `participation` 在两者之间，但仍显著受 pop_support 牵引。

### C) candidate overlap（state vs participation）
- Jaccard = **0.6658**（重合较高）
- state-only ratio in active = **0.1026**
- part-only ratio in active = **0.1026**

=> 两套 candidate 并非等价，但差异量级还不足以让 action 显著分离。

### D) cluster 行为（均值）
- far：
  - `state_score_mean` = **0.4229**
  - `participation_score_mean` = **0.7016**
  - boolean keep ratio = **0.9501**
  - opacity scale mean = **0.9808**

- near/mid 基本不衰减（keep≈1，scale≈1）。

=> 当前设定下，真正动作几乎只在 far，且幅度偏小（0.98 级别）。

---

## 6.3 四分组 probe：`state_only / part_only / both / neither`（更细拆）

额外 probe（10 batches）已保存到：
- `partition_probe.json`
- `C0_DIAGNOSIS_REPORT.md`

我们把 active set $\mathcal A$ 拆成四群：

1. `state_only = state_candidate \setminus part_candidate`
2. `part_only = part_candidate \setminus state_candidate`
3. `both = state_candidate \cap part_candidate`
4. `neither = \mathcal A \setminus (state_candidate \cup part_candidate)`

### (a) `state_only` 在做什么
它表示：**按 state 看应该动，但按 participation 看不一定该动**。

实测均值：
- ratio in active = `0.1026`
- ranking = `0.7190`（高）
- state = `0.4020`（低）
- participation = `0.7369`（高）
- support_norm / pop_support_norm = `0.7862 / 0.8085`

解释：
- 这些点的重要性并不低；
- 但按 state 语义，它们更像“当前状态较差 / 结构较弱”的点；
- 如果动作主要跟 state 走，boolean 更容易命中它们。

### (b) `part_only` 在做什么
它表示：**按 participation 看应该动，但按 state 看不一定该动**。

实测均值：
- ratio in active = `0.1026`
- ranking = `0.5790`（低）
- state = `0.4551`（反而更高）
- participation = `0.6466`（低）
- support_norm / pop_support_norm = `0.8997 / 0.9170`

解释：
- 这些点支撑更多、状态不差；
- 但按 importance/participation 语义，它们更像“虽然常出现，但不值得占太多参与权”的点；
- 这正是 opacity 路径理论上更该命中的群体。

### (c) `both` 在做什么
它表示：**state 与 participation 都同意这些点该被处理**。

实测均值：
- ratio in active = `0.3974`
- ranking = `0.6053`
- state = `0.2905`
- participation = `0.5715`
- support_norm / pop_support_norm = `0.3633 / 0.4364`

解释：
- 这是当前真正意义上的“共同低优先级群体”；
- 由于它占 active set 接近 40%，说明当前 quantile=0.5 把大量点都推入了共同 candidate。

### (d) `neither` 在做什么
它表示：**state 与 participation 都不认为当前该动**。

实测均值：
- ratio in active = `0.3973`
- ranking = `0.7618`
- state = `0.4835`
- participation = `0.7984`
- support_norm / pop_support_norm = `0.9423 / 0.9449`

解释：
- 这是 active set 里最稳定、最该保留的群体；
- 它们的重要性高、状态也高，因此 boolean / opacity 都不该轻易碰。

### 这四群体说明了什么
- `state_only` 和 `part_only` 确实证明：`state_score` 与 `participation_score` **不是换名**。
- 但 `both + neither` 合计接近 80%，说明当前 action 真正发生差异的群体仍然偏小。
- 这也是为什么 opacity 路径虽然“概念更对”，但指标上还没体现出明显改善。

---

## 7. 为什么 current opacity 路径还没赢（归因）

### 7.1 候选分离不够
虽然 state-only / part-only 存在，但总体重合仍高，且 quantile=0.5 让每 cluster 都接近“固定 50% candidate”。

### 7.2 衰减幅度偏弱
当前 far 平均 scale ≈ 0.981；对渲染贡献与梯度路径的调制很轻。

### 7.3 仍是 delayed 控制
iter t 统计、t+1 生效，难以精准作用到当前 residual landscape。

### 7.4 active-domain 仍主导
动作域仍锚在 `active_mask`，population 语义虽有统计但未成为主控制域。

---

## 8. O2a/b 现在该不该推进？

**不该。**

理由：
1. C1 的前置门槛（delayed opacity 至少不弱负）尚未满足；
2. 直接进 O2a/b 会叠加新变量，削弱归因能力；
3. 现阶段更高价值的是先把 C0 机制做清（score/candidate/action law）。

---

## 9. 下一步建议（只给可执行项）

1. **先做 C0-1（不改训练，只加统计）**
   - 固定每轮记录：`state_only / part_only / both / neither` 四群体的 depth/support/ranking/state/part 均值和 cluster 占比。

2. **再做 C0-2（小步 action-law）**
   - 不改 O2，不改 timing；
   - 仅改一处：candidate quantile（例如 far 单独 quantile）或 floor 映射曲线（例如非线性映射），保持 near/mid 不动。

3. **继续 compare 时只重跑新 opacity 臂**
   - 复用旧 summary/boolean controls（遵循当前项目约定）。

4. **只有当 delayed opacity 不再弱负时，才进入 O2a/b**。

---

## 10. 一句话总结

当前 B3 已经从“后向梯度缩放”推进到“前向参与控制”，并完成了 boolean 与 opacity 两种 delayed action-law 的正式对照；但从机制诊断看，`participation_score` 还没把 candidate 与 action 拉开到足够程度，导致 current opacity path 仍弱负。因此下一步不是 O2a/b，而是继续在 C0 层把分数语义与动作律做实。

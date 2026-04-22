# G_direct_brpo_alignment_engineering_plan.md

> 上位文档：`BRPO_A1_B3_reexploration_master_plan.md`、`part3_BRPO_A1_B3_vs_BRPO_detailed_analysis.md`
> 相关现状：`docs/design/GAUSSIAN_MANAGEMENT_DESIGN.md`、`docs/B3_opacity_participation_population_manager_engineering_plan.md`、`docs/current_step_joint_integration_timing_engineering_plan.md`
> 触发原因：用户明确要求 **G~ 直接对齐 BRPO**，不再把 `delayed deterministic opacity + C0-2 小步修补` 当成主推进路线。
> 文档角色：定义一条新的 **direct BRPO G~ path**，把当前系统从 “BRPO-like wiring” 推到 “BRPO semantics 真正落地”。

---

## 0. 结论先说

当前 G~ **还没有对齐 BRPO**。它已经完成的，只是第一层 BRPO-like wiring：
- pre-render participation control 已接通；
- `participation_score` 已显式拆出；
- `deterministic_opacity_participation` 已可运行；
- renderer 已能消费 `opacity_scale`。

但真正决定 BRPO 语义的四层仍未对齐：
1. **control universe**：当前主域仍是 `active_mask`，不是 scene-level `population_active_mask`；
2. **score semantics**：当前 action 不来自 paper unified score `S_i`，而来自工程化重写分数；
3. **action semantics**：当前是 deterministic candidate / top-k / floor 映射，不是 per-Gaussian Bernoulli opacity masking；
4. **timing semantics**：当前是 delayed controller，不是 current-step operator。

因此，这轮不再把 `deterministic_opacity_participation + C0-2` 当 landing 路线，而是直接实现一条并行的：

> **population universe → BRPO unified score → Bernoulli opacity masking → current-step effect**

旧 deterministic 路线保留，但只作为 compare control，不再作为目标态。

---

## 1. 当前代码事实（已核对 live code）

### 1.1 `pseudo_branch/spgm/stats.py`
当前 `collect_spgm_stats(...)` 已同时产出：
- `support_count`
- `population_support_count`
- `active_mask`
- `population_active_mask`

但真实主控制路径仍然锚在：
- `active_mask = support_count > 0`

也就是说，`population_active_mask` 已存在，但还不是主控制 universe。

### 1.2 `pseudo_branch/spgm/score.py`
当前 `build_spgm_importance_score(...)` 产出：
- `importance_score`
- `ranking_score`
- `state_score`
- `participation_score`

其中：
- `state_score = 0.45 * state_density_norm + 0.35 * population_support_norm + 0.20 * depth_score`
- `participation_score = 0.80 * importance_raw + 0.20 * population_support_norm`

这不是 BRPO paper 里的统一分数：
\[
S_i = \alpha \hat s_i^{(z)} + (1-\alpha) \hat s_i^{(\rho)}
\]

所以当前分数层是“工程化多分数 bundle”，不是 paper unified score semantics。

### 1.3 `pseudo_branch/spgm/manager.py`
当前 manager 的主动作有两条：
- `deterministic_participation`：`state_score -> candidate quantile -> keep-top-k -> boolean mask`
- `deterministic_opacity_participation`：`participation_score -> candidate quantile -> linear floor mapping -> opacity_scale`

这和 BRPO 的：
\[
p_i^{drop} = r \cdot w_{cluster(i)} \cdot S_i
\]
\[
m_i \sim \text{Bernoulli}(1-p_i^{drop})
\]
\[
\alpha_i^{eff} = \alpha_i \cdot m_i
\]

不是同一个 action law。

### 1.4 `scripts/run_pseudo_refinement_v2.py`
当前 StageB 的真实顺序是：
1. render 当前 pseudo views
2. build pseudo loss / backward
3. `maybe_apply_pseudo_local_gating(...)`
4. 生成 `_spgm_participation_render_mask` / `_spgm_participation_opacity_scale`
5. 下一轮再消费

因此当前真实语义是：
- `iter t` 统计
- `iter t+1` 生效

它是 delayed controller，不是 BRPO 式 current-step opacity operator。

### 1.5 `pseudo_branch/local_gating/gating_schema.py` + CLI
当前配置轴仍围绕 legacy / deterministic path：
- `spgm_manager_mode` 只有 `summary_only / xyz_lr_scale / deterministic_participation / deterministic_opacity_participation`
- `spgm_state_candidate_quantile`
- `spgm_state_participation_keep_*`
- `spgm_state_opacity_floor_*`

也就是说，CLI 和 dataclass 仍然把 “candidate quantile / keep ratio / floor mapping” 当主调参对象，而不是直接表达：
- control universe
- score semantics
- action semantics
- timing semantics

---

## 2. 这轮 direct BRPO G~ 的目标态

这轮要对齐的不是“更像 BRPO 的感觉”，而是以下四个明确语义轴。

### 2.1 Universe 对齐
目标：从 pseudo accepted active set 转向 scene-level population universe。

工程定义：
- G~ 主控制域从 `active_mask` 切到 `population_active_mask`
- `active_mask` 保留为 diagnostics / backward-compatibility field，不再作为 direct BRPO path 的主 action 域

### 2.2 Score 对齐
目标：action 直接来自 BRPO unified score `S_i`。

工程定义：
- 直接实现 BRPO unified score 路径：
\[
S_i = \alpha \hat s_i^{(z)} + (1-\alpha)\hat s_i^{(\rho)}
\]
- `state_score` / `ranking_score` 保留为 legacy diagnostics，不再给 direct BRPO action 用

### 2.3 Action law 对齐
目标：从 deterministic selector / floor mapping 转向 stochastic Bernoulli opacity masking。

工程定义：
- 直接实现：
\[
p_i^{drop} = r \cdot w_{cluster(i)} \cdot S_i
\]
\[
m_i \sim \text{Bernoulli}(1-p_i^{drop})
\]
\[
\alpha_i^{eff} = \alpha_i \cdot m_i
\]
- 不再先做 candidate quantile，再做 top-k keep，再做 floor mapping
- 不做永久 prune，只在 pseudo branch render 中施加 sampled opacity participation

### 2.4 Timing 对齐
目标：从 delayed action 转向 current-step action。

工程定义：
- 同一轮 iter 内完成：
  1. probe render
  2. collect stats
  3. build direct BRPO action
  4. loss render with current-step action
  5. backward

而不是：
- `iter t` render/backward
- `iter t` 决策
- `iter t+1` 才消费

---

## 3. 工程策略：不是继续补旧路，而是加一条平行 direct path

### 3.1 保留旧 deterministic 路线，但降级为 control
保留原因：
- compare 需要 control
- 旧实验结果需要可复现
- 直接覆盖旧 mode 容易让文档和历史产物失配

因此：
- `deterministic_participation`
- `deterministic_opacity_participation`
- delayed history fields

都保留，但只作为 legacy path。

### 3.2 direct BRPO path 必须显式命名，不能偷改旧语义
这轮不要把旧 mode 悄悄改得“更像 BRPO”。

应该显式新增 direct 语义轴，例如：
- `spgm_score_semantics = legacy_v1 | brpo_unified_v1`
- `spgm_control_universe = active | population_active`
- `spgm_action_semantics = deterministic_participation | deterministic_opacity | stochastic_bernoulli_opacity`
- `spgm_timing_mode = delayed | current_step_probe_loss`

这样 compare 时，才能明确知道现在跑的是哪条语义路径。

### 3.3 direct BRPO path 第一版只改 pseudo branch
第一版 direct BRPO 只作用于：
- pseudo render
- pseudo loss

明确不改：
- real branch render
- permanent prune / densify policy
- topology / T1 / A1 contract

否则变量会混在一起，无法判断 G~ 语义本身是否 landing。

---

## 4. 代码改动规划（按文件）

## 4.1 `pseudo_branch/local_gating/gating_schema.py`

### 要新增的配置轴
在 `PseudoLocalGatingConfig` 中新增 direct BRPO 所需字段：
- `spgm_score_semantics: str = 'legacy_v1'`
- `spgm_control_universe: str = 'active'`
- `spgm_action_semantics: str = 'deterministic_participation'`
- `spgm_timing_mode: str = 'delayed'`
- `spgm_drop_rate_global: float = 0.05`
- `spgm_cluster_weight_near: float = 1.0`
- `spgm_cluster_weight_mid: float = 1.0`
- `spgm_cluster_weight_far: float = 1.0`
- `spgm_sample_seed_mode: str = 'per_iter'`

### 要保留但标记为 legacy 的字段
以下字段继续保留，但在 direct BRPO path 下不作为主语义：
- `spgm_state_candidate_quantile`
- `spgm_state_participation_keep_*`
- `spgm_state_opacity_floor_*`
- `spgm_state_base_scale_*`

### 代码要求
- `uses_spgm()` 不应假设只有旧 deterministic mode；只要 direct BRPO path 开启，也应返回 True。
- `as_dict()` 应完整包含 direct path 字段，方便写进 `stageB_history.json`。

---

## 4.2 `scripts/run_pseudo_refinement_v2.py`

这是 direct BRPO G~ 改动最大的文件。

### 4.2.1 扩展 CLI
在 `parse_args()` 中新增对应 CLI：
- `--pseudo_local_gating_spgm_score_semantics`
- `--pseudo_local_gating_spgm_control_universe`
- `--pseudo_local_gating_spgm_action_semantics`
- `--pseudo_local_gating_spgm_timing_mode`
- `--pseudo_local_gating_spgm_drop_rate_global`
- `--pseudo_local_gating_spgm_cluster_weight_near`
- `--pseudo_local_gating_spgm_cluster_weight_mid`
- `--pseudo_local_gating_spgm_cluster_weight_far`
- `--pseudo_local_gating_spgm_sample_seed_mode`

`build_pseudo_local_gating_cfg()` 也要同步传进去。

### 4.2.2 拆开当前 `maybe_apply_pseudo_local_gating(...)`
当前这个函数把：
- stats collection
- score building
- update policy
- manager action
- diagnostics packaging

全揉在一起，不适合 current-step direct BRPO。

应拆成三类函数（即使第一版内部复用旧逻辑，也要先把 API 边界拆开）：
1. `collect_pseudo_local_gating_stats(...)`
2. `build_pseudo_local_gating_decision(...)`
3. `export_pseudo_local_gating_action(...)`

### 4.2.3 新增 StageB current-step 两段式结构
当 `spgm_timing_mode == 'current_step_probe_loss'` 时，StageB pseudo branch 改成：

1. `run_stageB_pseudo_probe_pass(...)`
   - 对当前 sampled pseudo views 做 probe render
   - probe render 默认不带 action（或带 neutral all-ones action）
   - 收集 visibility / support / depth / density 所需统计

2. `build_stageB_current_step_spgm_action(...)`
   - 基于 probe pass 结果计算 direct BRPO unified score
   - 生成 current-step Bernoulli opacity mask

3. `run_stageB_pseudo_loss_pass(...)`
   - 对同一批 sampled pseudo views，使用 current-step sampled opacity mask 再 render 一次
   - 用 loss render 的结果计算 pseudo loss / total loss / backward

### 4.2.4 StageB 主循环改造要求
当前的：
- `stageB_spgm_participation_render_mask`
- `stageB_spgm_participation_opacity_scale`

只适合 delayed mode。

在 direct current-step mode 下：
- action 不应再先存成 “下一轮状态” 再消费
- sampled mask / opacity scale 应是 **本轮局部变量**，只服务于本轮 loss render
- 下一轮最多保留 diagnostics summary，不应把 sampled action 直接沿用为下一轮先验动作

### 4.2.5 history / logging
`_init_gating_history_lists(...)`、`_append_gating_history(...)`、打印日志都要扩展，至少新增：
- `spgm_score_semantics`
- `spgm_control_universe`
- `spgm_action_semantics`
- `spgm_timing_mode`
- `spgm_unified_score_mean`
- `spgm_unified_score_p50`
- `spgm_drop_prob_mean`
- `spgm_drop_prob_p50`
- `spgm_drop_prob_mean_near/mid/far`
- `spgm_sampled_keep_ratio`
- `spgm_sampled_keep_ratio_near/mid/far`
- `spgm_probe_active_ratio`
- `spgm_losspass_active_ratio`

必须显式区分：
- probe-pass stats
- decision stats
- loss-pass effective action

否则 current-step landing 了也无法证明 action 真作用到了当前 loss landscape。

---

## 4.3 `pseudo_branch/spgm/stats.py`

### 当前问题
`collect_spgm_stats(...)` 已有 `population_active_mask`，但主控制逻辑和归一化仍然默认围绕 `active_mask`。

### direct BRPO 改造要求
新增可选参数，例如：
- `control_universe: str = 'active'`

函数返回中显式增加：
- `control_mask`
- `control_mask_name`

然后：
- legacy path 下：`control_mask = active_mask`
- direct BRPO path 下：`control_mask = population_active_mask`

### 注意事项
- `support_count`、`population_support_count` 都保留；
- `accepted_view_count` 仍保留；
- 但 score builder 不应再被迫只吃 `active_mask`。

也就是说，这一层的目标不是推翻现有统计，而是让 score/action 真正可以切到 population universe。

---

## 4.4 `pseudo_branch/spgm/score.py`

### 设计要求
不要继续把 direct BRPO action 建在 `state_score` 或 `participation_score` 上。

需要新增 direct BRPO score path，推荐两种写法二选一：

#### 方案 A：新增独立函数
新增：
- `build_spgm_brpo_unified_score(...)`

输出：
- `cluster_id`
- `depth_score`
- `density_score`
- `unified_score`
- `drop_prob_preclip`
- `drop_prob`

#### 方案 B：保留统一入口，但加语义分支
扩展 `build_spgm_importance_score(...)`，增加：
- `score_semantics='legacy_v1' | 'brpo_unified_v1'`

并在返回值里显式增加：
- `score_semantics_effective`
- `unified_score`

### direct BRPO score 的最低要求
1. `unified_score` 必须直接对应：
\[
S_i = \alpha \hat s_i^{(z)} + (1-\alpha)\hat s_i^{(\rho)}
\]
2. action 端必须直接使用 `unified_score`；
3. `state_score` / `ranking_score` 只保留为 diagnostics / legacy compare；
4. 所有归一化都要基于 `control_mask`，而不是写死 `active_mask`。

### 额外输出建议
为了 compare 可解释性，建议在返回值中额外输出：
- `unified_score_mean`
- `unified_score_p50`
- `unified_score_mean_near/mid/far`
- `density_entropy`
- `control_ratio`

---

## 4.5 `pseudo_branch/spgm/manager.py`

这是 direct BRPO action 的核心落点。

### 当前 legacy 逻辑
当前 `_build_participation_render_mask(...)` / `_build_opacity_participation_scale(...)` 都是：
- quantile candidate
- deterministic top-k 或 linear floor mapping

这套逻辑 direct BRPO path 下不再作为主语义。

### 需要新增的 direct BRPO action builder
推荐新增：
- `_build_stochastic_bernoulli_opacity_action(...)`

输入至少包括：
- `cluster_id`
- `control_mask`
- `unified_score`
- `drop_rate_global`
- `cluster_weights`
- `sample_seed` / `generator`

输出至少包括：
- `participation_keep_mask`
- `participation_opacity_scale`
- `drop_prob`
- `cluster_keep_ratio`
- `cluster_drop_prob_mean`

### direct BRPO action 公式
按当前 detailed analysis 里固化的公式，第一版直接写成：
\[
p_i^{drop} = r \cdot w_{cluster(i)} \cdot S_i
\]
\[
m_i \sim \text{Bernoulli}(1-p_i^{drop})
\]
\[
\alpha_i^{eff} = \alpha_i \cdot m_i
\]

工程要求：
1. `p_i^{drop}` 先 clamp 到 `[0, 1]`；
2. 非 `control_mask` 元素默认 `m_i = 1`；
3. `participation_opacity_scale = m_i.float()`；
4. 第一版只在 pseudo branch 生效；
5. sampled mask 要可追溯，至少要写 history summary。

### 不要做的事
- 不要在 direct BRPO mode 下继续走 `state_candidate_quantile`；
- 不要在 direct BRPO mode 下继续走 top-k keep；
- 不要把 Bernoulli sampling 包成 floor mapping 的微扰版；
- 不要 silently reuse `state_score` 作为 direct path 的 action score。

---

## 4.6 `third_party/S3PO-GS/gaussian_splatting/gaussian_renderer/__init__.py`

### 当前可复用事实
当前 renderer 已支持：
- `mask`
- `opacity_scale`

因此 direct BRPO 第一版 **不需要算法级重写 renderer**。

### 需要做的只是确认 contract
1. `opacity_scale` 传入全长 `[N]` 张量时，0/1 mask 语义正常；
2. mask branch / non-mask branch / opacity_scale branch 的返回 contract 一致；
3. `visibility_filter` 语义不会因为 sampled opacity mask 破坏 full-length contract。

也就是说，这个文件在 direct BRPO 里更像“验证和少量 debug 增强”，不是主战场。

---

## 5. 推荐执行顺序（直接对齐版）

### Phase G-BRPO-0：先把 direct path 接口搭出来
目标：不是调参，而是把 direct BRPO 四个语义轴变成代码里显式可选的配置。

本阶段必须落地：
- config/CLI 新增 direct path 字段；
- stats / score / manager / timing 的 API 分轴；
- legacy path 行为不变。

### Phase G-BRPO-1：落地 `population_active + unified_score + Bernoulli opacity`（先 delayed smoke）
目标：即使 timing 还没 current-step，也先把 direct BRPO 的 universe / score / action law 三层落地。

注意：
- 这一阶段不是 landing compare；
- 只是确认 direct BRPO action 已经不是 `state_score + top-k + floor`。

### Phase G-BRPO-2：落地 current-step probe → decide → loss render
目标：让 direct BRPO sampled opacity mask 真正作用到 **同一轮** pseudo loss render。

这是 direct alignment 的关键阶段。

### Phase G-BRPO-3：formal compare
本轮 formal compare 不再围绕 C0-2 微调，而是围绕语义分层：
1. `summary_only`（control）
2. `deterministic_participation`（legacy control）
3. `deterministic_opacity_participation`（legacy opacity control）
4. `direct_brpo_v1_delayed`（仅 universe/score/action 对齐）
5. `direct_brpo_v1_current_step`（四层全部对齐）

这样 compare 才能回答：
- 是 action law 本身无效；
- 还是 current-step 才是决定性差异；
- 还是 population universe 才是真正关键。

---

## 6. 这轮明确不做的事

1. 不再把 `candidate quantile` / `floor mapping` sweep 当主线；
2. 不再把 “把 opacity 从 0.98 压到 0.96” 当作 BRPO alignment 本体；
3. 不把 direct BRPO path 悄悄塞进旧 deterministic mode；
4. 不把 A1 / T1 / topology 绑进同一 patch；
5. 不在没有 current-step 的情况下宣称 G~ 已对齐 BRPO。

---

## 7. 验收标准

只有同时满足以下条件，G~ direct BRPO path 才能称为“实现到位”：

1. `control_universe` 真正切到 `population_active_mask`；
2. action score 真正来自 `unified_score S_i`，不是 `state_score`；
3. action 真正是 `Bernoulli opacity masking`，不是 deterministic top-k / floor；
4. sampled opacity mask 真正作用于 **current-step** pseudo loss render；
5. `stageB_history.json` 中能看到 direct path 的 `drop_prob` / sampled keep ratio / timing mode / score semantics / control universe；
6. formal compare 中 direct BRPO arm 可以与旧 control 同场对照。

少任何一条，都不应把它叫作 “G~ 已对齐 BRPO”。

---

## 8. 第一 patch 应该做什么

用户既然已经明确要求 direct alignment，这里的第一 patch 不应再是 C0-2 小修补。

第一 patch 的建议内容是：
1. 在 `gating_schema.py` / CLI 中加入 direct BRPO 四个语义轴；
2. 在 `stats.py` 中引入 `control_mask` 概念；
3. 在 `score.py` 中落地 `brpo_unified_v1`；
4. 在 `manager.py` 中新增 `stochastic_bernoulli_opacity` action builder；
5. 在 `run_pseudo_refinement_v2.py` 中把 stats / decision / action API 拆开。

这个 patch 的目标不是 compare 赢，而是：

> **先让 direct BRPO G~ 在代码语义上真实存在。**

第二 patch 再把 current-step probe/loss render 接起来，并做第一次 direct BRPO short compare。

---

## 9. 一句话 handoff

> **从现在开始，G~ 的主线不是再修 delayed deterministic opacity，而是新建一条 direct BRPO path：`population_active_mask -> unified score S_i -> Bernoulli opacity masking -> current-step loss render`。旧 deterministic 路线全部降级为 compare control。**

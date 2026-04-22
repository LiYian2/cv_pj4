# B3 opacity participation / population manager 工程落地文档

> 上位文档：`BRPO_A1_B3_reexploration_master_plan.md`
> 理论约束：`part3_BRPO_A1_B3_vs_BRPO_detailed_analysis.md`
> 目标：把 current B3 从 `state_score -> next-step boolean render selection` 推向更 BRPO-like 的 `importance score -> opacity participation manager`。

---

## 0. 结论先说

这轮 B3 的关键不是继续扫 keep-ratio，而是改三件根东西：

1. **动作变量**：从 boolean `participation_render_mask` 改成 continuous opacity attenuation；
2. **动作 score**：从 `state_score` 收回到更 BRPO-like 的 participation score；
3. **控制 universe**：从 `active_mask` 推到 `population_active_mask`。

这三者里，最优先的是 **动作变量**。

---

## 1. 当前代码事实

### 1.1 stats 层

`pseudo_branch/spgm/stats.py::collect_spgm_stats(...)` 当前输出：

- `support_count`
- `population_support_count`
- `depth_value`
- `density_proxy`
- `struct_density_proxy`
- `active_mask = support_count > 0`
- `population_active_mask = population_support_count > 0`

analysis 已经指出：

> current stats 已经有 population summary，但主逻辑仍主要锚在 `active_mask`。

### 1.2 score 层

`pseudo_branch/spgm/score.py::build_spgm_importance_score(...)` 当前显式区分：

- `importance_score`
- `ranking_score`
- `state_score`

其中当前 manager 真正拿去做 participation action 的是：

- `state_score`

而不是更接近 paper unified score 的：

- `importance_raw / importance_score`

### 1.3 manager 层

`pseudo_branch/spgm/manager.py` 当前 `deterministic_participation` 路径是：

- `_build_state_candidate_mask(...)`
- `_build_participation_render_mask(...)`
- cluster-wise quantile candidate selection
- keep-top-k by `state_score`
- 得到 boolean `participation_render_mask`

### 1.4 run loop 层

`run_pseudo_refinement_v2.py` 当前在 pseudo render 时：

- 读取 `stageB_spgm_participation_render_mask`
- `render(..., mask=current_pseudo_render_mask)`

也就是说，当前 B3 已进入 pre-render control，但动作对象仍然是 **select / drop**，不是 attenuation。

---

## 2. 设计原则

### 2.1 保留 current manager modes，不覆盖旧实现

必须保留：

- `summary_only`
- `xyz_lr_scale`
- `deterministic_participation`

新的主线应新增 mode，例如：

- `deterministic_opacity_participation`
- 之后必要时再加 `stochastic_opacity_participation`

### 2.2 先做 deterministic opacity，不直接跳 stochastic

原因：

- 现在最大的差异是 action variable 错位；
- stochastic 只会再引入方差，掩盖 action law 本身；
- 先把 deterministic opacity path 站稳，才有资格讨论 Bernoulli。

### 2.3 selector / ranking / diagnostics 不必同时废掉

当前 `state_score`、`ranking_score`、`update_policy` 都还有工程价值：

- `ranking_score` 仍可给 selector / diagnostics；
- `state_score` 仍可给状态日志 / 诊断；
- 但 participation action 不应继续主要依赖它。

---

## 3. 建议拆成三个阶段

## B3-O0：先把 participation score 显式拆出来

### 目标

在 `score.py` 里把“给 participation action 用的 score”显式命名，不再借用 `state_score`。

### 建议新增对象

在 `build_spgm_importance_score(...)` 里新增：

- `participation_score`
- `participation_score_mean`
- `participation_score_p50`

第一版建议：

- 以 `importance_raw` / `importance_score` 为核心；
- 允许再乘轻度 population term；
- 但不要直接等于 `state_score`。

建议原则：

- `ranking_score`: 给 update permission / selector 用
- `state_score`: 给 diagnostics 用
- `participation_score`: 给 opacity action 用

### 为什么先做这个

因为只有这样，后面 manager 才能明确回答：

> 当前被 attenuate 的，是“scene importance 低”的点，还是“当前状态低”的点？

---

## B3-O1：把动作变量从 boolean render mask 改成 opacity attenuation

### 目标

新增 manager mode：

- `deterministic_opacity_participation`

其语义不再是：

- drop or keep

而是：

- 为每个 Gaussian 生成一个 `participation_scale in [0, 1]`

推荐形态：

- near/mid/far cluster 仍可有 cluster-wise base decay
- 但最终动作是 continuous attenuation，不是 top-k selector

### 建议 manager 改造方式

在 `manager.py` 中新增函数，例如：

- `_build_opacity_participation_scale(...)`

输入：

- `cluster_id`
- `active_mask` or `population_active_mask`
- `participation_score`
- cluster-wise hyperparameters

输出：

- `participation_opacity_scale`，shape = `[N]`

建议规则：

- 非 active / non-population 点默认 scale = 1.0 或 no-op；
- 低 participation score 的点衰减更强；
- 第一版 deterministic，不采样；
- 不做硬 drop，除非后续明确需要 clip 到 very small epsilon。

### renderer 侧接线

当前 `render(..., mask=...)` 已能吃 boolean mask。
新的主线需要新增而不是替换：

- `render(..., opacity_scale=...)`

或更中性地：

- `render(..., participation_scale=...)`

renderer 中应在 opacity 进入 rasterization / compositing 前乘上该 scale。

### 注意

这一步是 B3 的主变更点，必须单独做 smoke / correctness check。

---

## B3-O2：把 control universe 从 `active_mask` 推到 `population_active_mask`

### 目标

让 participation control 不再只盯 accepted pseudo active set，而开始更接近 scene population manager。

### 当前问题

`active_mask = support_count > 0` 让当前 B3 很容易变成：

- 谁这轮 pseudo support 少，谁就更容易被处理。

这更像 local visibility controller，而不是 paper 里的 scene population manager。

### 建议做法

分两步走：

1. B3-O2a：score 统计仍以 active set 为主，但 manager action 开始看 `population_active_mask`
2. B3-O2b：score 统计本身也切到 `population_active_mask` 主 universe

这样做的原因是：

- 一次性把 score 统计和 action universe 同时切掉，太难定位问题；
- 分两步能分辨“统计问题”还是“action 域问题”。

---

## 4. 不建议保留为主线的旧路径

### 4.1 `state_score -> keep-top-k -> boolean render mask`

这条路径不是不能保留，而是：

- 作为 archived compare object 保留；
- 不再作为 current B3 主线。

### 4.2 继续只扫 keep ratios

继续扫：

- `state_participation_keep_near/mid/far`

只能回答“硬 selector 削弱到什么程度不那么伤”，
回答不了“BRPO-style participation 到底有没有价值”。

所以 keep-ratio sweep 只能作为旧路径的收尾 compare，不是新路径的主线。

---

## 5. 具体代码修改点

### 5.1 `pseudo_branch/spgm/score.py`

新增 / 改造：

- `participation_score`
- `participation_score_mean`
- `participation_score_p50`

并把 current `importance_raw / importance_score / state_score` 的职责写清楚。

### 5.2 `pseudo_branch/spgm/manager.py`

新增 mode：

- `deterministic_opacity_participation`

新增返回对象：

- `participation_opacity_scale`
- cluster-wise attenuation diagnostics

保留现有：

- `participation_render_mask`

但仅作为旧路线 compare support。

### 5.3 renderer

修改：

- `gaussian_renderer.render(...)`

新增参数：

- `opacity_scale` 或 `participation_scale`

注意保证：

- unmasked / masked / scaled branches 的返回 contract 完全一致；
- `visibility_filter` 仍是 full-length semantics；
- 不重蹈之前 mask branch contract mismatch 的坑。

### 5.4 `run_pseudo_refinement_v2.py`

新增：

- current opacity-scale plumbing
- 日志项：`spgm_participation_score_*`, `spgm_opacity_scale_*`

旧 boolean mask plumbing 保留，以方便 A/B compare。

---

## 6. compare 顺序

### Compare B3-C0：score diagnostics only

不改 action，只记录：

- `importance_score`
- `state_score`
- new `participation_score`
- 三者的 cluster stats / rank correlation / disagreement

目的：先确认 new participation score 确实不是 state-score 换名。

### Compare B3-C1：delayed deterministic opacity vs delayed boolean mask

在不改 timing 的前提下，只换动作变量：

1. `summary_only`
2. old boolean `deterministic_participation`
3. new delayed `deterministic_opacity_participation`

回答问题：

- 仅仅把动作变量从 hard selector 改成 opacity attenuation，是否就能减轻 weak-negative？

### Compare B3-C2：population universe shift

在 delayed deterministic opacity 站住后，再比：

- active universe
- population universe

不要和 timing 改造绑在一起。

---

## 7. 暂不做的事

1. 不直接上 stochastic Bernoulli masking；
2. 不先把 `state_score` 删掉；
3. 不一次性把 manager / renderer / timing 三层一起改完再测；
4. 不把旧 boolean path 删除。

---

## 8. 一句话版执行目标

> **先在 score 层显式分出 participation_score，再把 manager 动作变量从 boolean selector 改成 deterministic opacity attenuation，最后再把 control universe 从 active_mask 推到 population_active_mask。**

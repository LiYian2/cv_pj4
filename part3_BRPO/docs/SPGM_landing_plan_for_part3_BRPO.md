# Part3 BRPO：SPGM 原理分析与工程落地规划

> 面向当前 `LiYian2/cv_pj4` 仓库的 Part3 refine 代码流整理。
> 目标：把 BRPO 论文中的 **SPGM (Scene Perception Gaussian Management)** 以**渐进、可调试、可复用**的方式接入你当前的 `StageA.5 / StageB` refine 框架，供后续 agent 继续细化工程规划与执行。

---

## 0. 执行摘要

当前 repo 已经具备以下三个关键基础：

1. **上游 signal 已接通**：`build_brpo_v2_signal_from_internal_cache.py` 已经在 `prepare_root/signal_v2` 下生成 BRPO 风格的隔离 signal，包括 RGB correspondence mask v2 与 depth supervision v2。
2. **Gaussian 微调入口已存在**：`pseudo_branch/gaussian_param_groups.py` 里的 `build_micro_gaussian_param_groups()` 已经支持 `xyz` / `xyz_opacity` 两类微量参数组。
3. **后向级别的 per-Gaussian 控制点已存在**：`run_pseudo_refinement_v2.py` 当前已经走通了 `evaluate_sampled_views_for_local_gating -> build_visibility_weight_map -> apply_gaussian_grad_mask` 的 pseudo-side local gating 回路。

因此，**SPGM 的最佳落点不是上游 signal 生产器，也不是再做一个新的 view gate，而是把当前的“view-conditioned grad mask”升级为“Gaussian-state-aware grad management”。**

换句话说：

- 现在的 local gating 主要回答：**哪些 pseudo view 值得进入更新**；
- BRPO 的 SPGM 主要回答：**哪些 Gaussian 在当前 scene state 下应该被信任、被减弱、被保守更新**。

所以当前最合理的工程方向是：

- **保留现有 view-level signal gate** 作为样本过滤器；
- **新增 SPGM 模块**，在 backward 之后、optimizer.step 之前，把 `accepted pseudo views` 映射成 **per-Gaussian importance score / grad weight**；
- 第一版先做 **deterministic soft keep**，不要一开始就做论文式 stochastic drop；
- 第一阶段先验证 `StageA.5 + xyz-only`，第二阶段再推到 `StageB`，第三阶段再考虑 `xyz+opacity` 与 stochastic attenuation。

---

## 1. 现在的代码流，和 SPGM 应该插在哪

### 1.1 当前 Part3 refine 主线（按 repo 现状整理）

从当前 `STATUS.md / DESIGN.md` 与代码路径来看，当前主线已经不是旧的 train-mask-only 参数调试，而是：

```text
part2 full rerun
→ internal cache
→ internal prepare
→ difix / fusion
→ isolated signal_v2
→ run_pseudo_refinement_v2.py
   ├─ StageA   （主要稳 pose / exposure，不更新 Gaussian）
   ├─ StageA.5 （开始更新 Gaussian）
   └─ StageB   （pseudo + real branch 的 joint refine）
```

其中关键点：

- `StageA-only` 当前不会更新 Gaussian，因此 `refined_gaussians.ply` 与输入 `BASE_PLY` 一致，StageA-only 的 replay 只能当 sanity check，不是 Gaussian refine 判优指标。
- 真正会改 PLY / 高斯图的是 **StageA.5 / StageB**。
- 当前的 `local Gaussian gating / subset refine` 已经成为主线判断的一部分，而不是旁路线。

### 1.2 当前已接通的 pseudo-side local gating 回路

在 `run_pseudo_refinement_v2.py` 中，当前已经有如下导入与调用关系：

- `evaluate_sampled_views_for_local_gating`
- `build_visibility_weight_map`
- `apply_gaussian_grad_mask`
- `build_iteration_gating_summary`

而 `maybe_apply_pseudo_local_gating()` 当前流程是：

```text
sampled_views
→ evaluate_sampled_views_for_local_gating(...)
→ build_visibility_weight_map(...)
→ apply_gaussian_grad_mask(...)
```

也就是：

1. 对 sampled pseudo views 做 view-level gate；
2. 用 accepted pseudo views 的 `visibility_filter` 构造一个 visibility union / weight map；
3. 把这个 weight map 直接乘到 Gaussian grad 上。

### 1.3 现有 local gating 的本质

当前 local gating 的核心是：

- **view-side** 统计量：
  - `target_depth_verified_ratio`
  - `rgb_confidence_nonzero_ratio`
  - `target_depth_render_fallback_ratio`
  - `mean_abs_rel_correction_verified`
- **Gaussian-side** 执行动作：
  - 对 accepted pseudo views 的 `visibility_filter` 取 union / weight
  - 用 `apply_gaussian_grad_mask()` 在 backward 后直接裁剪 / 缩放 Gaussian grad

这说明你现在已经有一个非常合适的 **SPGM 插入点**：

```text
backward 后
view gate 之后
visibility union 之前 / 或替代 visibility union
optimizer.step 之前
```

### 1.4 SPGM 应该插在哪

**建议把 SPGM 插在 `evaluate_sampled_views_for_local_gating()` 和 `apply_gaussian_grad_mask()` 之间。**

也就是把当前：

```text
gate_results
→ build_visibility_weight_map
→ apply_gaussian_grad_mask
```

升级为：

```text
gate_results
→ collect_spgm_stats
→ build_spgm_importance_score
→ build_spgm_grad_weights
→ apply_gaussian_grad_mask
```

### 1.5 为什么插这里最合适

因为你现在已经具备：

- pseudo view 级别的 sample filtering；
- render package 里的 `visibility_filter`；
- backward 后可直接操作的 Gaussian grad；
- StageA.5 / StageB 的局部 Gaussian optimizer。

这意味着：

- 上游 signal builder 不需要推翻；
- optimizer 不需要重写；
- loss builder 不需要重写；
- 只要把“怎么生成 per-Gaussian grad weight”从 `visibility union` 升级成 `SPGM score -> policy` 即可。

这也是当前 repo 最低风险、最高复用率的落地方式。

---

## 2. 按 BRPO 原理，SPGM 可以拆成 4 个模块落地

下面把 BRPO 的 SPGM 拆成四个工程模块，并明确它们和你当前代码流怎么接。

---

### 模块 A：Pseudo-view gate（保留，但降级为样本过滤器）

#### A.1 模块职责

这部分不是 SPGM 本体，但必须保留。

它负责：

- 过滤明显坏的 pseudo view；
- 给 accepted pseudo views 一个 coarse sample weight；
- 作为后续 SPGM 统计的输入过滤器。

#### A.2 当前 repo 中已有实现

对应：

- `pseudo_branch/local_gating/signal_gate.py`
- `evaluate_sampled_views_for_local_gating()`
- `PseudoLocalGatingConfig`

当前 gate 主要基于：

- `verified_ratio`
- `rgb_mask_ratio`
- `fallback_ratio`
- `min_correction`

并支持 hard / soft gate。

#### A.3 在新 SPGM 体系中的角色

保留它，但明确定位为：

> **view-quality filter**，而不是 Gaussian management。

也就是说，SPGM 的输入不是所有 sampled views，而应该是：

```text
accepted pseudo views only
```

#### A.4 数学表达

对每个 pseudo view `v`，由现有 signal gate 产生一个 coarse weight：

\[
w_v \in [0,1]
\]

硬 gate 可以视作：

\[
w_v \in \{0,1\}
\]

软 gate 可以视作：

\[
w_v \in (0,1]
\]

这个 `w_v` 后面会进入 per-Gaussian support 统计。

#### A.5 与现有代码流怎么接

保持不变：

```text
sampled_views
→ evaluate_sampled_views_for_local_gating(...)
→ gate_results
```

然后把 `gate_results` 不再只喂给 `build_visibility_weight_map()`，而是喂给 SPGM 的 stats 模块。

---

### 模块 B：Gaussian statistics extractor（新增）

#### B.1 模块职责

这部分是 SPGM 的第一层，也是你现在 repo 里真正缺失的关键层。

它要做的是：

- 从 accepted pseudo views 中，提取 **per-Gaussian scene statistics**；
- 把“哪些 Gaussian 被 pseudo 看到 / 支持”显式编码出来；
- 产出后续 score 计算所需的基础量。

#### B.2 需要输出哪些统计量

建议第一版输出：

- `support_count: Tensor[N]`
- `depth_value: Tensor[N]`
- `density_proxy: Tensor[N]`
- `active_mask: BoolTensor[N]`

其中 `N` 是 Gaussian 数量。

#### B.3 数学原理

设当前 iter 的 accepted pseudo view 集合为 `A_t`，第 `i` 个 Gaussian 记为 `g_i`，其中心为 `x_i`。

##### (1) support count

定义第 `i` 个 Gaussian 的 pseudo 支持数：

\[
c_i = \sum_{v \in A_t} w_v \, \mathbf{1}[g_i \in \mathrm{vis}(v)]
\]

含义：

- 如果一个 Gaussian 只在极少数 accepted pseudo view 中被看到，它的支持度低；
- 如果它能在多个 accepted pseudo view 中稳定可见，则支持度高。

这一步不是 BRPO 论文原式，但非常适合你当前系统，因为你当前的核心问题就是 pseudo supervision 太弱、太局部、太稀疏。

##### (2) depth value

对每个被某个 accepted pseudo view 看见的 Gaussian，定义其相机坐标深度：

\[
z_i^{(v)} = (R_v x_i + t_v)_z
\]

然后对所有可见 view 做加权聚合。建议第一版用**加权中位数**：

\[
z_i = \mathrm{wmedian}_{v \in A_t, g_i \in \mathrm{vis}(v)} z_i^{(v)}
\]

为什么建议中位数而不是均值：

- 对偶发异常 pseudo view 更鲁棒；
- 更适合当前 sparse + weak pseudo support 的设定。

##### (3) density proxy

BRPO 论文用的是 density + entropy 的 Gaussian 管理思想，但你当前 repo 并没有现成暴露一个论文同款 density field。

因此第一版建议用工程近似：

\[
\rho_i = \tilde{\alpha}_i \cdot \tilde{c}_i
\]

其中：

- `\tilde{\alpha}_i`：归一化后的 opacity / sigmoid(opacity)
- `\tilde{c}_i`：归一化后的 support count

如果以后能稳定拿到 Gaussian scale / covariance，再升级成：

\[
\rho_i = \frac{\tilde{\alpha}_i}{\mathrm{vol}_i + \varepsilon} \cdot \tilde{c}_i
\]

也就是 volume-normalized density。

##### (4) active mask

定义 active Gaussian 集：

\[
\mathcal{G}_{\text{active}} = \{ i \mid c_i > 0 \}
\]

对应实现里就是：

- 被至少一个 accepted pseudo view 看见；
- 才进入后续 SPGM score / policy。

#### B.4 与现有代码流怎么接

新增：

```text
part3_BRPO/pseudo_branch/spgm/stats.py
```

建议接口：

```python
collect_spgm_stats(
    sampled_views,
    gate_results,
    render_packages,
    gaussians,
    device,
) -> dict
```

输出：

```python
{
    "support_count": Tensor[N],
    "depth_value": Tensor[N],
    "density_proxy": Tensor[N],
    "active_mask": BoolTensor[N],
}
```

接到当前代码流时，应替代 / 前置于 `build_visibility_weight_map()`。

---

### 模块 C：Depth partition + density entropy scorer（新增，SPGM 核心）

#### C.1 模块职责

这部分是 BRPO-SPGM 的数学核心。

它的目标是：

- 把 per-Gaussian 统计量转成 **importance score**；
- 实现 BRPO 风格的 depth-aware + density-aware scene perception；
- 把“哪些 Gaussian 更值得信、哪些更该保守更新”编码成连续分数。

#### C.2 子模块 1：depth partition

BRPO 论文先做的是 **1D optimal-transport-inspired depth partition**。

核心思想：

- 在一维上，Wasserstein 距离和 quantile function 有天然联系；
- 因此把 Gaussian 的 depth 分布按 quantile 分成连续 depth clusters，是一个很合理的近似分层方式。

##### 建议工程实现

对当前 active Gaussian 的 `z_i` 做 `K=3` 个 quantile 分桶：

- near
- mid
- far

得到 cluster id：

\[
k_i \in \{0,1,2\}
\]

##### depth score

沿用 BRPO 论文思路，定义：

\[
s_i^{(z)} = 1 - \frac{z_i - z_{\min}}{z_{\max} - z_{\min} + \delta}
\]

解释：

- 越近的 Gaussian，depth score 越高；
- 越远的 Gaussian，depth score 越低。

这符合你当前 refine 的经验：远处和稀薄区域更容易被 pseudo 弱信号误导。

#### C.3 子模块 2：density entropy

BRPO 论文随后对 density distribution 做 histogram-based entropy 估计。

设对 `\rho_i` 做直方图离散化，得到 `B` 个 bin，离散概率为 `p_b`，则归一化 Shannon entropy：

\[
\bar H(\rho) = -\frac{1}{\log B} \sum_{b=1}^{B} p_b \log(p_b + \varepsilon)
\]

其中：

- `B`：bin 数，第一版建议 32
- `\varepsilon`：数值稳定项

##### 解释

- `\bar H` 低：density 分布集中
- `\bar H` 高：density 分布更均匀 / 更散

#### C.4 子模块 3：density score

按 BRPO 论文思路构造 entropy-aware density score：

\[
\hat s_i^{(\rho)} = \tilde{\rho}_i (1 - \beta \bar H) + \gamma \bar H
\]

其中：

- `\tilde{\rho}_i`：归一化 density proxy
- `\beta, \gamma \in [0,1]`
- 第一版建议：`\beta = 0.5, \gamma = 0.5`

#### C.5 子模块 4：unified importance score

再把 depth score 与 density score 融合：

\[
S_i = \alpha s_i^{(z)} + (1-\alpha) \hat s_i^{(\rho)}
\]

第一版建议：

- `\alpha = 0.5`

#### C.6 结合你当前系统，建议增加 support modifier

因为你当前系统最大的实际问题之一是 **pseudo support 稀疏**，所以建议在工程版里再乘一个 support modifier：

\[
m_i^{(\mathrm{sup})} = \frac{c_i}{\max_j c_j + \varepsilon}
\]

然后构造 effective importance：

\[
\bar S_i = S_i \cdot (m_i^{(\mathrm{sup})})^{\eta}
\]

第一版建议：

- `\eta = 0.5`

这样做的含义是：

- depth / density 再好，如果 pseudo support 只来自一个非常窄的局部 view，它也不应该被赋予太高的更新权限；
- 只有 structure 和 support 同时较好的 Gaussian，才应该得到更高 keep-weight。

#### C.7 与现有代码流怎么接

新增：

```text
part3_BRPO/pseudo_branch/spgm/score.py
```

建议接口：

```python
build_depth_partition(...)
build_density_entropy_score(...)
build_spgm_importance_score(...)
```

输出：

```python
{
    "cluster_id": Tensor[N],
    "depth_score": Tensor[N],
    "density_entropy": float,
    "density_score": Tensor[N],
    "importance_score": Tensor[N],
}
```

然后把 `importance_score` 传给 policy 模块。

---

### 模块 D：Cluster-aware attenuation policy（新增，执行层）

#### D.1 模块职责

这部分是 SPGM 的执行层。

它要做的是：

- 把 importance score 变成 optimizer 可直接消费的 **per-Gaussian weight**；
- 决定哪些高斯更新得更激进，哪些更保守，哪些几乎冻结；
- 对接你当前已经存在的 `apply_gaussian_grad_mask()`。

#### D.2 BRPO 论文原理

BRPO 论文使用的是 cluster-aware drop probability，并通过 stochastic masking 去衰减高斯。

但对你现在的 repo，我**不建议第一版就直接照论文做 Bernoulli stochastic opacity drop**。

#### D.3 为什么第一版不建议直接 stochastic drop

因为你当前第一问题是：

- `StageB-20iter` 有轻微正向窗口；
- `StageB-120iter` 会 regression；
- 现阶段最需要的是**稳定可解释**的 refine 动力学，而不是一上来就引入额外随机性。

而且你已经有现成的：

- `apply_gaussian_grad_mask()`

它本来就适合先做 deterministic grad weighting。

#### D.4 第一版推荐策略：deterministic soft keep

建议先把 `importance_score` 映射成 keep-weight，而不是 drop-probability。

定义：

\[
w_i^{\mathrm{keep}} = w_{\min} + (1 - w_{\min}) \cdot \bar S_i
\]

其中：

- `w_min = 0.05 ~ 0.1`

再乘 cluster keep policy：

\[
w_i = w_i^{\mathrm{keep}} \cdot u_{k_i}
\]

建议第一版 cluster policy：

- near: `u = 1.0`
- mid: `u = 0.8`
- far: `u = 0.6`

#### D.5 解释

这等价于：

- 近处、结构更强、支持更稳的 Gaussian，被更完整地保留梯度；
- 远处、支持差、density 结构弱的 Gaussian，被更强地衰减；
- `weights == 0` 只出现在 inactive 或明确 reject 的高斯上。

#### D.6 第二版再做什么

等 deterministic 版本稳定后，再考虑：

- stochastic keep / drop
- opacity-specific attenuation
- cluster-aware Bernoulli mask

#### D.7 与现有代码流怎么接

新增：

```text
part3_BRPO/pseudo_branch/spgm/policy.py
```

建议接口：

```python
build_spgm_grad_weights(
    importance_score,
    cluster_id,
    active_mask,
    mode="soft_keep",
    cluster_keep=(1.0, 0.8, 0.6),
    weight_floor=0.05,
) -> dict
```

输出：

```python
{
    "weights": Tensor[N],
    "stats": {...},
}
```

然后直接喂给：

```python
apply_gaussian_grad_mask(gaussians, weights=..., params_mode=...)
```

也就是说：

**SPGM 第一版不需要重写 optimizer，不需要改 loss builder，只要把 `weights` 的来源从 visibility union 换成 SPGM importance policy。**

---

## 3. 具体到 repo，应怎么改文件

下面是面向当前 repo 的**具体改动图**。

---

### 3.1 保留不动的文件

这些先不要改主逻辑：

- `part3_BRPO/scripts/build_brpo_v2_signal_from_internal_cache.py`
- `part3_BRPO/pseudo_branch/pseudo_fusion.py`
- `part3_BRPO/pseudo_branch/pseudo_loss_v2.py`
- `part3_BRPO/pseudo_branch/gaussian_param_groups.py`（第一阶段只复用，不扩大自由度）
- `part3_BRPO/pseudo_refine_scheduler.py`（第一阶段只接新 gating mode，不重构 schedule）

#### 原因

这些组件已经完成了：

- 上游 signal 生产；
- StageA.5 / StageB 的基础训练 wiring；
- micro Gaussian optimizer 的最小参数组。

当前主要矛盾已经不在这里，而在 `Gaussian-side refine policy`。

---

### 3.2 新增一个 `spgm/` 子包

建议新增：

```text
part3_BRPO/pseudo_branch/spgm/
├── __init__.py
├── stats.py
├── score.py
└── policy.py
```

#### 各文件职责

- `stats.py`：抽取 per-Gaussian state
- `score.py`：depth partition + density entropy + importance score
- `policy.py`：importance → grad weights

---

### 3.3 改 `local_gating` 的 schema / config

当前 `PseudoLocalGatingConfig` 已经有：

- `mode`
- `params`
- `min_verified_ratio`
- `min_rgb_mask_ratio`
- `max_fallback_ratio`
- `min_correction`
- `soft_power`

建议扩展为支持 SPGM：

```text
mode:
  off | hard | soft | spgm_soft | spgm_keep

spgm_num_clusters
spgm_alpha_depth
spgm_beta_entropy
spgm_gamma_entropy
spgm_support_eta
spgm_weight_floor
spgm_density_mode
spgm_cluster_keep_near
spgm_cluster_keep_mid
spgm_cluster_keep_far
```

#### 原则

- 不要把 `signal_gate.py` 直接改造成 SPGM 核心；
- 它仍然只负责 **view gate**；
- SPGM 是新分支，通过 mode 切换进入。

---

### 3.4 核心改动：`run_pseudo_refinement_v2.py`

这是最关键的接入点。

#### 当前伪流程

```python
sampled_ids = [int(v['sample_id']) for v in sampled_views]
gate_results = evaluate_sampled_views_for_local_gating(sampled_views, gating_cfg)

if gating_cfg.enabled():
    visibility_stats = build_visibility_weight_map(...)
    grad_stats = apply_gaussian_grad_mask(..., weights=visibility_stats['weights'])
```

#### 建议改成

```python
sampled_ids = [int(v['sample_id']) for v in sampled_views]
gate_results = evaluate_sampled_views_for_local_gating(sampled_views, gating_cfg)

if gating_cfg.mode in {"hard", "soft"}:
    visibility_stats = build_visibility_weight_map(...)
    grad_stats = apply_gaussian_grad_mask(..., weights=visibility_stats['weights'])

elif gating_cfg.mode in {"spgm_soft", "spgm_keep"}:
    spgm_stats = collect_spgm_stats(
        sampled_views=sampled_views,
        gate_results=gate_results,
        render_packages=render_packages,
        gaussians=gaussians,
        device=gaussians._xyz.device,
    )

    spgm_score = build_spgm_importance_score(
        depth_value=spgm_stats['depth_value'],
        density_proxy=spgm_stats['density_proxy'],
        support_count=spgm_stats['support_count'],
        active_mask=spgm_stats['active_mask'],
        ...
    )

    spgm_policy = build_spgm_grad_weights(
        importance_score=spgm_score['importance_score'],
        cluster_id=spgm_score['cluster_id'],
        active_mask=spgm_stats['active_mask'],
        ...
    )

    grad_stats = apply_gaussian_grad_mask(
        gaussians=gaussians,
        weights=spgm_policy['weights'],
        params_mode=gating_cfg.params,
    )
```

#### 需要新增的日志 / summary 字段

建议同步写入 history：

- `spgm_active_ratio`
- `spgm_support_mean`
- `spgm_support_p50`
- `spgm_depth_entropy`（可选）
- `spgm_density_entropy`
- `spgm_weight_mean`
- `spgm_weight_p10/p50/p90`
- `spgm_cluster_count_near/mid/far`

这样 agent 后面做 stabilization 规划时，就能判断：

- regression 是不是来自权重过平；
- 还是 active subset 太大 / 太小；
- 还是 far cluster 被保留得太多。

---

## 4. 分阶段怎么落地

建议严格分阶段，不要一次做 full SPGM。

---

### 阶段 1：StageA.5-only + xyz-only + deterministic SPGM

#### 目标

验证 SPGM 第一版是不是 **真实工作、而不是 no-op**。

#### 配置建议

- `signal_pipeline = brpo_v2`
- `stage_mode = stageA5`
- `stageA5_trainable_params = xyz`
- `pseudo_local_gating_mode = spgm_keep`
- `pseudo_local_gating_params = xyz`

#### 为什么先做这一版

因为：

- `StageA.5` 已经开始更新 Gaussian；
- 但还没有 real branch 混进来，变量少；
- 非常适合做最小验证。

#### 这一阶段的验收指标

- 能完整跑通，不 crash
- `spgm_active_ratio` 不为 0
- `spgm_weight_mean` 不接近 0 / 1 的极端
- `grad_keep_ratio_xyz` 和旧 gating 有可观察差异
- replay 相比旧 hard/soft gating 至少不更差

---

### 阶段 2：StageA.5-only + xyz+opacity

#### 目标

验证 SPGM 是否能支持更完整一点的 Gaussian 自由度。

#### 原则

- 不要一开始就把主要精力放在 `xyz+opacity`
- 先在 StageA.5 里验证 `xyz-only` 有效
- 再把 opacity 纳入同一 SPGM grad weight

#### 建议

第一版对 opacity 先共用与 xyz 相同的 weight，不要额外做 opacity-specific stochastic drop。

---

### 阶段 3：StageB + SPGM 只作用 pseudo branch

#### 目标

让 SPGM 正式进入 joint refine，但不误伤 real branch。

#### 关键原则

**SPGM 只作用 pseudo branch 的 backward 后梯度，不直接压 real branch。**

因为：

- real branch 是当前 map 的 anchor；
- 当前 StageB regression 的一个核心问题，是 pseudo-side 弱信号长跑会把图推坏；
- 所以需要压的是 pseudo-side 的 map 改动权限，而不是把 real anchor 一起削弱。

#### 工程要求

如果当前 StageB 的 backward 混在一起，不方便区分伪梯度和真实梯度，建议：

- 先拆成 pseudo step / real step 两段式；
- 或至少先在 pseudo loss backward 后立刻应用 SPGM，再处理 real loss。

---

### 阶段 4：再考虑 stochastic drop / full BRPO-like attenuation

#### 目标

在 deterministic soft keep 已经稳定的前提下，进一步向论文式 SPGM 靠拢。

#### 此时才考虑

- Bernoulli keep/drop
- opacity attenuation
- cluster-aware stochastic policy

#### 不建议提前做的原因

你当前系统最需要的是：

- 稳定的解释性；
- 可重复的 StageB 窗口分析；
- 明确知道 regression 是由谁导致。

而 stochastic 版本会提高调试难度，应该排到后面。

---

## 5. 建议直接分配给 agent 的模块任务

下面是最适合 agent 接手的拆分方式。

---

### Task 1：新增 `spgm/stats.py`

#### 目标

从 accepted pseudo views + render packages + gaussians 中提取 per-Gaussian state。

#### 输入

- `sampled_views`
- `gate_results`
- `render_packages`
- `gaussians`

#### 输出

- `support_count: Tensor[N]`
- `depth_value: Tensor[N]`
- `density_proxy: Tensor[N]`
- `active_mask: BoolTensor[N]`

#### 数学定义

- `c_i = Σ_v w_v 1[g_i ∈ vis(v)]`
- `z_i = weighted median_v z_i^(v)`
- `ρ_i = normalized(opacity_i) * normalized(c_i)`

#### 验收

- `active_mask.sum() > 0`
- support histogram 合理
- 与当前 accepted visibility union 同量级

---

### Task 2：新增 `spgm/score.py`

#### 目标

实现 BRPO-style depth partition + density entropy + unified importance score。

#### 输入

- `depth_value`
- `density_proxy`
- `support_count`
- `active_mask`

#### 输出

- `cluster_id`
- `depth_score`
- `density_entropy`
- `density_score`
- `importance_score`

#### 数学定义

- depth quantile split，`K=3`
- `s_z = 1 - (z-zmin)/(zmax-zmin+δ)`
- `Hbar = -(1/log B) Σ p_b log(p_b+ε)`
- `s_rho = rho_tilde * (1 - βHbar) + γHbar`
- `S = α s_z + (1-α) s_rho`
- `S_bar = S * support_norm^η`

#### 验收

- `importance_score` 在 `[0,1]`
- `density_entropy` 不为 NaN
- near / mid / far 三个 cluster 数量合理

---

### Task 3：新增 `spgm/policy.py`

#### 目标

把 importance score 转成可直接作用于 grad 的 per-Gaussian weight。

#### 输入

- `importance_score`
- `cluster_id`
- `active_mask`

#### 输出

- `weights: Tensor[N]`
- `policy_stats`

#### 第一版策略

- deterministic soft keep
- `w_keep = w_floor + (1-w_floor) * S_bar`
- `w = w_keep * cluster_keep[k]`

#### 验收

- inactive Gaussian 权重为 0
- active Gaussian 权重分布合理
- 与旧 `visibility union` 相比更平滑

---

### Task 4：改 `run_pseudo_refinement_v2.py`

#### 目标

在现有 local gating 回路中新增 SPGM mode。

#### 需要做的事

- 保留现有 `evaluate_sampled_views_for_local_gating`
- 新增 `spgm_*` mode
- accepted views → `collect_spgm_stats`
- stats → score → policy → `apply_gaussian_grad_mask`
- 增加 history 统计字段

#### 验收

- `pseudo_local_gating_mode=spgm_keep` 时能完整跑通
- 不破坏旧 hard / soft gating
- history 中新增 `spgm_*` 统计

---

### Task 5：先做 StageA.5 ablation

#### 目标

确认 SPGM 不是 no-op，并确认它对当前最好分支是否带来更稳定的 early gain。

#### 对照组

- baseline：旧 hard / soft gating
- exp1：`spgm_keep + xyz`
- exp2：`spgm_keep + xyz+opacity`

#### 输出

- replay metrics
- `grad_keep_ratio_xyz`
- `spgm_weight_mean / p50 / p90`
- `spgm_density_entropy`
- `cluster histogram`

---

## 6. 重要原则（给后续 agent 规划时必须坚持）

### 原则 1：SPGM 不是新上游

不要把主要精力用在改 `build_brpo_v2_signal_from_internal_cache.py` 的主逻辑上。

它已经负责 signal_v2 生产。当前问题不是“再多造一个 signal”，而是“如何让现有 signal 更合理地作用到 Gaussian 上”。

### 原则 2：SPGM 不要和 signal gate 混成一个东西

- `signal_gate.py`：继续负责 **view-quality filter**
- `spgm/*.py`：负责 **Gaussian-state-aware management**

两者是前后级，不是同一层。

### 原则 3：第一版只做 deterministic grad weighting

先把解释性和稳定性建立起来。

不要第一版就上：

- stochastic drop
- opacity Bernoulli attenuation
- full paper-like stochastic management

### 原则 4：先 StageA.5，再 StageB

顺序必须是：

1. StageA.5 证明 SPGM work
2. StageB 只作用 pseudo branch
3. 再考虑更强自由度 / 随机衰减

### 原则 5：SPGM 的目标不是“增加监督覆盖率”，而是“重分配地图更新权限”

这是最关键的原理判断。

当前系统已经不是“完全没信号”，而是“有信号，但它弱而局部”。

所以 SPGM 的作用不是制造更多 pseudo signal，而是：

- 把 pseudo-side 的地图改动权限收缩到更可信的 Gaussian 子集；
- 让 `20 iter` 的轻微正向，不至于在 `120 iter` 被自己优化掉。

---

## 7. 最终建议（一句话版）

按你当前 repo 的结构，**SPGM 最自然、最稳妥、最符合 BRPO 原理的落点，就是把“accepted pseudo views 的 visibility union grad mask”升级成“accepted pseudo views 上的 per-Gaussian depth-density-support importance weight map”。**

也就是：

```text
view gate 继续保留
但只做样本过滤
真正的 Gaussian 管理由 SPGM 接管
并在 backward 后 / optimizer.step 前施加
```

如果要给 agent 一个总任务描述，可以直接用下面这段：

> 在现有 `run_pseudo_refinement_v2.py` 的 local gating 回路上，实现一个最小可用的 BRPO-style SPGM：
> - 保留现有 `evaluate_sampled_views_for_local_gating()` 作为 accepted pseudo view filter；
> - 新增 `pseudo_branch/spgm/`，分成 `stats.py / score.py / policy.py` 三层；
> - 在 `maybe_apply_pseudo_local_gating()` 中新增 `spgm_keep` / `spgm_soft` mode；
> - 用 accepted pseudo views 的 visibility、camera-space depth、opacity/support proxy 构建 per-Gaussian importance；
> - 第一版仅做 deterministic soft keep，先支持 `StageA.5 + xyz-only`；
> - 第二版推进到 `StageB`，但只作用 pseudo branch；
> - 第三版再考虑 stochastic attenuation / opacity-specific policy。


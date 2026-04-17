# BRPO-style Local Gaussian Gating / 子集 Refine 工程落地方案（2026-04-15）

## 1. 为什么这份文档现在必须单独写

当前问题已经不是“有没有 depth signal”，而是：

- `train_mask` 平均约 18.7%，真正 non-fallback depth 约 3.49%；
- support 区域内的平均 correction 也只有约 1.53%；
- 但 A.5 / StageB 当前会更新全局 Gaussian `xyz/opacity`；
- 实测上 A.5 / StageB 已经出现大量 Gaussian 被改动，而 replay 仍退化。

因此现在最该补的不是再加一个大正则，而是让：

`supervision scope` 和 `optimization scope`
重新匹配。

local Gaussian gating / 子集 refine 的作用就是：
- pseudo branch 只更新当前 sampled pseudo views 真正可见、且信号 gate 通过的 Gaussian 子集；
- real branch 继续保留较全局的纠偏能力；
- 把“弱局部监督 -> 全局地图扰动”改成“弱局部监督 -> 局部地图修正”。

---

## 2. 目标边界

## 2.1 这次要落地什么

第一版只做：
1. StageA.5 / StageB 的 pseudo-side local gradient gating；
2. 支持 `xyz` 或 `xyz+opacity`；
3. 支持 hard gating 和 soft gating 两种模式；
4. 保留 current refine loop 的大框架，不上 full SPGM。

## 2.2 这次明确不做什么

1. 不做 full SPGM；
2. 不做全新的 map backend；
3. 不把 densify/prune 一起复杂化；
4. 不让 pseudo branch 的 gating 影响 real branch 的全局纠偏梯度。

第一版目标是：
- 先把 pseudo 对 Gaussian 的影响局部化；
- 先验证 replay 是否更稳；
- 再决定要不要进一步做 management / stochastic masking / importance score。

---

## 3. 当前代码里，哪些锚点可以直接利用

## 3.1 `run_pseudo_refinement_v2.py` 已有的可复用基础

当前 v2 refine 已经具备：
1. `stage_mode = stageA / stageA5 / stageB`
2. `stageA5_trainable_params = xyz / xyz_opacity`
3. pseudo 与 real view 双分支采样
4. RGB/depth mask 分离
5. source-aware depth loss

也就是说，当前 v2 主循环已经是合适的 local gating 承载体，不需要重新换训练框架。

## 3.2 当前 v2 的缺口

`run_pseudo_refinement_v2.py:569-599` 和 `725-768` 的训练循环，当前对于 Gaussian 更新是：
- sampled pseudo views 算 loss
- `total_loss.backward()`
- `gaussian_optimizer.step()`

这里没有任何 per-Gaussian local mask。

## 3.3 老版 v1 里有一个非常关键的可借鉴点

`scripts/run_pseudo_refinement.py:601-612` 已经做过一次“pseudo backward / real backward 分开处理”：

- 先 pseudo backward；
- 再清掉 pseudo frozen param 的 grad；
- 再 real backward。

这说明：
- 当前工程里已经有“分 branch 后处理梯度”的先例；
- local gating 最自然的第一版，不是去改 optimizer 结构，而是继续沿这个思路，在 backward 之后对 Gaussian grad 做 mask。

## 3.4 renderer 返回的可见性信息也已有先例

在老版 `run_pseudo_refinement.py` 中，`render(...)` 返回包里已经用过：
- `viewspace_points`
- `visibility_filter`
- `radii`

因此第一版 local gating 的首选实现，不应该自己再做一遍 3D->2D 投影判断，而应优先复用 renderer 的 `visibility_filter` 作为“当前 view 下哪些 Gaussian 真正在成像”的近似可见集合。

---

## 4. 推荐的新文件组织

```text
part3_BRPO/
├── pseudo_branch/
│   └── local_gating/
│       ├── __init__.py
│       ├── visibility_union.py      # 从 sampled pseudo renders 收集可见 Gaussian union
│       ├── signal_gate.py           # 基于每个 pseudo sample 的信号质量决定是否允许参与 local set
│       ├── grad_mask.py             # 对 xyz/opacity 梯度做 hard/soft mask
│       ├── gating_schema.py         # dataclass / summary helpers
│       └── gating_io.py             # history / debug 文件写出
├── scripts/
│   └── run_pseudo_refinement_v2.py  # 修改：接入 local gating CLI 和 backward 顺序
└── docs/
    └── BRPO_local_gaussian_gating_engineering_plan_20260415.md
```

为什么单独建 `local_gating/`：
- 它是 map-side 机制，不应和 signal generation 混在同一文件里；
- 后续如果要从 local gating 继续扩到 SPGM / importance score，也有清晰的挂载点。

---

## 5. 第一版 local gating 的工程定义

## 5.1 gating 的最小数学对象

对每个 Gaussian `j`，定义一个 pseudo-side gate：

- hard 版：`m_j ∈ {0,1}`
- soft 版：`w_j ∈ [0,1]`

pseudo 分支对 Gaussian 参数的更新变为：

- hard：`grad_j <- m_j * grad_j`
- soft：`grad_j <- w_j * grad_j`

这里的 `grad_j` 只指 pseudo branch 对 `xyz/opacity` 的梯度，不包括 real branch。

## 5.2 gate 的来源

第一版推荐把 gate 写成两层乘积：

1. visibility gate
   - Gaussian 必须在 sampled pseudo views 里实际可见
2. signal gate
   - 该 pseudo sample 本身必须达到最低信号门槛

即：

`local_gate = visible_union(sampled_good_pseudo_views)`

不要第一版就搞复杂 importance score。先把“只更新可见且样本本身过线的 Gaussian”做对。

---

## 6. visibility gate：怎么取

## 6.1 首选来源

直接使用 render package 里的：
- `visibility_filter`

每个 sampled pseudo view 在 forward 时都能拿到一个 `visibility_filter`，它已经是“这个 view 里哪些 Gaussian 参与了成像”的近似布尔向量。

## 6.2 第一版的聚合方式

对一次 iteration 内所有 sampled pseudo views：

1. 对每个 view，得到 `visibility_filter_i`
2. 对通过 signal gate 的 view 做 union：
   - `visible_union = OR_i visibility_filter_i`

这个 `visible_union` 就是本轮 pseudo-side 允许更新的 Gaussian 子集。

## 6.3 为什么不用 intersection

intersection 会太严：
- sampled pseudo views 本来就少；
- 当前 signal 已经不强；
- intersection 很容易把 active set 压得过小。

第一版应使用 union，而不是 intersection。

---

## 7. signal gate：怎么定义 sampled pseudo view 是否合格

## 7.1 第一版不要做 per-pixel gate 到 Gaussian

第一版不建议上来就做“由 pixel 反投影到 Gaussian 的局部分数图”。
这太复杂，也不利于快速验证主假设。

第一版用 per-view gate 就够：
- 这个 sampled pseudo view 是否值得给 Gaussian 梯度；
- 如果值得，就把它的可见 Gaussian 全纳入 union；
- 如果不值得，这个 view 的 Gaussian 更新全部屏蔽。

## 7.2 per-view gate 的推荐输入

每个 `view` 当前已经能拿到：
- `rgb_confidence_nonzero_ratio`
- `depth_confidence_nonzero_ratio`
- `target_depth_verified_ratio`
- `target_depth_seed_ratio`
- `target_depth_dense_ratio`
- `target_depth_render_fallback_ratio`
- `source_meta / depth_meta`

如果切到新的 BRPO v2 signal path，后续还会有：
- `rgb_mask_v2_nonzero_ratio`
- `depth_supervision_mask_v2_nonzero_ratio`
- `mean_abs_rel_correction_verified`

## 7.3 第一版推荐 gate 规则

第一版建议先用 rule-based hard gate：

一个 sampled pseudo view 必须同时满足：
1. `verified_ratio >= min_verified_ratio`
2. `rgb_mask_nonzero_ratio >= min_rgb_mask_ratio`
3. `render_fallback_ratio <= max_fallback_ratio`
4. 可选：`mean_abs_rel_correction_verified >= min_correction`

推荐第一版默认：
- `min_verified_ratio = 0.01`
- `min_rgb_mask_ratio = 0.01`
- `max_fallback_ratio = 0.98`
- `min_correction = 0.0`（先不开）

## 7.4 signal gate 的输出

每个 iteration 导出：
- sampled pseudo ids
- accepted pseudo ids
- rejected pseudo ids
- 每个 rejected view 的原因
- accepted views 的 union-visible Gaussian ratio

这部分日志必须进 history，否则后面会不知道“是 gate 生效了，还是根本没 sample 到有效 view”。

---

## 8. 需要新增的 dataclass / helper

## 8.1 `gating_schema.py`

建议新增：

```python
@dataclass
class LocalGatingConfig:
    mode: str                     # off / hard_visible_union / hard_visible_union_signal / soft_visible_union_signal
    params: str                   # xyz / xyz_opacity
    min_verified_ratio: float
    min_rgb_mask_ratio: float
    max_fallback_ratio: float
    min_correction: float
    soft_power: float
    log_interval: int

@dataclass
class LocalGatingStepSummary:
    iter: int
    sampled_sample_ids: list[int]
    accepted_sample_ids: list[int]
    rejected_sample_ids: list[int]
    active_gaussian_ratio: float
    active_gaussian_count: int
    grad_norm_xyz_before: float
    grad_norm_xyz_after: float
    grad_norm_opacity_before: float | None
    grad_norm_opacity_after: float | None
```

## 8.2 `visibility_union.py`

建议提供：

```python
def build_visible_union_masks(render_pkgs: list[dict], accepted_flags: list[bool]) -> torch.Tensor:
    ...  # 返回 shape [N] bool tensor
```

## 8.3 `signal_gate.py`

建议提供：

```python
def evaluate_sample_signal_gate(view: dict, cfg: LocalGatingConfig) -> tuple[bool, dict]:
    ...  # 返回是否通过 + 失败原因/统计
```

## 8.4 `grad_mask.py`

建议提供：

```python
def apply_local_gaussian_grad_mask_(
    gaussians,
    active_mask: torch.Tensor,
    params: str = 'xyz',
    soft_weights: torch.Tensor | None = None,
) -> dict:
    ...
```

这个函数必须返回 grad before/after summary，方便写 history。

---

## 9. run_pseudo_refinement_v2.py 需要怎么改

## 9.1 新增 CLI 参数

建议新增：

```text
--pseudo_local_gating_mode {off,hard_visible_union,hard_visible_union_signal,soft_visible_union_signal}
--pseudo_local_gating_params {xyz,xyz_opacity}
--pseudo_local_min_verified_ratio 0.01
--pseudo_local_min_rgb_mask_ratio 0.01
--pseudo_local_max_fallback_ratio 0.98
--pseudo_local_min_correction 0.0
--pseudo_local_soft_power 1.0
--pseudo_local_log_interval 20
```

## 9.2 StageA.5 的改法

当前 StageA.5 没 real branch，所以相对简单。

原逻辑：
- sampled pseudo -> forward
- mean pseudo loss -> backward
- gaussian_optimizer.step

改成：
1. sampled pseudo -> forward，并保留每个 `pkg` 的 `visibility_filter`
2. pseudo loss backward
3. 用 signal gate 过滤 sampled views
4. 对 accepted views 的 `visibility_filter` 做 union
5. 对 `_xyz.grad` / `_opacity.grad` 做 mask
6. `gaussian_optimizer.step`

这能保证：
- pose / exposure 仍全局更新
- Gaussian 只在 local subset 上更新

## 9.3 StageB 的改法：必须把 backward 拆开

这是本次最关键的实现点。

当前 `run_pseudo_refinement_v2.py:764-768` 是：
- `total_loss = lambda_real * real + lambda_pseudo * pseudo`
- `total_loss.backward()`
- 两个 optimizer step

这不适合 local gating，因为一旦合并 backward，你就无法只屏蔽 pseudo-side 的 Gaussian grad 而保留 real-side 的 global grad。

正确改法应当参考老版 v1：

### 新的 StageB backward 顺序

1. `zero_grad()`
2. 先算 pseudo branch
3. `pseudo_backward = lambda_pseudo * pseudo_loss`
4. `pseudo_backward.backward(retain_graph=has_real_branch)`
5. 对 Gaussian grad 应用 local gating mask
6. 再算 real branch
7. `real_backward = lambda_real * real_loss`
8. `real_backward.backward()`
9. optimizer step

这样就能保证：
- pseudo branch 的 Gaussian 更新只落在 local subset
- real branch 仍能对全局 Gaussian 做纠偏

## 9.4 需要保留的日志

在 `stageA_history.json / stageB_history.json` 增加：
- `pseudo_local_gating_mode`
- `pseudo_local_gating_params`
- `pseudo_local_gate_summaries`
- `mean_active_gaussian_ratio`
- `mean_accepted_pseudo_views`
- `mean_rejected_pseudo_views`
- `grad_norm_xyz_before_gate`
- `grad_norm_xyz_after_gate`
- `grad_norm_opacity_before_gate`
- `grad_norm_opacity_after_gate`

否则无法判断是：
- gating 太强；
- sampled pseudo 本来就弱；
- 还是局部化确实让 replay 更稳。

---

## 10. 新版 local gating 与当前 StageA5/StageB 配置的关系

## 10.1 参数组层面

当前 `pseudo_branch/gaussian_param_groups.py` 已支持：
- `xyz`
- `xyz_opacity`

local gating 第一版正好可以建立在它上面：
- 如果 `params=xyz`，只 mask `gaussians._xyz.grad`
- 如果 `params=xyz_opacity`，同时 mask `gaussians._xyz.grad` 和 `gaussians._opacity.grad`

不需要改 optimizer group 结构。

## 10.2 densify / prune

第一版建议：
- local gating 验证阶段，不启用 densify / prune
- 尤其不要一边试 local subset，一边让 Gaussian 集合数量变化

原因：
- 一旦 densify/prune 开启，active mask 的索引空间会变化；
- 这会把“subset refine 是否有效”与“map topology 是否变化”混在一起。

所以第一版统一要求：
- StageA.5：关闭 densify/prune
- StageB：先做 conservative no-densify 版本

## 10.3 real branch 不该被 gating

这是第一版最重要的边界：
- local gating 只作用于 pseudo branch 的 Gaussian grad
- real branch 的 gradient 不做 local mask

否则你会把“pseudo 只做局部修正、real 负责全局拉回”的设计优势一起抹掉。

---

## 11. soft gating 要不要第一版一起做

我建议：接口保留，默认先跑 hard。

原因：
- hard gate 更容易解释；
- 如果连 hard 都没变稳，那 soft 大概率只是增加调参维度；
- 如果 hard 稳了，但 improvement 太保守，再开 soft。

## 11.1 hard 模式

```text
hard_visible_union
```
- 不看 signal，只对 sampled pseudo views 的 visible union 开门
- 这是最小 smoke 版

```text
hard_visible_union_signal
```
- sampled pseudo view 必须先过 signal gate，再把 visible gaussians 纳入 union
- 这是推荐默认第一版

## 11.2 soft 模式

```text
soft_visible_union_signal
```
- per-view 先有 signal score
- union 后的 Gaussian weight 由可见 view 的 score 聚合而来
- 对 grad 做乘权，不是硬清零

第一版只需要在接口和代码路径上留钩子，不必先作为主实验。

---

## 12. 建议的实验顺序

## G0：2-frame smoke

目标：
- 确认 render pkg 在 v2 路径下也能拿到 `visibility_filter`
- 确认 grad mask 不会报 shape / dtype 错

验收：
- history 里有 `active_gaussian_ratio`
- `_xyz.grad` 的 before/after 差值非零

## G1：E1 winner 上的 StageA.5 short compare

数据：
- E1 `signal-aware-8`

对照：
1. baseline StageA.5（无 gating）
2. `hard_visible_union`
3. `hard_visible_union_signal`

目标：
- 看 pseudo-side loss 是否还能下降
- 看 replay 是否更稳

如果 `hard_visible_union_signal` 已优于 `hard_visible_union`，说明 signal gate 是有必要的。

## G2：StageB short compare

前提：G1 至少没有明显反噬。

对照：
1. StageB baseline
2. StageB + `hard_visible_union_signal`

这里的目标不是立刻追求更低 pseudo loss，而是看：
- replay 是否更不容易退化
- Gaussian drift 是否收缩

## G3：必要时再开 soft mode

只有当：
- hard 模式确实让 replay 更稳
- 但 pseudo loss 收缩太慢

再开 `soft_visible_union_signal`。

---

## 13. 成功标准

第一版不要把成功定义成“loss 更低”。

更合理的成功标准是：
1. `active_gaussian_ratio` 明显小于全局 1.0，且不是塌到极小值；
2. pseudo-side loss 仍能下降；
3. 相比 baseline，replay 不再明显恶化，最好有轻微改善；
4. Gaussian xyz drift 的受影响比例明显下降；
5. 如果有 real branch，说明 global correction 没被一起锁死。

---

## 14. 风险点与规避

1. 最大风险：v2 render path 里没有把 `visibility_filter` 传回来
   - 规避：G0 smoke 先验证；如果缺，就先把 old run 里的用法迁到 v2

2. 第二个风险：StageB 继续用合并 backward，导致 local gating 同时屏蔽 real gradient
   - 规避：StageB 必须按 pseudo backward / gate / real backward 重构

3. 第三个风险：gate 过严，active set 太小
   - 规避：先上 `hard_visible_union`，再加 signal 条件

4. 第四个风险：日志不够，无法知道是 signal 弱还是 gating 过强
   - 规避：每轮记录 accepted/rejected pseudo ids 和 active ratio

---

## 15. 和 BRPO 文档的对应关系

这份 local gating 文档对应 BRPO 理论分析里的判断是：
- 当前最值得优先考虑的 map-side方向，不是 full SPGM；
- 而是先让 pseudo 分支只作用于局部 Gaussian 子集；
- 它直接命中当前主矛盾：弱局部监督去推动全局 Gaussian 扰动。

所以这份方案的定位非常明确：
- 它不是 paper-faithful full BRPO 的终点；
- 但它是当前 case 下最应该先做的 map-side落地件。

---

## 16. 一句话执行判断

如果现在就要开始改 map-side 逻辑，最稳的工程路线不是直接上 SPGM，而是：

- 在 `run_pseudo_refinement_v2.py` 里把 StageA.5 / StageB 的 pseudo-side Gaussian backward 改成“先分支 backward，再做 local grad mask，再 step”；
- 用 sampled pseudo views 的 `visibility_filter` 做 visible union；
- 再叠一个轻量 signal gate，只让真正过线的 pseudo view 把可见 Gaussian 纳入 active subset；
- 第一版先做 hard gating、先关 densify/prune、先看 replay 是否止跌。

只要这一步成立，后面再讨论 soft weighting、importance score 或 SPGM 才有意义。

# Part3 BRPO SPGM 工程落地方案（基于现代码审计，2026-04-16）

> 这份文档是对 `docs/SPGM_landing_plan_for_part3_BRPO.md` 的工程化收敛版：不是再讲一遍 SPGM 原理，而是把它压到当前 repo 的真实代码结构上，明确第一版该怎么改、先改到哪、哪些地方其实已经具备条件、哪些地方必须修正命名/接口假设。

---

## 1. 结论先行

当前代码已经具备把 SPGM 落在 pseudo-side local gating 回路上的全部必要锚点，因此推荐的推进顺序是：

1. 先做 `StageA.5 + xyz-only + deterministic keep` 的 SPGM v1；
2. 保留现有 `hard_visible_union_signal / soft_visible_union_signal` 作为 baseline，不覆盖旧路径；
3. 新增 `pseudo_branch/spgm/` 三层模块：`stats.py / score.py / policy.py`；
4. 只改 `run_pseudo_refinement_v2.py` 的 gating 分支、配置和 history 记录，不改上游 signal builder、不改 loss builder、不改 optimizer 框架；
5. StageA.5 验证通过后，再把同一套 SPGM 权重接到 StageB 的 pseudo backward 后、real backward 前；
6. 第一版不要做 stochastic drop，不要把 real branch 也一起 gate，不要先上 `xyz+opacity`。

一句话说，就是把现在的：

```text
accepted pseudo views -> visibility union -> grad mask
```

升级成：

```text
accepted pseudo views -> per-Gaussian support/depth/density stats -> importance score -> grad weights
```

而且这个升级点，当前代码已经有现成插口。

---

## 2. 按现代码核对后的关键事实

### 2.1 当前真实插口已经存在

当前 SPGM 最自然的接入点就在：

- `scripts/run_pseudo_refinement_v2.py`
- 函数：`maybe_apply_pseudo_local_gating(...)`

当前真实流程是：

```text
sampled_views
-> evaluate_sampled_views_for_local_gating(...)
-> build_visibility_weight_map(...)
-> apply_gaussian_grad_mask(...)
```

也就是 backward 已经结束、optimizer.step 还没发生时，代码已经允许我们拿到：

- `sampled_views`
- `gate_results`
- `render_packages`
- `gaussians`
- Gaussian grad

这正好就是 SPGM 应该插入的位置。

### 2.2 StageB 其实已经满足“只作用 pseudo branch”的要求

`run_pseudo_refinement_v2.py` 当前 StageB 顺序已经是：

```text
pseudo_backward.backward(retain_graph=has_real_branch)
-> maybe_apply_pseudo_local_gating(...)
-> real_backward.backward()
-> optimizer.step()
```

这点非常关键：

- 当前 StageB 已经不是“real/pseudo 完全混在一起无法拆”的状态；
- 因此 SPGM v2 推到 StageB 时，可以直接只压 pseudo-side Gaussian grad；
- 第一版不需要额外重构 StageB backward 顺序。

### 2.3 renderer 已经提供 SPGM v1 需要的可见性信息

`/home/bzhang512/CV_Project/third_party/S3PO-GS/gaussian_splatting/gaussian_renderer/__init__.py` 中，`render(...)` 当前会返回：

- `visibility_filter`
- `radii`
- `depth`
- `opacity`
- `n_touched`

其中：

- `visibility_filter` 已经足够做 per-view visible set；
- `n_touched` 在 S3PO-GS 里本来就被拿来构造 `occ_aware_visibility`；
- 所以 SPGM 的 `support_count` 完全可以直接站在现有 renderer 产物上实现，不需要去改 rasterizer。

### 2.4 当前 repo 已经有可复用的相机深度计算锚点

`pseudo_branch/pseudo_camera_state.py` 里已经有：

- `current_w2c(vp)`
- `current_c2w(vp)`
- `apply_pose_residual_(vp)`

这意味着 SPGM 里需要的 camera-space depth：

```text
z_i^(v) = (R_v x_i + t_v)_z
```

不应该从 `render_pkg['depth']` 反推，而应该直接：

- 用 `gaussians.get_xyz`
- 配合 `current_w2c(view['vp'])`
- 直接算每个 Gaussian 在当前 pseudo view 下的 z 值

这比从 image depth 倒推更对路，也更符合现在的代码结构。

### 2.5 现有 local gating 配置很薄，必须扩 schema

当前 `pseudo_branch/local_gating/gating_schema.py` 里的 `PseudoLocalGatingConfig` 只有：

- `mode`
- `params`
- `min_verified_ratio`
- `min_rgb_mask_ratio`
- `max_fallback_ratio`
- `min_correction`
- `soft_power`
- `log_interval`

现在还完全没有：

- SPGM 超参
- SPGM mode 判断
- SPGM/visibility-union 分流逻辑

所以第一步不是直接写 stats.py，而是先把 config 体系补齐。

---

## 3. 设计文档和现代码之间，已经确认的几个落差

### 3.1 CLI 名称不是 `pseudo_local_gating_mode`

当前真实 CLI 参数名是：

```text
--pseudo_local_gating
```

不是 landing 文档里写的 `pseudo_local_gating_mode`。

因此工程实现时应当：

- 继续保留 `--pseudo_local_gating` 这个现有参数名；
- 只扩 choices，不要另起一个新参数名，避免 run script 全部跟着改。

### 3.2 当前 soft 判定只认 `soft_*`，如果直接加 `spgm_soft` 会静默跑错

`PseudoLocalGatingConfig.is_soft()` 当前实现是：

```python
return (self.mode or '').startswith('soft_')
```

这意味着如果我们只是把 mode choice 增加为 `spgm_soft`：

- signal gate 不会走 soft gate 路径；
- 它会被当成 hard gate；
- 最后会出现“命名上是 spgm_soft，行为上却是 hard accept/reject”的静默偏差。

这个坑必须在第一版就修掉。

### 3.3 landing 文档里提到的 `pseudo_refine_scheduler.py` 路径和 repo 现状不一致

当前真实文件是：

```text
pseudo_branch/pseudo_refine_scheduler.py
```

不是 root 下的 `pseudo_refine_scheduler.py`。

这不影响 SPGM 主体，但写工程文档和 patch 时必须按真实路径落。

### 3.4 `render_pkg['depth']` 不是 per-Gaussian depth

当前 renderer 返回的 `depth` 是 image-space rendered depth map，不是 `Tensor[N]` 的 per-Gaussian depth。

因此 SPGM v1 里的：

- `depth_value: Tensor[N]`

必须由 `gaussians.get_xyz + current_w2c(vp)` 显式计算，不能误把 `render_pkg['depth']` 当现成输入。

### 3.5 history 扩展风险很低

当前 `visible_union_ratio / grad_keep_ratio_xyz / accepted_pseudo_sample_ids` 等 gating 字段，只在：

- `run_pseudo_refinement_v2.py`
- `pseudo_branch/local_gating/gating_io.py`

内部使用，没有发现其它 summary/plot 脚本强依赖固定 schema。

所以：

- 可以直接给 gating history 扩 `spgm_*` 字段；
- 旧 baseline 继续保留老字段；
- 风险主要在控制台打印和 `stageA_history.json / stageB_history.json` 的一致性，不在外部消费者兼容性。

---

## 4. 推荐的 SPGM v1 目标边界

### 4.1 第一版明确要做什么

SPGM v1 只做：

1. `StageA.5`
2. `xyz-only`
3. `deterministic keep weight`
4. `accepted pseudo views only`
5. `support + depth + opacity-proxy`
6. backward 后、optimizer.step 前的 grad weighting

### 4.2 第一版明确不做什么

这次先不要做：

1. 不改 `build_brpo_v2_signal_from_internal_cache.py`
2. 不改 `pseudo_loss_v2.py`
3. 不改 `gaussian_param_groups.py`
4. 不上 stochastic drop
5. 不上 opacity-specific stochastic attenuation
6. 不把 SPGM 压到 real branch
7. 不先做 `xyz_opacity`
8. 不把旧 `hard_visible_union_signal` 路径覆盖掉

---

## 5. 文件级工程落地方案

## 5.1 新增 `pseudo_branch/spgm/__init__.py`

职责：只做导出，不写逻辑。

建议导出：

```python
from .stats import collect_spgm_stats
from .score import build_spgm_importance_score
from .policy import build_spgm_grad_weights
```

这样 `run_pseudo_refinement_v2.py` 只需要从一个入口 import。

---

## 5.2 新增 `pseudo_branch/spgm/stats.py`

### 目标

从 accepted pseudo views 提取 per-Gaussian state，输出最小可用的：

- `support_count`
- `depth_value`
- `density_proxy`
- `active_mask`

### 建议接口

```python
collect_spgm_stats(
    sampled_views,
    gate_results,
    render_packages,
    gaussians,
    device,
) -> dict
```

### 建议内部 helper

```python
_camera_space_depths(xyz: torch.Tensor, vp) -> torch.Tensor
_normalize_active(x: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor
_weighted_median_over_views(depth_stack, weight_stack, valid_stack) -> torch.Tensor
```

### 关键实现建议

1. `support_count`
   - 对每个 accepted view：
   - 取 `pkg['visibility_filter']`
   - 令 `support_count[vis] += gate_weight`

2. `depth_value`
   - 对每个 accepted view 计算一个 `z_view: Tensor[N]`
   - 仅保留 `visibility_filter=True` 的位置有效
   - 因为每 iter 的 sampled pseudo views 很少（当前常用是 4），可以直接把 accepted view 的 `z_view` 堆成 `Tensor[V, N]`
   - 第一版仍然可以做 landing 文档里建议的 weighted median，不必退化成 mean

3. `density_proxy`
   - 当前 repo 没有论文同款 density field
   - 第一版按工程近似：

```text
density_proxy = normalize(sigmoid(opacity)) * normalize(support_count)
```

4. `active_mask`
   - `support_count > 0`

### 这里必须注意的现代码约束

- 深度计算必须走 `current_w2c(view['vp'])`；
- 不要用 `render_pkg['depth']` 代替 per-Gaussian depth；
- `visibility_filter` 长度必须与 `gaussians._xyz.shape[0]` 对齐，沿用当前 `visibility_union.py` 的 size-check 风格。

### 建议额外返回的 summary 字段

除了四个核心 tensor，建议顺手返回：

- `accepted_view_count`
- `active_ratio`
- `support_mean`
- `support_p50`
- `support_max`

这些字段后面直接喂 history/logging，很省事。

---

## 5.3 新增 `pseudo_branch/spgm/score.py`

### 目标

把 `support_count / depth_value / density_proxy / active_mask` 转成 SPGM importance。

### 建议接口

```python
build_spgm_importance_score(
    depth_value,
    density_proxy,
    support_count,
    active_mask,
    num_clusters=3,
    alpha_depth=0.5,
    beta_entropy=0.5,
    gamma_entropy=0.5,
    support_eta=0.5,
    entropy_bins=32,
) -> dict
```

### 建议内部子函数

```python
build_depth_partition(depth_value, active_mask, num_clusters=3) -> tuple[cluster_id, depth_score]
compute_density_entropy(density_proxy, active_mask, entropy_bins=32) -> float
```

### 实现要点

1. `cluster_id`
   - 只对 active Gaussian 做 quantile split
   - 第一版固定 `K=3`：near / mid / far
   - inactive 统一记为 `-1`

2. `depth_score`

```text
s_z = 1 - (z - z_min) / (z_max - z_min + eps)
```

3. `density_entropy`
   - 对 active 区域的 `density_proxy` 做 histogram entropy
   - 返回全局标量即可，不必先做 cluster-wise entropy

4. `density_score`

```text
s_rho = rho_norm * (1 - beta * Hbar) + gamma * Hbar
```

5. `importance_score`

```text
S = alpha * s_z + (1 - alpha) * s_rho
support_norm = support_count / (support_count.max() + eps)
S_bar = clamp(S, 0, 1) * support_norm ** eta
```

### 建议额外返回的 summary 字段

- `density_entropy`
- `cluster_count_near`
- `cluster_count_mid`
- `cluster_count_far`
- `importance_mean`
- `importance_p50`

### 退化情形处理

下面这些边界条件必须在第一版就写清楚：

- `active_mask.sum() == 0`：直接返回全零 score；
- `active_count < num_clusters`：自动退化为更少 quantile 桶，不报错；
- `z_max == z_min`：depth_score 直接置 1；
- `density_proxy` 全常数：entropy 允许为 0，不允许 NaN。

---

## 5.4 新增 `pseudo_branch/spgm/policy.py`

### 目标

把 importance score 转成可直接喂给 `apply_gaussian_grad_mask(...)` 的 `weights: Tensor[N]`。

### 建议接口

```python
build_spgm_grad_weights(
    importance_score,
    cluster_id,
    active_mask,
    weight_floor=0.05,
    cluster_keep=(1.0, 0.8, 0.6),
) -> dict
```

### 第一版策略

```text
base_keep = weight_floor + (1 - weight_floor) * importance_score
weights = base_keep * cluster_keep[cluster_id]
inactive -> 0
```

### 建议返回

```python
{
    'weights': Tensor[N],
    'stats': {
        'weight_mean': ...,
        'weight_p10': ...,
        'weight_p50': ...,
        'weight_p90': ...,
        'active_ratio': ...,
    }
}
```

### 工程判断

第一版只做 deterministic keep 就够了，因为：

- 现在主要目标是验证 SPGM 不是 no-op；
- 需要可解释、可复现的 StageA.5 / StageB 对照；
- 当前 repo 已经有现成的 grad weighting 承接点，不需要额外引入随机性。

---

## 5.5 修改 `pseudo_branch/local_gating/gating_schema.py`

### 需要新增的字段

建议把 dataclass 扩成：

```python
mode: str = 'off'
params: str = 'xyz'
min_verified_ratio: float = 0.01
min_rgb_mask_ratio: float = 0.01
max_fallback_ratio: float = 0.995
min_correction: float = 0.0
soft_power: float = 1.0
log_interval: int = 20
spgm_num_clusters: int = 3
spgm_alpha_depth: float = 0.5
spgm_beta_entropy: float = 0.5
spgm_gamma_entropy: float = 0.5
spgm_support_eta: float = 0.5
spgm_weight_floor: float = 0.05
spgm_entropy_bins: int = 32
spgm_density_mode: str = 'opacity_support'
spgm_cluster_keep_near: float = 1.0
spgm_cluster_keep_mid: float = 0.8
spgm_cluster_keep_far: float = 0.6
```

### 需要新增的方法

建议不要再只保留一个 `is_soft()`，而是显式写成：

```python
def enabled(self) -> bool

def uses_visibility_union(self) -> bool

def uses_spgm(self) -> bool

def is_soft(self) -> bool
```

其中：

- `uses_visibility_union()` 识别 `hard_visible_union_signal / soft_visible_union_signal`
- `uses_spgm()` 识别 `spgm_keep / spgm_soft`
- `is_soft()` 至少要覆盖 `soft_visible_union_signal` 和 `spgm_soft`

这样可以避免 mode 命名和行为脱钩。

---

## 5.6 修改 `pseudo_branch/local_gating/signal_gate.py`

这部分不要重写成 SPGM。

只做两件事：

1. 继续输出现有 `gate_results` 结构；
2. 让 `spgm_soft` 能正确走 soft gate 分支。

也就是说它仍然只负责：

```text
view-quality filter
```

而不是负责 Gaussian-side policy。

---

## 5.7 修改 `pseudo_branch/local_gating/gating_io.py`

当前 `build_iteration_gating_summary(...)` 只会写：

- accepted/rejected pseudo ids
- visible_union_ratio
- grad_keep_ratio_xyz
- grad_norm_xyz_pre/post

SPGM 接上后建议新增：

- `spgm_active_ratio`
- `spgm_support_mean`
- `spgm_support_p50`
- `spgm_density_entropy`
- `spgm_weight_mean`
- `spgm_weight_p10`
- `spgm_weight_p50`
- `spgm_weight_p90`
- `spgm_cluster_count_near`
- `spgm_cluster_count_mid`
- `spgm_cluster_count_far`

建议改法：

- 给 `build_iteration_gating_summary(...)` 多传一个 `extra_stats` 或 `spgm_stats` dict；
- 旧 visibility-union path 这些字段写 `None`；
- SPGM path 则把对应 summary 填进去。

这样 StageA/StageB 的 history append 逻辑最少改。

---

## 5.8 修改 `scripts/run_pseudo_refinement_v2.py`

这是唯一真正需要做 wiring 的核心文件。

### A. import 层

新增：

```python
from pseudo_branch.spgm import (
    collect_spgm_stats,
    build_spgm_importance_score,
    build_spgm_grad_weights,
)
```

### B. parse_args()

当前真实参数名是 `--pseudo_local_gating`，建议只扩 choices：

```text
off
hard_visible_union_signal
soft_visible_union_signal
spgm_keep
spgm_soft
```

同时新增这些 CLI：

- `--pseudo_local_gating_spgm_num_clusters`
- `--pseudo_local_gating_spgm_alpha_depth`
- `--pseudo_local_gating_spgm_beta_entropy`
- `--pseudo_local_gating_spgm_gamma_entropy`
- `--pseudo_local_gating_spgm_support_eta`
- `--pseudo_local_gating_spgm_weight_floor`
- `--pseudo_local_gating_spgm_entropy_bins`
- `--pseudo_local_gating_spgm_cluster_keep_near`
- `--pseudo_local_gating_spgm_cluster_keep_mid`
- `--pseudo_local_gating_spgm_cluster_keep_far`

命名建议继续挂在 `pseudo_local_gating_` 前缀下，和现有 run script 风格一致。

### C. build_pseudo_local_gating_cfg(args)

把上面这些新 CLI 全部映射进 dataclass。

### D. `_init_gating_history_lists()` / `_append_gating_history()`

补上所有 `spgm_*` 列表。

### E. `maybe_apply_pseudo_local_gating(...)`

建议改成三路分支：

```python
if not gating_cfg.enabled():
    ...  # no-op
elif gating_cfg.uses_visibility_union():
    ...  # 完整保留旧路径
elif gating_cfg.uses_spgm():
    gate_results = evaluate_sampled_views_for_local_gating(...)
    spgm_stats = collect_spgm_stats(...)
    spgm_score = build_spgm_importance_score(...)
    spgm_policy = build_spgm_grad_weights(...)
    grad_stats = apply_gaussian_grad_mask(..., weights=spgm_policy['weights'])
    ...
else:
    raise ValueError(...)
```

注意：

- `gate_results` 仍然先跑；
- SPGM 不替代 view gate，只替代 visibility-union 这一步；
- 旧 visibility-union path 完整保留，方便做 baseline compare。

### F. 控制台 logging

当前打印是围绕：

- accepted 数量
- visible_union_ratio
- keep_xyz
- grad pre/post

SPGM 模式下应改成打印：

- accepted 数量
- `spgm_active_ratio`
- `spgm_support_p50`
- `spgm_weight_p50`
- `spgm_weight_p90`
- `grad_norm_xyz_pre/post`

不要在 SPGM 模式下继续打印 `visible_union_ratio`，否则日志语义会混乱。

### G. history / effective_source_summary 序列化

以下两个地方都要记新 SPGM 配置：

- `stageA_history['effective_source_summary']`
- `stageB_history['pseudo_local_gating']`

否则后面复盘实验时，history 里只会看到 mode，看不到关键超参。

---

## 6. 推荐的实现顺序

### 第 0 步：只做 plumbing，不做算法

先把下面这些打通：

1. `spgm/` 新包创建
2. import 不报错
3. CLI 可解析
4. config 可序列化
5. history 可写 `spgm_*` 空字段
6. `spgm_keep` 路径能走到 `apply_gaussian_grad_mask()` 前

目标：先证明 wiring 是通的。

### 第 1 步：实现 `stats.py`

优先做：

- accepted-view filtering
- support_count
- active_mask
- camera-space depth
- weighted median depth
- density_proxy

目标：先把 `Tensor[N]` 级别的中间量稳定产出。

### 第 2 步：实现 `score.py`

优先做：

- active-only quantile split
- depth_score
- density_entropy
- importance_score

目标：先把 `importance_score` 稳定压到 `[0,1]`。

### 第 3 步：实现 `policy.py`

优先做：

- deterministic keep
- cluster keep
- inactive -> 0
- weight summary

目标：保证最终 `weights` 能直接替代 visibility-union weight map。

### 第 4 步：接到 StageA.5

只验证：

- 不 crash
- 不是 no-op
- 历史里真写入 `spgm_*`
- 与 baseline 有可观察差异

### 第 5 步：StageA.5 通过后，再接 StageB

因为当前 StageB 已经 pseudo/backward 分开，所以这一步不是结构性难题，主要是实验验证问题。

---

## 7. 建议的验证阶梯

## 7.1 代码级 smoke

实现完成后先跑：

```bash
cd /home/bzhang512/CV_Project/part3_BRPO
/home/bzhang512/miniconda3/envs/s3po-gs/bin/python - <<'PY'
from pseudo_branch.spgm import collect_spgm_stats, build_spgm_importance_score, build_spgm_grad_weights
print('spgm import smoke ok')
PY
```

验收：import 正常。

## 7.2 StageA.5 极短 smoke（5 iter）

推荐沿用当前 StageA.5 最优 baseline 的同一路输入，只把 gating 改成 `spgm_keep`。

当前可直接复用的基准输入来自：

- baseline PLY：`/data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache/after_opt/point_cloud/point_cloud.ply`
- pseudo cache：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/pseudo_cache_baseline`
- signal_v2：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/signal_v2`

建议 smoke root：

```text
/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_spgm_v1_smoke/
```

验收标准：

- `stageA_history.json` 正常生成；
- `pseudo_local_gating.mode == spgm_keep`；
- `history['spgm_active_ratio']` 非全 0；
- `history['spgm_weight_p50']` 非全 0 / 非全 1；
- `refined_gaussians.ply` 正常输出。

## 7.3 StageA.5 正式短对照（20 iter）

正式对照建议固定在当前 P2-F winner 分支上：

- baseline：`hard_visible_union_signal`
- exp：`spgm_keep`
- trainable params：`xyz`
- 其余参数与 `stageA5_v2rgbonly_xyz_gated_rgb0192_80` 保持一致

当前 canonical baseline 路径：

```text
/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_gated_rgb0192_80
```

建议正式对照输出单独开新 root，不要写进 smoke root。

验收标准：

- `spgm_*` 统计有明显非空分布；
- `grad_keep_ratio_xyz` 与 hard gating 有可观察差异；
- replay 不应明显劣于 hard gating；
- 至少能证明 SPGM 不是 no-op。

## 7.4 StageB 只在 StageA.5 pass 后再上

StageB formal compare 的 canonical baseline 用当前 P2-J winner：

```text
/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2j_stageB_bounded_schedule_compare_e1/stageB_post40_lr03_120
```

StageB 接入原则：

- 只压 pseudo backward 后的 Gaussian grad；
- real branch 保持 anchor，不做 SPGM gate；
- 第一轮只跑短预算（20 或 40 iter），先看动力学；
- 不要直接上 120 iter full compare。

---

## 8. 推荐的 v1 CLI 默认值

为了减少第一次调参维度，建议先固定：

- `spgm_num_clusters = 3`
- `spgm_alpha_depth = 0.5`
- `spgm_beta_entropy = 0.5`
- `spgm_gamma_entropy = 0.5`
- `spgm_support_eta = 0.5`
- `spgm_weight_floor = 0.05`
- `spgm_entropy_bins = 32`
- `spgm_cluster_keep_near = 1.0`
- `spgm_cluster_keep_mid = 0.8`
- `spgm_cluster_keep_far = 0.6`

第一版先不要扫这些超参，先看 wiring + direction 是否成立。

---

## 9. 这次实现里最容易踩的坑

1. `spgm_soft` 如果不修 `is_soft()`，会静默走 hard gate。
2. `render_pkg['depth']` 不是 per-Gaussian depth，不能直接拿来做 `depth_value`。
3. 如果把 SPGM summary 强行塞进 `visible_union_*` 字段，会让日志语义混乱。
4. 如果在 StageB 里把 SPGM 放到 real backward 之后，就会把 real anchor 也一起压到。
5. 如果第一版就引入 stochastic policy，很难判断 regression 是 wiring 问题还是策略问题。
6. 如果直接覆盖旧 gating 路径，会丢掉最重要的 baseline compare。

---

## 10. 最终执行建议

如果下一步直接开工，建议按下面的实际执行顺序做：

1. 先改 `gating_schema.py` + `run_pseudo_refinement_v2.py` 的 parser/config/history plumbing；
2. 新建 `pseudo_branch/spgm/` 三个文件，但先写最小骨架；
3. 先让 `spgm_keep` 走通并写出全零/占位 summary；
4. 再补 `stats.py`；
5. 再补 `score.py`；
6. 再补 `policy.py`；
7. 跑 5 iter StageA.5 smoke；
8. 过 smoke 后再开 20 iter StageA.5 正式 compare；
9. 只有 StageA.5 证明不是 no-op 后，再推进 StageB。

这条路径最稳，因为它利用了当前代码里已经存在的两个事实：

- local gating 插口已经有；
- StageB pseudo/real backward 已经分开。

所以 SPGM 现在不是“要不要大改系统”，而是“要不要把现有 visibility-union grad mask 升级成更有结构的 Gaussian importance weight map”。

答案是：可以直接开做，而且应该先从 `StageA.5 + xyz-only + deterministic keep` 做。

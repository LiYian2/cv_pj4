# Part3 BRPO：Stage B 之前的工程落地方案

> 范围：基于当前仓库中已经存在的 `run_pseudo_refinement_v2.py`、`pseudo_camera_state.py`、`pseudo_loss_v2.py`、`pseudo_refine_scheduler.py`、以及 v1 里的 Gaussian 参数控制逻辑，补齐 **Stage B 之前** 必须完成的三件事：
>
> 1. 分离且尺度化的 absolute pose prior
> 2. 各 loss 分支的真实梯度贡献诊断
> 3. Stage A.5 / micro-joint refine

Hermes 标注:
我的执行规划（与文档轻微偏离）：
A. 先做 Part 1 的代码改造，但只做最小闭环，不立刻全量网格扫描。  
- 落地 split+scaled abs prior（lambda_abs_t/lambda_abs_r + scene scale + robust penalty）  
- 打通日志字段（rho/theta、trans/rot loss、scene_scale）  
- 先做 3~4 组短 smoke，确认数值区间合理

B. 紧接着做 Part 2 的“Exp-4 单步梯度剖面”（轻量版先上）  
- 先回答一个关键问题：当前到底是 rgb 主导、abs prior 主导，还是 depth 分支真的在拉 pose  
- 这一步做完，再决定 Part 1 的网格边界（可显著减少盲扫）

C. 然后再做 Part 1 的二维扫描（缩窄后的网格）+ 300iter 候选  
- 用诊断结果缩小搜索范围，而不是 9x2 全量起跑

D. 最后进 Part 3（Stage A.5）  
- 先 xyz，再看 xyz+opacity 是否必要  
- 只要 A.5 没给出明确增益，就不建议进 full Stage B

---

## 0. 当前现状与总判断

### 0.1 当前已成立的部分

- upstream 的 `train_mask + M5 densified depth target + source-aware depth loss` 已接通；
- `apply_pose_residual_()` 已补回 S3PO 式 residual fold-back，当前 Stage A 不再是假优化；
- 但 `loss_depth` 只有弱下降，说明 **链路通了，但优化牵引仍偏弱**。

### 0.2 当前真正的主矛盾

现在的问题不再是：
- mask 没接上；
- verified depth 太 sparse 导致完全没法训；
- pose residual 闭环断开。

现在的问题是：

1. `pose_reg` 只约束单步 residual，而 residual 每步都会被折回 `R/T` 并清零，所以它**不能约束累计 drift**；
2. 当前 absolute pose prior 还是一个未尺度化的单一标量，`0.1` 太弱，`100` 太强；
3. 当前 Stage A 完全冻结 Gaussian，depth 误差主要只能解释到 pose / exposure 上，几何自由度不足，导致 depth 改善偏弱。

### 0.3 当前总策略

因此 Stage B 之前，不建议直接上 full joint refine，而是先完成三步：

```text
A. 分离且尺度化 abs prior
B. 做 loss 分支梯度贡献诊断
C. 做 Stage A.5 micro-joint refine
```

只有这三步至少部分成立，进入 Stage B 才更可解释。

---

## 1. 分离且尺度化的 absolute pose prior

### 1.1 现状

当前仓库里已经有：
- `pseudo_camera_state.py::current_w2c()`
- `pseudo_loss_v2.py::absolute_pose_prior_loss()`
- `run_pseudo_refinement_v2.py --stageA_lambda_abs_pose`

但当前实现是：

\[
L_{abs}^{old} = \|\log( T_{cur} T_0^{-1})\|_2^2
\]

其中 `SE3_log` 输出 \(\tau=[\rho,\theta]\)，直接混合了：
- 平移项 \(\rho\)
- 旋转项 \(\theta\)

问题在于：
- 平移与旋转量纲不同；
- 不同 scene / frame 的有效平移尺度也不同；
- 一个单独的 `lambda_abs_pose` 同时压两件事，几乎必然出现“要么没感觉，要么压太死”。

### 1.2 为什么必须做这一步

当前 Stage A 里：
- residual 每步被 fold-back 并清零；
- 所以 `loss_pose_reg` 只能抑制“本步更新过大”，不能抑制“累计位姿已经慢慢漂了”。

因此 abs prior 的职责不是替代 `pose_reg`，而是补上：

**对当前位姿相对初始位姿的累计偏移约束。**

它要解决的是：
- pseudo pose 在弱监督下慢慢漂走；
- depth loss 没把相机拉向正确几何，反而让它在自由度里找局部解释；
- 训练后期出现 drift 累积但 `pose_reg≈0` 的情况。

### 1.3 新公式：分离 + 尺度化

设：
- 当前相机位姿为 \(T_{cur}\)
- 初始位姿为 \(T_0\)
- 相对变换为

\[
\Delta T = T_{cur} T_0^{-1}
\]

令

\[
\log(\Delta T) = [\rho,\theta] \in \mathbb{R}^6
\]

其中：
- \(\rho\in\mathbb{R}^3\)：平移切空间向量
- \(\theta\in\mathbb{R}^3\)：旋转切空间向量

定义 scene-aware translation scale：

\[
s_{scene} = \operatorname{median}( D_{render}[M_{train}] )
\]

其中：
- \(D_{render}\) 是当前 pseudo sample 的 `render_depth`
- \(M_{train}\) 是当前 Stage A 实际使用的 `train_mask`

然后把平移项归一化：

\[
\tilde{\rho} = \frac{\rho}{s_{scene}+\varepsilon}
\]

推荐新的 absolute pose prior：

\[
L_{abs} = \lambda_t \, \phi(\|\tilde{\rho}\|_2) + \lambda_r \, \phi(\|\theta\|_2)
\]

其中 \(\phi(\cdot)\) 推荐首版使用 Charbonnier 或 Huber：

\[
\phi(x)=\sqrt{x^2+\delta^2}
\]

而不是再直接用平方项。

### 1.4 参数推荐

先固定 robust 形式，不扫 loss 形式，只扫权重。

首版建议：

```text
translation normalization:
  s_scene = median(render_depth within train_mask and render_depth>1e-6)

robust penalty:
  phi(x) = sqrt(x^2 + 1e-6)

initial weight candidates:
  lambda_t ∈ {0.3, 1.0, 3.0}
  lambda_r ∈ {0.03, 0.1, 0.3}
```

经验逻辑：
- 旋转项通常更敏感，所以 `lambda_r` 先比 `lambda_t` 小一个量级；
- 如果 scene 很近、depth 中值很小，translation 归一化后数值会变大，因此更不该继续用未归一化 L2。

### 1.5 二维扫描怎么做

不再扫单一 `lambda_abs_pose`，改扫二维网格：

```text
Grid-A (default rgb-depth balance)
  lambda_t ∈ {0.3, 1.0, 3.0}
  lambda_r ∈ {0.03, 0.1, 0.3}

Grid-B (depth-heavy)
  同一网格，但 beta_rgb 改成 depth-heavy 版本
```

每组先跑 80 iter smoke，再选 2~3 组进 300 iter。

### 1.6 代码改动

#### 修改文件 1：`pseudo_loss_v2.py`

把现在的：
- `absolute_pose_prior_loss(viewpoint)`

改成两层：

```python
compute_abs_pose_components(viewpoint, scene_scale) -> {
    "rho_norm": ...,
    "theta_norm": ...,
    "rho_vec": ...,
    "theta_vec": ...,
}

absolute_pose_prior_loss_scaled(
    viewpoint,
    scene_scale,
    lambda_abs_t,
    lambda_abs_r,
    robust_type="charbonnier",
)
```

并在 `stats` 中新增：

```python
loss_abs_pose_trans
loss_abs_pose_rot
abs_pose_rho_norm
abs_pose_theta_norm
scene_scale_used
```

#### 修改文件 2：`run_pseudo_refinement_v2.py`

新增 CLI：

```text
--stageA_lambda_abs_t
--stageA_lambda_abs_r
--stageA_abs_pose_robust {charbonnier,huber,l2}
--stageA_abs_pose_scale_source {render_depth_trainmask_median,render_depth_valid_median,fixed}
--stageA_abs_pose_fixed_scale
```

并在每个 sample 加载后预计算：

```text
stageA_scene_scale
```

存进 `view` 字典，供 loss 直接用。

#### 修改文件 3：`pseudo_camera_state.py`

保留现在的 `R0/T0` 与 `export_view_state()`，但建议额外导出：

```text
abs_pose_rho_norm
abs_pose_theta_norm
scene_scale_used
```

#### 修改文件 4：`pseudo_refine_scheduler.py`

`StageAConfig` 增加：

```python
lambda_abs_t: float = 0.0
lambda_abs_r: float = 0.0
abs_pose_robust: str = "charbonnier"
```

### 1.7 实验设计

#### Exp-1：未尺度化 vs 尺度化

比较：
- old single-scalar abs prior
- scaled + split abs prior

固定其余设置，先跑 80 iter。

看：
- `loss_depth_seed`
- `loss_depth_dense`
- `abs_pose_rho_norm`
- `abs_pose_theta_norm`
- `loss_abs_pose_trans/rot`

#### Exp-2：二维扫描

分两套：
- default
- depth-heavy

每套先 9 组 × 80 iter，筛出：
- drift 抑制明显
- 且 depth 不恶化

再做 2~3 组 × 300 iter。

#### Exp-3：推荐默认值固化

从二维扫描里选一组作为：
- `recommended_default`
- `recommended_depth_heavy`

写回文档与 CLI 默认口径。

### 1.8 成功标准

满足下面两条即可：

1. 相比 no-abs，`abs_pose_rho_norm/theta_norm` 明显收敛；
2. 相比 old scalar abs prior，`loss_depth_seed/dense` 不再被明显压坏。

---

## 2. 各 loss 分支的真实梯度贡献诊断

### 2.1 现状

现在你已经能记录：
- `loss_rgb`
- `loss_depth`
- `loss_depth_seed`
- `loss_depth_dense`
- `loss_depth_fallback`
- `loss_pose_reg`
- `loss_abs_pose_reg`

但还不知道：

**这些分支谁真的在给 pose 提供有用梯度。**

尤其当前最大的不确定性是：
- dense 区域覆盖已经上来了；
- 但 dense depth 那部分到底是在真正拉动 pose，还是只是“loss 数值存在、优化影响很弱”。

### 2.2 为什么必须做这一步

如果不做梯度贡献诊断，后面所有判断都会混在一起：
- 是 abs prior 压坏了 depth？
- 还是 depth dense 本身对 pose 没什么梯度？
- 还是 RGB 一直在主导更新？

所以这一部分的目的不是再看 loss 曲线，而是回答：

**每个 loss term 对 `cam_rot_delta / cam_trans_delta / exposure` 的真实贡献有多大。**

### 2.3 代码改动

#### 新增文件：`scripts/diagnose_stageA_loss_contrib.py`

独立脚本，避免把诊断逻辑塞进主训练脚本。

职责：
- 载入一份已有 Stage A 配置；
- 对固定 sampled pseudo views 做一次 forward；
- 分别对各 loss term 单独 backward；
- 统计各参数组梯度范数。

#### 复用文件：`pseudo_loss_v2.py`

要求把 loss builder 改成支持：

```python
return total, stats, loss_terms
```

其中 `loss_terms` 至少包含：

```python
{
  "rgb": l_rgb,
  "depth_total": l_depth,
  "depth_seed": l_depth_seed,
  "depth_dense": l_depth_dense,
  "depth_fallback": l_depth_fallback,
  "pose_reg": l_pose,
  "abs_pose_trans": l_abs_t,
  "abs_pose_rot": l_abs_r,
  "exp_reg": l_exp,
}
```

#### 修改文件：`run_pseudo_refinement_v2.py`

只做轻量增强：
- history 里记录每 10 或 20 iter 的 aggregated gradient summary；
- 默认可关，通过 flag 开启：

```text
--stageA_log_grad_contrib
--stageA_log_grad_interval 20
```

### 2.4 要记录什么

每次诊断至少记录：

```text
for each loss term:
  grad_norm_rot
  grad_norm_trans
  grad_norm_exp_a
  grad_norm_exp_b
```

如果进入 A.5，还要加：

```text
grad_norm_xyz
grad_norm_opacity
```

同时记录辅助统计：

```text
pixel_count_train_mask
pixel_count_seed
pixel_count_dense
pixel_count_fallback
weighted_residual_rgb
weighted_residual_depth_seed
weighted_residual_depth_dense
```

### 2.5 实验设计

#### Exp-4：单步梯度剖面

在当前 best Stage A 配置上，选固定 3 个 pseudo views，做一次梯度分支诊断。

目标：看清：
- `rgb` 是否远大于 `depth_dense`
- `depth_seed` 是否主要拉 translation
- `abs_pose_trans/rot` 是否已经压过数据项

#### Exp-5：训练中梯度剖面

在 80 iter 训练中，每 20 iter 记录一次 aggregated gradient summary。

目标：看清：
- 前期是谁主导；
- 中后期是不是 abs prior 逐渐压过 depth；
- `depth_dense` 有没有随着 pose 稳定而逐渐变强。

### 2.6 成功标准

成功不是“所有梯度都很大”，而是：

1. `depth_seed` 与 `depth_dense` 对 pose 至少是非零、稳定、有解释性的；
2. abs prior 不会在一开始就彻底压制数据项；
3. 可以据此解释后面的 Stage A.5 / Stage B 结果。

---

## 3. Stage A.5 / micro-joint refine

### 3.1 现状

当前 `run_pseudo_refinement_v2.py` 只有 `stageA`：
- 优化 pseudo pose residual + exposure；
- Gaussian 完全冻结；
- densify / prune 不参与。

这能回答“pose 会不会动”，但不能回答：

**depth supervision 一旦允许少量几何自由度，是否会明显更有效。**

### 3.2 为什么必须做这一步

当前 depth 弱下降，一个非常可能的原因是：
- pseudo depth 监督真正想修的是几何；
- 但你把 Gaussian 全冻住了；
- 所以它只能把误差尽量解释到 pose / exposure 上，能力上限很低。

因此在 full Stage B 之前，最合理的过渡不是直接 joint all params，而是：

**只放开极少量 Gaussian 参数，验证 geometry 自由度一旦存在，depth 是否会更有响应。**

### 3.3 推荐的最小版本

首推两档：

#### 方案 A：只放 `xyz`

最保守，最干净。

适合回答：
- geometry 只允许轻微位置调整时，depth 是否就会明显改善。

#### 方案 B：放 `xyz + opacity`

比纯 `xyz` 略强。

适合回答：
- 除了位置外，是否还需要一点 visibility / occupancy 调整。

不建议首版就放：
- scaling
- rotation
- SH features
- full appearance

因为那样解释会迅速变混。

### 3.4 学习率建议

原则：Gaussian 比 camera 低一个量级。

首版推荐：

```text
camera:
  lr_rot   = 3e-3
  lr_trans = 1e-3
  lr_exp   = 1e-2

gaussian micro-joint:
  lr_xyz      = 1e-4
  lr_opacity  = 5e-4   # 仅当启用 opacity
```

### 3.5 代码改动

#### 修改文件 1：`run_pseudo_refinement_v2.py`

把：

```text
--stage_mode {stageA}
```

扩成：

```text
--stage_mode {stageA, stageA5}
```

新增 CLI：

```text
--stageA5_trainable_params {xyz,xyz_opacity}
--stageA5_lr_xyz
--stageA5_lr_opacity
--stageA5_disable_densify
--stageA5_disable_prune
```

并在主 loop 里：
- `stageA` 只建 pseudo optimizer；
- `stageA5` 建 pseudo optimizer + gaussian micro optimizer。

#### 修改文件 2：`pseudo_refine_scheduler.py`

新增：

```python
@dataclass
class StageA5Config(StageAConfig):
    trainable_params: str = "xyz"
    lr_xyz: float = 1e-4
    lr_opacity: float = 5e-4
```

新增 builder：

```python
build_stageA5_optimizers(pseudo_views, gaussians, cfg)
```

#### 复用 v1 逻辑：`run_pseudo_refinement.py`

v1 已经有：
- `ALL_PARAM_GROUPS`
- `resolve_pseudo_trainable_params()`
- param group 控制的思路

A.5 不需要照抄全部，但建议借它的写法，给 Gaussian micro optimizer 做最小版参数组选择。

#### 可选新增文件：`pseudo_branch/gaussian_param_groups.py`

如果你不想把 v1 逻辑复制进 v2，可以单独抽一个轻量 helper：

```python
def build_micro_gaussian_param_groups(gaussians, mode, lr_xyz, lr_opacity):
    ...
```

### 3.6 训练规则

A.5 必须固定：

```text
densify = off
prune = off
real branch = off   # 首版先只看 pseudo side 机制
```

原因很简单：
- 这一步要回答的是“少量几何自由度是否有帮助”；
- 不是要做 full refine。

### 3.7 实验设计

#### Exp-6：Stage A vs A.5(xyz)

比较：
- Stage A
- Stage A.5 with `xyz`

其余保持一致。

看：
- `loss_depth_seed`
- `loss_depth_dense`
- replay metrics
- `grad_norm_xyz`
- pseudo pose drift

#### Exp-7：Stage A.5(xyz) vs Stage A.5(xyz+opacity)

目标：判断 opacity 是否必要。

如果 `xyz+opacity` 提升不明显，后面 full Stage B 就应先从 geometry-only 开始，而不是一上来 joint all。

#### Exp-8：A.5 + best scaled abs prior

用第 1 部分选出的最佳 abs prior 配置，跑 A.5。

目标：看“drift 抑制”和“少量几何自由度”是否能兼容，而不是互相打架。

### 3.8 成功标准

A.5 成功只要满足下面任一条：

1. 相比 Stage A，`loss_depth_seed/dense` 下降更明显；
2. replay 指标开始出现正向改善；
3. `xyz` 的梯度贡献可见且没有导致训练不稳定。

若这三条全不成立，就不建议直接进入 full Stage B。

---

## 4. 三部分的执行顺序

建议严格按这个顺序：

```text
Step 1  分离且尺度化 abs prior
Step 2  loss 分支梯度贡献诊断
Step 3  Stage A.5 micro-joint refine
Step 4  再决定是否进入 full Stage B
```

原因：
- 先做 abs prior，是先把 drift 约束问题从“不可调”变成“可调”；
- 再做梯度诊断，是为了知道谁在主导训练；
- 最后再做 A.5，才能解释 geometry 自由度到底有没有帮助。

---

## 5. 最终建议：什么情况下进入 Stage B

只有满足下面至少两条，才建议进入 full Stage B：

1. 已找到一组 scaled abs prior，使 drift 收敛且 depth 不被明显压坏；
2. 梯度诊断表明 `depth_seed/dense` 对 pose 或 geometry 不是“几乎无贡献”；
3. Stage A.5 相比 Stage A 有实质改善。

否则，应该继续在 A / A.5 层做结构排查，而不是把问题带进更重的 joint training。

---

## 6. 为什么我们现在要做 abs pose prior？BRPO 原论文似乎没有这个模块，它是否有相关问题？

### 6.1 我们现在为什么要做

因为你现在的实现继承了 S3PO 风格的 residual pose 更新：
- 每步优化 `cam_rot_delta / cam_trans_delta`；
- step 后立刻 fold-back 到 `R/T`；
- residual 被清零。

这样做的好处是闭环正确，但副作用也很明确：

**单步 `pose_reg` 不再等价于累计 drift 约束。**

所以 abs pose prior 不是为了“让 pose 更难动”，而是为了补上一个当前实现里缺失的东西：

**约束当前位姿不要在弱监督下长期漂离初始位姿。**

它解决的是：
- `pose_reg≈0` 但累计 drift 仍在慢慢发生；
- depth / RGB 数据项较弱时，相机在自由度里寻找局部解释；
- 后期 pose 虽然每步都不大，但总偏移越来越不可控。

### 6.2 为什么 BRPO 原论文里看起来没有这个模块

BRPO 论文的方法重点是：
- pseudo-view restoration
- overlap score fusion + confidence mask
- scene perception Gaussian management
- joint optimization of Gaussian attributes and camera poses

也就是说，论文假设的是一套**完整 joint optimization 管线**，而不是你现在这种：
- 先做 standalone Stage A
- 只优化 pseudo pose + exposure
- Gaussian 全冻结或只准备做很轻的 A.5

因此论文里不显式提出 abs pose prior，并不奇怪。

### 6.3 它是否有相关问题

有，但形式不一样。

BRPO 论文明确承认两个核心困难：
- sparse-view 下 joint optimization 很难；
- pseudo-views 可能几何不一致，会把优化带坏；
- sparse input 会导致 Gaussian 分布不均、joint optimization 困难，因此才引入 confidence mask 与 SPGM。

这说明它**当然面临 pose / geometry 不稳定问题**，只是它把问题主要归因于：
- pseudo-view 不可靠；
- sparse constraint 不足；
- Gaussian 管理与 joint optimization 难。

而你现在额外出现 abs pose prior 需求，是因为：

1. 你是在 S3PO-style residual pose 机制上做 standalone Stage A；
2. 你当前还没进入 BRPO 的 full joint optimization 与 SPGM；
3. 所以“累计 drift 缺乏约束”在你这里会被更清楚地暴露出来。

一句话说：

**BRPO 原论文没有显式写 abs pose prior，不代表它没有 pose stability 问题；而是它试图用 confidence mask、joint optimization 和 scene perception Gaussian management 去整体缓解。你们当前因为先停在 Stage A / A.5，必须用一个更显式的工程约束把累计 drift 先管住。**


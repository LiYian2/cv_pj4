# Part3 BRPO M5 工程落地方案：Depth Correction Densify + Source-Aware Depth Loss

> 适用范围：当前 `part3_BRPO` 的 **mask-problem route on top of internal route**
> 
> 对齐现状：基于当前已经完成的 `M1 / M2 / M2.5 / M3 / M4 / M4.5`
> 
> 当前目标：解决 **verified depth 过稀**，让 `target_depth_for_refine` 真正提供可用的新几何信号，而不是继续主要依赖 `render_depth fallback`

---

## 1. 当前问题的精确定义

当前问题已经不是：
- BRPO verification 没做出来
- train mask 太稀，无法训练

当前已经确认：
- `train_mask coverage` 已经大约在 `18% ~ 20%`，进入可训练区间；
- 但 `verified depth coverage` 仍只有大约 `1.5% ~ 1.6%`；
- 当前 `target_depth_for_refine.npy` 中，约 `98%+` 仍来自 `render_depth fallback`；
- 因此 Stage A 中 depth loss 虽然**非零且已接通**，但对优化的牵引依然很弱。

所以当前主问题应定义为：

**depth-flatness problem on top of solved train-mask problem**

也就是：

**mask 已经够大，但 depth target 中真正来自 verified projection 的几何新信息太少。**

---

## 2. 为什么当前不建议先主打“放宽 verify 阈值”

当前 verify 的角色应继续视为：
- **高精度 seed provider**，不是 dense geometry generator。

因此当前不建议把主要精力放在：
- 放宽 `tau_reproj_px`
- 放宽 `tau_rel_depth`
- 直接让 `support` 尽量变大

原因：

1. 这只能增加 seed 数量，不会自然得到高质量 dense depth。
2. 一旦 verify 变松，最先损失的是 seed 的可信度。
3. 当前真正缺的是：
   - 如何把 sparse verified depth 变成 **可训练的 dense correction signal**；
   - 而不是如何把 verify 本身做成大 coverage。

因此当前主线判断是：

**verify 保持偏保守；新增一层受控 densify；densify 的对象不是 absolute projected depth，而是对 render depth 的局部 correction。**

---

## 3. M5 的核心思想

### 3.1 不是 densify absolute depth，而是 densify correction field

当前我们已有：
- `render_depth.npy`：dense，但几何新信息不足
- `projected_depth_left/right.npy`：可信，但极 sparse

如果直接传播 `projected_depth` 的绝对值，风险很高：
- 错 seed 会把错误深度直接扩散到大片区域
- 很容易写坏几何 target

因此 M5 的核心不应该是：
- 直接扩散 `projected_depth`

而应该是：
- 以 `render_depth` 为 dense base
- 在 verified 区域估计局部 **depth correction**
- 再把 correction 传播到 train-mask 范围内的局部区域

### 3.2 推荐 correction 形式：log-depth correction

在 verified 像素上定义：

```text
delta = log(projected_depth) - log(render_depth)
```

然后传播的是 `delta`，不是原始 depth。

最后重建 densified target：

```text
target_depth_dense = render_depth * exp(delta_dense)
```

这么做的好处：

1. 传播的是相对修正，不是绝对值，数值更稳定；
2. 自动保留 render depth 的局部形状与连续性；
3. 更符合当前 Stage A 的性质：
   - 当前 Stage A 不是从零学习新几何
   - 而是在已有地图基础上做 pseudo pose / exposure alignment warm-up。

---

## 4. M5 的工程目标

M5 需要完成两件事：

### 4.1 Upstream：生成 dense-corrected depth target

新增一个新的 depth 目标层：

```text
projected sparse verified depth
  + render depth
  ↓
local depth correction densify
  ↓
target_depth_for_refine_v2.npy
```

目标不是把 verified depth 变成 100% dense，而是：
- 比当前 `~1.5%` verified depth 更强；
- 但仍保持几何可信；
- 不引入大面积错误传播。

### 4.2 Downstream：让 Stage A depth loss 按 source 拆开

当前 depth loss 是把所有 `target_depth_for_refine` 混在一起算 masked L1。

M5 之后应显式区分：
- verified seed region
- densified-from-verified region
- render fallback region

也就是说，loss 不应再把“新几何区域”和“保底 fallback 区域”视为同一类监督。

---

## 5. 当前代码结构上的落点

### 5.1 现有 upstream 相关文件

当前相关文件：

- `pseudo_branch/brpo_reprojection_verify.py`
  - 负责输出 sparse verified depth / support
- `pseudo_branch/brpo_train_mask.py`
  - 负责 `seed_support -> train_mask`
- `pseudo_branch/brpo_depth_target.py`
  - 负责构造当前 `target_depth_for_refine`
- `scripts/brpo_build_mask_from_internal_cache.py`
  - 负责 verify / train-mask / projected depth 总线
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
  - 负责 pack sample schema

### 5.2 现有 downstream 相关文件

- `scripts/run_pseudo_refinement_v2.py`
  - Stage A consumer
- `pseudo_branch/pseudo_loss_v2.py`
  - Stage A loss

---

## 6. M5-1：新增 local depth correction densify 层

### 6.1 建议新增文件

新增：

```text
part3_BRPO/pseudo_branch/brpo_depth_densify.py
```

职责：
- 从 sparse verified depth 构造 dense correction field
- 输出 densified depth target
- 输出 source map / diagnostic maps

### 6.2 输入

对每个 sample，需要以下输入：

```text
render_depth.npy
projected_depth_left.npy
projected_depth_right.npy
projected_depth_valid_left.npy
projected_depth_valid_right.npy
train_confidence_mask_brpo_fused.npy
seed_support_left/right/both/single.npy
```

可选附加输入：

```text
render_rgb.png
target_rgb_fused.png
```

其中：
- `render_depth` 作为 dense base
- `projected_depth_*` 作为 sparse verified anchors
- `train_mask` 用于限制 densify 区域
- `seed_support_*` 用于区分 correction 来源

### 6.3 输出

建议输出：

```text
target_depth_for_refine_v2.npy
target_depth_dense_source_map.npy
depth_correction_seed.npy
depth_correction_dense.npy
depth_seed_valid_mask.npy
depth_dense_valid_mask.npy
depth_densify_meta.json
```

同时保留兼容字段：

```text
target_depth_for_refine.npy
```

在 M5 稳定前可以考虑：
- 保留旧 `target_depth_for_refine.npy`
- 新增 `target_depth_for_refine_v2.npy`
- Stage A 通过 mode 明确切换

等 M5 稳定后，再考虑让 `target_depth_for_refine.npy` 指向新版本。

---

## 7. M5-1 的 densify 算法建议

### 7.1 Step A：先构造 sparse correction seed

在左/右 verified depth 上先构造 seed：

```text
delta_left  = log(projected_depth_left)  - log(render_depth)
delta_right = log(projected_depth_right) - log(render_depth)
```

对于 `both` 区域：
- 默认用平均：

```text
delta_both = 0.5 * (delta_left + delta_right)
```

对 `left_only / right_only`：
- 用对应单边 `delta`

得到：

```text
delta_seed_sparse
seed_valid_mask
```

### 7.2 Step B：只在 train-mask 内 densify

densify 的候选区域不能是整张图，而应限制为：

```text
candidate_region = train_confidence_mask_brpo_fused > 0
```

原因：
- train mask 已经是当前最合理的 supervision 区域定义；
- 把 correction densify 限制在 train-mask 内，能显著降低错误传播风险。

### 7.3 Step C：用局部 patch / region 拟合 correction，而不是逐像素 flood fill

不建议直接像 `brpo_train_mask.py` 那样做 BFS 像素级传播。

推荐两种可落地方案：

#### 方案 A：patch-wise constant correction

把图像分成固定 patch，例如：

```text
patch_size = 9 或 11
stride = 4 或 5
```

每个 patch 内：
- 统计落在 patch 内的 verified seeds
- 若 seed 数 >= 最小阈值（如 5 或 8）
- 且 patch 内 seed 的 correction 方差足够小
- 则该 patch 内使用 seed correction 的鲁棒均值（median / trimmed mean）作为 patch correction

优点：
- 工程简单
- 最稳
- 容易可视化与诊断

缺点：
- correction 边界可能较块状

#### 方案 B：local affine correction in log-depth space

patch 内拟合：

```text
delta = a * log(render_depth) + b
```

或更简单：

```text
projected_log_depth = a * render_log_depth + b
```

只有当 patch 内 seed 足够多且残差足够小，才接受该 patch 模型。

优点：
- 比常数 correction 更柔和
- 更适合局部深度 slope 变化

缺点：
- 工程复杂度更高
- 首版更难调

### 7.4 当前推荐首版

**先做方案 A：patch-wise constant correction**

原因：
- 当前首要目标是证明 densify 后的 depth signal 能真正推动 Stage A
- 不必一开始就做过复杂的局部模型

首版推荐参数：

```text
patch_size = 11
stride = 5
min_seed_count = 6
max_seed_delta_std = 0.08   # log-depth space
max_patch_rel_depth_var = 0.05
```

### 7.5 Step D：重建 dense target

得到 `delta_dense` 后，重建：

```text
target_depth_dense = render_depth * exp(delta_dense)
```

对于无 densify 的区域：
- 保持旧逻辑 fallback 到 `render_depth`

因此最终：

```text
target_depth_for_refine_v2 =
  dense_corrected_depth      if dense_valid
  render_depth               otherwise
```

---

## 8. M5-1 的 source map 设计

当前 source map 只有：
- `both_fused`
- `left_only`
- `right_only`
- `render_fallback`

M5 后建议扩展成：

```text
0 = render_fallback
1 = seed_both_fused
2 = seed_left_only
3 = seed_right_only
4 = densified_from_seed
255 = no_depth
```

注意：
- `densified_from_seed` 应与原始 `seed_*` 区分
- 便于下游 loss 拆项
- 也便于后续统计：
  - 真正 seed 比例
  - densified 比例
  - fallback 比例

---

## 9. M5-2：Stage A depth loss 拆开

### 9.1 当前问题

现在的 `pseudo_loss_v2.py` 中：

```text
L_depth = masked L1(render_depth, target_depth_for_refine, confidence_mask)
```

这会把三类区域混在一起：
- verified seed
- densified correction
- render fallback

而当前真正的新几何信号只占极小一部分，因此容易被 fallback 区域“冲淡”。

### 9.2 建议新增/修改

建议修改：

```text
part3_BRPO/pseudo_branch/pseudo_loss_v2.py
```

新增一个 source-aware depth loss：

```text
L_depth = λ_seed * L_depth_seed
        + λ_dense * L_depth_dense
        + λ_fb   * L_depth_fallback
```

其中：

#### 1) `L_depth_seed`
只在 source map 中：
- `seed_both_fused`
- `seed_left_only`
- `seed_right_only`

上计算。

建议：
- 权重大
- 这是当前最可信的新几何区域

#### 2) `L_depth_dense`
只在：
- `densified_from_seed`

区域上计算。

建议：
- 权重中等
- 这是 M5 真正新增的可训练深度区域

#### 3) `L_depth_fallback`
只在：
- `render_fallback`

区域上计算。

建议：
- Stage A 中极小，甚至直接设为 `0`
- 因为 fallback 的目标几乎不提供新几何信息

### 9.3 建议默认权重

首版建议：

```text
lambda_depth_seed = 1.0
lambda_depth_dense = 0.35
lambda_depth_fallback = 0.0
```

这意味着：
- Stage A 的 depth 分支只真正关注：
  - verified region
  - 及其 densified correction region
- 不再让 render fallback 参与 depth loss 主体

### 9.4 当前 total loss 推荐写法

Stage A 中建议改成：

```text
L_total = β * L_rgb
        + (1 - β) * L_depth_source_aware
        + λ_pose * L_pose_reg
        + λ_exp  * L_exp_reg
```

当前可保持：

```text
β = 0.7
```

但在引入 source-aware depth loss 后，再观察是否需要调成：

```text
β = 0.6 或 0.65
```

当前不建议在 M5 前先动这个全局系数。

---

## 10. 需要修改/新增的文件

### 10.1 upstream

#### 新增

```text
pseudo_branch/brpo_depth_densify.py
```

#### 修改

```text
pseudo_branch/brpo_depth_target.py
scripts/brpo_build_mask_from_internal_cache.py
scripts/prepare_stage1_difix_dataset_s3po_internal.py
```

### 10.2 downstream

#### 修改

```text
pseudo_branch/pseudo_loss_v2.py
scripts/run_pseudo_refinement_v2.py
```

---

## 11. 建议的具体改动点

### 11.1 `brpo_depth_densify.py`

新增函数建议：

```python
build_sparse_log_depth_correction(...)
densify_depth_correction_patchwise(...)
reconstruct_dense_depth_from_correction(...)
build_depth_source_map_v2(...)
```

### 11.2 `brpo_depth_target.py`

新增：

```python
build_blended_target_depth_v2(...)
```

职责：
- 调 `brpo_depth_densify.py`
- 输出：
  - `target_depth_for_refine_v2`
  - `target_depth_dense_source_map`
  - `depth_densify_meta`

### 11.3 `brpo_build_mask_from_internal_cache.py`

新增 mode：

```text
depth_target_mode = blended_m3 | blended_m5_dense
```

初期建议不要替换旧 M3，而是并行保留：
- M3 baseline
- M5 densify 版

### 11.4 `prepare_stage1_difix_dataset_s3po_internal.py`

pack 阶段新增 sample 输出：

```text
target_depth_for_refine_v2.npy
target_depth_dense_source_map.npy
depth_correction_seed.npy
depth_correction_dense.npy
depth_densify_meta.json
```

并把 provenance 写入：

```text
source_meta.json
```

### 11.5 `run_pseudo_refinement_v2.py`

新增 depth mode：

```text
stageA_target_depth_mode = blended_depth_m3 | blended_depth_m5 | render_depth_only
```

history 里新增统计：
- mean seed depth ratio
- mean densified depth ratio
- mean fallback ratio

### 11.6 `pseudo_loss_v2.py`

新增：

```python
masked_depth_loss_by_source(...)
build_stageA_loss_source_aware(...)
```

同时保留旧版：
- 便于回退
- 便于 ablation

---

## 12. 建议的实验顺序

### Step 1：先做 upstream 产物，不改 refine

目标：
- 单独确认 M5 densify 后的 depth coverage 到底到多少
- 不和优化器问题混在一起

优先看三件事：

1. `seed ratio`
2. `densified ratio`
3. `fallback ratio`

建议目标区间：

```text
seed ratio         ≈ 1% ~ 3%
densified ratio    ≈ 8% ~ 20%
fallback ratio     余下部分
```

注意：
- 不追求 densified depth 覆盖和 train mask 一样大
- 重点是“比 1.5% 明显更有力，但仍然受控”

### Step 2：只切 Stage A depth target，不拆 loss

先比较：

```text
M3 blended_depth
vs
M5 blended_depth_dense
```

目的：
- 先确认 densify 本身是否让 `loss_depth` 更有变化

### Step 3：再切 source-aware depth loss

比较：

```text
M5 + old depth loss
vs
M5 + source-aware depth loss
```

目的：
- 分离“target 改好”与“loss 改好”的贡献

### Step 4：确认后再考虑 Stage B

只有满足下面至少两条，才建议进入 Stage B：

1. `loss_depth` 不再完全平
2. `blended_depth_m5` 比 `render_depth_only` 有稳定差异
3. replay / eval 上有正向信号

---

## 13. 不建议做的事情

当前不建议：

1. **直接大幅放宽 verify 阈值**
   - 这会优先破坏 seed 精度
   - 不是当前主矛盾

2. **直接把 train-mask 的传播逻辑复制到 depth 上**
   - mask 传播错一点，影响是权重错
   - depth 传播错一点，影响是几何 target 错

3. **在 M5 前直接进入 Stage B**
   - 当前 depth 还是太靠 fallback
   - Stage B 会把问题耦合得更重

4. **直接把 densified depth 覆盖做得非常大**
   - 当前更适合保守增量
   - 先证明 correction field 有效，再扩覆盖

---

## 14. 当前推荐默认配置（首版）

### M5 densify

```text
patch_size = 11
stride = 5
min_seed_count = 6
max_seed_delta_std = 0.08
candidate_region = train_mask > 0
correction_space = log_depth
```

### Stage A source-aware depth loss

```text
lambda_depth_seed = 1.0
lambda_depth_dense = 0.35
lambda_depth_fallback = 0.0
beta_rgb = 0.7
```

---

## 15. 最终判断

当前最合理的下一步不是：
- 继续主打 verify 阈值放宽
- 或直接进 Stage B

而是：

```text
M5 = local depth correction densification
   + source-aware depth loss
```

这条路线和当前代码结构最兼容，也最符合目前真实瓶颈：
- mask 已够大
- verify seed 够准但太少
- depth 需要的是“受控变 dense”，不是“简单变多”

一句话总结：

**下一步不是把 sparse verified depth 直接放大成 dense truth，
而是把它变成对 render depth 的局部几何修正，并让 Stage A 只重点优化这些真正携带新几何信息的区域。**

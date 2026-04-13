# CURRENT_MASK_PROBLEM.md

> 最后更新：2026-04-12 20:40
> 主题：mask problem 第一阶段已基本解决；Stage A 的 S3PO residual pose 闭环已修回。当前新问题是：**depth 仍是弱下降，且 absolute pose prior 呈现明显权重尺度问题（0.1 太弱，100 太强）。**

---

## 1. 当前一句话结论

现在不能再把问题简单表述成“BRPO mask 太稀”。更准确的说法是：

- **train mask 这一层已经基本可用**：`train_mask coverage ≈ 19.4%`，处于此前判断更合理的 `10% ~ 25%` 区间；
- **原始 verified depth 的确太稀**：M3 下真正来自 verified projection 的区域只有 `~1.56%`；
- **M5 densify 已经把 depth 新信息区域显著抬高**：在 selected M5 参数下，`train_mask` 内非 fallback 区域已经到 `~81.2%`；
- **在补回闭环后，Stage A 的 depth loss 已经不是完全平线，但仍只表现为弱下降**；
- 最新 M5-4 对照显示：**absolute pose prior 已接入，但当前权重还没到可用区间**（0.1 无感，100 会压 depth）。

一句话说：

**mask problem 已经从“train mask 太稀”演变成“upstream depth target 已增强，但我们当前 Stage A 没有沿用 S3PO 原始的 `tau -> update_pose -> R/T` 闭环，导致 `theta/rho` residual 只在 backward 起作用、却没有在训练循环里被正确折回相机位姿”。**

---

## 2. 当前关键统计

当前统计基于：

```text
/home/bzhang512/CV_Project/output/part2_s3po/re10k-1/
  s3po_re10k-1_full_internal_cache/
  Re10k-1_part2_s3po/2026-04-11-05-33-58/
  internal_prepare/re10k1__internal_afteropt__brpo_proto_v4_stage3/pseudo_cache/samples/
```

样本帧：`10 / 50 / 120`

### 2.1 M3 之前已经确认的 coverage

- `seed_support_union_coverage ≈ 1.5608%`
- `train_mask_coverage ≈ 19.4126%`
- `verified_depth_coverage ≈ 1.5608%`

这说明：
- `seed_support -> propagation -> train_mask` 这条链已经把训练 mask 从 seed-level 扩成了可训练覆盖；
- 但 `verified depth` 在 M3 里仍然停留在 seed/support 级别。

### 2.2 M5-0：当前 depth signal 被覆盖结构严重限制

M5-0 的诊断重点不是再看整图，而是看 **train_mask 内部**。

结果：
- `mean_train_mask_coverage ≈ 19.41%`
- `mean_verified_depth_coverage ≈ 1.56%`
- `mean_verified_within_train_mask_ratio ≈ 8.05%`
- `mean_render_fallback_within_train_mask_ratio ≈ 91.95%`

解释：
- 在训练真正消费的 mask 区域里，只有大约 `8%` 的像素落在 verified depth 区域；
- 剩下大约 `92%` 虽然也在 mask 里，但 depth target 仍然只是 fallback。

所以 M4.5 的“depth 不动”不只是因为 depth loss 值小，而是因为：

**当前 train_mask 内真正携带新几何信息的区域比例太小。**

### 2.3 M5-1：densify 之后的 depth coverage 已经明显增强

默认参数（`patch=11, stride=5, min_seed=6, max_std=0.08`）过于保守，只把 densified 区域抬到 `~2.08%`，总 non-fallback 也只有 `~3.64%`，不够。

经过小范围 sweep 后，当前选中的一组参数是：

```text
patch_size = 11
stride = 5
min_seed_count = 4
max_seed_delta_std = 0.08
```

在这组参数下：
- `mean_seed_ratio ≈ 1.56%`
- `mean_densified_ratio ≈ 14.21%`
- `mean_total_nonfallback_ratio ≈ 15.77%`
- `mean_render_fallback_ratio ≈ 84.23%`

更关键的是，在 `train_mask` 内部：
- `nonfallback_within_train_mask_ratio ≈ 81.22%`
- `seed_within_train_mask_ratio ≈ 8.05%`
- `dense_within_train_mask_ratio ≈ 73.17%`

这说明 M5-1 已经把“真正有新几何信息的区域”从 seed 级抬到了**train_mask 的大部分区域**。

所以到这一步，问题已经不再是“depth 还是太 sparse，什么都没法做”。

---

## 3. 逻辑链：train mask、verified depth、densified depth 到底是什么关系

当前链路应理解为：

```text
left/right verify
  ↓
seed_support_left/right/both/single
  ├─→ propagation → train_confidence_mask_brpo_*
  └─→ projected_depth_left/right (仅 support 像素)
                      ↓
                M3: blended target_depth_for_refine
                      ↓
                M5: densify correction field
                      ↓
              target_depth_for_refine_v2.npy
```

这意味着当前有三层不同的概念：

1. **train mask**
   - 回答“哪些像素参与监督、权重多大”
   - 来自 `seed_support -> propagation`

2. **verified depth**
   - 回答“哪些 support 像素有高置信深度值”
   - 来自 `projected_depth_*`
   - M3 下只覆盖 `~1.56%`

3. **densified depth**
   - 回答“如何把 sparse correction 受控地扩进 train-mask 区域”
   - 来自 `render_depth + sparse log-depth correction + patch-wise densify`
   - M5-1 下已经能把非 fallback 提到 `~15.8%` 整图、`~81.2%` 的 train-mask 内部区域

因此当前不能再说：
- verified depth 会随着 train mask 变大而自动变大；
- 或 M5 之后的 dense 区域还只是“和以前一样太 sparse”。

到 M5-1 为止，**upstream depth target 已经从“太 sparse 无法讨论”提升到“足够强，应该能测试 downstream 是否真的在用”。**

---

## 4. 当前 loss 结构，以及为什么新的问题不再只是 coverage

### 4.1 旧的 Stage A 结构

当前 Stage A 的总 loss 仍然大致是：

```text
L = β * L_rgb
  + (1-β) * L_depth
  + λ_pose * L_pose_reg
  + λ_exp * L_exp_reg
```

这里：
- `L_rgb` 在 `train_mask` 上算；
- `L_depth` 在当前 target depth 上算；
- Stage A 只更新 pseudo pose delta 和 exposure，不改 Gaussian / PLY。

### 4.2 M5-2 之后的结果

第三阶段（M5-2）我已经把 consumer 接成两组：

1. `M5 densified target + old depth loss`
2. `M5 densified target + source-aware depth loss`

结果很关键：
- `M5 + old depth loss`：`loss_depth ≈ 0.02770`，300 iter 中基本不动；
- `M5 + source-aware depth loss`：`loss_depth ≈ 0.06867`，300 iter 中也基本不动；
- 其中 source-aware 已经显式拆成：
  - `loss_depth_seed ≈ 0.05796`
  - `loss_depth_dense ≈ 0.03062`
  - `loss_depth_fallback ≈ 0`

这意味着：
- 当前问题已经**不再只是 fallback 稀释**；
- 因为就算把 fallback 基本去掉，`seed` 和 `dense` 两项还是不动。

所以到这一步，问题从“depth 太 sparse”继续推进成：

**depth target 已经增强，但 Stage A 仍然没有把它转化成有效的 pose/render 对齐。**

---

## 5. 已确认的根因，以及最小修复后的新状态

### 5.1 根因不是“CUDA 单纯写坏了”，而是我们没有沿用 S3PO 的原始 residual pose 闭环

S3PO 原始代码里，`cam_rot_delta / cam_trans_delta` 是一步一清的 pose residual：
- `camera_utils.py` 把它们定义成 `nn.Parameter`；
- `slam_frontend.py` / `slam_backend.py` 在每次 `optimizer.step()` 后都会立即调用 `update_pose(viewpoint)`；
- `pose_utils.py::update_pose()` 会把 `tau=[rho, theta]` 通过 `SE3_exp(tau)` 折回到当前 `R/T`，然后把 residual 清零。

所以 S3PO 的原始闭环其实是：

```text
render(using current R/T)
  -> backward gives grad_tau
  -> optimizer.step()
  -> update_pose() folds tau into R/T
  -> next iteration render uses updated R/T
```

而我们之前的 Stage A 是：
- 保留了 `cam_rot_delta / cam_trans_delta`；
- 保留了 backward 给它们的梯度；
- 但**没有在每步后调用 `update_pose()` 或等价逻辑**。

因此真正的问题是：

**我们把 S3PO 的 residual pose 机制用断了。**

### 5.2 代码链路审计结果

代码审计确认了两件事：

1. `gaussian_renderer/__init__.py` 的确把 `theta/rho` 传到了 `diff_gaussian_rasterization` Python wrapper；
2. 但 `diff_gaussian_rasterization` 的 forward 路径实际只吃 `viewmatrix / projmatrix`，不把 residual 当成持久相机状态直接用于当前 render。

这和 S3PO 的原始设计并不矛盾，因为它本来就依赖 **step 后立刻 `update_pose()`** 来让下一轮 render 使用更新后的 base pose。

### 5.3 最小修复已经完成

当前已经在 Part3 的 pseudo viewpoint 层补了等价逻辑：
- 新增 `pseudo_branch/pseudo_camera_state.py::apply_pose_residual_()`；
- 在 `run_pseudo_refinement_v2.py` 的每次 `optimizer.step()` 后，对本轮参与优化的 pseudo views 调它。

它做的事情就是：
- 用当前 `tau=[cam_trans_delta, cam_rot_delta]` 计算新的 `w2c`；
- 折回 `vp.R / vp.T`；
- 刷新 `world_view_transform / full_proj_transform / camera_center`；
- 把 residual 清零。

这一步本质上就是把 S3PO 的 `update_pose()` 机制接回到 Part3 Stage A。

### 5.4 最小验证结果：闭环接回后，Stage A 不再是假优化

#### 验证 1：手动 residual -> fold 到 R/T -> render

修复前，哪怕把 `rot/trans` 调到 `0.2 / 0.2`，render RGB / depth 仍然 `0.0` 变化。

修复后，手动设置 residual 再调用 `apply_pose_residual_()`，render 会立刻变化：
- `(0.01, 0.01)`：`rgb_mean ≈ 0.0679`, `depth_mean ≈ 0.1648`
- `(0.05, 0.05)`：`rgb_mean ≈ 0.1604`, `depth_mean ≈ 0.4364`
- `(0.10, 0.10)`：`rgb_mean ≈ 0.2412`, `depth_mean ≈ 0.8877`

这说明修复后，pose 更新终于能真实影响下一轮 render。

#### 验证 2：M5 source-aware 80 iter smoke

对比：
1. `no_apply_pose_update`（旧坏链路）
2. `apply_pose_update`（修复后闭环）

结果：
- **旧坏链路**：
  - `loss_depth: 0.06867 -> 0.06867`（完全平）
  - `pose_updates_total = 0`
  - 最终 residual 仍堆在 `rot_norm ~ 0.27~0.41`, `trans_norm ~ 0.13`

- **修复后闭环**：
  - `loss_total: 0.02701 -> 0.02570`
  - `loss_depth: 0.06867 -> 0.06826`
  - `loss_depth_seed: 0.05796 -> 0.05760`
  - `loss_depth_dense: 0.03062 -> 0.03046`
  - `pose_updates_total = 240`
  - 最终 residual 全部回到 `0.0`

这说明：
- 修复后，Stage A 不再是之前那种“forward 不动、backward 乱推”的假优化；
- 但也要老实说：目前 depth 只是**开始轻微下降**，还没有出现非常强的下降趋势。

也就是说，这次修复证明了：

**之前 loss 不动的主要根因之一确实是 pose 闭环没接回；但把闭环接回之后，depth supervision 目前仍然只是弱有效，而不是一下子变得非常强。进一步的 300-iter 验证表明，这个“弱有效”不是 80 iter 偶然现象，而是当前参数/结构下的稳定状态。**


## 5.5 修复后的 300-iter 验证：不是“完全无效”，但目前仍然偏弱

在修回 `apply_pose_residual_()` 之后，又做了三组 300-iter 验证：

1. `default_300`
   - `beta_rgb=0.7`
   - `lambda_depth_seed=1.0`
   - `lambda_depth_dense=0.35`
   - `lr_exp=0.01`

2. `depth_heavy_300`
   - `beta_rgb=0.3`
   - `lambda_depth_seed=1.0`
   - `lambda_depth_dense=1.0`
   - `lr_exp=0.01`

3. `no_exposure_300`
   - `beta_rgb=0.7`
   - `lambda_depth_seed=1.0`
   - `lambda_depth_dense=0.35`
   - `lr_exp=0.0`

结果：

- `default_300`
  - `loss_total: 0.02701 -> 0.02577`
  - `loss_depth: 0.06867 -> 0.06816`
  - `loss_depth_seed: 0.05796 -> 0.05750`
  - `loss_depth_dense: 0.03062 -> 0.03047`

- `depth_heavy_300`
  - `loss_total: 0.06475 -> 0.06364`
  - `loss_depth: 0.08857 -> 0.08747`
  - `loss_depth_seed: 0.05796 -> 0.05731`
  - `loss_depth_dense: 0.03062 -> 0.03016`

- `no_exposure_300`
  - `loss_total: 0.02701 -> 0.02696`
  - `loss_rgb: 0.00915 -> 0.00928`（几乎不降，甚至略坏）
  - `loss_depth: 0.06867 -> 0.06822`

这说明：
- 修复后，depth 的确已经不是“完全平线”；
- 但无论 default、加重 depth、还是冻结 exposure，目前都只是**弱下降**；
- `depth_heavy_300` 比 default 稍强一点，但也没有出现质变。

### 5.6 修复后暴露出来的新结构问题

当前还有一个新的结构点需要明确：

**在每步都 `apply_pose_residual_()` 之后，当前 `pose_reg_loss = ||cam_rot_delta|| + ||cam_trans_delta||` 基本会回到 0。**

这意味着：
- `stageA_lambda_pose` 现在不再约束“累计位姿偏移”；
- 它只能约束当前一步 residual，但 residual 在 step 后立即清零；
- 所以从设计上说，当前 Stage A 已经缺少了一个“absolute pose drift prior”。

这和实验现象是吻合的：
- 300 iter 后真实 pose 的累计变化并不大，但确实存在；
- `default_300` 的平均位移变化约 `0.00138`，平均旋转变化约 `0.00498 rad`；
- `depth_heavy_300` 的平均位移变化约 `0.00390`，平均旋转变化约 `0.00453 rad`；
- `no_exposure_300` 的平均位移变化约 `0.00141`，平均旋转变化约 `0.00463 rad`。

这说明当前问题已经进一步收敛成：

**闭环修复后，优化终于是真的了；但当前 depth signal 只在较小尺度上推动位姿变化，而且当前 loss 结构里缺少一个对“累计 pose drift”的明确约束。**


## 6. 现在应该如何判断问题

到当前阶段，问题已经不再是单一的“depth coverage 少”。更准确地说，有两个层次：

### 6.1 Upstream 问题：基本已经推进到可用程度

- train mask 已基本可用；
- M5 densify 也已把 non-fallback depth 区域抬到了有训练意义的量级；
- 因此 upstream 现在**不是最先该怀疑的地方**。

### 6.2 Downstream 问题：根因已修，当前进入“修复后效果是否足够强”的判断阶段

当前最优先的问题已经变成：

**Stage A 的 pose 闭环已经接回，当前更需要判断的是：修复后这套 depth supervision 是否足够强，值得继续放大实验或推进下一阶段。**

在这个问题没查清之前：
- 现在已经可以重新开始相信 Stage A 的 loss 曲线有解释价值；
- 修复后的 300-iter 结果表明 depth 确实会下降，但仍然只是弱下降；
- 同时新的结构问题也暴露出来了：当前 `pose_reg` 无法约束累计位姿偏移；
- 因此下一步更合理的是在不进入 Stage B 的前提下，继续做参数/结构层面的诊断。

---

## 7. 当前结论（供后续引用）

当前结论固定为：

1. **train mask 问题已经基本解决。**
   当前 `train_mask coverage ≈ 19.4%`，已不再只是 seed-level support。

2. **verified depth 原始版本确实太稀，但 M5 densify 已显著改善。**
   当前 M5 selected 参数下，整图 `non-fallback ratio ≈ 15.8%`，在 `train_mask` 内部约 `81.2%` 的区域已是非 fallback depth。

3. **M5-2 说明仅靠 densify + source-aware loss 仍不足以让 Stage A 的 depth loss 动起来。**
   这意味着问题不再只是 fallback 稀释。

4. **当前最大的实现问题已经确认并完成最小修复。**
   Stage A 现在会在每次 step 后把 residual 折回 `R/T`，不再是之前的假闭环。

5. **因此当前仍不适合直接进入 Stage B。**
   根因虽然修了，但修复后 depth 只出现轻微下降，还需要先验证这是否足以支撑更大阶段。

6. **当前最合理的下一步是：**
   做 300-iter absolute prior 权重扫描（建议 10/30/100）与尺度化版本对照，先找出“抑制 drift 且不伤 depth”的区间，再决定是否推进下一阶段。


## 8. M5-4：absolute pose prior 之后，depth 现状怎么判断

基于 `2026-04-12_m56_abs_pose_eval` 的 60-iter 对照，当前 depth 现状可以明确成三点：

1. `lambda_abs_pose=0.1` 基本等同 noabs。
   - default 与 depth-heavy 两组里，`loss_depth` 与 `abs_pose_norm` 都几乎重合。

2. `lambda_abs_pose=100` 能明显抑制累计 drift。
   - `abs_pose_norm mean` 大约从 `0.001535` 降到 `0.000338`。

3. 但 `lambda_abs_pose=100` 没有带来 depth 改善。
   - default 组里 `loss_depth` 反而略变差（`0.068364 -> 0.068543`）。

结论：

当前 depth 的问题已经不是“有没有接上”，也不是“有没有 drift 抑制”，而是 **drift 抑制与 depth 对齐之间的权衡没有标定好**。

因此“100 也算不上很有效果”这个判断是成立的：
- 它对 drift 有效；
- 但对我们更关心的 depth 改善并不有效，至少在当前实现和权重下如此。

下一步应是 300-iter 权重扫描 + 尺度化 prior，而不是把 100 当默认解。

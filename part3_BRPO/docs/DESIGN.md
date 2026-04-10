# DESIGN.md - Part3 设计文档

> 本文档记录**当前实际采用**的设计决策、接口定义与默认实验方案。
> 最后更新：2026-04-10 12:18

---

## 1. 系统边界

### 1.1 核心约束

**Pseudo view 只参与 mapping，不参与 tracking。**

当前实现中：
- pseudo 不进入 S3PO frontend；
- pseudo 不参与 keyframe selection / tracking / window pose optimization；
- Stage 1 通过独立脚本 refinement 验证，不接入 `slam.py` 主队列。

### 1.2 Stage 划分

| Stage | 状态 | 说明 |
|-------|------|------|
| **Stage 1** | 当前主线 | external pseudo branch + standalone refine |
| **Stage 2** | 后续规划 | backend 集成、EDP 角色再设计、RAP/3D audit |

---

## 2. Stage 1 当前实现

### 2.1 数据准备

入口：`scripts/prepare_stage1_difix_dataset_s3po.py`

三阶段：
1. `select`：在 sparse train gap 中选择 pseudo sample；
2. `difix`：分别基于左右参考帧修复 pseudo RGB；
3. `pack`：写 `camera.json` / `refs.json`，并 symlink `render_rgb` / `render_depth` / `target_rgb_*`，形成 `pseudo_cache/`。

### 2.2 深度与置信度

入口：`pseudo_branch/build_pseudo_cache.py`

当前两种路线：
- **GT 重投影**：主要用于 405841；
- **EDP**：主要用于 Re10k-1 / DL3DV-2。

两条路线当前语义不同：
- **EDP 路线（Re10k-1 / DL3DV-2）**：`target_depth = left/right EDP fused depth`，`confidence_mask = fused EDP confidence`。`render_depth.npy` 会保留在 cache 里，但主要用于诊断图与辅助对齐，不是直接把 render depth 当 target depth 喂给 refine。
- **GT 重投影路线（405841）**：先把左右参考深度重投影到 pseudo 视角，再用 `render_depth.npy` 做一致性筛选；当前写出的 `target_depth` 本质上是“通过一致性检查保留下来的 render depth 子集”，`confidence_mask` 目前是二值有效性掩码。

注意：当前系统里 `target_depth` **不直接作为 refine loss 的几何监督**；refine 仍然使用 confidence-weighted pseudo RGB loss。

### 2.3 Refinement

入口：`scripts/run_pseudo_refinement.py`

当前支持三层控制：
1. **loss 组成**：pure pseudo / joint refine；
2. **densify 统计来源**：pseudo / real / mixed；
3. **pseudo 分支可更新参数范围**：all / appearance / geometry / none / 自定义子集。

---

## 3. Loss 与梯度控制设计

### 3.1 Loss 计算在哪里

- `slam_utils.py` 负责 **loss 的定义**：
  - `get_loss_mapping_rgb()`：real branch RGB reconstruction loss；
  - `get_loss_pseudo()`：pseudo branch confidence-weighted RGB loss。
- `run_pseudo_refinement.py` 负责 **loss 如何回传到哪些参数**。

### 3.2 当前 pseudo loss

`get_loss_pseudo()` 当前定义：

```python
L_pseudo = ||C ⊙ (I_render - I_target)||_1 / ||C||_1
```

其中：
- `C = confidence_mask`
- `target_depth` 不直接参与 loss
- depth 只表示可靠性，不做显式几何监督

### 3.3 当前 total loss

joint refine 下：

```text
L_total = λ_real * L_real + λ_pseudo * L_pseudo + L_reg
```

其中：
- `L_real`：真实 sparse train RGB reconstruction
- `L_pseudo`：confidence-weighted pseudo RGB
- `L_reg`：isotropic scaling regularization

### 3.4 参数更新控制

S3PO `GaussianModel` 当前 optimizer param groups：
- `xyz`
- `f_dc`
- `f_rest`
- `opacity`
- `scaling`
- `rotation`

当前 refine 采用 split backward：
1. 先对 pseudo branch backward；
2. 清掉 pseudo 不允许更新的 param-group gradients；
3. 再对 real branch backward（并带 reg）。

效果：
- pseudo 可仅改 appearance；
- real anchor 仍可更新几何；
- 不需改 `slam_utils.py` 的 loss 公式。

### 3.5 关键策略变量（解释）

- `lambda_pseudo`：**pseudo 分支权重**。值越大，pseudo 监督对总梯度影响越强。
- `freeze_geometry_for_pseudo`：禁止 pseudo loss 直接改 `xyz/scaling/rotation`。
- `densify_stats_source`：决定 densify 统计由 real/pseudo/mixed 哪一支驱动。

---

## 4. 当前实验结论与默认策略

### 4.1 A/B/C/D/E 结果（Re10k-1 sparse, external GT）

| 组别 | 配置摘要 | PSNR | SSIM | LPIPS | final_gaussians |
|------|----------|------|------|-------|-----------------|
| A | all-params + midpoint pseudo + real densify + λp=1.0 | 20.1598 | 0.7284 | 0.2960 | 48951 |
| B | freeze-geom + midpoint pseudo + real densify + λp=1.0 | 20.9542 | 0.7575 | 0.2613 | 39916 |
| C | freeze-geom + midpoint pseudo + real densify + λp=0.5 | 21.5173 | 0.7797 | 0.2242 | 41544 |
| D | freeze-geom + midpoint pseudo + disable_densify + λp=1.0 | 20.6968 | 0.7518 | 0.2484 | 35639 |
| E | freeze-geom + tertile pseudo + real densify + λp=0.5 | **21.7658** | **0.7879** | **0.2110** | 42454 |

baseline sparse GT（同口径 eval）：`PSNR 16.9367 / SSIM 0.6526 / LPIPS 0.1722`

### 4.2 这些结果目前**能支持什么**

到目前为止，这些结果稳妥支持的是：
1. 在 **external GT** 这个固定相机诊断口径下，pseudo refine 确实能把渲染结果往 GT 方向推；
2. `freeze_geometry_for_pseudo=True`、`pseudo_trainable_params=appearance`、`lambda_pseudo=0.5`、`tertile pseudo` 这组策略，在这个口径下是当前最好的 appearance-refine 组合；
3. DL3DV-2 sparse 上的 GT 口径结果也说明，这种 fixed-pose 渲染改善不是 Re10k-1 单数据集偶然现象。

### 4.3 这些结果目前**还不能支持什么**

它们**还不能直接支持**以下结论：
- refine 已经改善了 external infer 的 pose-sensitive 表现；
- refine 已经改善了 internal tracked-camera protocol；
- refine 已经真正提升了 end-to-end pipeline。

原因不是抽象怀疑，而是已有反例：
- Re10k-1 最新 `infer` 口径结果只有 `PSNR 13.6111 / SSIM 0.5067 / LPIPS 0.4061`；
- 对比 part2 sparse infer baseline：`13.3265 / 0.5027 / 0.3791`；
- `pose_success_rate = 67.4%`，说明 pose inference 本身就是当前瓶颈。

因此，当前更合理的解释是：**Stage 1 现有 refine 更像 fixed-pose appearance refinement，而不是 pose refinement。**

### 4.4 当前推荐默认策略（带前提）

如果目标是做 **external GT diagnostic 下的 appearance baseline**，当前推荐默认策略仍是：
- joint refine
- `freeze_geometry_for_pseudo=True`
- `pseudo_trainable_params=appearance`
- `densify_stats_source=real`
- `lambda_real=1.0`
- `lambda_pseudo=0.5`
- pseudo 采样优先使用 **tertile 双点**

但请注意：
- 这只是当前 **GT diagnostic baseline**；
- 不应再把它直接写成“当前最佳最终方案”。

### 4.5 当前最高优先级：internal replay 验证

在继续调 E / fused / lambda 之前，当前更高优先级是：
1. 导出 internal cache（before_opt / after_opt）
2. 对 baseline PLY 与 refined PLY 做 **同一组 internal camera states 的 replay render / replay eval**
3. 先回答“refined PLY 在 internal protocol 下是否真的更好”

只有这一步成立，后续 internal pseudo prepare、fused supervision、乃至 backend 集成才有明确方向。

### 4.6 DL3DV-2 sparse 迁移结果的重新解读

2026-04-09 DL3DV-2 sparse 的 E 结果：
- E：`PSNR 14.0617 / SSIM 0.4526 / LPIPS 0.6893`
- baseline sparse GT（同口径）：`PSNR 8.3061 / SSIM 0.2884 / LPIPS 0.7583`

这说明：E policy 在 **external GT** 下同样能改善 fixed-pose 渲染。

但在 infer/internal 口径没有补证前，它仍然只应被表述为：
**GT diagnostic 迁移成功**，而不是“完整协议已验证”。

---

## 5. 下一步优化方向（按优先级重排）

### 5.1 第一优先级：协议补证，而不是继续调分

1. **internal cache 导出**：先把 before_opt / after_opt 的 tracked camera states 与 render cache 存下来。
2. **replay_internal_eval**：对 baseline / refined PLY 做同协议 replay，对比 internal non-KF render 指标。
3. **判定 refine 性质**：如果 replay 也提升，说明 refine 至少对 internal map/render 有真实收益；如果 replay 不提升，则说明当前 Stage 1 更偏 fixed-pose appearance tuning。

### 5.2 第二优先级：只有 replay 成立后再继续策略改动

若 internal replay 能证明 refined PLY 确有收益，再继续：
1. `E` 基线上做 `lambda_pseudo=0.25`
2. `E` 基线上做 `disable_densify` 交叉对照
3. 继续推进 fused pseudo target
4. 细化 appearance 子集：`opacity only`、`f_dc+f_rest`

### 5.3 405841 的位置

405841 不应在协议尚未澄清时提前大规模推进。更合理顺序是：
- 先完成 internal replay 闭环；
- 再决定 405841 是沿用 external/EDP 还是直接转 internal 路线。

---

## 6. 评估口径与指标实现

### 6.1 external GT：固定相机诊断口径

- `pose_mode=gt`
- `origin_mode=test_to_sparse_first`
- `infer_init_mode=gt_first`

作用：
- 用于判断在**准确相机**下，这张 PLY 的渲染质量如何；
- 适合做 appearance/map 的固定相机诊断；
- 不应再单独承担“完整 pipeline 成功”的结论。

### 6.2 external infer：pose-sensitive stress test

- pose 由 `infer_poses_sequential()` 逐帧顺序估计；
- 估计过程中直接依赖当前 PLY 渲染深度与 MASt3R/PnP；
- 当前结果说明，该口径高度受 pose success rate 影响。

因此 external infer 更适合回答：
- 这张 PLY 是否更利于外部顺序 pose inference；
- map 变化是否足以在 pose-sensitive 协议下体现出来。

### 6.3 internal protocol：当前真正需要补齐的目标口径

internal protocol 的核心不是 pose 是否接近 GT，而是：
- camera state 与 map 在同一轮 SLAM 中共同形成；
- render 发生在 internal tracked camera states 下；
- 更能反映“这张 map 是否与系统自身的相机状态自洽”。

这也是当前 `01_INTERNAL_CACHE.md` 想解决的核心问题。

### 6.4 LPIPS 生成方式（明确说明）

Part3 external eval 的 LPIPS 来自：
- `third_party/S3PO-GS/utils/external_eval_utils.py`
- `LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)`

因此：
- **不是 VGGT 指标**；
- 与 Part1/Scaffold 常见的 `lpips(net=vgg)` 口径不同，跨 pipeline 比较时需注明。

### 6.5 数据接触边界说明

- Refine 阶段：输入为 sparse train + pseudo cache，不直接读取 test GT RGB/GT pose。
- external GT 评估：读取 test GT pose 与 test GT RGB。
- external infer 评估：不使用 GT pose 渲染，但仍用 GT RGB 计算指标。
- internal replay（待实现）：不使用 GT pose 渲染，但仍需 GT RGB 计算指标。

---
## 7. 参考与保留文档

当前 `docs/` 下仍有参考价值的非状态类文档：
- `project4_part3_merged_route.md`
- `pseudo_cache_schema_migration_plan.md`
- `pseudo_view.pdf` / `0_pseudo_view.pdf`

若临时分析笔记已被 `STATUS.md / DESIGN.md / CHANGELOG.md / charles.md` 吸收，可清理。

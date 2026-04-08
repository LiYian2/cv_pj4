# DESIGN.md - Part3 设计文档

> 本文档记录**当前实际采用**的设计决策、接口定义与默认实验方案。
> 最后更新：2026-04-08 21:56

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

当前 Re10k-1 sparse 实际使用：
- `target_depth = EDP fused depth`
- `confidence_mask = normalized EDP confidence`

注意：当前系统里 `target_depth` **不直接作为 refine loss 的几何监督**。

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

### 4.1 A/B/C/D/E 结果（Re10k-1 sparse）

| 组别 | 配置摘要 | PSNR | SSIM | LPIPS | final_gaussians |
|------|----------|------|------|-------|-----------------|
| A | all-params + midpoint pseudo + real densify + λp=1.0 | 20.1598 | 0.7284 | 0.2960 | 48951 |
| B | freeze-geom + midpoint pseudo + real densify + λp=1.0 | 20.9542 | 0.7575 | 0.2613 | 39916 |
| C | freeze-geom + midpoint pseudo + real densify + λp=0.5 | 21.5173 | 0.7797 | 0.2242 | 41544 |
| D | freeze-geom + midpoint pseudo + disable_densify + λp=1.0 | 20.6968 | 0.7518 | 0.2484 | 35639 |
| E | freeze-geom + tertile pseudo + real densify + λp=0.5 | **21.7658** | **0.7879** | **0.2110** | 42454 |

baseline sparse GT（同口径 eval）：`PSNR 16.9367 / SSIM 0.6526 / LPIPS 0.1722`

### 4.2 结论（当前最可靠）

1. **固定 pseudo 可学习参数为 appearance 是必要条件**（A→B 已验证）。
2. **`lambda_pseudo` 过大有副作用**：B→C 显示将 λp 从 1.0 降到 0.5 后，PSNR/SSIM/LPIPS 同时改善。
3. **在 C 策略下增加 pseudo 空间覆盖（midpoint→tertile）继续有效**：E 相对 C 三项指标继续同步提升。
4. **real-only densify 仍是当前更稳默认项**：D 虽略降 LPIPS 相对 B，但整体不如 C/E。

### 4.3 当前推荐 refine policy

当前推荐默认策略：
- joint refine
- `freeze_geometry_for_pseudo=True`
- `pseudo_trainable_params=appearance`
- `densify_stats_source=real`
- `lambda_real=1.0`
- `lambda_pseudo=0.5`
- pseudo 采样优先使用 **tertile 双点**，而不是单 midpoint

---

## 5. 下一步优化方向（PSNR + LPIPS）

### 5.1 实验优先级

1. **E 基线上做 `lambda_pseudo=0.25`**（继续测试 pseudo 权重下探）。
2. **E 基线上做 `disable_densify` 交叉对照**（看 densify 残余贡献）。
3. **细化 appearance 子集**：`opacity only`、`f_dc+f_rest`。

### 5.2 理论依据（为什么可能更好）

- pseudo target 来自 render+difix，含域偏差；降低 `lambda_pseudo` 可减少伪监督将偏差写入最终解。
- 让 pseudo 不直接改几何，可避免“颜色误差驱动几何漂移”的错误归因。
- densify 若受 pseudo 统计驱动，容易在伪误差处生长；real 驱动 densify 更贴近真实观测约束。
- LPIPS 对结构错位和纹理伪影敏感；上述策略本质是在压制这两类误差来源。

---

## 6. 评估口径与指标实现

### 6.1 当前主评估口径

- `pose_mode=gt`
- `origin_mode=test_to_sparse_first`
- `infer_init_mode=gt_first`

该口径用于与历史 A/B 结果严格可比。

### 6.2 LPIPS 生成方式（明确说明）

Part3 external eval 的 LPIPS 来自：
- `third_party/S3PO-GS/utils/external_eval_utils.py`
- `LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)`

因此：
- **不是 VGGT 指标**；
- 与 Part1/Scaffold 常见的 `lpips(net='vgg')` 口径不同，跨 pipeline 比较时需注明。

### 6.3 数据接触边界说明

- Refine 阶段：输入为 sparse train + pseudo cache，不直接读取 test GT RGB/GT pose。
- 评估阶段（`pose_mode=gt`）：会读取 test GT pose 与 test GT RGB 计算指标。

---

## 7. 参考与保留文档

当前 `docs/` 下仍有参考价值的非状态类文档：
- `project4_part3_merged_route.md`
- `pseudo_cache_schema_migration_plan.md`
- `pseudo_view.pdf` / `0_pseudo_view.pdf`

若临时分析笔记已被 `STATUS.md / DESIGN.md / CHANGELOG.md / charles.md` 吸收，可清理。

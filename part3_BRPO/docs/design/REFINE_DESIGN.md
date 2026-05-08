# REFINE_DESIGN.md - Refine 模块设计

> 更新时间：2026-05-05 05:30 (Asia/Shanghai)

> **书写规范**：
> 1. 只记录设计，不记录历史过程
> 2. 覆盖式更新，不追加
> 3. 状态用 ✅ ⚠️ ❌ 标记
> 4. 更新后修改文档顶部时间戳

---

## 1. 模块概览

Refine 模块负责 pseudo view 的 pose 和 scene refinement，包括：
- **Pose optimization**：优化 pseudo view 的 pose (theta/rho)
- **Scene optimization**：优化 Gaussians 的 xyz, opacity, scale, rotation
- **Exposure refinement**：优化 exposure 参数 (exposure_a/b)
- **Scale regularization**：防止 Gaussian scale 爆炸

---

## 2. 核心组件

### 2.1 Pose Delta 管理 (pseudo_camera_state.py)

| 函数 | 作用 | 状态 |
|------|------|------|
| `current_w2c(vp)` | 计算 pose-corrected w2c | ✅ |
| `refresh_viewpoint_transforms_(vp)` | 用 R/T 刷新 transforms | ✅ |
| `apply_pose_delta_before_render_(vp)` | **关键修复**：render 前应用 pose delta | ✅ |
| `apply_pose_residual_(vp)` | 将 delta 折叠到 R/T | ✅ |
| `make_viewpoint_trainable(vp)` | 初始化 pose delta 参数 | ✅ |

**CRITICAL FIX (2026-05-05)**：
- S3PO rasterizer forward.cu 不使用 theta/rho
- `apply_pose_delta_before_render_()` 在 render 前将 pose delta 应用到 world_view_transform
- 现在 gradient 可以从 RGB/depth loss 直接流回 theta/rho

### 2.2 Gauss-Newton Pose Optimization (pose_gauss_newton.py)

| 函数 | 作用 | 状态 |
|------|------|------|
| `compute_pose_jacobian_fd()` | Finite difference 计算 pose Jacobian | ✅ |
| `gauss_newton_pose_update()` | 单 viewpoint GN optimization | ✅ |
| `gauss_newton_batch_update()` | 批量 GN optimization | ✅ |
| `GaussNewtonPoseOptimizer` | 状态化 optimizer 类 | ✅ |

**设计要点**：
- 用 finite difference 计算 J = dLoss/dPose
- Levenberg-Marquardt damping: H = J.T @ J + damping * I
- 直接更新 theta/rho，不依赖 Adam

### 2.3 Loss Functions (pseudo_loss_v2.py)

| 函数 | 作用 | 状态 |
|------|------|------|
| `masked_rgb_loss()` | RGB L1 loss + exposure | ✅ |
| `masked_depth_loss()` | Depth L1 loss | ✅ |
| `pose_reg_loss()` | Pose delta L2 regularization | ✅ |
| `exposure_reg_loss()` | Exposure L1 regularization | ✅ |
| `scale_reg_loss()` | **新增**：Scale regularization | ✅ |
| `absolute_pose_prior_loss_scaled()` | Absolute pose prior | ✅ |
| `build_stageA_loss_paper_brpo_split()` | Paper chain loss | ✅ |

### 2.4 Backend Integration (slam_backend_brpo.py)

| 类/函数 | 作用 | 状态 |
|---------|------|------|
| `BRPOMappingConfig` | Online mapping 配置 | ✅ 已添加 lambda_scale |
| `BRPOBackEndContinuation` | Backend runner | ✅ |
| `_run_joint_pseudo_engine()` | Joint optimization loop | ✅ 已集成 scale_reg_loss |
| `run_brpo_pseudo_mapping()` | Online mapping 入口 | ✅ |

---

## 3. 配置参数

### 3.1 BRPOMappingConfig 新增参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `lambda_scale` | 0.01 | Scale regularization 权重 |
| `max_scale` | None | 最大允许 scale |
| `lambda_exp` | 0.001 | Exposure regularization 权重 |

### 3.2 使用方式

```yaml
Results:
  brpo_online_mapping:
    lambda_scale: 0.01  # 新增
    max_scale: 0.5      # 新增（可选）
    lambda_exp: 0.001   # 已有
```

---

## 4. Gradient Flow 修复

### 4.1 问题诊断

| 问题 | 原因 | 影响 |
|------|------|------|
| Pose gradient 不起作用 | forward.cu 不使用 theta/rho | Pose 无法从 RGB/depth refine |

### 4.2 修复方案

```
修复前：
render(viewpoint) → forward.cu 使用 viewmatrix(R/T) → theta/rho 不影响 render

修复后：
apply_pose_delta_before_render_(viewpoint) → 更新 world_view_transform
render(viewpoint) → forward.cu 使用 pose-corrected viewmatrix → theta/rho 影响 render
backward → gradient 流回 theta/rho
```

---

## 5. 一句话结论

> **Refine 模块已完成关键修复：pose gradient 问题通过 `apply_pose_delta_before_render_()` 解决，Gauss-Newton 模块已实现，scale regularization 已添加。所有修改已集成到 slam_backend_brpo.py 的 online mapping loop。**
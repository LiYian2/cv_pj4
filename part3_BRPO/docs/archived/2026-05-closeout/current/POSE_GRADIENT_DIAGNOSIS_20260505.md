# Pose Gradient 问题诊断

> 更新时间：2026-05-05 05:00 (Asia/Shanghai)

> **书写规范**：
> 1. 只记录"现在"，不记录历史过程
> 2. 覆盖式更新，不追加
> 3. 状态用 ✅ ⚠️ ❌ 标记
> 4. 更新后修改文档顶部时间戳

---

## 1. 问题摘要

**核心问题**：S3PO rasterizer 的 forward pass 不使用 theta/rho pose delta，导致 pose gradient 无法从 RGB/depth loss 直接反传。

---

## 2. 技术分析

### 2.1 Forward vs Backward 行为对比

| 组件 | 使用的 Pose | 包含 Pose Delta？ |
|------|------------|------------------|
| forward.cu (render) | viewmatrix (从 R/T 计算) | ❌ 不包含 |
| backward.cu | 从 viewmatrix 解析 SE3 | ✅ 计算 theta/rho gradient |

### 2.2 代码证据

**Python 层：Camera.world_view_transform**

```python
# utils/camera_utils.py:98
@property
def world_view_transform(self):
    return getWorld2View2(self.R, self.T).transpose(0, 1)
    # ❌ 不包含 cam_rot_delta / cam_trans_delta
```

**C++ 层：forward.cu**

```cpp
// forward.cu 只使用 viewmatrix，没有 theta/rho 相关代码
// transformPoint4x3(mean, viewmatrix) 直接用 viewmatrix
```

**C++ 层：backward.cu**

```cpp
// backward.cu:277-287
SE3 T_CW(view_matrix);  // 从 viewmatrix 解析当前 pose
mat33 dpC_drho = mat33::identity();
mat33 dpC_dtheta = -mat33::skew_symmetric(t);
// 计算 dL_dtau (theta/rho gradient)
dL_dtau[6 * idx + i] += dL_dt[i];
dL_dtau[6 * idx + 3] += dL_dtheta.x;
dL_dtau[6 * idx + 4] += dL_dtheta.y;
dL_dtau[6 * idx + 5] += dL_dtheta.z;
```

**Python 层：render 函数**

```python
# gaussian_renderer/__init__.py
rasterizer(
    ...
    theta=viewpoint_camera.cam_rot_delta,  # 传入 theta
    rho=viewpoint_camera.cam_trans_delta,  # 传入 rho
)
# 但 forward.cu 不使用这些参数！
```

---

## 3. 问题根源

### 3.1 设计意图 vs 实际行为

**设计意图**：
- theta/rho 是 pose delta 参数
- render 应该使用 pose-corrected pose (R/T + delta)
- gradient 从 RGB/depth loss 反传到 theta/rho

**实际行为**：
- forward.cu 只使用 viewmatrix（不包含 delta）
- theta/rho 被传入但被忽略
- backward.cu 计算的 theta/rho gradient 是"理论上的"
- theta/rho 改变不影响 render 结果

### 3.2 为什么 backward 还能计算 theta/rho gradient？

backward.cu 从 viewmatrix 解析出当前 pose (SE3)，然后计算：
- `dL/dtau = dL/dt * dt/dtau` (chain rule)
- 这里 `dt/dtau` 是 Jacobian，表示 pose 如何随 theta/rho 变化

但问题是：
- forward 不使用 theta/rho
- 所以 theta/rho gradient 无法实际影响 forward 结果
- gradient 计算是"数学上正确"但"工程上无效"

---

## 4. 当前 Pose Optimization 实际效果

### 4.1 pose_reg_loss 只是 L2 regularization

```python
# pseudo_loss_v2.py:87
def pose_reg_loss(viewpoint, trans_weight: float = 1.0) -> torch.Tensor:
    return torch.norm(viewpoint.cam_rot_delta, p=2) + \
           float(trans_weight) * torch.norm(viewpoint.cam_trans_delta, p=2)
```

这不是从 RGB/depth loss 反传的真正 pose gradient，只是惩罚 pose delta 变大。

### 4.2 为什么 ATE 有轻微改善？

可能的解释：
1. **lambda_pose regularization** 限制了 pose drift
2. **lambda_abs_pose prior** 保持了 pose 与初始值的接近
3. 但这些都不是真正的"pose refinement"

---

## 5. 修复方案

### 方案 A：修改 world_view_transform property

```python
# 修改 utils/camera_utils.py
@property
def world_view_transform(self):
    from part3_BRPO.pseudo_branch.refine.pseudo_camera_state import current_w2c
    w2c_current = current_w2c(self)  # 包含 pose delta
    return w2c_current.transpose(0, 1)
```

**优点**：简单，一行修改
**缺点**：可能影响其他使用 world_view_transform 的代码

### 方案 B：创建新的 render 函数

```python
def render_with_pose_delta(viewpoint_camera, pc, ...):
    w2c_current = current_w2c(viewpoint_camera)
    viewpoint_camera.world_view_transform = w2c_current.transpose(0, 1).contiguous()
    viewpoint_camera.full_proj_transform = (
        viewpoint_camera.world_view_transform.unsqueeze(0).bmm(
            viewpoint_camera.projection_matrix.unsqueeze(0)
        )
    ).squeeze(0)
    return render(viewpoint_camera, pc, ...)
```

**优点**：不影响原有 render 函数
**缺点**：需要修改所有调用点

### 方案 C：修改 C++ rasterizer forward.cu

在 forward.cu 中使用 theta/rho：
```cpp
// 在 forward.cu 中
SE3 T_delta(theta, rho);  // 从 theta/rho 构建 pose delta
SE3 T_final = T_delta * T_base;  // 应用 delta
viewmatrix = T_final.matrix();
```

**优点**：最彻底的修复
**缺点**：需要修改 C++/CUDA 代码，重新编译

---

## 6. Gauss-Newton 状态

### 6.1 现状

- ❌ `pose_gauss_newton.py` 文件不存在
- 当前使用 Adam optimizer
- Pose gradient 依赖间接 regularization，不是直接从 RGB/depth loss

### 6.2 Gauss-Newton 设计建议

参考 BRPO 论文 section 3.2：

```python
def gauss_newton_pose_update(viewpoint, render_loss_fn, max_iters=5):
    """
    Gauss-Newton pose optimization.
    
    1. 用 finite difference 计算 Jacobian J = dLoss/dPose
    2. 构建 H = J.T @ J + lambda * I (damping)
    3. 计算 delta = H^{-1} @ J.T @ residual
    4. 更新 pose
    """
    for i in range(max_iters):
        # Finite difference Jacobian
        J = compute_pose_jacobian_fd(viewpoint, render_loss_fn)
        
        # Gauss-Newton update
        H = J.T @ J + damping * torch.eye(6)
        delta = torch.linalg.solve(H, -J.T @ residual)
        
        # Apply delta
        viewpoint.cam_rot_delta += delta[3:6]
        viewpoint.cam_trans_delta += delta[0:3]
```

---

## 7. 下一步行动

| 优先级 | 任务 | 状态 |
|-------|------|------|
| P0 | 修复 forward 不使用 theta/rho | ⚠️ 待定方案 |
| P1 | 验证 theta/rho gradient 是否起作用 | ❌ 待验证 |
| P2 | 实现 Gauss-Newton pose optimization | ❌ 未实现 |
| P3 | 添加 exposure refine 和 scale regularization | ❌ 待规划 |

---

## 8. 一句话结论

> **Pose gradient 问题的根本原因是 S3PO rasterizer forward.cu 不使用 theta/rho pose delta。backward.cu 计算的 theta/rho gradient 是数学上正确但工程上无效的。需要修改 world_view_transform 或 forward.cu，让 pose delta 实际参与 render 计算。**
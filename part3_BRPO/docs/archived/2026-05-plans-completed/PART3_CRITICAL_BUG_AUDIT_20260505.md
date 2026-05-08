# Part3 BRPO 深度审计报告 - 第二轮

> 审计时间：2026-05-04 23:50 (Asia/Shanghai)
> 审计者：Claude (based on Charles's request)

---

## 1. 审计背景

用户报告两个严重问题：
1. S3PO 融合效果非常轻微（18.217 → 18.230），远达不到 BRPO 论文声称的 20→24 PSNR
2. Standalone refinement 大幅退化（14.29 vs 18.217 baseline）

---

## 2. 关键发现一：Pose Optimization 未生效

### 2.1 问题描述

检查 StageA history，发现：
```
loss_pose_reg: [0.0, 0.0, 0.0, ...]  ← 全程为 0！
```

**pose regularization loss 全程为 0，说明 pose delta (cam_rot_delta, cam_trans_delta) 没有任何变化。**

### 2.2 根本原因

**Render 和 Loss computation 使用不同的 pose：**

| 组件 | 使用的 Pose | 包含 Pose Delta？ |
|------|------------|------------------|
| **render(viewpoint)** | `viewpoint.R`, `viewpoint.T` | ❌ 不包含 |
| **loss computation** | `current_w2c(viewpoint)` | ✅ 包含 |

```python
# pseudo_camera_state.py
def current_w2c(vp):
    base = eye_matrix
    base[:3, :3] = vp.R
    base[:3, 3] = vp.T
    tau = torch.cat([vp.cam_trans_delta, vp.cam_rot_delta])  # ← 包含 pose delta
    return SE3_exp(tau) @ base  # ← pose-corrected w2c

# render 函数使用
def render(viewpoint_camera, ...):
    viewmatrix = viewpoint_camera.world_view_transform  # ← 从 vp.R/T 计算，不含 delta
```

### 2.3 为什么 pose optimization 不生效

1. **Render 不使用 pose delta**：Gaussians 渲染时用的是原始 pose (R/T)
2. **Loss computation 使用 pose delta**：但只是用来计算 prior loss
3. **梯度流断裂**：pose delta 参数不在 render pipeline 中，无法从 RGB/depth loss 获得梯度
4. **pose_reg_loss = 0**：因为 delta 初始为 0，没有变化

### 2.4 正确的设计应该是

render 应该使用 pose-corrected pose：

```python
# 正确做法
w2c_current = current_w2c(viewpoint)
viewpoint.world_view_transform = w2c_current.transpose(0, 1)
```

**目前的设计是错误的：pose delta 只在 loss 中被考虑，但在 render 中被忽略！**

---

## 3. 关键发现二：Standalone Confidence Mask Coverage 过低

### 3.1 问题描述

Standalone legacy script PSNR = 14.29（严重退化 4dB）

### 3.2 根本原因

**Confidence mask coverage 差异巨大：**

| 来源 | Coverage | Nonzero pixels |
|------|----------|----------------|
| legacy `confidence_mask_brpo_fused.npy` | **5.5%** | 14,421 |
| v2 `signal_v2_dense_paper_q070` | **16.8%** | 43,883 |

legacy 的 coverage 只有 5.5%，远低于 v2 的 16.8%！

### 3.3 为什么 Coverage 过低导致退化

- 只有 5.5% 的像素有 supervision
- 3000 iterations 只优化这少量区域
- 其他区域被忽略，导致整体质量下降

---

## 4. 关键发现三：Legacy Script 没有 Pose Optimization

### 4.1 Legacy Script 缺失功能

检查 legacy script (`archive_experiments/legacy_entry/run_pseudo_refinement.py`)：

- ❌ 没有 pose optimization 参数
- ❌ 没有 `lambda_pose` 参数
- ❌ 没有 `cam_rot_delta` / `cam_trans_delta` 参数
- ✓ 只优化 Gaussians appearance/geometry

### 4.2 与 v2 Script 的对比

| 功能 | Legacy Script | V2 Script |
|------|--------------|-----------|
| Pose optimization | ❌ 无 | ✓ 有（但目前不生效） |
| Exposure refinement | ❌ 无 | ✓ 有 |
| Absolute pose prior | ❌ 无 | ✓ 有 |
| Signal v2 pipeline | ❌ 无 | ✓ 有 |

---

## 5. 为什么 Part3 与 BRPO 论文有巨大鸿沟

### 5.1 BRPO 论文的 Pose Optimization

根据 BRPO 论文 section 3.2：
- Pseudo view pose 通过 optimization refine
- Pose correction 直接影响 rendering
- RGB loss 和 depth loss 都驱动 pose optimization

### 5.2 Part3 的问题

1. **Pose delta 不在 render pipeline 中**：render 用的是原始 pose
2. **Pose optimization 无法从 RGB/depth loss 获得梯度**：梯度流断裂
3. **Pose correction 只在 iteration 结束后应用**：而不是在每次 render 时

### 5.3 修复方向

**方向 A：让 render 使用 pose-corrected pose**

在每次 render 前调用：
```python
w2c_current = current_w2c(viewpoint)
viewpoint.world_view_transform = w2c_current.transpose(0, 1)
viewpoint.full_proj_transform = ...
refresh_viewpoint_transforms_(viewpoint)
```

**方向 B：直接优化 viewpoint.R/T**

不使用 pose delta，直接把 R/T 作为可训练参数。

---

## 6. 其他潜在问题

### 6.1 Scale Regularization

检查 `lambda_abs_pose` 参数：
- `stageA_lambda_abs_t = 3.0`
- `stageA_lambda_abs_r = 0.1`
- 使用 scene_scale 进行 scaling

但 pose optimization 不生效，这些参数也没有意义。

### 6.2 Exposure Refinement

Exposure 参数 (exposure_a, exposure_b) 存在，但没有被验证是否生效。

### 6.3 Densify/Prune

V2 script 的 densify/prune 默认禁用（stageA5 mode 才有，但用的是 stageB mode）。

---

## 7. 总结：三个 Critical Bug

| Bug | 严重程度 | 影响 |
|-----|---------|------|
| **Render 不使用 pose delta** | 🔴 Critical | Pose optimization 无法生效 |
| **Standalone confidence coverage 过低** | 🔴 Critical | 导致严重退化 |
| **Legacy script 缺 pose optimization** | 🟠 Major | Standalone 无法 refine pose |

---

## 8. 立即可做的修复

### 8.1 修复 Render 使用 Pose Delta

在 `run_pseudo_refinement_v2.py` 的 render 调用前：

```python
# Before render
for view in sampled_views:
    vp = view['vp']
    w2c_current = current_w2c(vp)
    vp.world_view_transform = w2c_current.transpose(0, 1).contiguous()
    vp.full_proj_transform = (
        vp.world_view_transform.unsqueeze(0).bmm(vp.projection_matrix.unsqueeze(0))
    ).squeeze(0)
```

### 8.2 使用 signal_v2 的 standalone

不使用 legacy script，改用 v2 script 的 stageA mode。

---

## 9. 一句话结论

> **Part3 失效的根本原因是：Pose delta 参数虽然存在于 loss computation 中，但 render 函数使用的是不包含 pose delta 的原始 pose (R/T)，导致 pose optimization 无法从 RGB/depth loss 获得梯度，pose correction 完全不生效。此外，Standalone legacy script 的 confidence mask coverage 只有 5.5%（vs v2 的 16.8%），且没有 pose optimization 功能，导致严重退化。要修复这些问题，需要让 render 使用 pose-corrected pose，并使用 v2 signal pipeline 进行 standalone refinement。**

---

## 10. 相关文件

- Pose delta definition: `pseudo_branch/refine/pseudo_camera_state.py:current_w2c()`
- Render function: `third_party/S3PO-GS/gaussian_splatting/gaussian_renderer/__init__.py`
- Refinement script: `scripts/run_pseudo_refinement_v2.py`
- Legacy script: `scripts/archive_experiments/legacy_entry/run_pseudo_refinement.py`
- Loss functions: `pseudo_branch/refine/pseudo_loss_v2.py`
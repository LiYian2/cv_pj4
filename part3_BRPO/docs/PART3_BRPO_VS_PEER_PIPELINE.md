# Part3 BRPO vs 同学 Pipeline 对比分析

> 分析日期: 2026-05-06

---

## 1. 整体架构对比

| 维度 | Part3 BRPO (D-Series) | 同学 Pipeline |
|------|----------------------|---------------|
| **SLAM Backbone** | S3PO-GS | S3PO-GS |
| **Tracking** | Monocular, 全量帧 | Monocular, 全量帧 |
| **Keyframe 稀疏度** | force_keyframe_indices / kf_interval | kf_interval |
| **Pseudo 触发** | 每个 keyframe 触发 online mapping | 相邻 keyframe 到达后触发 |
| **Pseudo 选择** | midpoint/quartile (slot selector) | 中间 tracked frame，fallback interpolation |
| **Frontend-Backend 通信** | runtime_state_payload (dict) | path + pose 数值 |

---

## 2. Pseudo View 创建对比

### Part3 BRPO

**流程**:
```
Frontend 缓存 runtime_camera_state (所有帧)
    ↓
Backend select_runtime_pseudo_slots()
    ↓
选择: midpoint_only / quartile 模式
    ↓
MASt3R dense matching (left_ref, pseudo_rgb, right_ref)
    ↓
Projected depth = pts3d.z
    ↓
创建 PseudoRecord (viewpoint, depth, mask)
```

**特点**:
- Pseudo pose 来自 **tracked frame** (有真实 image path)
- Depth 来自 **MASt3R projected depth** (3D matching)
- 不做 restoration，直接用 coarse render + MASt3R depth

### 同学 Pipeline

**流程**:
```
相邻 keyframe 到达
    ↓
选择中间 tracked frame pose
    ↓ (fallback)
若无 tracked frame → keyframe interpolation
    ↓
Coarse pseudo-view rendering
    ↓
Difix3D 双向 restoration
    ↓ (previous + current reference conditioned)
Overlap-score residual fusion
    ↓
BRPO-style confidence mask
    ↓
Skip 低 overlap/confidence pseudo
```

**特点**:
- Pseudo pose: **优先 tracked frame，fallback interpolation**
- Depth: **从相邻 KF depth 构建**
- Restoration: **Difix3D 双向 refinement**
- Fusion: **residual fusion（非 simple average）**
- Filter: **低 confidence 直接 skip**

---

## 3. Depth 来源对比

| | Part3 BRPO | 同学 Pipeline |
|---|-----------|---------------|
| **Depth 来源** | MASt3R projected depth (pts3d.z) | 相邻 KF depth 构建 |
| **Depth 类型** | Dense 3D matching 结果 | Local depth propagation |
| **依赖** | MASt3R model | Difix3D + KF depth |
| **精度** | Dense correspondence | Restoration refinement |

**分析**:
- Part3 用 MASt3R 做 dense 3D matching，depth 更准确但依赖 model
- 同学用 KF depth 构建 + restoration，更轻量但依赖 KF depth quality

---

## 4. Restoration vs No Restoration

### Part3 BRPO: 无 Restoration

直接使用:
- `pseudo_render_rgb` (coarse)
- `pseudo_projected_depth` (MASt3R)

**原因**: Paper BRPO 原设计不使用 restoration，直接用 projected depth supervision

### 同学 Pipeline: Difix3D 双向 Restoration

**双向 restoration**:
1. Previous reference conditioned refinement
2. Current reference conditioned refinement

**Residual fusion**:
- 基于 overlap score 计算权重
- Residual fusion (非 simple average)

**优势**: Restoration 提升 pseudo-view 质量，减少 coarse render 的 artifacts

---

## 5. Confidence Mask 对比

### Part3 BRPO

**Mask 来源**:
- MASt3R confidence (`conf > threshold`)
- Support mask (valid pts3d)

**实现**: `pseudo_scene_mask_mode = all_valid / both_only`

### 同学 Pipeline

**BRPO-style confidence mask**:
- Feature correspondence consistency
- Reprojection consistency
- Opacity filtering

**额外**: **Skip 低 overlap/confidence pseudo**

**分析**: 同学的 pipeline 有更完整的 confidence filtering，直接 skip 低质量 pseudo

---

## 6. Optimization 阶段对比

### Part3 BRPO

**单阶段**:
```
Real + Pseudo joint optimization
    ↓
apply_pose_delta_before_render_()
    ↓
Render + Loss
    ↓
Backward
    ↓ (可选)
Gauss-Newton pose optimization
    ↓ (可选)
Scale regularization
    ↓
Adam step
```

**特点**:
- Real + pseudo 同时优化
- GN 可选，scale reg 可选
- 不分阶段

### 同学 Pipeline

**两阶段**:
```
Phase 1: pose/exposure stabilization
    ↓
Phase 2: joint pose-Gaussian optimization
```

**特点**:
- 先稳定 pose/exposure
- 再联合优化
- 更稳定的 optimization trajectory

---

## 7. Loss 设计对比

### Part3 BRPO

**Loss 组成**:
```python
total_loss = (
    lambda_real * loss_real +       # RGB + depth (real KFs)
    lambda_pseudo * loss_pseudo +   # RGB + depth (pseudo)
    lambda_scale * scale_loss +     # Scale regularization
    lambda_pose * pose_loss +       # (可选)
)
```

**Depth loss**: Paper BRPO split v1 / exact shared cm v1

### 同学 Pipeline

**Masked RGB-D loss**:
- Confidence-weighted RGB loss
- Confidence-weighted depth loss

**额外**:
- **Local depth scale consistency regularization** (防止 scale drift)
- **Pseudo loss warmup** (避免 early-stage 过强 supervision)

**分析**: 同学的 loss design 更细致，有 warmup 和 local scale consistency

---

## 8. Densification/Pruning 对比

### Part3 BRPO

**Gaussian update**:
- Real KF mapping: densify/prune/opacity_reset
- Pseudo mapping: **默认关闭 densify/prune**
- 只通过 gradient 更新基本属性

### 同学 Pipeline

**明确分离**:
- Densification/pruning: **仅由真实 KF mapping 执行**
- Pseudo optimization: **不参与 densify/prune**

**相同点**: 两者都限制 pseudo 不参与 densify/prune

---

## 9. Frontend-Backend 通信对比

### Part3 BRPO

**发送内容**:
```python
runtime_state_payload = {
    frame_id: {
        "image_path": str,
        "pose_c2w": matrix,
        "is_keyframe": bool,
        "fx", "fy", "cx", "cy", ...
    }
}
```

**特点**: 发送完整 dict，包含 path + pose + camera params

### 同学 Pipeline

**发送内容**: **path + pose 数值，不发送 torch object**

**优势**: 更轻量的通信，减少 serialization overhead

---

## 10. Evaluation 对比

### Part3 BRPO

**Evaluation**:
- ATE (trajectory accuracy)
- PSNR/SSIM/LPIPS (rendering quality)
- 在 SLAM 结束时评估

### 同学 Pipeline

**Evaluation**: **Held-out non-keyframe images**

**特点**:
- 用非 keyframe 作为 test set
- 避免 KF bias
- 更 fair 的 evaluation

---

## 11. 关键差异总结

| 差异 | Part3 BRPO | 同学 Pipeline |
|------|-----------|---------------|
| **Pseudo pose 来源** | Tracked frame (slot selector) | Tracked frame + interpolation fallback |
| **Depth 来源** | MASt3R projected | KF depth 构建 + restoration |
| **Restoration** | 无 | Difix3D 双向 + residual fusion |
| **Confidence filter** | MASt3R conf mask | BRPO-style + skip 低 confidence |
| **Optimization 阶段** | 单阶段 | 两阶段 (stabilization + joint) |
| **Loss design** | Paper BRPO | Masked + warmup + local scale |
| **Communication** | dict payload | path + pose 数值 |
| **Evaluation** | ATE + rendering | Held-out non-KF |

---

## 12. 优势对比

### Part3 BRPO 优势

1. **MASt3R depth 更准确** - Dense 3D matching，直接获取 scene depth
2. **Gauss-Newton 效果显著** - D-Series 验证 ATE 改善 71%
3. **实现简洁** - 不依赖 Difix3D，流程更直接
4. **已验证效果** - D-Series 有定量结果

### 同学 Pipeline 优势

1. **Restoration 提升质量** - Difix3D refinement 减少 artifacts
2. **Confidence filtering 更完整** - Skip 低质量 pseudo
3. **两阶段 optimization** - 更稳定的 convergence
4. **Pseudo loss warmup** - 避免 early-stage 问题
5. **Local scale consistency** - 防止 scale drift
6. **Held-out evaluation** - 更 fair 的评估
7. **轻量通信** - 减少 serialization overhead

---

## 13. 潜在改进方向

### Part3 BRPO 可借鉴

1. **Pseudo pose fallback** - 若无 tracked frame，用 interpolation
2. **Restoration** - Difix3D 提升 pseudo-view 质量
3. **Two-phase optimization** - pose stabilization + joint
4. **Pseudo loss warmup** - 控制 early-stage supervision
5. **Local scale consistency** - 防止 local scale drift
6. **Held-out evaluation** - 更 fair 的评估方式

### 同学 Pipeline 可借鉴

1. **MASt3R depth** - Dense matching 比 KF depth propagation 更准确
2. **Gauss-Newton** - 二阶方法比 Adam 更高效
3. **Global scale regularization** - 防止 Gaussian scale explosion

---

## 14. 结论

**核心差异**:
- Part3 BRPO: **MASt3R depth + GN optimization**，已验证效果显著
- 同学 Pipeline: **Restoration + 两阶段 + warmup**，设计更完整细致

**建议**:
- Part3 可借鉴 restoration、warmup、local scale consistency
- 同学可借鉴 MASt3R depth、Gauss-Newton
- 两者各有优势，可互相借鉴优化
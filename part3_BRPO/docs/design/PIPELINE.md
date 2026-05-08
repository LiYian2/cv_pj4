# PIPELINE.md

> Purpose: compact source-of-truth for the current Part3 BRPO Online Mapping pipeline.
> Update: 2026-05-07 — 切换到 Online Mapping 主线，记录 D-Series 实验结果

---

## 1. 口径说明

- **M~** = Mask（confidence），定义监督域与监督强度
- **T~** = Target，定义监督目标数值（projected depth）
- **G~** = Gaussian Management，定义 per-Gaussian 参与控制
- **R~** = Topology，定义 joint refine 的 loss assembly / backward timing

---

## 2. 一句话总览

当前 Online Mapping 主线已验证：**Pose gradient fix + Gauss-Newton pose optimization + Scale regularization = ATE改善71%**。

关键组件：
- **Pose gradient fix**: `apply_pose_delta_before_render_()` 确保 pose delta 参与 render
- **Gauss-Newton**: 二阶优化，比 Adam 更高效
- **Scale regularization**: `scale_reg_loss()` 防止 Gaussian scale explosion
- **Difix restoration**: **缺失**，计划集成（D5 待运行）

---

## 3. Online Mapping 数据流

### 3.1 进程架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Main Process                            │
│  ┌─────────────────┐                                        │
│  │    Frontend     │ ──────► backend_queue ──────►          │
│  │  (SLAM Loop)    │                                        │
│  │                 │ ◄───── frontend_queue ◄─────           │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ mp.Process
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend Process                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Backend Loop                                        │    │
│  │  ┌───────────────┐  ┌─────────────────────────────┐ │    │
│  │  │  SLAM Map     │  │  BRPO Online Mapping       │ │    │
│  │  │  (Real KFs)   │  │  (Pseudo Views)            │ │    │
│  │  └───────────────┘  ┌─────────────────────────────┐ │    │
│  │                     │  Gauss-Newton Pose Opt     │ │    │
│  │                     │  Scale Regularization      │ │    │
│  │                     └─────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Frontend → Backend 数据流

```
Frontend                              Backend
────────────────────────────────────────────────────
runtime_camera_state_cache (所有帧)
    ↓
request_keyframe() ────────────► backend_queue
    ↓                              ↓
runtime_state_payload          brpo_runtime_camera_states
                                   ↓
                               select_runtime_pseudo_slots()
                                   ↓
                               slots (quartile: 3 pseudo per gap)
                                   ↓
                               build_runtime_exact_backend_bundle()
                                   ├─ MASt3R matching (left/right refs)
                                   ├─ Projected depth (pts3d.z)
                                   └─ PseudoRecord creation
                                   ↓
                               run_brpo_pseudo_mapping()
                                   ├─ apply_pose_delta_before_render_()
                                   ├─ Render + Loss + Backward
                                   ├─ Gauss-Newton (可选)
                                   └─ Scale regularization
                                   ↓
                               push_to_frontend()
```

---

## 4. 核心组件详解

### 4.1 Pose Gradient Fix (`apply_pose_delta_before_render_`)

**问题**: S3PO rasterizer 使用 `world_view_transform` 渲染，只依赖 R/T，不包含 `cam_rot_delta` / `cam_trans_delta`。

**修复**: 在 render 前将 pose delta fold 进 transform：

```python
def apply_pose_delta_before_render_(viewpoint):
    theta = viewpoint.cam_rot_delta  # pose residual
    rho = viewpoint.cam_trans_delta
    
    # Fold theta/rho into world_view_transform
    R_new = viewpoint.R @ exp_so3(theta)
    T_new = viewpoint.T + rho
    
    viewpoint.world_view_transform = build_world_view(R_new, T_new)
```

**文件**: `pseudo_branch/refine/pseudo_camera_state.py`

### 4.2 Gauss-Newton Pose Optimization

**原理**: 二阶优化，利用 Jacobian 信息高效更新 pose：

```python
class GaussNewtonPoseOptimizer:
    def step(self, viewpoint, loss_fn):
        # Finite difference Jacobian: ∂loss/∂(theta, rho)
        J = compute_jacobian_finite_diff(viewpoint, loss_fn)
        
        # Hessian: J.T @ J + damping * I
        H = J.T @ J + self.damping * I
        
        # Solve: delta = -H^-1 @ J.T @ residual
        delta = torch.linalg.solve(H, -J.T @ residual)
        
        # Direct update (no gradient)
        theta += delta[:3]
        rho += delta[3:]
```

**文件**: `pose_gauss_newton.py`

### 4.3 Scale Regularization

**原理**: 防止 Gaussian scale 在 pseudo supervision 下爆炸：

```python
def scale_reg_loss(gaussians, max_scale=0.01):
    scaling = gaussians.get_scaling
    # Penalize scale > max_scale
    excess = torch.clamp(scaling - max_scale, min=0)
    return excess.mean()
```

**文件**: `slam_backend_brpo.py`

---

## 5. Real KF vs Pseudo Mapping 对比

### 5.1 输入数据

| 数据 | Real KF Mapping | Pseudo Mapping |
|------|-----------------|----------------|
| **RGB source** | Dataset GT RGB | Coarse render (未修复) |
| **Depth source** | Dataset mono_depth | MASt3R projected depth |
| **Confidence mask** | 无 | MASt3R confidence + epipolar verify |

### 5.2 Optimization 差异

| 操作 | Real KF Mapping | Pseudo Mapping |
|------|-----------------|----------------|
| **Pose delta 应用** | ❌ 无 | ✓ `apply_pose_delta_before_render_()` |
| **Pose optimizer** | Adam | Gauss-Newton (可选) |
| **Gaussian densify** | ✓ 执行 | ❌ 禁用 |
| **Gaussian prune** | ✓ 执行 | ❌ 禁用 |
| **Scale regularization** | Isotropic (weight=10) | `scale_reg_loss` (λ_scale) |

### 5.3 Loss 组成

| Loss 项 | Real KF Mapping | Pseudo Mapping |
|---------|-----------------|----------------|
| **RGB loss** | GT RGB vs render | Pseudo RGB vs render |
| **Depth loss** | mono_depth vs render | Projected depth vs render (masked) |
| **Regularization** | Isotropic | Scale reg |

---

## 6. D-Series 实验结果

### 6.1 D-Series 配置对比

| 实验 | Pose Opt | GN | Scale Reg | λ_scale | λ_pseudo | ATE (m) |
|------|----------|-----|-----------|---------|----------|---------|
| D0 | ❌ | ❌ | ❌ | - | - | 0.475 |
| D1 | ✓ | ❌ | ❌ | - | - | 0.475 |
| D2 | ✓ | ✓ | ❌ | - | - | 0.395 |
| D3 | ✓ | ✓ | ✓ | 0.01 | - | 0.302 |
| D4 | ✓ | ✓ | ✓ | 0.1 | 2.0 | **0.135** |
| D5 | ✓ | ✓ | ✓ | 0.1 | 2.0 | (待运行) + Difix |

### 6.2 关键发现

1. **Pose gradient fix**: D0→D1 无改善（说明单独 fix 不够）
2. **Gauss-Newton**: D1→D2 ATE改善17%
3. **Scale regularization**: D2→D3 ATE改善24%
4. **λ调整**: D3→D4 ATE改善55%
5. **总计**: D0→D4 ATE改善 **71%**

---

## 7. 缺失功能：Difix Restoration

### 7.1 问题确认

Online mapping **缺失** Difix restoration：

| 功能 | Standalone Pipeline | Online Mapping |
|------|---------------------|----------------|
| **Difix restoration** | ✓ 双向修复 | ❌ 缺失 |
| **RGB fusion** | ✓ residual fusion | ❌ 缺失 |
| **Depth target** | 从 fused RGB 计算 | MASt3R projected depth |

### 7.2 集成计划

**文件**: `docs/DIFIX_FUSION_INTEGRATION_PLAN.md`

**流程**:
```
Coarse render → Difix 双向修复 → RGB Fusion → MASt3R matching → Projected depth
```

**D5 配置** (待运行):
```yaml
use_difix_restoration: true
difix_model_name: nvidia/difix_ref
difix_timestep: 100
difix_fusion_mode: brpo_overlap_confidence
```

---

## 8. 关键文件索引

| 文件 | 作用 |
|------|------|
| `slam.py` | 主入口，初始化 frontend/backend 进程 |
| `slam_frontend.py` | Frontend 主循环，runtime state 缓存 |
| `slam_backend.py` | Backend 主循环，pseudo mapping 触发 |
| `slam_backend_brpo.py` | Pseudo mapping 核心逻辑 |
| `pose_gauss_newton.py` | GN pose optimization |
| `pseudo_camera_state.py` | `apply_pose_delta_before_render_()` |
| `runtime_exact_backend.py` | Pseudo view 创建，MASt3R matching |
| `runtime_slot_selector.py` | Pseudo slot 选择 (quartile mode) |

---

## 9. 待完成事项

| 优先级 | 任务 | 状态 |
|--------|------|------|
| P0 | 运行 D5 实验 (Difix enabled) | 待执行 |
| P1 | 对比 D4 vs D5 结果 | 待分析 |
| P2 | 实施深度生成变更 (pts3d_1.z) | 规划完成 |
| P3 | Pseudo loss warmup | 待规划 |
| P4 | Local scale consistency | 待规划 |

---

## 10. 推荐阅读顺序

1. `docs/current/STATUS.md` — 当前状态与结论
2. `docs/current/DESIGN.md` — 系统边界与模块关系
3. `docs/current/CHANGELOG.md` — 已落地改动与时间线
4. `docs/DIFIX_FUSION_INTEGRATION_PLAN.md` — Difix 集成规划
5. `docs/DEPTH_GENERATION_CHANGE_PLAN_20260506.md` — 深度生成变更规划
6. `docs/PART3_BRPO_ONLINE_MAPPING_PIPELINE.md` — Pipeline 完整流程
7. `docs/S3PO_ONLINE_MAPPING_CODE_FLOW_ANALYSIS.md` — 代码调用链分析
8. `docs/design/MASK_DESIGN.md`
9. `docs/design/TARGET_DESIGN.md`
10. `docs/design/REFINE_DESIGN.md`

---

## 11. 历史：Standalone Pipeline (已归档)

旧版 standalone pipeline 文档已归档到 `docs/archived/2026-04-plans-landed/`。

Standalone 流程（供参考）：
```
Dataset split
  → part2 S3PO full rerun
  → internal_eval_cache
  → internal prepare (select / Difix / fusion / verify / pack)
  → signal branch (legacy or signal_v2)
  → refine (StageA / StageA.5 / StageB)
  → replay eval
```

此流程已不再作为主线，online mapping pipeline 是当前重点。
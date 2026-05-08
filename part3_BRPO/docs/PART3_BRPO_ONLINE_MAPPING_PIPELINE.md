# Part3 BRPO Online Mapping Pipeline

> 分析日期: 2026-05-06
> 目标: 总结 online mapping 路径的完整 pipeline

---

## 1. 入口和架构

### 主入口

**文件**: `third_party/S3PO-GS/slam.py`

```python
class SLAM:
    def __init__(self, config, mast3r_model, save_dir=None):
        # Frontend 和 Backend 初始化
        self.frontend = FrontEnd(self.config, mast3r_model, self.save_dir)
        self.backend = BackEnd(self.config, self.save_dir)

        # Queue 通信
        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

    def run(self):
        # Backend 作为独立进程
        backend_process = mp.Process(target=self.backend.run)
        backend_process.start()

        # Frontend 在主进程运行
        self.frontend.run()
```

### 进程架构

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

---

## 2. Frontend 处理流程

### 文件: `utils/slam_frontend.py`

### 2.1 主循环 (`run()`)

```
cur_frame_idx = 0
while True:
    if cur_frame_idx >= len(dataset):
        # 评估并退出
        eval_ate()
        save_gaussians()
        break

    # 1. 初始化 Camera
    viewpoint = Camera.init_from_dataset(dataset, cur_frame_idx, projection_matrix)

    # 2. Tracking (非关键帧)
    render_pkg = self.tracking(cur_frame_idx, viewpoint)

    # 3. 缓存 runtime_camera_state (BRPO)
    if brpo_online_mapping_enabled:
        self.runtime_camera_state_cache[cur_frame_idx] = self._export_runtime_camera_state(
            cur_frame_idx, viewpoint, is_keyframe=False
        )

    # 4. 关键帧检测
    create_kf = self.is_keyframe(...) or (cur_frame_idx in force_keyframe_indices)

    if create_kf:
        # 5. 标记为关键帧，更新缓存
        if brpo_online_mapping_enabled:
            self.runtime_camera_state_cache[cur_frame_idx] = self._export_runtime_camera_state(
                cur_frame_idx, viewpoint, is_keyframe=True
            )

        # 6. 发送 "keyframe" 消息给 Backend
        self.request_keyframe(cur_frame_idx, viewpoint, current_window, depth_map)

    cur_frame_idx += 1
```

### 2.2 Runtime Camera State 缓存

**函数**: `_export_runtime_camera_state(frame_idx, viewpoint, is_keyframe=False)`

**缓存内容**:
```python
{
    "frame_id": int(frame_idx),
    "uid": int(viewpoint.uid),
    "image_path": str(image_path),
    "image_name": str(image_name),
    "image_width": int(viewpoint.image_width),
    "image_height": int(viewpoint.image_height),
    "fx", "fy", "cx", "cy", "FoVx", "FoVy",
    "pose_c2w": pose_c2w_matrix,  # R/T 的 c2w
    "is_keyframe": bool(is_keyframe),
}
```

**关键**: Frontend 缓存所有帧的 state，通过 `runtime_state_payload` 传递给 Backend。

### 2.3 Keyframe 消息

**函数**: `request_keyframe(cur_frame_idx, viewpoint, current_window, depthmap)`

**消息格式**:
```python
msg = [
    "keyframe",
    cur_frame_idx,
    viewpoint,         # Camera 对象
    current_window,    # 当前 KF window
    depth_map,
    self.theta,        # pose delta
    runtime_state_payload,  # 所有缓存帧的 state dict
]
self.backend_queue.put(msg)
```

---

## 3. Backend 处理流程

### 文件: `utils/slam_backend.py`

### 3.1 主循环 (`run()`)

```python
def run(self):
    while True:
        if backend_queue.empty():
            # 正常 SLAM mapping
            self.map(self.current_window)

        else:
            data = backend_queue.get()

            if data[0] == "keyframe":
                cur_frame_idx = data[1]
                viewpoint = data[2]
                current_window = data[3]
                depth_map = data[4]
                theta = data[5]
                runtime_state_payload = data[6]  # Frontend 缓存

                # 1. 更新 runtime_camera_states
                self.brpo_runtime_camera_states = runtime_state_payload

                # 2. 添加新关键帧
                self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                # 3. SLAM mapping (Real KFs)
                self.map(self.current_window, iters=iter_per_kf, up_pose=True)
                self.map(self.current_window, prune=True)

                # 4. BRPO Online Mapping (Pseudo Views)
                prepare_payload = self._maybe_prepare_brpo_runtime_slots(cur_frame_idx)
                self._run_brpo_runtime_pseudo_mapping(cur_frame_idx, prepare_payload)

                # 5. 推送更新给 Frontend
                self.push_to_frontend("keyframe")
```

### 3.2 准备 Pseudo Slots

**函数**: `_maybe_prepare_brpo_runtime_slots(cur_frame_idx)`

**流程**:
1. 从 `brpo_runtime_camera_states` 获取可用帧（非关键帧）
2. 调用 `select_runtime_pseudo_slots()` 选择 pseudo slots
3. 对每个 slot:
   - 调用 `build_runtime_exact_backend_bundle()` - MASt3R matching, depth projection
   - 调用 `build_runtime_exact_signal_bundle()` - 计算信号
   - 调用 `build_runtime_pseudo_record_bundle()` - 创建 PseudoRecord
4. 缓存 pseudo records 到 `self.brpo_runtime_pseudo_records`

**Slot 选择逻辑** (`select_runtime_pseudo_slots`):
- `midpoint_only`: 每个 gap 中点，1 pseudo per gap
- `quartile`: 25%, 50%, 75% 位置，3 pseudo per gap

### 3.3 运行 Pseudo Mapping

**函数**: `_run_brpo_runtime_pseudo_mapping(cur_frame_idx, prepare_payload)`

**流程**:
1. 检查 `enable_pseudo_gradient` 和 `pseudo_map_iters > 0`
2. 创建 `BRPOMappingConfig` (包含 GN 和 scale reg 参数)
3. 调用 `run_brpo_pseudo_mapping()` 执行优化

---

## 4. Pseudo Mapping 核心

### 文件: `utils/slam_backend_brpo.py`

### 4.1 主函数: `run_brpo_pseudo_mapping()`

**输入**:
- `gaussians`: Gaussian scene
- `cameras`: 所有 viewpoint (包含 pose delta)
- `current_window`: 当前 KF window
- `mapping_cfg`: BRPOMappingConfig
- `runtime_pseudo_records`: PseudoRecord 列表

**流程**:
```
for iteration in range(num_iterations):
    # 1. Render Real Views
    for viewpoint in current_window:
        apply_pose_delta_before_render_(viewpoint)  # CRITICAL FIX
        render_pkg = render(viewpoint, gaussians, ...)
        loss_real = compute_loss(render_pkg, viewpoint)

    # 2. Render Pseudo Views
    for record in runtime_pseudo_records:
        apply_pose_delta_before_render_(record.viewpoint)  # CRITICAL FIX
        render_pkg = render(record.viewpoint, gaussians, ...)
        loss_pseudo = compute_depth_loss(render_pkg, record)

    # 3. Scale Regularization
    scale_loss = scale_reg_loss(gaussians, max_scale=max_scale)

    # 4. Total Loss
    total_loss = lambda_real * loss_real + lambda_pseudo * loss_pseudo + lambda_scale * scale_loss

    # 5. Backward
    total_loss.backward()

    # 6. Gauss-Newton (如果启用)
    if use_gauss_newton:
        gn_optimizer.step()

    # 7. Adam Optimizer Step (如果不用 GN)
    else:
        pose_optimizer.step()
        gaussian_optimizer.step()
```

### 4.2 Gauss-Newton Pose Optimization

**文件**: `pseudo_branch/refine/pose_gauss_newton.py`

**集成位置**: `_run_joint_pseudo_engine()` 中，backward 之后

**逻辑**:
1. 计算 finite difference Jacobian: `∂loss/∂(theta, rho)`
2. 构建 Hessian: `J.T @ J + damping * I`
3. 解线性系统: `delta = -H^-1 @ J.T @ residual`
4. 直接更新: `theta += delta[:3], rho += delta[3:]`
5. Fold 到 R/T: `_fold_pseudo_pose_residual_()`

---

## 5. 数据流图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              Frontend                                     │
│                                                                           │
│  Dataset ──► Camera.init_from_dataset() ──► viewpoint                    │
│                         │                                                 │
│                         ▼                                                 │
│              _export_runtime_camera_state()                               │
│                         │                                                 │
│                         ▼                                                 │
│              runtime_camera_state_cache (所有帧)                          │
│                         │                                                 │
│                         ▼ (keyframe 消息)                                 │
│              runtime_state_payload ──────────────────────────────────────►│
└──────────────────────────────────────────────────────────────────────────┘
                                                                            │
                                                                            │ backend_queue
                                                                            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              Backend                                      │
│                                                                           │
│  brpo_runtime_camera_states ◄── runtime_state_payload                    │
│         │                                                                 │
│         ▼                                                                 │
│  select_runtime_pseudo_slots() ──► slots (midpoint/quartile)             │
│         │                                                                 │
│         ▼                                                                 │
│  build_runtime_exact_backend_bundle()                                     │
│         │                                                                 │
│         ├─► MASt3R matching (left_ref, right_ref)                        │
│         ├─► Projected depth computation                                   │
│         └─► PseudoRecord (包含 viewpoint, depth, mask)                   │
│         │                                                                 │
│         ▼                                                                 │
│  brpo_runtime_pseudo_records                                              │
│         │                                                                 │
│         ▼                                                                 │
│  run_brpo_pseudo_mapping()                                                │
│         │                                                                 │
│         ├─► Render Real KFs + apply_pose_delta_before_render_()          │
│         ├─► Render Pseudo Views + apply_pose_delta_before_render_()      │
│         ├─► Compute depth loss (pseudo)                                   │
│         ├─► Scale regularization                                          │
│         ├─► Backward                                                      │
│         ├─► Gauss-Newton (如果启用)                                       │
│         └─► Adam step (如果不用 GN)                                       │
│         │                                                                 │
│         ▼                                                                 │
│  更新 gaussians + poses                                                   │
│         │                                                                 │
│         ▼                                                                 │
│  push_to_frontend() ──► frontend_queue                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Pseudo View 的关键处理

### 6.1 Frontend 不处理 Pseudo

Frontend 只负责:
- 缓存所有帧的 runtime state
- 传递给 Backend

**不涉及**: pseudo view 创建、render、optimization

### 6.2 Backend 处理 Pseudo

Backend 负责:
- 从 runtime_camera_states 选择可用帧作为 pseudo
- MASt3R matching 生成 projected depth
- 创建 PseudoRecord (包含 viewpoint + depth + mask)
- 执行 pseudo mapping optimization

### 6.3 Pseudo Viewpoint 创建

**位置**: `pseudo_branch/integration/runtime_exact_backend_bundle.py`

```python
class RuntimeExactBackendBundle:
    pseudo_viewpoint: Camera  # 从 runtime state 创建
    pseudo_render_rgb: Tensor
    pseudo_projected_depth: Tensor
    left_result: Dict
    right_result: Dict
```

**创建流程**:
1. 从 `slot.frame_id` 获取 pseudo state
2. 创建 Camera 对象: `Camera(..., uid=frame_id, ...)`
3. MASt3R matching: `matcher(left_rgb, pseudo_rgb, right_rgb)`
4. Projected depth: `depth = pts3d.z`
5. 保存为 PseudoRecord

---

## 7. 配置传递

### Frontend 配置

```yaml
Results:
  brpo_online_mapping:
    enabled: true
    trigger: keyframe
```

读取位置: `slam_frontend.py` 初始化时

### Backend 配置

```yaml
Results:
  brpo_online_mapping:
    enabled: true
    trigger: keyframe
    placement_mode: quartile
    pseudo_map_iters: 20
    use_gauss_newton: true
    lambda_scale: 0.1
    ...
```

读取位置: `slam_backend.py` `_resolve_brpo_online_mapping_cfg()`

---

## 8. 关键文件索引

| 文件 | 作用 |
|------|------|
| `slam.py` | 主入口，初始化 frontend/backend |
| `slam_frontend.py` | Frontend 主循环，runtime state 缓存 |
| `slam_backend.py` | Backend 主循环，pseudo mapping 触发 |
| `slam_backend_brpo.py` | Pseudo mapping 核心逻辑 |
| `pose_gauss_newton.py` | GN pose optimization |
| `pseudo_camera_state.py` | apply_pose_delta_before_render_() |
| `runtime_exact_backend_bundle.py` | Pseudo view 创建，MASt3R matching |
| `runtime_slot_selector.py` | Pseudo slot 选择逻辑 |

---

## 9. 总结

### Frontend vs Backend 责任划分

| | Frontend | Backend |
|---|---|---|
| **处理对象** | Real frames (tracking) | Keyframes + Pseudo |
| **Pseudo 涉及** | 缓存 state，传递给 backend | 创建、render、optimize |
| **主要循环** | 每帧 tracking | 每个 KF mapping + pseudo |
| **Pose optimization** | Tracking 时 pose delta | Mapping 时 GN/Adam |
| **Depth 来源** | Dataset mono_depth | MASt3R projected depth |

### Pipeline 关键点

1. **Frontend 缓存所有帧的 state** - 传递给 Backend 选择 pseudo
2. **Backend 每个 KF 触发 pseudo mapping** - online mapping 模式
3. **apply_pose_delta_before_render_()** - 关键修复，在 render 前应用 pose delta
4. **Gauss-Newton + Scale** - 最优配置，ATE 改善 71%
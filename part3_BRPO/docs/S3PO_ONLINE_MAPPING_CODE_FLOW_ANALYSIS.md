# S3PO-GS Online Mapping 完整代码调用链分析

> 分析日期: 2026-05-06
> 目标: 详细分析 KF 和 Pseudo 的 mapping 流程差异

---

## 1. Difix Restoration 确认

### 结论：所有 D 系列和 Online Mapping **都没有做 Difix Restoration**

**证据**:

| 文件 | 搜索 difix/restoration | 结果 |
|------|----------------------|------|
| `slam_backend.py` | `grep difix` | 无结果 |
| `slam_backend_brpo.py` | `grep difix` | 无结果 |
| `runtime_exact_backend.py` | `grep difix` | 无结果 |
| `runtime_pseudo_builder.py` | `grep difix` | 无结果 |

**Difix 只存在于 Standalone Pipeline**:
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py` - 有 `stage_difix()`
- 这是 offline continuation 的预处理脚本，不在 online mapping 中

**所以**:
- D0-D4 实验都没有 difix restoration
- D5 配置也没有 difix 相关参数
- Online mapping 直接使用 coarse render + MASt3R projected depth

---

## 2. 完整调用链

### 2.1 入口

```python
# slam.py
class SLAM:
    def run(self):
        # Backend 作为独立进程
        backend_process = mp.Process(target=self.backend.run)
        backend_process.start()

        # Frontend 在主进程运行
        self.frontend.run()
```

### 2.2 Frontend Loop

```python
# slam_frontend.py
def run(self):
    while True:
        if cur_frame_idx >= len(dataset):
            # 评估并退出
            break

        # 1. 初始化 Camera
        viewpoint = Camera.init_from_dataset(dataset, cur_frame_idx, projection_matrix)

        # 2. Tracking (每帧)
        render_pkg = self.tracking(cur_frame_idx, viewpoint)

        # 3. 缓存 runtime_camera_state (所有帧)
        if brpo_online_mapping_enabled:
            self.runtime_camera_state_cache[cur_frame_idx] = self._export_runtime_camera_state(
                cur_frame_idx, viewpoint, is_keyframe=False
            )

        # 4. Keyframe 检测
        create_kf = self.is_keyframe(...) or (cur_frame_idx in force_keyframe_indices)

        if create_kf:
            # 5. 标记为关键帧
            self.runtime_camera_state_cache[cur_frame_idx] = self._export_runtime_camera_state(
                cur_frame_idx, viewpoint, is_keyframe=True
            )

            # 6. 发送 keyframe 消息给 Backend
            self.request_keyframe(cur_frame_idx, viewpoint, current_window, depth_map)
            # 消息包含: runtime_state_payload (所有缓存帧的 state dict)

        cur_frame_idx += 1
```

### 2.3 Backend Loop

```python
# slam_backend.py
def run(self):
    while True:
        if backend_queue.empty():
            # 正常 SLAM mapping (idle 时)
            self.map(self.current_window)

        else:
            data = backend_queue.get()

            if data[0] == "keyframe":
                cur_frame_idx = data[1]
                viewpoint = data[2]
                current_window = data[3]
                depth_map = data[4]
                runtime_state_payload = data[6]  # Frontend 缓存

                # 1. 更新 runtime_camera_states
                self.brpo_runtime_camera_states = runtime_state_payload

                # 2. 添加新关键帧
                self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                # 3. ⭐ KF Mapping (Real KFs only)
                self.map(self.current_window, iters=iter_per_kf, up_pose=True)
                self.map(self.current_window, prune=True)

                # 4. ⭐ 准备 Pseudo Slots
                prepare_payload = self._maybe_prepare_brpo_runtime_slots(cur_frame_idx)

                # 5. ⭐ Pseudo Mapping
                self._run_brpo_runtime_pseudo_mapping(cur_frame_idx, prepare_payload)

                # 6. 推送更新给 Frontend
                self.push_to_frontend("keyframe")
```

---

## 3. runtime_exact_backend.py 调用位置

### 调用链

```
Backend.run()
    ↓
处理 "keyframe" 消息
    ↓
_maybe_prepare_brpo_runtime_slots(cur_frame_idx)
    ↓
select_runtime_pseudo_slots()
    ↓
对每个 slot:
    ↓
⭐ build_runtime_exact_backend_bundle()  ← runtime_exact_backend.py
    ↓
    ├─ render_rgb_depth_from_state() - Render pseudo view (coarse)
    ├─ matcher.match_pair() - MASt3R matching (left/right refs)
    ├─ verify_single_branch_exact() - Epipolar verification
    └─ 创建 RuntimeExactBackendBundle
    ↓
build_runtime_exact_signal_bundle()
    ↓
build_runtime_pseudo_record_bundle()
    ↓
保存到 brpo_runtime_pseudo_records
```

### runtime_exact_backend.py 的作用

**不涉及 difix restoration**，只做:

| 功能 | 描述 |
|------|------|
| **Coarse render** | 用当前 Gaussian map render pseudo view |
| **MASt3R matching** | Dense matching between (pseudo, left_ref) and (pseudo, right_ref) |
| **Projected depth** | 从 MASt3R pts3d.z 获取 depth |
| **Verification** | Epipolar + depth consistency check |
| **Fusion weight** | Confidence-based fusion weights |

**输出**:
- `pseudo_render_rgb`: Coarse RGB (未修复)
- `pseudo_render_depth`: Render depth
- `projected_depth_left/right`: MASt3R projected depth
- `fusion_weight_left/right`: Confidence weights

---

## 4. KF Mapping vs Pseudo Mapping 对比

### 4.1 KF Mapping (`self.map()`)

**文件**: `slam_backend.py` `map()` (line 463)

**输入**:
- `current_window`: List of KF indices
- `viewpoints`: Camera objects with GT RGB + mono_depth

**流程**:
```python
def map(self, current_window, prune=False, iters=1, up_pose=True):
    viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]

    for _ in range(iters):
        # 1. Render each KF
        for viewpoint in viewpoint_stack:
            render_pkg = render(viewpoint, self.gaussians, ...)

            # 2. Loss: RGB + mono_depth (from dataset)
            loss_mapping += get_loss_mapping(config, image, viewpoint, depth=depth, monodepth=True)

        # 3. Isotropic regularization
        isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
        loss_mapping += 10 * isotropic_loss.mean()

        # 4. Backward
        loss_mapping.backward()

        # 5. Gaussian maintenance (densify/prune)
        with torch.no_grad():
            if prune:
                self.gaussians.prune_points(to_prune)

        # 6. Optimizer step
        self.gaussians.optimizer.step()
        pose_optimizer.step()  # if up_pose
```

**特点**:
| 特点 | 描述 |
|------|------|
| **RGB source** | Dataset GT RGB |
| **Depth source** | Dataset mono_depth |
| **Loss** | RGB + mono_depth + isotropic |
| **Densify/Prune** | ✓ 执行 |
| **Pose optimization** | Adam (无 GN) |
| **Pose delta 应用** | ❌ 不调用 apply_pose_delta_before_render_() |

### 4.2 Pseudo Mapping (`_run_brpo_runtime_pseudo_mapping()`)

**文件**: `slam_backend_brpo.py` `_run_joint_pseudo_engine()` (line 412)

**输入**:
- `pseudo_records`: BackendPseudoViewRecord list
- `current_window`: KF indices (用于 real loss)

**流程**:
```python
def _run_joint_pseudo_engine(self, cfg, pseudo_records, current_window, ...):
    for step_idx in range(num_iterations):
        total_loss = 0

        # 1. ⭐ Real KF Loss (same as KF mapping)
        for cam_idx in current_window:
            viewpoint = self.cameras[cam_idx]
            if update_real_pose:
                apply_pose_delta_before_render_(viewpoint)  # ⭐ CRITICAL FIX
            render_pkg = render(viewpoint, self.gaussians, ...)
            real_loss = get_loss_mapping(config, render_pkg["render"], viewpoint, depth=render_pkg["depth"])
            total_loss += lambda_real * real_loss

        # 2. ⭐ Pseudo Loss (不同!)
        for record in sampled_pseudo:
            apply_pose_delta_before_render_(record.viewpoint)  # ⭐ CRITICAL FIX
            render_pkg = render(record.viewpoint, self.gaussians, ...)

            # ⭐ 使用 projected depth + confidence mask
            pseudo_loss = compute_backend_pseudo_exact_loss(
                render_rgb=render_pkg["render"],
                render_depth=render_pkg["depth"],
                record=record,  # 包含 projected_depth + mask
                cfg=loss_cfg,
            )
            total_loss += lambda_pseudo * pseudo_loss

        # 3. ⭐ Scale regularization (不同于 KF mapping 的 isotropic)
        scale_loss = scale_reg_loss(self.gaussians, max_scale=max_scale)
        total_loss += lambda_scale * scale_loss

        # 4. Backward
        total_loss.backward()

        # 5. ⭐ Gauss-Newton (KF mapping 没有)
        if use_gauss_newton:
            gn_optimizer.optimize(record.viewpoint, gn_loss_fn)

        # 6. Optimizer step
        with torch.no_grad():
            self.gaussians.optimizer.step()
            if not use_gn:
                pose_optimizer.step()
            else:
                # GN 已直接更新 theta/rho

            # 7. ⭐ 不执行 densify/prune
            # enable_densify = False (default)
            # enable_prune = False (default)
```

**特点**:
| 特点 | 描述 |
|------|------|
| **RGB source** | ⭐ Coarse render (未 difix 修复) |
| **Depth source** | ⭐ MASt3R projected depth |
| **Loss** | Real RGB + Pseudo RGB-D + Scale reg |
| **Densify/Prune** | ❌ 不执行 (default disabled) |
| **Pose optimization** | ⭐ Gauss-Newton (可选) |
| **Pose delta 应用** | ⭐ apply_pose_delta_before_render_() |

---

## 5. 核心差异总结

### 5.1 输入数据差异

| 数据 | KF Mapping | Pseudo Mapping |
|------|-----------|----------------|
| **RGB** | Dataset GT | Coarse render (未修复) |
| **Depth** | Dataset mono_depth | MASt3R projected depth |
| **Confidence mask** | 无 | MASt3R confidence + epipolar verify |

### 5.2 Optimization 差异

| 操作 | KF Mapping | Pseudo Mapping |
|------|-----------|----------------|
| **Pose delta 应用** | ❌ 无 | ⭐ apply_pose_delta_before_render_() |
| **Pose optimizer** | Adam | ⭐ GN (可选) |
| **Gaussian densify** | ✓ 执行 | ❌ 禁用 |
| **Gaussian prune** | ✓ 执行 | ❌ 禁用 |
| **Scale reg** | Isotropic (fixed weight=10) | ⭐ scale_reg_loss (lambda_scale) |

### 5.3 Loss 差异

| Loss 项 | KF Mapping | Pseudo Mapping |
|---------|-----------|----------------|
| **RGB loss** | GT RGB vs render | Pseudo RGB vs render |
| **Depth loss** | mono_depth vs render | ⭐ Projected depth vs render (masked) |
| **Regularization** | Isotropic (weight=10) | ⭐ Scale reg (lambda_scale) |

---

## 6. 为什么 Pseudo Mapping 不做 Densify/Prune?

**原因**: Paper BRPO 设计原则

1. **Pseudo view 只用于 pose/scene refinement**
   - 不应该改变 Gaussian 数量
   - 只通过 gradient 更新现有 Gaussian 属性

2. **Densify/Prune 只由真实 KF 触发**
   - Real KF 有 GT RGB + mono_depth
   - 可以准确判断 coverage 和 quality
   - Pseudo view 的 confidence mask 不适合 densify 判断

3. **防止 pseudo 引入的 scale drift**
   - Densify 基于 gradient threshold
   - Pseudo depth 可能有 scale bias
   - 禁止 densify 避免 pseudo 引入新的 Gaussians

---

## 7. 执行顺序

每个 keyframe 到达后:

```
1. add_next_kf() - 添加新 KF
    ↓
2. ⭐ KF Mapping (iters=iter_per_kf)
    ├─ Render current_window KFs
    ├─ Loss: GT RGB + mono_depth
    ├─ Backward
    ├─ Densify/Prune
    └─ Pose optimizer step
    ↓
3. ⭐ KF Prune (prune=True)
    └─ Prune based on n_obs
    ↓
4. ⭐ 准备 Pseudo Slots
    ├─ select_runtime_pseudo_slots()
    ├─ build_runtime_exact_backend_bundle() (MASt3R)
    └─ 创建 pseudo records
    ↓
5. ⭐ Pseudo Mapping (iters=pseudo_map_iters)
    ├─ Render current_window KFs + Pseudo views
    ├─ Loss: Real RGB + Pseudo RGB-D + Scale reg
    ├─ Backward
    ├─ ⭐ Gauss-Newton (可选)
    ├─ ⭐ apply_pose_delta_before_render_()
    └─ Pose optimizer step (或 GN 已更新)
    ↓
6. push_to_frontend() - 更新 Frontend
```

---

## 8. 关键问题：Pose Delta 应用差异

### KF Mapping 不应用 Pose Delta?

**代码**:
```python
# slam_backend.py map()
for viewpoint in viewpoint_stack:
    render_pkg = render(viewpoint, ...)  # ❌ 没有调用 apply_pose_delta_before_render_()
```

**问题**: KF mapping 的 pose optimization 也不流回 gradient?

**答案**: KF mapping 有 pose optimizer:
```python
self.keyframe_optimizers = torch.optim.Adam([
    {"params": [viewpoint.cam_rot_delta], "lr": lr_rot},
    {"params": [viewpoint.cam_trans_delta], "lr": lr_trans},
])
```

**但是**: Adam optimizer 会更新 `cam_rot_delta` 和 `cam_trans_delta`，但 render 不使用它们！

**这是 S3PO 的原始问题**:
- KF mapping 的 pose optimizer 也有 gradient 不流回的问题
- 但 KF mapping 没有 fix，只有 pseudo mapping 有 fix

---

## 9. 结论

1. **所有 D 系列都没有 Difix Restoration** - Online mapping 缺失这个环节

2. **runtime_exact_backend.py** - Backend 在准备 pseudo slots 时调用，只做 MASt3R matching + projected depth

3. **KF vs Pseudo Mapping 差异**:
   - KF: GT data + densify/prune + Adam pose (无 pose delta fix)
   - Pseudo: MASt3R depth + no densify/prune + GN pose (有 pose delta fix)

4. **Pose delta fix 只在 Pseudo Mapping** - KF mapping 仍然有原始 S3PO 问题

5. **链条完整但缺少 difix** - 需要添加 restoration 环节
# Pseudo Gradient and Persistence Fix Plan

> 创建时间：2026-05-05 06:00 (Asia/Shanghai)
> 状态：待实施

---

## 0. 问题摘要

Online mapping pseudo enhancement "完全无效"的根本原因：

1. **split_authority backward 梯度丢弃**：pseudo loss backward 后立即 zero Gaussians grad，导致 Gaussians 只从 real loss 获得梯度
2. **临时 viewpoint 不持久化**：pseudo pose delta 只 fold 进临时对象，不影响后续 SLAM pipeline
3. **`.detach().clone()` 断梯度链**：pose fold 操作断开了 theta/rho → R/T 的梯度传播（虽然这不是主要问题，因为 fold 在 `torch.no_grad()` 内）

---

## 1. 原理分析

### 1.1 问题一：split_authority backward 梯度丢弃

**当前代码**（`slam_backend_brpo.py:549-556`）：

\`\`\`python
if sampled_pseudo and float(cfg.lambda_pseudo) != 0.0:
    need_after_pose = bool(scene_active_count > 0)
    # Step 1: pseudo_pose_loss backward
    (float(cfg.lambda_pseudo) * pseudo_pose_loss_sum).backward(retain_graph=need_after_pose)
    # ❌ Gaussians grad 被 zero！
    self._zero_optimizer_grads_(self.gaussians.optimizer)
    
    if scene_active_count > 0:
        # Step 2: pseudo_scene_loss backward
        (float(cfg.lambda_pseudo) * pseudo_scene_loss_sum).backward(retain_graph=True)
        # ❌ pose grad 被 zero！
        self._zero_pseudo_viewpoint_grads_(sampled_pseudo)

# Step 3: main_scene_objective backward（只有 real + scale）
main_scene_objective.backward()
\`\`\`

**设计意图**（推测）：
- "split_authority" 想让 pose 和 scene 分开优化
- pose 从 pseudo_pose_loss 获得梯度
- scene (Gaussians) 从 pseudo_scene_loss 获得梯度
- 通过 zero 阻止"cross-talk"

**实际效果**：
- pseudo_pose_loss backward → Gaussians.grad 被计算
- 但立即 `_zero_optimizer_grads_` → Gaussians.grad = 0
- pseudo_scene_loss backward → Gaussians.grad 再次计算（但可能 retain_graph=True 意味着是累积？）
- 然后 `_zero_pseudo_viewpoint_grads_` 只 zero pose，Gaussians.grad 应该保留？
- 最后 main_scene_objective.backward() → Gaussians.grad = real + scale

**关键错误**：
- `main_scene_objective = lambda_real * real_loss_sum + scale_weight * scale_loss`
- **不包含 pseudo_loss**！所以 Gaussians 最终梯度只有 real contribution

**验证方式**：
- 在 backward 前后打印 Gaussians.grad norm
- 确认是否 pseudo 贡献被丢弃

---

### 1.2 问题二：临时 viewpoint 不持久化

**当前代码路径**：

\`\`\`python
# runtime_pseudo_builder.py:46-48
viewpoint = create_viewpoint_from_state(pseudo_state)  # 新建临时对象
make_viewpoint_trainable(viewpoint)                     # 添加 theta/rho delta
record = BackendPseudoViewRecord(viewpoint=viewpoint)   # 包装成 record

# slam_backend_brpo.py:637
self._fold_pseudo_pose_residual_(record.viewpoint)  # fold delta 进 R/T
\`\`\`

**问题**：
- 这个 `viewpoint` 不在 `self.cameras` dict 中
- `_fold_pseudo_pose_residual_` 只修改临时对象的 R/T
- mapping block 结束后，这个临时对象被丢弃
- 后续 SLAM 的 keyframe selection、render、densify 都不使用这个 pseudo pose

**影响**：
- pseudo pose optimization 完全是"局部优化"
- 不会影响全局 scene geometry
- Gaussians 没有"pose-corrected"监督视角

---

### 1.3 问题三：.detach().clone() 断梯度链

**当前代码**（`slam_backend_brpo.py:141-152`）：

\`\`\`python
def _fold_pseudo_pose_residual_(viewpoint) -> None:
    new_w2c = current_w2c(viewpoint)  # 依赖 theta/rho，有梯度
    if hasattr(viewpoint, "update_RT"):
        viewpoint.update_RT(new_w2c[:3, :3].detach().clone(),  # ❌ detach 断梯度
                            new_w2c[:3, 3].detach().clone())
    else:
        viewpoint.R = new_w2c[:3, :3].detach().clone()  # ❌ detach 断梯度
        viewpoint.T = new_w2c[:3, 3].detach().clone()
        refresh_viewpoint_transforms_(viewpoint)
    with torch.no_grad():
        viewpoint.cam_rot_delta.zero_()
        viewpoint.cam_trans_delta.zero_()
\`\`\`

**分析**：
- 这个函数在 `torch.no_grad()` block 内调用（slam_backend_brpo.py:620-642）
- 所以 `.detach().clone()` 本身不是问题（no_grad 已经断梯度）
- 真正的问题是：**在 no_grad 内 fold，意味着 fold 操作不会影响梯度计算**

**实际影响**：
- fold 是合理的（累积 pose delta 进 R/T）
- 但应该让 fold 后的 pose 用于后续 iteration 的 render
- 目前临时 viewpoint 被丢弃，所以这个问题是问题二的子问题

---

## 2. 修复方案

### 2.1 方案一：简化 split_authority backward（推荐）

**目标**：让 Gaussians 从 pseudo loss 获得梯度

**修改位置**：`slam_backend_brpo.py:545-560`

**修改内容**：

\`\`\`python
# 修改前（问题代码）
if sampled_pseudo and float(cfg.lambda_pseudo) != 0.0:
    need_after_pose = bool(scene_active_count > 0)
    (float(cfg.lambda_pseudo) * pseudo_pose_loss_sum).backward(retain_graph=need_after_pose)
    self._zero_optimizer_grads_(self.gaussians.optimizer)  # ❌ 删除这行
    if scene_active_count > 0:
        (float(cfg.lambda_pseudo) * pseudo_scene_loss_sum).backward(retain_graph=True)
        self._zero_pseudo_viewpoint_grads_(sampled_pseudo)  # ❌ 删除这行
main_scene_objective.backward()

# 修改后（简化方案）
total_loss.backward()  # 直接做一次 backward，包含所有 loss
\`\`\`

**具体实现**：

\`\`\`python
# 方案 A：统一 backward（最简单）
if not split_authority:
    total_loss = total_loss + scale_weight * scale_loss
    total_loss.backward()
else:
    # 方案 B：保留 split_authority 但修复 zero 问题
    total_loss = (
        float(cfg.lambda_real) * real_loss_sum
        + float(cfg.lambda_pseudo) * (pseudo_pose_loss_sum + pseudo_scene_loss_sum)
        + scale_weight * scale_loss
    )
    total_loss.backward()  # 一次 backward，所有梯度累积
    
    # 不再 zero Gaussians.grad
    # 如果需要分离 pose/Gaussians 优化频率，可以：
    #   - 使用不同 optimizer.step() 时机
    #   - 而不是 zero grad
\`\`\`

**风险评估**：
- 低风险：backward 逻辑简化，反而更符合 PyTorch 标准
- 需测试：确认 pseudo loss 对 Gaussians 的梯度确实生效

---

### 2.2 方案二：pseudo pose 持久化

**目标**：让 pseudo pose delta 影响后续 SLAM pipeline

**设计选项**：

#### Option A：传播到相邻 keyframe pose（推荐）

**原理**：
- pseudo viewpoint 位于 kf_left 和 kf_right 之间
- pseudo pose delta 可以"部分传播"到相邻 keyframe
- 例如：pseudo 偏移 10%，则 kf_left/kf_right 各获得 5% 偏移

**实现**：

\`\`\`python
# 新增函数：传播 pseudo pose delta 到相邻 keyframe
def _propagate_pseudo_pose_to_neighbors_(
    self,
    record: BackendPseudoViewRecord,
    left_ref_frame_id: int,
    right_ref_frame_id: int,
    alpha: float = 0.5,  # 传播权重
) -> None:
    """将 pseudo pose delta 传播到相邻 keyframe。
    
    原理：
    - pseudo 位于 kf_left 和 kf_right 之间
    - pseudo pose delta 反映了"理想 pose"
    - 将 delta 的一部分传播给相邻 keyframe
    """
    pseudo_vp = record.viewpoint
    
    # 获取 pseudo pose delta（在 fold 前读取）
    pseudo_rot_delta = pseudo_vp.cam_rot_delta.detach().clone()
    pseudo_trans_delta = pseudo_vp.cam_trans_delta.detach().clone()
    
    # 传播到 left keyframe
    left_vp = self.cameras[left_ref_frame_id]
    if hasattr(left_vp, "cam_rot_delta"):
        with torch.no_grad():
            left_vp.cam_rot_delta.add_(alpha * pseudo_rot_delta)
            left_vp.cam_trans_delta.add_(alpha * pseudo_trans_delta)
    
    # 传播到 right keyframe
    right_vp = self.cameras[right_ref_frame_id]
    if hasattr(right_vp, "cam_rot_delta"):
        with torch.no_grad():
            right_vp.cam_rot_delta.add_((1 - alpha) * pseudo_rot_delta)
            right_vp.cam_trans_delta.add_((1 - alpha) * pseudo_trans_delta)
\`\`\`

**调用位置**：
- 在 `_fold_pseudo_pose_residual_` 之前调用
- 或者在 mapping block 结束后统一处理

**问题**：
- 需要知道每个 record 对应的 left/right ref frame id
- 目前 BackendPseudoViewRecord 没有 slot 信息

**修改 BackendPseudoViewRecord**：

\`\`\`python
# backend_pseudo_view_loader.py:19-35
@dataclass
class BackendPseudoViewRecord:
    sample_id: int
    frame_id: int
    viewpoint: Any
    target_rgb: np.ndarray
    target_depth: np.ndarray
    confidence_mask: np.ndarray
    source_map: Optional[np.ndarray]
    valid_mask: Optional[np.ndarray]
    target_confidence: Optional[np.ndarray]
    support_both_mask: Optional[np.ndarray]
    stageA_scene_scale: Optional[float] = None
    # 新增字段
    left_ref_frame_id: Optional[int] = None   # 新增
    right_ref_frame_id: Optional[int] = None  # 新增
    ...
\`\`\`

---

#### Option B：存储 pseudo viewpoint 到 cameras dict

**原理**：
- 把 pseudo viewpoint 加入 `self.cameras`
- 后续 SLAM 可以访问 pseudo viewpoint
- densify/prune 可以使用 pseudo 视角的监督

**实现**：

\`\`\`python
# 在 run_runtime_pseudo_mapping 结束后
for frame_id, record in runtime_pseudo_records.items():
    self.cameras[frame_id] = record.viewpoint  # pseudo viewpoint 持久化
\`\`\`

**问题**：
- frame_id 可能与 real keyframe ID 冲突（pseudo 用 midpoint ID 如 1.5？）
- SLAM 的 keyframe selection 逻辑可能不兼容非整数 ID
- densify/prune 可能不处理 pseudo viewpoint

**风险**：较高，需要改动 SLAM 核心逻辑

---

#### Option C：pseudo pose delta 累积到全局 pose correction buffer

**原理**：
- 不直接修改 keyframe pose
- 维护一个全局 pose correction dict
- 每次 render 时动态应用 correction

**实现**：

\`\`\`python
# 在 BRPOBackEndContinuation 中新增
self.pseudo_pose_corrections: dict[int, torch.Tensor] = {}  # frame_id -> pose_delta

# fold pseudo delta 时
self.pseudo_pose_corrections[record.frame_id] = (
    record.viewpoint.cam_rot_delta.detach().clone(),
    record.viewpoint.cam_trans_delta.detach().clone()
)

# 后续 render 时应用
def render_with_pseudo_correction(viewpoint, gaussians, pipe, bg, corrections):
    if viewpoint.uid in corrections:
        rot_delta, trans_delta = corrections[viewpoint.uid]
        viewpoint.cam_rot_delta.data = rot_delta
        viewpoint.cam_trans_delta.data = trans_delta
        apply_pose_delta_before_render_(viewpoint)
    return render(viewpoint, gaussians, pipe, bg)
\`\`\`

**优点**：
- 不修改 keyframe pose
- 可以选择性应用 correction
- 容易 debug

---

### 2.3 推荐实施方案

**Phase 1**：修复 split_authority backward（最关键，立即生效）
- 文件：`slam_backend_brpo.py:545-560`
- 改动：简化 backward，删除 zero grad 操作
- 测试：确认 pseudo loss 对 Gaussians 的梯度生效

**Phase 2**：pseudo pose 持久化（结构性修复）
- 文件：`backend_pseudo_view_loader.py`, `slam_backend_brpo.py`
- 改动：
  1. BackendPseudoViewRecord 新增 left/right ref frame id
  2. 新增 `_propagate_pseudo_pose_to_neighbors_` 函数
  3. 在 mapping block 结束后调用传播函数
- 测试：确认 pseudo pose delta 影响后续 keyframe

**Phase 3**：验证与调参
- 跑完整 online mapping 实验
- 对比 baseline vs fixed
- 调整传播权重 alpha

---

## 3. 具体代码修改清单

### 3.1 Phase 1：split_authority backward 修复

**文件**：`/home/bzhang512/CV_Project/third_party/S3PO-GS/utils/slam_backend_brpo.py`

**修改位置**：第 545-560 行

**修改内容**：

\`\`\`python
# === 原代码（删除） ===
if not split_authority:
    total_loss = total_loss + scale_weight * scale_loss
    total_loss.backward()
else:
    total_loss = (
        float(cfg.lambda_real) * real_loss_sum
        + float(cfg.lambda_pseudo) * (pseudo_pose_loss_sum + pseudo_scene_loss_sum)
        + scale_weight * scale_loss
    )
    if sampled_pseudo and float(cfg.lambda_pseudo) != 0.0:
        need_after_pose = bool(scene_active_count > 0)
        (float(cfg.lambda_pseudo) * pseudo_pose_loss_sum).backward(retain_graph=need_after_pose)
        self._zero_optimizer_grads_(self.gaussians.optimizer)
        if scene_active_count > 0:
            (float(cfg.lambda_pseudo) * pseudo_scene_loss_sum).backward(retain_graph=True)
            self._zero_pseudo_viewpoint_grads_(sampled_pseudo)
    main_scene_objective = float(cfg.lambda_real) * real_loss_sum + scale_weight * scale_loss
    main_scene_objective.backward()

# === 新代码（统一 backward） ===
# 简化：不再分离 pose/scene backward，统一处理
total_loss = (
    float(cfg.lambda_real) * real_loss_sum
    + float(cfg.lambda_pseudo) * (pseudo_pose_loss_sum + pseudo_scene_loss_sum)
    + scale_weight * scale_loss
)
total_loss.backward()
\`\`\`

---

### 3.2 Phase 2：pseudo pose 持久化

**文件 1**：`/home/bzhang512/CV_Project/part3_BRPO/pseudo_branch/refine/backend_pseudo_view_loader.py`

**修改内容**：BackendPseudoViewRecord 新增字段

\`\`\`python
@dataclass
class BackendPseudoViewRecord:
    sample_id: int
    frame_id: int
    viewpoint: Any
    target_rgb: np.ndarray
    target_depth: np.ndarray
    confidence_mask: np.ndarray
    source_map: Optional[np.ndarray]
    valid_mask: Optional[np.ndarray]
    target_confidence: Optional[np.ndarray]
    support_both_mask: Optional[np.ndarray]
    stageA_scene_scale: Optional[float] = None
    target_rgb_path: Optional[str] = None
    target_depth_path: Optional[str] = None
    confidence_path: Optional[str] = None
    observation_meta_path: Optional[str] = None
    # === 新增字段 ===
    left_ref_frame_id: Optional[int] = None
    right_ref_frame_id: Optional[int] = None
\`\`\`

**文件 2**：`/home/bzhang512/CV_Project/part3_BRPO/pseudo_branch/integration/runtime_pseudo_builder.py`

**修改位置**：第 51-67 行（build_runtime_pseudo_record_bundle 函数）

**修改内容**：传递 left/right ref frame id

\`\`\`python
record = BackendPseudoViewRecord(
    sample_id=int(slot.frame_id),
    frame_id=int(slot.frame_id),
    viewpoint=viewpoint,
    target_rgb=...,
    ...
    # === 新增 ===
    left_ref_frame_id=int(slot.left_ref_frame_id),
    right_ref_frame_id=int(slot.right_ref_frame_id),
)
\`\`\`

**文件 3**：`/home/bzhang512/CV_Project/third_party/S3PO-GS/utils/slam_backend_brpo.py`

**新增函数**（第 150 行附近）：

\`\`\`python
@staticmethod
def _propagate_pseudo_pose_to_neighbors_(
    viewpoint,
    left_vp,
    right_vp,
    alpha: float = 0.3,
) -> None:
    """将 pseudo pose delta 传播到相邻 keyframe。
    
    Args:
        viewpoint: pseudo viewpoint (含 cam_rot_delta, cam_trans_delta)
        left_vp: 左侧 keyframe viewpoint
        right_vp: 右侧 keyframe viewpoint
        alpha: 传播给 left 的权重 (1-alpha 给 right)
    """
    pseudo_rot_delta = viewpoint.cam_rot_delta.detach().clone()
    pseudo_trans_delta = viewpoint.cam_trans_delta.detach().clone()
    
    # 传播到 left keyframe
    if hasattr(left_vp, "cam_rot_delta") and getattr(left_vp, "uid", None) != 0:
        with torch.no_grad():
            left_vp.cam_rot_delta.add_(alpha * pseudo_rot_delta)
            left_vp.cam_trans_delta.add_(alpha * pseudo_trans_delta)
    
    # 传播到 right keyframe
    if hasattr(right_vp, "cam_rot_delta") and getattr(right_vp, "uid", None) != 0:
        with torch.no_grad():
            right_vp.cam_rot_delta.add_((1 - alpha) * pseudo_rot_delta)
            right_vp.cam_trans_delta.add_((1 - alpha) * pseudo_trans_delta)
\`\`\`

**修改调用位置**（第 637 行附近）：

\`\`\`python
# 原代码
if cfg.update_pseudo_pose:
    for record in sampled_pseudo:
        self._fold_pseudo_pose_residual_(record.viewpoint)

# 新代码
if cfg.update_pseudo_pose:
    for record in sampled_pseudo:
        # 先传播到相邻 keyframe
        if record.left_ref_frame_id is not None and record.right_ref_frame_id is not None:
            left_vp = self.cameras.get(int(record.left_ref_frame_id))
            right_vp = self.cameras.get(int(record.right_ref_frame_id))
            if left_vp is not None and right_vp is not None:
                self._propagate_pseudo_pose_to_neighbors_(
                    viewpoint=record.viewpoint,
                    left_vp=left_vp,
                    right_vp=right_vp,
                    alpha=0.3,  # 可配置
                )
        # 然后 fold pseudo delta
        self._fold_pseudo_pose_residual_(record.viewpoint)
\`\`\`

---

## 4. 验证计划

### 4.1 Phase 1 验证

**测试命令**：

\`\`\`bash
# 在 mapping block 中插入 debug 打印
# 修改 slam_backend_brpo.py，在 backward 后打印 Gaussians.grad norm

# 运行实验
python scripts/run_online_mapping_experiment.py --config configs/D2_fixed.yaml
\`\`\`

**验证指标**：
- Gaussians.get_xyz.grad.norm() 在 backward 后是否非零
- pseudo loss 下降速度是否加快
- 最终 rendering quality 是否改善

---

### 4.2 Phase 2 验证

**测试命令**：

\`\`\`bash
# 运行完整 online mapping
python scripts/run_online_mapping_experiment.py --config configs/D3_persist.yaml

# 检查 keyframe pose delta 是否被更新
# 在日志中搜索 "cam_rot_delta" / "cam_trans_delta" 的变化
\`\`\`

**验证指标**：
- adjacent keyframe 的 pose delta 是否被更新
- 后续 keyframe render 是否使用 updated pose
- 最终 trajectory accuracy 是否改善

---

## 5. 风险评估

| 修改 | 风险等级 | 说明 |
|------|----------|------|
| Phase 1: backward 简化 | 低 | 符合 PyTorch 标准，预期效果明显 |
| Phase 2: pose 持久化 | 中 | 改动较多，需确保不破坏现有 SLAM 流程 |
| alpha 传播权重 | 低 | 可调参数，可从 0.3 开始测试 |

---

## 6. 时间线

- Day 1：Phase 1 实施 + 验证
- Day 2：Phase 2 实施 + 验证
- Day 3：完整实验对比 + 调参

---

## 7. 相关文件

- `slam_backend_brpo.py`：核心 mapping loop
- `pseudo_camera_state.py`：pose delta 处理
- `backend_pseudo_view_loader.py`：record 数据结构
- `runtime_pseudo_builder.py`：record 构建
- `POSE_GRADIENT_DIAGNOSIS_20260505.md`：问题诊断文档

---

> 状态：待实施，等待用户确认后执行

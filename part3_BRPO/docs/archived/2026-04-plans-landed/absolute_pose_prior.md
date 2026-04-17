# Absolute Pose Prior 落地方案

## 一、问题背景

当前 Stage A 的 `pose_reg_loss` 只约束每步的 residual (cam_rot_delta / cam_trans_delta)，但 residual 在 `apply_pose_residual_()` 后会被清零折回 R/T。这意味着 pose_reg 无法约束累计 drift。

Absolute Pose Prior 的核心思路：存储初始位姿 R0/T0，约束当前位姿相对初始位姿的 SE(3) 偏移量。

---

## 二、实现方案

### 2.1 新增 SE3_log 函数

**文件**: `third_party/S3PO-GS/utils/pose_utils.py`

在现有 `SE3_exp` 函数后添加：

```python
def SO3_log(R: torch.Tensor) -> torch.Tensor:
    """从 rotation matrix 提取 axis-angle (3D)"""
    device = R.device
    dtype = R.dtype
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))
    if angle < 1e-7:
        return torch.zeros(3, device=device, dtype=dtype)
    axis = torch.stack([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2 * torch.sin(angle))
    return angle * axis

def SE3_log(T: torch.Tensor) -> torch.Tensor:
    """从 SE(3) matrix 提取 6D tangent space vector [rho, theta]"""
    device = T.device
    dtype = T.dtype
    R = T[:3, :3]
    t = T[:3, 3]
    theta = SO3_log(R)
    angle = torch.norm(theta)
    if angle < 1e-7:
        rho = t
    else:
        V_inv = V(theta).inverse()
        rho = V_inv @ t
    return torch.cat([rho, theta], dim=0)
```

### 2.2 修改 make_viewpoint_trainable 存储初始位姿

**文件**: `pseudo_branch/pseudo_camera_state.py`

修改函数：

```python
def make_viewpoint_trainable(vp):
    vp.cam_rot_delta = _ensure_parameter(getattr(vp, 'cam_rot_delta', None), (3,))
    vp.cam_trans_delta = _ensure_parameter(getattr(vp, 'cam_trans_delta', None), (3,))
    vp.exposure_a = _ensure_parameter(getattr(vp, 'exposure_a', None), (1,))
    vp.exposure_b = _ensure_parameter(getattr(vp, 'exposure_b', None), (1,))
    # 新增：存储初始位姿
    vp.R0 = vp.R.detach().clone()
    vp.T0 = vp.T.detach().clone()
    return vp
```

### 2.3 新增 absolute_pose_prior_loss

**文件**: `pseudo_branch/pseudo_loss_v2.py`

添加 import 和函数：

```python
# 在文件顶部添加
from utils.pose_utils import SE3_log

def absolute_pose_prior_loss(viewpoint) -> torch.Tensor:
    """约束当前位姿相对初始位姿的 SE(3) 偏移"""
    # 构建 initial w2c
    T0 = torch.eye(4, device=viewpoint.R0.device, dtype=viewpoint.R0.dtype)
    T0[:3, :3] = viewpoint.R0
    T0[:3, 3] = viewpoint.T0
    
    # 获取当前 w2c
    current_w2c_mat = current_w2c(viewpoint)
    
    # 计算相对变换 ΔT = current_w2c @ T0^{-1}
    T0_inv = torch.linalg.inv(T0)
    delta_T = current_w2c_mat @ T0_inv
    
    # SE(3) log
    tau = SE3_log(delta_T)
    return torch.norm(tau, p=2) ** 2
```

**注意**: 需要在文件中 import `current_w2c` 或直接内联实现。

### 2.4 修改 build_stageA_loss_source_aware 加入 abs_pose loss

**文件**: `pseudo_branch/pseudo_loss_v2.py`

修改函数签名和逻辑：

```python
def build_stageA_loss_source_aware(
    render_rgb,
    render_depth,
    target_rgb,
    target_depth,
    confidence_mask,
    depth_source_map,
    viewpoint,
    beta_rgb: float,
    lambda_pose: float,
    lambda_exp: float,
    trans_weight: float,
    lambda_depth_seed: float = 1.0,
    lambda_depth_dense: float = 0.35,
    lambda_depth_fallback: float = 0.0,
    use_depth: bool = True,
    lambda_abs_pose: float = 0.0,  # 新增参数
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # ... 现有 loss 计算 ...
    
    l_abs_pose = absolute_pose_prior_loss(viewpoint) if lambda_abs_pose > 0 else torch.tensor(0.0, device=render_rgb.device)
    
    total = (
        float(beta_rgb) * l_rgb 
        + (1.0 - float(beta_rgb)) * l_depth 
        + float(lambda_pose) * l_pose 
        + float(lambda_exp) * l_exp
        + float(lambda_abs_pose) * l_abs_pose  # 新增
    )
    
    stats = {
        # ... 现有字段 ...
        'loss_abs_pose_reg': float(l_abs_pose.detach().item()),  # 新增
        'loss_total': float(total.detach().item()),
    }
    return total, stats
```

### 2.5 修改 StageAConfig

**文件**: `pseudo_branch/pseudo_refine_scheduler.py`

```python
@dataclass
class StageAConfig:
    num_iterations: int = 300
    beta_rgb: float = 0.7
    lambda_pose: float = 0.01
    lambda_exp: float = 0.001
    trans_reg_weight: float = 1.0
    lr_rot: float = 0.003
    lr_trans: float = 0.001
    lr_exp: float = 0.01
    num_pseudo_views: int = 4
    lambda_abs_pose: float = 0.0  # 新增
```

### 2.6 修改 run_pseudo_refinement_v2.py

**文件**: `scripts/run_pseudo_refinement_v2.py`

1. **添加 CLI 参数**:

```python
p.add_argument('--stageA_lambda_abs_pose', type=float, default=0.0,
               help='Weight for absolute pose prior loss (default: 0.0, disabled)')
```

2. **传递参数到 StageAConfig**:

```python
cfg = StageAConfig(
    # ... 现有参数 ...
    lambda_abs_pose=args.stageA_lambda_abs_pose,  # 新增
)
```

3. **传递参数到 loss 函数**:

在调用 `build_stageA_loss_source_aware` 时添加:

```python
loss, stats = build_stageA_loss_source_aware(
    # ... 现有参数 ...
    lambda_abs_pose=cfg.lambda_abs_pose,  # 新增
)
```

4. **添加 history 记录**:

```python
history = {
    # ... 现有字段 ...
    'loss_abs_pose_reg': [],  # 新增
}
```

5. **修改 export_view_state 导出 R0/T0** (可选，用于分析):

```python
def export_view_state(view: Dict[str, Any]) -> Dict[str, Any]:
    # ... 现有代码 ...
    state = ExportedPseudoCameraState(
        # ... 现有字段 ...
    )
    # 新增：导出初始位姿和偏移量
    result = state.to_dict()
    result['R0'] = vp.R0.detach().cpu().tolist() if hasattr(vp, 'R0') else None
    result['T0'] = vp.T0.detach().cpu().tolist() if hasattr(vp, 'T0') else None
    return result
```

---

## 三、验证步骤

### 3.1 单元测试 SE3_log

```python
# 测试脚本
import torch
from utils.pose_utils import SE3_exp, SE3_log, SO3_exp, SO3_log

# 测试 1: identity
tau = torch.zeros(6)
T = SE3_exp(tau)
tau_back = SE3_log(T)
print(f'Identity test: ||tau_back|| = {torch.norm(tau_back).item():.6f}')

# 测试 2: 随机 tau
tau = torch.randn(6) * 0.1
T = SE3_exp(tau)
tau_back = SE3_log(T)
error = torch.norm(tau - tau_back)
print(f'Random tau test: ||error|| = {error.item():.6f}')
```

### 3.2 集成测试

运行 Stage A with abs_pose prior:

```bash
python scripts/run_pseudo_refinement_v2.py \
    --ply_path /path/to/pointcloud.ply \
    --pseudo_cache /path/to/pseudo_cache \
    --output_dir /path/to/output_abs_pose_test \
    --stageA_lambda_abs_pose 0.1 \
    --stageA_iters 50
```

检查:
1. `loss_abs_pose_reg` 是否有效计算
2. `loss_depth` 是否有更明显下降
3. 最终 pose drift 是否被约束

---

## 四、实验配置

| 组别 | beta_rgb | lambda_pose | lambda_abs_pose | 说明 |
|------|----------|-------------|-----------------|------|
| baseline | 0.7 | 0.01 | 0.0 | 当前默认 |
| abs_pose_default | 0.7 | 0.01 | 0.1 | 默认配置 + abs pose |
| abs_pose_strong | 0.7 | 0.01 | 1.0 | 强约束 |
| abs_pose_depth_heavy | 0.3 | 0.01 | 0.1 | depth-heavy + abs pose |

---

## 五、风险与备选

1. **SE3_log 数值稳定性**: 当 rotation 接近 identity 时，需用 Taylor 展开。已在代码中用 `angle < 1e-7` 处理。
2. **过约束**: 如果 lambda_abs_pose 过大，可能阻止必要调整。建议从 0.01 开始尝试。
3. **与 pose_reg 重复**: 两者约束不同目标 - pose_reg 约束单步 residual，abs_pose 约束累计 drift。可以共存，但可能需要调整权重。

---

## 六、文件改动清单

| 文件 | 改动类型 | 内容 |
|------|----------|------|
| `third_party/S3PO-GS/utils/pose_utils.py` | 新增函数 | `SO3_log`, `SE3_log` |
| `pseudo_branch/pseudo_camera_state.py` | 修改函数 | `make_viewpoint_trainable` 增加 R0/T0 |
| `pseudo_branch/pseudo_loss_v2.py` | 新增函数 + 修改 | `absolute_pose_prior_loss`; 修改 `build_stageA_loss_source_aware` |
| `pseudo_branch/pseudo_refine_scheduler.py` | 修改 dataclass | `StageAConfig` 增加 `lambda_abs_pose` |
| `scripts/run_pseudo_refinement_v2.py` | 修改 | CLI 参数、参数传递、history 记录 |
# Paper 路线 C_m 生成修复方案

**日期**: 2026-05-07
**状态**: ✅ 已实现
**优先级**: 高

---

## 问题描述

### 现象

在 E2 (paper_split_rgbonly) 实验中，发现 `pseudo_confidence_exact_brpo_upstream_target_v1.png` (C_m) 和 `pseudo_valid_mask_exact_brpo_upstream_target_v1.png` (valid_mask) **完全一致**：

```
C_m positive: 42945 pixels (16.38%)
valid_mask positive: 42945 pixels (16.38%)
交集: 42945 pixels (完全重叠)
```

### 根本原因

**`brpo_reprojection_verify.py:157` 的 support 定义包含 depth 验证**：

```python
support = valid_ref_depth & in_bounds & valid_pseudo_depth & (reproj_err < tau_reproj_px) & (rel_depth_err < tau_rel_depth)
```

这意味着 `support_mask` = MASt3R RGB 匹配 **经过 depth 验证过滤后** 的结果。

### 数据证据

| 方向 | MASt3R 匹配数 | depth验证后 | 过滤率 |
|------|--------------|------------|--------|
| Left | 66538 | 14380 | **78.4%** |
| Right | 54324 | 33312 | **38.7%** |

**depth 验证过滤掉了 ~78% 和 ~39% 的 RGB 匹配！**

---

## Paper 路线需求

Paper 方法的 C_m 定义：
- **仅基于 RGB 匹配**（MASt3R matcher 输出 pts_pseudo/pts_ref）
- **不依赖 depth 验证**
- C_m = support_left ∩ support_right (both=1.0) ∪ support_left ⊕ support_right (xor=0.5)

当前代码错误地将 depth 验证混入了 C_m 生成流程。

---

## 代码流程分析

### 当前流程（错误）

```
MASt3R matcher.match_pair()
    → pts_pseudo, pts_ref, match_conf (纯 RGB 匹配)
    ↓
verify_single_branch_exact()  [brpo_reprojection_verify.py]
    → support = RGB + depth 双重验证
    → support_mask (已含 depth 过滤)
    ↓
build_exact_brpo_upstream_target_observation()  [pseudo_observation_brpo_style.py]
    → confidence_cm = support_left/support_right 的 overlap
    ↓
C_m = confidence_cm (被 depth 验证污染)
```

### 正确流程（Paper 路线）

```
MASt3R matcher.match_pair()
    → pts_pseudo, pts_ref, match_conf (纯 RGB 匹配)
    ↓
build_rgb_mask_from_correspondences()  [rgb_mask_inference.py]
    → support_left/right = 直接从 pts_fused 构建（无 depth 验证）
    → raw_rgb_confidence_v2 = 纯 RGB C_m
    ↓
C_m = raw_rgb_confidence_v2 (仅 RGB 匹配)
```

---

## 关键文件

| 文件 | 当前作用 | 需修改 |
|------|---------|--------|
| `brpo_reprojection_verify.py` | RGB+depth 验证 | 添加 rgb_only 模式 |
| `rgb_mask_inference.py` | 纯 RGB mask 生成（未被调用） | 需集成 |
| `runtime_exact_backend.py` | 调用 verify_single_branch_exact | 添加 rgb_only_verification 选项 |
| `runtime_signal_builder.py` | 构建 signal bundle | 根据 depth_loss_mode 选择 C_m 来源 |
| `pseudo_observation_brpo_style.py` | 构建 C_m | 需区分 rgb_only vs exact |

---

## 解决方案

### 方案 A：修改 verify_single_branch_exact（推荐）

在 `brpo_reprojection_verify.py` 中添加 `rgb_only` 参数：

```python
def verify_single_branch_exact(
    pseudo_state: Dict,
    ref_state: Dict,
    pseudo_depth: np.ndarray,
    ref_depth: np.ndarray,
    pts_pseudo: np.ndarray,
    pts_ref: np.ndarray,
    tau_reproj_px: float = 4.0,
    tau_rel_depth: float = 0.15,
    ref_side: str = "left",
    ref_frame_id: int = 0,
    rgb_only: bool = False,  # 新增参数
):
    """
    Args:
        rgb_only: If True, skip depth verification and use only RGB correspondence.
                  For paper_brpo_split mode.
    """
    # ... existing code ...

    if rgb_only:
        # Paper route: only RGB verification, no depth check
        support = in_bounds & (reproj_err < tau_reproj_px)
        # No depth validation, no rel_depth_err threshold
    else:
        # Exact route: RGB + depth verification
        support = valid_ref_depth & in_bounds & valid_pseudo_depth & (reproj_err < tau_reproj_px) & (rel_depth_err < tau_rel_depth)

    # ... rest of code ...
```

### 方案 B：直接使用 rgb_mask_inference

修改 `runtime_exact_backend.py`：

```python
def build_runtime_exact_backend_bundle(
    ...
    rgb_only_verification: bool = False,  # 新增参数
):
    if rgb_only_verification:
        from pseudo_branch.mask.rgb_mask_inference import build_rgb_mask_from_correspondences

        rgb_mask_result = build_rgb_mask_from_correspondences(
            fused_rgb_path=pseudo_rgb_path,
            left_ref_rgb_path=left_ref_rgb_path,
            right_ref_rgb_path=right_ref_rgb_path,
            matcher=matcher,
            size=size,
        )

        # 使用纯 RGB mask
        left_result = {
            "support_mask": rgb_mask_result["support_left"],
            "confidence_map": rgb_mask_result["raw_rgb_confidence_left_cont_v2"],
            ...
        }
        right_result = {
            "support_mask": rgb_mask_result["support_right"],
            "confidence_map": rgb_mask_result["raw_rgb_confidence_right_cont_v2"],
            ...
        }
    else:
        # 原有 exact 流程
        left_result = verify_single_branch_exact(...)
        right_result = verify_single_branch_exact(...)
```

### 方案 C：分离 C_m 和 depth target

在 `pseudo_observation_brpo_style.py` 中：

```python
def build_exact_brpo_upstream_target_observation(
    ...
    cm_source: str = "exact_backend",  # 新增参数: "exact_backend" | "rgb_only"
):
    if cm_source == "rgb_only":
        # Paper route: C_m from RGB mask, depth target from exact backend
        confidence_cm = ...  # 从 rgb_mask_inference 获取
        valid_mask = ...     # 从 exact backend 获取（depth target 覆盖）
    else:
        # Exact route: C_m 和 valid_mask 都来自 exact backend
        confidence_cm = ...
        valid_mask = ...
```

---

## 配置层面

在 YAML 配置中添加选项：

```yaml
brpo_online_mapping:
  # C_m 来源模式
  cm_generation_mode: "exact_backend"  # 或 "rgb_only"

  # 当 cm_generation_mode="rgb_only" 时:
  # - C_m 仅基于 RGB 匹配（无 depth 验证）
  # - depth target 仍使用 exact backend 的 projected depth
```

或通过 `depth_loss_mode` 自动推断：

```python
if depth_loss_mode == "paper_brpo_split_v1":
    rgb_only_verification = True  # Paper 路线
elif depth_loss_mode == "exact_shared_cm_v1":
    rgb_only_verification = False  # Exact 路线
```

---

## 实现步骤

1. **Step 1**: 在 `brpo_reprojection_verify.py` 添加 `rgb_only` 参数
2. **Step 2**: 在 `RuntimeExactBackendConfig` 添加 `rgb_only_verification` 字段
3. **Step 3**: 修改 `build_runtime_exact_backend_bundle` 根据配置选择验证模式
4. **Step 4**: 在 `runtime_signal_builder.py` 确认 C_m 正确传递
5. **Step 5**: 添加配置选项 `cm_generation_mode`
6. **Step 6**: 测试验证 C_m 覆盖率变化

---

## 验证方法

修复后，预期 C_m 覆盖范围显著增加：

| 方向 | 当前 (depth验证) | 修复后 (RGB only) |
|------|-----------------|------------------|
| Left | 5.49% | ~25.4% |
| Right | 12.71% | ~20.7% |
| Union | 16.38% | ~40%+ |

验证脚本：

```python
import numpy as np

base = '.../event_kf_0271/frame_0254'

# RGB only C_m
rgb_cm = np.load(base + '/signal_v2/raw_rgb_confidence_v2.npy')

# 当前 C_m (含 depth)
current_cm = np.load(base + '/signal_v2/pseudo_confidence_exact_brpo_upstream_target_v1.npy')

print(f'RGB-only C_m coverage: {(rgb_cm > 0).mean():.4f}')
print(f'Current C_m coverage: {(current_cm > 0).mean():.4f}')
# 预期: RGB-only > current
```

---

## 相关文件

- `DEPTH_GENERATION_CHANGE_PLAN_20260506.md` - Phase A depth 生成改动
- `PART3_BRPO_ONLINE_MAPPING_PIPELINE.md` - Pipeline 总览

---

## 附录：关键代码片段

### A. brpo_reprojection_verify.py:157（问题根源）

```python
# 当前: RGB + depth 双重验证
support = valid_ref_depth & in_bounds & valid_pseudo_depth & (reproj_err < tau_reproj_px) & (rel_depth_err < tau_rel_depth)

# Paper route 应该是:
# support = in_bounds & (reproj_err < tau_reproj_px)  # 仅 RGB 验证
```

### B. rgb_mask_inference.py:55-60（正确实现）

```python
support_left = left_maps['support_mask'] > 0.5
support_right = right_maps['support_mask'] > 0.5
support_both = support_left & support_right
support_single = support_left ^ support_right

discrete = np.full((h, w), fill_value=float(value_none), dtype=np.float32)
discrete[support_single] = float(value_single)
discrete[support_both] = float(value_both)
# 这是纯 RGB C_m，无 depth 验证
```

### C. exact_backend_meta.json 数据

```json
{
  "tau_rel_depth": 1.0,  // 已经很宽松 (100%)
  "left_stats": {
    "num_matches": 66538,  // MASt3R 匹配数
    "num_support": 14380,  // depth 验证后
    "support_ratio_vs_matches": 0.216  // 只有 21.6% 通过
  }
}
```

---

## 实现结果

### 采用方案

**方案 B（修改版）**：分离 C_m 和 depth target 来源

- C_m (support_mask) 来自 `_accumulate_match_maps`（纯 RGB，无 depth 验证）
- projected_depth 来自 `verify_single_branch_exact`（用于 depth target）

### 改动文件

**`runtime_exact_backend.py`**

1. 添加 `RuntimeExactBackendConfig.rgb_only_verification: bool = False` 字段

2. 添加 `_accumulate_match_maps` 导入：
   ```python
   from pseudo_branch.mask.rgb_mask_inference import _accumulate_match_maps
   ```

3. 在 `build_runtime_exact_backend_bundle` 中添加分支逻辑：
   ```python
   if cfg.rgb_only_verification:
       # Step 1: RGB-only mask for C_m
       left_rgb_maps = _accumulate_match_maps(image_shape=(h, w), pts_fused=pts_pseudo_left, conf=match_conf_left)
       right_rgb_maps = _accumulate_match_maps(image_shape=(h, w), pts_fused=pts_pseudo_right, conf=match_conf_right)

       # Step 2: Exact projected_depth for depth target
       left_exact_result = verify_single_branch_exact(...)
       right_exact_result = verify_single_branch_exact(...)

       # Step 3: Merge: RGB-only support_mask + exact projected_depth
       left_result = {
           "support_mask": left_rgb_maps["support_mask"],  # RGB-only for C_m
           "confidence_map": left_rgb_maps["conf_map"],    # RGB-only confidence
           "projected_depth_map": left_exact_result["projected_depth_map"],  # exact for depth target
           ...
       }
   ```

4. 在 `exact_meta` 中添加标记：
   ```python
   "rgb_only_verification": bool(cfg.rgb_only_verification),
   "cm_generation_mode": "rgb_only" if cfg.rgb_only_verification else "exact_backend",
   ```

### 验证状态

- ✅ 静态检查：导入正确，字段兼容
- ✅ Smoke 测试：`_accumulate_match_maps` 输出格式正确
- ⏳ 实验验证：需要实际跑 E2 实验确认 C_m 覆盖范围增大

### 预期效果

当 `rgb_only_verification=True` 时：

| 组件 | 来源 | 覆盖范围 |
|------|------|---------|
| C_m (support_mask) | RGB only | 大（~MASt3R 匹配数） |
| projected_depth | exact backend | 小（~depth 验证后） |
| valid_mask | projected_depth union | 小 |

**C_m 和 valid_mask 将不再一致！**

### 备份位置

原文件备份：`runtime_exact_backend.py.bak_20260507_rgbonly_fix`

### 下一步

1. 在 YAML 配置中启用 `rgb_only_verification: true`
2. 重跑 E2 实验
3. 验证 `exact_backend_meta.json` 中 `cm_generation_mode="rgb_only"`
4. 对比 C_m 覆盖范围变化
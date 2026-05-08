# Part3 BRPO Online Mapping 完整审查报告

> 审查日期: 2026-05-06
> 审查范围: Difix Restoration 缺失 + Hermes 修复 + 链条完整性

---

## 🚨 严重问题：Difix Restoration 缺失

### 问题确认

**Standalone Pipeline** (`prepare_stage1_difix_dataset_s3po_internal.py`) 包含 difix restoration:

```python
def stage_difix(args, run_root: Path):
    model = load_difix_model(args.difix_model_name, args.difix_model_path, args.timestep)

    for rec in records:
        # 左侧参考修复
        run_single_difix(
            model_bundle=model,
            image_path=Path(rec.render_input_path),
            ref_path=Path(rec.left_ref_input_path),
            output_path=Path(rec.left_fixed_path),
        )
        # 右侧参考修复
        run_single_difix(
            model_bundle=model,
            image_path=Path(rec.render_input_path),
            ref_path=Path(rec.right_ref_input_path),
            output_path=Path(rec.right_fixed_path),
        )
```

**Online Mapping Pipeline** (`runtime_exact_backend.py`) **缺失 difix restoration**:

```python
def build_runtime_exact_backend_bundle(...):
    # 只做：
    # 1. Render coarse pseudo view
    pseudo_render_rgb, pseudo_render_depth = render_rgb_depth_from_state(...)

    # 2. MASt3R matching (left/right refs)
    pts_pseudo, pts_ref_left, _ = matcher.match_pair(...)
    pts_pseudo, pts_ref_right, _ = matcher.match_pair(...)

    # 3. Verify exact branch
    left_result = verify_single_branch_exact(...)
    right_result = verify_single_branch_exact(...)

    # ❌ 没有 Difix restoration
    # ❌ 没有 RGB fusion
```

### 缺失功能

| 功能 | Standalone | Online Mapping |
|------|-----------|----------------|
| **Difix restoration** | ✓ 双向修复 | ❌ 缺失 |
| **RGB fusion** | ✓ residual fusion | ❌ 缺失 |
| **Depth target** | 从 fused RGB 计算 | MASt3R projected depth |
| **Restoration stage** | stage_difix() | 无 |

### 影响分析

**Online Mapping 当前流程**:
```
Coarse render → MASt3R matching → projected depth → optimization
```

**缺失的环节**:
```
Coarse render → [缺失: Difix修复] → [缺失: fusion] → MASt3R → optimization
```

**后果**:
- Pseudo RGB 是 coarse render 质量，有 artifacts
- 缺少 restoration 提升质量
- 与 paper 设计不符（paper 有 restoration）

---

## Hermes 修复审查

### 1. KF0 Pose 跳过逻辑移除 ✓

**文件**: `slam_backend.py`
**修改**: 删除 `if viewpoint.uid == 0: continue`

**审查结果**: **正确**

**原因**: DL3DV cameras.json 来自 COLMAP 重建，不是真实 GT，KF0 也需要优化

**验证**: `grep 'uid == 0' slam_backend.py` → 无结果（已删除）

### 2. Runtime Slot Selector Quartile 模式 ✓

**文件**: `runtime_slot_selector.py`
**修改**: 添加 `_pick_candidate_at_ratio()` 和 `_get_placement_ratios()`

**审查结果**: **正确**

**新增模式**:
- `midpoint_only`: 1 pseudo (legacy)
- `quartile`: 3 pseudos at 25%, 50%, 75%
- `quintile`: 4 pseudos at 20%, 40%, 60%, 80%
- `uniform`: 基于 max_pseudo_per_gap

**代码质量**:
- 函数命名清晰
- 参数验证完整
- 返回类型正确 (`List[RuntimePseudoSlot]`)

### 3. Runtime Pseudo Builder Phase 2 Fix ✓

**文件**: `runtime_pseudo_builder.py`
**修改**: 添加 `left_ref_frame_id` 和 `right_ref_frame_id` 传递

**审查结果**: **正确**

```python
# Phase 2 fix: pass reference frame IDs
left_ref_frame_id=int(slot.left_ref_frame_id),
right_ref_frame_id=int(slot.right_ref_frame_id),
```

---

## Online Mapping 链条完整性

### 链条检查

```
Frontend                              Backend
────────────────────────────────────────────────────
runtime_camera_state_cache
    ↓
request_keyframe() ────────────► backend_queue
    ↓                              ↓
runtime_state_payload          brpo_runtime_camera_states
                                   ↓
                               select_runtime_pseudo_slots()
                                   ↓
                               slots (quartile/midpoint)
                                   ↓
                               build_runtime_exact_backend_bundle()
                                   ↓
                               MASt3R matching + projected depth
                                   ↓
                               build_runtime_exact_signal_bundle()
                                   ↓
                               build_runtime_pseudo_record_bundle()
                                   ↓
                               brpo_runtime_pseudo_records
                                   ↓
                               run_brpo_pseudo_mapping()
                                   ↓
                               apply_pose_delta_before_render_()
                                   ↓
                               Render + Loss + Backward
                                   ↓
                               Gauss-Newton (可选)
                                   ↓
                               gaussians update
                                   ↓
                               push_to_frontend()
```

### 链条状态

| 环节 | 状态 | 问题 |
|------|------|------|
| Frontend 缓存 | ✓ | 无 |
| Backend 接收 payload | ✓ | 无 |
| Slot selection | ✓ | quartile 已添加 |
| MASt3R matching | ✓ | 无 |
| **Difix restoration** | ❌ | **缺失** |
| **RGB fusion** | ❌ | **缺失** |
| Signal bundle | ✓ | 无 |
| Record bundle | ✓ | Phase 2 fix |
| Pose delta 应用 | ✓ | 已修复 |
| GN optimization | ✓ | 已验证 |
| Scale regularization | ✓ | 已验证 |

---

## D5 配置审查

**文件**: `configs/d5_online_mapping_fix.yaml`

**关键配置**:
| 参数 | D5 值 | 审查 |
|------|--------|------|
| placement_mode | quartile | ✓ 正确（3 pseudo） |
| max_pseudo_per_gap | 3 | ✓ 匹配 quartile |
| update_real_pose | true | ✓ KF pose 可更新 |
| use_gauss_newton | true | ✓ 最优配置 |
| lambda_scale | 0.1 | ✓ D4 验证有效 |
| lambda_pseudo | 2.0 | ✓ 解决 22:1 比例 |
| dense3d_conf_quantile | 0.15 | ✓ 更宽松选择 |

**问题**: 配置正确，但缺少 difix restoration 配置项

---

## 结论与建议

### 结论

1. **严重问题**: Online mapping 缺少 Difix restoration 和 RGB fusion
2. **Hermes 修复**: 正确，quartile 模式和 Phase 2 fix 已正确实现
3. **链条**: 除 difix 外，其他环节完整

### 建议

**P0 - 必须修复**: 添加 Difix restoration 到 online mapping

方案:
```python
# 在 runtime_exact_backend.py 中添加
def run_difix_restoration(pseudo_rgb, left_ref_rgb, right_ref_rgb, model):
    left_fixed = run_single_difix(model, pseudo_rgb, left_ref_rgb)
    right_fixed = run_single_difix(model, pseudo_rgb, right_ref_rgb)
    fused_rgb = residual_fusion(left_fixed, right_fixed, left_weight, right_weight)
    return fused_rgb
```

**P1 - 配置补充**: D5 配置添加 difix 相关参数

```yaml
brpo_online_mapping:
  use_difix_restoration: true
  difix_model_name: "nvidia/difix_ref"
  difix_fusion_mode: "residual"
```

**P2 - 验证**: D5 实验暂时不运行，等 difix restoration 集成后再运行

---

## 文件清单

| 文件 | 需要修改 |
|------|---------|
| `runtime_exact_backend.py` | ✓ 添加 difix restoration |
| `runtime_pseudo_builder.py` | ✓ 使用 fused RGB |
| `slam_backend.py` | ✓ 添加 difix model 加载 |
| `D5 config` | ✓ 添加 difix 参数 |
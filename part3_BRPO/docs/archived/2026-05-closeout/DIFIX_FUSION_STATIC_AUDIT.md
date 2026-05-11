# Difix + Fusion 集成静态代码审查报告

> 审查日期: 2026-05-06
> 审查范围: Hermes 修复后的完整数据流链条

---

## 审查结论: Hermes 修复完全正确，无其他 Bug

---

## 1. 数据流链条审查

### 1.1 Render 输出 (起点)

| 节点 | 函数 | 输出范围 | 验证状态 |
|------|------|---------|---------|
| render() | gaussian_renderer | 0-1 float (CHW) | PASS |

**确认**: Gaussian Splatting render 输出标准范围 0-1 float

---

### 1.2 Render → uint8 转换

| 文件 | 代码 | 修复前 | 修复后 | 验证状态 |
|------|------|--------|--------|---------|
| runtime_exact_backend.py:251 | pseudo_rgb_uint8 | astype(np.uint8) (错误) | np.clip(*255, 0, 255).astype(np.uint8) | PASS |

**修复**: 先乘 255 再转 uint8

---

### 1.3 Difix 输入/输出

| 函数 | 输入类型 | 输出类型 | 验证状态 |
|------|---------|---------|---------|
| run_single_difix_pil() | PIL Image (uint8) | PIL Image (uint8) | PASS |
| run_difix_restoration() | np.uint8 array | np.uint8 array | PASS |

**确认**: Difix 使用标准 uint8 图像接口

---

### 1.4 save_rgb_png 调用

| 文件:行号 | 调用 | 输入范围 | 验证状态 |
|----------|------|---------|---------|
| runtime_exact_backend.py:262 | left_fixed.png | uint8/255 (错误修复前) | uint8/255 -> float/255 (正确) |
| runtime_exact_backend.py:263 | right_fixed.png | 同上 | PASS |
| runtime_exact_backend.py:305 | fused_rgb.png | 0-1 float | PASS |
| runtime_exact_backend.py:312 | pseudo_fused_rgb.png | 0-1 float | PASS |

**save_rgb_png 定义** (runtime_debug_export.py:23-29):
- 输入期望: 0-1 float
- 内部处理: clip(0,1) -> *255 -> uint8 -> PNG

---

### 1.5 Fusion 输入/输出

| 函数 | 输入类型 | 输出类型 | 验证状态 |
|------|---------|---------|---------|
| fuse_residual_targets() | uint8 array | uint8 array | PASS |
| normalize_branch_weights() | float32 array | float32 array | PASS |

**fuse_residual_targets 计算**:


---

### 1.6 fused_rgb 范围转换

| 文件:行号 | 代码 | 验证状态 |
|----------|------|---------|
| runtime_exact_backend.py:302 | fused_rgb = fused_rgb_uint8.astype(np.float32) / 255.0 | PASS |
| runtime_exact_backend.py:315 | fused_rgb = pseudo_render_rgb.astype(np.float32) | PASS |

**确认**: fused_rgb 最终是 0-1 float

---

### 1.7 MASt3R Matcher 输入

| 节点 | 输入来源 | PNG 内容 | 验证状态 |
|------|---------|---------|---------|
| matcher.match_pair() | save_rgb_png 输出 | 标准 0-255 uint8 PNG | PASS |

**load_images** (dust3r.utils.image):
- 从 PNG 文件读取
- 自动处理标准 0-255 uint8 图像
- MASt3R 模型内部会处理范围

---

### 1.8 RuntimeExactBackendBundle 返回

| 文件:行号 | 字段 | 范围 | 验证状态 |
|----------|------|------|---------|
| runtime_exact_backend.py:409 | final_pseudo_rgb | 0-1 float | PASS |
| runtime_exact_backend.py:413 | pseudo_render_rgb | 0-1 float | PASS |

---

### 1.9 BackendPseudoViewRecord 创建

| 文件:行号 | 字段 | 来源 | 范围 | 验证状态 |
|----------|------|------|------|---------|
| runtime_pseudo_builder.py:60 | target_rgb | pseudo_render_rgb | 0-1 float | PASS |

---

### 1.10 write_runtime_pseudo_record_frame

| 函数调用 | 输入 | 验证状态 |
|----------|------|---------|
| save_rgb_png(target_rgb_runtime.png, target_rgb) | 0-1 float | PASS |

---

### 1.11 Loss 计算

| 函数 | render_rgb | target_rgb | 验证状态 |
|------|-----------|-----------|---------|
| masked_rgb_loss() | 0-1 float (CHW) | 0-1 float (CHW) | PASS |

**Loss 计算**:


**关键**: render_rgb 和 target_rgb 范围必须一致，Hermes 修复确保了这一点

---

## 2. 其他潜在问题审查

### 2.1 Depth 范围

| 节点 | 类型 | 验证状态 |
|------|------|---------|
| pseudo_render_depth | float32 | PASS |
| left_ref_depth, right_ref_depth | float32 | PASS |
| projected_depth | float32 | PASS |

**确认**: Depth 不涉及 uint8 转换，始终 float32

---

### 2.2 Confidence Mask 范围

| 节点 | 范围 | 验证状态 |
|------|------|---------|
| overlap_confidence | 0-1 float | PASS |
| fusion_weight_left/right | 0-1 float | PASS |
| support_mask | 0-1 float | PASS |

**确认**: Confidence mask 始终 0-1 float

---

### 2.3 PIL Image 操作

| 函数 | 输入/输出 | 验证状态 |
|------|---------|---------|
| Image.fromarray() | 0-255 uint8 | PASS |
| PIL -> np.array | 0-255 uint8 | PASS |

---

## 3. 总结

### Hermes 修复的关键点

1. **uint8 转换**: 先乘 255 再转 uint8，避免 0/1 二值化
2. **save_rgb_png 输入**: uint8 结果先除 255 转回 0-1 float
3. **fused_rgb 范围**: 保持 0-1 float 用于 downstream loss
4. **else 分支**: pseudo_render_rgb 保持 0-1 float，不转 uint8

### 无其他 Bug

整个数据流链条审查完成：
- Render 输出 → uint8 转换 → Difix → Fusion → save_png → Matcher → Loss
- 所有范围转换正确
- 无遗漏的 Bug

---

## 4. 建议

1. **运行 D5 实验**: 验证 difix 效果
2. **对比 D4**: 定量分析改善程度
3. **记录结果**: 更新实验报告

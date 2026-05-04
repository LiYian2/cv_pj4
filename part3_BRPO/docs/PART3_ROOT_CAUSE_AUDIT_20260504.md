# Part3 BRPO 根本问题深度审计报告

> 审计时间：2026-05-04 23:30 (Asia/Shanghai)
> 审计者：Claude (based on Charles's request)

---

## 1. 审计背景

用户报告 Part3 的一切尝试都没有效果：
- Standalone refinement 无法提升 replay
- Online mapping integration 反而造成 scene degradation
- 各种实验轴（iteration、mask mode、contract、multi-view）都无效

---

## 2. 关键发现：Depth Target 的根本差异

### 2.1 三个系统的对比

| 系统 | Depth Supervision 来源 | 是否是 External Signal |
|------|----------------------|----------------------|
| **S3PO Real Mapping** | `viewpoint.mono_depth` = 数据集预计算的 monocular depth estimation | ✅ 是（来自独立网络） |
| **BRPO 论文** | Pseudo frame 的真实 sensor depth 或 multi-view reconstruction depth | ✅ 是（来自 sensor/reconstruction） |
| **Part3 Pseudo** | `target_depth` = blend(render_depth, projected_depth_left, projected_depth_right) | ❌ **不是 - Circular!** |

### 2.2 Part3 Depth Target 的来源

```python
# prepare_stage1_difix_dataset_s3po_internal.py line 471-473
render_depth = np.load(render_depth_path).astype(np.float32)  # ← Gaussians render
left_ref_depth = render_depth_from_state(gaussians, left_ref_state, pipe, background)  # ← Gaussians render
right_ref_depth = render_depth_from_state(gaussians, right_ref_state, pipe, background)  # ← Gaussians render

# brpo_reprojection_verify.py
def render_depth_from_state(gaussians, state, pipe, background):
    render_pkg = render(viewpoint, gaussians, pipe, background)  # ← Gaussians render!
    return render_pkg["depth"]
```

**关键：left_ref_depth, right_ref_depth 都是从 Gaussians render 派生的，不是真实 depth observation！**

---

## 3. Circular Loop 分析

### 3.1 Part3 Pipeline 的 Circular 结构

```
S3PO Gaussians (经过 real-only optimization)
    ↓ render
render_depth (Gaussians 输出)
    ↓ 投影到 pseudo viewpoint
projected_depth_left / projected_depth_right (仍是 Gaussians 输出)
    ↓ blend
target_depth = blend(projected_left, projected_right, render_fallback)
    ↓ 用作 supervision
optimize Gaussians ← 用 Gaussians 自己的输出来监督自己！
```

### 3.2 depth_supervision_v2.py 的逻辑

```python
# line 70-80
both_w_sum = fusion_weight_left + fusion_weight_right
target[valid_both_weight] = (
    both_left[valid_both_weight] * projected_depth_left[valid_both_weight]
    + both_right[valid_both_weight] * projected_depth_right[valid_both_weight]
)
target[left_only] = projected_depth_left[left_only]
target[right_only] = projected_depth_right[right_only]

# fallback_mode == 'render_depth' 时
target[fallback] = render_depth[fallback]  # ← 又是 Gaussians render!
```

**这是一个自循环：用 Gaussians 输出的 depth 来监督 Gaussians 本身。**

---

## 4. S3PO Real Mapping 的 Depth 来源

### 4.1 S3PO 代码

```python
# slam_utils.py line 106-107
gt_depth = torch.from_numpy(viewpoint.mono_depth).to(...)
l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
```

### 4.2 mono_depth 的加载

```python
# dataset.py line 102-103 (DL3DVParser)
self.depth_paths = sorted(glob.glob(f"{self.input_folder}/depth/*.png"))
self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/mono_depth/*.png"))
```

**S3PO 的 mono_depth 来自数据集预计算文件，是 DepthAnything/MiDaS 等独立网络的输出，不依赖当前 Gaussians state。**

---

## 5. BRPO 论文的 Depth 来源

根据 BRPO 论文 section 3.1 和 3.3：

1. Pseudo frame 通过 diffusion model 生成 RGB
2. Depth 来自 **真实 sensor** 或 **multi-view stereo reconstruction**
3. 关键：BRPO 的 pseudo depth **不依赖当前 Gaussians state**

BRPO 论文公式 (18-20)：
```
L = β L_rgb + (1-β) L_d + λ_s L_s

L_rgb = ||C_m ⊙ (I_t - Î_t)||_1
L_d = ||C_m ⊙ (D_t - D̂_t)||_1
```

这里的 D_t 是 pseudo view 的真实 depth observation。

---

## 6. 根本问题总结

### 6.1 Circular Supervision

Part3 的 pseudo depth target 是从 Gaussians render 派生的 estimation，不是 external observation。

**这不是 supervision，而是 self-consistency！**

### 6.2 具体问题列表

| 问题 | 详情 | 严重程度 |
|------|------|---------|
| **Circular Supervision** | target_depth = Gaussians render → 用自己监督自己 | 🔴 Critical |
| **No External Signal** | pseudo depth 没有独立来源 | 🔴 Critical |
| **Densify/Prune 禁用** | Standalone: disable_densify=True, disable_prune=True | 🟠 Major |
| **Iteration 不足** | Online: pseudo_map_iters=2 vs real mapping 300 | 🟡 Moderate |

---

## 7. 为什么一切努力都无效？

| 尝试方向 | 为什么无效 |
|---------|-----------|
| **增加 iteration** | 优化目标本身是 circular，iteration 再多也没用 |
| **放宽 mask mode** | coverage 变大但 supervision 还是 circular |
| **换 contract (exact/paper)** | contract 怎么换，depth 来源还是 circular |
| **启用 SPGM** | SPGM 只控制 optimization，不改变 supervision 来源 |
| **切到 online mapping** | 只是改变了 timing，supervision 还是 circular |
| **增加 pseudo views** | 更多 circular supervision 还是 circular |

---

## 8. 修复方向

### 8.1 方向 A：提供真正的 pseudo depth observation

需要检查数据集是否提供：
- DL3DV 是否有 pseudo viewpoint 的 ground truth depth？
- Re10k 是否有 pseudo depth？

如果有，直接用 ground truth depth 作为 target_depth。

### 8.2 方向 B：放弃 depth supervision

只做 RGB-only pose optimization：
- 不用 depth loss (`stageA_disable_depth=True`)
- 只用 RGB loss + confidence mask
- 让 pseudo 只优化 pose，不优化 Gaussians

### 8.3 方向 C：用独立的 depth estimation

对 pseudo viewpoint 用 DepthAnything/MiDaS 生成独立的 depth estimation。
虽然不完美，但至少是 external signal。

### 8.4 方向 D：检查数据集实际 depth 文件

检查 DL3DV 数据集的 `mono_depth/` 目录是否包含 pseudo frame 的 depth。

---

## 9. 立即可做的验证实验

### 9.1 验证 Circular Supervision 假设

| Arm | depth source | 预期结果 |
|------|-------------|---------|
| **control** | target_depth = blend(render, projected) | 无效果（当前 baseline） |
| **RGB-only** | 完全不用 depth loss | 可能有小正向效果（消除 circular） |
| **GT depth (如果有)** | 使用数据集的 pseudo depth GT | 应该有明显正向效果 |

### 9.2 实验代码修改

```bash
# RGB-only 实验
python scripts/run_pseudo_refinement_v2.py \
  --stageA_disable_depth \
  ... (其他参数保持不变)
```

---

## 10. 数据集检查建议

需要检查以下路径：

```bash
# DL3DV
ls /home/bzhang512/CV_Project/dataset/DL3DV-2/mono_depth/
ls /home/bzhang512/CV_Project/dataset/DL3DV-2/results/mono*.png

# Re10k
ls /home/bzhang512/CV_Project/dataset/Re10k-1/*/mono_depth/
```

---

## 11. 一句话结论

> **Part3 失败的根本原因是：pseudo depth target 是从 Gaussians render 派生的 circular estimation，不是真正的 external observation。优化目标形成了 Gaussians → render → project → blend → supervise → Gaussians 的自循环，没有引入新信息。BRPO 论文和 S3PO real mapping 都用 external depth signal（sensor/独立网络 estimation），这是 Part3 与它们的本质差异，也是一切尝试都无效的根本原因。**

---

## 12. 相关文档

- BRPO 论文方法提取：`docs/paper/BRPO_METHOD_extracted_20260424.md`
- Hermes Phase3 Compare Plan：`docs/S3PO_ONLINE_MAPPING_PHASE3_COMPARE_AND_PHASE4_PLAN_20260504.md`
- Charles DL3DV 实验规划：`/data2/bzhang512/CV_Project/output/part3_BRPO/tmp_docs/S3PO_DL3DV2_BRPO_EXPERIMENT_PLAN_20260504.md`
- 设计文档：`docs/current/DESIGN.md`
- S3PO backend：`third_party/S3PO-GS/utils/slam_backend.py`
- Pseudo depth builder：`pseudo_branch/target/depth_supervision_v2.py`
- Render depth function：`pseudo_branch/observation/brpo_reprojection_verify.py`
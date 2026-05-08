# C-Series Online Mapping 实验结果

> 更新时间：2026-05-05 05:00 (Asia/Shanghai)

> **书写规范**：
> 1. 只记录现在，不记录历史过程
> 2. 覆盖式更新，不追加
> 3. 数值在表格中清晰呈现
> 4. 状态用 ✅ ⚠️ ❌ 标记
> 5. 更新后修改文档顶部时间戳

---

## 1. 实验设计

**目标**：在 S3PO backend keyframe path 内集成 BRPO pseudo view enhancement，验证 online mapping 效果。

**数据集**：DL3DV-2 full sequence (306 frames)

**配置**：
- 
- 
- 
- 
- 
- （Paper 路线，RGB/depth 分离 confidence）
- 
- 

---

## 2. 实验结果

| 实验 | pseudo_map_iters | ATE (m) | PSNR | SSIM | ATE 改善 | PSNR 改善 |
|------|-----------------|---------|------|------|---------|----------|
| C1 (noop) | 0 | 0.0629 | 19.44 | 0.680 | baseline | baseline |
| C2 (active-10) | 10 | 0.0618 | 19.09 | 0.666 | -1.7% | -0.35 |
| C3 (active-20) | 20 | 0.0574 | 19.36 | 0.671 | **-8.9%** | -0.08 |

---

## 3. 关键发现

### 3.1 ATE 有轻微改善，PSNR 无改善

- C3 (pseudo_map_iters=20) 的 ATE 相比 C1 改善 **8.9%**
- PSNR 没有改善，甚至略有下降
- 这与 BRPO 论文声称的 20→24 PSNR 增益形成鲜明对比

### 3.2 Coverage 问题已解决

B-series coverage ablation 已确认：
-  +  mask mode → **27-34% coverage**
- 这已经是最优配置，coverage 不是主要瓶颈

### 3.3 Pose Gradient 问题

**根本原因诊断**：

| 组件 | 使用的 Pose | 包含 Pose Delta？ |
|------|------------|------------------|
| **forward.cu (render)** |  (从 R/T 计算) | ❌ 不包含 |
| **backward.cu** | 从  解析 SE3 | ✅ 计算 theta/rho gradient |

**问题**：
1. Forward pass 不使用 theta/rho
2. theta/rho 改变不影响 render 结果
3. Backward pass 计算的 theta/rho gradient 是理论上的，但 forward 不使用

**证据**：
- : 计算  (theta/rho gradient)
- : 没有 theta/rho 相关代码
-  property:  — 不包含 cam_rot_delta/cam_trans_delta

### 3.4 Gauss-Newton 未集成

-  文件不存在
- 当前使用 Adam optimizer + indirect pose regularization
- Pose gradient 只来自 ，不是从 RGB/depth loss 直接反传

---

## 4. Part2 Pseudo Cache 可用性

**发现**：Part2 有完整的 pseudo cache，但这是 standalone 产物，不是 online mapping 过程中生成的。

**Part2 pseudo cache 包含**：
-  — pseudo 帧 depth target
-  /  — 从相邻帧投影的 depth
-  — 各种 confidence mask

**路径**：


**局限**：
- 这是 after_opt 的产物，不是 before_opt（SLAM 过程中）
- Online mapping 应该实时生成，不能依赖 Part2 cache

---

## 5. 下一步

### 5.1 修复 Pose Gradient（优先级最高）

**方案 A：修改 world_view_transform property**



**方案 B：在 render 前手动更新**



### 5.2 实现 Gauss-Newton Pose Optimization

参考 BRPO 论文 section 3.2：
- Finite difference Jacobian
- 直接优化 pose，不依赖 Adam
- 每 N iterations 做一次 GN update

### 5.3 加入 Exposure Refine 和 Scale Regularization

| 组件 | 现状 | 建议 |
|------|------|------|
| Exposure refine | 已有 ,  | 保留，可增加权重 |
| Scale regularization | 未明确添加 | 添加  防止 Gaussian scale 爆炸 |

---

## 6. 实验产物位置



---

## 7. 一句话结论

> **C-series online mapping 实验完成，ATE 有 8.9% 改善但 PSNR 无改善。根本原因是 S3PO rasterizer 的 forward pass 不使用 theta/rho pose delta，导致 pose gradient 无法从 RGB/depth loss 直接反传。需要修改 world_view_transform 或在 render 前手动应用 pose delta，并考虑实现 Gauss-Newton pose optimization。**

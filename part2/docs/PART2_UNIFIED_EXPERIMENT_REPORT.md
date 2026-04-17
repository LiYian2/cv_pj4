# Part 2 Unified Experiment Report: RegGS vs S3PO-GS on Unposed Sparse Reconstruction

## Abstract

本报告总结了 Part 2 在 Unposed Sparse Reconstruction 任务上的实验对比，涉及两条方法路线：**RegGS**（增量式 3DGS 注册）与 **S3PO-GS**（monocular outdoor RGB-only 3DGS-SLAM）。实验覆盖三个数据集（Re10k-1、DL3DV-2、405841），并涉及两种评测协议（external eval 与 internal eval）。核心结论是：RegGS 在温和几何场景（Re10k）收敛，但在旋转主导/前向推进场景失稳；S3PO-GS 的 internal protocol 指标显著高于 external，说明 tracked camera 与地图的自洽性是关键，但也因此两种 protocol 不能直接横比。

## 1. Scope and Experimental Question

### 1.1 两条方法路线的目标差异

**RegGS**：面向“给定一组无位姿稀疏视图，离线做统一注册和重建”的场景。核心机制是增量式注册：先构建局部子图（mini），再与主图（main）对齐并融合，循环推进。该方法高度依赖每一步 pair 的可配准性，早期偏差可沿链式放大。

**S3PO-GS**：面向 monocular outdoor RGB-only 3DGS-SLAM。核心机制是 tracking-mapping 闭环：一边估计相机位姿，一边维护和更新 3D Gaussian 地图。该方法使用预训练 pointmap 模型（MASt3R）提供几何先验，但始终让系统尺度锚定在当前地图自身上。

### 1.2 实验设计

本实验的目标是：
1. 在相同数据集下对比 RegGS 与 S3PO-GS 的 external protocol 表现；
2. 明确 internal protocol 与 external protocol 的定义差异；
3. 分析 RegGS 在不同数据集上的失败模式；
4. 为 Part3（基于 internal route 的 refine）提供实验基础。

## 2. Protocol Definitions

### 2.1 External Protocol

External protocol 是离线评测协议：
- 输入：一张给定 PLY + test split
- 相机来源：重新构建相机，可选 GT pose 或 sequential infer pose
- 特点：地图与相机不是在同一轮 SLAM 过程中共同形成的

**External infer**：以第一帧 GT 或 identity 初始化，逐帧顺序估相对位姿。
**External gt**：直接使用数据集 GT pose。

### 2.2 Internal Protocol

Internal protocol 是 S3PO-GS 原生评测协议：
- 输入：SLAM 运行过程中保存的 internal map + tracked camera states
- 相机来源：frontend.cameras 中的当前位姿估计（R/T）
- 特点：地图与相机在同一系统中共同演化，天然处于同一尺度系

**before_opt**：使用 tracking/mapping 后的当前地图，直接做 non-KF 渲染评测。
**after_opt**：先运行 color refinement 进一步优化地图外观，再做同样的 non-KF 渲染评测。

### 2.3 为什么两种 protocol 不能直接横比

关键差异：
- External protocol 的相机是离线重建的，与当前地图的自洽性差；
- Internal protocol 的相机是与地图共同优化的，天然更自洽；
- 当前 external eval 的对齐只用了很弱的 origin_mode，不是完整的 map-to-test 刚体变换。

因此，**internal 分数高于 external 是结构必然，不应理解为“internal 方法更好”，而是“internal 测的是地图-相机自洽性”**。

## 3. RegGS Results and Failure Analysis

### 3.1 指标总览（External Protocol）

| Dataset | PSNR | SSIM | LPIPS | ATE RMSE |
|---------|------|------|-------|----------|
| Re10k-1 | 26.02 | 0.895 | 0.128 | 0.011 |
| DL3DV-2 | 15.79 | 0.443 | 0.482 | 1.87 |
| 405841 | 16.68 | 0.481 | 0.524 | 19.29 |

核心观察：
- Re10k：ATE 低（约 0.01 量级），test 指标明显优于 DL3DV/405841；
- DL3DV：train 指标极高，test 与 ATE 同时明显退化；
- 405841：最高级别漂移（ATE 约 17~18）。

### 3.2 失败模式分析

**Re10k：温和几何，可收敛区**
- 相邻 train 平移约 0.52，旋转约 4.05°
- 几何跨度温和，main 迭代可保持稳定

**DL3DV：rotation-dominant + low-effective-parallax 失稳**
- 相邻 train 平移约 2.71，旋转约 16.61°
- 局部子图可解释图像，但未必能稳定并入全局 main
- 增量式注册链路缺少足够视差锚点

**405841：forward-motion + scale-drift 累积漂移**
- 相邻 train 平移约 5.63，旋转约 1.51°
- 长段前冲 + 侧向视差弱，尺度歧义沿链路积累
- 后期 main 失去稳定参考作用

### 3.3 RegGS 失败的本质原因

同一套 RegGS 管线在三个数据集上表现差异巨大的主因，是 **数据几何与运动模式是否落在增量式 3DGS 注册链的稳定工作区间**，而不是单一 checkpoint、单一参数或单纯训练轮数。

关键结论：
- 失败主要发生在 infer/alignment 层；
- refine 层是放大器，不是首发故障点；
- metric 层存在解释注意项：当前 evaluator 对 test 帧执行 test-time pose 优化，所以 test 分数不是完全固定位姿直渲染。

## 4. S3PO External Eval Results

### 4.1 指标总览

| Dataset | PSNR | SSIM | LPIPS | ATE RMSE |
|---------|------|------|-------|----------|
| Re10k-1 | 13.15 | 0.496 | 0.389 | 0.048 |
| DL3DV-2 | 11.36 | 0.372 | 0.630 | 1.51 |
| 405841 | 16.11 | 0.564 | 0.542 | 2.79 |

核心观察：
- External infer pose 是主要瓶颈；
- sparse/full 的地图差异在该协议下不明显；
- Re10k 的 external 指标明显低于 RegGS external，说明 external infer pose 更弱。

### 4.2 External GT vs External Infer

以 Re10k 为例：
- External infer PSNR：约 13.15
- External gt PSNR：约 16.94
- GT pose 会改善 external eval，但提升仍有限

这说明 external eval 的瓶颈不只是 pose 来源，还包括：
- 相机是离线重建的；
- 当前对齐只用了很弱的 origin_mode。

## 5. S3PO Internal Eval Results

### 5.1 指标总览

| Dataset | Mode | Protocol | PSNR | SSIM | LPIPS | ATE RMSE |
|---------|------|----------|------|------|-------|----------|
| Re10k-1 | full | internal (before) | 16.89 | 0.700 | 0.281 | 0.007 |
| Re10k-1 | full | internal_after | 23.95 | 0.873 | 0.079 | 0.007 |
| DL3DV-2 | full | internal (before) | 16.59 | 0.569 | 0.434 | 0.463 |
| DL3DV-2 | full | internal_after | 17.48 | 0.615 | 0.354 | 0.463 |
| 405841 | full | internal (before) | 21.00 | 0.737 | 0.374 | 0.613 |
| 405841 | full | internal_after | 24.02 | 0.766 | 0.280 | 0.613 |

### 5.2 Internal vs External

以 Re10k-1 为例：
- External infer PSNR：约 13.15
- External gt PSNR：约 16.94
- Internal before_opt PSNR：约 16.89
- Internal after_opt PSNR：约 23.95

核心观察：
- Internal before_opt 已接近 External gt 水平；
- Internal after_opt 显著高于所有 External 结果；
- Color refinement 在 internal protocol 下大幅提升地图外观质量。

### 5.3 为什么 Internal ATE 更低

Internal protocol 的 ATE 只看 **KF**（关键帧），而不是全部 non-KF。
- Re10k-1 full run 的 ATE 只包含 9 个 KF：0, 34, 69, 104, 139, 173, 208, 243, 278；
- 这与 RegGS 的 ATE 计算方式不同（RegGS 看 test split）。

因此，**Internal ATE 与 External ATE 不能直接横比**。

## 6. Cross-Protocol Interpretation

### 6.1 为什么 Internal 指标高于 External

核心原因是 **internal tracked camera state 与地图更自洽**：
1. 相机位姿是在 SLAM 过程中与地图共同优化的；
2. 地图坐标系与相机坐标系天然处于同一尺度系；
3. 渲染评测用的是 tracked non-KF camera，而不是离线重建的 test camera。

这解释了为什么：
- Internal before_opt ≈ External gt（都用了 GT 或 tracked pose）；
- Internal after_opt >> Internal before_opt（color refinement 改善外观）。

### 6.2 External vs RegGS

在 External protocol 下：
- RegGS 在 Re10k 明显优于 S3PO external；
- RegGS 在 DL3DV/405841 的 ATE 更高，但 PSNR 相近。

这说明：
- RegGS 的增量注册在温和场景更稳；
- External infer pose 是 S3PO external 的主要瓶颈。

### 6.3 正确的比较方法

1. **RegGS vs S3PO external**：可在 External protocol 下比较，因为两者都用了离线 pose；
2. **S3PO internal vs external**：不能直接比较，因为 protocol 不同；
3. **S3PO internal vs Part3 refine**：可以比较，因为 Part3 基于 internal route。

## 7. Limitations and Next Steps

### 7.1 当前限制

1. RegGS 的 failure analysis 是定性诊断，尚未做系统参数网格验证；
2. S3PO internal eval 的 sparse 模式缺少 PSNR 数据（部分 run 未保存）；
3. External eval 的对齐机制较弱，不是完整的 map-to-test 刚体变换。

### 7.2 下一步方向

1. **Part3**：基于 internal route 做 standalone refine，验证 pseudo supervision 是否能改善 internal 指标；
2. **协议对齐**：如需横比 internal vs external，需补充 internal replay eval（方案 A）；
3. **RegGS 稳定化**：如需继续推进 RegGS，优先调整 aligner 参数（color/depth/mw2 权重）。

## 8. Reproducibility Assets

核心数据文件：
- RegGS external：
- S3PO internal：
- 合并表：

图表：
- 
- 
- 
- 

脚本：
- 
- 
- 

## 9. Summary

> **同一套 RegGS 管线在三个数据集上表现差异巨大的主因，是数据几何与运动模式是否落在增量式 3DGS 注册链的稳定工作区间。S3PO-GS 的 internal protocol 指标显著高于 external，说明 tracked camera 与地图的自洽性是关键，但也因此两种 protocol 不能直接横比。Part3 的 refine 实验应基于 internal route，而非 external。**

---

_本报告基于 2026-04-16 的实验数据整理生成。_

# Part2 RegGS Failure Analysis

本文档为阶段 1/2/3 与本轮 failure 调研的合并版，目标是保留完整证据链并减少重复叙述。主线是：先确认失败现象，再定位失败发生层，再给出可执行验证路径。文档范围覆盖 Re10k、DL3DV-2、405841（scene_A / scene_C）相关本地结果、论文对照与代码机制。

## 1. 总结结论（先看这个）

当前问题的主因不是“refine 迭代不够”，而是 RegGS 的增量注册链路（NoPo 局部高斯 -> 局部/全局配准 -> 融合）对运动几何非常敏感。在 Re10k 上，该链路多数时间位于稳定工作区；在 DL3DV 与 405841 上分别触发不同失败模式，导致一致的“train 高、test 低、ATE 大”。

可压缩为三句话：

1. **Re10k 是当前管线的相对舒适区**：运动温和、有效视差更稳定，累计主图更易维持正确几何。
2. **DL3DV 主要是 rotation-dominant + low-effective-parallax 失稳**：局部子图可解释图像，但未必能稳定并入全局 main。
3. **405841（尤其 scene_C）主要是 forward-motion + scale-drift 累积漂移**：局部偏差沿增量融合链放大，后期 main 成为“错误锚点”。

## 2. 证据总览：从结果到机制

### 2.1 指标层证据（跨阶段一致）

多个 run 在不同阶段得到同一模式：

- Re10k（示例：`sr30,nv8` 或 `sr50,nv9`）ATE 始终低（约 `0.01` 量级），test 指标明显优于 DL3DV/405841。
- DL3DV（`sr30,nv8`、`sr30,nv10`、`sr50,nv10`）普遍出现：train 指标极高，test 与 ATE 同时明显退化（ATE 常在 `2+`）。
- 405841（scene_A 到 scene_C）可见从“中等失稳”到“几何级失稳”的梯度；scene_C 在当前设置下出现最高级别漂移（ATE 约 `17~18`）。

这说明退化不是单一渲染质量问题，而是几何/位姿链路先失稳，随后 refine 在训练帧上放大记忆化。

### 2.2 轨迹与可视化证据

`vis_align`、`vis_integrate`、`eval_traj.png` 的共同信号：

- Re10k：main 结构连续，轨迹紧凑。
- DL3DV：局部子图常出现“各自解释画面但不共址”，integrate 后更易分叉、错位粘连。
- 405841：早期可能勉强对上，但沿前向推进逐步累积漂移，后期 main 失去稳定参考作用。

该现象与指标层完全同向，支持“失败发生在 infer/alignment 主链路”。

## 3. 失败发生在 pipeline 哪一层

阶段 3 的核心目标是把“可能原因”推进到“可归因层级”。合并结论如下：

1. **主要失败层：infer/alignment**。
2. **refine 层是放大器，不是首发故障点**；位姿底座错时，refine 可继续提高 train 观感，但难救 test 与轨迹。
3. **metric 层存在解释注意项**：当前 evaluator 对 test 帧执行 test-time pose 优化（每帧 400 步），所以 test 分数不是“完全固定位姿直渲染”的口径，读数需结合 ATE 与可视化共同判断。

## 4. 代码机制与论文对照（为何会这样）

### 4.1 本地实现机制

本地 RegGS 的核心行为是增量式注册：

1. `sample_rate` 先取 test 子采样帧。
2. 在剩余帧中按 `n_views` 等距取 train。
3. 对相邻 train pair 生成 mini，再与 main 对齐并融合，循环推进。

这决定了系统高度依赖“每一步 pair 可配准性”。一旦早期 pair 偏离，误差可被链式放大。

### 4.2 对齐损失与稳定锚点

当前 aligner 关键点：

- 深度项只在 `main_depth_i > 0.5 && mini_depth_i > 0.5` 区域生效。
- 尺度初始化来自主/子图深度中位数比，并受 `scale_min/scale_max/depth_valid_min` 约束。
- 三项损失当前固定权重：`color=1.0, depth=0.1, mw2=0.1`。

因此当有效深度区域崩塌、尺度初始化不稳、或 pair 重叠不足时，优化会失去几何锚点并更易滑向错误 basin。

### 4.3 论文一致性与边界

论文明确支持以下判断：

1. RegGS 依赖 `MW2 + Photo + Depth` 联合优化，单项不足以稳定全局。
2. 大帧间运动会导致注册不收敛。
3. 论文评测口径是“test=所有非训练帧”，而本地实现默认是 sample_rate 子采样 test。

换言之，本地失败现象与论文已承认边界一致；但论文数值与本地数值不能直接横比。

## 5. 不同数据集的失败模式（统一解释）

### 5.1 Re10k：可收敛区

典型统计（`sr50,nv9`）：相邻 train 平移约 `0.5188`，旋转约 `4.05°`。几何跨度相对温和，main 迭代可保持稳定。

### 5.2 DL3DV：旋转主导 + 有效视差不足

典型统计（`sr50,nv10`）：平移约 `2.7125`，旋转约 `16.61°`。此外早期诊断在 `sr30,nv8` 下也显示相邻 train 基线显著偏大：

- train ids: `[0, 43, 87, 130, 174, 217, 261, 305]`
- 相邻 train index 间隔均值：`43.57`
- 相邻 train 平移均值：`3.3473`

与同序列更密采样（16/8 风格）相比平移间隔约 `1.99x`，与 Re10k 的同 8/30 比约 `5.64x`。这解释了为什么在 DL3DV 上“训练可拟合，但注册难稳定”。

### 5.3 405841：前向推进引发尺度/深度耦合累积

典型统计（`sr30,nv12`）：平移约 `5.6266`，旋转约 `1.51°`。它不是“转太多”，而是“长段前冲 + 侧向视差弱”，导致尺度歧义更易沿链路积累，后期 main 漂移反过来拖坏新 pair。

## 6. 关于 `vis_align` 大面积蓝色的统一解释

本地 `vis_align` 五联图为：`GT Color / Main Color / Main Depth / Mini Color / Mini Depth`，且保存于 align 最后一轮。深度图使用 `jet`（`vmin=0,vmax=6`）时，深蓝在这里主要表示接近零深度渲染，即“当前视角下几乎无有效深度贡献”，而不是单纯“很远”。

这通常对应三类状态：

1. 位姿/尺度偏差使高斯整体投影失效（出视锥、到相机后方等）。
2. mini 子图本身退化，无法提供可注册几何。
3. main 已累计失稳，当前 pair 在错误锚点上继续恶化。

当蓝区大面积出现时，depth 锚点会显著收缩，颜色项与 MW2 被迫承担更多对齐责任，系统更容易进入错误局部最优。该现象应被视为“几何失锚报警”，不是可视化噪声。

## 7. 参数、checkpoint 与实现细节的角色

合并判断如下：

1. **P0：数据几何与运动模式** 决定是否进入稳定工作区，是首因。
2. **P1：NoPo 先验与域匹配** 影响初始 mini 质量，会放大或缓解失稳。
3. **P2：aligner 参数与实现细节** 决定收敛 margin；重要但不足以单独解释跨数据集差异。

同理，`stable_v2_overfit` 相比 `stable_v1` 的“train 更高但 test/ATE 未改善”再次说明：若底层几何没稳，后段优化不会自动带来泛化改进。

## 8. 已落地能力与当前可控参数

当前两个 run notebook 已接入关键参数接口（`sample_rate / n_views / new_submap_every` 与 aligner 稳定项等），可用于快速构造对照实验。对 DL3DV/405841 的可复现起始 profile 建议保持：

- `iterations=300`
- `cam_rot_lr=1e-4`
- `cam_trans_lr=1e-3`
- `filter_alpha=true`
- `alpha_thre=0.95`
- `mask_invalid_depth=true`
- `scale_min=0.1`
- `scale_max=10.0`
- `depth_valid_min=0.1`
- `scale_init_fallback=1.0`
- `mw2_warmup_iters=60`

该 profile 目标是“先稳几何”，不是直接追求最高画质。

## 9. 最小执行路径（合并版）

为避免在错误位姿底座上空耗 refine，建议统一执行顺序：

1. **先 infer+metric 小矩阵筛 ATE**，再进入 refine。
2. 在同数据内先比较 `n_views`，再比较稳定 profile，再比较 checkpoint。
3. 对 ATE 无改善组停止加 refine，回到采样与配准稳定性。

推荐最小矩阵可收敛为 4 组（先 DL3DV 验证）：

1. 基线：`re10k-ckpt, nv=8, sr=30, sm=2`
2. 域对照：`dl3dv-ckpt, nv=8, sr=30, sm=2`
3. 稳定性提升：`dl3dv-ckpt, nv=12, sr=30, sm=2 + stable profile`
4. 采样提升：`dl3dv-ckpt, nv=8, sr=20, sm=2 + stable profile`

准入建议：若某组 ATE 相对基线下降达到显著幅度（例如 30% 量级）并且 `vis_align`/`vis_integrate` 观感同步改善，再投入 refine。

## 10. 后续代码改造优先级（仅保留必要项）

在不扩大改动面的前提下，优先做两项：

1. 将 aligner 的 `color/depth/mw2` 权重暴露到 yaml。
2. 增加评测协议开关（`sampled_test` vs `all_non_train`）用于论文口径对齐。

这两项直接决定后续消融结论是否可比、可复现。

## 11. 最终判断

跨阶段证据已经足够支持以下统一结论：

> 同一套 RegGS 管线在 Re10k、DL3DV、405841 上表现差异巨大的主因，是数据几何与运动模式是否落在“增量式 3DGS 注册链”的稳定工作区间，而不是单一 checkpoint、单一参数或单纯训练轮数。

因此后续工作的主战场应是 infer/alignment 与采样协议，而非先加 refine。只有先把几何锚点稳住，train/test/ATE 才有机会同向改善。

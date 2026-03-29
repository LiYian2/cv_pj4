# Part2 RegGS 本地诊断（阶段 1）

> 范围说明
>
> 本文只基于本地代码与本地运行结果做诊断，不展开 GitHub 远端深挖（例如 NoPo 新权重来源、社区 issue 里的特定 trick）。
> 下一阶段再补充远端调研。

## 1. 结论先行（TL;DR）

当前 DL3DV 结果“明显模糊 + 大面积黑区 + 几何拖影”主要不是 refine 不够，而是 infer/alignment 阶段已发生较大位姿误差累积。最关键证据是：

1. ATE 在 DL3DV 上很高，且和同参数 Re10k 对比差异巨大。
2. 你当前 8/30 采样在这个 DL3DV 序列上的相邻训练视角位移跨度远超 Re10k，超出了当前默认对齐策略的稳定域。
3. 你使用的是 re10k NoPo 权重跑 DL3DV（且本地无 dl3dv 混合权重），存在明显域偏差风险。

因此调参优先级应是：

1. 先降低位姿估计难度（采样策略与子图策略）。
2. 再增强对齐稳定性（aligner 迭代与学习率/掩码）。
3. 最后再做 refine 细节提升。

## 2. 本地证据

### 2.1 配置对比：你当前设置 vs 官方默认

- 你当前（DL3DV）: `n_views=8, sample_rate=30, new_submap_every=2`
  - 见 [part2/configs/reggs_dl3dv2_sr30_nv8.yaml](part2/configs/reggs_dl3dv2_sr30_nv8.yaml#L6)
  - 见 [part2/configs/reggs_dl3dv2_sr30_nv8.yaml](part2/configs/reggs_dl3dv2_sr30_nv8.yaml#L7)
  - 见 [part2/configs/reggs_dl3dv2_sr30_nv8.yaml](part2/configs/reggs_dl3dv2_sr30_nv8.yaml#L9)
- 官方默认（re10k sample）: `n_views=16, sample_rate=8, new_submap_every=3`
  - 见 [third_party/RegGS/config/re10k.yaml](third_party/RegGS/config/re10k.yaml#L6)
  - 见 [third_party/RegGS/config/re10k.yaml](third_party/RegGS/config/re10k.yaml#L7)
  - 见 [third_party/RegGS/config/re10k.yaml](third_party/RegGS/config/re10k.yaml#L9)

你的设置稀疏度明显更高。

### 2.2 质量与轨迹误差（本地结果）

- DL3DV（当前 run）
  - Test 指标：PSNR 15.69 / SSIM 0.447 / LPIPS 0.466
  - 见 [output/part2/dl3dv_2/reggs_dl3dv2_sr30_nv8/eval_test.json](output/part2/dl3dv_2/reggs_dl3dv2_sr30_nv8/eval_test.json#L2)
  - 见 [output/part2/dl3dv_2/reggs_dl3dv2_sr30_nv8/eval_test.json](output/part2/dl3dv_2/reggs_dl3dv2_sr30_nv8/eval_test.json#L3)
  - 见 [output/part2/dl3dv_2/reggs_dl3dv2_sr30_nv8/eval_test.json](output/part2/dl3dv_2/reggs_dl3dv2_sr30_nv8/eval_test.json#L4)
  - ATE-RMSE 约 2.326
  - 见 [output/part2/dl3dv_2/reggs_dl3dv2_sr30_nv8/ate_aligned.json](output/part2/dl3dv_2/reggs_dl3dv2_sr30_nv8/ate_aligned.json#L1)

- Re10k（同样 8/30）
  - Test 指标：PSNR 25.00 / SSIM 0.879 / LPIPS 0.144
  - 见 [output/part2/re10k/reggs_re10k1_sr30_nv8/eval_test.json](output/part2/re10k/reggs_re10k1_sr30_nv8/eval_test.json#L2)
  - 见 [output/part2/re10k/reggs_re10k1_sr30_nv8/eval_test.json](output/part2/re10k/reggs_re10k1_sr30_nv8/eval_test.json#L3)
  - 见 [output/part2/re10k/reggs_re10k1_sr30_nv8/eval_test.json](output/part2/re10k/reggs_re10k1_sr30_nv8/eval_test.json#L4)
  - ATE-RMSE 约 0.0106
  - 见 [output/part2/re10k/reggs_re10k1_sr30_nv8/ate_aligned.json](output/part2/re10k/reggs_re10k1_sr30_nv8/ate_aligned.json#L1)

同等稀疏配置下，DL3DV 的位姿误差大幅高于 Re10k，说明问题不是“参数都错”，而是“当前配置在该序列上超出稳定范围 + 域偏差”。

### 2.3 采样跨度定量诊断（本地统计）

对当前 DL3DV 序列（306 帧）按你当前策略分帧：

- train ids: `[0, 43, 87, 130, 174, 217, 261, 305]`
- test ids: `[15, 45, 75, 105, 135, 165, 195, 225, 255, 285]`
- 相邻 train 帧 index 间隔均值：`43.57`
- 相邻 train 相机中心平移间隔均值：`3.3473`

在同一序列里，用“官方风格”16/8 采样时：

- 相邻 train 平移间隔均值：`1.6780`
- 当前/官方风格间隔比约：`1.99x`

再对比 Re10k 序列同样 8/30：

- DL3DV 相邻 train 平移均值：`3.3473`
- Re10k 相邻 train 平移均值：`0.5930`
- 比值：`5.64x`

这说明同样是 8/30，DL3DV 这一条序列的真实几何跨度大得多，pose-free 配准更容易失稳。

### 2.4 代码机制证据（关键路径）

1. 训练/测试帧划分就是按 `sample_rate` 与 `n_views` 固定抽样。
   - 见 [third_party/RegGS/src/entities/reggs.py](third_party/RegGS/src/entities/reggs.py#L41)
   - 见 [third_party/RegGS/src/entities/reggs.py](third_party/RegGS/src/entities/reggs.py#L49)

2. 子图切分由 `new_submap_every` 控制。
   - 见 [third_party/RegGS/src/entities/reggs.py](third_party/RegGS/src/entities/reggs.py#L64)
   - 见 [third_party/RegGS/src/entities/reggs.py](third_party/RegGS/src/entities/reggs.py#L66)

3. 对齐损失固定为 `color + 0.1*depth + 0.1*mw2`，并未暴露到 yaml。
   - 见 [third_party/RegGS/src/entities/gaussian_aligner.py](third_party/RegGS/src/entities/gaussian_aligner.py#L392)
   - 见 [third_party/RegGS/src/entities/gaussian_aligner.py](third_party/RegGS/src/entities/gaussian_aligner.py#L394)
   - 见 [third_party/RegGS/src/entities/gaussian_aligner.py](third_party/RegGS/src/entities/gaussian_aligner.py#L428)

4. 对齐阶段优化变量学习率由 `aligner.cam_rot_lr/cam_trans_lr` 控制，`gs_scale` lr 固定 `1e-3`。
   - 见 [third_party/RegGS/src/entities/gaussian_model.py](third_party/RegGS/src/entities/gaussian_model.py#L433)
   - 见 [third_party/RegGS/src/entities/gaussian_model.py](third_party/RegGS/src/entities/gaussian_model.py#L434)
   - 见 [third_party/RegGS/src/entities/gaussian_model.py](third_party/RegGS/src/entities/gaussian_model.py#L436)

5. 对齐迭代次数实际是 `aligner.iterations * 2`。
   - 见 [third_party/RegGS/src/entities/gaussian_aligner.py](third_party/RegGS/src/entities/gaussian_aligner.py#L385)

6. 对齐中颜色损失掩码逻辑受 `filter_alpha / alpha_thre / soft_alpha / mask_invalid_depth` 影响。
   - 见 [third_party/RegGS/src/entities/gaussian_aligner.py](third_party/RegGS/src/entities/gaussian_aligner.py#L110)
   - 见 [third_party/RegGS/src/entities/gaussian_aligner.py](third_party/RegGS/src/entities/gaussian_aligner.py#L119)
   - 见 [third_party/RegGS/src/entities/gaussian_aligner.py](third_party/RegGS/src/entities/gaussian_aligner.py#L122)

7. refine 阶段超参（densify/opacity reset）目前是代码内固定，不读 yaml。
   - 见 [third_party/RegGS/run_refine.py](third_party/RegGS/run_refine.py#L199)
   - 见 [third_party/RegGS/run_refine.py](third_party/RegGS/run_refine.py#L203)

8. evaluator 对每个 test 帧会再做 400 步 pose 优化，因此 test 指标是“已做 test-time pose 修正”的结果。
   - 见 [third_party/RegGS/src/evaluation/evaluator.py](third_party/RegGS/src/evaluation/evaluator.py#L257)
   - 见 [third_party/RegGS/src/evaluation/evaluator.py](third_party/RegGS/src/evaluation/evaluator.py#L311)

9. NoPo checkpoint 逻辑支持 `acid / dl3dv / re10k` 分支，但本地目前只有 `acid.ckpt` 和 `re10k.ckpt`，没有 `mixRe10kDl3dv.ckpt`。
   - 见 [third_party/RegGS/src/utils/nopo_utils.py](third_party/RegGS/src/utils/nopo_utils.py#L44)
   - 见 [third_party/RegGS/src/utils/nopo_utils.py](third_party/RegGS/src/utils/nopo_utils.py#L46)
   - 本地目录: [third_party/RegGS/pretrained_weights](third_party/RegGS/pretrained_weights)

## 3. 根因优先级判断

### P0（最高优先级）

视角跨度过大导致 pose-free 对齐失稳：

- 你现在 8/30 在这条 DL3DV 的邻接训练基线非常大，几何重叠不足，pair-level 对齐和尺度估计更容易陷入错误局部最优。
- 从可视化现象（重影拖拽、大块黑区）和 ATE 高误差看，核心是几何/位姿问题，不是仅仅纹理不足。

### P1

NoPo 权重域偏差：

- DL3DV 数据上使用 re10k checkpoint，且本地没有 dl3dv 混合权重，初始相对位姿与高斯质量可能偏差更大，后续配准负担更重。

### P2

对齐损失和掩码策略在极稀疏场景下不够鲁棒：

- 当前深度/MW2 权重固定，alpha 与无效深度屏蔽默认偏“宽松”，复杂场景下可能引入噪声梯度。

### P3

refine 阶段可能在错误位姿上“越训越糊”：

- refine 只用 train 帧做颜色优化；若位姿底座错了，更多迭代只能把错误几何拟合得更“平滑”。

## 4. 调参建议（按优先顺序）

## 4.1 第一组：先改采样与子图（优先级最高）

目标：降低 pair 对齐难度，先把 ATE 拉下来。

建议尝试顺序：

1. 固定 `sample_rate=30`，把 `n_views` 从 `8 -> 12 -> 16`。
2. 固定 `n_views=8`，把 `sample_rate` 从 `30 -> 20 -> 16`。
3. `new_submap_every` 从 `2` 试到 `3`（必要时 `4`），减少子图切换频率，提升单子图稳定性。

经验标准：先看 ATE，再看清晰度。若 ATE 不降，单纯加 refine 意义不大。

## 4.2 第二组：aligner 稳定性参数

你当前可直接在 yaml 调的：

- `aligner.iterations`: `200 -> 300/400`
- `aligner.cam_rot_lr`: `2e-4 -> 1e-4`
- `aligner.cam_trans_lr`: `2e-3 -> 1e-3`
- `aligner.filter_alpha`: `false -> true`
- `aligner.alpha_thre`: `0.98 -> 0.95`（开启过滤时建议）
- `aligner.mask_invalid_depth`: `false -> true`

已有但你未设置（代码有默认）的稳定性项，可加入 yaml：

- `aligner.scale_min`: 先试 `0.1`
- `aligner.scale_max`: 先试 `10.0`
- `aligner.depth_valid_min`: 先试 `0.1`
- `aligner.scale_init_fallback`: `1.0`（可保持）
- `aligner.mw2_warmup_iters`: `20 -> 50`

这些项来自：
- [third_party/RegGS/src/entities/gaussian_aligner.py](third_party/RegGS/src/entities/gaussian_aligner.py#L47)
- [third_party/RegGS/src/entities/gaussian_aligner.py](third_party/RegGS/src/entities/gaussian_aligner.py#L51)

## 4.3 第三组：NoPo checkpoint

本地当前配置为 `nopo_checkpoint: re10k`：
- 见 [part2/configs/reggs_dl3dv2_sr30_nv8.yaml](part2/configs/reggs_dl3dv2_sr30_nv8.yaml#L8)

若你后续拿到 `mixRe10kDl3dv.ckpt`，可直接切到 `nopo_checkpoint: dl3dv` 做对比实验。

## 4.4 第四组：refine 相关

- refine 迭代默认 30k（硬编码）：
  - 见 [third_party/RegGS/run_refine.py](third_party/RegGS/run_refine.py#L260)
- 训练优化默认参数（来自 3DGS 原始设置）：
  - 见 [third_party/RegGS/src/entities/arguments.py](third_party/RegGS/src/entities/arguments.py#L55)

在位姿未稳定前，不建议优先动 refine。若 ATE 已明显下降再做：

1. 降低 densify aggressiveness（需要改代码，当前不是 yaml 参数）。
2. 缩短 refine 迭代做快速筛选，再对最优配置拉满迭代。

## 5. 推荐实验矩阵（只做本地阶段）

每组建议只跑 infer + metric（先不 refine）筛选，保留前 2 组再跑 refine。

1. E0（当前基线）：`n_views=8, sample_rate=30, new_submap_every=2`
2. E1：`n_views=12, sample_rate=30, new_submap_every=2`
3. E2：`n_views=16, sample_rate=30, new_submap_every=3`
4. E3：`n_views=8, sample_rate=20, new_submap_every=2`
5. E4：在 E2 或 E3 上加稳定项：
   - `iterations=300`
   - `cam_rot_lr=1e-4`
   - `cam_trans_lr=1e-3`
   - `filter_alpha=true`
   - `alpha_thre=0.95`
   - `mask_invalid_depth=true`
   - `mw2_warmup_iters=50`

每组记录：

- ATE-RMSE
- Test PSNR/SSIM/LPIPS
- `vis_align` 是否仍出现大面积漂移
- `vis_integrate` 是否仍有大块黑区

## 6. 下一阶段（下个对话）建议

下一阶段再做远端调研并落地：

1. 核验/补齐 `mixRe10kDl3dv.ckpt` 的官方来源与版本。
2. 对照论文与 repo commit 记录，确认 sparse-view 推荐设置是否有更新。
3. 把 aligner 的 `color/depth/mw2` 权重改成 yaml 可配，做系统消融。
4. 如有必要，再把 refine 的 densify 参数也抽到配置文件。

---

更新日期：2026-03-29

## 7. 阶段 2：论文提取与阅读（本地完成）

本阶段已完成对论文 PDF 的本地提取与阅读：

- 原始 PDF: [part2/docs/Cheng_RegGS_Unposed_Sparse_Views_Gaussian_Splatting_with_3DGS_Registration_ICCV_2025_paper.pdf](part2/docs/Cheng_RegGS_Unposed_Sparse_Views_Gaussian_Splatting_with_3DGS_Registration_ICCV_2025_paper.pdf)
- 提取文本: [part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt](part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt)

### 7.1 对阶段 1 本地研究内容的理解

阶段 1 已经完成了“代码机制 + 运行结果 + 采样几何跨度”的闭环诊断，核心结论是：

1. 当前 DL3DV 的失败主因是 infer/alignment 的位姿误差累积，而不是 refine 不够。
2. 在你当前序列上，`8/30` 采样导致相邻训练视角基线过大，使 pose-free 对齐更易失稳。
3. NoPo 权重域偏差（re10k 权重跑 DL3DV）会放大初始误差，进一步增加配准难度。

这套判断与论文“稀疏大位移场景更易失稳”的限制项是同向的（见 7.2 与 7.3）。

### 7.2 论文中和调参直接相关的要点（含证据）

1. 论文采样协议强调等距采样 + 全量非训练帧测试：
   - 2-view 场景按运动强度用 40/60 帧步长：[part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt](part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt#L638), [part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt](part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt#L639)
   - 8/16/32-view 训练帧全视频等距采样：[part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt](part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt#L640), [part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt](part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt#L641)
   - 测试集为所有非训练帧：[part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt](part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt#L642)

2. 论文明确是三项联合损失，不是单项可替代：
   - 总损失形式：`Ltotal = λ1LMW2 + λ2LPhoto + λ3LDepth` [part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt](part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt#L468)
   - 同时提到“adaptive weight allocation” [part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt](part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt#L430)
   - 消融显示去掉任一项都会显著退化（w/o Photo / w/o Depth / w/o MW2）[part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt](part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt#L763), [part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt](part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt#L764), [part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt](part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt#L765)

3. 论文对深度项的稳定性强调“有效深度掩码”：
   - 深度约束通过有效掩码抑制尺度漂移：[part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt](part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt#L491)
   - `Mv` 为有效深度 mask：[part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt](part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt#L505)

4. 论文对尺度初始化给出明确方向：
   - 先做尺度归一化：[part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt](part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt#L575)
   - 再由主图/子图深度关系初始化相对尺度：[part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt](part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt#L583)

5. 论文承认大位移是失效边界：
   - 大帧间运动会导致注册不收敛：[part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt](part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt#L793)
   - 输入视角数增加会显著增加耗时（MW2 开销）：[part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt](part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt#L791)

### 7.3 论文结论与本地实现的映射（关键差异）

1. 论文是 `λ1/λ2/λ3` 形式且提到自适应权重；本地代码当前是固定权重：
   - 固定 `color=1.0, depth=0.1, mw2=0.1`：[third_party/RegGS/src/entities/gaussian_aligner.py](third_party/RegGS/src/entities/gaussian_aligner.py#L392), [third_party/RegGS/src/entities/gaussian_aligner.py](third_party/RegGS/src/entities/gaussian_aligner.py#L393), [third_party/RegGS/src/entities/gaussian_aligner.py](third_party/RegGS/src/entities/gaussian_aligner.py#L394)
   - 结论：后续应优先把三项权重暴露到 yaml，再做与论文一致的消融与权重调度实验。

2. 论文测试协议是“全量非训练帧”；本地实现按 `sample_rate` 子采样测试帧：
   - `test_frame_ids = frame_ids[int(sample_rate/2)::sample_rate]` [third_party/RegGS/src/entities/reggs.py](third_party/RegGS/src/entities/reggs.py#L45)
   - `train_frame_ids` 从剩余帧等距抽 `n_views`：[third_party/RegGS/src/entities/reggs.py](third_party/RegGS/src/entities/reggs.py#L48)
   - 结论：本地协议与论文不完全一致，比较论文数字时需注意不可直接横比。

3. 论文强调尺度归一化与尺度初始化；本地已做了鲁棒化实现：
   - `scale_min/scale_max/depth_valid_min/mw2_warmup_iters` 可配置：[third_party/RegGS/src/entities/gaussian_aligner.py](third_party/RegGS/src/entities/gaussian_aligner.py#L47), [third_party/RegGS/src/entities/gaussian_aligner.py](third_party/RegGS/src/entities/gaussian_aligner.py#L51)
   - 迭代轮数为 `iterations * 2`：[third_party/RegGS/src/entities/gaussian_aligner.py](third_party/RegGS/src/entities/gaussian_aligner.py#L385)
   - 结论：这组稳定项值得在 DL3DV 上前置开启（而不是后置）。

4. 论文强调深度有效区域；本地确有对应掩码链路但默认偏宽松：
   - 深度损失有效区门槛：`main_depth_i > 0.5 & mini_depth_i > 0.5` [third_party/RegGS/src/entities/gaussian_aligner.py](third_party/RegGS/src/entities/gaussian_aligner.py#L105)
   - 颜色损失可受 `filter_alpha / mask_invalid_depth` 影响：[third_party/RegGS/src/entities/gaussian_aligner.py](third_party/RegGS/src/entities/gaussian_aligner.py#L35), [third_party/RegGS/src/entities/gaussian_aligner.py](third_party/RegGS/src/entities/gaussian_aligner.py#L38), [third_party/RegGS/src/entities/gaussian_aligner.py](third_party/RegGS/src/entities/gaussian_aligner.py#L114)
   - 结论：你的阶段1建议（开启过滤和无效深度屏蔽）与论文方向一致，应提高优先级。

### 7.4 调参策略更新（结合阶段1 + 论文）

更新后的优先级：

1. 先把“帧间位移跨度”降到可收敛区（采样优先）。
2. 保持三项损失联合，不做单项关闭（尤其不要关 MW2 或 Depth）。
3. 提前开启尺度与掩码稳定项，减少早期错误梯度。
4. 仅在 ATE 明显下降后再加大 refine。

对应动作：

1. 采样/子图：
   - `n_views: 8 -> 12 -> 16`
   - 或 `sample_rate: 30 -> 20 -> 16`
   - `new_submap_every: 2 -> 3`（必要时 4）

2. 对齐稳定项（建议直接并入所有候选组，而非最后再加）：
   - `aligner.iterations: 300`（实际 600 步）
   - `aligner.cam_rot_lr: 1e-4`
   - `aligner.cam_trans_lr: 1e-3`
   - `aligner.filter_alpha: true`
   - `aligner.alpha_thre: 0.95`
   - `aligner.mask_invalid_depth: true`
   - `aligner.scale_min: 0.1`
   - `aligner.scale_max: 10.0`
   - `aligner.depth_valid_min: 0.1`
   - `aligner.mw2_warmup_iters: 50`

3. 代码级后续（下一阶段可做）：
   - 把 `color/depth/mw2` 权重暴露到 yaml，并支持简单 warmup/schedule。
   - 新增一个“论文协议评估模式”（test=all non-train）用于与论文口径对齐。

### 7.5 下一轮最小实验矩阵（先 infer + metric）

只保留 4 组，先筛 ATE：

1. P0：当前基线（8/30/submap2）
2. P1：12/30/submap2 + 稳定项全开
3. P2：16/30/submap3 + 稳定项全开
4. P3：8/20/submap2 + 稳定项全开

进入 refine 的门槛：

- ATE 至少较 P0 下降 30%
- `vis_align` 不再出现连续大漂移
- `vis_integrate` 黑区显著减少

若 P1/P2/P3 都未明显降 ATE，则优先处理 NoPo 权重域偏差（补齐 `mixRe10kDl3dv.ckpt`），再继续 refine。

### 8. 关键结论 （截止stage2）
1. 现阶段主要瓶颈是 infer/alignment 位姿误差累积，不是 refine 轮数不够。
2. 同样 8/30 配置下，DL3DV 明显失稳而 Re10k 正常，核心证据是 ATE 与画面退化同时出现。
3. 你这条 DL3DV 序列在 8/30 采样下相邻训练视角几何跨度过大，超出当前对齐稳定域。
4. 当前使用 re10k NoPo 权重跑 DL3DV，且本地缺少 dl3dv 混合权重，存在显著域偏差风险。
5. 论文也明确“大帧间运动会导致注册不收敛”，与本地现象一致。
6. 论文与消融都支持三项联合优化（MW2 + Photo + Depth），不应通过关闭单项损失来“救结果”。
7. 本地实现目前是固定损失权重（color/depth/mw2），与论文的可调权重表述不完全一致，后续应优先把权重开放到配置。
8. 本地评测协议与论文口径不完全一致（本地 test 是按 sample_rate 子采样，论文是所有非训练帧），论文数字不能直接横比。
9. 调参优先级已确定：先降采样难度与子图切换频率，再增强 aligner 稳定性，最后才做 refine。
10. 当前最小实验矩阵是 P0-P3（先 infer+metric），若 ATE 不显著下降，优先补齐 NoPo 权重再谈 refine。
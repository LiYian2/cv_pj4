# hermes.md

> 用途：给后续 Hermes 自己看的快速回忆文档。对话断了、上下文丢了、或者用户让我“先回忆一下”时，先看这份，再按这里给出的文档路径继续展开。
> 维护原则：轻量覆盖更新，不做第二份 changelog，只保留当前最该知道的状态、文档入口和下一步重点。

## 1. 现在先怎么用这份文档

如果用户让我回忆最近做了什么，默认按这个顺序：
1. 先看本文件 `docs/hermes.md`
2. 再看 `docs/P0_absprior_and_P1A_stageA_signal_compare_20260415.md`
3. 再看 `docs/BRPO_absprior_compare_local_gating_execution_plan_20260415.md`
4. 再看 `docs/STATUS.md`
5. 再看 `docs/DESIGN.md`
6. 再看 `docs/CHANGELOG.md`

如果用户问“现在为什么不继续看 StageA replay”，先去看：
- `docs/P0_absprior_and_P1A_stageA_signal_compare_20260415.md`
- 里面已经明确写了：当前纯 `stage_mode=stageA` 不更新 Gaussian，replay-on-PLY 只是 identity sanity check。

## 2. 当前项目位置（压缩后最新状态）

当前主线已经从“继续在旧 mask/depth 链路里叠规则”推进到更收敛的六步状态：
1. abs prior 固定成可复用背景
2. signal branch 完成最小必要比较
3. P2 local Gaussian gating 第一版已接通
4. 8-frame `StageA.5` gated vs ungated compare 已完成
5. 带 real branch 的 `StageB` gated vs ungated compare 已完成
6. legacy gate threshold 诊断 + calibration 已完成
7. `RGB-only v2` 的 `StageA.5` short compare 已完成，当前主线已转向这条支路
8. `RGB-only v2 + gated_rgb0192` 的 `StageB` real-branch short compare 已完成，并通过了 joint refine 短验证

已经完成的关键节点：
- `E1 / E1.5 / E2` 已完成，当前 pseudo selection default winner 仍是 `signal-aware-8 = [23, 57, 92, 127, 162, 196, 225, 260]`
- canonical E1 root 上的新版 fusion 已补齐
- canonical E1 root 上的 8-frame `signal_v2` 已生成完成
- `P0` split abs prior 标定已完成
- `P1A` StageA-only 的 `legacy / v2-rgb-only / v2-full` compare 已完成
- `P2` 第一版 local Gaussian gating 已完成代码接入与 smoke
- `P2-B` 8-frame `StageA.5` 的 `legacy ungated vs gated` short compare 已完成
- `P2-C` 带 real branch 的 `StageB ungated vs gated` short compare 已完成
- `P2-D` 已确认 current threshold 在 legacy 上几乎是 no-op
- `P2-E` 已完成 legacy `StageA.5` threshold calibration（`0.02 / 0.03`）
- `P2-F` 已完成 `RGB-only v2` 的 `StageA.5` ungated vs branch-specific gated compare
- `P2-G` 已完成 `RGB-only v2 + gated_rgb0192` 的 `StageB` real-branch short compare

当前真正未完成、也是最高优先级的事：
- 以 `RGB-only v2 + gated_rgb0192` 为主候选，做一轮更长一点的 `StageB` / 完整 schedule verify
- 只有在更长验证明确暴露 coverage 瓶颈后，再分析是否需要受几何约束的 support/depth expand；当前不急着做 raw RGB densify

## 3. 当前最重要的结论

### 3.1 关于 abs prior

当前可以先固定：
- `lambda_abs_t = 3.0`
- `lambda_abs_r = 0.1`
- `abs_pose_robust = charbonnier`
- `stageA_abs_pose_scale_source = render_depth_trainmask_median`

原因不是它“最优”已经完全证明，而是：
- 相比 noabs，它能显著收住 drift；
- 相比更弱配置，它更稳定；
- 相比更强配置，它没有明显多出来的收益；
- 梯度量级上它已经进入有效区间，但还没压过 depth_total 一个数量级以上。

### 3.2 关于 signal 侧

当前 `mask / depth / confidence` pipeline 的角色，更像 signal gate，而不是会持续制造大量新增监督的主引擎。

这一轮之后要记住五句话：
- `v2 RGB-only` 不只是“StageA-only 看起来不坏”，而是已经在 `StageA.5` 和 `StageB` 上都明显优于 legacy；
- `full v2 depth` 当前仍然过窄；
- `RGB-only v2 + gated_rgb0192` 已经成为当前主候选 refine 分支；
- raw RGB support 虽然看起来点状，但当前证据还不支持立刻去 densify raw RGB mask；
- 所以下一步更该做的是更长一点的 joint verify，而不是继续围着 full-v2 depth 或 raw RGB densify 打转。

目前最关键的 grounded 对照：
- legacy `StageA.5` ungated replay：`PSNR ≈ 23.8518 / SSIM ≈ 0.87064 / LPIPS ≈ 0.08093`
- `RGB-only v2` `StageA.5` ungated replay：`PSNR ≈ 23.9232 / SSIM ≈ 0.87228 / LPIPS ≈ 0.07995`
- `RGB-only v2` `StageA.5` gated_rgb0192 replay：`PSNR ≈ 23.9248 / SSIM ≈ 0.87232 / LPIPS ≈ 0.07993`
- `RGB-only v2` `StageB` ungated replay：`PSNR ≈ 23.9945 / SSIM ≈ 0.87287 / LPIPS ≈ 0.08038`
- `RGB-only v2` `StageB` gated_rgb0192 replay：`PSNR ≈ 24.0106 / SSIM ≈ 0.87302 / LPIPS ≈ 0.08036`

### 3.3 关于 replay / compare 口径

这是压缩后最容易丢的关键信息，必须记住：

当前纯 `stage_mode=stageA` 不更新 Gaussian。

已经实测验证：
- `BASE_PLY`
- `stageA_noabs_80/refined_gaussians.ply`
- `stageA_abs_t3_r0p1_80/refined_gaussians.ply`

三者 `sha256sum` 完全一致。

因此：
- `StageA-only replay-on-PLY` 不是判优指标；
- 它只能说明“没有把 PLY 写坏”；
- 如果用户问“为什么现在不看 replay delta”，就直接回答：因为当前这个 stage 根本没改 Gaussian，replay 对 PLY 没有区分度。

### 3.4 关于 map-side refine

当前最值得优先做的 map-side 改动不是 full SPGM，而是 `local Gaussian gating / 子集 refine`。

原因已经进一步明确：
- abs prior 已经够用，不是当前第一瓶颈；
- full-v2 depth 也已证明太窄；
- StageA-only replay 又没有信息量；
- 所以真正该解决的是：`weak local supervision -> global Gaussian perturbation`

## 4. 当前必看的专项文档

### 第一优先级：这次必须先看的
1. `docs/P0_absprior_and_P1A_stageA_signal_compare_20260415.md`
   - 用途：看 P0 和 P1A 的真实 grounded 结果
   - 重点：abs prior 为什么固定成 `3.0 / 0.1`，以及为什么 StageA-only replay 无信息
2. `docs/BRPO_absprior_compare_local_gating_execution_plan_20260415.md`
   - 用途：看当前总执行方案
   - 重点：后续顺序固定为 `P0 -> P1 -> P2`，且现在应直接进入 `P2`

### 第二优先级：实现与设计背景
1. `docs/BRPO_fusion_mask_spgm_subset_refine.md`
   - 用途：看为什么当前应该优先 local gating，而不是先上 full SPGM
2. `docs/STATUS.md`
   - 用途：看当前状态与实验根
3. `docs/DESIGN.md`
   - 用途：看为什么“StageA-only replay 无信息”已经变成设计结论，而不是临时口头判断
4. `docs/CHANGELOG.md`
   - 用途：看 2026-04-15 ~ 2026-04-16 这轮真实做过的工程与实验更新

## 5. 当前回答用户时的推荐口径

如果用户问“现在结论是什么”，优先回答：
- abs prior 先固定 `lambda_abs_t=3.0`、`lambda_abs_r=0.1`
- `RGB-only v2 + gated_rgb0192` 已经在 `StageA.5` 和 `StageB` 两层都优于 legacy，可作为当前主候选 refine 分支
- `full v2 depth` 当前仍过窄
- 当前纯 `StageA` 不改 Gaussian，所以 replay 不是有效判优指标
- 下一步更该做的是围绕当前主候选做更长一点的 joint verify，而不是先做 raw RGB densify

如果用户问“为什么不继续做 StageA compare”，优先回答：
- 因为当前 StageA 只改 pseudo camera / exposure，不改 Gaussian
- replay-on-PLY 在这个 stage 上没有区分度
- 后续真正依赖 replay 的 compare 要放到 `StageA.5` 或其他会更新 Gaussian 的 stage

如果用户问“下一步具体做什么”，优先回答：
1. 以 `RGB-only v2 + gated_rgb0192` 为当前主候选，做更长一点的 `StageB` / 完整 schedule verify
2. 如果更长验证仍然成立，再决定是否需要更大预算或更完整 joint schedule
3. 只有在这条线明确暴露 coverage 瓶颈时，再考虑受几何约束的 support/depth expand；先不要直接 densify raw RGB mask

## 6. 下次回来先检查什么

1. `RGB-only v2` 的最新主对照报告：`docs/P2F_stageA5_v2rgbonly_gating_compare_20260416.md`
2. `RGB-only v2` 的最新 `StageB` real-branch 报告：`docs/P2G_stageB_v2rgbonly_realbranch_compare_20260416.md`
3. 是否已经补做更长一点的 `StageB` / 完整 schedule verify
4. 若要继续做 gating，是否仍围绕 `min_rgb_mask_ratio` 做 branch-specific 微调，而不是回 legacy 侧细扫
5. 若讨论 densify，先确认讨论对象是不是 `raw_rgb_confidence_v2` 本身；默认更应考虑受几何约束的 support/depth expand
6. 实验输出默认是否已切到 `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments`

## 7. 给下一个 Hermes 的一句话

现在不要再把主要注意力放在 full-v2 depth 的细调上，也不要再把 StageA-only replay 当成有效指标。legacy 侧的 gate 诊断和 calibration 已经做完，`RGB-only v2 + gated_rgb0192` 也已经通过了 `StageB` real-branch 短验证；下一步应直接做更长一点的 joint verify，确认它是不是稳定主线，而不是先做 raw RGB densify。

## 8. 云服务器环境信息（固定配置）

### SSH alias
- `Group8DDY`（大小写敏感，必须精确匹配）

### Python 环境
s3po 部分：
- Conda env: `s3po-gs`
- 完整路径: `/home/bzhang512/miniconda3/envs/s3po-gs/bin/python`
- 激活命令: `source ~/.bashrc && conda activate s3po-gs`

difix 部分：
- Conda env: `reggs`
- 完整路径: `/home/bzhang512/miniconda3/envs/reggs/bin/python`
- 激活命令: `source ~/.bashrc && conda activate reggs`

### PYTHONPATH（远端执行必须显式设置）
- 必须包含: `/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO`

### 项目路径
- Part3 BRPO root: `/home/bzhang512/CV_Project/part3_BRPO`
- 实验输出默认优先: `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments`
- 旧输出: `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments`

### 执行模板
```bash
ssh Group8DDY "cd /home/bzhang512/CV_Project/part3_BRPO && export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO:$PYTHONPATH && /home/bzhang512/miniconda3/envs/s3po-gs/bin/python scripts/xxx.py --args"
```

### 注意事项
- SSH alias 必须用 `Group8DDY`（不是 `group8ddy`）
- 远端执行必须显式设置 `PYTHONPATH`
- 当前磁盘很紧：`/` 也偏满，`/data` 已满；新实验默认写 `/data2`，不要再往 `/data` 或低收益的大目录复制上堆东西

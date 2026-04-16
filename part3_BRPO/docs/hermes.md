# hermes.md

> 用途：给后续 Hermes 自己看的快速回忆文档。对话断了、上下文丢了、或者用户让我“先回忆一下”时，先看这份，再按这里给出的文档路径继续展开。
> 维护原则：轻量覆盖更新，不做第二份 changelog，只保留当前最该知道的状态、文档入口和下一步重点。

## 1. 现在先怎么用这份文档

如果用户让我回忆最近做了什么，默认按这个顺序：
1. 先看本文件 `docs/hermes.md`
2. 再看 `docs/P2H_stageB_v2rgbonly_verify120_20260416.md`
3. 再看 `docs/P2G_stageB_v2rgbonly_realbranch_compare_20260416.md`
4. 再看 `docs/P2F_stageA5_v2rgbonly_gating_compare_20260416.md`
5. 再看 `docs/STATUS.md`
6. 再看 `docs/DESIGN.md`
7. 再看 `docs/CHANGELOG.md`

如果用户问“为什么现在不继续看 StageA replay”，先看：
- `docs/P0_absprior_and_P1A_stageA_signal_compare_20260415.md`
- 核心原因已经固定：当前纯 `stage_mode=stageA` 不更新 Gaussian，replay-on-PLY 只是 identity sanity check。

## 2. 当前项目位置（最新压缩状态）

当前主线已经推进到这里：
1. abs prior 已固定成可复用背景；
2. signal branch 的最小必要 compare 已完成；
3. local Gaussian gating 第一版已接通；
4. `RGB-only v2` 已在 `StageA.5` 和 `StageB-20iter` 上优于 legacy；
5. 但最新 `P2-H` 已确认：把当前 `StageB` 预算拉到 `120iter` 后，两条 v2 臂都会退回负区间；
6. 所以下一个主问题已经不是“要不要切到 v2”，而是“怎么稳住 `StageB` 后段，而不是让 20iter 的正益在 120iter 被吃掉”。

已经完成的关键节点：
- `E1 / E1.5 / E2` 已完成，当前 pseudo selection default winner 仍是 `signal-aware-8 = [23, 57, 92, 127, 162, 196, 225, 260]`
- canonical E1 root 上的新版 fusion 已补齐
- canonical E1 root 上的 8-frame `signal_v2` 已生成完成
- `P0` split abs prior 标定已完成
- `P1A` StageA-only 的 `legacy / v2-rgb-only / v2-full` compare 已完成
- `P2` 第一版 local Gaussian gating 已完成代码接入与 smoke
- `P2-B / P2-C / P2-D / P2-E` 已完成，legacy 阈值诊断与 calibration 已基本收口
- `P2-F` 已确认 `RGB-only v2` 在 `StageA.5` 上明显优于 legacy，且 `gated_rgb0192` 有小幅正增益
- `P2-G` 已确认 `RGB-only v2 + gated_rgb0192` 在 `StageB-20iter` 上优于本支路 ungated，也优于 legacy
- `P2-H` 已确认：到了 `StageB-120iter`，ungated / gated 两臂都低于 `after_opt` baseline，也低于各自 `StageA.5` handoff 起点；gating 仍然真实工作，但只能轻微减缓退化，不能阻止 regression

## 3. 当前最重要的结论

### 3.1 关于 abs prior

当前先固定：
- `lambda_abs_t = 3.0`
- `lambda_abs_r = 0.1`
- `abs_pose_robust = charbonnier`
- `stageA_abs_pose_scale_source = render_depth_trainmask_median`

它的角色是稳定可复用背景，不是当前主瓶颈。

### 3.2 关于 signal / gating 主线

现在要记住六句话：
- `v2 RGB-only` 已经证明自己比 legacy 更值得保留；
- `full v2 depth` 当前仍然过窄；
- `RGB-only v2 + gated_rgb0192` 仍然是当前最好的一条 refine 分支；
- 但它目前只证明了 `StageA.5` 和 `StageB-20iter` 的短程优势；
- `P2-H` 已经说明它还不是稳定的中预算 `StageB` 主线；
- 所以下一步该做的是 `StageB stabilization`，不是 raw RGB densify。

目前最关键的 grounded 对照：
- `RGB-only v2` `StageA.5` gated_rgb0192 replay：`PSNR ≈ 23.9248 / SSIM ≈ 0.87232 / LPIPS ≈ 0.07993`
- `RGB-only v2` `StageB-20iter` gated_rgb0192 replay：`PSNR ≈ 24.0106 / SSIM ≈ 0.87302 / LPIPS ≈ 0.08036`
- `RGB-only v2` `StageB-120iter` gated_rgb0192 replay：`PSNR ≈ 23.9146 / SSIM ≈ 0.86873 / LPIPS ≈ 0.08485`

这三行本身就说明了当前问题：`StageB` 的短程 gain 没有自然延续到 `120iter`。

### 3.3 关于 replay / compare 口径

这是压缩后最容易丢的关键信息，必须记住：
- 当前纯 `stage_mode=stageA` 不更新 Gaussian；
- `BASE_PLY` 与 `stageA_noabs_80 / stageA_abs_t3_r0p1_80` 的 `refined_gaussians.ply` 已验证同 hash；
- 所以 `StageA-only replay` 不是判优指标；
- 真正有区分度的 replay compare 只放在会更新 Gaussian 的 `StageA.5 / StageB`。

### 3.4 关于当前真正的瓶颈

`P2-H` 之后，当前最该优先锁定的瓶颈已经不是：
- legacy threshold 太松；
- raw RGB mask 太稀；
- full-v2 depth 还没再调够。

当前更像是：
- gating 机制已经真实工作；
- real branch 没被明显误伤；
- 但现有 `StageB` 后段训练动力学会把 `20iter` 的正益吃掉。

所以优先级已经收束为：`StageB stabilization > support/depth expand > raw RGB densify`。

## 4. 当前必看的专项文档

### 第一优先级
1. `docs/P2H_stageB_v2rgbonly_verify120_20260416.md`
   - 用途：看最新 grounded 结论
   - 重点：为什么当前主候选虽然在 20iter 是正向的，但到 120iter 还会 regression
2. `docs/P2G_stageB_v2rgbonly_realbranch_compare_20260416.md`
   - 用途：看当前主候选是怎么从 short compare 里站出来的
3. `docs/P2F_stageA5_v2rgbonly_gating_compare_20260416.md`
   - 用途：看为什么 `RGB-only v2` 比 legacy 更值得保留

### 第二优先级
1. `docs/STATUS.md`
2. `docs/DESIGN.md`
3. `docs/CHANGELOG.md`
4. `docs/BRPO_absprior_compare_local_gating_execution_plan_20260415.md`

## 5. 当前回答用户时的推荐口径

如果用户问“现在结论是什么”，优先回答：
- `RGB-only v2 + gated_rgb0192` 仍然是当前最好的一条 refine 分支；
- 但 `P2-H` 已经确认：它在 `StageB-120iter` 上还不稳；
- gating 机制是真实工作的，real branch 也没明显被伤到；
- 所以当前主问题不是“gate 没生效”，而是 `StageB` 后段稳定性；
- 下一步不该先去 raw RGB densify，而该做 `StageB stabilization`。

如果用户问“下一步具体做什么”，优先回答：
1. 围绕 gated 主候选做 `StageB` 回落窗口定位（至少 `20 / 40 / 80 / 120`）
2. 以 `20iter` 之后的后段 `lr / lambda_real : lambda_pseudo` 调度为第一批稳定化变量
3. 只有在 `StageB stabilization` 仍然稳定失败时，再讨论 early-stop 是否成为默认策略
4. 在这之前不要先做 raw RGB densify

## 6. 下次回来先检查什么

1. `docs/P2H_stageB_v2rgbonly_verify120_20260416.md`
2. `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2h_stageB_v2rgbonly_verify120_e1/summary.json`
3. 是否已经开始做 `StageB stabilization` 的窗口定位或后段 schedule compare
4. 如果要讨论 support/densify，先确认是不是在 `StageB stabilization` 明确失败之后才谈

## 7. 给下一个 Hermes 的一句话

不要再把下一步表述成“先做更长一点的 StageB verify”。这件事已经由 `P2-H` 回答了：当前 `RGB-only v2 + gated_rgb0192` 在 `20iter` 好、到 `120iter` 会回落。现在真正该做的是 `StageB stabilization`（窗口定位 + 后段调度），而不是先去 densify raw RGB mask。

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
- 当前 `/` 盘仍很紧，`/data` 已满；新实验默认写 `/data2`，文档修改也尽量保持轻量，不要做低收益复制

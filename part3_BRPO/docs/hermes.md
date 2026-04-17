# hermes.md

> 用途：给后续 Hermes 自己看的快速回忆文档。对话断了、上下文丢了、或者用户让我“先回忆一下”时，先看这份，再按这里给出的文档路径继续展开。
> 维护原则：轻量覆盖更新，不做第二份 changelog，只保留当前最该知道的状态、文档入口和下一步重点。

## 1. 现在先怎么用这份文档

如果用户让我回忆最近做了什么，默认按这个顺序：
1. 先看本文件 `docs/hermes.md`
2. 再看 `docs/P2M_spgm_conservative_repair_compare_20260417.md`
3. 再看 `docs/P2L_spgm_canonical_stageB_compare_20260417.md`
4. 再看 `docs/P2K_spgm_v1_forensic_audit_20260416.md`
5. 再看 `docs/P2J_bounded_stageB_schedule_compare_20260416.md`
6. 再看 `docs/SPGM_landing_plan_for_part3_BRPO.md`
7. 再看 `docs/P2I_stageB_window_localization_20260416.md`
8. 再看 `docs/P2H_stageB_v2rgbonly_verify120_20260416.md`
9. 再看 `docs/STATUS.md`
10. 再看 `docs/DESIGN.md`
11. 再看 `docs/CHANGELOG.md`

如果用户问“为什么现在不继续看 StageA replay”，先看：
- `docs/P0_absprior_and_P1A_stageA_signal_compare_20260415.md`
- 核心原因已经固定：当前纯 `stage_mode=stageA` 不更新 Gaussian，replay-on-PLY 只是 identity sanity check。

## 2. 当前项目位置（最新压缩状态）

当前主线已经推进到这里：
1. abs prior 已固定成可复用背景；
2. signal branch 的最小必要 compare 已完成；
3. local Gaussian gating 第一版已接通；
4. `RGB-only v2` 已在 `StageA.5` 和 `StageB-20iter` 上优于 legacy；
5. `P2-I` 已把当前 winner 的窗口定位出来：`20 / 40 / 80` 仍有可用窗口，但 `80 -> 120` 会明显 cliff；
6. `P2-J` 又进一步表明：一个 bounded late-stage schedule 就能把 `120iter` 从 regression 区拉回来，其中 `post40_lr03_120` 是当前最强 bounded StageB baseline；
7. 但这个结果仍然只说明 “StageB 可以被 bounded 地救回来”，不说明继续深调 StageB 是主线；
8. `P2-L` 已进一步确认：即便 compare protocol 对齐到 canonical `post40_lr03_120 / gated_rgb0192`，当前 deterministic `SPGM v1` 仍低于 canonical baseline；
9. `P2-M` 又进一步确认：conservative deterministic repair 能明显追回这部分损失，但还不能完全追平 baseline；
10. 因而当前更合理的主工程方向已经从“先做 conservative repair”转成 `selector-first SPGM repair`，而不是继续把 StageB 或 soft-suppress-only SPGM 放大。

已经完成的关键节点：
- `E1 / E1.5 / E2` 已完成，当前 pseudo selection default winner 仍是 `signal-aware-8 = [23, 57, 92, 127, 162, 196, 225, 260]`
- canonical E1 root 上的新版 fusion 已补齐
- canonical E1 root 上的 8-frame `signal_v2` 已生成完成
- `P0` split abs prior 标定已完成
- `P1A` StageA-only 的 `legacy / v2-rgb-only / v2-full` compare 已完成
- `P2` 第一版 local Gaussian gating 已完成代码接入与 smoke
- `P2-F` 已确认 `RGB-only v2` 在 `StageA.5` 上明显优于 legacy
- `P2-G` 已确认 `RGB-only v2 + gated_rgb0192` 在 `StageB-20iter` 上优于本支路 ungated，也优于 legacy
- `P2-H` 已确认原始 `StageB-120iter` 会 regression
- `P2-I` 已确认当前 winner 的窗口不是只有 `20iter`，而是 `40 / 80` 仍可用，但 `80 -> 120` 有 cliff
- `P2-J` 已确认 bounded schedule 能显著缓解这个 cliff；`post40_lr03_120` 当前是最好的 bounded StageB baseline
- `scripts/run_pseudo_refinement_v2.py` 已新增 StageB post-switch 调度 CLI：
  - `--stageB_post_switch_iter`
  - `--stageB_post_lr_scale_xyz`
  - `--stageB_post_lr_scale_opacity`
  - `--stageB_post_lambda_real`
  - `--stageB_post_lambda_pseudo`

当前真正要盯住的三件事：
- 参考/回退臂：保留 `post40_lr03_120` 作为 canonical bounded StageB baseline
- repair 结论：`P2-M` 已确认 conservative deterministic 确实有效，其中 A 臂 `spgm_repair_a_keep111_eta0_wf025` 是当前最佳 repair anchor；但它仍略低于 baseline [参见 `docs/P2M_spgm_conservative_repair_compare_20260417.md`]
- 主线下一步：从 A 臂出发做 `selector-first SPGM repair`，不是继续只扫 soft suppress 强度，也不是直接跳 stochastic / xyz+opacity / 更长 iter

## 3. 当前最重要的结论

### 3.1 关于 abs prior

当前先固定：
- `lambda_abs_t = 3.0`
- `lambda_abs_r = 0.1`
- `abs_pose_robust = charbonnier`
- `stageA_abs_pose_scale_source = render_depth_trainmask_median`

它的角色是稳定可复用背景，不是当前主瓶颈。

### 3.2 关于 signal / gating / StageB 主线

现在要记住九句话：
- `v2 RGB-only` 已经证明自己比 legacy 更值得保留；
- `full v2 depth` 当前仍然过窄；
- `RGB-only v2 + gated_rgb0192` 仍然是当前最好的一条 refine 分支；
- 它不是只有 `20iter` 有效，当前窗口至少能延续到 `80iter`；
- 原始 schedule 下 `120iter` 会明显回落；
- 但 bounded late-stage schedule 能把这个 cliff 大幅拉回；
- 当前最强 bounded StageB baseline 是 `post40_lr03_120`；
- 即便如此，这个结果也更像“把 120 救回到接近 40iter 的水平”，而不是“发现了一个值得继续深调的新长程 regime”；
- 因此接下来应把主工程重心转向 SPGM，而不是继续无边界地磨 StageB。

目前最关键的 grounded 对照：
- `StageA.5 gated_rgb0192`：`23.9248 / 0.87232 / 0.07993`
- `StageB-40 gated_rgb0192`：`24.02235 / 0.87253 / 0.08122`
- `StageB-120 raw schedule`：`23.91443 / 0.86873 / 0.08484`
- `StageB-120 post40_lr03`：`24.03019 / 0.87229 / 0.08177`

这四行说明了现在的真实局面：
- bounded schedule 确实能救回 120；
- 但它并没有把 StageB 变成一个明显强于早期窗口的全新 regime；
- 所以 StageB 可以留一个 bounded baseline，但主线不该继续压在它上面。

### 3.3 关于 replay / compare 口径

这是压缩后最容易丢的关键信息，必须记住：
- 当前纯 `stage_mode=stageA` 不更新 Gaussian；
- `BASE_PLY` 与 `stageA_noabs_80 / stageA_abs_t3_r0p1_80` 的 `refined_gaussians.ply` 已验证同 hash；
- 所以 `StageA-only replay` 不是判优指标；
- 真正有区分度的 replay compare 只放在会更新 Gaussian 的 `StageA.5 / StageB`。

### 3.4 关于当前真正的瓶颈与 SPGM 的角色

现在最该优先锁定的瓶颈已经不是：
- legacy threshold 太松；
- raw RGB mask 太稀；
- full-v2 depth 还没再调够；
- 或再继续盲目拉长 StageB budget。

当前更像是：
- view gate 已经在真实工作；
- bounded schedule 也能把 late-stage cliff 部分救回；
- 但这条线的边际解释空间已经越来越小。

而 SPGM 文档给出的关键判断是：
- SPGM 不该被当成新的 signal 生产器；
- 它更像是把当前的 view-conditioned grad mask 升级成 Gaussian-state-aware grad management；
- 也就是：保留现有 view gate 作为 sample filter，把真正的 per-Gaussian 更新权限管理交给 SPGM。

因此当前推荐优先级应理解成：
- `主线工程：selector-first SPGM repair（以 repair A 为 anchor） >`
- `bounded StageB baseline 保留/对照 >`
- `raw RGB densify / support expand`

## 4. 当前必看的专项文档

### 第一优先级
1. `docs/P2M_spgm_conservative_repair_compare_20260417.md`
   - 用途：看 conservative deterministic repair 是否真的能把 replay 拉回来
   - 重点：为什么它虽然有效，但仍不足以替代 canonical baseline，以及为什么下一步应改 selector 而不是继续只改 suppress 强度
2. `docs/P2L_spgm_canonical_stageB_compare_20260417.md`
   - 用途：看对齐 canonical baseline 之后的正式 compare 结果
   - 重点：为什么现在已经不能再把问题主要归因于 protocol drift，而应转向 suppressor → selector 的 policy repair
3. `docs/P2J_bounded_stageB_schedule_compare_20260416.md`
   - 用途：看 bounded StageB compare 的最后结论
   - 重点：为什么 `post40_lr03_120` 可以作为参考 baseline，但还不足以继续把主线压在 StageB 上
4. `docs/SPGM_landing_plan_for_part3_BRPO.md`
   - 用途：看 SPGM 的原理分析与第一版落地切入点
   - 重点：SPGM 应插在现有 local gating 回路里，接管 per-Gaussian grad weight，而不是重写上游 signal
5. `docs/P2I_stageB_window_localization_20260416.md`

### 第二优先级
1. `docs/STATUS.md`
2. `docs/DESIGN.md`
3. `docs/CHANGELOG.md`

## 5. 当前回答用户时的推荐口径

如果用户问“现在结论是什么”，优先回答：
- bounded StageB schedule compare 已经做完，`post40_lr03_120` 仍是 canonical bounded StageB baseline；
- `P2-L` 已确认 current deterministic `SPGM v1` 在 canonical protocol 下仍低于 baseline；
- `P2-M` 又确认 conservative deterministic repair 能明显追回这部分损失，但仍没有完全追平 baseline；
- 所以下一步不是继续只扫软压制强度，而是从 repair A 出发做 selector-first 的 SPGM repair。

如果用户问“下一步具体做什么”，优先回答：
1. 保留 `post40_lr03_120` 作为 canonical bounded StageB baseline / 对照臂
2. 以 `spgm_repair_a_keep111_eta0_wf025` 为新 anchor
3. 把 policy 改成 cluster 内 quantile / top-k 的 selector-first 版本，让 active set 真正发生变化
4. 在这之前不要先做 stochastic drop、长 iter 扫描或 raw RGB densify

## 6. 下次回来先检查什么

1. `docs/hermes.md`
2. `docs/P2M_spgm_conservative_repair_compare_20260417.md`
3. `docs/P2L_spgm_canonical_stageB_compare_20260417.md`
4. `docs/P2K_spgm_v1_forensic_audit_20260416.md`
5. `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_p2m_spgm_conservative_repair_e1/summary.json`
6. `scripts/run_pseudo_refinement_v2.py` 里 `stageB_post_*` 与 SPGM 相关 CLI / history 字段是否保持一致

## 7. 给下一个 Hermes 的一句话

不要把当前状态误读成“再把 soft suppress 调轻一点就够了”。现在更准确的状态是：bounded StageB baseline 仍是 `post40_lr03_120`，canonical formal compare 和 conservative repair compare 都已经补齐；current deterministic SPGM 的 suppress 强度确实能被部分修回，但仅靠更温和的 soft suppress 仍不够。下一步应从 repair A 出发做 selector-first rewrite，再谈 stochastic 或长程推进。

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

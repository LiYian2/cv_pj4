# P2-H：RGB-only v2 StageB verify120（2026-04-16）

## 1. 目的

在 `P2-G` 已证明 `RGB-only v2 + gated_rgb0192` 能在 `StageB` 20iter short compare 中优于本支路 ungated 之后，这一轮不直接跳到 300iter，也不先去做 support/densify 扩张，而是先做一个基于存档记录的中预算 verify：
1. 复用 `P2-G` 的同一协议，只把 `stageB_iters` 从 `20` 提到 `120`；
2. 检查 `RGB-only v2 + gated_rgb0192` 的短跑优势能否延续到更长一点的 StageB；
3. 如果不能延续，优先把问题收束到 `StageB` 后段稳定性，而不是过早解释成 coverage 不足。

之所以先做 `120iter` 而不是直接做 `300iter`，是因为旧的 legacy `StageB` 记录已经明确出现过“120iter 仍正、300iter 退化”的形态；因此当前最稳妥的下一步，是先回答这条新主候选支路在中等预算下是否还能站住。

## 2. 协议

### 2.1 固定输入
- internal cache root:
  - `/data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache`
- base ply / replay baseline:
  - `<internal_cache_root>/after_opt/point_cloud/point_cloud.ply`
- pseudo cache:
  - `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/pseudo_cache_baseline`
- signal_v2 root:
  - `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/signal_v2`
- real branch:
  - `train_manifest=/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/split_manifest.json`
  - `train_rgb_dir=/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/rgb`
- StageA.5 handoff roots（来自 `P2-F`）:
  - ungated: `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_ungated_80`
  - gated: `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_gated_rgb0192_80`

### 2.2 共同 StageB 参数
- `signal_pipeline=brpo_v2`
- `stage_mode=stageB`
- `stageA_iters=0`
- `stageB_iters=120`
- `num_pseudo_views=4`
- `num_real_views=2`
- `lambda_real=1.0`
- `lambda_pseudo=1.0`
- `stageA_rgb_mask_mode=brpo_v2_raw`
- `stageA_depth_mask_mode=train_mask`
- `stageA_target_depth_mode=target_depth_for_refine_v2`
- `stageA_depth_loss_mode=source_aware`
- `stageA_lambda_depth_seed=1.0`
- `stageA_lambda_depth_dense=0.35`
- `stageA_lambda_depth_fallback=0.0`
- `stageA_lambda_abs_t=3.0`
- `stageA_lambda_abs_r=0.1`
- `stageA_abs_pose_robust=charbonnier`
- `stageA_abs_pose_scale_source=render_depth_trainmask_median`
- `stageA5_trainable_params=xyz`
- `stageA5_lr_xyz=1e-4`
- `seed=0`
- `init_pseudo_reference_mode=keep`

### 2.3 对照臂
1. `stageB_from_stageA5_v2rgbonly_ungated_120`
   - 起点：`P2-F` ungated StageA.5 输出
   - gating：`off`
2. `stageB_from_stageA5_v2rgbonly_gated_rgb0192_120`
   - 起点：`P2-F` gated StageA.5 输出
   - gating：
     - `pseudo_local_gating=hard_visible_union_signal`
     - `pseudo_local_gating_params=xyz`
     - `pseudo_local_gating_min_verified_ratio=0.01`
     - `pseudo_local_gating_min_rgb_mask_ratio=0.0192`
     - `pseudo_local_gating_max_fallback_ratio=0.995`

### 2.4 输出路径
- run root:
  - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2h_stageB_v2rgbonly_verify120_e1`

## 3. 结果

### 3.1 replay baseline（after_opt）
- PSNR = `23.948911476135255`
- SSIM = `0.8734854695973573`
- LPIPS = `0.0787797965799217`

### 3.2 本轮 120iter replay

| arm | PSNR | SSIM | LPIPS | ΔPSNR vs after_opt | ΔSSIM | ΔLPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ungated_120 | `23.910415366843896` | `0.8686565275545474` | `0.08484659913789343` | `-0.03849610929135849` | `-0.004828942042809925` | `+0.006066802557971734` |
| gated_rgb0192_120 | `23.91462847391764` | `0.8687342405319214` | `0.08484732096117956` | `-0.034283002217613046` | `-0.004751229065435902` | `+0.006067524381257863` |

### 3.3 和前序关键锚点的关系

#### 对比 `P2-F` 自己的 StageA.5 起点
- `ungated_120 - stageA5_ungated_80`:
  - PSNR `-0.012805945784954531`
  - SSIM `-0.0036238630612690503`
  - LPIPS `+0.004897576598105607`
- `gated_120 - stageA5_gated_80`:
  - PSNR `-0.010172596684210333`
  - SSIM `-0.003585341020866717`
  - LPIPS `+0.004915932717698593`

也就是说：到 `120iter` 时，两条 `StageB` 臂都已经不仅输给 `after_opt` baseline，也输给了自己对应的 `StageA.5` handoff 起点。

#### 对比 `P2-G` 的 StageB 20iter short compare
- `ungated_120 - ungated_20`:
  - PSNR `-0.08411824261700573`
  - SSIM `-0.00421483582920501`
  - LPIPS `+0.004461936335320826`
- `gated_120 - gated_20`:
  - PSNR `-0.09597775494610872`
  - SSIM `-0.004283723566267272`
  - LPIPS `+0.004484285372826788`

这说明：`P2-G` 里看到的 20iter joint refine 正益，到了 `120iter` 并没有保持住；当前主候选仍然呈现出明显的“短跑好、往后掉”的形态。

### 3.4 gated vs ungated（120iter 内部比较）
- PSNR：gated `+0.004213107073745448`
- SSIM：gated `+7.771297737402261e-05`
- LPIPS：gated `+7.218232861289087e-07`

结论：
- gating 在 `120iter` 时仍然略优于 ungated，至少 PSNR/SSIM 方向没有翻转；
- 但这个增益已经显著小于 `P2-G` 20iter 的 `+0.0161 PSNR / +0.000147 SSIM`；
- 而且两条臂都已经整体退到负区间，所以它不足以支持“当前 StageB 已稳定可放大预算”的结论。

## 4. gating 机制是否还在真实工作

### ungated_120
- `iters_with_rejection = 0 / 120`
- `grad_keep_ratio_xyz_mean = 1.0`
- `loss_real_last = 0.1190696656703949`
- `true_pose_mean_trans = 0.000669498241991307`
- `true_pose_mean_rotF = 0.0006795773042378821`

### gated_rgb0192_120
- `iters_with_rejection = 96 / 120`
- `total_rejected_sample_evals = 116`
- `unique_rejected_ids = [225, 260]`
- `grad_keep_ratio_xyz_mean = 0.7290216875572999`
- `grad_keep_ratio_xyz_last = 0.6688389182090759`
- `loss_real_last = 0.11909361183643341`
- `true_pose_mean_trans = 0.0006502531947487961`
- `true_pose_mean_rotF = 0.0006630452800087086`

这说明：
1. 这一轮不是“gating 又退化成 no-op”——它依然在大多数迭代里真实拒掉 `225 / 260`，并持续裁剪 pseudo-side xyz 更新范围；
2. real branch 也没有被明显伤到，`loss_real_last` 与 pose aggregate 都和 ungated 基本同量级；
3. 因此当前问题已经不能再归因为“阈值太松 / gating 没生效”。更像是：即便 gating 生效了，现有 `StageB` 120iter 训练动力学仍然会把 20iter 的正益吃掉。

## 5. 当前判断

1. `RGB-only v2 + gated_rgb0192` 依然是这条分支里更好的 `StageB` 臂，但它还不是一个已经证明能稳定放大预算的主线配置；
2. 当前主候选已经暴露出清晰的 `20iter PASS -> 120iter regression` 形态；
3. 这次 regression 不是由 “gating 失效” 或 “real branch 被误伤” 直接导致的；
4. 所以当前最该优先探索的，不是 raw RGB densify，也不是 support/depth expand，而是 `StageB` 的后段稳定性：
   - 回落窗口在什么时候出现；
   - 后段学习率 / 权重 / 变量放开节奏怎么改，才能保住短程正益；
   - 如果仍保不住，默认策略应退回 `StageA.5 winner` 或 `StageB` 早停窗口，而不是继续盲目加长。

## 6. 下一步建议

下一步更合理的顺序是：
1. 不再把“更长一点的 verify”当作未回答问题；`P2-H` 已经回答了：当前主候选在 `120iter` 上还不稳。
2. 接下来转入 `StageB stabilization`，优先做最小代价、最有解释力的两类验证：
   - `window localization`：围绕 gated 主候选补 `20 / 40 / 80 / 120` 的 replay 节点，先定位回落窗口；
   - `post-short-run schedule`：从 `20iter` 之后开始做后段 `lr` 降档或 `lambda_real : lambda_pseudo` 再平衡，而不是立刻加新 signal 语义。
3. 在这条主线没证明“中预算 StageB 只是因为后段调度不稳”之前，仍不建议：
   - 直接做 raw RGB densify；
   - 回到 legacy 侧继续磨 threshold；
   - 先上 `xyz+opacity`、soft gating 或 full SPGM。

# P2-J：bounded StageB schedule compare（2026-04-16）

## 1. 目的

在 `P2-I` 已把当前 winner 的窗口定位到 `20 / 40 / 80 / 120` 之后，这一轮不再做无边界长跑，而是给 `StageB` 一次有边界的 follow-up：
1. 固定当前最佳分支 `RGB-only v2 + gated_rgb0192`；
2. 固定总 budget 仍为 `120iter`；
3. 只测试简单、可解释的 late-stage schedule，看看能否把 `80 -> 120` 的 cliff 压住；
4. 以此回答：在正式推进 SPGM 前，`StageB` 是否还值得继续投入更多 schedule 调参预算。

## 2. 这轮新增的最小工程支持

为了做 bounded compare，这一轮对 `scripts/run_pseudo_refinement_v2.py` 加了最小的 StageB post-switch 调度支持：
- `--stageB_post_switch_iter`
- `--stageB_post_lr_scale_xyz`
- `--stageB_post_lr_scale_opacity`
- `--stageB_post_lambda_real`
- `--stageB_post_lambda_pseudo`

并且把这些 late-stage 实际生效值写进 `stageB_history.json`：
- `lambda_real_effective`
- `lambda_pseudo_effective`
- `gaussian_lr_xyz_effective`
- `post_switch_applied*`

这一步的作用是：后续若还要做一次 bounded schedule compare，不需要再临时改代码。

## 3. 协议

### 3.1 固定输入
- StageA.5 handoff：
  - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_gated_rgb0192_80`
- pseudo cache：
  - `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/pseudo_cache_baseline`
- signal_v2 root：
  - `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/signal_v2`
- real branch：
  - `train_manifest=/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/split_manifest.json`
  - `train_rgb_dir=/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/rgb`

### 3.2 共同参数
- `signal_pipeline=brpo_v2`
- `stage_mode=stageB`
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
- gating：
  - `pseudo_local_gating=hard_visible_union_signal`
  - `pseudo_local_gating_min_rgb_mask_ratio=0.0192`

### 3.3 三个 bounded compare 臂
1. `stageB_post40_lr03_120`
   - `post_switch_iter=40`
   - `xyz lr *= 0.3`
2. `stageB_post80_lr03_120`
   - `post_switch_iter=80`
   - `xyz lr *= 0.3`
3. `stageB_post80_lr03_real05_120`
   - `post_switch_iter=80`
   - `xyz lr *= 0.3`
   - `lambda_real -> 0.5`

### 3.4 输出路径
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2j_stageB_bounded_schedule_compare_e1`

## 4. 结果

### 4.1 参考锚点（来自 P2-I）
- after_opt baseline：`23.94891 / 0.87349 / 0.07878`
- StageA.5 start：`23.92480 / 0.87232 / 0.07993`
- P2-I best 40iter：`24.02235 / 0.87253 / 0.08122`
- P2-I 80iter：`24.01980 / 0.87179 / 0.08247`
- P2-I 120iter：`23.91443 / 0.86873 / 0.08484`

### 4.2 本轮 compare

| arm | PSNR | SSIM | LPIPS | ΔPSNR vs P2-I 120 | ΔPSNR vs P2-I 40 |
| --- | ---: | ---: | ---: | ---: | ---: |
| post40_lr03_120 | `24.03019` | `0.87229` | `0.08177` | `+0.11576` | `+0.00784` |
| post80_lr03_120 | `23.99358` | `0.87099` | `0.08307` | `+0.07915` | `-0.02877` |
| post80_lr03_real05_120 | `24.01828` | `0.87167` | `0.08255` | `+0.10384` | `-0.00407` |

best arm：
- by PSNR: `post40_lr03_120`
- by SSIM: `post40_lr03_120`
- by LPIPS: `post40_lr03_120`

## 5. 怎么解读这些结果

### 5.1 bounded schedule 确实能把 120iter 的 cliff 拉回来

和 `P2-I 120` 比：
- `post40_lr03_120`:
  - PSNR `+0.11576`
  - SSIM `+0.00355`
  - LPIPS `-0.00307`
- `post80_lr03_real05_120`:
  - PSNR `+0.10384`
  - SSIM `+0.00294`
  - LPIPS `-0.00229`

所以当前可以明确说：
- `80 -> 120` 的 cliff 不是不可救；
- 一个非常小的 late-stage schedule 改动就能把 `120iter` 从 regression 区拉回到明显正区间。

### 5.2 但它也没有把 StageB 变成“继续深调就一定值得”的局面

虽然 `post40_lr03_120` 是本轮最佳臂，但它相对 `P2-I 40` 的关系是：
- PSNR `+0.00784`
- SSIM `-0.00024`
- LPIPS `+0.00055`

也就是说：
- 它把 `120iter` 救回来了；
- 但并没有在所有指标上明显超出 `40iter` 这个更简单的窗口；
- 更像是“把后段崩坏抑制住”，而不是“找到了一个显著更强的新长程 regime”。

### 5.3 三个臂里最有信息量的判断

1. `post40_lr03_120` 最好：
   - 说明 current schedule 的失稳，确实可能在 `40` 之后就开始积累，只是到 `120` 才完全暴露；
   - 比起 `80` 后再补救，`40` 后先降速更有效。
2. `post80_lr03_120` 仍能明显优于原始 `120`，但不如 `post40_lr03_120`：
   - 说明只在 cliff 临近时出手，有点晚了。
3. `post80_lr03_real05_120` 比 `post80_lr03_120` 好不少，但仍略逊于 `post40_lr03_120`：
   - 说明 late-stage real/pseudo balance 的确是问题的一部分；
   - 但当前最强的一阶手段仍然是更早地收住 Gaussian update step。

## 6. 当前判断

1. `StageB` 不是完全不值得继续碰：bounded schedule compare 已经证明，late-stage cliff 可以被一个很小的调度改动显著缓解；
2. 但这个结果还不支持继续做无边界的 StageB 深调：
   - 当前最优臂只是把 `120` 救回到接近/略高于 `40` 的水平；
   - 它没有把 StageB 变成一个显著强于早期窗口的新主线。
3. 因此更合理的项目判断是：
   - `StageB` 现在保留一个可用的 bounded baseline（当前首选是 `post40_lr03_120`）；
   - 但主工程重心不应继续放在 StageB schedule 微调上；
   - 接下来应准备推进 `SPGM` 第一版落地。

## 7. 下一步建议

1. 若还要给 `StageB` 留一个参考配置，当前可先把：
   - `post40_lr03_120`
   当作 bounded StageB baseline / 对照臂；
2. 之后不要继续做开放式 StageB 网格；
3. 主线应转到 `docs/SPGM_landing_plan_for_part3_BRPO.md`，先做：
   - `StageA.5 + xyz-only + deterministic keep` 的 SPGM 第一版；
4. 等 SPGM 第一版跑通并完成最小 compare 后，再决定：
   - 是否把 SPGM 推进到 StageB；
   - 以及当前这个 bounded StageB baseline 是否还值得保留为对照组。

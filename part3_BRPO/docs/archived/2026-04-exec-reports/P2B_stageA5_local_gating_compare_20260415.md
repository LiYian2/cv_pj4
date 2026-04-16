# P2-B 8-frame StageA.5 gated vs ungated short compare（2026-04-15）

## 1. 目的

在已经完成 P2 第一版 local Gaussian gating 接入后，这一轮 compare 的目标不是再验证 wiring，而是回答三个更实的问题：
1. 在 8-frame `StageA.5` 上，gated 是否比 ungated 的 replay 更稳；
2. `grad_norm_xyz` 是否明显收缩；
3. `loss_depth` 是否没有被 gating 压没。

## 2. 固定口径

### 输入根
- base ply:
  - `/data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache/after_opt/point_cloud/point_cloud.ply`
- pseudo cache:
  - `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/pseudo_cache_baseline`
- StageA handoff init states:
  - `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_signal_compare_stageAonly_e1/stageA_legacy_80/pseudo_camera_states_stageA.json`
- replay internal cache root:
  - `/data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache`

### 统一参数
- `signal_pipeline=legacy`
- `stage_mode=stageA5`
- `stageA_iters=80`
- `num_pseudo_views=4`
- `stageA_rgb_mask_mode=train_mask`
- `stageA_depth_mask_mode=train_mask`
- `stageA_target_depth_mode=target_depth_for_refine_v2`
- `stageA_depth_loss_mode=source_aware`
- `stageA_lambda_abs_t=3.0`
- `stageA_lambda_abs_r=0.1`
- `stageA_abs_pose_robust=charbonnier`
- `stageA_abs_pose_scale_source=render_depth_trainmask_median`
- `stageA5_trainable_params=xyz`
- `stageA5_lr_xyz=1e-4`
- `seed=0`

## 3. 对照臂

### Arm-U：ungated xyz
输出目录：
- `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_p2b_stageA5_local_gating_compare_e1/stageA5_legacy_xyz_ungated_80`

### Arm-G：gated xyz
输出目录：
- `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_p2b_stageA5_local_gating_compare_e1/stageA5_legacy_xyz_gated_80`

额外参数：
- `pseudo_local_gating=hard_visible_union_signal`
- `pseudo_local_gating_params=xyz`
- `pseudo_local_gating_min_verified_ratio=0.01`
- `pseudo_local_gating_min_rgb_mask_ratio=0.01`
- `pseudo_local_gating_max_fallback_ratio=0.995`

## 4. replay 结果

part2 after_opt baseline（internal cache 自带）：
- PSNR = `23.94890858685529`
- SSIM = `0.8734854221343994`
- LPIPS = `0.0787798319839769`

### Arm-U：ungated
- replay PSNR = `23.851770499900535`
- replay SSIM = `0.8706381656505443`
- replay LPIPS = `0.08092637193147782`
- delta vs after_opt:
  - PSNR `-0.09713808695475379`
  - SSIM `-0.0028472564838550873`
  - LPIPS `+0.002146539947500928`

### Arm-G：gated
- replay PSNR = `23.852209585684317`
- replay SSIM = `0.8706391842276962`
- replay LPIPS = `0.08092523099923575`
- delta vs after_opt:
  - PSNR `-0.09669900117097185`
  - SSIM `-0.0028462379067032417`
  - LPIPS `+0.002145399015258856`

### gated vs ungated
- PSNR: gated `+0.000439085783782`
- SSIM: gated `+0.000001018577152`
- LPIPS: gated `-0.000001140932242`

结论：
- gating 没有带来明显级别的 replay 改善；
- 但它也没有把 replay 变差；
- 在这个 fixed-threshold 版本上，gated 相比 ungated 只有极轻微、几乎可忽略的 replay 优势。

## 5. loss / grad / gating summary

### Arm-U：ungated
- `loss_depth_last = 0.059482134878635406`
- `loss_depth_mean_last10 = 0.05720929810777307`
- `grad_norm_xyz_last = 0.16134321689605713`
- `grad_norm_xyz_mean_last10 = 0.11884148642420769`
- `grad_keep_ratio_xyz_last = 1.0`
- `true_pose_mean_trans = 0.00027645077032378894`
- `true_pose_mean_rotF = 0.0006259299875714242`

### Arm-G：gated
- `loss_depth_last = 0.05946585815399885`
- `loss_depth_mean_last10 = 0.057207750622183084`
- `grad_norm_xyz_last = 0.16231262683868408`
- `grad_norm_xyz_mean_last10 = 0.11905288267880679`
- `grad_keep_ratio_xyz_last = 0.8302343487739563`
- `visible_union_ratio_last = 0.8302343487739563`
- `true_pose_mean_trans = 0.0002745584346031686`
- `true_pose_mean_rotF = 0.0006347938829385439`

### gated 侧附加观察
- `iters_with_rejection = 0 / 80`
- `mean_accepted = 4.0`
- `mean_keep_ratio_xyz = 0.7246640816330909`
- `min_keep_ratio_xyz = 0.5240942239761353`
- `max_keep_ratio_xyz = 0.8307724595069885`

这说明：
- 这轮 gated compare 中，所有 sampled pseudo views 都通过了 signal gate，没有出现真正的 per-view rejection；
- 但 `visibility_filter union` 仍然把 active Gaussian subset 收到约 `52% ~ 83%`；
- 即便如此，`grad_norm_xyz` 并没有出现明显收缩，说明当前几何梯度主要集中在本来就可见的那部分 Gaussian 上。

## 6. 当前判断

1. 这轮 8-frame `StageA.5` compare 没有证明第一版 gating 已经带来显著 replay 改善。
2. 但它也证明了：
   - gating 不会自动把 depth loss 压空；
   - gating 也没有造成明显额外退化；
   - current threshold 下，因为所有 sampled pseudo views 都通过 signal gate，第一版 gating 的作用主要只剩 visible union，而不是 rejection。
3. 因此当前更合理的解释是：
   - 第一版 gating 的真正价值点还是“拒绝 weak pseudo views”；
   - 如果 compare 中根本没有 rejection，gating 自然很难带来强区分度。
4. 所以下一步 StageB compare 仍值得做，但应该明确把它当成：
   - 验证 real branch 打开后 pseudo-side gating 不会把全局纠偏一并裁掉；
   - 同时继续观察 gating 是否至少不比 baseline 更差。

## 7. 这轮之后的执行选择

基于 replay 指标，Arm-G 相比 Arm-U 略好且没有额外损失，因此后续 `StageB` short compare 先使用：
- `StageA.5 gated` 的输出 `refined_gaussians.ply`
- `StageA.5 gated` 的 `pseudo_camera_states_final.json`
作为共同起点，再做 `StageB ungated vs StageB gated` real-branch compare。

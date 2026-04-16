# P2-C real-branch StageB gated vs ungated short compare（2026-04-15）

## 1. 目的

在 P2-B 已经表明 `StageA.5 gated vs ungated` 只有极轻微区别之后，这一轮最关键的问题不再是“gating 会不会改善 replay”，而是：
1. 打开 real branch 后，pseudo-side local gating 会不会误伤 global correction；
2. 在 real branch 存在时，gated 是否至少不会比 ungated 更差；
3. 当前第一版 gating 是否已经有足够强的证据值得继续保留为默认路线。

## 2. 固定口径

### 共同起点
本轮 `StageB` 两个对照臂都从同一个 `StageA.5 gated` 输出起跑，以隔离 `StageB` gating 自身的影响：
- start ply:
  - `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_p2b_stageA5_local_gating_compare_e1/stageA5_legacy_xyz_gated_80/refined_gaussians.ply`
- init pseudo states:
  - `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_p2b_stageA5_local_gating_compare_e1/stageA5_legacy_xyz_gated_80/pseudo_camera_states_final.json`

### 其他固定输入
- pseudo cache:
  - `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/pseudo_cache_baseline`
- real branch manifest:
  - `/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/split_manifest.json`
- real branch rgb dir:
  - `/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/rgb`
- replay internal cache root:
  - `/data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache`

### 统一参数
- `signal_pipeline=legacy`
- `stage_mode=stageB`
- `stageA_iters=0`
- `stageB_iters=20`
- `num_pseudo_views=4`
- `num_real_views=2`
- `lambda_real=1.0`
- `lambda_pseudo=1.0`
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

### Arm-BU：StageB ungated
输出目录：
- `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_p2c_stageB_local_gating_compare_e1/stageB_from_stageA5gated_ungated_20`

### Arm-BG：StageB gated
输出目录：
- `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_p2c_stageB_local_gating_compare_e1/stageB_from_stageA5gated_gated_20`

额外参数：
- `pseudo_local_gating=hard_visible_union_signal`
- `pseudo_local_gating_params=xyz`
- `pseudo_local_gating_min_verified_ratio=0.01`
- `pseudo_local_gating_min_rgb_mask_ratio=0.01`
- `pseudo_local_gating_max_fallback_ratio=0.995`

## 4. replay 结果

part2 after_opt baseline：
- PSNR = `23.94890858685529`
- SSIM = `0.8734854221343994`
- LPIPS = `0.0787798319839769`

### Arm-BU：ungated
- replay PSNR = `23.902851705197936`
- replay SSIM = `0.8708849849524322`
- replay LPIPS = `0.08175702903557706`
- delta vs after_opt:
  - PSNR `-0.04605688165735344`
  - SSIM `-0.00260043718196723`
  - LPIPS `+0.0029771970516001695`

### Arm-BG：gated
- replay PSNR = `23.90280670589871`
- replay SSIM = `0.870884425993319`
- replay LPIPS = `0.08175759694918439`
- delta vs after_opt:
  - PSNR `-0.04610188095658074`
  - SSIM `-0.002600996141080447`
  - LPIPS `+0.002977764965207491`

### gated vs ungated
- PSNR: gated `-0.0000450`
- SSIM: gated `-0.00000056`
- LPIPS: gated `+0.00000057`

结论：
- real-branch StageB 下，gated 与 ungated 的 replay 差别也极小；
- 这次 gated 甚至比 ungated 略差一点，但量级仍然几乎可忽略；
- 因而 current threshold 的第一版 gating 还没有展现出明确可复用收益。

## 5. loss / grad / real-branch 观察

### Arm-BU：ungated
- `loss_total_last = 0.17194749414920807`
- `loss_real_last = 0.1479826420545578`
- `loss_pseudo_last = 0.02396484836935997`
- `loss_depth_last = 0.05203331355005503`
- `loss_depth_mean_last5 = 0.06068725474178791`
- `grad_norm_xyz_last = 0.08340691030025482`
- `grad_norm_xyz_mean_last5 = 0.09613939896225929`
- `true_pose_mean_trans = 0.0005945999592716588`
- `true_pose_mean_rotF = 0.0023801601701485607`

### Arm-BG：gated
- `loss_total_last = 0.17194876074790955`
- `loss_real_last = 0.14798256754875183`
- `loss_pseudo_last = 0.023966185748577118`
- `loss_depth_last = 0.05203594360500574`
- `loss_depth_mean_last5 = 0.06068743821233511`
- `grad_norm_xyz_last = 0.08340376615524292`
- `grad_norm_xyz_mean_last5 = 0.09621561467647552`
- `grad_keep_ratio_xyz_last = 0.8227609992027283`
- `visible_union_ratio_last = 0.8227609992027283`
- `true_pose_mean_trans = 0.000595148501887667`
- `true_pose_mean_rotF = 0.002380498310316282`

### gated 侧附加观察
- `iters_with_rejection = 0 / 20`
- `mean_keep_ratio_xyz = 0.7403085142374038`
- `min_keep_ratio_xyz = 0.5392801761627197`
- `max_keep_ratio_xyz = 0.8309518098831177`

这说明：
- real branch 打开后，gated 的 `loss_real` 与 ungated 几乎完全一致；
- 至少从这一轮短 compare 看，没有出现“pseudo-side gating 把 real branch global correction 一并裁掉”的明显证据；
- 但由于仍然没有发生任何 view-level rejection，这轮 compare 对“signal gate 真的帮忙了吗”这个问题仍然给不出强肯定答案。

## 6. 当前判断

1. `StageB` real branch 已经在当前第一版 gating 上完成真实 short compare，不再只是 no-real smoke。
2. 这轮结果支持一个保守判断：
   - pseudo-side local gating 不会明显破坏 real branch；
   - 但 current threshold 下，它也没有带来足够明确的 replay / loss 优势。
3. 当前最合理的设计解释是：
   - 真正需要发生的是 signal gate 对 weak pseudo view 的 rejection；
   - 如果 compare 中一直 0 rejection，那么第一版 gating 大部分时候只是一个 visibility-subset mask，而不是结构性筛选器。
4. 因而“是否默认保留这版 gating”目前仍不能下强结论；
   - 可以保留代码路径；
   - 但不应把当前 threshold 的 hard gating 直接宣称为默认 winner。

## 7. 下一步建议

如果继续做 P2 后续，我建议优先顺序是：
1. 不急着上 `xyz+opacity` 或 soft gating；
2. 先检查当前 signal gate 阈值为什么在 StageA.5 / StageB compare 中持续 `0 rejection`；
3. 再决定是：
   - 轻微上调 gate 阈值做新的 short compare，还是
   - 换到 `RGB-only v2` 分支上看 signal gate 是否更容易产生有意义的 rejection；
4. 只有当 signal gate 真能筛掉一部分 weak pseudo views 时，local gating 才更有可能体现出明显结构收益。

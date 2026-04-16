# P2-F：RGB-only v2 StageA.5 ungated vs branch-specific gated short compare（2026-04-16）

## 1. 目的

在 `P2-E` 已确认 legacy `StageA.5` 上继续细磨 threshold 的收益很低之后，这一轮转到 `RGB-only v2`，回答两个问题：
1. `RGB-only v2` 在会真实更新 Gaussian 的 `StageA.5` 上，是否比 legacy 更值得继续推进？
2. 在 `RGB-only v2` 自己的量纲上做 branch-specific gating 后，是否会带来额外收益？

## 2. 协议与输入

### 固定输入
- base ply:
  - `/data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache/after_opt/point_cloud/point_cloud.ply`
- pseudo cache:
  - `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/pseudo_cache_baseline`
- signal_v2 root:
  - `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/signal_v2`
- StageA handoff init states（RGB-only v2 自己的 StageA 输出）:
  - `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_signal_compare_stageAonly_e1/stageA_v2_rgbonly_80/pseudo_camera_states_stageA.json`
- replay internal cache root:
  - `/data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache`

### 统一参数
- `signal_pipeline=brpo_v2`
- `signal_v2_root=<above>`
- `stage_mode=stageA5`
- `stageA_iters=80`
- `num_pseudo_views=4`
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

### 输出路径
本轮实验已按用户要求写到 `/data2`：
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1`

## 3. 对照臂

### Arm-U：RGB-only v2 ungated
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_ungated_80`

### Arm-G：RGB-only v2 gated
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_gated_rgb0192_80`

额外 gate 参数：
- `pseudo_local_gating=hard_visible_union_signal`
- `pseudo_local_gating_params=xyz`
- `pseudo_local_gating_min_verified_ratio=0.01`
- `pseudo_local_gating_min_rgb_mask_ratio=0.0192`
- `pseudo_local_gating_max_fallback_ratio=0.995`

这个 `0.0192` 不是随意拍的：基于 `P1A` 的 `RGB-only v2` 8-frame signal 统计，它会稳定拒掉最弱的两档 pseudo（`225` / `260`），但不会像 `0.02` 那样一下拒掉 `6/8`。

## 4. replay 结果

part2 after_opt baseline（internal cache 自带）：
- PSNR = `23.94890858685529`
- SSIM = `0.8734854221343994`
- LPIPS = `0.0787798319839769`

### Arm-U：ungated
- replay PSNR = `23.92322131262885`
- replay SSIM = `0.8722803906158164`
- replay LPIPS = `0.07994902253978782`
- delta vs after_opt:
  - PSNR `-0.02568727422643846`
  - SSIM `-0.0012050315185829774`
  - LPIPS `+0.0011691905558109256`

### Arm-G：gated rgb=0.0192
- replay PSNR = `23.924801070601852`
- replay SSIM = `0.8723195815527881`
- replay LPIPS = `0.07993138824348096`
- delta vs after_opt:
  - PSNR `-0.02410751625343721`
  - SSIM `-0.0011658405816112882`
  - LPIPS `+0.001151556259504069`

### gated vs ungated
- PSNR: gated `+0.00157975797300125`
- SSIM: gated `+3.919093697168918e-05`
- LPIPS: gated `-1.7634296306856534e-05`

结论：
- `RGB-only v2` 上的 branch-specific gating 是正向的，而且比 legacy 上的 gating gain 更可见；
- 但它仍然不是大幅提升，更多是“方向正确的小增益”。

## 5. 和 legacy StageA.5 的关系（谨慎解释）

与之前 legacy `StageA.5` 参考臂相比：
- `v2 ungated - legacy ungated`:
  - PSNR `+0.07145081272831533`
  - SSIM `+0.00164222496527211`
  - LPIPS `-0.0009773493916900022`
- `v2 gated - legacy gated`:
  - PSNR `+0.07259148491753464`
  - SSIM `+0.0016803973250919535`
  - LPIPS `-0.0009938427557547869`

这个差值很大，说明：在 `StageA.5` 这个会真实更新 Gaussian 的阶段，`RGB-only v2` 这条支路明显比当前 legacy 更值得保留。

但要注意一个 caveat：
- 这里每个分支都接的是**自己对应的 StageA handoff states**；
- 所以这个对比更像“各分支自然 StageA→A.5 流程下谁更好”，而不是把一切中间状态强行固定后的纯局部 ablation。

即便如此，这个结果仍然足够强，至少可以支持：`RGB-only v2` 不是只在 StageA-only 上“看起来不坏”，而是在真正改高斯图的 `StageA.5` 上也已经展现出明显更好的 replay 区间。

## 6. gating summary / loss / pose

### Arm-U：ungated
- `mean_confidence_nonzero_ratio = 0.01958179473876953`
- `mean_target_depth_verified_ratio = 0.04133939743041992`
- `iters_with_rejection = 0 / 80`
- `loss_depth_last = 0.062414358370006084`
- `grad_keep_ratio_xyz_mean = 1.0`
- `true_pose_mean_trans = 0.00023091466827484077`
- `true_pose_mean_rotF = 0.0007192542076189835`

### Arm-G：gated rgb=0.0192
- `iters_with_rejection = 59 / 80`
- `total_rejected_sample_evals = 74 / 320`
- `rejected_ids = {225, 260}`
- `rejected_reason = rgb_mask_ratio`
- `grad_keep_ratio_xyz_mean = 0.7201281029731035`
- `grad_keep_ratio_xyz_last = 0.8233887553215027`
- `loss_depth_last = 0.06254712678492069`
- `true_pose_mean_trans = 0.00023718911615531362`
- `true_pose_mean_rotF = 0.0007850109908322927`

这说明：
- `RGB-only v2` 的 branch-specific gate 已经不再是 no-op，而是真正按 RGB 量纲在筛样本；
- 并且它不是把系统掐死：depth loss 仍正常存在，replay 反而略好；
- 所以从机制上说，这条线已经比 legacy 上“调进 reject 区但没什么变化”的状态更有推进价值。

## 7. 现在是否要做 RGB densify？

我的判断是：**现在还不应该直接做 RGB mask densify。**

原因有三层：
1. 当前 `RGB-only v2` 虽然 raw mask 看起来点状/seed-like，但它在 `StageA.5` 上已经明显优于 legacy；这说明“point-like raw support”本身还不足以构成必须立刻 densify 的证据。
2. 现在 raw RGB confidence 的作用更像高精度 correspondence evidence，而不是最终要被无约束铺开的训练区域。如果现在直接把 raw RGB mask 做 morphology-style 扩张，很容易把它重新变回旧 `train_mask` 语义，反而丢掉 `v2` 的主要价值。
3. 真正仍然显著偏窄的，是 `full v2 depth` 那条线，而不是 `RGB-only v2 + legacy depth target` 这条线。也就是说，如果后面要做“扩张”，更合理的对象应该是：
   - `depth supervision v2` 的 support expand / densify
   - 或者一个受几何约束的 downstream support expansion
   而不是先把 raw RGB correspondence map 直接糊开。

更直接地说：
- `RGB-only v2` 已经先证明了“稀疏但语义更准”的支路是值得保留的；
- 因此现在不该因为它看起来像点，就立刻去做 RGB densify；
- 如果后面要扩张，也应该是 controlled expansion，而不是无约束 diffusion。

## 8. 当前判断

1. `RGB-only v2` 已经从“StageA-only 看起来不坏”推进到“StageA.5 replay 明显优于 legacy”；
2. branch-specific gating（`min_rgb_mask_ratio=0.0192`）在这条支路上确实带来小幅正增益；
3. 因而当前后续最值得推进的是：
   - 保留 `RGB-only v2` 作为下一阶段主候选；
   - 若继续推进 gating，就围绕这条支路做更小心的 branch-specific compare；
   - 不急着做 raw RGB densify。
4. 如果后面要做“扩张”，优先考虑的是受几何约束的 support/depth expand，而不是对 raw RGB correspondence mask 直接做形态学式铺开。

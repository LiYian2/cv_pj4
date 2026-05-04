# M3D_RGB_ONLY_STAGEB120_REPLAY_COMPARE_20260424.md

> 更新时间：2026-04-24 21:41 (Asia/Shanghai)

## 1. 目的

验证当前 exact-upstream consumer 若完全不使用 pseudo depth loss、只保留 RGB supervision，是否能解释并修复 dense3d 路线的 replay regression。

这次实验只做 consumer-side ablation：
- 不重建 mask / exact backend / signal_v2
- 直接复用已经裁剪为 mainline-only 的 `sparse_signal` 与 `dense3d_q070_signal`
- 仅在 refine 侧增加 `--stageA_disable_depth`

因此它回答的是：在当前 fixed route 下，去掉 pseudo depth loss 本身，是否能改善结果。

## 2. 协议

- compare root: `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260424_m3d_rgb_only_compare_stageB120_replay`
- common anchor:
  - `/data/bzhang512/CV_Project/output/part3_BRPO/experiments/20260415_p2b_stageA5_local_gating_compare_e1/stageA5_legacy_xyz_gated_80/refined_gaussians.ply`
  - `/data/bzhang512/CV_Project/output/part3_BRPO/experiments/20260415_p2b_stageA5_local_gating_compare_e1/stageA5_legacy_xyz_gated_80/pseudo_camera_states_final.json`
- pseudo cache: `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/pseudo_cache_baseline`
- train manifest: `/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/split_manifest.json`
- StageB protocol:
  - `stage_mode=stageB`
  - `stageA_iters=0`
  - `stageB_iters=120`
  - `stageB_post_switch_iter=40`
  - `stageB_post_lr_scale_xyz=0.3`
  - `joint_topology_mode=brpo_joint_v1`
  - `pseudo_observation_mode=exact_brpo_upstream_target_v1`
  - `stageA_depth_loss_mode=exact_shared_cm_v1`
  - `lambda_real=1.0`
  - `lambda_pseudo=1.0`
  - `num_real_views=2`
  - `num_pseudo_views=4`
  - clean summary-only SPGM control 保持不变

唯一变量：
- current arms：默认 RGB + depth
- rgb_only arms：额外加 `--stageA_disable_depth`

## 3. Arms

1. `sparse_current`
   - signal root: `sparse_signal`
   - 默认 exact shared RGB+depth
2. `sparse_rgb_only`
   - signal root: `sparse_signal`
   - 同协议，仅加 `--stageA_disable_depth`
3. `q070_current`
   - signal root: `dense3d_q070_signal`
   - 默认 exact shared RGB+depth
4. `q070_rgb_only`
   - signal root: `dense3d_q070_signal`
   - 同协议，仅加 `--stageA_disable_depth`

## 4. Smoke 确认

在正式 compare 前，先跑了 1-iter smoke：
- output: `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260424_rgb_only_smoke_q070_stageb1`
- 结果确认：
  - `stageA_disable_depth=true`
  - StageB iter 1 日志中 `depth=0.0000`

说明 no-depth contract 已真实被 consumer 吃到，不是空参数。

## 5. Replay 结果

- anchor replay: `PSNR 23.85221`, `SSIM 0.87064`, `LPIPS 0.08093`
- `sparse_current`: `PSNR 24.00351`, `SSIM 0.87177`, `LPIPS 0.08247`
- `sparse_rgb_only`: `PSNR 23.92435`, `SSIM 0.87166`, `LPIPS 0.08221`
- `q070_current`: `PSNR 23.66595`, `SSIM 0.86585`, `LPIPS 0.08577`
- `q070_rgb_only`: `PSNR 23.66449`, `SSIM 0.86666`, `LPIPS 0.08500`

关键差分：
- `sparse_rgb_only - sparse_current`
  - `PSNR -0.07916`
  - `SSIM -0.00011`
  - `LPIPS -0.00026`
- `q070_rgb_only - q070_current`
  - `PSNR -0.00146`
  - `SSIM +0.00082`
  - `LPIPS -0.00077`
- `q070_current - sparse_current`
  - `PSNR -0.33756`
- `q070_rgb_only - sparse_rgb_only`
  - `PSNR -0.25986`

## 6. 训练侧结果

- `sparse_current`
  - `loss_total_last 0.16893`
  - `loss_pseudo_last 0.04658`
  - `loss_rgb_last 0.00785`
  - `loss_depth_last 0.04108`
- `sparse_rgb_only`
  - `loss_total_last 0.12360`
  - `loss_pseudo_last 0.00362`
  - `loss_rgb_last 0.00517`
  - `loss_depth_last 0.0`
- `q070_current`
  - `loss_total_last 0.17211`
  - `loss_pseudo_last 0.05186`
  - `loss_rgb_last 0.00948`
  - `loss_depth_last 0.04523`
- `q070_rgb_only`
  - `loss_total_last 0.12381`
  - `loss_pseudo_last 0.00430`
  - `loss_rgb_last 0.00614`
  - `loss_depth_last 0.0`

这说明去掉 depth 后，train-side pseudo/depth totals 会自然下降；但 replay 并没有因此统一改善。

## 7. 结论

结论很直接：当前 dense3d regression 不能简单归因于 pseudo depth loss 本身。

更细一点说：
- 对 sparse 而言，depth supervision 是有帮助的；全局关掉 depth 后 replay 明显变差（`-0.079 PSNR`）。
- 对 q070 dense3d 而言，全局关掉 depth 并没有把它救回来；只带来了极小的表面变化（PSNR 基本不动，SSIM/LPIPS 略好），但它仍明显落后 sparse。
- 因此，“depth 质量差”至多是 dense 路线中的次级因素，不是当前 replay gap 的主解释。

当前更合理的解释仍是：dense3d 的主要问题在 supervision composition / support quality，尤其是大量 single-branch 新增区域如何进入 exact shared contract，而不是 depth loss 这一项单独坏掉。

## 8. 下一步更值得测的方向

如果继续做 loss contract 排查，更值得做的是局部合同 ablation，而不是全局关掉 depth：
1. only-both-depth：只让 both-support 区域进 depth，single-support 区域不进 depth
2. single-depth-suppressed：保留 both-depth，但压掉 single-branch depth supervision
3. single/both 分开记账：把 replay 变化与 single/both 质量组成明确对应起来

在当前 fixed route 下，`rgb_only` 不是更优主线。

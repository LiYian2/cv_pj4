# StageB 执行简报（Phase 3）

时间：2026-04-13

## 1) 运行配置
- run: `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_stageB_phase3_longrun/stageB300_conservative_xyz_opacity`
- mode: `stageB`
- iters: `300`
- warm start: A.5 xyz+opacity refined ply
- gaussian trainable: `xyz_opacity`
- num_pseudo_views=4, num_real_views=2
- lambda_real=1.0, lambda_pseudo=1.0

## 2) 训练末10均值
- loss_total: 0.103643
- loss_real: 0.097858
- loss_pseudo: 0.005785
- loss_rgb: 0.005452
- loss_depth: 0.006548
- grad_norm_xyz: 0.079770

## 3) Replay 对照（vs A.5 baseline）
A.5: PSNR 23.970262, SSIM 0.873834, LPIPS 0.078717

StageB300: PSNR 23.790987, SSIM 0.868885, LPIPS 0.081147

Delta(StageB300 - A.5):
- PSNR: -0.179275
- SSIM: -0.004948
- LPIPS: +0.002430

Gate(non-regression): FAIL

## 4) 产物
- replay json: `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_stageB_phase3_longrun/stageB300_conservative_xyz_opacity/replay/stageB300_conservative_xyz_opacity/replay_eval.json`
- summary json: `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_stageB_phase3_longrun/stageB300_conservative_xyz_opacity/summary_phase3_stageB300_vs_a5.json`

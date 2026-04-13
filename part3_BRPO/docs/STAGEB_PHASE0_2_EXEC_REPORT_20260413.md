# StageB 执行简报（Phase 0/1/2）

时间：2026-04-13

## 0) 执行目标
按 `docs/STAGEB_CONSERVATIVE_ENTRY_PLAN.md` 先完成：
- Phase 0（预检+基线冻结）
- Phase 1（StageB conservative 最小可跑）
- Phase 2（短程 replay gate）

## 1) Phase 0 简报（预检+基线冻结）

冻结基线（A.5 xyz_opacity）路径：
- warm-start PLY：
  `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_stageA5_midpoint_300iter_withply/stageA5_xyz_opacity/refined_gaussians.ply`
- baseline replay：
  `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_stageA5_midpoint_300iter_withply/replay/stageA5_xyz_opacity_300/replay_eval.json`
- pseudo cache（midpoint 8帧）：
  `/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_prepare/re10k1__internal_afteropt__midpoint_proto_v1/pseudo_cache`

检查结论：
- manifest: `num_samples=8`, `sample_ids=[17,51,86,121,156,190,225,260]`
- 每个 sample 均具备 `confidence_mask_brpo.npy` 与 `target_depth_for_refine.npy`
- depth finite=1.0，mask ratio 约 0.147~0.220，满足最小可跑要求

## 2) Phase 1 简报（StageB conservative 最小可跑）

实现变更：
- 文件：`scripts/run_pseudo_refinement_v2.py`
- 新增 `--stage_mode stageB` 及参数：
  - `--stageB_iters`
  - `--train_manifest`, `--train_rgb_dir`
  - `--lambda_real`, `--lambda_pseudo`, `--num_real_views`
- StageB 逻辑：
  - pseudo 分支：沿用 StageA RGBD masked loss + pose/exposure 正则
  - real 分支：加载 sparse train 视角，加入 real anchor RGB mapping loss
  - joint optimize：pseudo camera params + micro gaussian params（xyz/xyz_opacity）

本轮运行：
- 输出目录：
  `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_stageB_conservative_gate/phase1_stageB120`
- 关键设置：
  - `stage_mode=stageB`
  - `stageA_iters=0`（直接从 A.5 warm start 进入 StageB）
  - `stageB_iters=120`
  - `train_manifest=/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/split_manifest.json`
  - `num_real_views=2`, `num_pseudo_views=4`
  - `trainable_params=xyz_opacity`, `lr_xyz=1e-4`, `lr_opacity=5e-4`

稳定性结果（last10 mean）：
- loss_total: 0.11545
- loss_real: 0.11050
- loss_pseudo: 0.00495
- loss_rgb: 0.00453
- loss_depth: 0.00593
- grad_norm_xyz: 0.07560

结论：
- 无发散，中短程可稳定运行。

## 3) Phase 2 简报（短程 replay gate）

StageB replay：
- `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_stageB_conservative_gate/phase1_stageB120/replay/stageB_conservative_120/replay_eval.json`

A.5 baseline replay：
- `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_stageA5_midpoint_300iter_withply/replay/stageA5_xyz_opacity_300/replay_eval.json`

对比（StageB120 - A.5）：
- PSNR: +0.29383
- SSIM: +0.004369
- LPIPS: -0.0002153

Gate 判定（非退化条件：PSNR/SSIM不降且LPIPS不升）：
- `PASS`

## 4) 产物清单
- StageB run目录：
  `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_stageB_conservative_gate/phase1_stageB120`
- 关键文件：
  - `stageA_history.json`
  - `stageB_history.json`
  - `refinement_history.json`
  - `refined_gaussians.ply`
  - `replay/stageB_conservative_120/replay_eval.json`
  - `summary_phase0_1_2.json`

## 5) 结论
Phase 0/1/2 已按计划执行完成，且 replay gate 通过。
可进入 Phase 3（长程 StageB + 逐步放开参数组）。

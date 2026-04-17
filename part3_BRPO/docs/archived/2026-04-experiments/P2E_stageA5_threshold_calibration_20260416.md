# P2-E：legacy StageA.5 signal gate threshold calibration（2026-04-16）

## 1. 目的

在 `P2-D` 已确认 current threshold 过松之后，这一轮不再问“gate 能不能接通”，而是只回答一个更具体的问题：

- 当 legacy `StageA.5` 的 hard gate 真正进入 non-zero rejection 区后，replay / loss / pose 会不会出现可解释收益？

## 2. 协议与输入

### 固定输入
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

### 说明：输出路径改到 home tmp
本轮没有继续写到 canonical `output/part3_BRPO/experiments/...`，而是写到：
- `/home/bzhang512/tmp_part3_brpo_outputs/20260416_p2e_stageA5_threshold_calibration_e1`

原因不是协议改变，而是 `output/` 实际落在 `/data`，当时 `/data` 已满，无法再创建新实验目录；本轮只是把 run root 临时放到了 `/home` 可写路径，输入协议与 compare 口径保持不变。

## 3. 对照臂

已有参考臂：
1. ungated（旧 P2-B）
   - `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_p2b_stageA5_local_gating_compare_e1/stageA5_legacy_xyz_ungated_80`
2. gate vr=0.01（旧 P2-B current threshold）
   - `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_p2b_stageA5_local_gating_compare_e1/stageA5_legacy_xyz_gated_80`

本轮新增：
3. gate vr=0.02
   - `/home/bzhang512/tmp_part3_brpo_outputs/20260416_p2e_stageA5_threshold_calibration_e1/stageA5_legacy_xyz_gatevr002_80`
4. gate vr=0.03
   - `/home/bzhang512/tmp_part3_brpo_outputs/20260416_p2e_stageA5_threshold_calibration_e1/stageA5_legacy_xyz_gatevr003_80`

统一 gate 设置：
- `pseudo_local_gating=hard_visible_union_signal`
- `pseudo_local_gating_params=xyz`
- `pseudo_local_gating_min_rgb_mask_ratio=0.01`
- `pseudo_local_gating_max_fallback_ratio=0.995`
- `pseudo_local_gating_min_correction=0.0`

## 4. replay 结果

| arm | replay PSNR | replay SSIM | replay LPIPS | delta PSNR vs after_opt | delta SSIM | delta LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| ungated | 23.85177050 | 0.87063817 | 0.08092637 | -0.09713809 | -0.00284726 | +0.00214654 |
| gate vr=0.01 | 23.85220959 | 0.87063918 | 0.08092523 | -0.09669900 | -0.00284624 | +0.00214540 |
| gate vr=0.02 | 23.85247908 | 0.87065239 | 0.08092866 | -0.09642951 | -0.00283303 | +0.00214883 |
| gate vr=0.03 | 23.85223916 | 0.87065089 | 0.08093010 | -0.09666942 | -0.00283453 | +0.00215027 |

与 `gate vr=0.01` 对比：
- `vr=0.02`: PSNR `+0.00026949`，SSIM `+1.321e-05`，LPIPS `+3.434e-06`（更差）
- `vr=0.03`: PSNR `+2.958e-05`，SSIM `+1.171e-05`，LPIPS `+4.866e-06`（更差）

结论：
- `vr=0.02 / 0.03` 都把 replay 指标推到了和旧 `vr=0.01` 几乎重合的区域；
- 其中 `vr=0.02` 是本轮 PSNR / SSIM 最好的一臂，但幅度仍然极小；
- `vr=0.03` 没有带来额外收益，反而把 LPIPS 再推高了一点。

## 5. gating summary / loss / pose

| arm | accepted mean | iters with rejection | total rejected sample evals | rejected ids | reason | keep ratio mean | loss_depth_last |
|---|---:|---:|---:|---|---|---:|---:|
| ungated | 4.000 | 0 / 80 | 0 / 320 | - | - | 1.0000 | 0.05948213 |
| gate vr=0.01 | 4.000 | 0 / 80 | 0 / 320 | - | - | 0.7247 | 0.05946586 |
| gate vr=0.02 | 3.5375 | 37 / 80 | 37 / 320 | `260` | `verified_ratio` | 0.7204 | 0.05963119 |
| gate vr=0.03 | 3.0750 | 59 / 80 | 74 / 320 | `225, 260` | `verified_ratio` | 0.7202 | 0.05970583 |

补充观察：
- `vr=0.02` 如预期开始只拒掉最弱那一档 pseudo（`sample_id=260`）；
- `vr=0.03` 则进一步稳定拒掉 `225 + 260`；
- 所有 rejection 都只来自 `verified_ratio`，而不是 `rgb_mask_ratio` / `fallback_ratio`；
- 但即便 rejection 数已经明显非零，`grad_keep_ratio_xyz` 的均值只从 `0.7247` 轻微降到 `0.7204 / 0.7202`，变化非常小。

这意味着：
- 第一轮 calibration 已经证明 gate 确实进入了真正的 reject 区；
- 但在 legacy `StageA.5` 上，单纯把最弱 1~2 个 pseudo view 剔掉，并没有显著改变 map-side active Gaussian subset 的整体 footprint；
- 因此 replay 也自然没有出现明显级别的改善。

## 6. 当前判断

1. `P2-D` 的诊断被实跑验证了：`min_verified_ratio=0.02 / 0.03` 的确能把 gate 从 `0 rejection` 拉进有效工作区。
2. 但这一步也同时说明：**legacy `StageA.5` 的瓶颈已经不再是“threshold 太松导致完全不 reject”，而是“即便开始 reject 最弱 pseudo，replay 仍几乎不变”。**
3. 如果只需要一个“已进入有效 reject 区的 legacy calibrated reference”，本轮更合理的候选是 `vr=0.02`：
   - 它已经开始稳定拒掉最弱 pseudo；
   - 比 `vr=0.03` 更保守；
   - replay 也没有比 `vr=0.03` 更差。
4. 但更大的工程结论是：**继续在 legacy `StageA.5` 上细磨 threshold 的边际收益已经很低。** 当前更值得做的，不是把 `0.02 / 0.03 / 0.025 / 0.028` 再细扫一遍，而是把同一套 gating 逻辑转到更有语义差异的分支上做 branch-specific calibration（优先 `RGB-only v2`），或者只把 `vr=0.02` 作为 legacy 参考臂保留。

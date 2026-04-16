# P0 abs prior 标定 + P1A StageA-only signal compare（2026-04-15）

## 1. 这次做了什么

本轮实际完成了两件事：
1. 在 E1 winner `signal-aware-8` canonical root 上完成一轮 `legacy` 背景下的 split abs prior 标定；
2. 在固定 abs prior 背景后，完成一轮 `StageA-only` 的 `legacy / v2-rgb-only / v2-full` 信号短对照。

注意：这轮不是 StageA.5 / StageB 对照，而是纯 `stage_mode=stageA`。因此它主要回答：
- abs prior 是否进入有效区间；
- signal 分支改变后，mask / depth / drift 指标怎么变；
- full v2 是否因为 coverage 过窄而变成“更纯但太弱”。

它不能回答 replay-on-PLY 的优劣，因为当前代码下纯 `StageA` 不会更新 Gaussian。

---

## 2. 实验根与输入固定口径

### P0 run root
`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_absprior_calibration_legacy_e1`

### P1A run root
`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_signal_compare_stageAonly_e1`

### canonical pseudo set
E1 winner `signal-aware-8`：
`[23, 57, 92, 127, 162, 196, 225, 260]`

### prepare root
`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare`

### StageA canonical cache used in this run
`pseudo_cache_baseline/`

原因：该 root 中存在 `target_depth_for_refine_v2.npy` 与 `target_depth_dense_source_map.npy`，而旧 `pseudo_cache/` 只保留 M3 层 target，不足以承载本轮 P0 的 `target_depth_for_refine_v2 + source_aware` 口径。

---

## 3. P0：split abs prior 标定

## 3.1 背景

本轮固定：
- `signal_pipeline=legacy`
- `stageA_rgb_mask_mode=train_mask`
- `stageA_depth_mask_mode=train_mask`
- `stageA_target_depth_mode=target_depth_for_refine_v2`
- `stageA_depth_loss_mode=source_aware`
- `num_pseudo_views=4`
- `stageA_iters=80`

扫描的组合：
- noabs = `(0.0, 0.0)`
- `(1.5, 0.1)`
- `(3.0, 0.1)`
- `(6.0, 0.1)`
- `(3.0, 0.05)`
- `(3.0, 0.2)`

其中 tuple 表示 `(lambda_abs_t, lambda_abs_r)`。

## 3.2 关键结果

### noabs
- `loss_depth_last ≈ 0.07838`
- `abs_pose_rho_norm_last ≈ 0.00648`
- `abs_pose_theta_norm_last ≈ 0.00094`
- `final_mean_trans_norm ≈ 0.00380`
- `final_mean_rot_fro_norm ≈ 0.00114`

### `(1.5, 0.1)`
- `loss_depth_last ≈ 0.07718`
- `loss_abs_pose_trans_last ≈ 0.00174`
- `loss_abs_pose_rot_last ≈ 0.000116`
- `final_mean_trans_norm ≈ 0.000704`
- `final_mean_rot_fro_norm ≈ 0.000631`

### `(3.0, 0.1)`
- `loss_depth_last ≈ 0.07735`
- `loss_abs_pose_trans_last ≈ 0.00312`
- `loss_abs_pose_rot_last ≈ 0.000115`
- `abs_pose_rho_norm_last ≈ 0.000471`
- `abs_pose_theta_norm_last ≈ 0.000547`
- `final_mean_trans_norm ≈ 0.000407`
- `final_mean_rot_fro_norm ≈ 0.000527`

### `(6.0, 0.1)`
- `loss_depth_last ≈ 0.07738`
- `final_mean_trans_norm ≈ 0.000265`
- `final_mean_rot_fro_norm ≈ 0.000592`

### `(3.0, 0.05)`
- `loss_depth_last ≈ 0.07746`
- `loss_abs_pose_rot_last ≈ 0.0000583`
- `final_mean_trans_norm ≈ 0.000392`
- `final_mean_rot_fro_norm ≈ 0.000601`

### `(3.0, 0.2)`
- `loss_depth_last ≈ 0.07725`
- `loss_abs_pose_rot_last ≈ 0.000226`
- `final_mean_trans_norm ≈ 0.000408`
- `final_mean_rot_fro_norm ≈ 0.000526`

## 3.3 梯度量级判断

iter1 的 `grad_contrib` 显示：
- `depth_total`: `grad_norm_trans ≈ 1.92`, `grad_norm_rot ≈ 7.21`
- 当 `(lambda_abs_t, lambda_abs_r)=(3.0, 0.1)` 时：
  - `abs_pose_trans`: `grad_norm_trans ≈ 0.225`
  - `abs_pose_rot`: `grad_norm_rot ≈ 0.049`

这说明：
- abs prior 已经进入有效区间；
- 但没有强到压倒 depth 分支一个数量级以上；
- 因而它适合作为后续 compare 的固定背景配置。

## 3.4 P0 结论

当前最合适的固定背景配置是：
- `lambda_abs_t = 3.0`
- `lambda_abs_r = 0.1`
- `abs_pose_robust = charbonnier`
- `stageA_abs_pose_scale_source = render_depth_trainmask_median`

原因：
- 相比 noabs，它能显著收住 drift；
- 相比 `(1.5, 0.1)`，作用更稳定；
- 相比 `(6.0, 0.1)`，没有额外收益到值得把平移 prior 再加重；
- 在当前梯度量级下，它更像“有效稳定器”，不是“纯压制器”。

---

## 4. P0 的结构性修正：纯 StageA replay 不是有效比较指标

这次 P0 还确认了一个必须写进设计判断的事实：

当前 `run_pseudo_refinement_v2.py` 的纯 `stage_mode=stageA` 不会更新 Gaussian，只更新：
- pseudo camera residual
- exposure

实测已用 `sha256sum` 验证：
- `BASE_PLY`
- `stageA_noabs_80/refined_gaussians.ply`
- `stageA_abs_t3_r0p1_80/refined_gaussians.ply`

三者 hash 完全一致。

因此：
- 纯 `StageA` 上的 replay-on-PLY 只能作为 identity sanity check；
- 它不能作为 abs prior 或 signal branch 的主比较指标；
- 如果后续想用 replay 判断优劣，必须进入会真实更新 Gaussian 的 stage，例如 `StageA.5 / StageB`。

---

## 5. P1A：StageA-only signal compare

## 5.1 为什么先做 P1A

虽然纯 `StageA` 的 replay 无信息，但它仍然可以回答：
- signal coverage / verified ratio 怎么变；
- depth seed / dense 分支在不同 signal branch 下的损失表现；
- 在固定 abs prior 背景下，full v2 是否已经弱到不可消费。

因此本轮先做一个 `StageA-only` 的三臂短对照，作为信号层诊断，而不是最终 replay 验收。

## 5.2 前置工作

在 canonical E1 root 上重新补齐新 fusion，并成功生成 8-frame `signal_v2`：
`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/signal_v2`

本轮对照统一固定：
- pseudo set = E1 winner `signal-aware-8`
- `stageA_iters=80`
- `num_pseudo_views=4`
- abs prior = `(3.0, 0.1)`
- `stageA_depth_loss_mode=source_aware`

## 5.3 三臂定义

### Arm-L：legacy
- `signal_pipeline=legacy`
- `stageA_rgb_mask_mode=train_mask`
- `stageA_depth_mask_mode=train_mask`
- `stageA_target_depth_mode=target_depth_for_refine_v2`

### Arm-V1：v2 RGB-only
- `signal_pipeline=brpo_v2`
- `stageA_rgb_mask_mode=brpo_v2_raw`
- `stageA_depth_mask_mode=train_mask`
- `stageA_target_depth_mode=target_depth_for_refine_v2`

### Arm-V2：full v2
- `signal_pipeline=brpo_v2`
- `stageA_rgb_mask_mode=brpo_v2_raw`
- `stageA_depth_mask_mode=brpo_v2_depth`
- `stageA_target_depth_mode=brpo_v2`

## 5.4 核心结果

### Arm-L：legacy
- `mean_confidence_nonzero_ratio ≈ 0.18869`
- `mean_target_depth_verified_ratio ≈ 0.04134`
- `mean_target_depth_render_fallback_ratio ≈ 0.95866`
- `loss_depth_last ≈ 0.07737`
- `loss_depth_seed_last ≈ 0.06404`
- `loss_depth_dense_last ≈ 0.03807`
- `final_mean_trans_norm ≈ 0.000406`
- `final_mean_rot_fro_norm ≈ 0.000527`

### Arm-V1：v2 RGB-only
- `mean_confidence_nonzero_ratio ≈ 0.01958`
- `mean_target_depth_verified_ratio ≈ 0.04134`
- `mean_target_depth_render_fallback_ratio ≈ 0.95866`
- `loss_depth_last ≈ 0.07750`
- `loss_depth_seed_last ≈ 0.06420`
- `loss_depth_dense_last ≈ 0.03798`
- `final_mean_trans_norm ≈ 0.000386`
- `final_mean_rot_fro_norm ≈ 0.000748`

### Arm-V2：full v2
- `mean_confidence_nonzero_ratio ≈ 0.01958`
- `mean_target_depth_verified_ratio ≈ 0.01957`
- `mean_target_depth_render_fallback_ratio ≈ 0.00001`
- `loss_depth_last ≈ 0.11127`
- `loss_depth_seed_last ≈ 0.11093`
- `loss_depth_dense_last ≈ 0.00098`
- `final_mean_trans_norm ≈ 0.000321`
- `final_mean_rot_fro_norm ≈ 0.000693`

## 5.5 P1A 结论

1. `v2 RGB-only` 不是明显坏方向。
   - 它把 RGB mask coverage 从 `18.9%` 收到 `~1.96%`，但 depth target 仍沿用 legacy M5 target，因此 depth 侧表现总体仍接近 legacy。
   - 这说明“RGB 语义重排”本身不是当前的主要问题。

2. `full v2` 当前确实过窄。
   - verified ratio 从 `~4.13%` 直接降到 `~1.96%`；
   - `loss_depth_dense_last` 也几乎掉到 0；
   - 它更像“更纯但太弱”的 supervision，而不是当前可直接替代 legacy 的默认主线。

3. 因此当前若保留 v2，优先保留的是：
   - `RGB-only v2` 作为 signal semantics probe；
   - 而不是直接把 `full v2 depth` 当默认训练分支。

---

## 6. 当前总判断

本轮结束后，当前最稳妥的项目判断是：

1. abs prior 先固定成 `(lambda_abs_t, lambda_abs_r) = (3.0, 0.1)`；
2. 当前纯 `StageA` 不更新 Gaussian，replay-on-PLY 不具辨别力；
3. `v2` 的 full depth 分支现在仍然太窄；
4. 如果继续推进 v2，应该优先保留 `RGB-only` 入口，而不是直接让 full v2 depth 接管；
5. 后续主线优先级应转到 `P2 local Gaussian gating`，并把真正依赖 replay 的比较移动到 `StageA.5` 或其他会真实更新 Gaussian 的 stage 上。

---

## 7. 下一步推荐

下一步不再继续围绕 full v2 depth 做高强度细调，而是：

1. 以 `abs prior = (3.0, 0.1)` 作为固定背景；
2. 开始实现 `local Gaussian gating` 第一版；
3. 第一版范围固定成：
   - `StageA.5`
   - pseudo-side only
   - `xyz-only`
   - hard gating
   - `visibility_filter union`
4. 等 gating 第一版接通后，再决定 replay compare 是先在：
   - `legacy gated/ungated` 上做，还是
   - `v2 RGB-only gated/ungated` 上做。

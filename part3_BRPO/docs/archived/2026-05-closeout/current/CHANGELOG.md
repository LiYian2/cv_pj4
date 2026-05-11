# CHANGELOG.md - Part3 BRPO 历史记录

> 更新时间：2026-05-05 06:00 (Asia/Shanghai)

---

## 2026-05-05 (下午) - GN 集成 + D-Series 规划

### Gauss-Newton 集成到 Online Mapping

**事件**：
- 在 `BRPOMappingConfig` 中添加 GN 参数：`use_gauss_newton`, `gn_max_iters`, `gn_damping`, `gn_every_n_steps`
- 在 `_run_joint_pseudo_engine()` 中集成 GN：
  - GN 在 backward 之后执行（需要 gradient）
  - GN 直接更新 theta/rho，然后 fold 到 R/T
  - 如果使用 GN，跳过 Adam pose optimizer
- 生成 D-series 实验配置（D0-D4）

**修改文件**：
- `third_party/S3PO-GS/utils/slam_backend_brpo.py`：集成 GN
- `scripts/generate_d_series_configs.py`：D-series 配置生成器
- `scripts/run_d_series.sh`：D-series 运行脚本

**D-Series 实验规划**：

| 实验 | 目标 | 关键配置 |
|------|------|---------|
| D0_baseline | 无 pseudo mapping 的 baseline | `pseudo_map_iters=0` |
| D1_pose_fix | 验证 pose gradient 修复 | `pseudo_map_iters=20`, Adam |
| D2_gauss_newton | 验证 GN 效果 | `pseudo_map_iters=20`, GN |
| D3_scale_reg | 验证 scale regularization | `lambda_scale=0.1` |
| D4_gn_scale | 组合验证 | GN + lambda_scale=0.1 |

---

## 2026-05-05 (上午) - Pose Gradient 修复 + Gauss-Newton 实现 + Scale Regularization

**事件**（略，详见上文）

---

## 2026-05-08 13:18
- Fixed E5 Difix CUDA OOM risk in spawn backend path:
  - third_party/S3PO-GS/slam.py now passes CPU background payload to BackEnd.
  - third_party/S3PO-GS/utils/slam_backend.py initializes backend CUDA context dynamically and passes target_device into Difix loader.
  - prepare_stage1_difix_dataset_s3po.py and runtime_exact_backend.py use model_bundle target_device instead of implicit cuda.
- Verification: remote py_compile and direct import/BackEnd constructor smoke passed.


## 2026-05-08 14:10
- Runtime exact backend now supports `pseudo_rgb_source` (`render` default, `gt` upper-bound).
- BackEnd resolver forwards `pseudo_rgb_source` into RuntimeExactBackendConfig.
- Created E6a/E6b GT-pseudo no-Difix projected-depth configs and launchers.


## 2026-05-08 14:14
- Launched E6 GT-pseudo no-Difix pair on GPU1.
- Added a conservative BackEnd idle-loop guard: skip background real-window mapping when `keyframe_optimizers is None`; this prevents joint-primary/no-legacy-map crashes and is a no-op for normal SLAM once legacy optimizers exist.


## 2026-05-09
- Fixed E7a 2IMG+PAIR binary cap bug: apply_cm_cap no longer multiplies depth targets by C_m=0.5 on single-support pixels; it uses (C_m > 0) support.
- Verification: py_compile passed; direct artifact smoke on event_kf_0033 showed single/both depth_effective/depth_calibrated median = 1.0 and no non-C_m depth.
- Reran E7a in-place with unchanged config. New result: before_opt PSNR 18.429, after_opt PSNR 20.597, SSIM 0.685, LPIPS 0.262, stats_final RMSE 0.329. This improves old E7a PSNR but remains below E5c and worsens ATE, so further diagnosis should target 2img depth geometry/scale consistency.


## 2026-05-09 — E7a depth-loss ablation
- Ran post-fix forensics comparing E7a binarycap 2IMG targets against E5c projected targets and E7a PAIR anchors. Shared-support median ratio is about 1.01, median abs-rel about 0.20; left-anchor abs-rel about 0.15, right-anchor abs-rel about 0.31, with late right-side failures around 0.63.
- Located trajectory spikes for depth-enabled E7a: frame 264 non-keyframe about 1.55m, frames 302/303 about 0.57/1.17m, and keyframe 305 about 1.18m after sim3 alignment.
- Launched clean E7a_binarycap_depthoff ablation after discovering match_real_loss_weights=true overrides lambda_depth. Final clean config uses match_real_loss_weights=false and lambda_depth=0.0.
- Result: before_opt PSNR 18.646; after_opt PSNR 21.633, SSIM 0.712, LPIPS 0.236; stats_final RMSE 0.0619. This confirms current 2IMG dense depth loss is harmful while pseudo RGB/C_m online mapping is beneficial.


## 2026-05-10 E8 C_m local expansion audit

- Fixed C_m expansion metadata/stat reporting: observation summaries now separate raw reciprocal C_m stats from consumed soft-C_m stats; diagnostic sidecar dry-run no longer writes frame outputs; sidecar summaries include depth target filled before/after and complete reject counters.
- Audited E8 at /home/bzhang512/my_storage2_1T/part3_online_mapping_experiments/E8_cm_local_expand_r1_soft. The run is protocol-aligned with E5c except cm_expansion_mode=local_soft_v1 and cm_expansion_apply_to_depth_scope=false.
- Final metrics: E8 after_opt PSNR 20.5222, SSIM 0.6629, LPIPS 0.2684, stats_final RMSE 0.0875. E5c reference: PSNR 21.2012, RMSE 0.0645. E8 degraded.
- Code/production-flow check: raw support is preserved; signal confidence equals cm_expanded_soft and not cm_raw; depth target/valid scope stays raw/projected. No current hard wiring bug was found in the audited E8 artifacts.
- Mechanistic diagnosis: local expansion adds about 8-16 percent image area as RGB-only soft C_m; depth_in_added is 0 for all audited events because apply_to_depth_scope=false and projected depth target remains raw. Since paper_brpo_split_v1 RGB loss normalizes by confidence_mask.sum, added easy/weak pixels dilute raw reciprocal seed RGB gradients by about 10-25 percent while adding no geometric anchor. This is the leading explanation for lower training pseudo losses but worse final PSNR/ATE.
- Next recommendation: do not continue full local_soft_v1 as-is. Test conservative variants: both-only/near-depth-valid expansion, or budget-preserving reweighting that keeps raw seed gradient mass constant; optionally pair with depth-off if isolating pure RGB expansion.


## 2026-05-10 — dense_match_v1 standalone RGB-only support mode
- Added `pseudo_branch/mask/dense_match_densify.py` implementing reciprocal point densify: disk rasterization, Gaussian blur, normalization, thresholding, plus raw-vs-dense support summaries.
- Updated `pseudo_branch/integration/runtime_exact_backend.py`:
  - new config fields on `RuntimeExactBackendConfig`
  - new `rgb_only_support_mode` branch inside `rgb_only_verification`
  - dense-match artifact export under `exact_backend_v1/dense_match_v1/`
  - hard guard rejecting `dense_match_v1` + `cm_expansion_mode != none`
  - exact meta now records `rgb_only_support_mode` and `dense_match_meta`
- Updated `third_party/S3PO-GS/utils/slam_backend.py` to parse and forward the new dense-match config keys from YAML.
- Verification:
  - remote `py_compile` passed for `dense_match_densify.py`, `runtime_exact_backend.py`, and `slam_backend.py`
  - direct import smoke confirmed new dataclass fields exist
  - direct function smoke confirmed dense support expands beyond raw reciprocal support
  - BackEnd resolver smoke confirmed YAML fields reach runtime config

## 2026-05-10 — E9 dense-match static bridge fix
- Patched `third_party/S3PO-GS/utils/slam_backend.py` mainline consuming path so `_maybe_prepare_brpo_runtime_slots()` now forwards `rgb_only_support_mode`, all `cm_dense_*` knobs, and `cm_expansion_*` knobs from `Results.brpo_online_mapping` into `RuntimeExactBackendConfig`.
- Added frame-level `exact_cfg_debug.json` export and a hard runtime check that raises on `expected_support_mode != resolved_support_mode`, preventing another silent fallback from `dense_match_v1` to `reciprocal_seed`.
- Verification: remote `py_compile` passed; resolver smoke on `configs/e9_dense_match_v1_depthoff.yaml` returns `dense_match_v1`, radius `2`, expansion `none`. No rerun launched.


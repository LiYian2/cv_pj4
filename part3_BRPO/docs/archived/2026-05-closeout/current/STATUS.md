# STATUS.md - Part3 BRPO Online Mapping 当前状态

> 更新时间：2026-05-06 16:30 (Asia/Shanghai)

---

## 0. 最新修改（2026-05-06）

### 0.1 KF0 Pose 跳过逻辑已移除

**文件**：`third_party/S3PO-GS/utils/slam_backend.py`
**修改**：删除第 631-632 行的 `if viewpoint.uid == 0: continue`
**原因**：DL3DV 的 `cameras.json` 来自 COLMAP 重建，不是真实 GT，KF0 也需要优化

**备份**：`slam_backend.py.bak_kf0_skip`

### 0.2 D5 实验配置已生成

**文件**：`/home/bzhang512/CV_Project/part3_BRPO/configs/d5_online_mapping_fix.yaml`

**关键配置**：
| 参数 | D5 值 | 说明 |
|---|---|---|
| placement_mode | quartile | 3 pseudo per gap |
| update_real_pose | true | 允许真实 KF pose 更新 |
| use_gauss_newton | true | 启用 GN |
| lambda_scale | 0.1 | Scale 正则化 |
| lambda_pseudo | 2.0 | 增强 pseudo 权重 |
| dense3d_conf_quantile | 0.15 | 更宽松的 pseudo 选择 |

### 0.3 待解决问题：22:1 Loss 比例失衡

**现状**：
| | 数量 | 像素覆盖 | 总监督 |
|---|---|---|---|
| Real | 10 frames | 100% | 10 单位 |
| Pseudo | 3 frames | ~15% | 0.45 单位 |

**已执行**：
- `lambda_pseudo = 2.0` ✓
- `placement_mode = quartile` (3 pseudo) ✓

**待讨论**：
- 方案 c：分离 pose/scene loss（pose loss 不受 mask 限制）

---

## 1. Config 传递 Bug（已修复 2026-05-05）

**问题**：`_resolve_brpo_online_mapping_cfg()` 遗漏读取关键参数
**修复**：已在 `slam_backend.py` 中补全参数传递

---

## 2. Gaussians 更新范围

**S3PO 主循环**：全部 6 个参数
- `_xyz`, `_features_dc/_rest`, `_opacity`, `_scaling`, `_rotation`

**Pseudo mapping**：
- densify/prune/opacity_reset 默认关闭
- 基本属性通过 gradient backward 更新
- Phase 1 修复后梯度传递完整 ✓

---

## 3. 关键文件

| 文件 | 用途 |
|---|---|
| `configs/d5_online_mapping_fix.yaml` | D5 实验配置 |
| `third_party/S3PO-GS/utils/slam_backend.py` | KF0 跳过已移除 |
| `pseudo_branch/refine/pseudo_camera_state.py` | pose delta 应用 |
| `pseudo_branch/integration/runtime_slot_selector.py` | quartile 模式 |

---

## 4. 下一步

| 优先级 | 任务 | 状态 |
|-------|------|------|
| P0 | 运行 D5 实验 | ❌ 待执行 |
| P1 | 验证 KF0 pose 优化效果 | ❌ 待验证 |
| P2 | 评估 22:1 比例改善 | ❌ 待评估 |

---

## 5. 一句话结论

> **KF0 跳过已移除，D5 配置已生成（quartile + update_real_pose + GN）。下一步运行 D5 实验验证效果。**


## 2026-05-08 13:18 — E5 Difix spawn CUDA device fix
- Implemented dynamic CUDA device inheritance for E5 Difix online mapping: BackEnd child now initializes logical cuda:0 under CUDA_VISIBLE_DEVICES and Difix receives the resolved backend device.
- Avoided passing the parent CUDA background tensor into the spawned BackEnd process; backend rebuilds background from CPU payload after device selection.
- Verified py_compile/import smoke; killed stale E5a OOM processes and submitted E5a/E5b via wait_gpu_and_run_e5_pair.sh.


## 2026-05-08 14:10 — E6 GT-pseudo upper-bound arm created
- Added E6a/E6b configs cloned from E5 but with `use_difix_restoration=false` and `pseudo_rgb_source=gt`.
- Purpose: upper-bound test where pseudo RGB target/matching/mask use dataset GT image instead of rendered pseudo RGB; depth remains E5 projected bidirectional exact backend.
- Added launchers `run_e6a...sh`, `run_e6b...sh`, and `run_e6_gtpseudo_pair.sh`. Verified py_compile, bash -n, and resolver fields. Not auto-started to avoid racing current E5 wait queue.


## 2026-05-08 14:14 — E6 GT-pseudo pair launched
- Launched `scripts/run_e6_gtpseudo_pair.sh` with `CUDA_VISIBLE_DEVICES=1`; E6a currently running.
- Log: `/data3/bzhang512/part3_online_mapping_experiments/e6_logs/e6_gtpseudo_pair_20260508_141247.log`.
- GPU1 usage observed around 4.1GB for main E6a process plus small child process; no Difix load.
- Added idle backend guard so joint-primary/no-legacy-map runs do not call real-window `map()` when `keyframe_optimizers` is unset.


## 2026-05-09 — E7a binary C_m cap rerun
- P0 bug fixed in pseudo_branch/common/twoimg_pair_proxy_depth.py: apply_cm_cap now uses binary support (C_m > 0) instead of multiplying depth values by discrete C_m. This prevents single-support pixels (C_m=0.5) from halving target depth.
- Overwrote/reran E7a with unchanged config: depth_generation_mode=twoimg_pair_proxy_cm_capped_v1, lambda_depth=0.025, color_refinement_use_pseudo=false.
- Smoke verified new event_kf_0033 artifacts: depth_effective/depth_calibrated median is 1.0 on both single and both support; signal depth equals fixed effective depth.
- Result: after_opt PSNR improved vs old E7a (20.04 -> 20.60) but still below E5c (21.20); ATE/stats_final worsened (0.329), so binary cap fixes one real bug but 2img depth remains unsafe for trajectory/geometry. Next: diagnose post-fix 2img depth quality/pose drift, not more lambda tuning first.


## 2026-05-09 — E7a post-binarycap forensics and depth-off ablation
- Artifact comparison: fixed E7a 2IMG depth has no global shared-support scale drift versus E5c projected depth (median ratio about 1.01), but shared-support median abs-rel is still about 0.20.
- E7a 2IMG target agrees much better with left PAIR anchor than right PAIR anchor: left median abs-rel about 0.15, right about 0.31, with some late events around 0.63. Added dense support is mostly single-support (about 80-94 percent), i.e. the least constrained region.
- Trajectory failure with depth enabled is localized but severe: non-keyframe frame 264 spikes to about 1.55m, and frames 302/303 plus keyframe 305 spike to about 0.57/1.17/1.18m after the kf_271 -> kf_305 interval.
- Clean ablation E7a_binarycap_depthoff used match_real_loss_weights=false and lambda_depth=0.0; runtime verified lambda_depth=0.0. Result: after_opt PSNR 21.633, SSIM 0.712, LPIPS 0.236, stats_final RMSE 0.0619. This beats E5c PSNR 21.201 and matches/improves E5c stats_final RMSE 0.0645.
- Current conclusion: pseudo RGB/C_m online mapping is useful; current dense 2IMG depth loss is the component corrupting geometry. Default next branch should be depth-off or gated depth, not full 2IMG dense depth.


## 2026-05-10 E8 C_m local expansion audit

- Fixed C_m expansion metadata/stat reporting: observation summaries now separate raw reciprocal C_m stats from consumed soft-C_m stats; diagnostic sidecar dry-run no longer writes frame outputs; sidecar summaries include depth target filled before/after and complete reject counters.
- Audited E8 at /home/bzhang512/my_storage2_1T/part3_online_mapping_experiments/E8_cm_local_expand_r1_soft. The run is protocol-aligned with E5c except cm_expansion_mode=local_soft_v1 and cm_expansion_apply_to_depth_scope=false.
- Final metrics: E8 after_opt PSNR 20.5222, SSIM 0.6629, LPIPS 0.2684, stats_final RMSE 0.0875. E5c reference: PSNR 21.2012, RMSE 0.0645. E8 degraded.
- Code/production-flow check: raw support is preserved; signal confidence equals cm_expanded_soft and not cm_raw; depth target/valid scope stays raw/projected. No current hard wiring bug was found in the audited E8 artifacts.
- Mechanistic diagnosis: local expansion adds about 8-16 percent image area as RGB-only soft C_m; depth_in_added is 0 for all audited events because apply_to_depth_scope=false and projected depth target remains raw. Since paper_brpo_split_v1 RGB loss normalizes by confidence_mask.sum, added easy/weak pixels dilute raw reciprocal seed RGB gradients by about 10-25 percent while adding no geometric anchor. This is the leading explanation for lower training pseudo losses but worse final PSNR/ATE.
- Next recommendation: do not continue full local_soft_v1 as-is. Test conservative variants: both-only/near-depth-valid expansion, or budget-preserving reweighting that keeps raw seed gradient mass constant; optionally pair with depth-off if isolating pure RGB expansion.


## 2026-05-10 — dense_match_v1 landed as standalone RGB-only support mode
- Implemented peer-style densify module at `pseudo_branch/mask/dense_match_densify.py`: reciprocal match points -> disk rasterization -> Gaussian blur -> normalize -> threshold.
- Added a new non-overlapping runtime switch for `rgb_only_verification` path: `rgb_only_support_mode = reciprocal_seed | dense_match_v1`.
- `dense_match_v1` is intentionally isolated from existing `cm_expansion_mode`; runtime now rejects combining `dense_match_v1` with `cm_expansion_mode != none` so the compare identity stays clean.
- Current contract: `dense_match_v1` only changes RGB-only branch support/C_m coverage; exact projected depth target path is unchanged. Final C_m remains the existing discrete both/xor contract because support is still consumed as binary left/right support.
- New config fields exposed through `slam_backend.py`: `cm_dense_point_radius`, `cm_dense_blur_sigma`, `cm_dense_blur_kernel`, `cm_dense_corr_threshold`, `cm_dense_seed_mode`, `cm_dense_normalize_mode`.
- Verification passed: remote py_compile for the three touched files; import smoke for `RuntimeExactBackendConfig`; direct `build_dense_match_maps()` smoke showed raw support ratio 0.005 -> dense support ratio 0.325 on a toy sample; BackEnd resolver smoke confirmed new YAML fields propagate into runtime cfg.
- New artifacts for this mode: raw reciprocal support remains saved separately, and `exact_backend_v1/dense_match_v1/` stores dense support, dense soft confidence, dense seed masks, and `dense_match_meta.json`.

## 2026-05-10 — E9 dense-match static bridge fix
- Located the real residual bridge bug in `third_party/S3PO-GS/utils/slam_backend.py`: `_maybe_prepare_brpo_runtime_slots()` constructed `RuntimeExactBackendConfig` without forwarding `rgb_only_support_mode` or the `cm_dense_*` knobs, so live runtime could silently fall back to `reciprocal_seed` even when YAML/resolver said `dense_match_v1`.
- Patched the mainline consuming path to forward all dense-match and `cm_expansion_*` fields from `Results.brpo_online_mapping` into `RuntimeExactBackendConfig`.
- Added frame-level `exact_cfg_debug.json` plus a hard runtime mismatch check (`expected_support_mode` vs `resolved_support_mode`) so future runs fail loudly instead of silently producing another reciprocal-seed artifact.
- Static verification only: remote `py_compile` passed and resolver smoke now reports `rgb_only_support_mode=dense_match_v1`, `cm_dense_point_radius=2`, `cm_expansion_mode=none`. No relaunch performed in this step.


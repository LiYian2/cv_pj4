# CHANGELOG.md - Part3 Stage1 过程记录

> **书写规范**：
> 1. 增量更新，倒序排列（最新在最上）
> 2. 每条提炼为 3-5 行：做了什么 → 发现什么 → 结论是什么
> 3. 口径统一：M~（Mask）、T~（Target）、G~（Gaussian Management）、R~（Topology）
> 4. 实验细节引用归档文档

---

## 2026-04-22

### 工程整理：scripts final audit Stage 3 / Stage 4
- 完成 compat shim 收口：新增 `scripts/compat/run_pseudo_refinement.py` 作为内部 compatibility entry；顶层 `scripts/run_pseudo_refinement.py` 收窄为仅维持旧 CLI 路径稳定的外部 wrapper。
- `scripts/run_pseudo_refinement_v2.py` 与 `scripts/diagnostics/diagnose_stageA_gradients.py` 已改为直接加载 `scripts/compat/run_pseudo_refinement.py`；内部调用不再依赖顶层 wrapper。
- 远端验证通过：`py_compile` 覆盖 compat / wrapper / current refine / diagnostic caller；最小 real import / smoke 成功确认 v2 与 diagnose 路径都能经 compat 层拿到 legacy module。
- 新增 `docs/SCRIPTS_FINAL_AUDIT_STAGE34_20260422.md`；current/design/hermes/scripts README 同步改写为“scripts 侧代码整理已完成，后续只剩 backend-only integration 与 working-tree/commit 收尾”。

### 工程整理：scripts final audit Stage 1 / Stage 2
- 完成 Stage 1 低风险 diagnostics 收纳：`analyze_m5_depth_signal.py`、`diagnose_stageA_gradients.py`、`diagnose_stageA_loss_contrib.py`、`summarize_stageA_compare.py` 已迁入 `scripts/diagnostics/`；同时修正 `diagnose_stageA_gradients.py` / `diagnose_stageA_loss_contrib.py` 的 repo-root 解析，并把 stageA archived shell runner 对 `summarize_stageA_compare.py` 的调用切到新路径。
- 完成 Stage 2 legacy prepare 下沉：`prepare_stage1_difix_dataset.py`、`prepare_stage1_difix_dataset_s3po.py`、`build_a2_expand_from_a1_signal_v2.py` 已迁入 `scripts/legacy_prepare/`；顶层 `scripts/` 现只保留 live core + `run_pseudo_refinement.py` compat shim。
- 远端验证通过：`py_compile` 覆盖 7 个新路径脚本 + `run_pseudo_refinement.py` / `run_pseudo_refinement_v2.py`；最小 real import / smoke 成功导入 diagnostics、legacy prepare、compat shim 与 current refine 入口。
- 新增 `docs/SCRIPTS_FINAL_AUDIT_STAGE12_20260422.md` 作为本轮收尾记录；current/design/hermes 同步改写为“scripts Stage 1 / 2 已完成，下一步若继续只审 compat shim”。

### 工程整理：pseudo_branch Phase 6（residual T~ top-level builder cleanup）
- 完成 `brpo_depth_target.py`、`brpo_depth_densify.py`、`depth_target_builder.py` → `pseudo_branch/target/...` 的 direct migration；`scripts/materialize_m5_depth_targets.py`、`scripts/select_signal_aware_pseudos.py`、`scripts/prepare_stage1_difix_dataset_s3po_internal.py` 与 `pseudo_branch/__init__.py` 已切到新路径。
- 远端验证通过：`py_compile` 覆盖三个 target residual 文件、`pseudo_branch/target/__init__.py`、`pseudo_branch/__init__.py` 和三个直接 caller；最小 real import / smoke 成功导入三个 caller、`pseudo_branch` 与 `pseudo_branch.target`，并真实执行 `build_blended_target_depth()`、`build_blended_target_depth_v2()`、`build_sparse_log_depth_correction()`、`densify_depth_correction_patchwise()`、`reconstruct_dense_depth_from_correction()`、`build_depth_source_map_v2()`、`load_depth()`、`get_intrinsic_matrix()`、`reproject_depth()`。
- 新增 `docs/PSEUDO_BRANCH_T_RESIDUAL_CLEANUP_PHASE6_20260422.md` 记录本轮范围、回滚备份与验证结论；layout/current/design/hermes 同步改写为“pseudo_branch 第二轮整理已正式收尾”。
- 本轮完成后，`pseudo_branch/` 顶层只剩 `__init__.py`；第二轮 direct migration 已完整闭环，后续若继续整理，应转向 `scripts/` 顶层 live 入口与历史 runner 的二次审计。

### 工程整理：pseudo_branch Phase 5（common direct migration）
- 完成 `align_depth_scale.py`、`build_pseudo_cache.py`、`diag_writer.py`、`epipolar_depth.py`、`flow_matcher.py` → `pseudo_branch/common/...` 的 direct migration；`scripts/brpo_verify_single_branch.py`、`scripts/brpo_build_mask_from_internal_cache.py`、`scripts/select_signal_aware_pseudos.py`、`scripts/build_brpo_v2_signal_from_internal_cache.py` 与 `pseudo_branch/__init__.py` 已切到新路径。
- 远端验证通过：`py_compile` 覆盖 common 下迁移文件、`pseudo_branch/__init__.py` 与四个直接 caller；最小 real import / smoke 成功导入四个 caller、`pseudo_branch` 与 `pseudo_branch.common`，并真实执行 `align_edp_depth()`、`load_depth()`、`get_intrinsic_matrix()`、`reproject_depth()`、`write_depth_consistency_map()`、`pose_c2w_to_w2c()`、`compute_fundamental_matrix()`、`compute_epipolar_distance()`。
- 新增 `docs/PSEUDO_BRANCH_COMMON_MIGRATION_PHASE5_20260422.md` 记录本轮范围、回滚备份与验证结论；layout/current/design/hermes 同步改写为“Phase 5 common 已完成，下一步处理 residual T~ top-level builder 收口”。
- Phase 5 验收时额外发现：顶层 `pseudo_branch/` 还剩 `brpo_depth_target.py`、`brpo_depth_densify.py`、`depth_target_builder.py` 三个 residual T~ flat files，它们不属于本轮 common 迁移，但应作为下一步最终收口对象。

### 工程整理：pseudo_branch Phase 4（M~ direct migration）
- 完成 `brpo_confidence_mask.py`、`brpo_train_mask.py`、`confidence_builder.py`、`joint_confidence.py`、`rgb_mask_inference.py` → `pseudo_branch/mask/...` 的 direct migration；`scripts/brpo_build_mask_from_internal_cache.py`、`scripts/select_signal_aware_pseudos.py`、`scripts/build_brpo_v2_signal_from_internal_cache.py` 与 `pseudo_branch/brpo_v2_signal/__init__.py` 已切到新路径。
- 远端验证通过：`py_compile` 覆盖 mask 下迁移文件、`brpo_v2_signal` package glue 和三个直接 caller；最小 real import / smoke 成功导入三个 caller，并真实执行 `build_brpo_confidence_mask()`、`build_train_confidence_masks()`、`build_confidence_from_target_depth()`、`build_joint_confidence_from_rgb_and_depth()`、`build_rgb_mask_from_correspondences()`。
- 新增 `docs/PSEUDO_BRANCH_M_MIGRATION_PHASE4_20260422.md` 记录本轮范围、回滚备份与验证结论；layout/current/design/hermes 同步改写为“Phase 4 已完成，下一步 Phase 5 common”。

### 工程整理：pseudo_branch Phase 3（T~/observation direct migration）
- 完成 `pseudo_fusion.py`、`brpo_reprojection_verify.py`、`pseudo_observation_brpo_style.py`、`pseudo_observation_verifier.py`、`joint_observation.py` → `pseudo_branch/observation/...`，以及 `depth_supervision_v2.py`、`support_expand.py` → `pseudo_branch/target/...` 的 direct migration；相关 caller 与 `brpo_v2_signal` package glue 已切到新路径。
- 远端验证通过：`py_compile` 覆盖迁移文件和直接 caller；最小 real import / smoke 成功导入 6 个 caller，并真实执行 verifier / fusion / target / observation / A2 expand 五类核心函数。
- 新增 `docs/PSEUDO_BRANCH_T_OBSERVATION_MIGRATION_PHASE3_20260422.md` 记录本轮范围、回滚备份与验证结论；layout/current/design/hermes 同步改写为“Phase 3 已完成，下一步 Phase 4 M~”。

### 工程整理：pseudo_branch Phase 2（R~ direct migration）
- 完成 `pseudo_branch/pseudo_camera_state.py`、`pseudo_branch/pseudo_loss_v2.py`、`pseudo_branch/pseudo_refine_scheduler.py` → `pseudo_branch/refine/...` 的 direct migration；`scripts/run_pseudo_refinement_v2.py`、两个 diagnose 脚本，以及 `pseudo_branch/gaussian_management/spgm/stats.py` 已切到新路径 import，未保留旧 top-level shim。
- 远端验证通过：`py_compile` 覆盖迁移文件和直接 caller；最小 real import / smoke 成功导入三个 caller，并真实执行 `current_w2c()`、`build_stageA_optimizer()`、`masked_rgb_loss()`。
- 新增 `docs/PSEUDO_BRANCH_R_MIGRATION_PHASE2_20260422.md` 记录本轮范围、回滚备份与验证结论；layout/current/design/hermes 同步改写为“Phase 2 已完成，下一步 Phase 3 T~/observation”。

### 工程整理：pseudo_branch Phase 1（G~ direct migration）
- 完成 `pseudo_branch/local_gating/*`、`pseudo_branch/spgm/*`、`pseudo_branch/gaussian_param_groups.py` → `pseudo_branch/gaussian_management/...` 的 direct migration；`pseudo_branch/pseudo_refine_scheduler.py` 与 `scripts/run_pseudo_refinement_v2.py` 已改为新路径 import，未保留旧 top-level shim。
- 远端验证通过：`py_compile` 覆盖迁移文件和直接 caller；最小 real import / smoke 成功导入 `scripts/run_pseudo_refinement_v2.py` 与 `pseudo_branch.gaussian_management`，`build_micro_gaussian_param_groups()` / `build_spgm_grad_weights()` 已真实执行。
- 新增 `docs/PSEUDO_BRANCH_G_MIGRATION_PHASE1_20260422.md` 记录本轮范围、回滚备份与验证结论；layout/current/design/hermes 同步改写为“Phase 1 已完成，下一步 Phase 2 R~”。

### 工程整理：scripts 第二轮清理 + pseudo_branch 迁移骨架
- 把历史 A1 / StageA / topology / legacy refine runner 统一收进 `scripts/archive_experiments/`，顶层 `scripts/` 只保留 live 入口、active builder 和仍在被当前 docs 消费的 utility。
- 新建 `pseudo_branch/{common,observation,mask,target,gaussian_management,refine}/` 目录骨架，并落 `docs/design/PSEUDO_BRANCH_LAYOUT.md` 记录 direct migration mapping；该计划明确不走最小兼容 shim。
- 同步刷新 current/design/agent handoff 中的 stale path 和 stale next-step：G~ clean compare 长文改指向 archived 路径，T~/R~ 口径改为 T4 已完成、下一阶段转 backend-only integration。

### T~-Phase T4.0：exact-upstream smoke chain
- 完成 `T4.0a branch-native exact backend smoke`、`T4.0b exact-upstream signal smoke`、`T4.0c consumer smoke`，确认 exact backend / exact-upstream signal / `exact_shared_cm_v1` consumer 路径都被真实消费。
- 同步修正两类 live mismatch：branch-native difix 真路径应为 `_flat_difix_from_pseudo_cache_baseline/...`；`summary_only` G~ control 必须走 passthrough policy，避免先做 grad-weight 再统计导致假 clean control。
- smoke 结论：`exact_brpo_upstream_target_v1 + exact_shared_cm_v1` 已经不是“builder-only 接线”，而是完整可训练的 T~ exact-upstream 路径。

### T~-Phase T4.1：full exact-upstream formal compare
- 完成 full exact backend build、full exact-upstream signal build，以及 4-arm fixed-control compare：`oldA1_newT1_summary_only`、`exactBrpoCm_oldTarget_v1_newT1_summary_only`、`exactBrpoFullTarget_v1_newT1_summary_only`、`exactBrpoUpstreamTarget_v1_newT1_summary_only`。
- compare summary 明确确认：`oldA1` arm 使用 `signal_v2_root=/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_a1_full_brpo_target_signal_full`；G~ 末态仍为 clean `summary_only` passthrough control。
- replay 结果：upstream vs full `+0.035853`、vs exact-cm-oldtarget `+0.022966`、vs oldA1 `+0.022358`。
- 结论：T4 exact-upstream 第一轮正式落地成功，`exact_brpo_upstream_target_v1` 应升为新的 T~ control / mainline。

### 文档同步：STATUS / DESIGN / TARGET_DESIGN
- 把“Phase T4 待执行 / old T~ 仍是当前主线”的旧口径覆盖更新为：T4 compare 已完成，当前 standalone 最优组合为 `exact M~ + exact upstream T~ + clean summary G~ + T1`。
- 同步固化新的设计判断：决定性增益来自 upstream verifier/backend/projected-depth/target field 对齐，而不是继续在 proxy backend 上做 consumer-side exact 化。

## 2026-04-21

### T~-Phase T1/T2/T3：exact upstream backend + target field + loss contract
- 完成 `verify_single_branch_exact()`：保留 provenance / hit_count / occlusion_reason / continuous confidence
- 完成 `build_exact_upstream_depth_target()`：exact depth target，无 render fallback，metadata 明确 `no_render_fallback=true`
- 完成 `build_exact_brpo_upstream_target_observation()`：exact M~ + exact T~ bundle，`recommended_stageA_depth_loss_mode=exact_shared_cm_v1`
- 完成 `build_stageA_loss_exact_shared_cm()`：shared C_m loss contract，RGB/depth 使用同一 mask，无 fallback tier
- 完成 `pseudo_fusion.py` 改动：新增 `exact_conf_left/right` 参数，支持用 exact backend confidence 替换 proxy fusion weights
- 完成 `run_pseudo_refinement_v2.py` 集成：新增 `--stageA_depth_loss_mode=exact_shared_cm_v1`，StageA/StageB loop 调用逻辑已集成
- 结论：T~ upstream exact path 已完整落地（Phase T1-T3），随后在 2026-04-22 完成 Phase T4 formal compare 并升格为当前 T~ mainline。


### G~-C2：clean compare cleanup + smoke verify
- 修正 `spgm_action_semantics` 默认覆盖问题，`summary_only` 现在可作为 true no-action control 使用；同时 direct BRPO path 改为 `neutral_passthrough`，不再混 legacy grad-weight policy。
- 修正 current-step history：每个 iter 只保留一条 `current_step_probe_loss` 记录，不再 probe/post-backward 双 append。
- 直接 smoke 验证显示：`baseline_summary_only` 的 `grad_weight_mean_xyz=1.0`、`spgm_weight_mean=1.0`、`spgm_state_participation_ratio=1.0`；`direct_brpo_current_step` 的 `timing_mode_effective=current_step_probe_loss` 且 `grad_weight_mean_xyz=1.0`。
- 结论：G~ compare 现在有了干净 control，也有了较干净的 direct BRPO action/timing arm，旧 compare 污染点已清掉。

### G~-C3：最小 clean compare（20-iter + replay）
- 在 `20260421_g_brpo_clean_compare_v1` 跑三臂：`baseline_summary_only`、`legacy_delayed_opacity`、`direct_brpo_current_step`；详细长文见 `docs/archived/2026-04-experiments/G_BRPO_CLEAN_COMPARE_20260421.md`。
- Replay 结果：baseline `24.021169`，legacy delayed opacity `24.004704`（vs baseline `-0.016466`），direct current-step `24.026548`（vs baseline `+0.005379`）。
- 结论：旧 `+0.0114` 结论来自脏 baseline；clean compare 下 direct BRPO 仍小幅正向，但收益有限，legacy delayed opacity 明确负向。
- 项目判断同步更新：G~ 现在应冻结为已完成语义对齐的 side branch，下一条主线转向 T~ upstream backend，而不是继续优先扩 G~。

## 2026-04-20

### 文档重组：口径统一为 M~/T~/G~/R~
- 创建 TARGET_DESIGN.md（T~）、GAUSSIAN_MANAGEMENT_DESIGN.md（G~）、新 REFINE_DESIGN.md（R~）
- 更新 MASK_DESIGN.md（M~），分离 T~ 内容
- 更新 DESIGN.md，四大模块口径统一
- 更新 CHANGELOG.md，历史记录口径统一
- 结论：M~ 与 T~ 最初耦合实现但语义独立，现在各自有独立设计文档。

### M~-T~-E1：exact BRPO target-side compare
- 在 pseudo_observation_brpo_style.py 新增 exact_brpo_full_target_v1：strict M~ + exact T~ + shared-M~ depth loss
- Compare 结果：exact M~ + exact T~ = 24.174488，exact M~ + old T~ = 24.187495，old M~ + old T~ = 24.187737
- 结论：exact M~ 已对齐，exact T~ 在当前 proxy backend 下仍弱 -0.013 PSNR，瓶颈在上游 Layer B。

### M~-S1：strict BRPO M~ checklist + exact/hybrid split
- 新增 A1_STRICT_BRPO_ALIGNMENT_CHECKLIST.md，明确 M~ 与 T~ 分离
- 接入三条 exact M~ ablation：exact_brpo_cm_old_target_v1、exact_brpo_cm_hybrid_target_v1、exact_brpo_cm_stable_target_v1
- Compare 结果：exact M~ + old T~ ≈ old M~ + old T~（差 < 1e-5），说明 M~ 已对齐
- 结论：剩余 gap 在 T~ contract，不在 M~。

### M~-D1：direct BRPO builder + fixed R~ compare
- 新增 build_brpo_direct_observation，产出 hybrid M~ + hybrid T~ bundle
- Compare 结果：hybrid M~ + hybrid T~ = 24.175408，优于 new M~ + new T~ 但弱于 old 约 -0.012
- 结论：hybrid 分支（historical brpo_direct_v1）不是 strict BRPO，应明确标记为 hybrid。

### G~-O1-C1：delayed opacity participation compare
- 在 score.py 拆出 participation_score，在 manager.py 新增 deterministic_opacity_participation
- Compare 结果：opacity vs summary = -0.000573 PSNR，仍弱负
- 结论：O0/O1 wiring 成立，但 delayed opacity 当前不应推进 O2a/b，应先做 C0 诊断。

### M~-V2-H1：fixed R~ 下 conf-only / depth-only hybrid compare
- 补做 conf-only（M~ brpo_style_v2 + T~ old）与 depth-only（M~ old + T~ brpo_style_v2）
- Compare 结果：三臂差异 ~1e-4，不能归因成单侧问题
- 结论：剩余 gap 需 M~ + T~ 联合改动。

### M~-R1：BRPO-style v1 builder + fixed R~ compare
- 新增 pseudo_observation_brpo_style.py，产出 hybrid M~ + hybrid T~ bundle
- Compare 结果：hybrid M~ + hybrid T~ = 24.175377，优于 new M~ + new T~ 但弱于 old
- 结论：shared M~ 方向对，但 verifier/target builder 不够强。

### M~-V1：verifier proxy + fixed R~ compare（negative）
- 新增 pseudo_observation_verifier.py，产出 verify M~
- Compare 结果：verify M~ + new T~ = 24.067703，显著差于 old 和 new
- 结论：verify proxy 已完成排错职责，证明保守 decoupling 不够强。

## 2026-04-19

### G~-R1：deterministic participation controller + compare
- 把 G~ 从 xyz_lr_scale 推进到 pre-render boolean participation control
- Compare 结果：boolean vs summary = -0.0032 PSNR，weak-negative
- 结论：G~ 方法对象已切对，但 keep 配置当前 no-go。

---

## 口径说明

- M~ = Mask（confidence）
- T~ = Target（depth target）
- G~ = Gaussian Management（per-Gaussian gating）
- R~ = Topology（joint loop）

历史记录中的 A1 → M~ + T~，B3 → G~，T1 → R~。
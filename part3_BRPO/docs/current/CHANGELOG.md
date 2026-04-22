# CHANGELOG.md - Part3 Stage1 过程记录

> **书写规范**：
> 1. 增量更新，倒序排列（最新在最上）
> 2. 每条提炼为 3-5 行：做了什么 → 发现什么 → 结论是什么
> 3. 口径统一：M~（Mask）、T~（Target）、G~（Gaussian Management）、R~（Topology）
> 4. 实验细节引用归档文档

---

## 2026-04-22

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
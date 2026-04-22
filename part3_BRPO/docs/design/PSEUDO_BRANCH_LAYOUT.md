# PSEUDO_BRANCH_LAYOUT.md

> 更新时间：2026-04-22 15:04 (Asia/Shanghai)
> 目标：记录 `pseudo_branch/` 第二轮整理的最终 live layout，固化已完成的 G~/R~/T~/M~/common direct migration 与 residual T~ cleanup，并给出收尾后的边界。

---

> 当前进度：Phase 0（目录骨架）、Phase 1（G~ direct migration）、Phase 2（R~ direct migration）、Phase 3（T~/observation direct migration）、Phase 4（M~ direct migration）、Phase 5（common direct migration）与 Phase 6（residual T~ cleanup）已完成，并已通过 `py_compile` + 最小 real import/smoke 验证；第二轮 `pseudo_branch/` 目录整理已正式收尾。

---

## 1. 迁移原则

1. 目录组织按当前统一口径对齐：M~ / T~ / G~ / R~，另保留 `common/` 与 `observation/` 两个工程层目录承接跨模块 glue。
2. **不做最小兼容 shim**：每一轮迁移都直接改 import / caller / doc 引用，不保留“旧路径继续转发”的长期壳层。
3. 可以分步迁移，但每一步都必须是可运行闭环：同一 patch 内完成文件移动、import 更新、`py_compile` 验证。
4. 顶层 `pseudo_branch/` 最终只保留包入口和清晰分层，不再继续堆平铺文件。

---

## 2. 目标目录骨架

```text
pseudo_branch/
├── common/
├── observation/
├── mask/
├── target/
├── gaussian_management/
└── refine/
```

目录职责：
- `common/`: 几何/缓存/诊断等跨模块工具
- `observation/`: 同时产出 M~/T~ bundle 的 builder、verifier、fusion glue
- `mask/`: confidence / C_m / train-mask 相关逻辑
- `target/`: projected-depth / target-depth / supervision 相关逻辑
- `gaussian_management/`: `local_gating/`、`spgm/`、Gaussian 参数组
- `refine/`: loss、scheduler、camera state、refine loop glue

---

## 3. 文件映射（当前 → 目标）

### 3.1 G~
- `pseudo_branch/local_gating/*` → `pseudo_branch/gaussian_management/local_gating/*`
- `pseudo_branch/spgm/*` → `pseudo_branch/gaussian_management/spgm/*`
- `pseudo_branch/gaussian_param_groups.py` → `pseudo_branch/gaussian_management/gaussian_param_groups.py`

### 3.2 R~
- `pseudo_branch/pseudo_loss_v2.py` → `pseudo_branch/refine/pseudo_loss_v2.py`
- `pseudo_branch/pseudo_refine_scheduler.py` → `pseudo_branch/refine/pseudo_refine_scheduler.py`
- `pseudo_branch/pseudo_camera_state.py` → `pseudo_branch/refine/pseudo_camera_state.py`

### 3.3 Observation glue（跨 M~/T~）
- `pseudo_branch/pseudo_fusion.py` → `pseudo_branch/observation/pseudo_fusion.py`
- `pseudo_branch/brpo_reprojection_verify.py` → `pseudo_branch/observation/brpo_reprojection_verify.py`
- `pseudo_branch/brpo_v2_signal/pseudo_observation_brpo_style.py` → `pseudo_branch/observation/pseudo_observation_brpo_style.py`
- `pseudo_branch/brpo_v2_signal/pseudo_observation_verifier.py` → `pseudo_branch/observation/pseudo_observation_verifier.py`
- `pseudo_branch/brpo_v2_signal/joint_observation.py` → `pseudo_branch/observation/joint_observation.py`

### 3.4 M~
- `pseudo_branch/brpo_confidence_mask.py` → `pseudo_branch/mask/brpo_confidence_mask.py`
- `pseudo_branch/brpo_train_mask.py` → `pseudo_branch/mask/brpo_train_mask.py`
- `pseudo_branch/confidence_builder.py` → `pseudo_branch/mask/confidence_builder.py`
- `pseudo_branch/brpo_v2_signal/joint_confidence.py` → `pseudo_branch/mask/joint_confidence.py`
- `pseudo_branch/brpo_v2_signal/rgb_mask_inference.py` → `pseudo_branch/mask/rgb_mask_inference.py`

### 3.5 T~
- `pseudo_branch/brpo_depth_target.py` → `pseudo_branch/target/brpo_depth_target.py`
- `pseudo_branch/brpo_depth_densify.py` → `pseudo_branch/target/brpo_depth_densify.py`
- `pseudo_branch/depth_target_builder.py` → `pseudo_branch/target/depth_target_builder.py`
- `pseudo_branch/brpo_v2_signal/depth_supervision_v2.py` → `pseudo_branch/target/depth_supervision_v2.py`
- `pseudo_branch/brpo_v2_signal/support_expand.py` → `pseudo_branch/target/support_expand.py`

### 3.6 Common
- `pseudo_branch/align_depth_scale.py` → `pseudo_branch/common/align_depth_scale.py`
- `pseudo_branch/build_pseudo_cache.py` → `pseudo_branch/common/build_pseudo_cache.py`
- `pseudo_branch/diag_writer.py` → `pseudo_branch/common/diag_writer.py`
- `pseudo_branch/epipolar_depth.py` → `pseudo_branch/common/epipolar_depth.py`
- `pseudo_branch/flow_matcher.py` → `pseudo_branch/common/flow_matcher.py`

---

## 4. 推荐迁移顺序（direct migration，不做 shim）

### Phase 0（已完成）
- 建立目录骨架
- 补 layout 文档
- 为 direct migration 先清出目标层级

### Phase 1：G~ 先迁（已完成）
原因：`local_gating/` 与 `spgm/` 已经是相对独立的子树，最适合先整体进 `gaussian_management/`。
- 已迁：`local_gating/`、`spgm/`、`gaussian_param_groups.py` → `pseudo_branch/gaussian_management/`
- 已改：`pseudo_branch/pseudo_refine_scheduler.py`、`scripts/run_pseudo_refinement_v2.py` import 已切到新路径
- 已清：旧 top-level G~ 路径已删除，未保留 shim
- 已验：`py_compile` + 最小 real import/smoke 已通过；详见 `docs/PSEUDO_BRANCH_G_MIGRATION_PHASE1_20260422.md`

### Phase 2：R~ 壳层迁移（已完成）
- 已迁：`pseudo_camera_state.py`、`pseudo_loss_v2.py`、`pseudo_refine_scheduler.py` → `pseudo_branch/refine/`
- 已改：`scripts/run_pseudo_refinement_v2.py`、两个 diagnose 脚本、`pseudo_branch/gaussian_management/spgm/stats.py` import 已切到新路径
- 已清：旧 top-level R~ 路径已删除，未保留 shim
- 已验：`py_compile` + 最小 real import/smoke 已通过；详见 `docs/PSEUDO_BRANCH_R_MIGRATION_PHASE2_20260422.md`

### Phase 3：T~/observation 主干迁移（已完成）
- 先迁当前 T~ 主线真实入口：`pseudo_fusion.py`、`brpo_reprojection_verify.py`、`depth_supervision_v2.py`、`pseudo_observation_brpo_style.py`
- 再迁 verifier / joint observation 衍生文件

### Phase 4：M~ 辅助文件迁移（已完成）
- 已迁：`brpo_confidence_mask.py`、`brpo_train_mask.py`、`confidence_builder.py` → `pseudo_branch/mask/`
- 已迁：`brpo_v2_signal/joint_confidence.py`、`brpo_v2_signal/rgb_mask_inference.py` → `pseudo_branch/mask/`
- 已改：`scripts/brpo_build_mask_from_internal_cache.py`、`scripts/select_signal_aware_pseudos.py`、`scripts/build_brpo_v2_signal_from_internal_cache.py` 与 `pseudo_branch/brpo_v2_signal/__init__.py` import 已切到新路径
- 已清：旧 top-level M~ 路径与 `brpo_v2_signal` 内对应平铺文件已删除，未保留 shim
- 已验：`py_compile` + 最小 real import/smoke 已通过；详见 `docs/PSEUDO_BRANCH_M_MIGRATION_PHASE4_20260422.md`

### Phase 5：common 收尾 + 清理旧平铺路径（已完成）
- 已迁：`align_depth_scale.py`、`build_pseudo_cache.py`、`diag_writer.py`、`epipolar_depth.py`、`flow_matcher.py` → `pseudo_branch/common/`
- 已改：`scripts/brpo_verify_single_branch.py`、`scripts/brpo_build_mask_from_internal_cache.py`、`scripts/select_signal_aware_pseudos.py`、`scripts/build_brpo_v2_signal_from_internal_cache.py` 与 `pseudo_branch/__init__.py` import 已切到新路径
- 已清：旧 top-level common 路径已删除，未保留 shim
- 已验：`py_compile` + 最小 real import/smoke 已通过；详见 `docs/PSEUDO_BRANCH_COMMON_MIGRATION_PHASE5_20260422.md`

### Phase 6：residual T~ builder cleanup（已完成）
- 已迁：`brpo_depth_target.py`、`brpo_depth_densify.py`、`depth_target_builder.py` → `pseudo_branch/target/`
- 已改：`scripts/materialize_m5_depth_targets.py`、`scripts/select_signal_aware_pseudos.py`、`scripts/prepare_stage1_difix_dataset_s3po_internal.py` 与 `pseudo_branch/__init__.py` import 已切到新路径
- 已清：旧 top-level T~ flat paths 已删除，未保留 shim
- 已验：`py_compile` + 最小 real import/smoke 已通过；详见 `docs/PSEUDO_BRANCH_T_RESIDUAL_CLEANUP_PHASE6_20260422.md`

---

## 5. 每一阶段的验收

1. `python -m py_compile` 覆盖本阶段移动的 Python 文件与它们的直接 caller
2. 至少做一次最小 real import / smoke，证明新路径被真实加载
3. `docs/current/STATUS.md`、`docs/current/DESIGN.md`、相关 design doc 的路径引用同步更新
4. 不留下长期兼容壳、双路径并存 import、或“先 move 以后再修”的半成品状态

---

## 6. 当前结论

当前已完成“骨架 + G~ Phase 1 + R~ Phase 2 + Phase 3 observation/target + Phase 4 M~ + Phase 5 common + Phase 6 residual T~ cleanup”七步，live G~ 入口已经收敛到 `pseudo_branch/gaussian_management/`，live R~ 入口已经收敛到 `pseudo_branch/refine/`，live observation 入口已经收敛到 `pseudo_branch/observation/`，live M~ 入口已经收敛到 `pseudo_branch/mask/`，live target 入口已经收敛到 `pseudo_branch/target/`，live common 入口已经收敛到 `pseudo_branch/common/`。当前顶层 `pseudo_branch/` 只剩 `__init__.py`，第二轮目录整理已正式结束；后续若继续工程清理，应转向 `scripts/` 顶层 live 入口与历史 runner 的边界审计，而不是再回头保留平铺路径。

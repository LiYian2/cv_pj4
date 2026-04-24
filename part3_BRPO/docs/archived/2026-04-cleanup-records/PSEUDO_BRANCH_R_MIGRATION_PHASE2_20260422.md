# PSEUDO_BRANCH_R_MIGRATION_PHASE2_20260422.md

> 更新时间：2026-04-22 12:43 (Asia/Shanghai)
> 目标：按 `docs/design/PSEUDO_BRANCH_LAYOUT.md` 完成 Phase 2 的 R~ direct migration，并给出最小可复用验证结论。

---

## 1. 本轮范围

> 后续更新：Phase 3（T~/observation direct migration）已完成，见 `docs/archived/2026-04-cleanup-records/PSEUDO_BRANCH_T_OBSERVATION_MIGRATION_PHASE3_20260422.md`。

本轮只做 R~ 壳层迁移，不改 M~/T~/G~ 语义，不保留旧 top-level shim。

迁移内容：
- `pseudo_branch/pseudo_camera_state.py` → `pseudo_branch/refine/pseudo_camera_state.py`
- `pseudo_branch/pseudo_loss_v2.py` → `pseudo_branch/refine/pseudo_loss_v2.py`
- `pseudo_branch/pseudo_refine_scheduler.py` → `pseudo_branch/refine/pseudo_refine_scheduler.py`
- `scripts/run_pseudo_refinement_v2.py` import 改到新 R~ 路径
- `scripts/diagnostics/diagnose_stageA_gradients.py`、`scripts/diagnostics/diagnose_stageA_loss_contrib.py` import 改到新 R~ 路径
- `pseudo_branch/gaussian_management/spgm/stats.py` 对 `current_w2c` 的引用改到新 R~ 路径

旧路径处理：
- 已删除 repo 内旧 `pseudo_branch/pseudo_camera_state.py`
- 已删除 repo 内旧 `pseudo_branch/pseudo_loss_v2.py`
- 已删除 repo 内旧 `pseudo_branch/pseudo_refine_scheduler.py`
- 未保留长期兼容转发壳

回滚备份：
- `/data2/bzhang512/.hermes_backups/part3_BRPO/r_migration_20260422_code/`

---

## 2. 验证

### 2.1 Syntax
已在远端执行 `python -m py_compile`，覆盖：
- `pseudo_branch/refine/__init__.py`
- `pseudo_branch/refine/pseudo_camera_state.py`
- `pseudo_branch/refine/pseudo_loss_v2.py`
- `pseudo_branch/refine/pseudo_refine_scheduler.py`
- `pseudo_branch/gaussian_management/spgm/stats.py`
- `scripts/run_pseudo_refinement_v2.py`
- `scripts/diagnostics/diagnose_stageA_gradients.py`
- `scripts/diagnostics/diagnose_stageA_loss_contrib.py`

### 2.2 Minimal real import / smoke
已在远端做最小 real import / smoke，验证两层：
1. `scripts/run_pseudo_refinement_v2.py`、`scripts/diagnostics/diagnose_stageA_gradients.py`、`scripts/diagnostics/diagnose_stageA_loss_contrib.py` 均可真实 import，说明 caller 已消费新 R~ 路径
2. `pseudo_branch.refine` package path 可真实加载并执行代表性函数

smoke 输出要点：
- `caller_imports_ok=True`
- `w2c_shape=(4, 4)`
- `w2c_t03=0.1`
- `optimizer_groups=4`
- `optimizer_group_names=['pseudo_0_7_cam_rot_delta', 'pseudo_0_7_cam_trans_delta', 'pseudo_0_7_exposure_a', 'pseudo_0_7_exposure_b']`
- `masked_rgb_loss=0.5`

说明：
- `current_w2c()` 已通过真实执行
- `build_stageA_optimizer()` 已通过真实执行
- `masked_rgb_loss()` 已通过真实执行
- 这次 smoke 覆盖了相机状态、scheduler 与 loss 三类 R~ 核心组件，而不只是 import 层

---

## 3. 当前结论

1. R~ Phase 2 direct migration 已完成，并满足 layout 文档要求的“move + import 更新 + syntax/smoke 验证”闭环。
2. `pseudo_branch/` 顶层不再保留旧 R~ 平铺路径，当前 live R~ 入口已经收敛到 `pseudo_branch/refine/`。
3. 下一步应进入 Phase 3：T~/observation 主干迁移，先处理 `pseudo_fusion.py`、`brpo_reprojection_verify.py`、`depth_supervision_v2.py`、`pseudo_observation_brpo_style.py` 这些当前真实主入口。

# PSEUDO_BRANCH_G_MIGRATION_PHASE1_20260422.md

> 更新时间：2026-04-22 06:19 (Asia/Shanghai)
> 目标：按 `docs/design/PSEUDO_BRANCH_LAYOUT.md` 完成 Phase 1 的 G~ direct migration，并给出可复用的验证结论。

---

## 1. 本轮范围

> 后续更新：Phase 2（R~ direct migration）已完成，见 `docs/PSEUDO_BRANCH_R_MIGRATION_PHASE2_20260422.md`。

本轮只做 G~ 迁移，不改 M~/T~/R~ 语义，不保留旧 top-level shim。

迁移内容：
- `pseudo_branch/local_gating/*` → `pseudo_branch/gaussian_management/local_gating/*`
- `pseudo_branch/spgm/*` → `pseudo_branch/gaussian_management/spgm/*`
- `pseudo_branch/gaussian_param_groups.py` → `pseudo_branch/gaussian_management/gaussian_param_groups.py`
- `pseudo_branch/pseudo_refine_scheduler.py` import 改到新 G~ 路径
- `scripts/run_pseudo_refinement_v2.py` import 改到新 G~ 路径

旧路径处理：
- 已删除 repo 内旧 `pseudo_branch/local_gating/`
- 已删除 repo 内旧 `pseudo_branch/spgm/`
- 已删除 repo 内旧 `pseudo_branch/gaussian_param_groups.py`
- 未保留长期兼容转发壳

回滚备份：
- `/data2/bzhang512/.hermes_backups/part3_BRPO/g_migration_20260422_code/`

---

## 2. 验证

### 2.1 Syntax
已在远端执行 `python -m py_compile`，覆盖：
- `pseudo_branch/gaussian_management/__init__.py`
- `pseudo_branch/gaussian_management/gaussian_param_groups.py`
- `pseudo_branch/gaussian_management/local_gating/*.py`
- `pseudo_branch/gaussian_management/spgm/*.py`
- `pseudo_branch/pseudo_refine_scheduler.py`
- `scripts/run_pseudo_refinement_v2.py`

### 2.2 Minimal real import / smoke
已在远端做最小 real import / smoke，验证两层：
1. `scripts/run_pseudo_refinement_v2.py` 可被真实 import，说明 caller 已消费新 G~ 路径
2. `pseudo_branch.gaussian_management` package path 可真实加载并执行代表性函数

smoke 输出要点：
- `script_import_ok=True`
- `config_mode='spgm_keep'`
- `group_names=['xyz', 'opacity']`
- `spgm_policy_mode_effective='dense_keep'`
- `spgm_selected_ratio=0.666667`

说明：
- `build_micro_gaussian_param_groups()` 已通过真实执行
- `build_spgm_grad_weights()` 已通过真实执行
- `build_stageA5_optimizers()` 在 dummy viewpoint 上触发预期 `AttributeError`，说明调用已经进入真实 StageA.5 optimizer 构建路径，而不是停在 import 层

---

## 3. 当前结论

1. G~ Phase 1 direct migration 已完成，并满足 layout 文档要求的“move + import 更新 + syntax/smoke 验证”闭环。
2. `pseudo_branch/` 顶层不再保留旧 G~ 平铺路径，当前 live G~ 入口已经收敛到 `pseudo_branch/gaussian_management/`。
3. 下一步应进入 Phase 2：R~ 壳层迁移，把 `pseudo_loss_v2.py`、`pseudo_refine_scheduler.py`、`pseudo_camera_state.py` 收进 `pseudo_branch/refine/`，并同步 `scripts/run_pseudo_refinement_v2.py` import。

---

## 4. 本轮涉及文件

新增/落位：
- `pseudo_branch/gaussian_management/__init__.py`
- `pseudo_branch/gaussian_management/README.md`
- `pseudo_branch/gaussian_management/gaussian_param_groups.py`
- `pseudo_branch/gaussian_management/local_gating/*`
- `pseudo_branch/gaussian_management/spgm/*`

改动 caller：
- `pseudo_branch/pseudo_refine_scheduler.py`
- `scripts/run_pseudo_refinement_v2.py`

同步文档：
- `docs/design/PSEUDO_BRANCH_LAYOUT.md`
- `docs/current/STATUS.md`
- `docs/current/DESIGN.md`
- `docs/current/CHANGELOG.md`
- `docs/design/GAUSSIAN_MANAGEMENT_DESIGN.md`
- `docs/design/REFINE_DESIGN.md`
- `docs/agent/hermes.md`

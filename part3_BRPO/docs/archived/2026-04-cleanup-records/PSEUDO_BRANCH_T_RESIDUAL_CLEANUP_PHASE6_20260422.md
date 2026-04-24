# PSEUDO_BRANCH_T_RESIDUAL_CLEANUP_PHASE6_20260422

> 更新时间：2026-04-22 15:04 (Asia/Shanghai)
> 范围：`pseudo_branch/` 第二轮整理的 Phase 6（residual T~ top-level builder cleanup）
> 结论：`brpo_depth_target.py`、`brpo_depth_densify.py`、`depth_target_builder.py` 已全部收进 `pseudo_branch/target/`；旧 top-level T~ flat paths 已退役；第二轮 `pseudo_branch/` 目录整理正式收尾完成。

---

## 1. 本轮收口范围

本轮处理的是 Phase 5 final audit 暴露出的三个 residual T~ top-level builder：

- `pseudo_branch/brpo_depth_target.py` → `pseudo_branch/target/brpo_depth_target.py`
- `pseudo_branch/brpo_depth_densify.py` → `pseudo_branch/target/brpo_depth_densify.py`
- `pseudo_branch/depth_target_builder.py` → `pseudo_branch/target/depth_target_builder.py`

同步修改的直接 caller / 包入口：
- `scripts/materialize_m5_depth_targets.py`
- `scripts/select_signal_aware_pseudos.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
- `pseudo_branch/__init__.py`
- `pseudo_branch/target/__init__.py`
- `pseudo_branch/target/README.md`

其中 `brpo_depth_target.py` 内部对 `brpo_depth_densify.py` 的引用也已切为同包相对导入。

---

## 2. 运行安全与回滚

收口前已确认当前用户下无活跃 `part3_BRPO` / `run_pseudo_refinement_v2` / `brpo` 相关进程占用这些文件。

本轮代码回滚备份：
- `/data2/bzhang512/.hermes_backups/part3_BRPO/t_residual_cleanup_20260422_code/`

本轮文档回滚备份：
- `/data2/bzhang512/.hermes_backups/part3_BRPO/t_residual_cleanup_20260422_docs/`

---

## 3. 验证

### 3.1 `py_compile`

已覆盖并通过：
- `pseudo_branch/__init__.py`
- `pseudo_branch/target/__init__.py`
- `pseudo_branch/target/brpo_depth_target.py`
- `pseudo_branch/target/brpo_depth_densify.py`
- `pseudo_branch/target/depth_target_builder.py`
- `scripts/materialize_m5_depth_targets.py`
- `scripts/select_signal_aware_pseudos.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`

### 3.2 最小 real import / smoke

真实导入成功：
- `scripts/materialize_m5_depth_targets.py`
- `scripts/select_signal_aware_pseudos.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
- `pseudo_branch`
- `pseudo_branch.target`

真实执行成功：
- `build_blended_target_depth()`
- `build_blended_target_depth_v2()`
- `build_sparse_log_depth_correction()`
- `densify_depth_correction_patchwise()`
- `reconstruct_dense_depth_from_correction()`
- `build_depth_source_map_v2()`
- `load_depth()`
- `get_intrinsic_matrix()`
- `reproject_depth()`

### 3.3 旧路径审计

live Python 代码中已无以下旧路径引用：
- `pseudo_branch.brpo_depth_target`
- `pseudo_branch.brpo_depth_densify`
- `pseudo_branch.depth_target_builder`

---

## 4. 收口后的树状态

本轮完成后，`pseudo_branch/` 顶层只剩：
- `__init__.py`

也就是说第二轮目录整理定义下的六个子层：`common/`、`observation/`、`mask/`、`target/`、`gaussian_management/`、`refine/` 已全部收齐，不再残留旧平铺 Python 模块。

---

## 5. 文档同步

本轮已同步更新：
- `docs/current/STATUS.md`
- `docs/current/DESIGN.md`
- `docs/current/CHANGELOG.md`
- `docs/agent/hermes.md`
- `docs/design/PSEUDO_BRANCH_LAYOUT.md`

---

## 6. 当前结论 / 下一步

- Phase 6 已完成：residual T~ top-level builder 已全部收进 `pseudo_branch/target/`
- 第二轮 `pseudo_branch/` 目录整理正式结束
- 如果还要继续做工程整理，下一步不应再是 `pseudo_branch/` 平铺收口，而应转向 `scripts/` 顶层 live 入口 / 历史 runner / util 的二次审计与归档边界

# SCRIPTS_FINAL_AUDIT_STAGE12_20260422

> 更新时间：2026-04-22 17:30 (Asia/Shanghai)
> 范围：scripts 侧 final audit 的 Stage 1（低风险 diagnostics 收纳）+ Stage 2（legacy prepare 下沉）
> 结论：顶层 `scripts/` 已从 16 个 Python 文件收敛为 8 个 live core + 1 个 compat shim；非 live 诊断脚本已收进 `scripts/diagnostics/`，旧 prepare / historical utility 已收进 `scripts/legacy_prepare/`。

---

## 1. 本轮收纳范围

### Stage 1：diagnostics 下沉
- `scripts/analyze_m5_depth_signal.py` → `scripts/diagnostics/analyze_m5_depth_signal.py`
- `scripts/diagnose_stageA_gradients.py` → `scripts/diagnostics/diagnose_stageA_gradients.py`
- `scripts/diagnose_stageA_loss_contrib.py` → `scripts/diagnostics/diagnose_stageA_loss_contrib.py`
- `scripts/summarize_stageA_compare.py` → `scripts/diagnostics/summarize_stageA_compare.py`

### Stage 2：legacy prepare 下沉
- `scripts/prepare_stage1_difix_dataset.py` → `scripts/legacy_prepare/prepare_stage1_difix_dataset.py`
- `scripts/prepare_stage1_difix_dataset_s3po.py` → `scripts/legacy_prepare/prepare_stage1_difix_dataset_s3po.py`
- `scripts/build_a2_expand_from_a1_signal_v2.py` → `scripts/legacy_prepare/build_a2_expand_from_a1_signal_v2.py`

同步修改：
- `scripts/archive_experiments/stageA/run_p0_absprior.sh`
- `scripts/archive_experiments/stageA/run_p1_stageA_signal_compare.sh`
- `scripts/README.md`
- `docs/archived/2026-04-cleanup-records/PSEUDO_BRANCH_R_MIGRATION_PHASE2_20260422.md`
- `docs/archived/2026-04-cleanup-records/PSEUDO_BRANCH_T_OBSERVATION_MIGRATION_PHASE3_20260422.md`
- `docs/current/STATUS.md`
- `docs/current/DESIGN.md`
- `docs/current/CHANGELOG.md`
- `docs/agent/hermes.md`

另外新增：
- `scripts/diagnostics/README.md`
- `scripts/legacy_prepare/README.md`

---

## 2. 运行安全与回滚

收纳前已确认当前用户下无活跃 `part3_BRPO` / `run_pseudo_refinement_v2` / `prepare_stage1_difix` / `brpo` 相关进程占用这些脚本。

本轮代码回滚备份：
- `/data2/bzhang512/.hermes_backups/part3_BRPO/scripts_stage12_cleanup_20260422_code/`

本轮文档回滚备份：
- `/data2/bzhang512/.hermes_backups/part3_BRPO/scripts_stage12_cleanup_20260422_docs/`

---

## 3. 验证

### 3.1 `py_compile`
已覆盖并通过：
- `scripts/diagnostics/analyze_m5_depth_signal.py`
- `scripts/diagnostics/diagnose_stageA_gradients.py`
- `scripts/diagnostics/diagnose_stageA_loss_contrib.py`
- `scripts/diagnostics/summarize_stageA_compare.py`
- `scripts/legacy_prepare/prepare_stage1_difix_dataset.py`
- `scripts/legacy_prepare/prepare_stage1_difix_dataset_s3po.py`
- `scripts/legacy_prepare/build_a2_expand_from_a1_signal_v2.py`
- `scripts/run_pseudo_refinement.py`
- `scripts/run_pseudo_refinement_v2.py`

### 3.2 最小 real import / smoke
真实导入成功：
- 4 个 diagnostics 脚本
- 3 个 legacy prepare 脚本
- `scripts.run_pseudo_refinement` compatibility shim
- `scripts.run_pseudo_refinement_v2`

额外检查：
- `diagnose_stageA_gradients.py` 现在仍能定位顶层 `scripts/run_pseudo_refinement.py`
- `archive_experiments/stageA/*.sh` 已改为调用 `scripts/diagnostics/summarize_stageA_compare.py`

### 3.3 顶层路径审计
本轮完成后，顶层 `scripts/` 只剩：
- `brpo_build_mask_from_internal_cache.py`
- `brpo_verify_single_branch.py`
- `build_brpo_v2_signal_from_internal_cache.py`
- `materialize_m5_depth_targets.py`
- `prepare_stage1_difix_dataset_s3po_internal.py`
- `replay_internal_eval.py`
- `run_pseudo_refinement.py`
- `run_pseudo_refinement_v2.py`
- `select_signal_aware_pseudos.py`

---

## 4. 当前结论 / 下一步

- scripts final audit 的 Stage 1 / Stage 2 已完成
- 顶层 `scripts/` 已基本收敛到 live core + 1 个 compat shim
- 后续 Stage 3 / 4 已在 `docs/archived/2026-04-cleanup-records/SCRIPTS_FINAL_AUDIT_STAGE34_20260422.md` 完成：`run_pseudo_refinement.py` 已收窄为外部 CLI wrapper，内部 compatibility boundary 已收进 `scripts/compat/`

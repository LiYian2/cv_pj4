# PSEUDO_BRANCH_COMMON_MIGRATION_PHASE5_20260422

> 更新时间：2026-04-22 14:04 (Asia/Shanghai)
> 范围：`pseudo_branch/` 第二轮整理的 Phase 5（common direct migration）
> 结论：live common 入口已收敛到 `pseudo_branch/common/`；旧 top-level common 路径已退役；验证通过。Phase 5 验收时同时发现仍有三个 residual T~ top-level builder 留在 `pseudo_branch/` 顶层。

---

## 1. 本轮迁移范围

本轮按 `docs/design/PSEUDO_BRANCH_LAYOUT.md` 执行 Phase 5 common direct migration，完成以下移动：

- `pseudo_branch/align_depth_scale.py` → `pseudo_branch/common/align_depth_scale.py`
- `pseudo_branch/build_pseudo_cache.py` → `pseudo_branch/common/build_pseudo_cache.py`
- `pseudo_branch/diag_writer.py` → `pseudo_branch/common/diag_writer.py`
- `pseudo_branch/epipolar_depth.py` → `pseudo_branch/common/epipolar_depth.py`
- `pseudo_branch/flow_matcher.py` → `pseudo_branch/common/flow_matcher.py`

同步修改的直接 caller / glue：
- `pseudo_branch/__init__.py`
- `pseudo_branch/common/__init__.py`
- `pseudo_branch/common/README.md`
- `scripts/brpo_verify_single_branch.py`
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/select_signal_aware_pseudos.py`
- `scripts/build_brpo_v2_signal_from_internal_cache.py`

额外修正了 `build_pseudo_cache.py` 内部仍残留的旧本地导入：
- `diag_writer` → `pseudo_branch.common.diag_writer`
- `epipolar_depth` → `pseudo_branch.common.epipolar_depth`
- `flow_matcher` → `pseudo_branch.common.flow_matcher`
- `pseudo_fusion` → `pseudo_branch.observation.pseudo_fusion`
- 顶部 `sys.path` 注入改为 repo root，避免迁移后脚本级绝对导入失效

---

## 2. 运行安全与回滚

迁移前已检查当前用户下无活跃 `part3_BRPO` / `run_pseudo_refinement_v2` / `brpo` 相关进程占用这些文件。

本轮代码回滚备份：
- `/data2/bzhang512/.hermes_backups/part3_BRPO/common_migration_20260422_code/`

本轮文档回滚备份：
- `/data2/bzhang512/.hermes_backups/part3_BRPO/common_migration_20260422_docs/`

---

## 3. 验证

### 3.1 `py_compile`

已覆盖并通过：
- `pseudo_branch/__init__.py`
- `pseudo_branch/common/__init__.py`
- `pseudo_branch/common/align_depth_scale.py`
- `pseudo_branch/common/build_pseudo_cache.py`
- `pseudo_branch/common/diag_writer.py`
- `pseudo_branch/common/epipolar_depth.py`
- `pseudo_branch/common/flow_matcher.py`
- `scripts/brpo_verify_single_branch.py`
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/select_signal_aware_pseudos.py`
- `scripts/build_brpo_v2_signal_from_internal_cache.py`

### 3.2 最小 real import / smoke

真实导入成功：
- `scripts/brpo_verify_single_branch.py`
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/select_signal_aware_pseudos.py`
- `scripts/build_brpo_v2_signal_from_internal_cache.py`
- `pseudo_branch`
- `pseudo_branch.common`

真实执行成功：
- `align_edp_depth()`
- `load_depth()`
- `get_intrinsic_matrix()`
- `reproject_depth()`
- `write_depth_consistency_map()`
- `pose_c2w_to_w2c()`
- `compute_fundamental_matrix()`
- `compute_epipolar_distance()`
- `FlowMatcher` class import / interface check（未实例化模型）

关键 smoke 输出：
- `align_success = True`
- `align_scale = 2.0`
- `align_overlap = 3`
- `loaded_depth_sum = 10.0`
- `reproj_valid_count = 4`
- `reproj_depth_sum = 10.0`
- `fundamental_finite = True`
- `epi_mean = 0.0`
- `diag_png_exists = True`
- `diag_consistency_sum = 2.006772`

### 3.3 旧路径审计

live Python 代码中已无以下旧路径引用：
- `pseudo_branch.align_depth_scale`
- `pseudo_branch.build_pseudo_cache`
- `pseudo_branch.diag_writer`
- `pseudo_branch.epipolar_depth`
- `pseudo_branch.flow_matcher`

---

## 4. Phase 5 结束后的树状态

Phase 5 完成后，`pseudo_branch/` 顶层只剩：
- `__init__.py`
- `brpo_depth_target.py`
- `brpo_depth_densify.py`
- `depth_target_builder.py`

也就是说 common 这一层已经收口完成；但 final audit 同时暴露出这三个 residual T~ builder 仍未从早先 Phase 3 的 target 规划里收走。

---

## 5. 文档同步

本轮已同步更新：
- `docs/current/STATUS.md`
- `docs/current/DESIGN.md`
- `docs/current/CHANGELOG.md`
- `docs/design/PSEUDO_BRANCH_LAYOUT.md`
- `docs/agent/hermes.md`

---

## 6. 当前结论 / 下一步

- Phase 5 已完成：live common 入口已收敛到 `pseudo_branch/common/`
- 顶层 old common flat paths 已退役
- 但第二轮目录整理还不能宣称完全结束，因为 final audit 明确还剩三个 residual T~ top-level builder
- 下一步应只做一个小范围 residual cleanup：
  1. `brpo_depth_target.py`
  2. `brpo_depth_densify.py`
  3. `depth_target_builder.py`
  4. 同步它们的 direct caller import 与最终路径索引

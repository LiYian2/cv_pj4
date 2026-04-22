# PSEUDO_BRANCH_M_MIGRATION_PHASE4_20260422

> 更新时间：2026-04-22 13:45 (Asia/Shanghai)
> 范围：`pseudo_branch/` 第二轮整理的 Phase 4（M~ direct migration）
> 结论：live M~ 入口已收敛到 `pseudo_branch/mask/`；旧 top-level M~ 路径与 `brpo_v2_signal` 内对应平铺文件已退役；验证通过。

---

## 1. 本轮迁移范围

本轮按 `docs/design/PSEUDO_BRANCH_LAYOUT.md` 执行 M~ direct migration，完成以下移动：

- `pseudo_branch/brpo_confidence_mask.py` → `pseudo_branch/mask/brpo_confidence_mask.py`
- `pseudo_branch/brpo_train_mask.py` → `pseudo_branch/mask/brpo_train_mask.py`
- `pseudo_branch/confidence_builder.py` → `pseudo_branch/mask/confidence_builder.py`
- `pseudo_branch/brpo_v2_signal/joint_confidence.py` → `pseudo_branch/mask/joint_confidence.py`
- `pseudo_branch/brpo_v2_signal/rgb_mask_inference.py` → `pseudo_branch/mask/rgb_mask_inference.py`

同步修改的直接 caller / glue：
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/select_signal_aware_pseudos.py`
- `scripts/build_brpo_v2_signal_from_internal_cache.py`
- `pseudo_branch/brpo_v2_signal/__init__.py`
- `pseudo_branch/mask/__init__.py`
- `pseudo_branch/mask/README.md`

策略保持不变：不保留兼容 shim，直接切 live import。

---

## 2. 运行安全与回滚

迁移前已检查当前用户下无活跃 `part3_BRPO` / `run_pseudo_refinement_v2` / `brpo` 相关进程占用这些文件。

本轮代码回滚备份：
- `/data2/bzhang512/.hermes_backups/part3_BRPO/m_migration_20260422_code/`

本轮文档回滚备份：
- `/data2/bzhang512/.hermes_backups/part3_BRPO/m_migration_20260422_docs/`

---

## 3. 验证

### 3.1 `py_compile`

已覆盖并通过：
- `pseudo_branch/mask/__init__.py`
- `pseudo_branch/mask/brpo_confidence_mask.py`
- `pseudo_branch/mask/brpo_train_mask.py`
- `pseudo_branch/mask/confidence_builder.py`
- `pseudo_branch/mask/joint_confidence.py`
- `pseudo_branch/mask/rgb_mask_inference.py`
- `pseudo_branch/brpo_v2_signal/__init__.py`
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/select_signal_aware_pseudos.py`
- `scripts/build_brpo_v2_signal_from_internal_cache.py`

### 3.2 最小 real import / smoke

真实导入成功：
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/select_signal_aware_pseudos.py`
- `scripts/build_brpo_v2_signal_from_internal_cache.py`
- `pseudo_branch.brpo_v2_signal`

真实执行成功：
- `build_brpo_confidence_mask()`
- `build_train_confidence_masks()`
- `build_confidence_from_target_depth()`
- `build_joint_confidence_from_rgb_and_depth()`
- `build_joint_depth_target()`
- `build_rgb_mask_from_correspondences()`（使用最小 dummy matcher + 临时图片）

关键 smoke 输出：
- `mask_conf_nonzero_ratio = 0.75`
- `mask_cont_mean_positive = 0.555526`
- `train_support_ratio_both = 0.5`
- `simple_conf_sum = 2.0`
- `joint_nonzero_ratio = 0.5`
- `joint_depth_sum = 10.0`
- `rgb_mask_support_ratio_both = 0.0625`
- `rgb_mask_nonzero_ratio = 0.1875`

### 3.3 旧路径审计

live Python 代码中已无以下旧路径引用：
- `pseudo_branch.brpo_confidence_mask`
- `pseudo_branch.brpo_train_mask`
- `pseudo_branch.confidence_builder`
- `pseudo_branch.brpo_v2_signal.joint_confidence`
- `pseudo_branch.brpo_v2_signal.rgb_mask_inference`

---

## 4. 文档同步

本轮已同步更新：
- `docs/current/STATUS.md`
- `docs/current/DESIGN.md`
- `docs/current/CHANGELOG.md`
- `docs/design/PSEUDO_BRANCH_LAYOUT.md`
- `docs/design/MASK_DESIGN.md`
- `docs/agent/hermes.md`

---

## 5. 当前结论 / 下一步

- Phase 4 已完成：live M~ 入口已收敛到 `pseudo_branch/mask/`
- 当前 `pseudo_branch/` 已完成 G~ / R~ / T~/observation / M~ 四轮 direct migration
- 下一步进入 Phase 5：把 `align_depth_scale.py`、`build_pseudo_cache.py`、`diag_writer.py`、`epipolar_depth.py`、`flow_matcher.py` 等 common helper 收进 `pseudo_branch/common/`，并清理残余旧平铺路径

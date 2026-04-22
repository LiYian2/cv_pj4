# PSEUDO_BRANCH_T_OBSERVATION_MIGRATION_PHASE3_20260422.md

> 更新时间：2026-04-22 13:45 (Asia/Shanghai)
> 目标：按 `docs/design/PSEUDO_BRANCH_LAYOUT.md` 完成 Phase 3 的 T~/observation direct migration，并给出最小可复用验证结论。

---

## 1. 本轮范围

本轮完成 observation / target 主干入口及其直接衍生文件迁移，不改 M~/G~/R~ 语义，不保留旧 top-level shim。

迁移内容：
- `pseudo_branch/pseudo_fusion.py` → `pseudo_branch/observation/pseudo_fusion.py`
- `pseudo_branch/brpo_reprojection_verify.py` → `pseudo_branch/observation/brpo_reprojection_verify.py`
- `pseudo_branch/brpo_v2_signal/pseudo_observation_brpo_style.py` → `pseudo_branch/observation/pseudo_observation_brpo_style.py`
- `pseudo_branch/brpo_v2_signal/pseudo_observation_verifier.py` → `pseudo_branch/observation/pseudo_observation_verifier.py`
- `pseudo_branch/brpo_v2_signal/joint_observation.py` → `pseudo_branch/observation/joint_observation.py`
- `pseudo_branch/brpo_v2_signal/depth_supervision_v2.py` → `pseudo_branch/target/depth_supervision_v2.py`
- `pseudo_branch/brpo_v2_signal/support_expand.py` → `pseudo_branch/target/support_expand.py`
- 直接 caller 与 package glue 已切到新路径：`scripts/brpo_build_mask_from_internal_cache.py`、`scripts/brpo_verify_single_branch.py`、`scripts/build_brpo_v2_signal_from_internal_cache.py`、`scripts/legacy_prepare/build_a2_expand_from_a1_signal_v2.py`、`scripts/prepare_stage1_difix_dataset_s3po_internal.py`、`scripts/select_signal_aware_pseudos.py`、`pseudo_branch/brpo_v2_signal/__init__.py`（其后 `joint_confidence.py` / `rgb_mask_inference.py` 又在 Phase 4 继续迁入 `pseudo_branch/mask/`）

旧路径处理：
- 已删除 repo 内旧 `pseudo_branch/pseudo_fusion.py`
- 已删除 repo 内旧 `pseudo_branch/brpo_reprojection_verify.py`
- 已删除 repo 内旧 `pseudo_branch/brpo_v2_signal/depth_supervision_v2.py`
- 已删除 repo 内旧 `pseudo_branch/brpo_v2_signal/pseudo_observation_brpo_style.py`
- 已删除 repo 内旧 `pseudo_branch/brpo_v2_signal/pseudo_observation_verifier.py`
- 已删除 repo 内旧 `pseudo_branch/brpo_v2_signal/joint_observation.py`
- 已删除 repo 内旧 `pseudo_branch/brpo_v2_signal/support_expand.py`
- 未保留长期兼容转发壳

回滚备份：
- `/data2/bzhang512/.hermes_backups/part3_BRPO/phase3_migration_20260422_code/`

---

## 2. 验证

### 2.1 Syntax
已在远端执行 `python -m py_compile`，覆盖：
- `pseudo_branch/observation/*.py`（本轮新增/迁入入口）
- `pseudo_branch/target/depth_supervision_v2.py`
- `pseudo_branch/target/support_expand.py`
- `pseudo_branch/brpo_v2_signal/__init__.py`（后续 Phase 4 已继续切到 `pseudo_branch/mask/joint_confidence.py` / `pseudo_branch/mask/rgb_mask_inference.py`）
- 直接 caller 脚本：`brpo_build_mask_from_internal_cache.py`、`brpo_verify_single_branch.py`、`build_brpo_v2_signal_from_internal_cache.py`、`build_a2_expand_from_a1_signal_v2.py`、`prepare_stage1_difix_dataset_s3po_internal.py`、`select_signal_aware_pseudos.py`

### 2.2 Minimal real import / smoke
已在远端做最小 real import / smoke，验证两层：
1. 上述 6 个直接 caller 都能真实 import，说明 live 调用面已消费新 observation/target 路径
2. observation / target 下的代表性函数已真实执行，而不只是 import 层

smoke 输出要点：
- `caller_imports_ok=True`
- `neighbors=(5, 12)`
- `translation_score=0.606531`
- `depth_verified_ratio=1.0`
- `obs_valid_ratio=1.0`
- `verify_union_ratio=1.0`
- `joint_valid_ratio=1.0`
- `a2_gain_ratio=0.25`

说明：
- `brpo_reprojection_verify.get_intrinsic_matrix_from_state()` / `find_neighbor_kfs()` 已通过真实执行
- `pseudo_fusion._translation_consistency_scalar()` 已通过真实执行
- `build_depth_supervision_v2()`、`build_brpo_style_observation()`、`build_pseudo_observation_verifier()`、`build_joint_observation_from_candidates()`、`build_support_expand_from_a1()` 均已通过真实执行
- 这次 smoke 覆盖了 verifier / fusion / target / observation / A2 expand 五类 Phase 3 核心路径

---

## 3. 当前结论

1. Phase 3 的 T~/observation 主干与直接衍生文件迁移已完成，并满足 layout 文档要求的“move + import 更新 + syntax/smoke 验证”闭环。
2. live observation 入口已收敛到 `pseudo_branch/observation/`，live target-side 主入口已收敛到 `pseudo_branch/target/`。
3. 下一步进入 Phase 4：M~ 辅助文件迁移，把 `brpo_confidence_mask.py`、`brpo_train_mask.py`、`confidence_builder.py`、`joint_confidence.py`、`rgb_mask_inference.py` 收到 `pseudo_branch/mask/`，并继续收口 `brpo_v2_signal/` 残余平铺内容。

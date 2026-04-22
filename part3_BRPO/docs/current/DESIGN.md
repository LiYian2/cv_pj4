# DESIGN.md - Part3 BRPO 设计文档

> 更新时间：2026-04-22 20:04 (Asia/Shanghai)

> **书写规范**：
> 1. 只记录"设计原则、架构决策、接口定义"，不记录实验数据
> 2. 覆盖式更新，直接修改对应版块，不追加
> 3. 设计判断用一句话固化，不展开过程
> 4. 引用格式：`[参见 STATUS §X]`
> 5. 更新后修改文档顶部时间戳

---

## 1. 系统边界与模块口径

### 1.1 四大模块

| 模块 | 口径 | 功能 | 详细文档 |
|------|------|------|---------|
| Mask | M~ | 监督域 + 监督强度 | MASK_DESIGN.md |
| Target | T~ | 监督目标数值 | TARGET_DESIGN.md |
| Gaussian Management | G~ | Per-Gaussian gating | GAUSSIAN_MANAGEMENT_DESIGN.md |
| Joint Refine | R~ | Topology joint loop | REFINE_DESIGN.md |

**注**：Fusion 已定，不纳入设计文档范围。

### 1.2 模块关系

```
M~ → pseudo RGB/depth mask（监督域）
T~ → pseudo depth target（监督目标）
G~ → pseudo render Gaussian gating（参与控制）
R~ → joint loss assembly + backward timing（拓扑）

Pipeline:
  Prepare → Fusion → M~/T~ builder → G~ gating → R~ topology → Backward → Optimize
```



### 1.3 T~ 状态

**Phase T1-T4 已完成**：
- exact backend bundle（`verify_single_branch_exact`）
- exact target field（`build_exact_upstream_depth_target`）
- exact loss contract（`build_stageA_loss_exact_shared_cm`）
- branch-native verifier input + exact-upstream signal + consumer smoke
- fixed clean G~ / fixed T1 formal compare

**核心语义**：
- `no_render_fallback=true`：不支持区域保持 invalid/zeroed，不悄悄 fallback
- `shared C_m`：RGB 和 depth 使用同一 confidence mask
- `verifier_backend_semantics=exact_branch_native_v1`：branch-native provenance
- `exact_brpo_upstream_target_v1`：当前 winning T~ bundle

**设计结论**：T~ 的决定性增益来自 upstream verifier/backend/target field 的整体对齐，而不是继续在 proxy backend 上做 consumer-side exact 化。

### 1.4 工程目录组织（第二轮整理）

- `scripts/` 顶层现在保留 8 个 live core 入口 + 1 个外部 CLI wrapper（`run_pseudo_refinement.py`）；历史 compare / one-off runner 归档到 `scripts/archive_experiments/`
- 内部 compatibility boundary 已收进 `scripts/compat/`，内部调用不再直接依赖顶层 `run_pseudo_refinement.py`
- non-live diagnostics / summary helper 已收进 `scripts/diagnostics/`
- legacy prepare / historical utility 已收进 `scripts/legacy_prepare/`
- `pseudo_branch/` 已建立 `common/ / observation/ / mask/ / target/ / gaussian_management/ / refine/` 六个骨架目录
- G~ Phase 1 已落地：`local_gating/`、`spgm/`、`gaussian_param_groups.py` 已迁入 `pseudo_branch/gaussian_management/`，直接 caller 已切到新路径，旧 top-level G~ 路径已退役
- R~ Phase 2 已落地：`pseudo_camera_state.py`、`pseudo_loss_v2.py`、`pseudo_refine_scheduler.py` 已迁入 `pseudo_branch/refine/`，直接 caller 已切到新路径，旧 top-level R~ 路径已退役
- Phase 3 已落地：observation / target 主干入口与直接衍生文件已迁入 `pseudo_branch/observation/` 与 `pseudo_branch/target/`，直接 caller 已切到新路径，旧 top-level T~/observation 路径已退役
- Phase 4 已落地：`brpo_confidence_mask.py`、`brpo_train_mask.py`、`confidence_builder.py`、`joint_confidence.py`、`rgb_mask_inference.py` 已迁入 `pseudo_branch/mask/`，直接 caller 与 `brpo_v2_signal` package glue 已切到新路径，旧 top-level M~ 路径已退役
- Phase 5 已落地：`align_depth_scale.py`、`build_pseudo_cache.py`、`diag_writer.py`、`epipolar_depth.py`、`flow_matcher.py` 已迁入 `pseudo_branch/common/`，直接 caller 与包入口已切到新路径，旧 top-level common 路径已退役
- Phase 6 residual T~ cleanup 已落地：`brpo_depth_target.py`、`brpo_depth_densify.py`、`depth_target_builder.py` 已迁入 `pseudo_branch/target/`，直接 caller 与包入口已切到新路径，旧 top-level T~ flat paths 已退役
- 第二轮迁移现已完整闭环：`pseudo_branch/` 顶层只剩 `__init__.py`，live 代码路径全部落到 `common/ / observation/ / mask/ / target/ / gaussian_management/ / refine/` 六层目录下
- 详细 mapping 与阶段顺序见 `docs/design/PSEUDO_BRANCH_LAYOUT.md`；本轮记录见 `docs/PSEUDO_BRANCH_G_MIGRATION_PHASE1_20260422.md`、`docs/PSEUDO_BRANCH_R_MIGRATION_PHASE2_20260422.md`、`docs/PSEUDO_BRANCH_T_OBSERVATION_MIGRATION_PHASE3_20260422.md`、`docs/PSEUDO_BRANCH_M_MIGRATION_PHASE4_20260422.md`、`docs/PSEUDO_BRANCH_COMMON_MIGRATION_PHASE5_20260422.md` 与 `docs/PSEUDO_BRANCH_T_RESIDUAL_CLEANUP_PHASE6_20260422.md`

---

## 2. 当前主线

### 2.1 固定参照线

**RGB-only v2 + gated_rgb0192 + post40_lr03_120**（canonical StageB protocol）

### 2.2 当前候选主线

| 模块 | 当前状态 | 说明 |
|------|---------|------|
| M~ | exact M~ | 已基本对齐 BRPO semantics；`exact_brpo_cm_old_target_v1` 保留为 semantics-clean control |
| T~ | exact upstream T~ | T4 formal compare 已证实这是当前 winning target path |
| G~ | clean summary G~ | clean compare 证明 G~ 不是当前主瓶颈 |
| R~ | T1 (brpo_joint_v1) | 当前 topology 主线 |

**当前主线判断**：`exact M~ + exact upstream T~ + clean summary G~ + T1` 是当前 standalone 最优组合；`exact M~ + old T~ + clean summary G~ + T1` 与 `old A1 + new T1` 保留为对照 control。

---

## 3. 设计判断（固化）

### 3.1 M~ 结论
- exact M~ 与 old M~ 基本等价（差 < 1e-5 PSNR）
- strict BRPO $C_m$ 已基本对齐
- 剩余 gap 不在 M~

### 3.2 T~ 结论
- `exact_brpo_full_target_v1` 证明：只做 proxy backend 下的 consumer-side exact 化，不足以赢 old A1
- `exact_brpo_upstream_target_v1` 证明：把 verifier backend / projected-depth / target field 整体拉到 exact upstream 之后，strict BRPO T~ 可以转成正向 winner
- 因此 T~ 当前主线应固定为 **exact upstream T~**，old T~ 仅保留为历史 control
- 后续大工程的重点不再是继续扫 standalone T~ compare，而是把这套 winner 以 backend-only 方式集成进 S3PO

### 3.3 G~ 结论
- clean compare 之后，direct BRPO current-step 仍有小幅正向，但幅度只有约 `+0.005 PSNR`
- 旧 `+0.0114` 结论来自脏 baseline，不能继续当设计依据
- legacy delayed opacity 明确负向，不能作为 landing 路线
- 因此 G~ 应定位为：**语义已对齐、收益有限的 side branch**；下一步不优先继续扩 G~，而是转到 T~ upstream

### 3.4 R~ 结论
- T1 (brpo_joint_v1) 是稳定 topology 主线
- 与 BRPO sequential topology 结构不同但有效
- G~ timing：T1 delayed，BRPO current-step

---

## 4. 接口定义

### 4.1 M~ 接口

| mode | confidence 来源 | 输出文件 |
|------|----------------|---------|
| old | rgb + geometry tier min | joint_confidence_v2.npy |
| new | score_stack 派生 | pseudo_confidence_joint_v1.npy |
| hybrid | support sets | pseudo_confidence_brpo_style_v1.npy |
| exact | strict BRPO semantics | pseudo_confidence_exact_brpo_*.npy |

### 4.2 T~ 接口

| mode | target 来源 | 输出文件 |
|------|------------|---------|
| old | projected depth + rgb gate | target_depth_for_refine_v2_brpo.npy |
| new | score_prob weighted | pseudo_depth_target_joint_v1.npy |
| hybrid | verified composition | pseudo_depth_target_brpo_style_v1.npy |
| exact | strict BRPO target | pseudo_depth_target_exact_brpo_full_target_v1.npy |

### 4.3 G~ 接口

| mode | action 类型 | 输出 |
|------|------------|------|
| clean summary | 无动作 / no-action control | diagnostics only |
| legacy opacity | delayed opacity scale | participation_opacity_scale |
| direct current-step | stochastic Bernoulli opacity masking | participation_opacity_scale + current-step history |

### 4.4 R~ 接口

| mode | topology | 说明 |
|------|---------|------|
| off | sequential | pseudo → real sequential backward |
| brpo_joint_v1 | joint | pseudo + real → joint backward |

---

## 5. 不做的事

- 不继续打磨 verify proxy（已完成 negative proof）
- 不在 M~ contract 上继续微调（已对齐）
- 不再把旧 `+0.0114` G~ 结论当成当前依据
- 不在 G~ legacy delayed opacity 仍负向时推进 O2a/b
- 不把 T~ 剩余 gap 简化成单侧问题
- 不在 observation compare 里同时改 topology 或 G~

---

## 6. 参考

- 状态：[STATUS.md]
- 过程：[CHANGELOG.md]
- G~ clean compare：[docs/archived/2026-04-experiments/G_BRPO_CLEAN_COMPARE_20260421.md]
- T4 compare 执行文档：[docs/T4_EXACT_UPSTREAM_COMPARE_PLAN_20260421.md]
- pseudo_branch 目录迁移：[docs/design/PSEUDO_BRANCH_LAYOUT.md]
- pseudo_branch G~ 迁移记录：[docs/PSEUDO_BRANCH_G_MIGRATION_PHASE1_20260422.md]
- pseudo_branch R~ 迁移记录：[docs/PSEUDO_BRANCH_R_MIGRATION_PHASE2_20260422.md]
- pseudo_branch T~/observation 迁移记录：[docs/PSEUDO_BRANCH_T_OBSERVATION_MIGRATION_PHASE3_20260422.md]
- M~ 详细：[MASK_DESIGN.md]
- T~ 详细：[TARGET_DESIGN.md]
- G~ 详细：[GAUSSIAN_MANAGEMENT_DESIGN.md]
- R~ 详细：[REFINE_DESIGN.md]

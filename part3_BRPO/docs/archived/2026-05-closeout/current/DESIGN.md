# DESIGN.md - Part3 BRPO 设计文档

> 更新时间：2026-05-04 21:30 (Asia/Shanghai)

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

**并行 compare branch（2026-05-04 新增）**：
- `paper_brpo_target_v1`：把 T~ 改成 depth-only producer，取消 `raw_rgb_confidence` gating 与 `render_depth` fallback，只保留 `projected_depth_left/right + projected_valid + fusion_weight + light tau_rel_depth`。
- `paper_brpo_split_v1` / `paper_brpo_split_depthconf_v1`：把 consumer 改成 RGB 只用 `C_m`，depth 再决定是否乘 depth-only `target_confidence`。
- 当前判断：这条 paper-realign 分支已经落成可复现 compare branch，但 first full9 compare 尚未推翻 `exact_brpo_upstream_target_v1 + exact_shared_cm_v1` 的 mainline verdict。

### 1.3 R~ 重构状态（2026-05-04）

R~ 当前必须区分成三层：

1. **legacy standalone R~**：`run_pseudo_refinement_v2.py` 里的 `brpo_joint_v1` / StageA / StageB。它仍是历史 compare 与 replay baseline，但不再代表未来 refine 主线。
2. **backend continuation R~**：`third_party/S3PO-GS/utils/slam_backend_brpo.py` + `third_party/S3PO-GS/slam.py::_maybe_run_brpo_pseudo_continuation()`。这条路径现在降级为 after_opt control / bridge route，不再是最终 landing 目标。
3. **online mapping R~**：`pseudo_branch/integration/*.py` + `third_party/S3PO-GS/utils/slam_frontend.py` + `third_party/S3PO-GS/utils/slam_backend.py`。当前工程主线是把 exact pseudo supervision 放进 backend keyframe event 内部的 runtime slot activation / runtime bundle build / backend debug-export shell，并进一步接入 conservative pseudo-aware mapping block。

当前 landed 结论：supervision 语义保留，但 refine 执行器已经进一步从 continuation 壳转向 online mapping 壳。`run_pseudo_refinement_v2.py` 与 `after_opt continuation` 现在都只保留 reference/control 身份；当前 live landing 已经从“只建 bundle”推进到“single-gap pseudo 真正进入 backend mapping loop”。

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
- 详细 mapping 与阶段顺序见 `docs/design/PSEUDO_BRANCH_LAYOUT.md`；本轮记录见 `docs/archived/2026-04-cleanup-records/PSEUDO_BRANCH_G_MIGRATION_PHASE1_20260422.md`、`docs/archived/2026-04-cleanup-records/PSEUDO_BRANCH_R_MIGRATION_PHASE2_20260422.md`、`docs/archived/2026-04-cleanup-records/PSEUDO_BRANCH_T_OBSERVATION_MIGRATION_PHASE3_20260422.md`、`docs/archived/2026-04-cleanup-records/PSEUDO_BRANCH_M_MIGRATION_PHASE4_20260422.md`、`docs/archived/2026-04-cleanup-records/PSEUDO_BRANCH_COMMON_MIGRATION_PHASE5_20260422.md` 与 `docs/archived/2026-04-cleanup-records/PSEUDO_BRANCH_T_RESIDUAL_CLEANUP_PHASE6_20260422.md`

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
| R~ | online backend mapping shell | 当前 live 主线已从 keyframe-event runtime slot activation + runtime exact bundle/signal/record build，推进到 conservative single-gap pseudo-aware mapping block；当前验证通过的是 `both_only` + small-iter smoke，不是 full online compare |

**当前主线判断**：standalone best 仍是 `exact M~ + exact upstream T~ + clean summary G~ + T1`，但工程主线已经切到 **S3PO online backend mapping shell**；`after_opt continuation` 与 standalone 现在都只保留 control / replay / reference 身份。

**Paper-realign 分支状态**：`paper_cm_only / paper_brpo_target_v1 / paper_brpo_split_*` 已经真实接通并完成 full9 compare，但当前只保留为 diagnostics / compare branch，尚未升为 standalone 或 backend mapping 主线。

---

## 3. 设计判断（固化）

### 3.1 M~ 结论
- exact M~ 与 old M~ 基本等价（差 < 1e-5 PSNR）
- strict BRPO $C_m$ 已基本对齐
- 当前 live exact `C_m` 的 matching layer 已支持两种入口：`sparse_desc_2d`（旧 `FlowMatcher`）与 `dense_pts3d_3d`（新 `Dense3DMatcher`）；升级点保持在 matching layer，自始至终不改 BRPO 离散三档语义
- 落地方案仍分两条：`docs/archived/2026-04-m3d-experiments/BRPO_MASK_DENSE_2D_MATCHING_PLAN_20260424.md`（dense2d，低风险 control / side option）与 `docs/archived/2026-04-m3d-experiments/BRPO_MASK_MAST3R_3D_MATCHING_PLAN_20260424.md`（MASt3R 3D matching，更值得优先落地的主线候选）
- 2026-04-24 已完成 M~ 3D live wiring：`scripts/brpo_build_mask_from_internal_cache.py` 与 `scripts/build_brpo_v2_signal_from_internal_cache.py` 已切到 matcher factory，并把 matcher config / matcher meta 写入产物
- full mechanism validation 与后续 `StageB120 + replay` compare 已完成，详见 `docs/archived/2026-04-m3d-experiments/M3D_FULL_MECHANISM_VALIDATION_20260424.md` 与 `docs/archived/2026-04-m3d-experiments/M3D_STAGEB120_REPLAY_COMPARE_20260424.md`。新的 grounded 结论是：dense3d 的机制增密是真的，`q0.70` 也仍是 dense 内部最优候选，但在真正的长程 replay compare 中，sparse 仍然是 winner。Replay PSNR 为 sparse `24.0045`、q0.70 `23.6660`、q0.80 `23.5816`
- 进一步的 structural forensic（`docs/archived/2026-04-m3d-experiments/M3D_STRUCTURAL_FORENSICS_20260424.md`）表明：旧 live exact M~ 的低 coverage 确实来自 `FlowMatcher + fast_reciprocal_NNs(..., subsample_or_initxy1=8)` 这条 sparse 2D reciprocal matching 路径；因此 `~1.5%–2%` 并不是额外异常，而是旧设计自然结果。
- 同一轮 forensic 也表明 dense3d 接通后，`exact_brpo_upstream_target_v1` 的 target depth / target confidence / source map 确实随 matcher 一起变化；问题不是“只改了 C_m，target depth 没同步”，而是新增 supervision 的结构质量。
- 现在更像是 supervision composition mismatch：dense3d 新增区域主要来自 single-branch 支持，而不是 both-branch 几何；以 q0.70 为例，新增 valid 区域里约 `64.1%` 是 `C_m=0.5`，仅约 `35.9%` 是 `C_m=1.0`。同时 `exact_shared_cm_v1` 实际使用的 effective mask 为 `C_m × valid_mask × target_confidence`，其 valid 区域均值也低于 sparse（q0.70 `0.403` vs sparse `0.480`）。
- 这解释了为什么 dense3d 的 coverage 增长没有转成更好的 downstream replay：下游被喂入了更多单边、较弱、可能更 noisy 的 supervision，最终在 StageB120 中表现为更高的 pseudo/depth loss 与更差 replay。
- 新的 `cm_only` ablation 进一步把问题收窄：在当前 fixed route 下，把 `exact_shared_cm_v1` 改成只用裸 `C_m`（不乘 `valid_mask` / `target_confidence`）后，sparse 与 q070 两个 arm 都出现 replay 小幅退化，同时 pseudo/RGB/depth loss 全部上升。由于当前 exact-upstream 导出中 `C_m>0` 与 `valid_mask>0` 重合，这说明真正起作用的不是 valid-mask 裁剪，而是 `target_confidence` 对 supervision 强度的连续抑制。
- 新的 `rgb_only` ablation 则进一步否定了“depth target 本身是主因”的简单解释：在 fixed route 下，用 `--stageA_disable_depth` 完全移除 pseudo depth loss 后，sparse replay 明显下降（PSNR `-0.0792`），q070 只得到极小表面变化（PSNR `-0.0015`，SSIM/LPIPS 略好），但仍显著落后于 sparse。更像是 dense supervision 的组成/quality mismatch，而不是 depth 项单独出了问题。
- 因此当前 M~ 决策不应是继续做同类 q-sweep，更不应直接把 dense3d 默认化；如果继续推进，应优先回到 BRPO 原方法检查 both-vs-single contract、single-branch target composition 与 confidence weighting 是否与 live 语义存在方法级偏差。

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
- `run_pseudo_refinement_v2.py` 里的 T1 (`brpo_joint_v1`) 现在只应被视为 **legacy standalone topology**，不是未来 refine 主引擎。
- `third_party/S3PO-GS/utils/slam_backend_brpo.py` + `slam.py::_maybe_run_brpo_pseudo_continuation()` 仍保留，但角色已经降级为 after_opt control / bridge route，而不是最终 landing 目标。
- 2026-05-04 已新增 `pseudo_branch/integration/` runtime package，并把 `third_party/S3PO-GS/utils/slam_frontend.py` / `utils/slam_backend.py` 接到 keyframe-event runtime slot activation path：backend 现在可以在真实 keyframe 事件里直接构建 exact backend bundle、exact-upstream signal 和 `BackendPseudoViewRecord`。
- 同日已把 `third_party/S3PO-GS/utils/slam_backend_brpo.py` 提升为 shared pseudo-aware backend engine 的第一版：保留 `run_brpo_pseudo_continuation(...)` control 入口，同时新增 `run_brpo_pseudo_mapping(...)` 供 online mapping 路径直接消费 runtime pseudo records；`utils/slam_backend.py` 也已在 keyframe path 里真正调用该 mapping block，而不再只停留在 bundle/debug export。
- 已完成 three-step grounding：1) `frame_0023` exact-upstream parity 五个核心数组 bitwise 一致；2) `current_window=[0,34]` 的 backend trigger smoke 成功激活 midpoint pseudo slot `frame_id=17`，并在 `enable_pseudo_gradient=false`、`pseudo_map_iters=0` 下保持 Gaussian `xyz` 完全不变；3) conservative Phase 3 smoke 在同一 gap 上成功执行 `pseudo_map_iters=2` 的在线 pseudo-aware mapping，history 同时写出 `loss_real / loss_pseudo / loss_pseudo_pose / loss_pseudo_scene`，Gaussian `xyz max_abs_delta≈0.01921`，且 `update_real_pose=false` 下 real pose 没有被误更新。
- 因此 R~ 当前设计判断已经更新为：**保留 exact supervision，主壳切到 S3PO online backend mapping；先完成 runtime slot/bundle shell，再在下一阶段把 pseudo 真正接进 mapping iterations。**

### 3.5 refine forensic 结论
- `docs/REFINE_FORENSICS_MASTER_20260425.md` 已把当前 refine forensic 六步完整落盘，并与 archived M3D 报告形成主从关系：master doc 负责当前结论与恢复入口，执行长文留在 `docs/archived/2026-04-m3d-experiments/`。
- 旧 live M~ 的低 coverage 现在已经可以定性为“旧 matching 设计自然结果”而不是额外接线 bug：它确实走 `sparse_desc_2d` / `FlowMatcher + fast_reciprocal_NNs(..., subsample_or_initxy1=8)`，所以 `~1.5%–2%` coverage 是旧稀疏 2D reciprocal matching 的直接结果。
- 但 dense3d 已证明“把 coverage 做大”并不足以自动转成更好 replay。更强 pseudo route 的失败表现为：`iter020` 前可略有正收益，`20 -> 40` 后开始持续恶化；`pose_only` 与 `gaussians_only` 都不会单独复现这一点，因此当前更像 joint pseudo pose + Gaussian feedback instability。
- 同一轮 forensic 还说明坏 route 同时具备三件事：初始 target mismatch 更大、single-branch effective supervision mass 更高、以及 real anchor 太弱。最关键的证据是：real-train RGB loss 可以继续下降，而 replay 明显变差。
- 因而当前设计判断应继续保持：不要把 repair 方向简化成“默认 dense3d”或“把离散 `C_m` 改成 continuous mask”。更合理的第一修复轴是 both-vs-single contract、single-branch target composition、pseudo/real balance 与 stronger pseudo route 的稳定化。
- 2026-04-27 的 `dense q070 + 4real+2pseudo` follow-up 又把这个判断收紧了一步：单纯提高 real view 数、降低 pseudo view 数，并不能让 current exact dense contract 自然转正；`current_4r2p_exact` 与 `rgb_only_4r2p_exact` 都比旧 `2+4` dense q070 controls 更差。只有当 pseudo contract 被进一步弱化成 `all-ones + RGB-only` 时，`4real+2pseudo` 才明显转正并达到 `24.1126 PSNR`。因此后续如果要继续修 dense exact route，优先级仍应放在 pseudo contract 本身，而不是先把 minibatch ratio 当主修复轴。
- 同日追加的 `continuous-confidence + RGB-only` dense q0.70 follow-up 也没有改变这个判断：把离散 `C_m` 直接换成连续 `target_confidence` 后，`contconf_rgb_only_4r2p` 只有 `23.5406 PSNR`，相对离散 exact RGB-only 仅 `+0.0210 PSNR`，但仍明显落后于 `allones_rgb_only_4r2p`。因此当前更不该把修复方向理解成"只要从离散改成连续就会更好"；continuous confidence 在这里最多只是很弱的局部修正。
- 2026-05-04 完成 `full8 dense3d q070 active mask compare`：StageB120 下 sparse replay `24.0045` vs dense3d q0.70 `23.9133`，sparse 仍胜 `+0.0912 PSNR`；active mask 统计确认 sparse mean `0.01523`（~1.5%）、dense3d mean `0.18760`（~18.8%）、dense3d 新增 `+0.17424`。这再次验证：dense3d 确实大幅提升 mask coverage，但 replay 未改善，与前序结论一致。问题不是 coverage 本身，而是 supervision composition 与 joint coupling instability。

---

## 4. 接口定义

### 4.1 M~ 接口

| mode | confidence 来源 | 输出文件 |
|------|----------------|---------|
| old | rgb + geometry tier min | joint_confidence_v2.npy |
| new | score_stack 派生 | pseudo_confidence_joint_v1.npy |
| hybrid | support sets | pseudo_confidence_brpo_style_v1.npy |
| paper | fused-domain direct support sets（no geometry verifier） | pseudo_confidence_paper_brpo_*.npy |
| exact | strict BRPO semantics | pseudo_confidence_exact_brpo_*.npy |

### 4.2 T~ 接口

| mode | target 来源 | 输出文件 |
|------|------------|---------|
| old | projected depth + rgb gate | target_depth_for_refine_v2_brpo.npy |
| new | score_prob weighted | pseudo_depth_target_joint_v1.npy |
| hybrid | verified composition | pseudo_depth_target_brpo_style_v1.npy |
| paper | light geometry-only bidirectional projection（depth-only） | pseudo_depth_target_paper_brpo_target_v1.npy |
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
- T4 compare 执行文档：[docs/archived/2026-04-plans-landed/T4_EXACT_UPSTREAM_COMPARE_PLAN_20260421.md]
- pseudo_branch 目录迁移：[docs/design/PSEUDO_BRANCH_LAYOUT.md]
- pseudo_branch G~ 迁移记录：[docs/archived/2026-04-cleanup-records/PSEUDO_BRANCH_G_MIGRATION_PHASE1_20260422.md]
- pseudo_branch R~ 迁移记录：[docs/archived/2026-04-cleanup-records/PSEUDO_BRANCH_R_MIGRATION_PHASE2_20260422.md]
- pseudo_branch T~/observation 迁移记录：[docs/archived/2026-04-cleanup-records/PSEUDO_BRANCH_T_OBSERVATION_MIGRATION_PHASE3_20260422.md]
- M~ 详细：[MASK_DESIGN.md]
- T~ 详细：[TARGET_DESIGN.md]
- G~ 详细：[GAUSSIAN_MANAGEMENT_DESIGN.md]
- R~ 详细：[REFINE_DESIGN.md]


## 2026-05-08 13:18 — Difix device policy
- Difix device policy is dynamic, not physical-GPU hardcoded: launcher chooses CUDA_VISIBLE_DEVICES; backend uses logical cuda:0 mapped by that environment.
- Config switch remains use_difix_restoration; optional difix_enforce_backend_device defaults true for backend-loaded Difix.


## 2026-05-08 14:10 — E6 upper-bound semantics
- `pseudo_rgb_source=gt` means pseudo RGB supervision, matching, and RGB-only C_m/mask use the dataset image.
- Depth supervision remains projected bidirectional exact backend (`depth_generation_mode=projected`), so render depth is still used for geometric verification; only rendered pseudo RGB is bypassed.


## 2026-05-09 — 2IMG+PAIR C_m cap semantics
- C_m is a loss/support confidence, not a depth-value scale. For dense 2IMG+PAIR target generation, C_m cap must be binary support: depth_effective = depth_calibrated * (C_m > 0).
- Discrete C_m values (1.0 both, 0.5 single) should be applied by the consumer loss as weights, never multiplied into target depth.
- Post-fix E7a improves PSNR but still trails E5c and has poor ATE, indicating remaining risk is 2img depth value quality/geometry consistency rather than the now-fixed cap bug alone.


## 2026-05-09 — E7a depth-loss interpretation
- Binary C_m cap is necessary but not sufficient. After fixing it, full 2IMG dense depth still causes trajectory/geometry corruption.
- Shared support does not show a simple global scale error: median E7a/E5c depth ratio is about 1.01, but median abs-rel is about 0.20. The problem is local/value consistency, not one scalar scale.
- Current implementation calibrates against left PAIR projected depth, then applies the same 2IMG target to both branches. Right-branch disagreement is large; late events show right-anchor abs-rel around 0.63.
- Added 2IMG coverage is dominated by single-support pixels, so most new depth supervision is weakly verified. Until repaired, 2IMG depth should be disabled or gated to safer regions such as both-support / anchor-valid / right-left-consistent pixels.
- Evidence: E7a_binarycap_depthoff (same config except match_real_loss_weights=false and lambda_depth=0.0) reaches PSNR 21.633 and stats_final RMSE 0.0619.


## 2026-05-10 E8 C_m local expansion audit

- Fixed C_m expansion metadata/stat reporting: observation summaries now separate raw reciprocal C_m stats from consumed soft-C_m stats; diagnostic sidecar dry-run no longer writes frame outputs; sidecar summaries include depth target filled before/after and complete reject counters.
- Audited E8 at /home/bzhang512/my_storage2_1T/part3_online_mapping_experiments/E8_cm_local_expand_r1_soft. The run is protocol-aligned with E5c except cm_expansion_mode=local_soft_v1 and cm_expansion_apply_to_depth_scope=false.
- Final metrics: E8 after_opt PSNR 20.5222, SSIM 0.6629, LPIPS 0.2684, stats_final RMSE 0.0875. E5c reference: PSNR 21.2012, RMSE 0.0645. E8 degraded.
- Code/production-flow check: raw support is preserved; signal confidence equals cm_expanded_soft and not cm_raw; depth target/valid scope stays raw/projected. No current hard wiring bug was found in the audited E8 artifacts.
- Mechanistic diagnosis: local expansion adds about 8-16 percent image area as RGB-only soft C_m; depth_in_added is 0 for all audited events because apply_to_depth_scope=false and projected depth target remains raw. Since paper_brpo_split_v1 RGB loss normalizes by confidence_mask.sum, added easy/weak pixels dilute raw reciprocal seed RGB gradients by about 10-25 percent while adding no geometric anchor. This is the leading explanation for lower training pseudo losses but worse final PSNR/ATE.
- Next recommendation: do not continue full local_soft_v1 as-is. Test conservative variants: both-only/near-depth-valid expansion, or budget-preserving reweighting that keeps raw seed gradient mass constant; optionally pair with depth-off if isolating pure RGB expansion.


## 2026-05-10 — dense_match_v1 standalone C_m support builder
- Added a new RGB-only support-generation branch distinct from `cm_expansion_mode`: `rgb_only_support_mode=dense_match_v1`.
- Semantics: use the existing reciprocal match points as seeds, then build branch-local support by `disk(radius) -> Gaussian blur -> normalize -> threshold`. No reprojection/depth overlap is applied in this mode.
- Consumer contract is intentionally conservative in v1: this mode replaces the branch `support_mask/confidence_map` used by the `rgb_only_verification` path, but does not modify exact projected depth generation and does not inject a soft `confidence_cm_override`.
- Therefore v1 isolates one question only: whether larger RGB-only support coverage changes online-mapping behavior when depth target semantics stay fixed.
- Non-overlap rule: `dense_match_v1` must not be combined with `cm_expansion_mode`; runtime raises a hard error to keep this branch separate from local soft C_m expansion.
- Config surface:
  - `rgb_only_support_mode: reciprocal_seed | dense_match_v1`
  - `cm_dense_point_radius`
  - `cm_dense_blur_sigma`
  - `cm_dense_blur_kernel` (`0` => auto)
  - `cm_dense_corr_threshold`
  - `cm_dense_seed_mode: binary | confidence_weighted`
  - `cm_dense_normalize_mode: max | p99 | none`
- Debug/production artifacts now preserve both raw and dense views of the support field, enabling direct raw-vs-dense coverage audits per pseudo event.

## 2026-05-10 — dense_match_v1 bridge hardening
- Dense-match enablement requires three distinct layers to agree: YAML config, resolver output, and the actual `RuntimeExactBackendConfig` constructed inside `_maybe_prepare_brpo_runtime_slots()`.
- The residual E9 failure mode was exactly a layer-3 drop: the runtime constructor omitted `rgb_only_support_mode` and `cm_dense_*` fields, so the consumer could execute `reciprocal_seed` despite a dense YAML/config identity.
- Mainline guardrail now writes `exact_cfg_debug.json` for each prepared frame and raises immediately on support-mode mismatch. This makes future dense-route bring-up artifact-verifiable and prevents another silent fallback run.


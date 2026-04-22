# STATUS.md - Part3 Stage1 当前状态

> 更新时间：2026-04-22 04:22 (Asia/Shanghai)

> **书写规范**：
> 1. 只记录"现在"，不记录历史过程（过程在 CHANGELOG）
> 2. 覆盖式更新，直接修改对应版块，不追加
> 3. 不写实验数据（数值在 CHANGELOG / compare json），只写结构/状态/判断
> 4. 状态用 ✅ ⚠️ ❌ 标记，"已完成"不重复列出
> 5. 更新后修改文档顶部时间戳

---

## 1. 概览

当前固定参照线仍是：**RGB-only v2 + gated_rgb0192 + post40_lr03_120**（canonical StageB protocol）。

A/T/B 主线现在更明确了：
- **exact_brpo_cm_old_target_v1 ≈ old A1** 仍成立：strict BRPO `C_m` 已基本对齐，M~ 不再是主瓶颈
- **T4 formal compare 已完成并给出新 winner**：`exact_brpo_upstream_target_v1 + exact_shared_cm_v1` 在 fixed clean G~ / fixed T1 protocol 下赢过 `old A1`、`exact_brpo_cm_old_target_v1`、`exact_brpo_full_target_v1`
- **old A1** 从 observation / target 主线降级为历史强 control；**new T1**（`joint_topology_mode=brpo_joint_v1`）仍是当前 topology 主线
- **exact_brpo_full_target_v1** 的 negative result 仍有效：只做 consumer-side exact target contract 不够，真正增益来自更 upstream 的 verifier backend / projected-depth field
- **A1 verify proxy（`brpo_verify_v1`）与 A1 BRPO-style v1 / v2** 都已完成其 probe 职责，不再是当前落地主线
- **历史 `brpo_direct_v1` 已被重新定性为 hybrid 分支**：它应被理解为 `hybrid_brpo_cm_geo_v1`，不是 strict BRPO
- **G~ clean compare verdict 不变**：direct current-step 仅小幅正向，legacy delayed opacity 明确负向；G~ 当前仍是“已完成语义对齐、但收益有限的 side branch”
- **StageA.5** 继续只保留 optional warmup / control 地位


### 1.3 T~ Alignment 状态

| Phase | 目标 | 状态 | 验收 |
|-------|------|------|------|
| T0 | 锁定 compare arms | ✅ 完成 | `exact_brpo_cm_old_target_v1` 作为 semantics-clean M~/T1 control |
| T1 | exact verifier/backend bundle + branch-native input | ✅ 完成 | `exact_backend_v1/` 导出 + provenance/hit_count/occlusion_reason |
| T2 | exact target field builder | ✅ 完成 | `exact_brpo_upstream_target_v1` + `no_render_fallback=true` |
| T3 | exact loss contract | ✅ 完成 | `build_stageA_loss_exact_shared_cm` + 训练端集成 |
| T4 | formal compare | ✅ 完成 | `exact_brpo_upstream_target_v1` 在 fixed clean G~ / fixed T1 compare 中赢过所有 control arms |

**Phase T4 关键结论**：
- compare root：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260422_t4_exact_upstream_compare_e1`
- `oldA1_newT1_summary_only` arm 最终确认使用 `signal_v2_root=/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_a1_full_brpo_target_signal_full`
- `exactBrpoUpstreamTarget_v1_newT1_summary_only` 明确高于 `oldA1_newT1_summary_only`、`exactBrpoCm_oldTarget_v1_newT1_summary_only`、`exactBrpoFullTarget_v1_newT1_summary_only`
- 当前 T~ 主线应更新为：`exact_brpo_upstream_target_v1 + exact_shared_cm_v1`

一句话判断：**G~ clean compare 之后，真正把 strict BRPO 主线拉成正向 winner 的不是继续 consumer-side 微调，而是 T~ upstream exact backend / projected-depth / target field 的整体对齐。当前 standalone 最优组合已经更新为 `exact M~ + exact upstream T~ + clean summary G~ + T1`。**



### 1.1 数据集 case 状态

| 数据集 | 当前状态 | 当前判断 |
|------|------|------|
| Re10k-1 | ✅ A1/T1 收敛，✅ exact BRPO-C_m compare 已完成，✅ exact BRPO target-side compare 已完成，✅ G~ clean compare 已完成，✅ T4 exact-upstream formal compare 已完成 | 当前 standalone 最优组合已更新为 `exact M~ + exact upstream T~ + clean summary G~ + T1`；`exact_brpo_cm_old_target_v1 ≈ old A1` 继续成立，而 `exact_brpo_upstream_target_v1` 在 fixed clean G~ / fixed T1 compare 下已赢过 `old A1`、`exact_brpo_cm_old_target_v1`、`exact_brpo_full_target_v1`，说明 decisive gain 来自更 upstream 的 verifier backend / projected-depth / target field 对齐 |
| DL3DV-2 | ✅ canonical baseline + repair A 已打通 | 暂未平移 strict BRPO A1 target-contract 路线 / B3 |

---

## 2. 数据结构

### 2.1 Internal cache（源数据）

```text
<run_root>/internal_eval_cache/
├── manifest.json
├── camera_states.json
├── before_opt/
│   ├── point_cloud/point_cloud.ply
│   ├── render_rgb/ / render_depth_npy/
├── after_opt/
│   ├── point_cloud/point_cloud.ply
│   ├── render_rgb/ / render_depth_npy/
```

### 2.2 Pseudo cache（训练输入）

```text
internal_prepare/<prepare_key>/pseudo_cache/
├── manifest.json
└── samples/<frame_id>/
    ├── render_rgb.png / render_depth.npy
    ├── target_rgb_fused.png
    ├── train_confidence_mask_brpo_*.npy
    ├── target_depth_for_refine.npy
    ├── target_depth_for_refine_v2.npy
    ├── projected_depth_left/right.npy
    └── signal_v2/（独立路径）
```

### 2.3 Signal v2（当前并存六条 A1 语义）

```text
signal_v2/frame_<frame_id>/
├── raw_rgb_confidence_v2.npy
├── target_depth_for_refine_v2_brpo.npy
├── depth_supervision_mask_v2_brpo.npy
├── joint_confidence_v2.npy
├── joint_confidence_cont_v2.npy
├── joint_depth_target_v2.npy
├── pseudo_depth_target_joint_v1.npy
├── pseudo_confidence_joint_v1.npy
├── pseudo_confidence_rgb_joint_v1.npy
├── pseudo_confidence_depth_joint_v1.npy
├── pseudo_uncertainty_joint_v1.npy
├── pseudo_source_map_joint_v1.npy
├── pseudo_valid_mask_joint_v1.npy
├── pseudo_confidence_verify_v1.npy
├── pseudo_valid_mask_verify_v1.npy
├── pseudo_depth_target_brpo_style_v1.npy
├── pseudo_confidence_brpo_style_v1.npy
├── pseudo_source_map_brpo_style_v1.npy
├── pseudo_valid_mask_brpo_style_v1.npy
├── pseudo_depth_target_brpo_style_v2.npy
├── pseudo_confidence_brpo_style_v2.npy
├── pseudo_source_map_brpo_style_v2.npy
├── pseudo_valid_mask_brpo_style_v2.npy
├── pseudo_depth_target_brpo_direct_v1.npy
├── pseudo_confidence_brpo_direct_v1.npy
├── pseudo_source_map_brpo_direct_v1.npy
├── pseudo_valid_mask_brpo_direct_v1.npy
└── rgb_mask_meta_v2.json / depth_meta_v2_brpo.json / joint_meta_v2.json /
   joint_observation_meta_v1.json / pseudo_verify_meta_v1.json /
   brpo_style_observation_meta_v1.json / brpo_style_observation_meta_v2.json /
   brpo_direct_observation_meta_v1.json
```

说明：
- **old A1**：`joint_confidence_v2 + joint_depth_target_v2`
- **new A1**：`pseudo_*_joint_v1`
- **verify proxy**：复用 `pseudo_depth_target_joint_v1`，只改 confidence
- **BRPO-style v1**：`pseudo_confidence_brpo_style_v1 + pseudo_depth_target_brpo_style_v1 + pseudo_source_map_brpo_style_v1`
- **BRPO-style v2**：`pseudo_confidence_brpo_style_v2 + pseudo_depth_target_brpo_style_v2 + pseudo_source_map_brpo_style_v2`
- **direct BRPO v1**：`pseudo_confidence_brpo_direct_v1 + pseudo_depth_target_brpo_direct_v1 + pseudo_source_map_brpo_direct_v1`

---

## 3. Pipeline 状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| Internal cache 导出 | ✅ | before/after PLY + render cache |
| Same-ply replay | ✅ | 可用于 baseline/refined 比较 |
| Prepare: select→difix→fusion→verify→pack | ✅ | canonical schema 打通 |
| StageA: source-aware depth loss | ✅ | 闭环可用 |
| StageA: abs prior | ✅ | 固定背景 λ_t=3.0, λ_r=0.1 |
| StageA.5: local gating micro-tune | ⚠️ | 保留，但已降级为可选 warmup / 对照 |
| StageB: bounded baseline | ✅ | post40_lr03_120 |
| SPGM: repair A anchor | ✅ | 当前最稳 policy 配置 |
| SPGM: B1/B2 | ✅ | manager shell + decoupled score + diagnostics 已接通 |
| B3 旧版：xyz deterministic lr scaling | ❌ | 已降级为 diagnostic probe，不再当主线 |
| B3 新版：deterministic participation controller | ⚠️ | boolean selector 路径已完成 conservative compare，但当前仍 weak-negative |
| B3 O0/O1：`deterministic_opacity_participation` | ⚠️ | `participation_score` + opacity attenuation + delayed C1 compare 已完成；当前相对复用 controls 仍微弱负向，暂不进入 O2a/b |
| A1 旧语义：joint support filter | ✅ | 当前 observation 主线 |
| A1 新语义：joint observation rewrite | ⚠️ | 已完成并确认优于 control，但在固定 new T1 下仍低于 old A1 |
| A1 verifier proxy：`brpo_verify_v1` | ❌ | 已完成 compare，当前 negative；保留为已验证过的失败 probe |
| A1 BRPO-style v1：`brpo_style_v1` | ⚠️ | builder + consumer + compare 已完成；方向正确，但已被更严格的 exact / hybrid 分解分析取代 |
| A1 BRPO-style v2：`brpo_style_v2` | ⚠️ | continuous quality + stable-blend 版 compare 已完成；现主要保留为 stable-target contract 对照 |
| A1 hybrid BRPO-C_m 几何门控分支：historical `brpo_direct_v1` / semantic label `hybrid_brpo_cm_geo_v1` | ⚠️ | 已完成 builder + consumer + formal compare；这是 hybrid 分支，不再应被当作 strict BRPO |
| A1 exact BRPO-C_m + old target：`exact_brpo_cm_old_target_v1` | ✅ | 与 old A1 基本等价，是当前 semantics-clean BRPO `C_m` control |
| A1 exact BRPO target-side：`exact_brpo_full_target_v1` | ⚠️ | 已完成 exact target-side compare；结论已固化为“仅做 proxy backend 下的 consumer-side exact 化仍不足以赢 old A1” |
| A1 exact BRPO-C_m + hybrid target：`exact_brpo_cm_hybrid_target_v1` | ⚠️ | 已完成 compare；仍属于 exact-upstream 落地前的过渡 target contract 对照 |
| A1 exact BRPO-C_m + stable target：`exact_brpo_cm_stable_target_v1` | ⚠️ | 已完成 compare；保留为 stable-target contract 对照 |
| A1 exact BRPO upstream target：`exact_brpo_upstream_target_v1` | ✅ | T4 formal compare 已完成；在 fixed clean G~ / fixed T1 protocol 下赢过 old A1 / exact-oldtarget / exact-fulltarget，已升为当前 T~ 主线 |
| T1：`joint_topology_mode=brpo_joint_v1` | ✅ | topology mode + compare + confirmation 已完成 |
| T1 orchestration：StageA.5 optional warmup/control | ✅ | 角色已经固化 |
| A2 geometry-constrained expand | ❌ | 当前 widening 方案不作为主线 |
| B3 stochastic masking | ❌ | 暂未启动；需等待 deterministic participation 更稳的版本 |
| G-BRPO universe control | ✅ |  已实现，与 BRPO 论文对齐 |
| G-BRPO unified score | ✅ |  已实现，公式与 BRPO 论文一致 |
| G-BRPO stochastic action | ✅ |  已实现，Bernoulli sampling 对齐 |
| G-BRPO current-step timing | ✅ |  已实现，probe → action → formal render |
| G-BRPO formal compare | ✅ | clean compare 已完成：`baseline_summary_only` 为 true summary-only control，`direct_brpo_current_step` 为 action-only clean direct BRPO compare；结论是 clean gain 仅小幅正向，legacy delayed opacity 明确负向，详见 `docs/archived/2026-04-experiments/G_BRPO_CLEAN_COMPARE_20260421.md` |

---

## 4. 当前锁定配置

| 类别 | 参数 | 当前值 |
|------|------|--------|
| Abs prior | lambda_abs_t / lambda_abs_r | 3.0 / 0.1 |
| StageB baseline | schedule | post40_lr03_120 |
| StageB baseline | upstream gate | gated_rgb0192 |
| StageB baseline | signal pipeline | RGB-only v2 + depth-sidecar（control） |
| A1 参考主线 | observation semantics | `joint_confidence_v2 + joint_depth_v2` |
| T1 主线 | topology | `joint_topology_mode=brpo_joint_v1` |
| 当前默认候选主线 | observation + target + topology | `exact_brpo_upstream_target_v1 + exact_shared_cm_v1 + clean summary G~ + brpo_joint_v1` |
| A1/T~ 当前 control | semantics-clean control | `exact_brpo_cm_old_target_v1 + brpo_joint_v1 + clean summary G~` |
| A1/T~ 历史强基线 | historical control | `old A1 + new T1` |
| StageA.5 role | orchestration | optional warmup / control |
| SPGM repair A | keep / eta / weight_floor | (1,1,1) / 0.0 / 0.25 |
| B3 v1 | manager mode | `deterministic_participation` |
| B3 v1 | first tested keep | near=1.0 / mid=0.9 / far=0.75 |
| B3 O1/C1 | manager mode | `deterministic_opacity_participation` |
| B3 O1/C1 | last tested opacity floors | near=1.0 / mid=1.0 / far=0.9 |

---

## 5. 当前判断

### 5.1 StageA
- 闭环已修，pose 可更新；StageA-only replay 不是判优指标。

### 5.2 StageB / T1
- `post40_lr03_120` 仍是固定 protocol。
- `joint_topology_mode=brpo_joint_v1` 仍是当前稳定 topology 主线。

### 5.3 StageA.5 角色
- StageA.5 不再是默认主线必经阶段。
- 当前工程定位仍是：**可选 warmup / 对照 / orchestration anchor**。

### 5.4 A1 当前结论
- **old A1** 仍可继续作为当前 observation control，但它已经不再是唯一的语义参考点。
- **exact_brpo_cm_old_target_v1** 与 `old A1` 基本等价（约 `-6.4e-06 PSNR`），因此 strict BRPO-style `C_m` 在当前 fused pseudo-frame proxy 下已经对齐到足够接近 old A1 的水平。
- **new A1** 已证明不是旧 A1 换壳，但在固定 `new T1` 下当前仍明显低于 old / exact-oldtarget control。
- **verify proxy** 已完成 formal compare，当前结果比 old A1 和 current new A1 都更差，已经完成它的排错职责。
- **BRPO-style v1 / v2** 仍然有价值，但现在它们更适合作为中间 builder / stable-target 对照，不再是主对齐入口。
- **historical `brpo_direct_v1`** 现在应明确写成 hybrid：`hybrid_brpo_cm_geo_v1`。它不是 strict BRPO，因为它把 `C_m` 和 geometry gate 混在了一起。

### 5.5 T~ 当前结论
- `exact_brpo_full_target_v1` 的 negative result 继续有效：**只做 proxy backend 下的 consumer-side exact 化，不足以赢 old A1**。
- T4 formal compare 已把结论推进了一步：**`exact_brpo_upstream_target_v1 + exact_shared_cm_v1` 在 fixed clean G~ / fixed T1 protocol 下赢过 `old A1`、`exact_brpo_cm_old_target_v1`、`exact_brpo_full_target_v1`。**
- 这说明 strict BRPO target-side 的决定性收益不在 consumer-side 微调，而在更 upstream 的 verifier backend / projected-depth field / target field 整体对齐。
- 因此当前 T~ 主线应从 “old T~ / exact-oldtarget control” 正式切换到 **exact upstream T~**；old A1 与 exact-oldtarget 继续只保留为 control。

### 5.6 当前对 target-side 的更新判断
- 当前证据已经支持把 `exact_brpo_upstream_target_v1` 升为新的 T~ control / mainline。
- 更准确的说法是：**M~ 已基本对齐；proxy-backend exact target-side 证明了只改 consumer 不够；exact upstream compare 则证明把 verifier/backend/target field 整体拉齐之后，strict BRPO 路线可以转正。**
- 因此 standalone 线上的下一步不应再继续做同口径的小型 T~ compare，而应冻结这套 winner，转向后续集成与复用。


### 5.7 SPGM / G~
- G~ clean compare 已经修干净：`summary_only` 现在是真正的 no-action control；direct BRPO arm 不再混 legacy grad-weight policy；current-step history 也是一 iter 一条记录。
- clean compare 结果是：`direct_brpo_current_step` 相对 clean `baseline_summary_only` 只保留**小幅正向**；`legacy_delayed_opacity` 相对 clean baseline 为**明确负向**。
- 因此当前 G~ 的工程定位应固定为：**语义对齐已完成，但收益有限；可以保留为 side branch，不再当主瓶颈突破口。**

### 5.8 是否继续做完全 BRPO 版本
- **G~ 方向本身不用回退，但也不值得继续优先扩。**
- standalone 线上，当前更合理的动作不是继续在 G~ 或同口径 T~ compare 上打转，而是先**冻结 `exact M~ + exact upstream T~ + clean summary G~ + T1` 这套 winner**。
- 后续大工程的主问题会从“standalone 是否能赢”切换成“这套 winner 怎样以 backend-only 方式集成回 S3PO，而不污染 tracking / frontend”。

### 5.9 DL3DV case
- 暂维持 canonical baseline + repair A 现状；不在 Re10k 的 A1/B3 还未稳定前贸然平移。

---

## 6. 待办

### 当前进行中 ⚠️
- A1 / T~：冻结 `exact_brpo_upstream_target_v1 + exact_shared_cm_v1 + clean summary G~ + brpo_joint_v1` 这套 standalone winner；`exact_brpo_cm_old_target_v1` 与 `old A1` 继续仅作为 semantics-clean / historical controls。
- G~：clean compare 定位不变：direct current-step 只有小幅正向，legacy delayed opacity 明确负向；G~ 维持 side branch，不再继续优先做 O2a/b / aggressive 调参。

### 下一阶段 ⏳
- 大工程主线从 standalone compare 转向 **S3PO backend-only integration**，优先做：
  1. 把 exact M~/T~ winner 的 builder / loss contract 从当前 standalone 实验入口中抽成可复用 backend
  2. 保持 pseudo supervision 只进 mapping / backend refine，不回灌 tracking / frontend
  3. 用 fixed control 复核集成后是否还能复现 standalone winner 的增益
  4. 在不改语义的前提下继续做工程整理：`scripts/` 顶层只保留 live 入口；`pseudo_branch/` 已完成 Phase 1 G~ direct migration（`local_gating/`、`spgm/`、`gaussian_param_groups.py` → `gaussian_management/`），下一步进入 Phase 2 R~ 壳层迁移
- 文档口径继续固定：
  1. G~ 使用 clean compare 口径，不再引用旧 `+0.0114` baseline
  2. 任何后续 compare 都固定 clean G~ / fixed T1 control，不再混入新的 G~ 改动

### 明确不做 🚫
- 不回去继续打磨 `brpo_verify_v1` proxy。
- 不把当前 `brpo_direct_v1` 误写成完整 BRPO implementation。
- 不把当前 A1 剩余 gap 简化成“只差 confidence”或“只差 depth target”单侧问题。
- 不在 observation compare 里同时改 topology 或 G~。
- 不再把 delayed opacity / C0-2 当作当前 G~ 主推进路线。

---

## 7. 参考

- 设计原则：[DESIGN.md]
- 过程记录：[CHANGELOG.md]
- A1 细化分析：[MASK_DESIGN.md]
- B3 / refine 设计：[REFINE_DESIGN.md]
- T~ 落地文档：[docs/T_direct_brpo_alignment_engineering_plan.md]
- T4 compare 执行文档：[docs/T4_EXACT_UPSTREAM_COMPARE_PLAN_20260421.md]
- G~ clean compare 记录：[docs/archived/2026-04-experiments/G_BRPO_CLEAN_COMPARE_20260421.md]
- pseudo_branch G~ 迁移进度：[docs/PSEUDO_BRANCH_G_MIGRATION_PHASE1_20260422.md]

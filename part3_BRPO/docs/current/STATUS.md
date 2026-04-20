# STATUS.md - Part3 Stage1 当前状态

> 更新时间：2026-04-20 16:35 (Asia/Shanghai)

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
- **old A1** 仍是当前 observation control
- **new T1**（`joint_topology_mode=brpo_joint_v1`）仍是当前 topology 主线
- **A1 verify proxy（`brpo_verify_v1`）已完成 negative proof，不再是当前执行方向**
- **A1 BRPO-style v1 / v2 都已完成 builder + consumer + formal compare，但都只是过渡 probe，不是当前落地主线**
- **历史 `brpo_direct_v1` 已被重新定性为 hybrid 分支**：它应被理解为 `hybrid_brpo_cm_geo_v1`，不是 strict BRPO
- **strict BRPO 对齐的关键新结论已经拿到**：`exact_brpo_cm_old_target_v1` 与 `old A1` 基本等价，说明当前主要瓶颈已经从 `C_m` 转移到 target / fallback / stabilization contract
- **exact BRPO target-side compare 也已经补完**：`exact_brpo_full_target_v1`（strict `C_m` + exact BRPO-style target-side proxy + shared-`C_m` depth loss）只有 `24.174488`，仍约 `-0.01325 PSNR` 低于 old A1，且几乎与 `exact_brpo_cm_hybrid_target_v1` 持平
- **B3 deterministic participation** 与 **B3 delayed deterministic opacity participation** 都已完成 C1 compare；当前仍是 weak-negative / no-go，暂不进入 O2a/b
- **StageA.5** 继续只保留 optional warmup / control 地位

一句话判断：**当前 Re10k 默认候选主线仍可记作 `old A1 + new T1`，但 A1 的语义判断已经前进一步：strict BRPO-style `C_m` 本身并不比 old A1 差，真正未对齐的是 target-side contract。未来的 BRPO 对齐主线不该再围绕 `C_m` 猜测，而应固定 exact `C_m` 后直接攻 target / fallback。**

补充判断（来自 `20260420_a1_strict_brpo_alignment_compare_e1` 与 `20260420_a1_exact_brpo_target_compare_e1`）：`exact_brpo_cm_old_target_v1` 相对 `old A1` 只有 `-6.4e-06 PSNR`；而 `exact_brpo_cm_hybrid_target_v1` / `exact_brpo_cm_stable_target_v1` 仍约 `-0.012 PSNR`，`exact_brpo_full_target_v1` 约 `-0.01325 PSNR`。这说明剩余 gap 不只是 current source-aware fallback/stable 契约的问题；在当前 proxy `I_t^{fix}` / projected-depth backend 下，纯 exact BRPO target-side 本身也没有赢过 old A1。

### 1.1 数据集 case 状态

| 数据集 | 当前状态 | 当前判断 |
|------|------|------|
| Re10k-1 | ✅ A1/T1 收敛，✅ A1 verify proxy compare 已完成，✅ A1 BRPO-style v1/v2 compare 已完成，✅ hybrid / exact BRPO-C_m compare 已完成，✅ exact BRPO target-side compare 已完成，✅ B3 boolean compare 已完成，✅ B3 opacity C1 compare 已完成 | 主线仍是 `old A1 + new T1`；`exact_brpo_cm_old_target_v1 ≈ old A1` 说明 strict BRPO `C_m` 已基本对齐；但 `exact_brpo_full_target_v1` 仍约 `-0.01325 PSNR` 低于 old A1，说明 current proxy `I_t^{fix}` / projected-depth backend 下，纯 exact BRPO target-side 也未 landing；historical `brpo_direct_v1` 仍应视为 hybrid 分支；B3 boolean / opacity 两条 delayed path 当前都仍 weak-negative |
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
| A1 exact BRPO target-side：`exact_brpo_full_target_v1` | ⚠️ | 已完成 exact target-side compare；在 shared-`C_m` depth loss 下仍约 `-0.01325 PSNR` 低于 old A1，几乎与 exact-hybrid-target 持平 |
| A1 exact BRPO-C_m + hybrid target：`exact_brpo_cm_hybrid_target_v1` | ⚠️ | 已完成 compare；仍约 `-0.013 PSNR` 低于 old A1，说明 target contract 仍未对齐 |
| A1 exact BRPO-C_m + stable target：`exact_brpo_cm_stable_target_v1` | ⚠️ | 比 exact+hypbrid-target 略好，但仍约 `-0.012 PSNR` 低于 old A1 |
| T1：`joint_topology_mode=brpo_joint_v1` | ✅ | topology mode + compare + confirmation 已完成 |
| T1 orchestration：StageA.5 optional warmup/control | ✅ | 角色已经固化 |
| A2 geometry-constrained expand | ❌ | 当前 widening 方案不作为主线 |
| B3 stochastic masking | ❌ | 暂未启动；需等待 deterministic participation 更稳的版本 |

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
| 当前默认候选主线 | observation + topology | `old A1 + new T1`（但 future BRPO 对齐应以 `exact_brpo_cm_old_target_v1` 为 semantics-clean control） |
| A1 当前研究线 | observation semantics | `exact BRPO C_m` 固定 + target/fallback contract 对齐（old / hybrid / stable target 三臂已完成） |
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

### 5.5 strict BRPO `C_m` 与 current hybrid / target contract 的关系
- `exact_brpo_cm_old_target_v1 ≈ old A1`：说明 **old A1 不是靠某个必须保留的非 BRPO confidence 才赢**。
- `exact_brpo_cm_hybrid_target_v1` 仍比 old A1 低约 `-0.013 PSNR`：说明把 `C_m` 改成 strict BRPO 并不会自动救回 hybrid target contract。
- `exact_brpo_cm_stable_target_v1` 比 `exact_brpo_cm_hybrid_target_v1` 略好（约 `+0.00096 PSNR`），但仍比 old A1 低约 `-0.01203 PSNR`：说明 stable-target blend 有一点帮助，但 target/fallback contract 仍未接住 old A1 的稳定性。
- `exact_brpo_full_target_v1`（strict `C_m` + exact BRPO-style target-side proxy + shared-`C_m` depth loss）只有 `24.174488`，相对 old A1 约 `-0.01325 PSNR`，且只比 `exact_brpo_cm_hybrid_target_v1` 高约 `+0.00014 PSNR`：说明当前 proxy `I_t^{fix}` / projected-depth backend 下，纯 exact BRPO target-side 本身也没有带来新的 replay 优势。
- builder 级检查也支持这个判断：`exact_brpo_full_target_v1` 与 `exact_brpo_cm_hybrid_target_v1` 的 target/source contract 几乎相同；frame 级 target array 差异只有浮点噪声量级，source map 也相同。现在的主要限制更像是 upstream proxy supervision field，而不只是 consumer-side source-aware 权重。

### 5.6 当前对 target-side 的更新判断
- 当前证据不再支持“只要把 consumer 再往 exact BRPO 挪一步就能救回 old A1”。
- 更准确的说法是：**现在 `C_m` 已基本对齐；target-side exactness 也已直接试过；但在 current proxy `I_t^{fix}` / projected-depth backend 下，纯 exact BRPO target-side 仍未落地。**
- 因此下一步若还坚持 exact BRPO 路线，重心应上移到更 upstream 的 Layer B proxy：`I_t^{fix}` 质量、verifier backend、以及 projected-depth supervision field 本身，而不是继续只在 stable/fallback 权重上打转。


### 5.7 SPGM / B3
- B3 第一版已经真正把动作位置从 **post-backward grad scaling** 推到 **pre-render participation control**。
- 本轮又进一步把动作变量从 **boolean render selection** 推到 **delayed deterministic opacity attenuation**，说明 O0/O1 wiring 已真实成立。
- 但 C1 formal compare 中，opacity path 相对复用的 `summary_only` 与旧 boolean conservative arm 仍是**轻微负向**，因此当前结论仍不是 landing，也**不应直接进入 O2a/b**。

### 5.8 是否继续做完全 BRPO 版本
- **应该继续，而且现在路径比之前更清楚。**
- 不应回退去打磨 proxy，也不应再把“是不是 BRPO”集中在 `C_m` 上反复猜。
- 更合理的下一步是：**固定 exact `C_m`（以 `exact_brpo_cm_old_target_v1` 为 control），直接做 target-side contract 对齐：old target → hybrid target → stable target → next exact BRPO target patch。**

### 5.9 DL3DV case
- 暂维持 canonical baseline + repair A 现状；不在 Re10k 的 A1/B3 还未稳定前贸然平移。

---

## 6. 待办

### 当前进行中 ⚠️
- A1：`exact_brpo_cm_old_target_v1` 继续作为 semantics-clean control；`exact_brpo_full_target_v1` 已证明“纯 exact BRPO target-side + shared-C_m depth loss”在 current proxy backend 下仍不赢 old A1。下一步若坚持 exact BRPO，应上移到 `I_t^{fix}` / verifier / projected-depth backend 本身，而不是继续只打 stable/fallback 权重。
- B3：当前 delayed opacity C1 仍未过关；如果继续，应先回到 C0-2 的小步 candidate-law / opacity-law 诊断，而不是直接推进 O2a/b。

### 下一阶段 ⏳
- 如果继续推进 A1，优先做：
  1. 明确选择路线：`exact BRPO upstream re-entry` vs `replay-first hybrid target contract`
  2. 若走 exact BRPO：直接检查 `I_t^{fix}` proxy / verifier backend / projected-depth field，避免再把时间花在 consumer-side source-aware 权重细修上
  3. 若走 replay-first：承认 `hybrid/stable target` 是工程分支，并在该标签下继续优化，而不要再把它表述成纯 BRPO
- 如果继续推进 B3，优先做：
  1. 用已接入的 C0 history 字段指导 C0-2 小步修改
  2. 只重跑新 opacity 臂，复用旧 control
  3. **只有当 delayed opacity 不再弱负时**，才进入 O2a/b

### 明确不做 🚫
- 不回去继续打磨 `brpo_verify_v1` proxy。
- 不把当前 `brpo_direct_v1` 误写成完整 BRPO implementation。
- 不把当前 A1 剩余 gap 简化成“只差 confidence”或“只差 depth target”单侧问题。
- 不在 observation compare 里同时改 topology 或 B3。
- 不在 current delayed opacity path 仍弱负时直接推进 O2a/b。

---

## 7. 参考

- 设计原则：[DESIGN.md]
- 过程记录：[CHANGELOG.md]
- A1 细化分析：[MASK_DESIGN.md]
- B3 / refine 设计：[REFINE_DESIGN.md]
- 总规划：[BRPO_A1_B3_reexploration_master_plan.md]
- A1 落地文档：[A1_verifier_decoupled_pseudo_observation_engineering_plan.md]

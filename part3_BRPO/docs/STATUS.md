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
- **old A1** 仍是当前 observation 主线
- **new T1**（`joint_topology_mode=brpo_joint_v1`）仍是当前 topology 主线
- **A1 verify proxy（`brpo_verify_v1`）已完成 negative proof，不再是当前执行方向**
- **A1 BRPO-style v1 / v2 都已完成 builder + consumer + formal compare，但都只是过渡 probe，不是当前落地主线**
- **A1 direct BRPO v1（`brpo_direct_v1`）已完成 builder + consumer + formal compare；它是当前 A1 里最接近“直接 BRPO 语义”的已落地版本，也是当前最值得继续的 A1 研究线**
- **B3 deterministic participation** 与 **B3 delayed deterministic opacity participation** 都已完成 C1 compare；当前仍是 weak-negative / no-go，暂不进入 O2a/b
- **StageA.5** 继续只保留 optional warmup / control 地位

一句话判断：**当前 Re10k 默认候选主线仍是 `old A1 + new T1`。A1 方向已经从 verify proxy / style-v1/v2 进一步推进到 `brpo_direct_v1`；它明显优于 current new A1，也优于 style_v2，但仍未超过 old A1，因此 direct BRPO 语义本身已被证明确实更对，却还没有完成对 old A1 的接管。**

补充判断（来自最新 direct compare + signal meta）：`brpo_direct_v1` 与 `brpo_style_v2` 的平均 valid coverage 基本相同，但 `brpo_direct_v1` 的正样本 confidence 均值已经接近 old A1；这说明剩余 gap 更像是 **target builder / exact supervision contract** 问题，而不再像“把 confidence 再调强一点”就能解决的问题。

### 1.1 数据集 case 状态

| 数据集 | 当前状态 | 当前判断 |
|------|------|------|
| Re10k-1 | ✅ A1/T1 收敛，✅ A1 verify proxy compare 已完成，✅ A1 BRPO-style v1/v2 compare 已完成，✅ A1 direct BRPO compare 已完成，✅ B3 boolean compare 已完成，✅ B3 opacity C1 compare 已完成 | 主线仍是 `old A1 + new T1`；`brpo_direct_v1` 是当前最强的新 A1 分支，但仍略低于 old A1；B3 boolean / opacity 两条 delayed path 当前都仍 weak-negative |
| DL3DV-2 | ✅ canonical baseline + repair A 已打通 | 暂未平移新 A1 direct BRPO / B3 |

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
| A1 BRPO-style v1：`brpo_style_v1` | ⚠️ | builder + consumer + compare 已完成；方向正确，但已被 direct BRPO v1 超过 |
| A1 BRPO-style v2：`brpo_style_v2` | ⚠️ | continuous quality + stable-blend 版 compare 已完成；当前仍低于 direct BRPO v1 与 old A1 |
| A1 direct BRPO v1：`brpo_direct_v1` | ⚠️ | builder + consumer + formal compare 已完成；当前是最强的新 A1 分支，但仍未超过 old A1 |
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
| 当前默认候选主线 | observation + topology | `old A1 + new T1` |
| A1 当前研究线 | observation semantics | `pseudo_observation_mode=brpo_direct_v1` |
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
- **old A1** 仍是当前 observation 主线。
- **new A1** 已证明不是旧 A1 换壳，但在固定 `new T1` 下当前仍未取代 old A1。
- **verify proxy** 已完成 formal compare，当前结果比 old A1 和 current new A1 都更差，已经完成它的排错职责。
- **BRPO-style v1 / v2** 都已证明“朝 BRPO-style semantics 走”方向是对的，但它们都还不是最终 landing 版本。
- **direct BRPO v1** 已完成 formal compare：它明显强于 current new A1，也优于 style_v2，但整体仍略低于 old A1，当前还不能 landing。

### 5.5 direct BRPO v1 与真正 BRPO 的关系
- 当前 `brpo_direct_v1` 已经更直接抓到了 BRPO A1 的核心形式：**shared `C_m` + fused-frame reciprocal correspondence support + 同源 projected-depth composition**。
- 但它还不是完整 BRPO implementation，至少还有三处差异：
  1. pseudo-frame 目前仍复用现有 `target_rgb_fused.png` 作为工程代理，不是完整论文版 `I_t^{fix}`；
  2. `valid_left / valid_right` 仍建立在现有 matcher support + overlap validity 上，不是更独立的 direct verifier backend；
  3. `depth_target_brpo_direct_v1` 当前是 overlap-conf weighted projected depth composition，没有接住 old A1 / 更完整 BRPO builder 里的稳定化 / fallback contract。

### 5.6 为什么当前 direct BRPO v1 仍差于 old A1
- 从最新 compare 看，**大方向已经再次被确认是对的**：`brpo_direct_v1` 相比 current new A1 回升明显，也比 style_v2 更强。
- 但它仍略低于 old A1，且 latest signal meta 显示：
  1. `brpo_direct_v1` 与 `brpo_style_v2` 的平均 valid coverage 基本相同；
  2. `brpo_direct_v1` 的正样本 confidence 均值已经接近 old A1；
  3. 但 old A1 仍然赢。
- 这说明当前剩余缺口更像是：
  1. **target builder 稳定性 / fallback contract** 仍不如 old A1；
  2. direct verifier support 的 exact pixel set 与 old A1 的有效 supervision contract 仍有细小但重要的偏差；
  3. 问题已不再像“再调一下 confidence 强度”就能解决。

### 5.7 SPGM / B3
- B3 第一版已经真正把动作位置从 **post-backward grad scaling** 推到 **pre-render participation control**。
- 本轮又进一步把动作变量从 **boolean render selection** 推到 **delayed deterministic opacity attenuation**，说明 O0/O1 wiring 已真实成立。
- 但 C1 formal compare 中，opacity path 相对复用的 `summary_only` 与旧 boolean conservative arm 仍是**轻微负向**，因此当前结论仍不是 landing，也**不应直接进入 O2a/b**。

### 5.8 是否继续做完全 BRPO 版本
- **应该继续，但下一步要更有针对性。**
- 不应回退去打磨 proxy，也不应继续做 conf-only / depth-only 单侧开关扫描。
- 更合理的下一步是：**先对 `old A1` 与 `brpo_direct_v1` 做 residual-gap diagnosis，直接找 target builder / verifier backend / exact valid set 的差异，再决定下一版 direct BRPO patch。**

### 5.9 DL3DV case
- 暂维持 canonical baseline + repair A 现状；不在 Re10k 的 A1/B3 还未稳定前贸然平移。

---

## 6. 待办

### 当前进行中 ⚠️
- A1：围绕 `old A1` vs `brpo_direct_v1` 做一轮 residual-gap diagnosis，重点看 valid-set disagreement、target residual、fallback / stable-target 语义差异，而不是继续做单侧 toggle。
- B3：当前 delayed opacity C1 仍未过关；如果继续，应先回到 C0-2 的小步 candidate-law / opacity-law 诊断，而不是直接推进 O2a/b。

### 下一阶段 ⏳
- 如果继续推进 A1，优先做：
  1. old A1 vs direct BRPO v1 的 supervision-set / target-builder 差异诊断
  2. 更接近 BRPO 的 direct verifier backend 或更稳的 direct target builder（两者一次只改一侧）
  3. 再做新一轮 `old A1 / current new A1 / brpo_style_v2 / brpo_direct_v1 / next direct patch` compare
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

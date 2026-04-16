# STATUS.md 写作规范

> 更新时间：2026-04-16 01:49 CST
> 本文件记录 Part3 Stage1 的**当前状态**，每次有实质性进展时更新对应版块。

## 写作规范

### 1. 结构要求

必须包含以下版块（按顺序）：
1. **概览**：一句话描述当前阶段目标
2. **数据结构**：目录结构、文件格式、schema
3. **代码脚本**：脚本清单、职责、位置、状态
4. **Pipeline**：流程图、输入输出
5. **配置参数**：关键参数及其含义
6. **状态**：已完成 / 进行中 / 待办

### 2. 禁止

- 同一信息在多个版块重复记录
- 追加式更新（应直接修改对应版块）
- 过程细节（放到 CHANGELOG.md）
- 历史信息（放到 CHANGELOG.md 或删除）

### 3. 引用规则

- 引用其他版块：`[参见 §数据结构]`
- 引用其他文档：`[参见 DESIGN.md §3]`
- 引用历史记录：`[参见 CHANGELOG.md 2026-04-12]`

### 4. 更新规则

1. 每次更新前，先确认信息属于哪个版块
2. 直接修改对应版块，不要追加
3. 更新后修改文档顶部的 `更新时间`
4. 如有重要变更，在 CHANGELOG.md 记录

### 5. 格式规范

- 目录结构用代码块 + 注释
- 表格用标准 markdown
- 关键路径用行内代码
- 状态用 ✅ ⚠️ ❌ 标记

---

# Part3 Stage1 现状

## 1. 概览

当前主线仍然是 **mask-problem route on top of Re10k-1 full internal route**，但 2026-04-15 的最新实测已经把优先级进一步推进到六个明确状态：
- 已完成 `P0`：在 E1 winner 的 canonical legacy root 上完成 split abs prior 标定，当前固定背景配置收敛为 `lambda_abs_t=3.0`、`lambda_abs_r=0.1`；
- 已完成 `P1A`：在同一 E1 winner 上补齐 8-frame `signal_v2` 并跑完 `legacy / v2-rgb-only / v2-full` 的 StageA-only 短对照；
- 已确认一个关键结构事实：当前纯 `stage_mode=stageA` 不会更新 Gaussian，`refined_gaussians.ply` 与输入 `BASE_PLY` hash 完全一致，因此 replay-on-PLY 在 StageA-only compare 中只是 identity sanity check，不是判优指标；
- 已完成 `P2` 第一版工程接入：`run_pseudo_refinement_v2.py` 已新增 `pseudo_local_gating_*` CLI，`pseudo_branch/local_gating/` 已创建，StageA.5 / StageB 的 pseudo-side local gating 路径已经接通并通过 smoke；
- 已完成 `P2-B`：8-frame `StageA.5` 的 `legacy ungated vs gated` short compare 已跑完，当前结论是 gated 相比 ungated 只有极轻微 replay 优势，但没有出现显著改善，也没有把 depth loss 压没；
- 已完成 `P2-C`：带 real branch 的 `StageB ungated vs gated` short compare 也已跑完，当前结论是 gated 不会明显误伤 real branch，但仍然没有展现出足够明确的收益；
- 已确认若要保留 v2 分支，当前更值得保留的是 `RGB-only v2`，而不是直接让 `full v2 depth` 接管默认训练口径；
- 当前 `P2-D` 诊断已完成：current threshold（`min_verified=0.01 / min_rgb=0.01 / max_fallback=0.995`）在 legacy 上基本等于 no-op，因此 StageA.5 / StageB 的 `0 rejection` 是阈值过松而不是实现失效；
- 已完成 `P2-E`：legacy `StageA.5` threshold calibration（`min_verified_ratio=0.02 / 0.03`）已把 gate 拉进真实 reject 区，但 replay 提升仍然极弱；当前若只保留一个 calibrated legacy gate 参考，`vr=0.02` 比 `vr=0.03` 更合适；
- 已完成 `P2-F`：`RGB-only v2` 的 `StageA.5` ungated vs branch-specific gated（`min_rgb_mask_ratio=0.0192`）short compare 已跑完，结果表明这条支路在 `StageA.5` replay 上明显强于 legacy，而且 gating 还能带来小幅正增益；
- 已完成 `P2-G`：固定 `signal-aware-8` 的 `StageB` real-branch short compare 已跑完，结果表明 `RGB-only v2 + gated_rgb0192` 在 joint refine 下仍然明显优于 legacy，也优于同支路 ungated；
- 因而当前新的判断是：`RGB-only v2 + gated_rgb0192` 已经可以提升为当前主候选 refine 分支；下一步更该做的是更长一点的 `StageB` / 完整 schedule verify，而不是先做 raw RGB densify。

一句话说：**现在已经不是“要不要从 legacy 切到 v2”的问题，而是：`RGB-only v2 + gated_rgb0192` 已经在 `StageA.5` 和 `StageB` 两层都站住了；当前更合理的下一步是做更长一点的验证来确认它是不是稳定主线，而不是急着做 raw RGB densify。**

## 2. 数据结构

### 2.1 当前正式源数据（Re10k-1 full internal rerun）

```text
/home/bzhang512/CV_Project/output/part2_s3po/re10k-1/
  s3po_re10k-1_full_internal_cache/
  Re10k-1_part2_s3po/2026-04-11-05-33-58/
```

其中正式 internal 源输入在：

```text
<run_root>/internal_eval_cache/
├── manifest.json
├── camera_states.json
├── before_opt/
│   ├── point_cloud/point_cloud.ply
│   ├── render_rgb/
│   ├── render_depth_npy/
│   └── stage_meta.json
├── after_opt/
│   ├── point_cloud/point_cloud.ply
│   ├── render_rgb/
│   ├── render_depth_npy/
│   └── stage_meta.json
├── replay_eval/
├── brpo_phaseB/
└── brpo_phaseC/
```

说明：
- `camera_states.json` 当前共享一份；
- `before_opt / after_opt` 分别保存自己的 PLY 与 render cache；
- 正式输入统一认 `internal_eval_cache/<stage>/...`。

### 2.2 当前 internal prepare 正式层

当前主 prototype（BRPO完整链路）：

```text
<run_root>/internal_prepare/re10k1__internal_afteropt__brpo_proto_v4_stage3/
```

A.5 midpoint 扩展实验使用：

```text
<run_root>/internal_prepare/re10k1__internal_afteropt__midpoint_proto_v1/
```

说明：该 midpoint 版采用 `kf-gap midpoint` 选帧（8帧：17/51/86/121/156/190/225/260）；由于当前环境中 `brpo_build_mask_from_internal_cache.py` 依赖的 MASt3R 加载报错（`load_model_as_safetensor`），此前 A.5 先使用 fused legacy mask 路径完成对照。当前代码状态已前移三步：1. `prepare_stage1_difix_dataset_s3po_internal.py::stage_fusion()` 已输出 BRPO-style `target ↔ reference overlap confidence` 融合结果；2. 新增 `signal_v2/` 路径可独立产出 fused RGB mask v2 与 depth supervision v2，且不与 legacy `verify / pack` 混写；3. `run_pseudo_refinement_v2.py` 已新增 `signal_pipeline=brpo_v2` 读取路径，并已在 3-frame StageA smoke 中真实读取 `signal_v2` 产物。

核心结构：

```text
internal_prepare/<prepare_key>/
├── manifests/
├── inputs/
├── difix/
├── fusion/
├── verification/
└── pseudo_cache/
    ├── manifest.json
    └── samples/<frame_id>/
        ├── render_rgb.png
        ├── render_depth.npy
        ├── target_rgb_left.png
        ├── target_rgb_right.png
        ├── target_rgb_fused.png
        ├── target_depth.npy
        ├── target_depth_for_refine.npy
        ├── target_depth_for_refine_source_map.npy
        ├── target_depth_for_refine_meta.json
        ├── target_depth_for_refine_v2.npy
        ├── target_depth_dense_source_map.npy
        ├── depth_correction_seed.npy
        ├── depth_correction_dense.npy
        ├── depth_seed_valid_mask.npy
        ├── depth_dense_valid_mask.npy
        ├── depth_densify_meta.json
        ├── projected_depth_left.npy
        ├── projected_depth_right.npy
        ├── projected_depth_valid_left.npy
        ├── projected_depth_valid_right.npy
        ├── seed_support_*.npy
        ├── train_confidence_mask_brpo_*.npy
        ├── confidence_mask_brpo_*.npy
        ├── support_*.npy
        ├── fusion_meta.json
        └── verification_meta.json
```

### 2.3 当前锁定中的 canonical schema

当前 sample schema 已经明确分层：

```text
samples/<frame_id>/
├── render_rgb / render_depth
├── target_rgb_left / target_rgb_right / target_rgb_fused
├── seed_support_*.npy
├── train_confidence_mask_brpo_*.npy
├── projected_depth_{left,right}.npy
├── target_depth_for_refine.npy           # M3 target
├── target_depth_for_refine_v2.npy        # M5 densified target
├── *_source_map.npy                      # 区域来源图
└── source_meta / fusion_meta / verification_meta / depth_densify_meta
```

同时，当前已新增一个**与旧链路隔离**的输出分支：

```text
signal_v2/frame_<frame_id>/
├── raw_rgb_confidence_v2.npy
├── raw_rgb_confidence_cont_v2.npy
├── rgb_support_{left,right,both,single}_v2.npy
├── target_depth_for_refine_v2_brpo.npy
├── target_depth_source_map_v2_brpo.npy
├── depth_supervision_mask_v2_brpo.npy
├── rgb_mask_meta_v2.json
└── depth_meta_v2_brpo.json
```

当前状态：
- 旧 `train_mask` 与 `verified depth` 仍保留，用于 legacy 路径；
- 新 `raw_rgb_confidence_v2` 来自 fused RGB ↔ reference RGB correspondence，不依赖旧 depth seed；
- 新 `target_depth_for_refine_v2_brpo` 只写入 `signal_v2/`，没有覆盖旧 `target_depth_for_refine{,_v2}`。

### 2.4 当前实验输出层

```text
/home/bzhang512/CV_Project/output/part3_stage1_internal/re10k-1/full/
├── 2026-04-12_m45_stageA_eval/
├── 2026-04-12_m50_m51_eval/
├── 2026-04-12_m52_stageA_loss_eval/
├── 2026-04-12_m54_pose_fix_smoke/
├── 2026-04-12_m55_pose_fix_scale_eval/
└── 2026-04-12_m56_abs_pose_eval/
```

当前最重要结果目录：
- **M4.5**：`.../2026-04-12_m45_stageA_eval/`
- **M5-0 / M5-1**：`.../2026-04-12_m50_m51_eval/analysis/`
- **M5-2**：`.../2026-04-12_m52_stageA_loss_eval/`
- **M5-3**：`.../2026-04-12_m54_pose_fix_smoke/` 与 `.../2026-04-12_m55_pose_fix_scale_eval/`
- **M5-4**：`.../2026-04-12_m56_abs_pose_eval/`

## 3. 代码脚本

### 3.1 当前主线脚本

路径：`/home/bzhang512/CV_Project/part3_BRPO/scripts/`

| 文件 | 职责 | 状态 | 备注 |
|------|------|------|------|
| `replay_internal_eval.py` | 基于 saved internal camera states 做 replay eval | ✅ 可用 | same-ply consistency 已验证通过 |
| `prepare_stage1_difix_dataset_s3po_internal.py` | internal `select / difix / fusion / verify / pack` | ✅ 可用 | `stage_fusion()` 已切到 BRPO-style overlap-confidence 口径；`verify / pack` 仍是旧 mask/depth 路径 |
| `brpo_build_mask_from_internal_cache.py` | 从 internal cache 构建 verification / seed / train mask / projected depth | ✅ 可用 | 当前仍是旧 mask/depth 主线；尚未切到 fused-RGB-first 的 v2 mask 语义 |
| `run_pseudo_refinement.py` | v1 standalone refine 入口 | ✅ 可用 | fixed-pose appearance tuning |
| `run_pseudo_refinement_v2.py` | BRPO-style refine v2 / Stage A 入口 | ✅ 可用 | 已支持 split+scaled abs prior（`lambda_abs_t/r + robust + scene_scale`）、`signal_pipeline=brpo_v2`，并已新增 `pseudo_local_gating_*` CLI、StageA.5 grad mask 与 StageB split backward |
| `analyze_m5_depth_signal.py` | M5-0：分析当前 depth signal 在 train-mask 内的结构 | ✅ 可用 | 不改训练，仅诊断 |
| `materialize_m5_depth_targets.py` | M5-1：写出 `target_depth_for_refine_v2` 与 densify 产物 | ✅ 可用 | 当前 selected 参数已验证 |
| `diagnose_stageA_gradients.py` | M5 diagnosis：检查 forward sensitivity 与 grad 连通性 | ✅ 可用 | 当前已发现 Stage A pose path 异常 |
| `diagnose_stageA_loss_contrib.py` | A+B 诊断：单步按 loss 分支统计 rot/trans/exp 梯度贡献 | ✅ 可用 | 用于筛选 abs prior 网格与解释 tradeoff |
| `build_brpo_v2_signal_from_internal_cache.py` | 从 fused RGB 生成隔离的 mask v2 + depth supervision v2 | ✅ 可用 | 输出到 `signal_v2/`，不改写 legacy `verify / pack` 产物 |

### 3.2 pseudo_branch

路径：`/home/bzhang512/CV_Project/part3_BRPO/pseudo_branch/`

| 文件 | 职责 | 状态 | 备注 |
|------|------|------|------|
| `brpo_reprojection_verify.py` | 单分支几何验证 + pseudo-view sparse verified depth | ✅ 可用 | 当前用于 M1 / M3 upstream verify |
| `brpo_confidence_mask.py` | seed-support / confidence alias 输出 | ✅ 可用 | 当前已保留 compatibility alias |
| `brpo_train_mask.py` | `seed_support → train_confidence_mask` propagation | ✅ 可用 | 当前默认研究区间接近 `10% ~ 25%` coverage |
| `brpo_depth_target.py` | 组装 M3/M5 depth target | ✅ 可用 | 当前已支持 `v1 + v2` 双版本 |
| `brpo_depth_densify.py` | M5：densify log-depth correction field | ✅ 可用 | 当前首版 patch-wise densify 已跑通 |
| `pseudo_camera_state.py` / `pseudo_loss_v2.py` / `pseudo_refine_scheduler.py` | Stage A pseudo camera + loss + optimizer | ✅ 可用 | 已补 `apply_pose_residual_()`；已接入 absolute pose prior（`R0/T0`, `SE3_log`, `lambda_abs_pose`） |
| `pseudo_branch/local_gating/` | pseudo-side local Gaussian gating / grad mask / history schema | ✅ 首版已接通 | 当前已包含 signal gate、visibility union、grad mask、history 序列化；默认先服务 `StageA.5 + xyz-only + hard gating` |
| `pseudo_fusion.py` | left/right repaired RGB 融合 | ✅ 已接入 | 当前已改为 `target ↔ reference overlap confidence` 权重，并导出 `fusion_weight_* / overlap_conf_* / overlap_mask_*` |
| `pseudo_branch/brpo_v2_signal/` | 隔离的 fused-RGB mask v2 + depth supervision v2 | ✅ 可用 | `rgb_mask_inference.py` 不依赖旧 depth seed；`depth_supervision_v2.py` 只写 `signal_v2/` |

### 3.3 下一阶段预计触达的代码

| 文件 | 计划改动 |
|------|----------|
| `run_pseudo_refinement_v2.py` | 已完成 `pseudo_local_gating_*` CLI、StageA.5 grad mask 与 StageB split backward；已跑完 8-frame `StageA.5` gated vs ungated short compare，下一步转为做带 real branch 的 `StageB` gated vs ungated short compare |
| `pseudo_branch/local_gating/` | 已完成第一版 hard gating 骨架；下一步视 compare 结果决定是否补 `xyz+opacity`、soft gating 与更细的 signal rule |
| `pseudo_cache / signal_v2` 桥接层 | 当前暂不扩张 schema；先等 gating compare 回答 `legacy` 还是 `RGB-only v2` 更值得继续挂接 |
| `docs/*` | 根据 P2 smoke 与后续 compare 更新默认 gate、验收口径和下一阶段边界 |

## 4. Pipeline

### 4.1 当前已跑通主线（到 M5-2）

```text
part2 full rerun
  ↓
internal_eval_cache
  ↓
same-ply replay consistency
  ↓
internal prepare: select
  ↓
Difix (left/right)
  ↓
BRPO-style fusion v1
  ↓
verification (branch_first | fused_first)
  ↓
seed_support
  ├── propagation -> train_mask
  └── projected_depth_left/right
            ↓
      M3 blended target_depth_for_refine
            ↓
      M5 densify -> target_depth_for_refine_v2
            ↓
run_pseudo_refinement_v2.py Stage A consumer
```

说明：当前主线已拆成新旧两条并行语义：1. `fusion` 已切到基于 pseudo render depth + ref rendered depth + camera states 的 `target ↔ reference overlap confidence`；2. `signal_v2` 已新增 fused RGB-first 的 mask/depth 分支，其中 mask 来自 fused RGB ↔ reference RGB correspondence，depth supervision 单独从 projected depth + fusion weight 生成；legacy `verification / train_mask / target_depth` 仍保留但不与其混写。

### 4.2 当前最关键实验闭环

```text
M4.5 long eval
  ├── blended_depth_long
  └── render_depth_only_long
  ↓
M5-0 signal diagnosis
  ↓
M5-1 densify target
  ↓
M5-2 source-aware depth loss
  ↓
M5-3 pose闭环修复 + M5-4 absolute pose prior接入与烟雾对照
```

### 4.3 当前结论对应的流程位置

```text
verify / pack upstream           -> 已打通 ✅
M5 densified target generation   -> 已打通 ✅
source-aware depth loss wiring   -> 已打通 ✅
Stage A 闭环 pose 优化           -> 已打通（弱有效）⚠️
absolute pose prior wiring       -> 已打通（权重未标定）⚠️
Stage B                          -> 明确不建议现在进入 ❌
```

## 5. 配置参数

### 5.1 当前 BRPO verification / train-mask / depth-target 参数

| 类别 | 参数 | 当前口径 |
|------|------|----------|
| verification | `verification_mode` | `fused_first` 主线，保留 `branch_first` |
| verification | `tau_reproj_px` | `4.0` |
| verification | `tau_rel_depth` | `0.15` |
| train_mask | `prop_radius_px` | `2` |
| train_mask | `prop_tau_rel_depth` | `0.01` |
| train_mask | `prop_tau_rgb_l1` | `0.05` |
| M3 depth target | `fallback_mode` | `render_depth` |
| M3 depth target | `both_mode` | `average` |

### 5.2 当前 M5 densify selected 参数

| 参数 | 当前选中值 |
|------|------------|
| `patch_size` | `11` |
| `stride` | `5` |
| `min_seed_count` | `4` |
| `max_seed_delta_std` | `0.08` |
| `correction_space` | `log-depth` |
| `candidate_region` | `train_mask > 0` |

### 5.3 当前 Stage A / M5-2 口径

| 类别 | 参数 | 当前口径 |
|------|------|----------|
| consumer | `target_side` | `fused` |
| consumer | `confidence_mask_source` | `brpo` |
| consumer | `stageA_mask_mode` | `train_mask` |
| consumer | `stageA_target_depth_mode` | `blended_depth_m5` |
| consumer | `stageA_depth_loss_mode` | `legacy` 或 `source_aware` |
| source-aware loss | `lambda_depth_seed` | `1.0` |
| source-aware loss | `lambda_depth_dense` | `0.35` |
| source-aware loss | `lambda_depth_fallback` | `0.0` |
| Stage A abs prior | `stageA_lambda_abs_t` | 当前扫描：`{0.3,1.0,3.0}` |
| Stage A abs prior | `stageA_lambda_abs_r` | 当前扫描：`{0.03,0.1}` |
| Stage A abs prior | `stageA_abs_pose_robust` | `charbonnier` |
| Stage A abs prior | `stageA_abs_pose_scale_source` | `render_depth_trainmask_median` |
| Stage A | `stageA_iters` | `300` |
| Stage A | `num_pseudo_views` | `3` |
| Stage A (legacy) | `stageA_lambda_abs_pose` | 仅兼容保留，不再作为主扫描维度 |

### 5.4 当前关键 coverage / diagnosis 口径

| 信号 | 当前量级 | 含义 |
|------|----------|------|
| `train_mask coverage` | `~19.4%` | 当前训练消费的 supervision 区域 |
| `M3 verified depth ratio` | `~1.56%` | 原始 verified depth 覆盖 |
| `M5 non-fallback ratio` | `~15.8%` | densify 后整图新几何区域 |
| `M5 non-fallback within train_mask` | `~81.2%` | train-mask 内真正有新几何信息的区域 |
| `A小网格(6x80) depth_last` | `0.06837 ~ 0.06854` | 各权重组 depth 末值差异很小 |
| `A小网格(6x80) rho_norm_last` | `1.83e-4 ~ 7.93e-4` | `lambda_abs_t` 增大时 drift 明显收紧 |
| `top3深跑(3x300) best_drift` | `lt3.0_lr0.1` | `rho≈2.19e-4, theta≈1.17e-4` |
| `top3深跑(3x300) best_depth` | `lt1.0_lr0.1` | `loss_depth≈0.06837` 但 drift 更大 |
| `depth-heavy top3(3x300) best_drift` | `lt3.0_lr0.1` | `rho≈5.19e-4, theta≈2.40e-4` |
| `depth-heavy top3(3x300) best_depth` | `lt1.0_lr0.1` | `loss_depth≈0.06814`，但 drift 最差 |
| `下游价值验证（3组）` | `270帧 replay 几乎不变` | 三组 `PSNR/SSIM/LPIPS` 差异仅 1e-4~1e-6 量级 |

补充：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_signal_v2_coverage_tmp/signal_v2/` 上 3 个有真实 DiFix 的 frame（10/50/120）平均为：`rgb support left≈1.56%`、`right≈1.20%`、`both≈0.93%`、`raw_rgb_confidence_nonzero≈1.82%`、`depth verified≈1.82%`、`both_weighted≈1.40%`、`left_only≈0.42%`、`render_fallback=0`、`mean_abs_rel_correction_verified≈3.15%`。这说明当前 v2 是一个更窄但更干净的 fused-RGB correspondence mask，depth supervision 也已严格跟着新 RGB 可信区和 projected depth 走。


### 5.5 当前固化口径（A+B 后）

| 口径 | 建议配置 | 目标 | 已知代价 |
|------|----------|------|----------|
| drift-prioritized | `beta_rgb=0.7, lambda_abs_t=3.0, lambda_abs_r=0.1` | 尽可能压低累计漂移 | depth 改善不明显 |
| depth-prioritized | `beta_rgb=0.3, lambda_abs_t=1.0, lambda_abs_r=0.1` | 在当前扫描内尽量保留 depth 末值 | drift 明显更大 |

说明：两套都属于 Stage A 临时口径，不代表已得到统一最优。

## 6. 状态

### 6.1 已完成 ✅

- [x] Phase 1：internal cache 导出
- [x] Phase 2：same-ply replay consistency
- [x] Phase 3：internal prepare + Difix + BRPO verification 并入
- [x] Phase 4：mask-only ablation（legacy vs brpo）
- [x] Phase 5：auditability / provenance 修补
- [x] Phase 6：canonical schema 第一轮打通
- [x] M1：`fused-first verification` + seed-support 层建立
- [x] M2：`seed_support → train_mask` propagation 接入 upstream
- [x] M2.5：propagation 合理区间收紧到约 `10% ~ 25%`
- [x] M3：`projected_depth_*` + blended `target_depth_for_refine` 落地
- [x] M4：Stage A consumer 显式支持 `train_mask / seed_support_only / blended_depth / render_depth_only`
- [x] M4.5：完成 `blended_depth_long vs render_depth_only_long` 的 300-iter Stage A 对照
- [x] M5-0：诊断当前 depth signal 在 train-mask 内部的结构
- [x] M5-1：完成 `target_depth_for_refine_v2` densify target 并选中一组可用参数
- [x] M5-2：完成 M5 target + source-aware depth loss wiring 与 300-iter 对照
- [x] M5-3：补回 S3PO residual pose 闭环并完成 80/300 iter 验证
- [x] A：split+scaled abs prior 工程落地（`lambda_abs_t/r + robust + scene_scale`）
- [x] B：loss 分支梯度贡献诊断落地（在线日志 + 单步诊断脚本）
- [x] A小网格：6组×80iter 扫描完成
- [x] top3深跑：3组×300iter 完成
- [x] depth-heavy top3深跑：3组×300iter 完成
- [x] grad contrib 汇总：完成 rot/trans 主导项统计
- [x] Stage A 双口径固化：drift-prioritized / depth-prioritized
- [x] value-eval：StageA三口径下游验证完成（270帧 replay）
- [x] BRPO-style fusion v1：`pseudo_fusion.py` 与 `stage_fusion()` 已按 `target ↔ reference overlap confidence` 口径落地并完成 smoke
- [x] BRPO-style signal v2 第一步：已新增 `build_brpo_v2_signal_from_internal_cache.py` 与 `pseudo_branch/brpo_v2_signal/`，可独立产出 fused RGB mask v2 + depth supervision v2
- [x] `signal_v2` 3-frame coverage eval：frame `10 / 50 / 120` 平均 `raw/depth verified≈1.82%`，当前确认是“窄但干净”的 fused-RGB-first supervision
- [x] refine consumer 最小接入完成：`run_pseudo_refinement_v2.py` 已支持 `signal_pipeline=brpo_v2`，并在 3-frame StageA 3-iter smoke 中产出 `history / states / refined_gaussians`

### 6.2 当前判断 ⚠️

- [x] 当前已确认：split abs prior 在当前 E1 winner + legacy StageA 背景下已经进入有效区间，当前固定背景配置可先收敛为 `lambda_abs_t=3.0`、`lambda_abs_r=0.1`
- [x] 当前已确认：纯 `stage_mode=stageA` 不会更新 Gaussian；`BASE_PLY` 与 `stageA_noabs_80 / stageA_abs_t3_r0p1_80` 的 `refined_gaussians.ply` 已用 `sha256sum` 验证为完全同 hash，因此 replay-on-PLY 在 StageA-only compare 中不是判优指标
- [x] 当前已完成 canonical E1 root 的 8-frame `signal_v2` 构建，并完成 `legacy / v2-rgb-only / v2-full` 的 `StageA-only` 短对照
- [x] 当前结论是：`v2 RGB-only` 不是明显坏方向，但 `full v2 depth` 当前仍然过窄；其 `mean_target_depth_verified_ratio` 约 `1.96%`，明显低于 legacy 的约 `4.13%`
- [x] 当前已完成 P2 第一版工程接入：`run_pseudo_refinement_v2.py` 已新增 `pseudo_local_gating_*` CLI、StageA.5 grad mask、StageB split backward，以及 `pseudo_branch/local_gating/` 模块目录
- [x] 当前已完成最小 smoke：StageA.5 / StageB history 会真实记录 accepted/rejected pseudo ids、visible union、grad keep ratio、pre/post grad norm；严格阈值下已实测可把 xyz grad 从非零裁到 0
- [x] 当前已完成 P2-B：8-frame `StageA.5` gated vs ungated short compare 已表明 gating 不会把 depth loss 压空，也不会显著恶化 replay，但在当前阈值下只带来极轻微 replay 改善
- [x] 当前已完成 P2-C：带 real branch 的 `StageB` gated vs ungated short compare 已表明 gating 不会明显误伤 real branch，但 replay / loss 仍然与 ungated 几乎重合
- [x] 当前新的结构判断是：无论在 StageA.5 还是 StageB compare 中，current threshold 下都持续 `0 rejection`；因此第一版 gating 的作用主要只剩 `visibility_filter union`，而不是通过 signal gate 主动拒绝 weak pseudo views
- [x] 当前 `P2-D` 统计诊断已确认：legacy sampled view 的 `verified_ratio` 最低约 `1.89%`、`rgb_mask_ratio` 最低约 `14.68%`、`fallback_ratio` 最高约 `98.11%`，而 current gate 默认阈值仅为 `1% / 1% / 99.5%`，所以 `0 rejection` 在当前代码口径下是结构必然
- [x] 当前进一步确认：如果不先重标定阈值，就直接把同一版 gating 挂到 `RGB-only v2` 上，仍大概率只会得到另一个 `0 rejection`；而若直接抬高 `v2` 的 RGB 阈值，又会因为量纲差异导致过度 reject，因此先做了 legacy 上的 gate threshold calibration
- [x] 当前已完成 `P2-E`：`min_verified_ratio=0.02 / 0.03` 已在 legacy `StageA.5` 上产生真实 rejection（`vr=0.02` 主要拒 `260`，`vr=0.03` 继续拒 `225+260`），但 replay 只出现极小幅度变化；其中 `vr=0.02` 是更保守的 calibrated legacy 参考臂
- [x] 当前新的判断是：legacy 侧的 threshold softness 已不是主瓶颈；继续在 legacy `StageA.5` 上细扫阈值的 ROI 很低，后续若继续推进 gating，更值得转向 `RGB-only v2` 的 branch-specific calibration
- [x] 当前已完成 `P2-F`：`RGB-only v2` 的 `StageA.5` ungated vs gated short compare 已表明，这条支路在自然 `StageA→A.5` handoff 下明显优于 legacy；branch-specific gating（`min_rgb_mask_ratio=0.0192`）会稳定拒掉 `225/260` 两档弱 pseudo，并带来小幅 replay 正增益
- [x] 当前对 densify 的判断也更清楚了：raw RGB mask 虽然看起来点状，但它并没有阻止 `RGB-only v2` 在 `StageA.5` 上明显优于 legacy；因此现在不应急着对 raw RGB mask 做 densify，若后续需要扩张，更应考虑受几何约束的 support/depth expand
- [x] 当前已完成 `P2-G`：固定 `signal-aware-8` 的 `StageB` real-branch short compare 已表明，`RGB-only v2 + gated_rgb0192` 在 StageB 下也优于本支路 ungated（`+0.0161 PSNR / +0.000147 SSIM / -2.16e-05 LPIPS`），并且明显优于 legacy StageB；real branch 未被明显误伤
- [x] 当前新的判断是：`RGB-only v2 + gated_rgb0192` 已可提升为当前主候选 refine 分支；下一步应做更长一点的 StageB / 完整 schedule verify，而不是先做 raw RGB densify

### 6.3 当前进行中 ⚠️

- [x] 做一轮 apples-to-apples `StageA-only` short compare：`legacy / v2-rgb-only / v2-full`，保持同一组 pseudo、同一组 iter、同一组 abs prior 背景
- [x] 先在会真实更新 Gaussian 的 stage 上做第一版 gating compare，并确认它对 replay / real branch 基本无害
- [x] 完成 signal gate 阈值 / 统计诊断，确认 current threshold 下的 `0 rejection` 来自阈值过松而不是实现失效
- [x] 完成 legacy `StageA.5` 第一轮 threshold calibration short compare（`min_verified_ratio=0.02 / 0.03`），确认 gate 已进入真实 reject 区
- [x] 完成 `RGB-only v2` `StageA.5` ungated vs branch-specific gated compare，并确认这条支路比 legacy 更值得保留
- [x] 完成 `RGB-only v2` `StageB` real-branch short compare，并确认当前主候选已延续到 joint refine
- [ ] 继续围绕 `RGB-only v2 + gated_rgb0192` 做更长一点的 StageB / 完整 schedule verify
- [ ] 若后续继续扩展，再决定是否补 `xyz+opacity`、soft gating 与更重的 SPGM

### 6.4 下一阶段待办 ⏳

- [ ] 以 `RGB-only v2 + gated_rgb0192` 为当前主候选，补一轮更长一点的 `StageB` / 完整 schedule verify
- [ ] 若继续做 gating compare，优先围绕 `RGB-only v2` 微调 branch-specific threshold，不再回 legacy 侧高密度细扫
- [ ] 只有在更长验证明确暴露 coverage 瓶颈后，再讨论受几何约束的 support/depth expand；先不做 raw RGB densify

### 6.5 当前明确不建议做的事 🚫

- [ ] 现在直接进入 Stage B
- [ ] 把 `lambda_abs_pose`（legacy 单标量）当默认值
- [ ] 把 `lambda_abs_t=3.0` 视为“能自动改善 depth”的最终解


## 7. StageB 保守入口推进（2026-04-13）

- [x] 已新增执行计划文档：`docs/STAGEB_CONSERVATIVE_ENTRY_PLAN.md`
- [x] Phase0/1/2 已完成，执行简报：`docs/STAGEB_PHASE0_2_EXEC_REPORT_20260413.md`
- [x] Phase2 gate 结论：`PASS`（相对 A.5 baseline，PSNR/SSIM 提升且 LPIPS 下降）
- [ ] 下一步：Phase3 长程 StageB（300iter 级别）


## 8. StageB Phase3（300iter long run）

- [x] 已完成 StageB 300iter conservative long run：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_stageB_phase3_longrun/stageB300_conservative_xyz_opacity`
- [x] replay 对照 A.5 结果已产出：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_stageB_phase3_longrun/stageB300_conservative_xyz_opacity/replay/stageB300_conservative_xyz_opacity/replay_eval.json`
- [x] 阶段报告：`/home/bzhang512/CV_Project/part3_BRPO/docs/STAGEB_PHASE3_EXEC_REPORT_20260413.md`


## 9. StageB 120iter vs 300iter 对比记录（2026-04-13）

对比基线：A.5 xyz+opacity (300iter)

- StageB 120iter（Phase2 gate）：PASS
  - delta(vs A.5): PSNR +0.293830, SSIM +0.004369, LPIPS -0.000215
- StageB 300iter（Phase3 long run）：FAIL
  - delta(vs A.5): PSNR -0.179275, SSIM -0.004948, LPIPS +0.002430

结论：
- StageB 短程有效，长程退化；
- 当前主问题是 StageB 后段稳定性，而不是 StageB 链路不可用。


## 10. StageB 网格扫描（计划）

- [x] 规划文档：`docs/STAGEB_GRID_SCAN_PLAN_20260413.md`
- [ ] 待执行：第一轮6组（权重×后段降lr）


## 11. midpoint8 + M5 + source-aware strict full pipeline（2026-04-13 夜）

- [x] 规范执行文档：`docs/MIDPOINT8_M5_STAGEA_A5_B_EXEC_REPORT_20260413.md`
- [x] 运行根目录：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_midpoint8_m5_fullpipeline`
- [x] Phase0+1：midpoint 8 帧已补齐 M5 产物（`target_depth_for_refine_v2.npy` 等）
- [x] Phase2：StageA(source-aware) + 270 replay
- [x] Phase3：StageA.5(`xyz_opacity`) + 270 replay
- [x] Phase4：StageB120(conservative) + 270 replay

结果表（vs part2 after_opt baseline）：

| run | PSNR | SSIM | LPIPS | ΔPSNR | ΔSSIM | ΔLPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 23.94891 | 0.87349 | 0.07878 | 0 | 0 | 0 |
| StageA | 23.94891 | 0.87349 | 0.07878 | +2.89e-06 | +4.75e-08 | -3.54e-08 |
| StageA.5 xyz_opacity | 23.83825 | 0.86555 | 0.08756 | -0.11066 | -0.00794 | +0.00878 |
| StageB120 conservative | 23.16104 | 0.85045 | 0.09781 | -0.78787 | -0.02304 | +0.01903 |

本轮最关键的纠偏结论：
1. StageA 并非“完全没动”，而是 `final_delta_summary` 只记录 fold-back 之后清零的 residual；真实相机状态有小幅更新，但量级很小；
2. StageA replay 几乎不变的根因不是“pose 完全没动”，而是 **StageA 根本不改 Gaussian，导出的 PLY 与输入 PLY hash 完全相同**，因此 replay 指标天然不变；
3. 当前 `StageA -> A.5 -> B` 不是严格顺序 handoff：A.5 / B 都会从 pseudo cache 重新加载 pseudo viewpoints，并不会读前一阶段导出的 pseudo camera states；
4. 当前 midpoint8 M5 的有效几何监督比预期弱很多：平均 `train_mask≈18.7%`，而 `seed+dense≈3.49%`，即 train-mask 内只有约 `18.65%` 具备 non-fallback depth；
5. A.5 / B 的 loss 下降不能说明下游几何更好；在本轮里它们更像是在 pseudo-side 局部过拟合，且 StageB 的 sparse real RGB anchor 不足以拉回。

状态判断更新：
- 当前主问题已不能再简单归因为“参数没调好”；
- 更深层的是 **pipeline 结构问题 + 中点伪帧几何信号不足**；
- 下轮优先级应转向：`stage handoff`、`StageA 评估口径修正`、`StageB 几何锚点增强 / paper-faithful joint机制补齐`。


## 12. P0 repair validation + repaired sequential rerun（2026-04-13 夜）

- [x] 已修复 Stage handoff：`run_pseudo_refinement_v2.py` 新增 `--init_pseudo_camera_states_json`
- [x] 已修复 StageA 报告口径：新增 `true_pose_delta_summary / aggregate`，旧 `final_delta_summary` 语义改名为 `residual_delta_summary_post_foldback`
- [x] 已完成 sequential smoke/short rerun：
  - `repair_seq_check/`
  - `repair_seq_rerun/`

结构验证结果：
- `StageA final -> StageA.5 init` 完全一致（pose/exposure diff = 0）
- `StageA.5 final -> StageB init` 完全一致（pose/exposure diff = 0）
- 说明此前“阶段实质互不相连”的问题已被修掉

修复后 sequential rerun 指标（vs baseline）：
- StageA.5(80, from StageA): `23.89296 / 0.87007 / 0.08178`，仍负，但好于修复前 strict run
- StageB120(from StageA.5): `23.17635 / 0.85271 / 0.09420`，仍负，但也略好于修复前 strict run

更新后的状态判断：
1. handoff bug 是真实结构问题，且修复后确实改善了 A.5 / B 的结果；
2. 但修复 handoff 后，pipeline 仍未转正；
3. 当前剩余主矛盾是：`midpoint8 M5` 几何信号偏弱 + 现有 joint refine 设计仍偏窄/偏弱。

## 13. P1 bottleneck review（2026-04-13 夜）

在 P0 handoff 修复后，继续沿真实代码链路复查 `pseudo_cache -> source-aware loss -> A.5/B refine -> replay`，当前结论进一步收敛为：

1. 问题不再主要是“阶段没连上”，而是 `midpoint8 M5` 的有效几何监督太弱；
2. 平均 `train_mask≈18.7%`，但真正 non-fallback depth 只占 `3.49%`，`render_fallback≈96.5%`；
3. 更关键的是：在有 support 的区域里，`target_depth_for_refine_v2` 相对 `render_depth` 的平均修正也只有约 `1.53%`；折算到整图，相当于只有约 `0.053%` 的全图平均相对修正量；
4. 与此同时，A.5 / StageB 打开的是全局 Gaussian `xyz/opacity`，导致“弱而局部的 supervision”去驱动“大范围的全局参数扰动”；
5. 这解释了为什么 pseudo-side loss 可以下降，但 replay 仍然容易退化。

本轮新增的关键量化证据：
- StageA80: `mean_trans_norm≈0.00376`，相对 `scene_scale≈3.196` 仅约 `0.118%`，说明 pose 调整本身也很小；
- A.5 相对 baseline PLY：mean xyz drift `≈0.00394`，且 `74.1%` Gaussians 的 xyz 改动 `>1e-3`；
- StageB120 相对 baseline PLY：mean xyz drift `≈0.00734`，且 `86.2%` Gaussians 的 xyz 改动 `>1e-3`。

状态判断更新：
- 当前更像是 `signal weak + supervision scope / optimization scope mismatch`，而不只是 handoff bug 或单纯参数问题；
- 下一步优先级应转向：
  1. pseudo 选点与 signal 质量复查；
  2. 缩小 A.5 / B 的可训练 Gaussian 作用域；
  3. 将 support/correction 量化检查设为长跑前固定 gate。

## 14. 下一阶段计划：signal semantics + stable refinement（2026-04-14）

基于更新后的 `DEBUG_CONVERSATION.md` 和真实代码链路复查，当前下一阶段已形成独立执行计划：
- `docs/SIGNAL_SEMANTICS_AND_STABLE_REFINEMENT_PLAN_20260414.md`

当前固定优先级：
1. `continuous confidence + agreement-aware support`
2. `RGB/raw confidence` 与 `depth/train-mask` 语义分离
3. `confidence-aware densify`
4. StageB 两段式 curriculum（前强后稳）
5. pseudo 分支的 local Gaussian gating
6. active-support-aware depth reweight

状态判断补充：
- B 层（更密 pseudo 选点）可以后置；
- C 层不应先上 full SPGM，而应先做 loss reweight + local gating + curriculum；
- 当前主线应先把 pseudo supervision 语义做对，再把 StageB 变稳。
## 15. S1.1 进展（2026-04-14）

- [x] `continuous confidence + agreement-aware support` 已完成第一版工程落地
- [x] verification / pack 产物已新增 raw discrete + raw continuous confidence 文件
- [ ] 下一步：让 refine consumer 真正区分 RGB/raw confidence 与 depth/train-mask
## 16. S1.2 进展（2026-04-14）

- [x] refine consumer 已支持 RGB/raw confidence 与 depth/train-mask 分离
- [x] loss 侧已支持 RGB / depth 使用不同 confidence mask
- [ ] 下一步：把 confidence-aware densify 做完，并开始小规模 semantics ablation
## 17. S1.3 + smoke/ablation（2026-04-14）

- [x] `confidence-aware densify` 已完成第一版工程落地
- [x] 2-frame verify/pack smoke 已通过，new confidence artifacts 能真实进入 pseudo_cache
- [x] 2-frame StageA-10iter semantics ablation 已跑通
- [ ] 当前阻塞不在代码链路，而在 S1.3 首轮阈值过严；下一步需回调 densify 阈值后再做 8-frame 短跑对照
## 18. 8-frame 短跑回调结果（2026-04-14 晚）

- [x] 已完成 8-frame verify/pack 联调：`20260414_signal_semantics_mid8`
- [x] 已完成 retuned confidence-aware densify 对照：`20260414_signal_semantics_mid8_compare`
- [x] 已完成 8-frame StageA-20iter baseline vs conf-aware 小对照

关键结果：
- baseline densify：`mean_dense_valid_ratio≈0.02269`
- retuned conf-aware densify：`mean_dense_valid_ratio≈0.01634`
- 相比首轮过严版本（≈0.00129），当前已回到可用区间，但仍偏保守
- StageA-20iter 下，conf-aware 组 `loss_total/loss_rgb/loss_depth` 仍高于 baseline

当前判断：
- S1 wiring 已基本完成；
- 但 E1 还没收口，当前还不建议直接进入 `4. StageB 两段式 curriculum`；
- 下一步应先做一轮更系统的 8-frame semantics 短跑/小网格回调。


## 19. E1：support-aware pseudo selection（2026-04-14 夜）

- [x] 已实现 `scripts/select_signal_aware_pseudos.py`
- [x] 已完成 8-gap × `{1/3, 1/2, 2/3}` lightweight candidate scoring
- [x] 已生成 `signal_aware_selection_report / manifest`
- [ ] 下一步：基于 `signal-aware-8` 做 apples-to-apples verify/pack + 8-frame StageA short compare，再决定是否进入 E2

本轮关键结果：
- midpoint8：`[17, 51, 86, 121, 156, 190, 225, 260]`
- signal-aware-8：`[23, 57, 92, 127, 162, 196, 225, 260]`
- `6/8` 个 gap 改选，且前 6 个 gap 全部偏向 `2/3`

和 midpoint8 的 aggregate 对照：
- `support_ratio_both`: `0.00754 -> 0.00738`
- `verified_ratio`: `0.01498 -> 0.01543`
- `continuous_confidence_mean_positive`: `0.69049 -> 0.69546`
- `balance_ratio`: `0.01019 -> 0.01086`
- `score`: `0.10101 -> 0.10313`

当前判断：
- `midpoint` 不是当前 case 的稳定最优 pseudo 位置；
- 但这轮仍是 lightweight render-based verify，不是 full fused-first apples-to-apples rerun；
- 因此 E1 方向已成立，但最终验收仍需补一轮 `signal-aware-8` 的正式 verify/pack + short refine compare。

## 20. E1.5：support-aware pseudo selection 正式短对照（2026-04-14 深夜）

- [x] 已完成 `signal-aware-8` 的正式 `fusion -> verify -> pack`
- [x] 已完成 apples-to-apples M5 densify 对照（baseline / conf-aware）
- [x] 已完成 8-frame StageA-20iter 对照（baseline / conf-aware）
- [x] 当前可进入 E2：dual-pseudo allocation

关键结论：
- raw signal：`verified_ratio 0.01498 -> 0.01543`，`continuous_confidence_mean_positive 0.69049 -> 0.69546`，`support_ratio_both 0.00754 -> 0.00738`
- baseline densify：`mean_dense_valid_ratio 0.02269 -> 0.02961`
- conf-aware densify：`mean_dense_valid_ratio 0.01634 -> 0.01831`
- baseline StageA20：`loss_total 0.03673 -> 0.03659`
- conf-aware StageA20：`loss_total 0.04392 -> 0.04340`

当前判断：
- support-aware pseudo selection 已不只是“方向成立”，而是已经通过正式短对照；
- 它的主要收益首先体现在 `verified signal -> densify coverage`，随后才体现在短跑 loss 的小幅改善；
- 因此 E1 可以收口，下一步建议进入 E2（dual-pseudo allocation）。


## 21. E2：dual-pseudo allocation 正式对照（2026-04-14 深夜）

- [x] 已完成 E2 工程落地：selection manifest / schema / pseudo_cache 已支持每 gap 多 pseudo
- [x] 已完成 `top2 per gap` 的正式 `select -> fusion -> verify -> pack`
- [x] 已完成 E2 的 baseline / conf-aware M5 densify 对照
- [x] 已完成 E2 的 16-pseudo StageA-20iter 对照
- [x] 已产出总结报告：`docs/SIGNAL_DUAL_PSEUDO_E2_COMPARE_20260414.md`

本轮实际选中的 E2 pseudo set：
- `top2 per gap = [23, 17, 57, 51, 92, 86, 127, 121, 162, 156, 196, 190, 225, 231, 260, 266]`
- 当前 case 中，`top2` 稳定落成 `1/2 + 2/3`，而不是 `1/3 + 2/3`

关键结果：
- raw signal（vs midpoint8）：`verified_ratio 0.01498 -> 0.01520`，`continuous_confidence_mean_positive 0.69049 -> 0.69470`，`support_ratio_both 0.00754 -> 0.00741`
- baseline densify：midpoint8 `0.02269`，E1 `0.02961`，E2 `0.02621`
- conf-aware densify：midpoint8 `0.01634`，E1 `0.01831`，E2 `0.01737`
- baseline StageA20：E1 `loss_total 0.03659`，E2 `0.04555`
- conf-aware StageA20：E1 `loss_total 0.04340`，E2 `0.05815`

当前判断：
- E2 相比 midpoint8 仍有轻微正向，但弱于 E1 的 `signal-aware-8`；
- 当前 16-pseudo 的多帧方案没有把更多样本转化成更好的 short refine，反而出现明显回落；
- 说明这轮主问题不是“pseudo 数量还不够”，而是“第二个 pseudo 的质量不够强，会稀释 supervision”；
- 因此当前 default winner 仍应保持 E1 `signal-aware-8`，而不是直接切到 E2。

下一步建议：
- 如果进入 E3（multi-anchor verify），应优先基于 E1 winner `signal-aware-8` 做，而不是基于当前 E2 16-pseudo set。


## 22. 对 mask/depth/confidence pipeline 的当前判断（2026-04-14 更深夜）

基于 `E1 -> E1.5 -> E2` 的连续结果，当前项目状态应更新为：

- 这条 pipeline 不是完全错误；
- 但它作为主增益来源的边际收益已经非常接近见顶；
- E1 已经吃到了这条线上最有价值的一段收益；
- E2 则说明“增加 pseudo 数量”不能替代“提高 winner 质量”，次优 pseudo 会在当前 consumer 机制下稀释 supervision。

因此当前 default winner 仍保持：
- `E1 signal-aware-8 = [23, 57, 92, 127, 162, 196, 225, 260]`

当前建议的下一步不再是继续扩张 E2，而是：
- 若继续给这条 pipeline 一次机会，只做一次基于 E1 winner 的 E3 final probe；
- 若 E3 仍不能在 short compare 上优于 E1，就应把这条线从“主线增益来源”降级为“已基本收口的 signal gate 层”。

## 23. P0 abs prior 标定 + P1A StageA-only signal compare（2026-04-15）

### run roots
- P0：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_absprior_calibration_legacy_e1`
- P1A：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_signal_compare_stageAonly_e1`
- 报告：`docs/P0_absprior_and_P1A_stageA_signal_compare_20260415.md`

### 本轮关键结果
- [x] 已完成 split abs prior 标定：当前固定背景配置收敛为 `lambda_abs_t=3.0`、`lambda_abs_r=0.1`
- [x] 已确认纯 `StageA` 不更新 Gaussian；`refined_gaussians.ply` 与 `BASE_PLY` 的 hash 完全一致，因此 replay-on-PLY 只剩 identity sanity check 作用
- [x] 已完成 `legacy / v2-rgb-only / v2-full` 的 `StageA-only` 短对照
- [x] `v2 RGB-only` 可作为后续 signal probe 保留；`full v2 depth` 当前则明显过窄：`verified_ratio` 约 `1.96%`，而 legacy 约 `4.13%`

### 当前结论
- 这轮之后，纯 `StageA` 上已经没有必要继续纠缠 replay；
- 如果后续想让 replay 成为有区分度的验收指标，必须进入会真实更新 Gaussian 的 stage；
- 因此当前下一步应切到 `P2 local Gaussian gating`，优先在 `StageA.5` 上落地 `xyz-only + hard gating + pseudo-side only`。

## 24. P2 local Gaussian gating 第一版实现 + smoke（2026-04-15）

### run roots
- smoke 根：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_local_gating_smoke`
- 报告：`docs/P2_local_gaussian_gating_first_impl_smoke_20260415.md`

### 本轮关键结果
- [x] 已在 `run_pseudo_refinement_v2.py` 中新增 `pseudo_local_gating_*` CLI
- [x] 已新增 `pseudo_branch/local_gating/`，拆出 signal gate / visibility union / grad mask / history schema
- [x] 已接通 `StageA.5`：`backward -> local gate mask -> gaussian step`
- [x] 已接通 `StageB`：split backward 后只裁 pseudo-side Gaussian grad，不直接裁 real branch
- [x] `stageA_history.json / stageB_history.json` 已真实记录 accepted/rejected pseudo ids、rejected reasons、visible union、grad keep ratio 与 pre/post grad norm
- [x] 严格阈值 smoke 已验证：当 sampled pseudo views 被 gate 全拒绝时，`xyz` grad 可从非零真实裁到 0

### 当前结论
- 第一版 local gating 已从计划进入真实代码路径；
- 在 `StageA.5 pseudo-only` 下，若 sampled pseudo views 全部通过 signal gate，则第一版 `visibility_filter union` 对 xyz grad norm 基本不再额外收缩；
- 因此第一版 gating 的主要价值，体现在“拒绝 weak pseudo views 时阻断它们对 Gaussian 的更新”；
- 当前下一步应直接进入 8-frame `StageA.5` gated vs ungated short compare，而不是继续停在 smoke 级别。

## 25. P2-B：8-frame StageA.5 gated vs ungated short compare（2026-04-15）

### run roots
- compare 根：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_p2b_stageA5_local_gating_compare_e1`
- 报告：`docs/P2B_stageA5_local_gating_compare_20260415.md`

### 本轮关键结果
- [x] 已完成 `legacy ungated xyz` vs `legacy hard_visible_union_signal xyz` 的 8-frame `StageA.5` short compare
- [x] gated replay 相比 ungated 仅有极轻微优势：
  - PSNR `+0.000439`
  - SSIM `+0.000001`
  - LPIPS `-0.000001`
- [x] gated 没有把 depth loss 压没：`loss_depth_last` 约 `0.05947`，与 ungated 的 `0.05948` 基本一致
- [x] gated 的 `grad_keep_ratio_xyz` 最后一轮约 `0.8302`，80 iter 平均约 `0.7247`
- [x] 但这轮 compare 中 `iters_with_rejection = 0 / 80`，说明所有 sampled pseudo views 都通过了 signal gate

### 当前结论
- 第一版 gating 在当前阈值下是“基本无害但提升极弱”；
- 这轮没有出现真正的 view-level rejection，因此 gating 的主要作用只剩 visible union；
- 这也解释了为什么 replay 改善极弱、`grad_norm_xyz` 也没有出现明显收缩；
- 因此当前下一步应切到带 real branch 的 `StageB` short compare，验证 pseudo-side gating 不会把 global correction 一并裁掉。

## 26. P2-C：real-branch StageB gated vs ungated short compare（2026-04-15）

### run roots
- compare 根：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_p2c_stageB_local_gating_compare_e1`
- 报告：`docs/P2C_stageB_local_gating_realbranch_compare_20260415.md`

### 本轮关键结果
- [x] 已完成带 real branch 的 `StageB ungated` vs `StageB hard_visible_union_signal` short compare
- [x] 两臂都从同一个 `StageA.5 gated` 输出起跑，以隔离 `StageB` gating 本身的影响
- [x] gated 不会明显误伤 real branch：`loss_real_last` 与 ungated 几乎完全一致（约 `0.14798`）
- [x] gated replay 相比 ungated 反而略差一点，但量级几乎可忽略：
  - PSNR `-0.000045`
  - SSIM `-0.00000056`
  - LPIPS `+0.00000057`
- [x] 这轮 compare 中同样 `iters_with_rejection = 0 / 20`

### 当前结论
- 第一版 gating 在 real branch 打开后仍然是“基本无害但收益极弱”；
- 当前最关键的问题已经不再是 real branch 会不会被误伤，而是：为什么 current threshold 下 signal gate 在 StageA.5 / StageB 都持续 `0 rejection`；
- 因而后续优先级应切到解释和重设 signal gate，而不是继续默认把当前 hard gating 当 winner 直接推进。


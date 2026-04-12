# STATUS.md 写作规范

> 更新时间：2026-04-12 20:40
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

当前主线仍然是 **mask-problem route on top of Re10k-1 full internal route**，但阶段判断已经更新：
- 已完成 `M1 / M2 / M2.5 / M3 / M4 / M4.5`，证明 upstream schema 和 Stage A consumer 已接通；
- 已完成 `M5-0 / M5-1 / M5-2 / M5-3`：depth signal 结构诊断、densified depth target、source-aware depth loss、S3PO residual pose 闭环修回；
- 已完成 `M5-4`：absolute pose prior 工程接入（`SE3_log + R0/T0 + stageA_lambda_abs_pose`）并做 smoke 对照；
- 当前确认：**depth 在修复后是“弱有效”，而 absolute pose prior 在小权重（0.1）几乎无感；大权重（100）能压 drift，但未改善 depth，甚至略伤。**

一句话说：**当前最优先的问题已经从“链路是否断开”切换成“absolute pose prior 如何做尺度化/权重标定，才能在抑制累计 drift 的同时不压制 depth 对齐”。**

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

当前主 prototype：

```text
<run_root>/internal_prepare/re10k1__internal_afteropt__brpo_proto_v4_stage3/
```

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

当前状态：
- `train_mask` 与 `verified depth` 已明确拆层；
- `target_depth_for_refine.npy` 表示 M3 blended depth target；
- `target_depth_for_refine_v2.npy` 表示 M5 densified depth target；
- `target_depth_dense_source_map.npy` 当前可区分：`seed / densified / render_fallback`。

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
| `prepare_stage1_difix_dataset_s3po_internal.py` | internal `select / difix / fusion / verify / pack` | ✅ 可用 | 当前已支持 M1/M2/M3/M5 sample 固化 |
| `brpo_build_mask_from_internal_cache.py` | 从 internal cache 构建 verification / seed / train mask / projected depth | ✅ 可用 | 当前已支持 `branch_first|fused_first` 与 M3 depth 输出 |
| `run_pseudo_refinement.py` | v1 standalone refine 入口 | ✅ 可用 | fixed-pose appearance tuning |
| `run_pseudo_refinement_v2.py` | BRPO-style refine v2 / Stage A 入口 | ✅ 可用 | 当前已补回 `apply_pose_residual_()` 闭环，并支持 `stageA_lambda_abs_pose` 与 abs-drift 导出 |
| `analyze_m5_depth_signal.py` | M5-0：分析当前 depth signal 在 train-mask 内的结构 | ✅ 可用 | 不改训练，仅诊断 |
| `materialize_m5_depth_targets.py` | M5-1：写出 `target_depth_for_refine_v2` 与 densify 产物 | ✅ 可用 | 当前 selected 参数已验证 |
| `diagnose_stageA_gradients.py` | M5 diagnosis：检查 forward sensitivity 与 grad 连通性 | ✅ 可用 | 当前已发现 Stage A pose path 异常 |

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
| `pseudo_fusion.py` | left/right repaired RGB 融合 | ✅ 已接入 | 当前已作为 fused-first pseudo source |

### 3.3 下一阶段预计触达的代码

| 文件 | 计划改动 |
|------|----------|
| `run_pseudo_refinement_v2.py` | 做 300-iter abs prior 权重扫描（含 default/depth-heavy）并固化推荐口径 |
| `pseudo_loss_v2.py` | 将 abs pose prior 从固定权重改为尺度化版本（rotation/translation 分离或按 scene scale 归一） |
| `pseudo_camera_state.py` | 增加 drift 统计导出（分 `rho/theta` 聚合）供实验比较 |
| `brpo_depth_target.py` / `brpo_depth_densify.py` | 当前 upstream 已够用，优先级低于 Stage A 结构标定 |

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
fusion (target_rgb_fused)
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
| Stage A | `stageA_iters` | `300` |
| Stage A | `num_pseudo_views` | `3` |
| Stage A | `stageA_lambda_abs_pose` | 当前已接入，实验显示 `0.1` 过弱、`100` 过强 |

### 5.4 当前关键 coverage / diagnosis 口径

| 信号 | 当前量级 | 含义 |
|------|----------|------|
| `train_mask coverage` | `~19.4%` | 当前训练消费的 supervision 区域 |
| `M3 verified depth ratio` | `~1.56%` | 原始 verified depth 覆盖 |
| `M5 non-fallback ratio` | `~15.8%` | densify 后整图新几何区域 |
| `M5 non-fallback within train_mask` | `~81.2%` | train-mask 内真正有新几何信息的区域 |
| `M5 source-aware loss_depth (fixed path)` | `0.06867 -> 0.06816` | 修复后开始轻微下降，但仍偏弱 |
| `default_300 mean pose change` | `trans ~0.00138`, `rot ~0.00498 rad` | 修复后真实累计位姿变化不大但存在 |
| `pose_reg during fixed-path training` | `≈ 0` | residual 每步清零后，`stageA_lambda_pose` 不约束累计 drift |
| `m56 default abs(0.1) vs noabs` | 几乎重合 | `loss_depth` 与 `abs_pose_norm` 变化都很小，权重过弱 |
| `m56 default abs(100)` | drift 明显下降 | `abs_pose_norm mean 0.00154 -> 0.00034`，但 `loss_depth` 略变差 |

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
- [x] M5-3：补回  闭环并完成 80/300 iter 验证

### 6.2 当前判断 ⚠️

- [x] upstream depth coverage 已不再是唯一主矛盾：M5 已把 train-mask 内非 fallback 区域抬到约 `81%`
- [x] M5-3 修复后，Stage A 已从“假优化”进入“弱有效”：depth 能降，但幅度小
- [x] 现阶段核心结构问题是：`pose_reg` 约束 residual 而非累计 drift，导致 absolute prior 成为必要项
- [x] M5-4 已证明 absolute prior 需要重标定：`0.1` 无感，`100` 能压 drift 但会压 depth

### 6.3 当前进行中 ⚠️

- [ ] 做 300-iter `lambda_abs_pose` 扫描（建议 `10/30/100`，含 default 与 depth-heavy）
- [ ] 设计尺度化 absolute prior（rotation/translation 分离权重，或按场景尺度归一）
- [ ] 固化“抑制 drift 且不伤 depth”的 Stage A 默认参数

### 6.4 下一阶段待办 ⏳

- [ ] 产出 m56 扩展对照汇总（含 drift-depth tradeoff 图表）
- [ ] 将 absolute prior 方案写入 `absolute_pose_prior.md` 的默认执行口径
- [ ] 在 Stage A 达到“非弱下降”前，不进入 Stage B

### 6.5 当前明确不建议做的事 🚫

- [ ] 现在直接进入 Stage B
- [ ] 把  当成“已接入就够用”的默认值
- [ ] 把  当成最终解（当前会压 depth）

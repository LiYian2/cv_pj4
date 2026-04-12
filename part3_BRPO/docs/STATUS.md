# STATUS.md 写作规范

> 更新时间：2026-04-12 14:25
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
- 已完成 `M1 / M2 / M2.5 / M3 / M4 / M4.5`，证明 upstream schema 和 Stage A consumer 都已接通；
- 已完成 `M5-0 / M5-1 / M5-2`：depth signal 结构诊断、densified depth target、source-aware depth loss；
- 当前已确认：**upstream depth coverage 问题已被显著缓解，并且 Stage A 已补回 S3PO 原始的 `tau -> update_pose -> R/T` 闭环；当前进入修复后效果评估阶段**；
- 最新 diagnosis 表明：renderer 前向对 `cam_rot_delta / cam_trans_delta` 的响应几乎为零，但 backward 却返回非零 pose 梯度，因此当前 Stage A pose refine 机制不可信。

一句话说：**当前最优先的问题已经从“找根因”切换成“根因已修后，这套 M5 depth supervision 是否足够强、值得继续放大实验并最终推进下一阶段”。**

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
└── 2026-04-12_m53_stageA_diagnosis/
```

当前最重要结果目录：
- **M4.5**：`.../2026-04-12_m45_stageA_eval/`
- **M5-0 / M5-1**：`.../2026-04-12_m50_m51_eval/analysis/`
- **M5-2**：`.../2026-04-12_m52_stageA_loss_eval/`
- **当前 diagnosis**：`.../2026-04-12_m53_stageA_diagnosis/m53_gradients.json`

## 3. 代码脚本

### 3.1 当前主线脚本

路径：`/home/bzhang512/CV_Project/part3_BRPO/scripts/`

| 文件 | 职责 | 状态 | 备注 |
|------|------|------|------|
| `replay_internal_eval.py` | 基于 saved internal camera states 做 replay eval | ✅ 可用 | same-ply consistency 已验证通过 |
| `prepare_stage1_difix_dataset_s3po_internal.py` | internal `select / difix / fusion / verify / pack` | ✅ 可用 | 当前已支持 M1/M2/M3/M5 sample 固化 |
| `brpo_build_mask_from_internal_cache.py` | 从 internal cache 构建 verification / seed / train mask / projected depth | ✅ 可用 | 当前已支持 `branch_first|fused_first` 与 M3 depth 输出 |
| `run_pseudo_refinement.py` | v1 standalone refine 入口 | ✅ 可用 | fixed-pose appearance tuning |
| `run_pseudo_refinement_v2.py` | BRPO-style refine v2 / Stage A 入口 | ✅ 可用 | 当前已补回每步后 `apply_pose_residual_()` 的 S3PO 闭环，并支持 M5 source-aware loss |
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
| `pseudo_camera_state.py` / `pseudo_loss_v2.py` / `pseudo_refine_scheduler.py` | Stage A pseudo camera + loss + optimizer | ✅ 可用 | 已补 `apply_pose_residual_()`，当前 residual 会在每步后折回 `R/T` |
| `pseudo_fusion.py` | left/right repaired RGB 融合 | ✅ 已接入 | 当前已作为 fused-first pseudo source |

### 3.3 下一阶段预计触达的代码

| 文件 | 计划改动 |
|------|----------|
| `gaussian_splatting/gaussian_renderer/__init__.py` / 底层 rasterizer | 审计 `theta/rho` 在 forward/backward 中是否一致生效 |
| `run_pseudo_refinement_v2.py` | 在 pose path 修清前，不继续扩 Stage B；必要时先限制/冻结 pose 分支 |
| `pseudo_loss_v2.py` | 当前已完成 source-aware depth loss，下一步取决于 pose path 审计结果 |
| `brpo_depth_target.py` / `brpo_depth_densify.py` | 当前 upstream 已足够开展诊断，不再是最优先怀疑点 |

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
M5-3 diagnosis: pose/render forward-backward consistency audit
```

### 4.3 当前结论对应的流程位置

```text
verify / pack upstream           -> 已打通 ✅
M5 densified target generation   -> 已打通 ✅
source-aware depth loss wiring   -> 已打通 ✅
Stage A depth/pseudo pose use it -> 结论不可信 ⚠️
pose/render forward consistency  -> 存在异常 ⚠️
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

### 5.4 当前关键 coverage / diagnosis 口径

| 信号 | 当前量级 | 含义 |
|------|----------|------|
| `train_mask coverage` | `~19.4%` | 当前训练消费的 supervision 区域 |
| `M3 verified depth ratio` | `~1.56%` | 原始 verified depth 覆盖 |
| `M5 non-fallback ratio` | `~15.8%` | densify 后整图新几何区域 |
| `M5 non-fallback within train_mask` | `~81.2%` | train-mask 内真正有新几何信息的区域 |
| `M5 source-aware loss_depth` | `~0.0687` | 已接通但仍基本不下降 |
| `pose forward sensitivity probe` | `0.0` | 手动加 pose delta 后 render RGB/depth 几乎不变 |

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
- [x] M5-3（诊断部分）：确认 Stage A pose/render 路径存在 forward/backward 一致性异常迹象

### 6.2 当前判断 ⚠️

- [x] upstream depth coverage 问题已不再是唯一主矛盾：M5 已把 train-mask 内的非 fallback 区域抬到约 `81%`
- [x] `M5 + legacy depth loss` 与 `M5 + source-aware depth loss` 都表明：loss 已接通，但 depth 仍基本不动
- [x] 当前更大的异常是：**renderer 前向对 pose delta 几乎不敏感，但 backward 却给出非零 pose 梯度**
- [x] 因此当前 Stage A 的 pose refine 结果不能直接当成“真实可优化性”的证据

### 6.3 当前进行中 ⚠️

- [ ] 审计 `theta / rho` 在 renderer / rasterizer forward 与 backward 中是否一致生效
- [ ] 判断问题是底层 rasterizer camera-delta path 未生效，还是上层接口/封装存在错位
- [ ] 在 pose path 修清之前，不继续把当前 Stage A loss 曲线当成 depth supervision 失效的最终结论

### 6.4 下一阶段待办 ⏳

- [ ] 先做 renderer / pose-delta 连通性修复或最小复现实验
- [ ] 修清后再重跑 `M5 + source-aware depth loss`，确认 loss 是否仍然平
- [ ] 只有在 Stage A pose path 正常后，再讨论是否进入 Stage B

### 6.5 当前明确不建议做的事 🚫

- [ ] 现在直接进入 Stage B
- [ ] 在 pose/render 前向都不敏感的情况下继续解释“为什么 depth 不动”而不先修底层连通性
- [ ] 把当前 Stage A pose drift 误当成“depth / rgb 真的推动了相机对齐”

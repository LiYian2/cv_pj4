# STATUS.md 写作规范

> 更新时间：2026-04-12 01:45
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

当前主线已经明确切到 **mask-problem route on top of Re10k-1 full internal route**：
- 已完成 `part2 full rerun → internal cache → replay consistency → internal prepare + Difix + BRPO verification → mask-only ablation`；
- 已完成 **M1 / M2 / M2.5 / M3 / M4 / M4.5**：`fused-first verification`、`seed_support → train_mask`、propagation 收紧、`blended target_depth_for_refine`、Stage A consumer 接入、以及 `blended_depth vs render_depth_only` 长一点对照；
- 当前已证明：**upstream schema 与 Stage A consumer 都已接通**，`blended_depth` 在 Stage A 中确实产生非零 depth loss；
- 但当前 `blended_depth` 的 depth loss 在 300-iter Stage A eval 中基本不下降，说明**depth 信号已接入，但当前优化利用仍偏弱**。

一句话说：**mask-problem route 已完成从 verify 到 Stage A consumer 的闭环接通；当前最优先的不再是补字段，而是诊断为什么 depth signal 已接上却基本不动。**

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
- 正式输入统一认 `internal_eval_cache/<stage>/...`，不认顶层 S3PO 原生空目录。

### 2.2 当前 internal prepare 正式层

当前 M3 主线 prototype：

```text
<run_root>/internal_prepare/re10k1__internal_afteropt__brpo_proto_v4_stage3/
```

当前已跑通结构：

```text
internal_prepare/<prepare_key>/
├── manifests/
│   ├── source_manifest.json
│   ├── pseudo_selection_manifest.json
│   ├── selection_summary.json
│   ├── difix_manifest.json
│   ├── fusion_manifest.json
│   ├── verification_manifest.json
│   └── pack_manifest.json
├── inputs/
│   ├── raw_render/
│   ├── left_ref/
│   └── right_ref/
├── difix/
│   ├── left_fixed/
│   └── right_fixed/
├── fusion/
│   └── samples/<frame_id>/
│       ├── target_rgb_fused.png
│       ├── confidence_mask_fused.npy
│       └── fusion_meta.json
├── verification/
│   ├── brpo_phaseC/after_opt/
│   └── brpo_phaseC_fused_first/after_opt/
└── pseudo_cache/
    ├── manifest.json
    └── samples/<frame_id>/
        ├── camera.json
        ├── refs.json
        ├── source_meta.json
        ├── render_rgb.png
        ├── render_depth.npy
        ├── ref_rgb_left.png
        ├── ref_rgb_right.png
        ├── target_rgb_left.png
        ├── target_rgb_right.png
        ├── target_rgb_fused.png
        ├── target_depth.npy
        ├── target_depth_for_refine.npy
        ├── target_depth_for_refine_source_map.npy
        ├── target_depth_for_refine_meta.json
        ├── verified_depth_mask.npy
        ├── projected_depth_left.npy
        ├── projected_depth_right.npy
        ├── projected_depth_valid_left.npy
        ├── projected_depth_valid_right.npy
        ├── seed_support_left.npy
        ├── seed_support_right.npy
        ├── seed_support_both.npy
        ├── seed_support_single.npy
        ├── train_confidence_mask_brpo_left.npy
        ├── train_confidence_mask_brpo_right.npy
        ├── train_confidence_mask_brpo_fused.npy
        ├── train_support_left.npy
        ├── train_support_right.npy
        ├── train_support_both.npy
        ├── train_support_single.npy
        ├── confidence_mask_brpo.npy
        ├── confidence_mask_brpo_left.npy
        ├── confidence_mask_brpo_right.npy
        ├── confidence_mask_brpo_fused.npy
        ├── support_left.npy
        ├── support_right.npy
        ├── support_both.npy
        ├── support_single.npy
        ├── fusion_meta.json
        ├── verification_meta.json
        └── diag_brpo/
```

### 2.3 当前锁定中的 canonical schema

当前 sample schema 已经锁到三层：

```text
samples/<frame_id>/
├── render_rgb.png
├── render_depth.npy
├── target_rgb_left.png
├── target_rgb_right.png
├── target_rgb_fused.png
├── target_depth.npy                        # 兼容字段
├── target_depth_for_refine.npy             # M3 blended depth target
├── target_depth_for_refine_source_map.npy  # 区域来源图
├── seed_support_*.npy                      # 几何 seed 层
├── train_confidence_mask_brpo_*.npy        # 训练 mask 层
├── projected_depth_{left,right}.npy        # pseudo-view sparse verified depth
├── projected_depth_valid_{left,right}.npy
├── confidence_mask_brpo_*.npy              # alias -> 当前训练消费 mask
├── support_*.npy                           # alias -> seed support
└── source_meta / fusion_meta / verification_meta
```

当前状态：
- `seed_support_*`、`train_confidence_mask_*`、`projected_depth_*`、`target_depth_for_refine*` 均已真实落盘；
- `target_depth_for_refine.npy` 已不再只是 render-depth symlink，而是 **M3 blended depth target**；
- `target_depth_for_refine_source_map.npy` 当前约显示：verified 区域 `~1.5%`，render fallback `~98.4%`；
- `confidence_mask_brpo_*` 当前 alias 指向真正训练消费的 `train_confidence_mask_brpo_*`，兼容现有 consumer。[参见 §配置参数]

### 2.4 当前实验输出层

```text
/home/bzhang512/CV_Project/output/part3_stage1_internal/re10k-1/full/
├── 2026-04-11_mask_only_ablation_proto/
├── 2026-04-11_phase6_schema_smoke/
├── 2026-04-11_stageA_v2_smoke/
├── 2026-04-12_m4_stageA_smoke/
└── 2026-04-12_m45_stageA_eval/
```

当前最重要结果目录：
- **M3 prototype**：
  - `.../internal_prepare/re10k1__internal_afteropt__brpo_proto_v4_stage3/`
- **M4 smoke**：
  - `.../2026-04-12_m4_stageA_smoke/blended_depth/`
  - `.../2026-04-12_m4_stageA_smoke/render_depth_only/`
- **M4.5 eval**：
  - `.../2026-04-12_m45_stageA_eval/blended_depth_long/`
  - `.../2026-04-12_m45_stageA_eval/render_depth_only_long/`
  - `.../2026-04-12_m45_stageA_eval/analysis/summary.json`
  - `.../2026-04-12_m45_stageA_eval/analysis/compare.txt`

## 3. 代码脚本

### 3.1 当前主线脚本

路径：`/home/bzhang512/CV_Project/part3_BRPO/scripts/`

| 文件 | 职责 | 状态 | 备注 |
|------|------|------|------|
| `replay_internal_eval.py` | 基于 saved internal camera states 做 replay eval | ✅ 可用 | same-ply consistency 已验证通过 |
| `prepare_stage1_difix_dataset_s3po_internal.py` | internal `select / difix / fusion / verify / pack` | ✅ 可用 | 当前已支持 M1/M2/M3 产物固化与 provenance |
| `brpo_build_mask_from_internal_cache.py` | 从 internal cache 构建 verification / seed / train mask / projected depth | ✅ 可用 | 当前已支持 `branch_first|fused_first` 与 M3 depth 输出 |
| `run_pseudo_refinement.py` | v1 standalone refine 入口 | ✅ 可用 | 当前支持 `legacy|brpo` confidence mask |
| `run_pseudo_refinement_v2.py` | BRPO-style refine v2 / Stage A 入口 | ✅ 可用 | 当前已支持显式 `stageA_mask_mode` / `stageA_target_depth_mode` |

### 3.2 pseudo_branch

路径：`/home/bzhang512/CV_Project/part3_BRPO/pseudo_branch/`

| 文件 | 职责 | 状态 | 备注 |
|------|------|------|------|
| `brpo_reprojection_verify.py` | 单分支几何验证 + pseudo-view sparse verified depth | ✅ 可用 | 当前用于 M1 / M3 upstream verify |
| `brpo_confidence_mask.py` | seed-support / confidence alias 输出 | ✅ 可用 | 当前已保留 compatibility alias |
| `brpo_train_mask.py` | `seed_support → train_confidence_mask` propagation | ✅ 可用 | 当前默认研究区间接近 `10% ~ 25%` coverage |
| `brpo_depth_target.py` | 组装 blended `target_depth_for_refine` | ✅ 可用 | 当前用于 M3 pack 阶段 |
| `pseudo_camera_state.py` / `pseudo_loss_v2.py` / `pseudo_refine_scheduler.py` | Stage A trainable pseudo camera + loss + optimizer | ✅ 可用 | 当前服务于 M4 / M4.5 |
| `pseudo_fusion.py` | left/right repaired RGB 融合 | ✅ 已接入 | 当前已作为 fused-first pseudo source |
| `epipolar_depth.py` / `build_pseudo_cache.py` | 旧 EDP 线 | ✅ 保留 | 与 BRPO 新线隔离，不混写 |

### 3.3 下一阶段预计触达的代码

| 文件 | 计划改动 |
|------|----------|
| `run_pseudo_refinement_v2.py` | 继续诊断为什么 `blended_depth` 已接通但 depth loss 基本不变 |
| `pseudo_loss_v2.py` | 检查当前 depth loss 的有效梯度与作用范围 |
| `brpo_depth_target.py` / `brpo_train_mask.py` | 如有需要，做受控扩张或更严格 depth-valid 规则 |
| `replay_internal_eval.py` | 如进入下一轮可评估实验，补 refine 输出的 replay 对照 |

## 4. Pipeline

### 4.1 当前已跑通主线（到 M4）

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
  ↓
train_mask propagation
  ↓
projected_depth_left/right
  ↓
pack → blended target_depth_for_refine
  ↓
pseudo_cache canonical sample
  ↓
run_pseudo_refinement_v2.py Stage A consumer
```

### 4.2 当前最关键实验闭环

```text
M3 prototype
  ↓
M4 smoke
  ├── train_mask + blended_depth
  └── train_mask + render_depth_only
  ↓
M4.5 long eval
  ├── blended_depth_long
  └── render_depth_only_long
  ↓
analysis/summary.json + compare.txt
```

### 4.3 当前结论对应的流程位置

```text
verify / pack upstream       -> 已打通 ✅
Stage A consumer wiring      -> 已打通 ✅
blended depth nonzero signal -> 已验证 ✅
depth signal drives training -> 仍待诊断 ⚠️
Stage B                      -> 尚未开始 ⏳
```

## 5. 配置参数

### 5.1 当前 BRPO verification / train-mask / depth-target 原型参数

| 类别 | 参数 | 当前口径 |
|------|------|----------|
| verification | `verification_mode` | `fused_first` 主线，保留 `branch_first` |
| verification | `tau_reproj_px` | `4.0` |
| verification | `tau_rel_depth` | `0.15` |
| train_mask | `train_mask_mode` | `propagate` |
| train_mask | `prop_radius_px` | `2` |
| train_mask | `prop_tau_rel_depth` | `0.01` |
| train_mask | `prop_tau_rgb_l1` | `0.05` |
| depth target | `depth_fallback_mode` | `render_depth` |
| depth target | `depth_both_mode` | `average` |

### 5.2 当前 Stage A / M4 / M4.5 口径

| 类别 | 参数 | 当前主线 |
|------|------|----------|
| consumer | `target_side` | `fused` |
| consumer | `confidence_mask_source` | `brpo` |
| consumer | `stageA_mask_mode` | `train_mask` |
| consumer | `stageA_target_depth_mode` | `blended_depth` 或 `render_depth_only` |
| Stage A | `stageA_iters` | smoke `60`，long eval `300` |
| Stage A | `num_pseudo_views` | `3` |
| Stage A | `stageA_disable_depth` | 默认关闭（启用 depth） |

### 5.3 当前关键 coverage 口径

| 信号 | 当前量级 | 含义 |
|------|----------|------|
| `train_mask` coverage | `~18% ~ 20%` | 当前训练真正消费的 RGB/depth mask，处于 M2.5 建议区间内 |
| `verified depth` ratio | `~1.5% ~ 1.6%` | 当前 source map 中真正来自 BRPO verified depth 的区域 |
| `render fallback` ratio | `~98.4%` | `target_depth_for_refine` 中来自 render depth 的区域 |

说明：
- `10% ~ 25%` 是 **train_mask** 的合理研究区间；
- `~1.5%` 是当前 **verified depth correction source** 的覆盖，不应与 train_mask coverage 混为一谈。

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
- [x] M4.5：完成 `blended_depth_long vs render_depth_only_long` 的 300-iter Stage A 对照，并生成分析汇总

### 6.2 当前判断 ⚠️

- [x] `blended_depth` 已在 Stage A 中产生**非零 depth loss**，说明 M3→M4 链路真实生效
- [x] `render_depth_only` 的 depth loss 近似为零，说明 consumer 口径切换是可分辨的
- [x] 当前 Stage A 中 RGB loss 会下降，但 `blended_depth` 的 depth loss 在长一点 eval 中几乎不变
- [x] 当前更像是“**depth signal 已接通，但优化利用偏弱**”，而不是“depth 没有被读到”

### 6.3 当前进行中 ⚠️

- [ ] 诊断为什么 `blended_depth` 的 depth loss 在 Stage A 中基本不动
- [ ] 判断问题更偏向：verified depth 太稀、loss 设计不敏感、还是 Stage A 当前自由度/权重配置不合适
- [ ] 决定下一步是优先改 Stage A loss/weight，还是回 upstream 做受控 depth-valid 扩张

### 6.4 下一阶段待办 ⏳

- [ ] 做一轮 M4.6 depth-flatness diagnosis
- [ ] 在必要时补 replay eval，验证 `blended_depth` 是否带来可见外部收益
- [ ] 只有在 Stage A 对 depth signal 的利用更清楚后，再决定是否进入 Stage B

### 6.5 当前不建议优先推进的事项 🚫

- [ ] 在当前 depth signal 仍基本“平的”情况下直接进入 full Stage B
- [ ] 把 `~1.5%` 的 verified depth 覆盖误当成 train_mask 覆盖，并据此下游做错误判断
- [ ] 在没有新诊断的情况下重新回到 `50% ~ 70%` 的宽 propagation 口径

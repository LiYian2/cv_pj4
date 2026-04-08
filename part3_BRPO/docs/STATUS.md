# STATUS.md 写作规范

> 更新时间：2026-04-08 21:56
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
- 引用其他文档：`[参见 DESIGN.md §Stage1]`
- 引用历史记录：`[参见 CHANGELOG.md 2026-04-08]`

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

当前 Stage 1 的主目标是：**在 Re10k-1 sparse 上找到“几何更稳 + 感知更好”的 pseudo refine 策略。**

当前主线判断（已由 A/B/C/D/E 支撑）：
- joint refine 是必要的；
- pseudo 分支默认不应更新几何；
- densify 统计应优先由 real views 驱动；
- `lambda_pseudo` 是关键权重，过大容易把 pseudo 偏差写进最终外观。

[参见 DESIGN.md §4]

## 2. 数据结构

### 2.1 当前主用数据

当前最规范、已验证可直接使用的是：**Re10k-1 sparse**。

```text
PLY:
/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/
  s3po_re10k-1_sparse/Re10k-1_part2_s3po/2026-04-04-00-43-29/
  point_cloud/final/point_cloud.ply

Sparse train manifest:
/home/bzhang512/CV_Project/dataset/Re10k-1/part2_s3po/sparse/split_manifest.json

Sparse train RGB:
/home/bzhang512/CV_Project/dataset/Re10k-1/part2_s3po/sparse/rgb/

Pseudo cache:
/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part3_stage1/
  re10k1__s3po__tertile__sparse_v1/pseudo_cache/

对照 pseudo cache（旧）：
/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part3_stage1/
  re10k1__s3po__midpoint__sparse_v1/pseudo_cache/
```

### 2.2 Pseudo cache 结构

```text
pseudo_cache/
├── manifest.json
└── samples/{frame_id}/
    ├── camera.json
    ├── refs.json
    ├── render_rgb.png
    ├── render_depth.npy
    ├── target_rgb_left.png
    ├── target_rgb_right.png
    ├── target_depth.npy
    ├── confidence_mask.npy
    ├── confidence_mask.png
    └── diag/
```

### 2.3 当前正式实验输出结构

```text
/home/bzhang512/my_storage_500G/CV_Project/output/part3_stage1/re10k-1/sparse/
├── 2026-04-08_joint_refine_freezegeom_ablation/
│   ├── A_joint_realdensify_allparams/
│   ├── B_joint_realdensify_freezegeom/
│   ├── _baseline_sparse_gt/
│   └── summary.json
├── 2026-04-08_B_followup_lambda_vs_nodensify/
│   ├── C_joint_realdensify_freezegeom_lambda0p5/
│   └── D_joint_nodensify_freezegeom_lambda1p0/
└── 2026-04-08_joint_refine_tertile_freezegeom_lambda0p5/
    └── E_joint_realdensify_freezegeom_lambda0p5_tertile/
```

### 2.4 数据状态

| 数据集 | Sparse pseudo cache | Sparse refine formal run | 说明 |
|--------|---------------------|--------------------------|------|
| Re10k-1 | ✅ | ✅ | 当前主线，tertile 版本已完成 |
| DL3DV-2 | ⚠️ 未完成规范化验证 | ❌ | 后续再做 |
| 405841 | ⚠️ 未完成规范化验证 | ❌ | 后续再做 |

## 3. 代码脚本

### 3.1 Part3 脚本

路径：`/home/bzhang512/CV_Project/part3_BRPO/scripts/`

| 文件 | 职责 | 状态 | 备注 |
|------|------|------|------|
| `prepare_stage1_difix_dataset_s3po.py` | select / difix / pack 三阶段数据准备 | ✅ 可用 | Re10k-1 sparse 已用于生成规范输入 |
| `run_pseudo_refinement.py` | Stage1 refine 入口 | ✅ 当前主入口 | 支持 joint refine / densify source / pseudo 参数控制 |

### 3.2 pseudo_branch

路径：`/home/bzhang512/CV_Project/part3_BRPO/pseudo_branch/`

| 文件 | 职责 | 状态 |
|------|------|------|
| `build_pseudo_cache.py` | 生成 `target_depth.npy` 与 `confidence_mask.npy` | ✅ 可用 |
| `epipolar_depth.py` | EDP 深度与置信度 | ✅ 已修正 confidence 归一化 |
| `flow_matcher.py` | pseudo ↔ ref 匹配 | ✅ 可用 |
| `diag_writer.py` | 诊断图输出 | ✅ 可用 |

### 3.3 S3PO 修改点

| 文件 | 作用 | 状态 |
|------|------|------|
| `third_party/S3PO-GS/utils/slam_utils.py` | `get_loss_pseudo()` 定义 | ✅ 当前使用 |
| `third_party/S3PO-GS/utils/slam_backend.py` | 存在 `pseudo_refinement()` 方法 | ⚠️ 未接入 queue |

### 3.4 关键事实

- **loss 公式**在 `slam_utils.py`；
- **pseudo 可更新参数集合**在 `run_pseudo_refinement.py`；
- 目前正式实验全部走独立 refine 脚本（非 backend 集成）；
- `lambda_pseudo` 是 `L_total = λ_real*L_real + λ_pseudo*L_pseudo + L_reg` 中 pseudo 分支权重。

## 4. Pipeline

### 4.1 数据准备

```text
part2 sparse split + external eval render/trj
    ↓
prepare_stage1_difix_dataset_s3po.py
    ↓
pseudo_cache/
    ↓
build_pseudo_cache.py
    ↓
complete pseudo_cache (camera / refs / target_depth / confidence)
```

### 4.2 Refinement

```text
sparse PLY + sparse train manifest + pseudo_cache
    ↓
run_pseudo_refinement.py
    ├── 采样 real sparse train views → L_real
    ├── 采样 pseudo views → L_pseudo
    ├── 通过 pseudo_trainable_params/freeze 控制可更新参数
    ├── densify stats source: real/pseudo/mixed
    └── 输出 refined.ply + refinement_history.json
```

### 4.3 Evaluation

```text
refined.ply
    ↓
third_party/S3PO-GS/eval_external.py
    ↓
external_eval/
    ├── render_rgb/
    ├── eval_external.json
    └── eval_external_meta.json
```

当前同口径正式评估使用：
- `pose_mode=gt`
- `origin_mode=test_to_sparse_first`
- `infer_init_mode=gt_first`

### 4.4 指标实现说明（LPIPS）

- Part3 external eval 的 LPIPS 来自：
  `third_party/S3PO-GS/utils/external_eval_utils.py`
- 具体实现：
  `torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)`
- **不是 VGGT 指标**，与 VGGT 模型本身无直接绑定。

## 5. 配置参数

### 5.1 当前主线 refine 开关（推荐）

| 参数 | 当前主线值 | 说明 |
|------|------------|------|
| `--train_manifest` | 必填 | 打开 joint refine |
| `--lambda_real` | `1.0` | real branch 权重 |
| `--lambda_pseudo` | `0.5` | pseudo branch 权重（当前最优主线） |
| `--num_real_views` | `2` | 每轮采样 real views 数 |
| `--num_pseudo_views` | `2` | 每轮采样 pseudo views 数 |
| `--densify_stats_source` | `real` | densify 仅由 real 驱动 |
| `--densify_from_iter` | `800` | densify 起点 |
| `--densify_interval` | `200` | densify 周期 |
| `--densify_until_iter` | `1600` | densify 截止 |
| `--densify_grad_threshold` | `0.0002` | densify 梯度阈值 |
| `--min_opacity` | `0.01` | prune 相关阈值 |
| `--freeze_geometry_for_pseudo` | `True` | pseudo 不更新几何 |
| `--pseudo_trainable_params` | `appearance` | 等价于 `f_dc,f_rest,opacity` |

### 5.2 当前有效对照配置

- **B**：`lambda_pseudo=1.0 + midpoint pseudo + real densify + freeze-geom`
- **C**：`lambda_pseudo=0.5 + midpoint pseudo + real densify + freeze-geom`
- **D**：`lambda_pseudo=1.0 + midpoint pseudo + disable_densify + freeze-geom`
- **E**：`lambda_pseudo=0.5 + tertile pseudo + real densify + freeze-geom`（当前最佳）

## 6. 状态

### 6.1 已完成 ✅

- [x] Re10k-1 sparse pseudo cache 规范化完成
- [x] `run_pseudo_refinement.py` joint refine / densify source / pseudo param 控制就绪
- [x] Re10k-1 sparse 正式 A/B 完成
- [x] Follow-up C/D（lambda 与 densify 对照）完成
- [x] tertile pseudo 版本 E 完成
- [x] 同口径 external eval（GT pose）完成

### 6.2 当前最佳已验证配置 ⚠️

当前最佳：**E (`lambda_pseudo=0.5`, freeze-geom, real-densify, tertile pseudo)**

- `avg_psnr = 21.7658`
- `avg_ssim = 0.7879`
- `avg_lpips = 0.2110`
- `final_gaussians = 42454`

相对 C（`21.5173 / 0.7797 / 0.2242`）：
- PSNR +0.25
- SSIM +0.008
- LPIPS -0.013

说明：
- 在保持 C 策略不变的情况下，将 pseudo 从 midpoint 扩展到 tertile 后，三项指标继续同步提升；
- 但相对 baseline sparse GT（`LPIPS=0.1722`）仍有差距。

### 6.3 进行中 ⚠️

- [ ] 在 E 基础上继续做 `lambda_pseudo=0.25` 对照
- [ ] 在 E 基础上做 `disable_densify` 交叉对照
- [ ] 做更细粒度 pseudo 参数子集（`opacity only` / `f_dc+f_rest`）

### 6.4 待办 ❌

- [ ] `render_depth + improved confidence` 路线对照
- [ ] 加一套 `pose_mode=infer` 的外评口径（与 `pose_mode=gt` 并行报告）
- [ ] DL3DV-2 sparse 主线规范化与正式实验
- [ ] 405841 sparse 主线规范化与正式实验
- [ ] backend 集成（非当前优先级）

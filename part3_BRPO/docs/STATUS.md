# STATUS.md 写作规范

> 更新时间：2026-04-10 12:18
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

当前 Stage 1 的核心问题已经从“继续堆 GT 口径分数”切换为：**先判断当前 pseudo refine 到底改善了什么，以及它是否真的在 pose-sensitive / internal protocol 下带来收益。**

当前已经明确：
- **E policy** 在 `external eval + GT pose` 口径下，能稳定提高固定相机条件下的渲染指标；
- 但这**还不能直接证明** refine 真正改善了完整 pipeline；
- 最新 `infer` 口径结果显示，refined PLY 相对 baseline 只有很小变化，说明当前提升很可能主要体现在 **fixed-pose appearance refinement**，而不是 pose-aware refinement。

因此，当前主线目标应改为：
- 保留 E 作为 **external GT diagnostic** 下的最强 appearance baseline；
- 优先建立 **internal cache + replay eval**，验证 refined PLY 在 internal camera protocol 下是否真的优于 baseline；
- 在协议问题澄清前，不再把 GT 口径结果直接表述为“已验证成功”。

[参见 `0_RENDER_EVAL_PROTOCOL.md` 与 `01_INTERNAL_CACHE.md`]
## 2. 数据结构

### 2.1 当前主用数据

当前已完成规范化并跑通正式 E policy 的 sparse 数据有两组：**Re10k-1** 与 **DL3DV-2**。

```text
Re10k-1 sparse
- PLY:
  /home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/
    s3po_re10k-1_sparse/Re10k-1_part2_s3po/2026-04-04-00-43-29/
    point_cloud/final/point_cloud.ply
- Sparse train manifest:
  /home/bzhang512/CV_Project/dataset/Re10k-1/part2_s3po/sparse/split_manifest.json
- Sparse train RGB:
  /home/bzhang512/CV_Project/dataset/Re10k-1/part2_s3po/sparse/rgb/
- Pseudo cache:
  /home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part3_stage1/
    re10k1__s3po__tertile__sparse_v1/pseudo_cache/

DL3DV-2 sparse
- PLY:
  /home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/dl3dv-2/
    s3po_dl3dv-2_sparse/DL3DV-2_part2_s3po/2026-04-04-00-43-38/
    point_cloud/final/point_cloud.ply
- Sparse train manifest:
  /home/bzhang512/CV_Project/dataset/DL3DV-2/part2_s3po/sparse/split_manifest.json
- Sparse train RGB:
  /home/bzhang512/CV_Project/dataset/DL3DV-2/part2_s3po/sparse/rgb/
- Pseudo cache:
  /home/bzhang512/CV_Project/dataset/DL3DV-2/part3_stage1/
    dl3dv2__s3po__tertile__sparse_v1/pseudo_cache/
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

/home/bzhang512/my_storage_500G/CV_Project/output/part3_stage1/dl3dv-2/sparse/
└── 2026-04-09_joint_refine_tertile_freezegeom_lambda0p5/
    ├── E_joint_realdensify_freezegeom_lambda0p5_tertile/
    ├── _baseline_sparse_gt/
    └── summary.json
```
### 2.4 数据状态

| 数据集 | Sparse pseudo cache | Sparse refine formal run | 说明 |
|--------|---------------------|--------------------------|------|
| Re10k-1 | ✅ | ✅ | external GT diagnostic 已完成；infer/internal 仍待补证 |
| DL3DV-2 | ✅ | ✅ | external GT diagnostic 已完成；infer/internal 仍待补证 |
| 405841 | ⚠️ 路线待定 | ❌ | GT-reprojection / EDP 路线仍待明确 |

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

当前评估需要分三层理解：
- **external GT**：固定准确相机下的渲染诊断；
- **external infer**：pose-sensitive stress test；
- **internal protocol**：目标协议，但当前尚未建立 replay 闭环。

因此，当前历史 A/B/C/D/E 主结果虽然都基于同一口径、彼此可比，但它们**只说明 fixed-pose 条件下的 map/render 变化**，还不能单独证明完整 pipeline 变好。

当前 external 评估相关参数：
- `pose_mode=gt`（历史主结果）
- `origin_mode=test_to_sparse_first`
- `infer_init_mode=gt_first`（infer stress test 使用）

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
- **E**：`lambda_pseudo=0.5 + tertile pseudo + real densify + freeze-geom`（当前 **external GT diagnostic** 下最佳）

## 6. 状态

### 6.1 已完成 ✅

- [x] Re10k-1 sparse pseudo cache 规范化完成
- [x] `run_pseudo_refinement.py` joint refine / densify source / pseudo param 控制就绪
- [x] Re10k-1 sparse 正式 A/B/C/D/E 完成
- [x] DL3DV-2 sparse 单组 E 完成
- [x] Re10k-1 / DL3DV-2 的 external GT-pose 同口径对照完成
- [x] Re10k-1 sparse external infer 口径补跑完成
- [x] internal / external protocol 差异已整理成文档
- [x] internal cache / replay 路线已有可执行规划

### 6.2 当前最强“诊断口径”配置 ⚠️

当前在 **external eval + GT pose** 这一固定相机诊断口径下，最强配置仍是：
**E (`lambda_pseudo=0.5`, freeze-geom, real-densify, tertile pseudo)**

Re10k-1 sparse：
- `avg_psnr = 21.7658`
- `avg_ssim = 0.7879`
- `avg_lpips = 0.2110`
- `final_gaussians = 42454`

DL3DV-2 sparse：
- `avg_psnr = 14.0617`
- `avg_ssim = 0.4526`
- `avg_lpips = 0.6893`

**但这一定义必须带协议前缀。** 当前它只能说明：在固定准确相机下，refined PLY 的渲染更接近 GT；它还**不能单独证明** pose-aware 或 internal protocol 下的完整收益。

### 6.3 当前关键未决问题 ⚠️

当前最大的不确定性不是 E 还能不能再调，而是：**我们到底 refine 了什么。**

目前已有证据指向：
- 当前 refine 更像 **fixed-pose appearance refinement**；
- 它未必改善 external infer，未必改善 internal tracked-camera protocol。

Re10k-1 sparse 的最新 infer 结果：
- refined（fused RGB-only）：`PSNR 13.6111 / SSIM 0.5067 / LPIPS 0.4061`
- baseline sparse infer：`PSNR 13.3265 / SSIM 0.5027 / LPIPS 0.3791`
- `pose_success_rate = 67.4% (182/270)`

说明：
- infer 口径下相对 baseline 只有很小变化；
- 当前还不能说“refine 真正提升了 end-to-end pipeline”；
- 之前所有 GT 提升都需要重新解读为：**fixed-pose 条件下 map/render 更优**，而不是自动等价于“位姿/协议也更优”。

### 6.4 当前最高优先级进行中 ⚠️

- [ ] 在 S3PO 侧导出 **internal cache**（建议 before_opt / after_opt 都能导）
- [ ] 建立 **replay_internal_eval.py**，用同一组 saved internal camera states 回放 baseline / refined PLY
- [ ] 回答核心问题：refined PLY 在 internal protocol 下是否真的优于 baseline

### 6.5 条件性后续任务 ⏳

只有当 internal replay 证明 refined PLY 确有收益后，再继续：
- [ ] internal prepare → pseudo cache → refine 的完整 internal 路线
- [ ] fused pseudo / lambda / densify 等更细的策略对照
- [ ] 405841 sparse 路线选择（GT-reprojection vs EDP）

### 6.6 当前不建议优先推进的事项 🚫

在 internal replay 结果出来之前，暂不建议优先：
- [ ] 单纯继续堆 `external GT` 分数
- [ ] 直接把 GT 口径结果写成“已验证成功”
- [ ] 先做 backend 集成式 pseudo refine

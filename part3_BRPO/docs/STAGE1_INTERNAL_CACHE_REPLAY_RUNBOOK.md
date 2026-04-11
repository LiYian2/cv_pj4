# STAGE1_INTERNAL_CACHE_REPLAY_RUNBOOK.md

> 用途：给后续 agent 复用 **internal 路线的完整协议**。
> 范围：覆盖 `part2 full rerun → internal cache → replay consistency → internal prepare → difix → fuse → BRPO mask → pseudo cache → refine → replay eval`。
> 原则：先把输入资产和 schema 层级定清楚，再做训练实验；可复用输入放在 `part2 output`，实验结果放在 `part3 output`。
> 当前状态：Phase 6 第一轮已在 `re10k1__internal_afteropt__brpo_proto_v3` 上真实跑通 `fusion → verify → pack`，并完成 `fused + brpo` refine smoke。Phase 7A 的 `run_pseudo_refinement_v2.py` 也已完成最小版并通过 2-iter Stage A smoke。

## 1. 先看什么

按顺序看：
1. `STATUS.md` —— 当前 internal 路线进度、哪些阶段已跑通。
2. `DESIGN.md` —— 当前 schema 决策与 Phase 6 / 7 的主线判断。
3. `CHANGELOG.md` —— 最近一次 internal prepare / BRPO / ablation 的过程记录。
4. `1_INTERNAL_CACHE.md` —— 当前工程边界和阶段划分。

## 2. 这条 internal 路线的协议目标

当前 internal 路线的目标不是继续沿旧 sparse external GT 思路堆实验，而是建立下面这条稳定链：

```text
1. part2 full rerun
   → 保存 internal camera states / render cache
2. replay consistency
   → 验证 replay 与官方 internal eval 足够接近
3. internal prepare (selection)
   → 从 internal cache 选择 pseudo samples
4. difix
   → 生成 left/right repaired target RGB
5. fuse
   → 生成 fused target RGB 与 fusion metadata
6. BRPO-style mask
   → 用 bidirectional verification 生成 side-aware / fused confidence
7. pseudo cache
   → 形成可被 refine v1/v2 消费的 canonical 输入
8. refine + replay eval
   → 对 baseline / refined 在同一组 internal camera states 下闭环比较
```

## 3. 数据结构分层（必须固定）

### 3.1 Layer A：part2 full rerun 的正式源数据

位置：
```text
/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/
```

当前正式 run：
```text
/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/
  s3po_re10k-1_full_internal_cache/
  Re10k-1_part2_s3po/2026-04-11-05-33-58/
```

标准结构：
```text
<run_root>/
├── config.yml
├── point_cloud/final/point_cloud.ply
├── psnr/
│   ├── before_opt/final_result.json
│   └── after_opt/final_result.json
├── plot/
└── internal_eval_cache/
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

注意：
- 顶层 S3PO 原生 `render_rgb/` 可能是空目录；正式输入统一认 `internal_eval_cache/<stage>/...`。
- 当前 `camera_states.json` 只保存一份，因为 `color_refinement()` 只改 gaussians，不改 pose。

### 3.2 Layer B：BRPO Phase B/C 原型输出

位置：
```text
<run_root>/internal_eval_cache/brpo_phaseB/
<run_root>/internal_eval_cache/brpo_phaseC/
```

作用：
- 调 `tau_reproj_px / tau_rel_depth`
- 验证单分支 / 双分支几何核验链是否成立
- 观察 `support_left/right/both/single` 是否几乎全空

它是 **原型验证层**，不是最终 canonical training input。

### 3.3 Layer C：internal prepare 的 canonical training-input 层

位置：
```text
<run_root>/internal_prepare/<prepare_key>/
```

当前 Phase 6 要锁定为：

```text
<run_root>/internal_prepare/<prepare_key>/
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
│       ├── confidence_mask_fused.npy      # 若继续沿旧 fusion 语义保留
│       └── fusion_meta.json
├── verification/
│   └── brpo_phaseC/<stage_tag>/
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
        ├── confidence_mask_brpo_left.npy
        ├── confidence_mask_brpo_right.npy
        ├── confidence_mask_brpo_fused.npy
        ├── support_left.npy
        ├── support_right.npy
        ├── support_both.npy
        ├── support_single.npy
        ├── fusion_meta.json
        ├── verification_meta.json
        └── diag/
```

约束：
- `left/right/fused` target 必须同时保留；
- BRPO confidence 不再只保留单个 `confidence_mask_brpo.npy`，要明确三种消费语义；
- `target_depth_for_refine.npy` 是给 v2 / Stage A 预留，不要求本阶段就启用 depth loss。

### 3.4 Layer D：part3 实验输出层

位置建议：
```text
/home/bzhang512/my_storage_500G/CV_Project/output/part3_stage1_internal/re10k-1/full/<experiment_name>/
```

建议结构：
```text
<experiment_root>/
├── COMMANDS.txt
├── refine/
│   ├── refined.ply
│   ├── refinement_history.json
│   └── refine.log
├── replay_eval/
│   ├── after_opt/
│   └── before_opt/
└── summaries/
```

不要把正式实验输出继续塞回 `part2 output` 根目录。

## 4. 环境分工

统一先做：
```bash
export TMPDIR=/data/bzhang512/tmp
mkdir -p "$TMPDIR"
```

当前已验证环境：
- `s3po-gs`：`slam.py`、`replay_internal_eval.py`、`run_pseudo_refinement.py`
- `reggs`：`prepare_stage1_difix_dataset_s3po_internal.py`、`Difix`、`BRPO verification`

不要默认用系统 `python3`。

## 5. Re10k-1 full 的 KF 协议

当前 full rerun 使用固定 KF：
```yaml
force_keyframe_indices: [0, 34, 69, 104, 139, 173, 208, 243, 278]
```

因此：
- `num_frames = 279`
- `num_kf = 9`
- `num_non_kf = 270`

后续任何 replay / prepare / BRPO verification，都应以：
- `internal_eval_cache/manifest.json` 的 `kf_indices / non_kf_indices`
- `stage_meta.json` 的 `rendered_non_kf_frames`

为准，不要自己再写另一套 KF 列表。

## 6. 标准执行协议（1–8）

### Step 1. part2 full rerun

准备 full yaml，把 `Results.save_dir` 指到：
```text
/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache
```

运行：
```bash
source /home/bzhang512/miniconda3/etc/profile.d/conda.sh
conda activate s3po-gs
export TMPDIR=/data/bzhang512/tmp
mkdir -p "$TMPDIR"
export S3PO_EXPORT_INTERNAL_CACHE=1

cd /home/bzhang512/CV_Project/third_party/S3PO-GS
python slam.py --config /tmp/s3po_re10k1_full_internal_cache.yaml
```

### Step 2. replay consistency check

先做 same-ply replay，验证 replay 与官方 internal eval 是否接近：
```bash
source /home/bzhang512/miniconda3/etc/profile.d/conda.sh
conda activate s3po-gs
export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO:$PYTHONPATH

cd /home/bzhang512/CV_Project/part3_BRPO
python scripts/replay_internal_eval.py \
  --internal-cache-root <run_root>/internal_eval_cache \
  --stage-tag after_opt \
  --ply-path <run_root>/internal_eval_cache/after_opt/point_cloud/point_cloud.ply
```

### Step 3. internal prepare（selection）

入口：
```bash
source /home/bzhang512/miniconda3/etc/profile.d/conda.sh
conda activate reggs
export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO:$PYTHONPATH

cd /home/bzhang512/CV_Project/part3_BRPO
python scripts/prepare_stage1_difix_dataset_s3po_internal.py \
  --stage select \
  --internal-cache-root <run_root>/internal_eval_cache \
  --run-key <prepare_key> \
  --scene-name Re10k-1 \
  --stage-tag after_opt \
  --placement tertile
```

支持：
- 自动按 gap 选点（`midpoint / tertile / both`）
- 或 `--frame-ids ...` 显式指定

### Step 4. internal Difix（left/right）

当前仍使用 `reggs` 环境。输入是 internal `render_rgb`，条件参考是左右真实 `ref_rgb`。

```bash
python scripts/prepare_stage1_difix_dataset_s3po_internal.py \
  --stage difix \
  --internal-cache-root <run_root>/internal_eval_cache \
  --run-key <prepare_key> \
  --scene-name Re10k-1 \
  --stage-tag after_opt \
  --prompt "remove degradation"
```

输出：
```text
internal_prepare/<prepare_key>/difix/left_fixed/
internal_prepare/<prepare_key>/difix/right_fixed/
```

### Step 5. fuse（left/right → fused）

Phase 6 之后，正式 protocol 要把这一步放进 prepare pipeline。

目标：
- 以 `render_rgb + left_fixed + right_fixed` 生成 `target_rgb_fused.png`
- 写 `fusion_meta.json`
- 先做 `rgb-only fused`，不要默认开启 `A_depth`

推荐复用：
- `pseudo_branch/pseudo_fusion.py`

当前要求：
- `fuse` 是 schema 中的正式环节，不再只是文档预留项；
- 即使这一阶段先不跑 fused ablation，也要把 fused 输出纳入 canonical pseudo_cache。

### Step 6. BRPO verification / side-aware confidence

当前实现原则：
- 不走 EDP
- 使用 **方案 B**：按需用 `ref pose + stage PLY` 现渲染 KF depth
- verification 消费 `Difix left/right` repaired 图，而不是只看 raw render

当前入口：
- `scripts/brpo_build_mask_from_internal_cache.py`

Phase 6 的协议要求：
- 产出 `support_left/right/both/single`
- 进一步显式写出：
  - `confidence_mask_brpo_left.npy`
  - `confidence_mask_brpo_right.npy`
  - `confidence_mask_brpo_fused.npy`

不要继续只保留一个 `confidence_mask_brpo.npy` 给所有 `target_side` 通吃。

### Step 7. pack 成 canonical internal pseudo_cache

入口仍为：
```bash
python scripts/prepare_stage1_difix_dataset_s3po_internal.py \
  --stage pack \
  --internal-cache-root <run_root>/internal_eval_cache \
  --run-key <prepare_key> \
  --scene-name Re10k-1 \
  --stage-tag after_opt \
  --target-rgb-source difix
```

Phase 6 的 pack 目标：
- 把 `left/right/fused` target 都写齐；
- 把 side-aware / fused BRPO confidence 都写齐；
- `source_meta.json`、`verification_meta.json`、`fusion_meta.json` 都要可审计；
- 形成 refine v1 / v2 都能消费的 canonical `pseudo_cache/`。

### Step 8. refine + replay eval

当前分两层：

1. **v1 当前口径**
   - `run_pseudo_refinement.py`
   - 先做 schema 验证与最小 consumer 验证

2. **Phase 7 主线**
   - `run_pseudo_refinement_v2.py`
   - 当前已落地并跑通 `Stage A` 最小版：pseudo pose delta + exposure stabilization
   - 当前 smoke 输出示例：
     `/home/bzhang512/my_storage_500G/CV_Project/output/part3_stage1_internal/re10k-1/full/2026-04-11_stageA_v2_smoke/fused_brpo_stageA_2iter`

无论 v1 还是 v2，最终都要再做：
```text
replay_internal_eval under same internal camera states
```

这是 internal route 的正式闭环。

## 7. 当前 depth / mask 协议

当前 internal 路线里，depth 分三层：

1. `render_depth.npy`
   - 当前地图渲染深度
   - 用于诊断 / verification / consistency
2. `target_depth.npy`
   - 当前兼容字段
   - 暂不作为 v1 主几何监督
3. `target_depth_for_refine.npy`
   - 给 Phase 7 / v2 预留
   - 可作为 mask-aware depth target

当前 mask 协议：
- `support_left / right / both / single` 是 verification 基础输出；
- `left/right/fused` 三种 `target_side` 不应继续共用同一个 BRPO confidence；
- side-aware / fused confidence 必须显式区分。

## 8. smoke / dev 目录怎么处理

```text
/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/internal_cache_smoke
```

它的角色仍然是：
- 开发测试产物
- replay / BRPO 原型验证时的轻量样本

它不是 canonical 正式链路。full 路线稳定后可清理，但在 docs / protocol 沉淀完成前建议先保留。

## 9. 当前不在这份 runbook 里展开实现的内容

这份 runbook 当前不展开：
- full Stage B two-stage joint optimize
- backend pseudo integration
- EDP 与 BRPO 进一步融合
- 405841 的路线细化

如果任务开始要求这些内容，说明已经进入下一阶段，需要在 `DESIGN.md` / `1_INTERNAL_CACHE.md` 上重新收敛。

## 10. 最低可交付结果

一次完整 internal route 推进，至少应交付：
- `part2 full rerun` 的 `run_root`
- 完整 `internal_eval_cache/`
- same-ply replay consistency 结果
- `internal_prepare/<prepare_key>/pseudo_cache/`
- `Difix left/right` 结果
- `fused target` 结果
- `support_left/right/both/single` 与 side-aware / fused BRPO confidence
- 一段简洁总结，明确写：
  - 可复用输入放在哪
  - prepare 输出放在哪
  - 实验输出放在哪
  - 当前阶段是 schema 固化、Stage A，还是已经进入 full two-stage

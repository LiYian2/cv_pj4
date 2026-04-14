# STAGE1_SPARSE_E_RUNBOOK.md

> 用途：给后续 agent 复用 **sparse 单组 E policy** 跑法。
> 例子：`DL3DV-2 sparse / 2026-04-09`。
> 原则：先核对输入与环境，再按 `select → check → difix → pack → cache → refine → eval` 执行。

## 1. 开跑前先看什么

按顺序看：
1. `STATUS.md` —— 当前哪些数据已经跑过、哪些还没跑。
2. `DESIGN.md` —— 当前 E policy 与 `target_depth / confidence_mask` 的真实语义。
3. `CHANGELOG.md` —— 最近一次同类实验是怎么跑通的。
4. `RENDER_EVAL_PROTOCOL.md` —— 避免混淆 internal / external eval 口径。

不要上来直接复用旧命令；先确认这次要跑的是 **哪个 scene、哪个 split、哪个 part2 run timestamp**。

## 2. 必查输入

至少确认下面这些路径和文件：
- `ply_path`：part2 sparse 的最终 PLY
- `sparse_manifest`：`dataset/<scene>/part2_s3po/sparse/split_manifest.json`
- `test_manifest`：`dataset/<scene>/part2_s3po/test/split_manifest.json`
- `sparse_rgb_dir`：`dataset/<scene>/part2_s3po/sparse/rgb/`
- `render_rgb_dir`：对应 external eval run 的 `render_rgb/`
- `render_depth_dir`：同一 run 的 `render_depth_npy/`
- `trj_json`：同一 run 的 `plot/trj_external_infer.json`
- `eval config`：`part2_s3po/configs/s3po_<scene>_test.yaml`

额外检查三件事：
1. `sparse_manifest` 与 `test_manifest` 的 `selected_indices` 是否符合预期；
2. `render_rgb_dir / render_depth_dir / trj_json` 是否来自**同一个 timestamp**；
3. `eval config` 指向的 dataset 是否是 test split，而不是 sparse split。

## 3. 环境与运行前设置

统一先做：
```bash
export TMPDIR=/data/bzhang512/tmp
mkdir -p "$TMPDIR"
```

当前已验证的环境分工：
- `reggs`：`prepare_stage1_difix_dataset_s3po.py`、`build_pseudo_cache.py`
- `s3po-gs`：`run_pseudo_refinement.py`、`eval_external.py`

不要默认用系统 `python3`。如果脚本一上来报 `numpy` / `munch` / `diffusers` 缺失，优先切对环境，不要先装包。

## 4. 当前路线判断

- **Re10k-1 / DL3DV-2**：当前默认走 **EDP route**。`target_depth` 来自 EDP fused depth，`confidence_mask` 来自 EDP confidence。
- **405841**：当前默认仍是 **GT-reprojection route**；如果要改成 EDP，要明确说明，不要把“改成 EDP”和“直接拿 render depth 当 target depth”混为一谈。

## 5. 标准执行顺序

下面用变量写法，先把路径填对，再执行。

```bash
SCENE=DL3DV-2
RUN_KEY=dl3dv2__s3po__tertile__sparse_v1
PART3_DATA_ROOT=/home/bzhang512/CV_Project/dataset/$SCENE/part3_stage1/$RUN_KEY
OUT_ROOT=/home/bzhang512/my_storage_500G/CV_Project/output/part3_stage1/dl3dv-2/sparse/2026-04-09_joint_refine_tertile_freezegeom_lambda0p5
EXP=E_joint_realdensify_freezegeom_lambda0p5_tertile
```

### Step 1. select

用 `reggs` 跑：
```bash
python scripts/prepare_stage1_difix_dataset_s3po.py \
  --stage select \
  --scene-name $SCENE \
  --run-key $RUN_KEY \
  --dataset-root /home/bzhang512/CV_Project/dataset \
  --sparse-manifest <.../sparse/split_manifest.json> \
  --test-manifest <.../test/split_manifest.json> \
  --sparse-rgb-dir <.../sparse/rgb> \
  --render-rgb-dir <.../render_rgb> \
  --render-depth-dir <.../render_depth_npy> \
  --trj-json <.../plot/trj_external_infer.json> \
  --placement tertile \
  --prompt "remove degradation"
```

执行后必须检查：
- `manifests/selection_summary.json`
- `manifests/pseudo_selection_manifest.json`

重点核对：
- `num_selected` 是否合理；tertile 通常是 `2 × gap_count`；
- `selected_frame_ids` 是否分布在每个 sparse gap 的两个三分位附近；
- `test_idx` 和 `render_rgb/render_depth_npy` 文件是否能对上。

### Step 2. check

这一步不要省。至少看：
- `selection_summary.json`
- 几个 `inputs/raw_render/*.png`、`inputs/left_ref/*.png`、`inputs/right_ref/*.png` 是否链接正确
- `sparse_manifest` / `test_manifest` 的 frame naming 是否和 `scene_name` 的规则一致

如果这里就发现 frame id、test idx 或输入路径不对，先修，再进 Difix。

### Step 3. difix

用 `reggs` 跑：
```bash
python scripts/prepare_stage1_difix_dataset_s3po.py \
  --stage difix \
  ...（其余参数与 select 保持一致）...
```

执行后检查：
- `difix/left_fixed/*.png` 数量
- `difix/right_fixed/*.png` 数量
- `manifests/difix_run_manifest.json`

如果 HuggingFace 外网不通，但日志里显示 “Will try to load from local cache”，且继续产图，就不用管。

### Step 4. pack

用 `reggs` 跑：
```bash
python scripts/prepare_stage1_difix_dataset_s3po.py \
  --stage pack \
  ...（其余参数与 select 保持一致）...
```

执行后检查：
- `pseudo_cache/manifest.json`
- `pseudo_cache/samples/<frame_id>/camera.json`
- `pseudo_cache/samples/<frame_id>/refs.json`
- `render_rgb.png / render_depth.npy / target_rgb_left.png / target_rgb_right.png` 是否都在

### Step 5. cache

用 `reggs` 跑：
```bash
python pseudo_branch/build_pseudo_cache.py \
  --pseudo-cache-root "$PART3_DATA_ROOT/pseudo_cache" \
  --sparse-rgb-dir <.../sparse/rgb>
```

执行后检查：
- `target_depth.npy` 数量是否等于 sample 数
- `confidence_mask.npy` / `confidence_mask.png` 数量是否等于 sample 数
- 日志里是否 `Errors: 0`

这里顺手记录路线：
- Re10k-1 / DL3DV-2：应该看到 `method=edp`
- 405841：默认应是 GT-reprojection 路线，除非任务明确要求切到 EDP

### Step 6. refine

用 `s3po-gs` 跑，**显式写出 E policy 参数**，不要依赖隐式默认：
```bash
python scripts/run_pseudo_refinement.py \
  --ply_path <part2 sparse ply> \
  --pseudo_cache "$PART3_DATA_ROOT/pseudo_cache" \
  --output_path "$OUT_ROOT/$EXP/refined.ply" \
  --num_iterations 2000 \
  --sh_degree 0 \
  --train_manifest <.../sparse/split_manifest.json> \
  --train_rgb_dir <.../sparse/rgb> \
  --lambda_real 1.0 \
  --lambda_pseudo 0.5 \
  --num_real_views 2 \
  --num_pseudo_views 2 \
  --rgb_boundary_threshold 0.01 \
  --freeze_geometry_for_pseudo \
  --pseudo_trainable_params appearance \
  --densify_interval 200 \
  --densify_from_iter 800 \
  --densify_until_iter 1600 \
  --densify_grad_threshold 0.0002 \
  --densify_stats_source real \
  --min_opacity 0.01 \
  --late_size_threshold 20 \
  --seed 0
```

执行后至少保留：
- `refined.ply`
- `refinement_history.json`
- `refine.log`
- `COMMAND.refine_eval.txt`

### Step 7. eval

正式结果优先用 **GT pose external eval**，保证和历史 A/B/C/D/E 同口径：
```bash
python /home/bzhang512/CV_Project/third_party/S3PO-GS/eval_external.py \
  --config <part2_s3po/configs/s3po_<scene>_test.yaml> \
  --ply_path "$OUT_ROOT/$EXP/refined.ply" \
  --save_dir "$OUT_ROOT/$EXP/external_eval_gt" \
  --pose_mode gt \
  --origin_mode test_to_sparse_first \
  --infer_init_mode gt_first \
  --begin 0 --end <test frame count>
```

执行后汇总：
- `external_eval_gt/psnr/gt/final_result.json`
- `external_eval_gt/eval_external.json`
- `external_eval_gt/eval_external_meta.json`

如果需要证明“这次 refine 是否真的比原始 sparse ply 更好”，再补跑一个 **baseline sparse GT**，口径必须完全一样，不能拿 infer-pose baseline 混进来。

## 6. agent 自查清单

交付前至少确认：
1. sample 数、`target_depth.npy` 数、`confidence_mask.npy` 数一致；
2. `refined.ply` 已生成；
3. `external_eval_gt/psnr/gt/final_result.json` 已生成；
4. 若做 baseline 对照，baseline 与 refine 必须是**同口径 GT pose external eval**；
5. 总结里明确写清：
   - scene / split
   - pseudo frame 数与 placement
   - refine 核心参数
   - final loss / final gaussians
   - final metrics
   - delta vs baseline（如果补跑了 baseline）

## 7. 本次 DL3DV-2 例子结果

- pseudo frame 数：`18`
- policy：`freeze-geom + appearance-only pseudo + real-densify + lambda_pseudo=0.5 + tertile`
- refine 结果：`final_loss_total=0.1609`，`final_gaussians=32041`
- external eval（GT pose）：`PSNR 14.0617 / SSIM 0.4526 / LPIPS 0.6893`
- baseline sparse GT：`PSNR 8.3061 / SSIM 0.2884 / LPIPS 0.7583`

后续 agent 如果只是复用这套流程，优先沿着这份 runbook 执行，不要自己改 protocol。

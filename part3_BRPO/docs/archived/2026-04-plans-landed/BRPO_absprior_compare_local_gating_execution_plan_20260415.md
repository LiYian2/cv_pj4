# Part3 BRPO：abs prior 标定 + legacy vs v2 短对照 + local Gaussian gating 工程落地方案（2026-04-15）

## 1. 这份方案解决什么问题

当前项目已经不再是“新链路还没接上”的阶段，而是：

- `signal_v2` 已能独立产出，并被 `run_pseudo_refinement_v2.py` 真实读取；
- 当前卡点变成：`abs prior` 还没标定、`legacy vs v2` 还没做真正强对照、`pseudo supervision scope -> Gaussian optimization scope` 的结构失配还没通过 `local Gaussian gating` 收住；
- 后续如果继续推进 StageA.5 / StageB 或更重的 SPGM，而不先把这三件事做完，解释空间会越来越差。

这份文档的目标是：基于当前真实代码结构，给出一条可以按阶段逐步执行的工程落地顺序。后续执行默认遵守这份方案，而不是继续用临时口头判断推进。

---

## 2. 当前代码结构的已确认事实（基于真实代码，不是文档想象）

### 2.1 已经存在、可以直接复用的部分

1. `scripts/run_pseudo_refinement_v2.py`
   - 已支持 `--signal_pipeline {legacy, brpo_v2}`；
   - 已支持 `brpo_v2_raw / brpo_v2_cont / brpo_v2_depth / brpo_v2 / target_depth_for_refine_v2_brpo`；
   - 已支持 split abs prior：`--stageA_lambda_abs_t`、`--stageA_lambda_abs_r`、`--stageA_abs_pose_robust`、`--stageA_abs_pose_scale_source`；
   - `stageA_history.json` 已记录：`loss_depth_seed / loss_depth_dense / loss_abs_pose_trans / loss_abs_pose_rot / abs_pose_rho_norm / abs_pose_theta_norm / scene_scale_used / final_true_pose_delta_aggregate`；
   - `--stageA_log_grad_contrib` 已可直接在短跑里记录 `rgb / depth_total / abs_pose_rot` 等梯度贡献，不需要先另写一个全新诊断入口。

2. `pseudo_branch/pseudo_loss_v2.py`
   - 已实现 `compute_abs_pose_components()`；
   - 已实现 scene-scale 归一的 `absolute_pose_prior_loss_scaled()`；
   - 已支持 robust penalty：`charbonnier / huber / l2`。

3. `scripts/replay_internal_eval.py`
   - 已可在固定 internal camera states 上，对任意 `ply` 做 replay，并产出 `avg_psnr / avg_ssim / avg_lpips` 与 internal eval 的 delta；
   - 可以直接作为 short compare 的“前后差值”度量入口。

4. `scripts/build_brpo_v2_signal_from_internal_cache.py`
   - 已能在现有 prepare root 上，从 `fusion/samples/<frame_id>/` 产出隔离的 `signal_v2/`；
   - 默认输出到 `<prepare_root>/signal_v2`，不会覆写 legacy `pseudo_cache/` 旧文件名。

5. `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
   - 已支持 `--selection-manifest` 或 `--frame-ids`；
   - 当前 canonical E1 winner 可以直接继续复用，不需要重新发明 prepare schema。

6. `scripts/run_pseudo_refinement.py`（v1 老版）
   - 已有“pseudo backward / real backward 分开处理”的先例；
   - 已直接使用过 `render(...)` 返回包里的 `visibility_filter / viewspace_points / radii`；
   - 这正是后续 `local Gaussian gating` 第一版最该借的工程锚点。

### 2.2 当前明确存在的缺口

1. `local Gaussian gating` 还没有实际落地。
   - 当前 `run_pseudo_refinement_v2.py` 的 StageA.5 / StageB 仍是单次 `total_loss.backward()` + `gaussian_optimizer.step()`；
   - 还没有 per-Gaussian local gate，也没有 pseudo-side grad mask。

2. `legacy vs v2` 还没有 apples-to-apples 短对照。
   - 当前只有 `20260415_signal_v2_coverage_tmp` 的 3-frame coverage + 3-iter smoke；
   - 这只能证明 wiring，不足以证明价值。

3. `abs prior` 虽已接入，但还没有被“固定成 compare 背景口径”。
   - 现在不能再用单纯 `lambda_abs_pose=0 / 0.1 / 1 / 10 / 100` 这种旧粗扫逻辑继续推进；
   - 需要固定到 split 的 `lambda_abs_t / lambda_abs_r` 口径，并尽快给出一个“够稳但不过强”的背景配置。

4. 现有一些诊断脚本仍是旧口径产物。
   - 例如 `scripts/diagnose_stageA_loss_contrib.py` 还没有跟上 `signal_pipeline / stageA_rgb_mask_mode / stageA_depth_mask_mode / signal_v2_root` 这套新入口；
   - 因此本方案不把它作为第一优先级依赖，而是优先复用 `run_pseudo_refinement_v2.py --stageA_log_grad_contrib`。

### 2.3 当前推荐的 canonical 输入根（直接复用，避免重复拷贝）

考虑到当前磁盘已经比较紧：`/` 约 97% 使用，`/data` 已接近满盘，因此后续短对照默认不复制整个 prepare root，而是在现有 canonical root 上做隔离输出。

固定使用：

```bash
PROJECT_ROOT=/home/bzhang512/CV_Project/part3_BRPO
PY=/home/bzhang512/miniconda3/envs/s3po-gs/bin/python
INTERNAL_CACHE_ROOT=/data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache
BASE_PLY=$INTERNAL_CACHE_ROOT/after_opt/point_cloud/point_cloud.ply
PREPARE_ROOT=/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare
PSEUDO_CACHE=$PREPARE_ROOT/pseudo_cache
SIGNAL_V2_ROOT=$PREPARE_ROOT/signal_v2
```

其中：
- `PREPARE_ROOT` 已经是 E1 winner 的 8-frame canonical root；
- `selected_frame_ids = [23, 57, 92, 127, 162, 196, 225, 260]`；
- `20260415_signal_v2_coverage_tmp` 只保留为 3-frame smoke，不作为正式 compare anchor。

---

## 3. 总执行顺序与阶段门禁

固定执行顺序：

1. `P0`: 标定 abs prior
2. `P1`: 做 legacy vs v2 短对照
3. `P2`: 实现并验证 local Gaussian gating

固定门禁：

- `P0` 没完成前，不进入正式 `P1` 结论判断；
- `P1` 没完成前，不把当前窄监督直接扩到 A.5 / StageB 长跑；
- `P2` 没完成前，不把 StageB grid / full SPGM 当主线；
- 所有阶段默认都以 `E1 signal-aware-8` 作为 pseudo set，不再退回 midpoint8 重新混口径。

---

## 4. Phase P0：abs prior 标定

## 4.1 目标

把 Stage A 固定成“稳 pose 的短跑诊断阶段”，给出一个后续 compare 统一复用的 abs prior 背景配置。

这里的目标不是一次性找全局最优值，而是尽快确定：

- `lambda_abs_t / lambda_abs_r` 是否都进入了有效区间；
- 它们与 `depth_total` 的梯度量级是否至少同阶；
- 它们是否能降低 drift，而不明显压死 depth 分支；
- 后续 `P1 legacy vs v2` 应该在什么 abs prior 背景下进行。

## 4.2 本阶段不做什么

- 不做 StageB；
- 不做 local gating；
- 不在这个阶段引入 `signal_v2`，避免把 signal 分支变化和 pose prior 标定混在一起；
- 不再做旧式大跳格粗扫。

## 4.3 当前代码上怎么落地

本阶段先不改训练主循环，直接复用现有入口：

- 训练：`scripts/run_pseudo_refinement_v2.py`
- 梯度观察：`--stageA_log_grad_contrib --stageA_log_grad_interval 1`
- replay：`scripts/replay_internal_eval.py`

固定使用 legacy branch 作为标定背景：

```text
signal_pipeline=legacy
stageA_rgb_mask_mode=train_mask
stageA_depth_mask_mode=train_mask
stageA_target_depth_mode=target_depth_for_refine_v2
stageA_depth_loss_mode=source_aware
num_pseudo_views=4
```

这样做的原因：
- abs prior 标定的目的是先稳住 pose 侧背景条件；
- 不应让 `signal_v2` 的覆盖变化先把 P0 搅浑。

## 4.4 建议新增的最小工程件

建议新增一个汇总脚本，而不是先改训练主逻辑：

```text
scripts/summarize_stageA_compare.py
```

职责：
- 读取每个 run 的 `stageA_history.json`；
- 提取：
  - `mean_confidence_nonzero_ratio`
  - `mean_target_depth_verified_ratio`
  - `loss_depth_seed / loss_depth_dense`（可取末值、前20iter均值、整体均值）
  - `loss_abs_pose_trans / loss_abs_pose_rot`
  - `abs_pose_rho_norm / abs_pose_theta_norm`
  - `final_true_pose_delta_aggregate`
- 读取 replay summary；
- 产出 `summary.json + compare_table.csv + compare_table.md`。

注意：P0/P1 共用这一份脚本，不额外再生造第二套 compare 汇总器。

## 4.5 推荐执行顺序

### P0-A：先生成统一 replay baseline

只跑一次：

```bash
cd /home/bzhang512/CV_Project
export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO

$PY $PROJECT_ROOT/scripts/replay_internal_eval.py \
  --internal-cache-root $INTERNAL_CACHE_ROOT \
  --stage-tag after_opt \
  --ply-path $BASE_PLY \
  --label before_refine_baseline \
  --save-dir /home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_absprior_calibration/replay_before
```

默认不要开 `--save-render-rgb / --save-render-depth / --save-render-depth-npy`，先省盘。

### P0-B：做一轮 split abs prior 短跑

每组只跑短 StageA（例如 80 iter 或 120 iter），并打开梯度日志：

```bash
$PY $PROJECT_ROOT/scripts/run_pseudo_refinement_v2.py \
  --ply_path $BASE_PLY \
  --pseudo_cache $PSEUDO_CACHE \
  --output_dir <RUN_DIR> \
  --target_side fused \
  --confidence_mask_source brpo \
  --signal_pipeline legacy \
  --stage_mode stageA \
  --stageA_iters <SHORT_ITERS> \
  --stageA_rgb_mask_mode train_mask \
  --stageA_depth_mask_mode train_mask \
  --stageA_target_depth_mode target_depth_for_refine_v2 \
  --stageA_depth_loss_mode source_aware \
  --stageA_lambda_depth_seed 1.0 \
  --stageA_lambda_depth_dense 0.35 \
  --stageA_lambda_depth_fallback 0.0 \
  --stageA_abs_pose_robust charbonnier \
  --stageA_abs_pose_scale_source render_depth_trainmask_median \
  --stageA_lambda_abs_t <LAMBDA_T> \
  --stageA_lambda_abs_r <LAMBDA_R> \
  --stageA_log_grad_contrib \
  --stageA_log_grad_interval 1 \
  --num_pseudo_views 4
```

本阶段不再扫旧的 `lambda_abs_pose` 单标量，只看 split 版。

### P0-C：对每组短跑做 replay

```bash
$PY $PROJECT_ROOT/scripts/replay_internal_eval.py \
  --internal-cache-root $INTERNAL_CACHE_ROOT \
  --stage-tag after_opt \
  --ply-path <RUN_DIR>/refined_gaussians.ply \
  --label <RUN_LABEL> \
  --save-dir <RUN_DIR>/replay_eval
```

实跑修正（2026-04-15 P0 live run）：
- 当前纯 `stageA` 只更新 pseudo camera / exposure，不更新 Gaussian；
- 已用 `sha256sum` 验证：`<RUN_DIR>/refined_gaussians.ply` 与输入 `BASE_PLY` 完全同 hash；
- 因此这里的 replay 只用于确认“没有意外写坏 PLY”，不能作为 abs prior 好坏的主比较指标。

### P0-D：统一汇总

由 `scripts/summarize_stageA_compare.py` 输出总表，按以下规则选一个固定背景配置：

1. `loss_abs_pose_trans / loss_abs_pose_rot` 都非零且稳定；
2. `grad_contrib` 里 abs prior 对 rot/trans 的梯度不应明显压倒 `depth_total` 一个数量级以上；
3. `final_true_pose_delta_aggregate` 相比 no-abs 更小；
4. 注意：`run_pseudo_refinement_v2.py` 的纯 `stageA` 当前不会更新 Gaussian，`refined_gaussians.ply` 与输入 `ply` 的 hash 会保持一致，因此这里的 replay 只能作为 identity sanity check，不能作为区分 abs prior 的主决策量。

## 4.6 P0 的验收标准

通过的标准不是“loss 看着好看”，而是同时满足：

- 能找到一个 split abs prior 组合，后续可作为 compare 固定背景；
- 该组合下，`abs prior` 明显进入有效区间，但不是纯压制项；
- 能给出一句明确口径：
  - “后续 legacy/v2 compare 固定用哪一组 `lambda_abs_t / lambda_abs_r`”。

---

## 5. Phase P1：legacy vs v2 强对照

## 5.1 目标

回答的不是“v2 能不能跑”，而是：

注意：根据 2026-04-15 的 P0 实跑，当前纯 `StageA` 不会改变 Gaussian PLY，因此如果 P1 仍然只停在 `stage_mode=stageA`，那么 replay delta 也会天然接近 0。也就是说：
- `StageA-only P1` 可以回答 signal / depth / drift / abs-prior-compatibility；
- 但如果要把 replay 作为主要验收指标，P1 compare 必须升级到 `StageA.5` 或其他真实更新 Gaussian 的 stage。

在同一组 pseudo frames、同一组 Stage A iter、同一组 abs prior、同一组 replay/eval 口径下：

- `brpo_v2` 相对 `legacy` 是否更稳；
- 它是不是只是更窄、更干净，但没有带来更好的训练可消费性；
- `v2` 的价值主要来自 RGB mask 语义，还是完整 depth supervision v2 也值得保留。

## 5.2 本阶段的比较臂

默认做三臂；如果预算紧，至少保留前两臂中的 legacy 与 full-v2：

### Arm-L：legacy baseline

```text
signal_pipeline=legacy
stageA_rgb_mask_mode=train_mask
stageA_depth_mask_mode=train_mask
stageA_target_depth_mode=target_depth_for_refine_v2
stageA_depth_loss_mode=source_aware
```

### Arm-V1：只换 RGB mask（推荐保留）

```text
signal_pipeline=brpo_v2
stageA_rgb_mask_mode=brpo_v2_raw
stageA_depth_mask_mode=train_mask
stageA_target_depth_mode=target_depth_for_refine_v2
stageA_depth_loss_mode=source_aware
signal_v2_root=$SIGNAL_V2_ROOT
```

这个臂的意义：把“RGB 语义重排”与“depth supervision v2 过窄”拆开看。

### Arm-V2：完整 v2

```text
signal_pipeline=brpo_v2
stageA_rgb_mask_mode=brpo_v2_raw
stageA_depth_mask_mode=brpo_v2_depth
stageA_target_depth_mode=brpo_v2
stageA_depth_loss_mode=source_aware
signal_v2_root=$SIGNAL_V2_ROOT
```

## 5.3 本阶段前置准备

先在 canonical E1 root 上生成 8-frame `signal_v2`：

```bash
cd /home/bzhang512/CV_Project
export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO

$PY $PROJECT_ROOT/scripts/build_brpo_v2_signal_from_internal_cache.py \
  --internal-cache-root $INTERNAL_CACHE_ROOT \
  --prepare-root $PREPARE_ROOT \
  --stage-tag after_opt \
  --output-root $SIGNAL_V2_ROOT
```

注意：
- 不重新复制 `PREPARE_ROOT`；
- 不改写 `pseudo_cache/` 旧文件；
- 只在 `signal_v2/` 下增量生成新产物。

## 5.4 本阶段建议新增的最小工程件

继续复用 `scripts/summarize_stageA_compare.py`，不再额外写第二套 compare 汇总器。

如果后续发现手工命令容易出错，再加一个轻量 orchestrator：

```text
scripts/run_stageA_short_compare.py
```

但第一版不强制。原则是：先回答问题，再决定是否需要 orchestration 脚本。

## 5.5 执行方式

三臂共享完全相同的：
- `PSEUDO_CACHE`
- `selected_frame_ids`
- `stageA_iters`
- `num_pseudo_views`
- `seed`
- `lambda_abs_t / lambda_abs_r`（来自 P0）
- replay/eval 口径

每个 arm 都执行：
1. StageA short run
2. replay_internal_eval
3. summarize_stageA_compare 汇总入总表

## 5.6 核心比较指标

必须一起看，不允许只看 loss：

1. supervision side
   - `mean_confidence_nonzero_ratio`
   - `mean_target_depth_verified_ratio`
   - `mean_target_depth_render_fallback_ratio`
   - `loss_depth_seed`
   - `loss_depth_dense`

2. pose/drift side
   - `loss_abs_pose_trans`
   - `loss_abs_pose_rot`
   - `abs_pose_rho_norm`
   - `abs_pose_theta_norm`
   - `final_true_pose_delta_aggregate`

3. replay side
   - `avg_psnr / avg_ssim / avg_lpips`
   - replay 前后 delta

## 5.7 P1 的决策规则

- 如果 `Arm-V1 > Arm-L`，但 `Arm-V2 <= Arm-V1`：
  - 说明 `RGB mask v2` 有价值，但 `depth supervision v2` 当前过窄，后续应优先考虑轻量 expand/reweight，而不是直接把 full-v2 当默认。

- 如果 `Arm-V2 >= Arm-V1 >= Arm-L`：
  - 说明完整 `v2` 值得继续保留并作为后续 gating 的默认 signal branch。

- 如果 `Arm-L` 仍明显更稳：
  - 说明 `v2` 目前仍停留在“干净但太弱”的状态；
  - 此时不要继续盲目扩大 v2，而应把重点转回 `local Gaussian gating` 与 map-side 作用域收缩。

## 5.8 P1 的验收标准

P1 结束时必须能给出一句明确结论：

- “后续 local gating 默认挂在哪个 signal branch 上：legacy / RGB-only / full-v2”。

如果这个问题还答不出来，说明 compare 设计不够强，对照还要补，而不是直接往 StageB 走。

---

## 6. Phase P2：local Gaussian gating / local GS gating

## 6.1 目标

把当前的结构问题从：

```text
weak local pseudo supervision -> global Gaussian perturbation
```

改成：

```text
weak local pseudo supervision -> local visible Gaussian subset update
```

这一步的目标不是放大 signal，而是让 supervision scope 和 optimization scope 重新匹配。

## 6.2 第一版固定边界

第一版只做：

1. pseudo branch 的 local gradient gating；
2. 先做 `StageA.5`，再接 `StageB`；
3. 先做 `xyz`，通过后再加 `xyz+opacity`；
4. 先做 hard gating，再做 soft gating；
5. real branch 保持全局纠偏，不被 pseudo gate 连带裁掉。

这一步明确不做：
- full SPGM
- 新 map backend
- 复杂 importance score
- per-pixel 到 Gaussian 的精细反投影打分

## 6.3 代码锚点（真实可改位置）

### 已有锚点

1. `scripts/run_pseudo_refinement_v2.py`
   - StageA.5：当前在 StageA 主循环中，若 `gaussian_optimizer is not None`，会直接参与 `total_loss.backward()` 后的 `gaussian_optimizer.step()`；
   - StageB：当前 `total_loss = lambda_real * real + lambda_pseudo * pseudo`，然后单次 backward；
   - 这里就是 local gating 要接入的主位置。

2. `scripts/run_pseudo_refinement.py`
   - 已有 pseudo / real backward 分开处理的工程先例；
   - 这是第一版 local gating 最该借的 backward 结构。

3. `render(...)` 返回包
   - v1 已直接使用 `visibility_filter / viewspace_points / radii`；
   - 因此第一版 gate 不需要自己重写投影器，先直接用 `visibility_filter` 做 visible union。

4. 已加载到 view dict 的 signal 统计
   - `rgb_confidence_nonzero_ratio`
   - `depth_confidence_nonzero_ratio`
   - `target_depth_verified_ratio`
   - `target_depth_render_fallback_ratio`
   - 若切到 v2，还能直接读到 `signal_v2` 的 mask/depth summary

### 建议新增目录

```text
pseudo_branch/local_gating/
  __init__.py
  visibility_union.py
  signal_gate.py
  grad_mask.py
  gating_schema.py
  gating_io.py
```

### 必改文件

```text
scripts/run_pseudo_refinement_v2.py
```

## 6.4 第一版 gate 定义

### visibility gate

- 对本 iteration 中 sampled pseudo views 的 `visibility_filter` 做 union；
- 但只对通过 signal gate 的 pseudo views 纳入 union；
- 第一版固定使用 `union`，不用 `intersection`。

### signal gate

第一版用 rule-based hard gate，不做复杂 learned score。

每个 sampled pseudo view 至少检查：
- `target_depth_verified_ratio >= min_verified_ratio`
- `rgb_confidence_nonzero_ratio >= min_rgb_mask_ratio`
- `target_depth_render_fallback_ratio <= max_fallback_ratio`
- `min_correction` 第一版默认可先关闭

建议 CLI：

```text
--pseudo_local_gating {off,hard_visible_union_signal,soft_visible_union_signal}
--pseudo_local_gating_params {xyz,xyz_opacity}
--pseudo_local_gating_min_verified_ratio
--pseudo_local_gating_min_rgb_mask_ratio
--pseudo_local_gating_max_fallback_ratio
--pseudo_local_gating_min_correction
--pseudo_local_gating_soft_power
--pseudo_local_gating_log_interval
```

## 6.5 backward 结构怎么改

### StageA.5（先实现）

当前：

```text
pseudo-only loss -> total_loss.backward() -> pose opt step -> gaussian opt step
```

改成：

```text
pseudo-only loss
-> backward()
-> 对 gaussian grad 做 local gate mask
-> pose opt step
-> gaussian opt step
```

说明：
- StageA.5 没 real branch，所以第一版不需要 split real/pseudo backward；
- 这里只需要在 backward 之后、gaussian optimizer step 之前，对 `xyz`（以及后续可选的 `opacity`）grad 做 mask。

### StageB（第二步接）

当前：

```text
real_loss + pseudo_loss -> 单次 backward -> pseudo opt step + gaussian opt step
```

改成：

```text
1. pseudo_loss.backward(retain_graph=True)
2. 对 gaussian grad 做 pseudo-side local gate mask
3. real_loss.backward()
4. step pseudo optimizer
5. step gaussian optimizer
```

关键点：
- 只裁 pseudo-side 对 Gaussian 的更新；
- 不裁 real branch 的全局纠偏梯度；
- 这一步应直接借 v1 的 split backward 结构，不要重新设计一套 optimizer 框架。

## 6.6 日志与验收产物

必须把 gating 信息写进 history，否则后面会无法判断到底是 gate 生效了，还是 sampled view 本身就没信号。

建议新增到 `stageA_history.json / stageB_history.json`：

- `pseudo_local_gating_mode`
- `pseudo_local_gating_params`
- `sampled_pseudo_sample_ids`
- `accepted_pseudo_sample_ids`
- `rejected_pseudo_sample_ids`
- `rejected_reasons`
- `visible_union_ratio`
- `grad_keep_ratio_xyz`
- `grad_keep_ratio_opacity`

## 6.7 P2 的最小验证梯度

固定三步，不要直接长跑：

### P2-A：StageA.5 2-frame / 3-frame smoke

目标：
- CLI 接通；
- history 里有 gating summary；
- `gating=off` 与 `gating=hard_visible_union_signal` 对 `grad_keep_ratio_xyz` 有真实差异。

### P2-B：8-frame StageA.5 short compare

比较：
- ungated xyz
- gated xyz

只回答：
- replay 是否更稳；
- `grad_norm_xyz` 是否明显收缩；
- `loss_depth` 是否没有被完全压没。

### P2-C：StageB short smoke（仅在 P2-B 通过后）

先接：
- `xyz-only`
- `hard gate`
- `real branch on`

只有这一步通过后，才考虑 `xyz+opacity` 或 soft gating。

## 6.8 P2 的验收标准

通过的标准不是 pseudo loss 降更多，而是：

- gating summary 真实记录下来；
- pseudo-side Gaussian 更新范围被收窄；
- replay/held-out 的退化至少比 ungated 版本更轻；
- 如果 `xyz-only hard gating` 都不能改善 replay，就不要急着上 `xyz+opacity` 或 soft gating。

---

## 7. 不允许再发生的几件事

1. 不再用 3-frame smoke 去替代正式 compare 结论。
2. 不再在 `P0/P1` 没完成前就把当前窄监督直接扩展到 A.5 / StageB 长跑。
3. 不再把 `v2` 的 wiring 成功表述成“已经证明 v2 更好”。
4. 不再把 local gating 和 full SPGM 一起塞进第一版实现。
5. 不再复制大 prepare root 做 compare；当前磁盘不允许这种低收益复制。

---

## 8. 最终一句话执行口径

后续执行默认按下面顺序推进：

1. 先用 legacy branch 在当前 E1 winner 上把 split abs prior 标定到一个“够稳但不过强”的固定背景；
2. 再在同一 pseudo set、同一 StageA 预算、同一 replay 口径下做 `legacy / RGB-only-v2 / full-v2` 的短对照；
3. 最后再把 local Gaussian gating 接到 StageA.5 -> StageB，先 `xyz-only + hard gate + pseudo-only local mask`，确认 replay 更稳后再考虑 `xyz+opacity` 与更重的 SPGM。

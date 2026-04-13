# CHANGELOG.md - Part3 Stage1 过程记录

> 本文件记录每次工作的过程、发现和结果。
> 更新规则：以日期为单位记录；允许为修正文档一致性而整理旧条目，但不要把“现状”写进本文件。

---

## 2026-04-12

### Mask problem route：M1 fused-first verification + compatibility layer

根据 `CURRENT_MASK_PROBLEM.md` 与 `SOLVE_MASK_PROBLEM.md` 的判断，开始把主线从“继续扩 Stage A”转到“先解决 upstream mask problem”。

这一轮的重点不是 refine，而是 verification / pack / consumer 之间的 upstream 结构修正。

代码修改：
- `pseudo_branch/brpo_confidence_mask.py`
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
- `docs/SOLVE_MASK_PROBLEM.md`

关键实现：
- `verification_mode=branch_first|fused_first` 双模式落地；
- `fused_first` 模式可直接消费 `fusion/samples/<id>/target_rgb_fused.png`；
- verification 输出新增 `seed_support_left/right/both/single`；
- 同时保留 `support_*` 与 `confidence_mask_brpo_*` 兼容层，不污染旧实验与现有 consumer；
- `pack` 已能在 `verification_mode=fused_first` 下把 `seed_support_*` 带入 sample。

真实检查：
- `fused_first` 已在当前 3-frame prototype 上跑通；
- 但与旧 `branch_first` 相比，coverage 改善并不明显；
- 这验证了：**单纯改 verification 顺序还不够，M2 的 propagation 仍然必要。**

### Mask problem route：M2 train mask propagation 第一轮实现

在 M1 之后，开始做 `seed_support → train_confidence_mask` 的第一轮工程化传播。

新增文件：
- `pseudo_branch/brpo_train_mask.py`

关键实现：
- verify 阶段新增 `train_mask_mode=propagate`；
- 新增：
  - `train_confidence_mask_brpo_{left,right,fused}`
  - `train_support_{left,right,both,single}`
- 当前训练真正消费的 mask 通过 alias 保持兼容：
  - `confidence_mask_brpo_*` → 指向 train mask
  - `support_*` → 指向 seed support
- propagation 仍保留在 verify/pack upstream，不塞回 refine。

中途问题与修复：
- 第一版 M2 因重复落盘（seed/train/alias 全部物理保存）触发 `No space left on device`；
- 随后把 alias 改成 symlink，保留兼容的同时避免重复写大文件；
- 清掉失败的中间 verify 目录后重跑，验证通过。

真实结果：
- 机制成立，coverage 从 `~1%` 量级显著抬升；
- 但默认参数明显过宽：
  - frame 10: `~75.1%`
  - frame 50: `~68.0%`
  - frame 120: `~68.9%`
- 说明 propagation 机制是对的，但默认超参不能直接作为最终 training mask 定义。

### Mask problem route：M2.5 propagation 合理区间小范围研究

为了判断当前 propagation 的“合理区间”，没有继续大规模重跑 verify，而是基于现有 `fused_first + seed_support` 结果做了离线小范围 sweep。

目的：
- 不是寻找唯一最优超参；
- 而是先判断：哪些组合明显太稀、哪些组合明显太稠、哪些组合更像训练可用区间。

代表性结果：
- `radius=1, tau_rel_depth=0.01, tau_rgb_l1=0.03` → `avg_fused_nonzero ≈ 8.2%`
- `radius=1, tau_rel_depth=0.02, tau_rgb_l1=0.03` → `≈ 9.7%`
- `radius=2, tau_rel_depth=0.01, tau_rgb_l1=0.05` → `≈ 19.4%`
- `radius=2, tau_rel_depth=0.02, tau_rgb_l1=0.05` → `≈ 25.1%`
- `radius=3, tau_rel_depth=0.03, tau_rgb_l1=0.08` → `≈ 52.3%`
- `radius=5, tau_rel_depth=0.03, tau_rgb_l1=0.08` → `≈ 70.7%`

阶段判断：
- 当前 propagation 的合理研究区间明显更接近 `10% ~ 25%` coverage；
- 因此后续不应继续使用当前过宽默认值，而应优先围绕：
  - `radius ∈ {1,2}`
  - `tau_rel_depth ∈ [0.01, 0.02]`
  - `tau_rgb_l1 ∈ [0.03, 0.05]`
  做进一步收紧。

### Mask problem route：M3 blended `target_depth_for_refine` 落地

在 M2.5 之后，正式进入 M3：把 upstream depth target 从“render-depth fallback 占位符”升级成可审计的 blended target。

新增文件：
- `pseudo_branch/brpo_depth_target.py`

代码修改：
- `pseudo_branch/brpo_reprojection_verify.py`
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`

关键实现：
- verification 侧新增 pseudo-view sparse verified depth 输出：
  - `projected_depth_left.npy`
  - `projected_depth_right.npy`
  - `projected_depth_valid_left.npy`
  - `projected_depth_valid_right.npy`
- pack 阶段正式构造：
  - `target_depth_for_refine.npy`
  - `target_depth_for_refine_source_map.npy`
  - `target_depth_for_refine_meta.json`
  - `verified_depth_mask.npy`
- `target_depth_for_refine` 当前定义为：
  - `both` → 双边 verified depth average
  - `left/right` → 单边 verified depth
  - unsupported → `render_depth` fallback

真实 smoke：
- 新开 prototype 目录：
  - `.../internal_prepare/re10k1__internal_afteropt__brpo_proto_v4_stage3/`
- 在 3-frame（10/50/120）上重跑 `verify → pack` 成功；
- sample 层已真实出现：
  - `projected_depth_*`
  - `target_depth_for_refine.npy`
  - `target_depth_for_refine_source_map.npy`
  - `target_depth_for_refine_meta.json`

代表性结果：
- frame 10 verified depth ratio：`~1.63%`
- frame 50 verified depth ratio：`~1.51%`
- frame 120 verified depth ratio：`~1.55%`
- 其余区域为 `render_depth` fallback

阶段判断：
- M3 已完成 upstream depth target 的真实落地；
- 当前 `target_depth_for_refine.npy` 已不再只是 schema 预留位；
- 但当前 verified depth 覆盖偏保守，更像 sparse correction source，而不是大覆盖 supervision 主体。

提交：
- `c7965b4` — `part3: implement Stage M3 blended target_depth_for_refine pipeline`

### Mask problem route：M4 Stage A consumer 显式接入新 target

在 M3 之后，继续做 M4：让 `run_pseudo_refinement_v2.py` 明确消费新的 train mask 与 blended depth target，而不是继续依赖含糊 fallback。

代码修改：
- `scripts/run_pseudo_refinement_v2.py`

关键实现：
- Stage A 现在显式支持：
  - `stageA_mask_mode=train_mask|seed_support_only|legacy|auto`
  - `stageA_target_depth_mode=blended_depth|render_depth_only|target_depth_for_refine|target_depth|render_depth|auto`
- history 现在会显式记录：
  - 实际 confidence source / path
  - 实际 depth source / path
  - effective mask/depth mode
  - mask coverage
  - verified depth ratio / render fallback ratio
- 通过 sample 内显式文件读取 `train_confidence_mask_brpo_*` 与 `target_depth_for_refine.npy`，不再让 consumer 靠隐式默认值漂移。

真实 smoke：
- 目录：
  - `.../2026-04-12_m4_stageA_smoke/blended_depth/`
  - `.../2026-04-12_m4_stageA_smoke/render_depth_only/`
- `train_mask + blended_depth`：
  - mean mask coverage `~19.4%`
  - mean verified depth ratio `~1.56%`
  - `loss_depth ≈ 0.00478`（非零）
- `train_mask + render_depth_only`：
  - `loss_depth ≈ 0`
- 额外 debug quick smoke：
  - `seed_support_only + blended_depth` 2 iter 可跑，mean mask coverage `~1.56%`

阶段判断：
- M4 consumer 已真实接通；
- `blended_depth` 与 `render_depth_only` 在 Stage A 中已能被区分；
- 这说明 M3→M4 链路不是“名义接通”，而是 depth signal 已实际进入 loss。

提交：
- `63abe22` — `part3: wire Stage A to explicit M4 mask and depth modes`

### Mask problem route：M4.5 Stage A 长一点对照（blended vs render-only）

在 M4 smoke 之后，继续做一轮更长一点的 Stage A eval，目的不是追最终指标，而是回答：
- depth signal 是否真的被消费；
- 当前 depth signal 在更长一点的 Stage A 中是否会动起来；
- `blended_depth` 与 `render_depth_only` 的差异是否只停留在 smoke 级别。

实验目录：
- `.../2026-04-12_m45_stageA_eval/blended_depth_long/`
- `.../2026-04-12_m45_stageA_eval/render_depth_only_long/`
- `.../2026-04-12_m45_stageA_eval/analysis/summary.json`
- `.../2026-04-12_m45_stageA_eval/analysis/compare.txt`

实验口径：
- `target_side=fused`
- `confidence_mask_source=brpo`
- `stageA_mask_mode=train_mask`
- `num_pseudo_views=3`
- `stageA_iters=300`
- 唯一变量：
  - `stageA_target_depth_mode=blended_depth`
  - vs `stageA_target_depth_mode=render_depth_only`

结果：
- 两组 `loss_rgb` 基本一致：
  - `0.00915 -> 0.00734`
- `blended_depth_long`：
  - `loss_depth ≈ 0.004783`（非零）
  - `loss_total first -> last: 0.00784 -> 0.02269`
- `render_depth_only_long`：
  - `loss_depth ≈ 4.4e-07 ≈ 0`
  - `loss_total first -> last: 0.00641 -> 0.02207`
- 当前 `pose_reg` 会随着迭代持续增长，两组量级接近；
- 当前 `blended_depth` 虽然已非零，但在 300 iter 中基本不下降。

阶段判断：
- `blended_depth` 与 `render_depth_only` 的差异已经真实存在；
- 但当前差异主要体现在“depth loss 是否非零”，而不是“depth 是否明显驱动了更好的优化轨迹”；
- 更准确的结论是：

**depth signal 已接入，但当前 Stage A 对它的利用仍偏弱。**

因此下一步应进入 **M4.6 / depth-flatness diagnosis**，而不是直接跳到 Stage B。


### Mask problem route：M5-0 depth signal diagnosis

在 M4.5 之后，没有直接继续推进 Stage B，而是先做一轮更结构化的 diagnosis，确认当前 depth signal 到底卡在哪里。

新增文件：
- `scripts/analyze_m5_depth_signal.py`

实验目录：
- `.../2026-04-12_m50_m51_eval/analysis/m50_depth_signal.json`

关键结果：
- `train_mask coverage ≈ 19.41%`
- `verified depth coverage ≈ 1.56%`
- `verified_within_train_mask_ratio ≈ 8.05%`
- `render_fallback_within_train_mask_ratio ≈ 91.95%`

解释：
- 当前 depth loss 不是没接上；
- 但在训练真正消费的 mask 区域里，绝大多数像素仍然只是 fallback depth；
- 这说明旧的 M3 target 在 train-mask 内部的“新几何信息占比”确实太低。

阶段判断：
- M5-0 证明旧问题真实存在；
- 下一步值得先尝试 densify correction field，而不是继续堆 M4.5 式对照。

### Mask problem route：M5-1 densify depth correction field

根据 M5 规划，开始新增一层受控的 local depth correction densify，而不是去放宽 verify 阈值。

新增文件：
- `pseudo_branch/brpo_depth_densify.py`
- `scripts/materialize_m5_depth_targets.py`

代码修改：
- `pseudo_branch/brpo_depth_target.py`

关键实现：
- 在 verified seed 上构造 `log-depth correction`，而不是传播 absolute depth；
- 先实现首版 `patch-wise constant correction`；
- 输出：
  - `target_depth_for_refine_v2.npy`
  - `target_depth_dense_source_map.npy`
  - `depth_correction_seed.npy`
  - `depth_correction_dense.npy`
  - `depth_seed_valid_mask.npy`
  - `depth_dense_valid_mask.npy`
  - `depth_densify_meta.json`

默认参数第一次运行结果：
- 只把 densified 区域提升到 `~2.08%`
- 总 non-fallback 只有 `~3.64%`
- 结论：太保守，不足以支撑下游对照

随后做了小范围 sweep，目标不是追求极大 coverage，而是找一个“明显增强但仍受控”的配置。

当前选中的一组参数：
- `patch_size = 11`
- `stride = 5`
- `min_seed_count = 4`
- `max_seed_delta_std = 0.08`

在这组参数下：
- `seed_ratio ≈ 1.56%`
- `densified_ratio ≈ 14.21%`
- `total_nonfallback_ratio ≈ 15.77%`
- `render_fallback_ratio ≈ 84.23%`

更关键的是，在 train-mask 内部：
- `nonfallback_within_train_mask_ratio ≈ 81.22%`
- `seed_within_train_mask_ratio ≈ 8.05%`
- `dense_within_train_mask_ratio ≈ 73.17%`

阶段判断：
- M5-1 upstream 是可行的；
- 这一步已经把真正有新几何信息的区域从 seed 级抬到了 train-mask 的大部分区域；
- 因此后续值得测试 source-aware depth loss，而不是继续纠结 upstream coverage 是否还太低。

提交：
- `8a9c224` — `part3: add M5 depth signal analysis and densify targets`

### Mask problem route：M5-2 source-aware depth loss 接入 Stage A

在 M5-1 之后，继续把 Stage A 接到新的 densified target，并把 depth loss 按 source 拆开。

代码修改：
- `pseudo_branch/pseudo_loss_v2.py`
- `scripts/run_pseudo_refinement_v2.py`

关键实现：
- `run_pseudo_refinement_v2.py` 新增：
  - `stageA_target_depth_mode=blended_depth_m5`
  - `stageA_depth_loss_mode=legacy|source_aware`
- `pseudo_loss_v2.py` 新增：
  - `loss_depth_seed`
  - `loss_depth_dense`
  - `loss_depth_fallback`
- 当前 source-aware 默认权重：
  - `lambda_depth_seed = 1.0`
  - `lambda_depth_dense = 0.35`
  - `lambda_depth_fallback = 0.0`

实验目录：
- `.../2026-04-12_m52_stageA_loss_eval/`

对照结果：
1. `M5 + legacy depth loss`
   - `loss_depth ≈ 0.02770`
   - 300 iter 中基本完全不动
2. `M5 + source-aware depth loss`
   - `loss_depth ≈ 0.06867`
   - 其中：
     - `loss_depth_seed ≈ 0.05796`
     - `loss_depth_dense ≈ 0.03062`
     - `loss_depth_fallback ≈ 0`
   - 300 iter 中同样基本完全不动

阶段判断：
- source-aware depth loss 已经真实接通；
- fallback 也已基本从 depth loss 主体中剔除；
- 但 depth 仍然不动，说明问题已经不再只是 coverage 或 fallback 稀释。

提交：
- `d700194` — `part3: add M5 source-aware stageA depth loss`

### Stage A diagnosis：确认 `theta/rho` 在 renderer 中 forward 丢弃、backward 仍回梯度

在 M5-2 之后，继续追查为什么 loss 仍完全不动，并审核 refine 过程本身是否存在更底层的问题。

新增文件：
- `scripts/diagnose_stageA_gradients.py`

实验目录：
- `.../2026-04-12_m53_stageA_diagnosis/m53_gradients.json`

代码审计先确认：
- `cam_rot_delta / cam_trans_delta` 确实传进了 `gaussian_renderer`
- optimizer 也确实在更新它们

但真正关键的 diagnosis 结果是：

1. **forward sensitivity probe 极其异常**
   - 人工把 `cam_rot_delta / cam_trans_delta` 设到 `0.01 / 0.01`、`0.05 / 0.05`、`0.1 / 0.1`、`0.2 / 0.2`
   - 重新 render RGB / depth
   - 与 base render 比较，结果始终：
     - `rgb_mean_abs_change = 0.0`
     - `depth_mean_abs_change = 0.0`

2. **backward 却返回了非零 pose 梯度**
   - `mean_grad_rgb_rot ≈ 0.265`
   - `mean_grad_rgb_trans ≈ 0.0786`
   - `mean_grad_depth_legacy_rot ≈ 0.504`
   - `mean_grad_depth_legacy_trans ≈ 0.197`
   - `mean_grad_depth_src_rot ≈ 1.732`
   - `mean_grad_depth_src_trans ≈ 0.550`

3. **pose regularization 在初始化为 0 时梯度也是 0**
   - 因此最开始没有任何“拉回 pose”的约束力

阶段判断：
- 当前最可疑的问题已经被代码链路审计确认：

**Stage A 的 `theta/rho` 在 Python 层被传入，但在 C++/CUDA forward 中被完全丢弃；与此同时 backward 仍然单独构造并返回 `dL_dtau`。**

这也解释了为什么：
- loss 基本不下降；
- pose delta 却会持续增大；
- 当前 Stage A 的 pose refine 结果不应被直接信任。

进一步的代码证据是：
- `diff_gaussian_rasterization/__init__.py::_RasterizeGaussians.forward()` 的 `args` 不包含 `theta/rho`；
- `rasterize_points.h/.cu::RasterizeGaussiansCUDA` 的 forward 函数签名也没有 `theta/rho`；
- 但 `_RasterizeGaussians.backward()` 会从 `_C.rasterize_gaussians_backward(...)` 收到 `grad_tau`，再拆成 `grad_theta / grad_rho`。

因此当前明确不建议直接进入 Stage B。

补充修正判断：
- 经过进一步代码链路审计，这里更准确的结论不是“CUDA 单纯写坏了一个本该 forward 直接吃 `theta/rho` 的接口”；
- S3PO 原始代码里，`cam_rot_delta / cam_trans_delta` 本来就是一步一清的增量 residual，`slam_frontend.py` / `slam_backend.py` 会在每次 `optimizer.step()` 后立刻调用 `update_pose(viewpoint)`，把 `tau` 折回 `R/T` 并清零；
- 当前 Part3 Stage A 的真正问题是：**我们沿用了 residual 参数和 backward 梯度，但没有沿用 `update_pose()` 这一步，因此把 S3PO 的增量式 pose 优化闭环用断了。**


### Stage A 修复：补回 S3PO residual pose 闭环 + 最小验证

在继续追根因之后，进一步梳理了 S3PO 原始代码对 `cam_rot_delta / cam_trans_delta` 的使用方式。

关键发现：
- 原始 S3PO 不是把 `theta/rho` 当作持续直接作用于 forward render 的状态量；
- 它把它们当作**一步一清的 pose residual**；
- `slam_frontend.py` / `slam_backend.py` 在每次 `optimizer.step()` 后都会立刻调用 `update_pose(viewpoint)`；
- `pose_utils.py::update_pose()` 会把 `tau=[rho, theta]` 折回 `R/T` 并清零。

所以当前 Part3 Stage A 的真正问题不是“CUDA 单纯坏了”，而是：

**我们沿用了 residual 参数和 backward 梯度，但没有沿用 `update_pose()` 这一步，因此把 S3PO 的增量式 pose 优化闭环用断了。**

代码修改：
- `pseudo_branch/pseudo_camera_state.py`
- `scripts/run_pseudo_refinement_v2.py`

关键实现：
- 新增 `apply_pose_residual_()`：
  - 用当前 `tau=[cam_trans_delta, cam_rot_delta]` 计算新的 `w2c`；
  - 折回 `vp.R / vp.T`；
  - 刷新 `world_view_transform / full_proj_transform / camera_center`；
  - 清零 residual。
- `run_pseudo_refinement_v2.py` 现在在每次 `optimizer.step()` 后，默认会对本轮参与优化的 pseudo views 调 `apply_pose_residual_()`；
- 同时新增开关：
  - `--stageA_apply_pose_update`（默认开启）
  - `--stageA_no_apply_pose_update`（用于保留旧坏链路做对照）

最小验证 1：手动 residual -> fold 到 R/T -> render
- 修复后，render 会立刻随 pose 变化：
  - `(0.01, 0.01)`：`rgb_mean ≈ 0.0679`, `depth_mean ≈ 0.1648`
  - `(0.05, 0.05)`：`rgb_mean ≈ 0.1604`, `depth_mean ≈ 0.4364`
  - `(0.10, 0.10)`：`rgb_mean ≈ 0.2412`, `depth_mean ≈ 0.8877`

最小验证 2：M5 source-aware 80 iter smoke 对照
- `no_apply_pose_update`（旧坏链路）：
  - `loss_depth: 0.06867 -> 0.06867`（完全平）
  - `pose_updates_total = 0`
  - 最终 residual 堆积在 `rot_norm ~ 0.27~0.41`、`trans_norm ~ 0.13`
- `apply_pose_update`（修复后闭环）：
  - `loss_total: 0.02701 -> 0.02570`
  - `loss_depth: 0.06867 -> 0.06826`
  - `loss_depth_seed: 0.05796 -> 0.05760`
  - `loss_depth_dense: 0.03062 -> 0.03046`
  - `pose_updates_total = 240`
  - 最终 residual 全部回到 `0.0`

阶段判断：
- 这说明当前 Stage A 已经不再是之前那种“forward 不动、backward 乱推”的假优化；
- 但也要老实说：修复闭环之后，depth 只是开始轻微下降，还没有出现非常强的下降趋势；
- 所以下一步不应直接进入 Stage B，而应先做修复后的中等规模验证，再决定是否需要继续加强 loss / 训练长度 / 更深层 renderer 改动。


### Stage A 修复后 300-iter 参数/规模验证

在补回 `apply_pose_residual_()` 之后，没有直接进入下一阶段，而是先做一轮修复后的 300-iter 参数/规模验证，确认当前问题更像训练规模、权重还是结构。

实验目录：
- `.../2026-04-12_m55_pose_fix_scale_eval/default_300`
- `.../2026-04-12_m55_pose_fix_scale_eval/depth_heavy_300`
- `.../2026-04-12_m55_pose_fix_scale_eval/no_exposure_300`
- 汇总：`.../analysis/m55_summary.json`

实验口径：
1. `default_300`
   - `beta_rgb=0.7`
   - `lambda_depth_seed=1.0`
   - `lambda_depth_dense=0.35`
   - `lr_exp=0.01`
2. `depth_heavy_300`
   - `beta_rgb=0.3`
   - `lambda_depth_seed=1.0`
   - `lambda_depth_dense=1.0`
   - `lr_exp=0.01`
3. `no_exposure_300`
   - `beta_rgb=0.7`
   - `lambda_depth_seed=1.0`
   - `lambda_depth_dense=0.35`
   - `lr_exp=0.0`

结果：
- `default_300`
  - `loss_total: 0.02701 -> 0.02577`
  - `loss_depth: 0.06867 -> 0.06816`
  - `loss_depth_seed: 0.05796 -> 0.05750`
  - `loss_depth_dense: 0.03062 -> 0.03047`
- `depth_heavy_300`
  - `loss_total: 0.06475 -> 0.06364`
  - `loss_depth: 0.08857 -> 0.08747`
  - `loss_depth_seed: 0.05796 -> 0.05731`
  - `loss_depth_dense: 0.03062 -> 0.03016`
- `no_exposure_300`
  - `loss_total: 0.02701 -> 0.02696`
  - `loss_rgb: 0.00915 -> 0.00928`
  - `loss_depth: 0.06867 -> 0.06822`

附加分析：真实 pose 累计变化（init -> final）
- `default_300`
  - mean `trans_delta_l2 ≈ 0.00138`
  - mean `rot_delta_rad ≈ 0.00498`
- `depth_heavy_300`
  - mean `trans_delta_l2 ≈ 0.00390`
  - mean `rot_delta_rad ≈ 0.00453`
- `no_exposure_300`
  - mean `trans_delta_l2 ≈ 0.00141`
  - mean `rot_delta_rad ≈ 0.00463`

阶段判断：
- 修复后，depth 已经不是完全平线；
- 但三组设置都只表现出“弱下降”，没有出现质变；
- `depth_heavy_300` 的确更强一点，但提升仍有限；
- 冻结 exposure 会让 RGB 更差，但不会显著改善 depth；
- 这说明问题已经从“闭环断了”进一步收敛成：

**当前更像是权重/尺度与结构设计问题，而不再是链路完全断开。**

新的结构发现：
- 因为现在每步都会 `apply_pose_residual_()` 并清零 residual，当前 `pose_reg_loss = ||cam_rot_delta|| + ||cam_trans_delta||` 会在训练中几乎始终为 0；
- 这意味着 `stageA_lambda_pose` 当前已经无法约束“累计位姿偏移”；
- 因此下一步需要重点判断：是否要补一个基于 `R/T` 相对初值的 **absolute pose prior**，而不是继续只对 residual 本身做正则。

### 文档恢复后的同步补写

由于误删并恢复了 `docs/`，本轮重新按写作规范同步补回：
- `docs/STATUS.md`
- `docs/DESIGN.md`
- `docs/CHANGELOG.md`

补回原则：
- `STATUS.md` 只写当前状态，不追加过程细节；
- `DESIGN.md` 只写当前设计判断与默认实验方案；
- `CHANGELOG.md` 记录 M1 / M2 / M2.5 / M3 / M4 / M4.5 的真实过程与结果；
- 避免把“现状”和“历史过程”混写。


### Mask problem route：M5-4 absolute pose prior 接入与 smoke 对照

目标：把 `absolute_pose_prior.md` 从设计文档落地为可执行代码，并验证它在当前 Stage A 闭环下是否有效。

代码修改：
- `third_party/S3PO-GS/utils/pose_utils.py`
- `pseudo_branch/pseudo_camera_state.py`
- `pseudo_branch/pseudo_loss_v2.py`
- `pseudo_branch/pseudo_refine_scheduler.py`
- `scripts/run_pseudo_refinement_v2.py`

关键实现：
- 新增 `SO3_log / SE3_log`；
- `make_viewpoint_trainable()` 存 `R0/T0`；
- 新增 `absolute_pose_prior_loss`（基于当前 `w2c` 相对初始 `w2c0` 的 `SE3_log`）；
- Stage A 增加 `--stageA_lambda_abs_pose`，并写入 history 与导出状态；
- `export_view_state` 新增 `pose_w2c_initial / abs_pose_rho / abs_pose_theta / abs_pose_norm`。

最小验证：
- `py_compile` 全通过；
- CLI `--help` 可见 `--stageA_lambda_abs_pose`；
- `SE3_exp -> SE3_log` roundtrip 误差约 `3.95e-09`。

60-iter smoke（同一 pseudo_cache，seed=0）：
- `default_noabs` vs `default_abs(0.1)`：几乎重合；
- `depth_heavy_noabs` vs `depth_heavy_abs(0.1)`：几乎重合；
- `default_abs(100)`：
  - `abs_pose_norm mean 0.001535 -> 0.000338`（drift 明显收紧）
  - 但 `loss_depth` 从 `0.068364` 变为 `0.068543`（略变差）。

阶段判断：
- absolute prior“接入成功”≠“参数可用”；
- 当前结果更像权重尺度问题：`0.1` 太弱，`100` 太强；
- 下一步应做 300-iter 权重扫描与尺度化 prior，而不是直接拿 `100` 当默认值。

### 与 2026-04-11（mask problem 初始阶段）的设计差异

相较昨天，当前设计变化是实质性的：

1. 昨天主矛盾是 **upstream coverage 不足**（先做 verify/seed/train-mask/M3/M5）；
2. 现在 upstream 已可用，主矛盾转为 **Stage A 结构标定**（闭环已修，开始做 abs prior）；
3. 昨天主要在“把 depth 信号接进来”，现在是“如何在 drift 约束与 depth 对齐之间找平衡”；
4. 因此当前不进入 Stage B 的理由也变化了：
   - 不是“链路没接通”；
   - 而是“已接通但参数区间未定”。

## 2026-04-11

### Internal cache / replay：Phase 1 与 Phase 2 跑通

在 Re10k-1 的 80-frame smoke rerun 上，完成了 internal cache 导出与 replay 机制验证。

新增/修改：
- `third_party/S3PO-GS/utils/internal_eval_utils.py`
- `third_party/S3PO-GS/slam.py`（opt-in internal cache 导出）
- `part3_BRPO/scripts/replay_internal_eval.py`

导出结果结构：
- `internal_eval_cache/manifest.json`
- `internal_eval_cache/camera_states.json`
- `internal_eval_cache/before_opt/`
- `internal_eval_cache/after_opt/`

关键确认：
- `color_refinement()` 只优化 gaussians，不更新 pose；
- 因此当前 cache 设计改为：**camera states 共享一份，before/after 各自保存自己的 PLY 与 render cache**。

same-ply consistency check：
- after_opt：internal `31.4016 / 0.95257 / 0.04713` vs replay `31.3581 / 0.95206 / 0.04761`
- before_opt：internal `23.7945 / 0.88950 / 0.12889` vs replay `23.7256 / 0.88842 / 0.13007`

结论：replay 与官方 internal eval 已足够接近，可以用于后续 baseline/refined 公平比较。

### Internal cache / replay：Phase 3 smoke 协议验证

使用同一份 Re10k-1 smoke internal cache，对比：
- baseline sparse PLY：`/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_sparse/Re10k-1_part2_s3po/2026-04-04-00-43-29/point_cloud/final/point_cloud.ply`
- current E refined PLY：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_stage1/re10k-1/sparse/2026-04-08_joint_refine_tertile_freezegeom_lambda0p5/E_joint_realdensify_freezegeom_lambda0p5_tertile/refined.ply`

after_opt replay：
- baseline sparse：`21.3486 / 0.79084 / 0.10690`
- E refined：`19.0380 / 0.69438 / 0.23764`

delta（E - baseline）：
- `PSNR -2.3106`
- `SSIM -0.09646`
- `LPIPS +0.13073`

结论：在 smoke internal protocol 下，current E refined PLY 明显劣于 baseline sparse PLY。
但这仍是 **80-frame smoke**，还不是 full rerun 的最终正式结论。

### BRPO confidence mask：Phase B / Phase C 原型

基于 full internal cache：
`/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache/`

新增文件：
- `part3_BRPO/pseudo_branch/brpo_reprojection_verify.py`
- `part3_BRPO/pseudo_branch/brpo_confidence_mask.py`
- `part3_BRPO/scripts/brpo_verify_single_branch.py`
- `part3_BRPO/scripts/brpo_build_mask_from_internal_cache.py`

工程决策：
- BRPO 新线与旧 EDP 线隔离，不复用 `epipolar_depth.py` / `build_pseudo_cache.py`；
- verification 采用 **方案 B**：按需用 `ref pose + stage PLY` 现渲染 KF depth；
- 当前阶段不改 refine loss，只先跑 verified support / confidence mask。

Phase B（left 单分支）在 `after_opt` 的 3 个样本帧上跑通：
- 平均 `support_ratio_vs_matches = 0.8687`
- 平均 `mean_reproj_error = 1.1048 px`
- 平均 `mean_rel_depth_error = 0.0196`

Phase C（left + right + mask fusion）在同 3 个样本帧上跑通：
- 平均 `support_ratio_left = 0.01352`
- 平均 `support_ratio_right = 0.01003`
- 平均 `support_ratio_both = 0.00803`
- 平均 `support_ratio_single = 0.00749`
- 平均 `mean_reproj_error_left = 1.1048 px`
- 平均 `mean_reproj_error_right = 1.1354 px`
- 平均 `mean_rel_depth_error_left = 0.0196`
- 平均 `mean_rel_depth_error_right = 0.0241`

说明：
- verified support 没有出现“几乎全空”的坏现象；
- 双边交集 `support_both` 能稳定落到可见结构区域；
- 这说明 `internal cache + on-demand ref depth render + bidirectional reprojection verification` 这条新链已具备进入 mask-only replacement ablation 的条件。

### Internal prepare：Phase 3 正式并入 `select → difix → verify → pack`

在 full internal cache 上，把 internal prepare 正式扩成：

```text
select → difix → verify → pack
```

关键实现：
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py` 增加 `verify` stage；
- `scripts/brpo_build_mask_from_internal_cache.py` 支持 `--pseudo-left-root / --pseudo-right-root`；
- verification 可以直接消费 `difix/left_fixed` 与 `difix/right_fixed`；
- `pack` 阶段可自动把 verification 产物接入 `pseudo_cache/samples/<id>/`。

关键 prototype run：
- `re10k1__internal_afteropt__difix_proto_v2`
- `re10k1__internal_afteropt__brpo_proto_v3`

当前 3-frame prototype 使用显式 frame ids：
- `10`
- `50`
- `120`

说明：
- 这轮不是自动 `midpoint / tertile` 选点，而是显式 `frame_ids`；
- `Difix left/right`、`BRPO verification` 与 `pseudo_cache` 已在同一条 internal prepare 流程内打通。

### Phase 4：mask-only ablation（legacy vs brpo）

在同一份 after_opt PLY、同一份 3-frame internal pseudo_cache、同一份 v1 fixed-pose RGB-only refine 配置下，完成：
- `confidence_mask_source=legacy`
- `confidence_mask_source=brpo`

两组 refine 输出：
- legacy：`.../2026-04-11_mask_only_ablation_proto/legacy_left/refined.ply`
- brpo：`.../2026-04-11_mask_only_ablation_proto/brpo_left/refined.ply`

训练内现象：
- `brpo` 的 pseudo loss 下降更快、最终更低；
- 但 replay 结果并未更好。

replay eval（same internal camera states, after_opt）结果：
- baseline internal after_opt：`23.9489 / 0.87349 / 0.07878`
- legacy refine：`22.3862 / 0.84365 / 0.10834`
- brpo refine：`21.9324 / 0.82049 / 0.13134`

结论：
- 当前 BRPO verification 链条**能跑通**；
- 但当前 BRPO mask 在 `v1 fixed-pose RGB-only refine` 下**不优于 legacy mask**；
- 更可能的问题是：当前 mask 语义与当前 refine 消费方式不匹配，而不是 verification 完全失效。

### Phase 5：auditability / provenance 修补

根据 Phase 4 结果，继续检查了 `verify → pack → refine` 的 provenance 是否完整可审计。

发现的问题：
- verification 之前虽然已经支持消费 Difix left/right repaired 图，但 `verification_meta.json` 的 provenance 记录不够完整；
- `prepare_stage1_difix_dataset_s3po_internal.py` 的 sample manifest 对 `.json` 类型 BRPO 产物路径命名不够规范；
- `run_pseudo_refinement.py` 的 history 之前没有写出每个 sample 实际吃到的 target / confidence 来源。

对应补丁：
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
- `scripts/run_pseudo_refinement.py`

补丁后能力：
- verification 会记录 branch 实际输入、override 状态、policy 与 matcher/source 信息；
- pack 会记录更完整的 `source_meta` 与 `verification_*_path`；
- refine history 会记录每个 pseudo sample 的 `target_rgb_path / confidence_path / confidence_source_kind / coverage`。

说明：
- 这一步当前只完成了**代码与编译验证**；
- 若要让产物也带上新 provenance，需要在后续 Phase 6/7 中重跑 verify / pack / refine。

### Phase 6：schema solidification 第一轮实现与真实 smoke

本轮开始正式进入 Phase 6，不再停留在文档规划。

代码修改：
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
- `pseudo_branch/brpo_confidence_mask.py`
- `scripts/run_pseudo_refinement.py`

关键实现：
- `prepare_stage1_difix_dataset_s3po_internal.py` 新增 `fusion` stage，`all` 流程改为 `select → difix → fusion → verify → pack`；
- `fusion` 直接复用 `pseudo_fusion.py`，先实现 **rgb-only fused**，产出 `target_rgb_fused.png`、`confidence_mask_fused.npy/png`、`fusion_meta.json`；
- `pack` 阶段新增 `target_depth_for_refine.npy` 预留位，并将 fused 产物接入 `pseudo_cache/samples/<id>/`；
- `brpo_confidence_mask.py` 从单一 `confidence_mask_brpo.npy` 扩展为：
  - `confidence_mask_brpo_left.npy/png`
  - `confidence_mask_brpo_right.npy/png`
  - `confidence_mask_brpo_fused.npy/png`
  同时保留旧的 `confidence_mask_brpo.npy/png` 作为兼容 alias；
- `run_pseudo_refinement.py` 的 BRPO consumer 改为按 `target_side` 解析 confidence。

真实验证：
- 在现有 prototype `re10k1__internal_afteropt__brpo_proto_v3` 上，实际重跑了：
  `fusion → verify → pack`
- `verify` 目录中已真实产出新的 `left/right/fused` BRPO masks；
- `pack` 初次验证时发现一个真实漏链：verify 已产出新 masks，但 `maybe_link_brpo_artifacts()` 仍按旧白名单链接，导致 sample 里没有带入新字段；
- 随后补上 pack linkfix，并只重跑 `pack`，确认 `pseudo_cache/samples/10/` 中已真实出现：
  - `target_rgb_fused.png`
  - `target_depth_for_refine.npy`
  - `confidence_mask_brpo_left/right/fused.npy`

consumer smoke：
- 直接调用 `load_pseudo_viewpoints(...)`，确认以下三种模式都能正确解析本地 sample 内的新文件：
  - `left + brpo`
  - `right + brpo`
  - `fused + brpo`
- 进一步跑通了一次真实 refine smoke：
  - `target_side=fused`
  - `confidence_mask_source=brpo`
  - `num_iterations=2`
  - 成功进入训练循环并正常落盘。

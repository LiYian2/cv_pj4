# REFINE_SLAM_STYLE_AUDIT_20260430

更新时间：2026-04-30

## 目标

本文件记录一个明确判断：当前 Part3 live refine 不是 S3PO / SLAM 风格的 joint pose-map optimization。这里不讨论“小修补是否还能挽回一点结果”，而是审计当前壳本身为什么不对，以及为什么后续应转向结构性重构。

## 审计结论（一句话）

当前 `run_pseudo_refinement_v2.py` 的 StageA / StageA.5 / StageB 更像“离线 pseudo consumer + micro Gaussian tune”，而不是“真实 keyframe window + pseudo constraints + full Gaussian map”的 SLAM-style backend mapping。它会同时更新 pseudo pose 和一小部分 Gaussian 参数，但 joint 对象太窄，real branch 太弱，Gaussian optimizer 壳也没有复用 S3PO backend 的完整 mapping 机制。

## 审计基线与证据来源

本次判断来自以下 live 代码与真实 artifact：

- `scripts/run_pseudo_refinement_v2.py`
- `pseudo_branch/refine/pseudo_refine_scheduler.py`
- `pseudo_branch/refine/pseudo_camera_state.py`
- `pseudo_branch/gaussian_management/gaussian_param_groups.py`
- `third_party/S3PO-GS/utils/slam_backend.py`
- `third_party/S3PO-GS/utils/slam_utils.py`
- `third_party/S3PO-GS/slam.py`
- `docs/paper/BRPO_METHOD_extracted_20260424.md`
- `docs/design/REFINE_DESIGN.md`
- current exact sparse run artifact:
  `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260424_m3d_consumer_compare_stageB120_replay/sparse/stageA_history.json`

## 1. 当前 refine 实际优化了什么

### 1.1 StageA 只优化 pseudo pose / exposure

`pseudo_refine_scheduler.py` 里的 `build_stageA_optimizer()` 只为每个 pseudo viewpoint 建四组参数：

- `cam_rot_delta`
- `cam_trans_delta`
- `exposure_a`
- `exposure_b`

这些参数由 `pseudo_camera_state.py::viewpoint_optimizer_groups()` 定义。也就是说，StageA 的职责只是 pseudo pose/exposure stabilization，这一点本身与 BRPO paper 的“先稳定位姿与曝光”方向基本一致。

### 1.2 StageA.5 / StageB 的 Gaussian 自由度非常窄

`gaussian_param_groups.py::build_micro_gaussian_param_groups()` 只支持两种模式：

- `xyz`
- `xyz_opacity`

没有 `scaling`、`rotation`、`SH/color features` 等完整 Gaussian 参数组。当前 exact sparse 真实 run 的 `stageA_history.json` 进一步确认：

- `stage_mode = stageB`
- `joint_topology_mode = brpo_joint_v1`
- `stageA5_trainable_params = xyz`
- `num_real_views = 2`
- `num_pseudo_views = 4`
- `init_pseudo_camera_states_json` 非空，且 `init_handoff_summary.loaded_count = 8`

这说明 current live exact sparse 的 joint stage 实际只在改：

- pseudo pose / exposure
- Gaussian xyz

不是 full Gaussian map optimization。

## 2. 当前 StageB 的 real branch 不是 SLAM-style real mapping

### 2.1 real branch 只吃 RGB anchor

`run_pseudo_refinement_v2.py` 的 StageB 里，real branch 来自：

- `v1.load_real_viewpoints(...)`
- 每轮对 sampled real views 调 `v1.get_loss_mapping_rgb(...)`

这里用的是 RGB-only 的 mapping loss，不是 RGB-D mapping loss。

### 2.2 real viewpoints 不是 backend 窗口里的可优化 keyframes

real views 是从 `scripts/archive_experiments/legacy_entry/run_pseudo_refinement.py::load_real_viewpoints()` 读出的。该路径里 `create_viewpoint()` 会构建一个只有 `original_image` 的真实 viewpoint；它不是 S3PO backend 当前 window 中那种带 `mono_depth`、可进入 backend pose optimizer 的 live keyframe 对象。

当前 StageB 的 real branch 做的事情实际是：

- 从 train manifest 取固定 real views
- render
- 与真实 RGB 做 loss
- 把这个 loss 当作 anchor 混入总 loss

但 real camera pose 本身不进 optimizer，也没有像 `slam_backend.py::map()` 那样对 real keyframe 走 `update_pose()`。

### 2.3 这解释了为何 “real train RGB 更好但 replay 更差”

此前 forensic 已经说明：坏 route 中 real-train RGB loss 可以下降，但 replay 明显恶化。这和当前结构是一致的，因为 real branch 本质上只是弱 RGB anchor，而不是强几何/位姿锚点。

## 3. 当前 Gaussian 优化壳不是 S3PO backend mapping

### 3.1 current refine 重新起了一个 micro optimizer

`run_pseudo_refinement_v2.py` 的流程是：

- `GaussianModel(sh_degree=args.sh_degree)`
- `gaussians.load_ply(args.ply_path)`
- 然后自己用 `build_micro_gaussian_param_groups(...)` 建 `torch.optim.Adam`

这不是 S3PO backend 那个完整的 `gaussians.training_setup(opt_params)` / `gaussians.optimizer` 路线。

### 3.2 当前 live refine 没有真正的 densify / prune / update_lr mapping 机制

虽然 CLI 上保留了：

- `--stageA5_disable_densify`
- `--stageA5_disable_prune`

但 live 代码里这两个开关目前只有参数记录和 history 记录，没有对应的完整 mapping 逻辑。当前 refine 主体没有复用 S3PO backend 的：

- `densify_and_prune(...)`
- `reset_opacity_nonvisible(...)`
- `gaussians.update_learning_rate(...)`
- `gaussians.optimizer.step()`

所以它不是一个“把 pseudo supervision 接进 S3PO mapping”的实现，而是一个脱离 S3PO backend 的单独 consumer shell。

## 4. S3PO backend 实际在做什么

`third_party/S3PO-GS/utils/slam_backend.py::map()` 的 live 结构清楚表明，S3PO 的 backend mapping 是：

- 维护 real keyframe window
- 对 window views 与随机非窗口 views 做 render
- 用 `get_loss_mapping(...)` 做 real RGB-D mapping loss
- 同时 step `gaussians.optimizer`
- 同时 step `keyframe_optimizers`
- 对 real keyframe pose 调 `update_pose(viewpoint)`
- 在 mapping 过程中做 `densify_and_prune(...)`
- 做 `reset_opacity_nonvisible(...)`
- 做 `gaussians.update_learning_rate(...)`

而 `slam.py` 在 `after_opt` 前后还明确保留了 evaluation/export 流程，以及单独的 `color_refinement()` 后处理阶段。

这才是 S3PO / SLAM 风格的 pose-map joint optimization：真实 keyframe pose 与 Gaussian map 一起处在同一个 backend optimization loop 里。

## 5. 与 BRPO paper 的结构差异

`docs/paper/BRPO_METHOD_extracted_20260424.md` 里 3.3 写得非常明确：

1. 先 stabilize pose / exposure
2. 然后 joint refine Gaussians and poses
3. joint stage 用 weighted RGB-D reconstruction loss
4. 最后再做 confidence-weighted appearance refinement（L1 + SSIM）

论文里 joint stage 更新的是：

- Gaussian parameters `G = {(μ_i, Σ_i, c_i, α_i)}`
- camera poses `{T_t}`

当前 Part3 live refine 与它的差异不是“命名稍有不同”，而是结构上少了三层：

- Gaussian joint 自由度被压缩成 `xyz` 或 `xyz_opacity`
- real branch 不是真实 pose-map joint BA，只是固定 real RGB anchor
- 没有一个真正独立的 final appearance refinement stage 去对应 paper 的最后一步

## 6. 这为什么不是继续做小修补的时候

前面的 M~/T~/dense3d forensic 已经说明，当前 exact route 的问题不是单一 coverage 数字，也不是一个 target 权重就能解释完。现在再去继续做：

- `2 real + 4 pseudo -> 4 real + 2 pseudo`
- `xyz -> xyz_opacity`
- `离散 C_m -> continuous confidence`
- `关掉 depth / 再扫一个 matcher q`

本质上都还是在当前这个 consumer shell 内转旋钮。只要 joint 壳仍是“pseudo pose + micro Gaussian + fixed real RGB anchor”，就仍然有很大概率继续得到同类 failure：前期略有收益，之后 joint feedback 把 replay 拉坏。

## 7. 当前最可靠的诊断

如果按“是不是已经具备 SLAM-style joint pose-map optimization”来判断，答案是否定的。

更准确地说，当前 refine 是：

- 有 joint backward
- 但 joint 对象太窄
- real 分支不是 backend mapping
- Gaussian optimizer 不是 backend optimizer
- 没有完整 map maintenance
- 没有 paper 对应的 final appearance refinement

所以它更像“离线 pseudo-aware micro-refine shell”，而不像“真实 keyframe + pseudo constraint 的 backend continuation”。

## 8. 直接工程含义

后续主方向不应是继续把 `run_pseudo_refinement_v2.py` 打磨成更复杂的 consumer，而应是：

- 保留当前已审清的 exact M~/T~ supervision 语义
- 替换当前 refine 执行器
- 把 pseudo supervision 接入一个 S3PO-style backend continuation
- 让优化中心重新回到真实 keyframe window + full Gaussian map 上

这不是“更安全的小改动”，而是必要的结构性换壳。只有先把 refine 壳改成真正的 backend mapping，后面的 M~/T~/G~ 调整才有机会不再原地打转。

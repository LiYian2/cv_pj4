# E5a / E5b 实验设置文档：Joint-Primary Online Mapping + Masked Pseudo Color Refinement

**日期**: 2026-05-07
**状态**: 可落地配置定义
**适用对象**: DL3DV-2 / part2_s3po / Part3 BRPO online mapping 主线

---

## 0. 本文目标

定义 E5 family 的实验设置，统一基于两条已落地的 live 改动：

1. **online mapping** 已切到 joint-primary 语义：pseudo 在 refine loss window 内与 real member 同批优化，但 pseudo 不作为 Gaussian maintenance source。
2. **final color refinement** 已接入 masked pseudo：real 仍走 full-image L1+SSIM，pseudo 走 masked L1+SSIM，且该阶段不更新 pose。

E5 family 只分两支：

- **E5a**: `update_real_pose: false`
- **E5b**: `update_real_pose: true`

除此之外，其余在线映射与 final color refinement 语义保持一致。

---

## 1. E5 family 的目标语义

E5 family 要验证的是：

1. online mapping 中，pseudo 是否在 **joint-primary refine loop** 里真正作为与 real 等价的 member 生效；
2. final color refinement 中，masked pseudo 是否能把 pseudo supervision 的 appearance 收益保留下来，而不是被 real-only refinement 冲淡；
3. 仅切换 `update_real_pose` 的开关时，是否会带来额外收益或额外不稳定性。

因此，E5 family 的对照轴必须严格收敛到：

- E5a：joint-primary + masked pseudo color refinement + `update_real_pose=false`
- E5b：joint-primary + masked pseudo color refinement + `update_real_pose=true`

不再混入 GN、split authority、pseudo scene duplication、depth-conf 加权、pseudo->neighbor heuristic 等额外变量。

---

## 2. 当前 live 代码已确认的语义基础

以下结论直接来自 live 代码检查，E5 文档据此定义。

### 2.1 Online mapping 已支持 joint-primary 拓扑

`third_party/S3PO-GS/utils/slam_backend.py`

当前 keyframe event 在 `topology_mode == "joint_primary"` 时，会先准备 runtime pseudo slots，再直接调用 `_run_brpo_runtime_pseudo_mapping(...)` 作为 primary mapping block；成功后只按配置决定是否额外跑 legacy prune。

### 2.2 `match_real_loss_weights: true` 已能把 pseudo loss 权重对齐到 real mapping 的 `Training.alpha`

`third_party/S3PO-GS/utils/slam_backend.py`
`third_party/S3PO-GS/utils/slam_backend_brpo.py`

当前 live resolver 逻辑是：

- 若 `match_real_loss_weights: true`
- 则解析成：
  - `beta_rgb = Training.alpha`
  - `lambda_depth = 1 - Training.alpha`
  - 若 `use_depth=false`，则 `lambda_depth=0`

对当前 DL3DV-2 主线 config，`Training.alpha = 0.975`，因此 E5 family 的 pseudo loss 权重应实际解析为：

- `beta_rgb = 0.975`
- `lambda_depth = 0.025`

这对应 real mapping 的 `alpha * L_rgb + (1-alpha) * L_depth` 语义，而不是旧 E4 的未归一化 `49:1` 写法。

### 2.3 Pseudo 已可作为单条 masked member 被消费

`third_party/S3PO-GS/utils/slam_backend_brpo.py`

当前 live 逻辑在 `pseudo_window_equivalence=true` 时：

- `split_authority = bool(cfg.split_pseudo_authority) and not pseudo_window_equivalence`
- `scene_mask_mode = "none" if pseudo_window_equivalence else ...`

因此，只要开启 `pseudo_window_equivalence=true`，pseudo 就不会再被拆成 pose loss + scene loss 两份重复消费。

### 2.4 Densify / opacity reset 已能在 joint-primary 中开启

`third_party/S3PO-GS/utils/slam_backend.py`
`third_party/S3PO-GS/utils/slam_backend_brpo.py`

当前 live joint-primary dispatch 会传入：

- `enable_densify = True`
- `enable_prune = False`
- `enable_opacity_reset = True`

且 `gaussian_maintenance_source = real_only` 时，只允许 real-window / extra-real 视图贡献 maintenance statistics；pseudo 只贡献 supervision loss，不参与高斯维护来源。

### 2.5 Final color refinement 已支持 masked pseudo，但配置入口有两个层次

`third_party/S3PO-GS/utils/slam_backend.py`

当前 live 实现中：

1. `Results.color_refinement` 决定是否执行 final color refinement；
2. `Results.color_refinement_iters` 决定迭代数；
3. `Results.brpo_online_mapping.*` 内的以下字段控制 pseudo color refinement 行为：
   - `color_refinement_use_pseudo`
   - `color_refinement_pseudo_ratio`
   - `color_refinement_pseudo_weight`
   - `color_refinement_pseudo_mask_source`
   - `color_refinement_log_every`

也就是说，E5 配置时需要同时写：

- top-level `Results.color_refinement` / `Results.color_refinement_iters`
- 以及 `Results.brpo_online_mapping` 里的 pseudo color refinement 开关

### 2.6 当前 pseudo color refinement 的监督语义

`third_party/S3PO-GS/utils/slam_backend.py`
`third_party/S3PO-GS/gaussian_splatting/utils/loss_utils.py`

当前 live masked pseudo color refinement 的 pseudo path 是：

- pseudo 视图 render
- `target_rgb = record.target_rgb`
- `mask = record.confidence_mask`（若 `color_refinement_pseudo_mask_source=confidence_mask`）
- loss = masked L1 + masked SSIM
- 不更新 pose

因此，若 E5 family 要坚持“RGB 的 `C_m` 不由 depth 监督，不再额外乘 valid mask / target confidence”，则 color refinement 也应保持：

- `color_refinement_pseudo_mask_source: confidence_mask`

而不是 `valid_mask` 或 `support_both_mask`。

---

## 3. E5 family 的统一设置原则

以下设置对 E5a / E5b 两支都相同。

### 3.1 Online mapping 拓扑与监督语义

必须保持：

- `topology_mode: joint_primary`
- `pseudo_window_equivalence: true`
- `match_real_loss_weights: true`
- `split_pseudo_authority: false`
- `pseudo_scene_mask_mode: both_only`  （在 equivalence=true 下实际上应失效，但保留成非重复消费语义）
- `propagate_pseudo_delta_to_neighbors: false`
- `gaussian_maintenance_source: real_only`

含义：

1. pseudo 进入 joint-primary refine loop；
2. pseudo 按单条 masked member 消费；
3. pseudo 不走 split pose/scene duplication；
4. pseudo 不再通过 heuristic neighbor propagation 影响 real pose；
5. pseudo 不作为 densify / opacity reset / prune 的 maintenance source。

### 3.2 Loss contract

E5 family 要坚持 paper decoupled RGB route：

- `rgb_only_verification: true`
- `depth_generation_mode: projected`
- `depth_loss_mode: paper_brpo_split_v1`
- `use_depth: true`
- `match_real_loss_weights: true`

语义要求：

1. RGB 侧 `C_m` 不由 depth validity / depth confidence 再次修饰；
2. RGB loss 只由 paper-route `confidence_mask` 决定；
3. depth 仍然乘同一个 `C_m`；
4. depth target 仍来自双向投影 / projected depth，而不是 direct MASt3R depth。

当前 `paper_brpo_split_v1` + `rgb_only_verification: true` 对应的就是这条定义。

### 3.3 Gaussian maintenance

E5 family 明确要求：

- densify: 开
- opacity reset: 开
- prune: 关

对当前 live joint-primary 实现，这需要同时保证：

- joint loop 内部：`enable_densify=True`, `enable_prune=False`, `enable_opacity_reset=True`
- **额外**：`joint_primary_run_legacy_prune: false`

注意：如果保留 `joint_primary_run_legacy_prune: true`，那么即使 joint loop 内部 `enable_prune=False`，成功路径后仍会额外调用一次 legacy `map(..., prune=True)`，这与“保持关闭 prune”不一致。

因此，E5 family 必须显式写：

- `joint_primary_run_legacy_prune: false`

### 3.4 Final color refinement

E5 family 要启用 masked pseudo color refinement：

- `Results.color_refinement: true`
- `Results.color_refinement_iters: 26000`  （若无特殊说明，沿用 live default）
- `color_refinement_use_pseudo: true`
- `color_refinement_pseudo_mask_source: confidence_mask`
- `color_refinement_pseudo_ratio: 0.5`
- `color_refinement_pseudo_weight: 1.0`
- `color_refinement_log_every: 200`

这里的选择理由是：

1. `confidence_mask` 最贴近作者所说的 `C_m`；
2. `pseudo_ratio=0.5` / `pseudo_weight=1.0` 保持当前 live 默认 mixed refinement 语义，先不再引入额外加权变量；
3. color refinement 阶段只改 gaussians，不改 pose，因此其作用应主要体现在 PSNR / SSIM / LPIPS，而不是轨迹。

### 3.5 其余参数

若本文未显式覆盖，其余参数全部 **继承 corrected E4 baseline**，包括但不限于：

- `placement_mode: midpoint_only`
- `max_pseudo_per_gap: 1`
- `num_pseudo_views_per_step: 1`
- `pseudo_map_iters: 20`
- `extra_real_views: 2`
- `update_real_exposure: true`
- `lambda_real: 1.0`
- `lambda_pseudo: 1.0`
- `lambda_pose: 0.01`
- `lambda_exp: 0.001`
- `lambda_scale: 0.01`
- `trans_weight: 1.0`
- `lambda_abs_pose: 0.0`
- `lambda_abs_t: 3.0`
- `lambda_abs_r: 0.1`
- `abs_pose_robust: charbonnier`
- `matcher_mode: dense_pts3d_3d`
- `dense3d_conf_quantile: 0.15`
- `tau_reproj_px: 20.0`
- `tau_rel_depth: 1.0`
- `use_difix_restoration: true`
- `use_gauss_newton: false`

---

## 4. E5a / E5b 的唯一区别

### 4.1 E5a

```yaml
update_real_pose: false
update_pseudo_pose: true
```

语义：

- real KF pose 不由 pseudo joint refine 直接更新；
- pseudo pose 仍可更新；
- 用于回答：仅靠 joint-primary + masked pseudo appearance refinement，本身是否已经足够带来收益。

### 4.2 E5b

```yaml
update_real_pose: true
update_pseudo_pose: true
```

语义：

- pseudo joint refine 允许直接进入 real KF pose 优化闭环；
- 用于回答：在 E5a 已经接通 joint-primary 与 masked pseudo color refinement 的基础上，额外放开 real pose 是否还能带来进一步收益，或引入新的不稳定。

---

## 5. 推荐的配置块（E5 family 公共部分）

以下为建议直接落入 `Results` / `Results.brpo_online_mapping` 的公共配置块。

```yaml
Results:
  color_refinement: true
  color_refinement_iters: 26000

  brpo_online_mapping:
    enabled: true
    trigger: keyframe

    # joint-primary online mapping
    topology_mode: joint_primary
    placement_mode: midpoint_only
    max_pseudo_per_gap: 1
    num_pseudo_views_per_step: 1
    pseudo_map_iters: 20
    enable_pseudo_gradient: true

    # equal-member semantics
    pseudo_window_equivalence: true
    extra_real_views: 2
    propagate_pseudo_delta_to_neighbors: false
    update_real_exposure: true
    gaussian_maintenance_source: real_only
    joint_primary_run_legacy_prune: false

    # loss weights: mirror real mapping alpha exactly
    lambda_real: 1.0
    lambda_pseudo: 1.0
    match_real_loss_weights: true
    beta_rgb: 0.975
    lambda_depth: 0.025

    lambda_pose: 0.01
    lambda_exp: 0.001
    lambda_scale: 0.01
    max_scale: null
    trans_weight: 1.0
    lambda_abs_pose: 0.0
    lambda_abs_t: 3.0
    lambda_abs_r: 0.1
    abs_pose_robust: charbonnier

    # pose switches
    # E5a: update_real_pose: false
    # E5b: update_real_pose: true
    update_pseudo_pose: true

    # paper-decoupled RGB route
    use_depth: true
    split_pseudo_authority: false
    pseudo_scene_mask_mode: both_only
    isotropic_weight: 10.0
    depth_generation_mode: projected
    depth_loss_mode: paper_brpo_split_v1
    rgb_only_verification: true

    # matcher / verifier
    matcher_mode: dense_pts3d_3d
    dense3d_conf_quantile: 0.15
    tau_reproj_px: 20.0
    tau_rel_depth: 1.0

    # no GN
    use_gauss_newton: false
    gn_max_iters: 5
    gn_damping: 0.01
    gn_every_n_steps: 1

    # masked pseudo color refinement
    color_refinement_use_pseudo: true
    color_refinement_pseudo_ratio: 0.5
    color_refinement_pseudo_weight: 1.0
    color_refinement_pseudo_mask_source: confidence_mask
    color_refinement_log_every: 200
```

说明：

- `beta_rgb: 0.975` / `lambda_depth: 0.025` 这里即使写死，也会与 `match_real_loss_weights: true` 的当前 live `Training.alpha=0.975` 一致；
- 若后续 `Training.alpha` 改动，建议同步更新文档里的显式数值，避免文字和运行时解析漂移；
- 对当前 E5 family，`joint_primary_run_legacy_prune: false` 是必须显式写出的关键项。

---

## 6. E5 family 的预期 artifact 检查点

正式运行 E5a / E5b 后，应首先检查以下 artifact，确认实验确实按本文设定执行，而不是 silent fallback。

### 6.1 Online mapping event summary

路径示例：

- `.../brpo_debug/event_kf_XXXX/event_summary.json`
- `.../brpo_debug/event_kf_XXXX/pseudo_mapping_summary.json`

应看到：

- `topology_mode = joint_primary`
- `pseudo_window_equivalence = true`
- `match_real_loss_weights = true`
- `beta_rgb = 0.975`
- `lambda_depth = 0.025`
- `gaussian_maintenance_source = real_only`
- `joint_primary_used = true`

且不应出现：

- `joint_primary_status = fallback_real_only`

### 6.2 Joint history

路径示例：

- `.../joint_primary_mapping/brpo_pseudo_history.json`

应重点检查：

- `split_pseudo_authority = false`
- `resolved_beta_rgb = 0.975`
- `resolved_lambda_depth = 0.025`
- `num_real_window_members > 0`
- `num_pseudo_members > 0`
- `num_extra_real_members >= 0`
- `neighbor_pose_propagation_enabled = false`

E5a / E5b 的主要差异检查点：

- E5a: `num_real_pose_optimized = 0`
- E5b: `num_real_pose_optimized > 0`

### 6.3 Color refinement summary

路径：

- `.../color_refinement_summary.json`

应看到：

- `use_pseudo = true`
- `pseudo_mask_source = confidence_mask`
- `pseudo_ratio = 0.5`
- `pseudo_weight = 1.0`
- `num_pseudo_steps > 0`
- `color_refinement_updates_pose = false`

---

## 7. 本文最终定义

E5 family 的标准定义如下：

1. **paper decoupled RGB route**：`rgb_only_verification=true`，RGB 的 `C_m` 不再由 depth-side valid/conf 再次修饰；depth 仍乘 `C_m`，depth target 仍是 projected depth。
2. **joint-primary online mapping**：pseudo 在 refine 环节与 real member 同批进入 joint loop；pseudo 不拆成 pose+scene 两份，不走 neighbor heuristic，不作为 Gaussian maintenance source。
3. **Gaussian maintenance**：densify 开、opacity reset 开、prune 全关（包括 legacy prune）。
4. **masked pseudo final color refinement**：pseudo 以 `confidence_mask` 作为 `C_m` 进入 final appearance refinement，且该阶段只更新 gaussians，不更新 pose。
5. **唯一实验变量**：
   - E5a: `update_real_pose=false`
   - E5b: `update_real_pose=true`

如果后续没有新的用户指令，E5 family 就按本文定义执行，不再额外引入 GN、split authority、depth-conf weighting、pseudo ratio/weight 调参等新变量。
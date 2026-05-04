# M3D_STRUCTURAL_FORENSICS_20260424.md

> 更新时间：2026-04-24 13:33 (CST)

## 1. 问题
在 `StageB120 + replay` compare 中，MASt3R 3D dense matching 虽然把 exact mask / exact-upstream target coverage 显著抬高，但 replay 仍输给 sparse。需要回答两个问题：
1. 当前 live M~ 低覆盖率是否真的是因为旧路径仍在使用 sparse 2D matching？
2. dense3d 接通后，`exact_brpo_upstream_target_v1` 的 target depth / target confidence / source map 是否真的同步切到了新 exact backend，还是只改了 `C_m`？

## 2. grounded 结论
- 旧 live exact M~ 的匹配路径确实是 `FlowMatcher + fast_reciprocal_NNs(..., subsample_or_initxy1=8)` 的 sparse 2D MASt3R reciprocal descriptor matching，而不是 dense matching，也不是 MASt3R 3D matching。
- 因而 `~1.5%–2%` 的 raw exact `C_m` coverage 是实现自然结果，不是额外异常。512×512 下 stride-8 seed 只有约 `4096 / 262144 = 1.5625%`；live summary 的 sparse `cm_nonzero_ratio ≈ 0.01543` 与此严格一致。
- dense3d 接通后，`exact_brpo_upstream_target_v1` 的 target depth 并没有停留在旧路径。代码和产物都表明：`pseudo_depth_target_exact_brpo_upstream_target_v1.npy`、`pseudo_target_confidence_exact_brpo_upstream_target_v1.npy`、`pseudo_source_map_exact_brpo_upstream_target_v1.npy` 会随 matcher 一起变化，不是只换了 `C_m`。
- 当前 replay 变差的更可能根因不是“target depth 没同步”，而是 dense3d 新增 supervision 的组成偏单边、有效权重偏弱，导致更多 noisy / asymmetric 的 supervision 被下游优化吃进去。

## 3. code path tracing
### 3.1 旧 sparse M~ 路径
- `pseudo_branch/common/flow_matcher.py`
  - `FlowMatcher.match_pair()` 内部使用 `fast_reciprocal_NNs(..., subsample_or_initxy1=8, dist="dot")`
- `scripts/brpo_build_mask_from_internal_cache.py`
  - live exact backend 旧主路由通过 `FlowMatcher` 生成 reciprocal matches
- `pseudo_branch/mask/rgb_mask_inference.py`
- `pseudo_branch/mask/brpo_confidence_mask.py`
- `pseudo_branch/observation/brpo_reprojection_verify.py`

这条路径对应的是 sparse 2D reciprocal descriptor matching，不是 dense matcher。

### 3.2 dense3d exact-upstream target 路径
- `scripts/build_brpo_v2_signal_from_internal_cache.py`
  - `exact_brpo_upstream_target_v1` 分支会加载 exact backend bundle：
    - `support_left_exact.npy`
    - `support_right_exact.npy`
    - `projected_depth_left_exact.npy`
    - `projected_depth_right_exact.npy`
    - `confidence_left_exact.npy`
    - `confidence_right_exact.npy`
  - 然后调用 `build_exact_brpo_upstream_target_observation(...)`
- `pseudo_branch/target/depth_supervision_v2.py`
  - `build_exact_upstream_depth_target(...)` 直接用上述 exact supports / exact projected depths / exact confidences 生成：
    - `pseudo_depth_target_exact_upstream_v1`
    - `pseudo_source_map_exact_upstream_v1`
    - `pseudo_valid_mask_exact_upstream_v1`
    - `pseudo_confidence_exact_upstream_v1`
  - 注释与实现都明确：`no_render_fallback = True`
- `scripts/run_pseudo_refinement_v2.py`
  - `pseudo_observation_mode == exact_brpo_upstream_target_v1` 时强制要求 `stageA_depth_loss_mode == exact_shared_cm_v1`
- `pseudo_branch/refine/pseudo_loss_v2.py`
  - `build_stageA_loss_exact_shared_cm(...)` 实际使用：shared discrete `C_m`、optional `valid_mask`、optional continuous `target_confidence`
  - 实际 effective mask 为 `C_m × valid_mask × target_confidence`

## 4. artifact comparison: sparse vs dense3d
比较根目录：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260424_m3d_live_smoke_full`
比较对象：`sparse_signal` / `dense3d_q090_signal` / `dense3d_q080_signal` / `dense3d_q070_signal`
字段：`exact_brpo_upstream_target_v1`

### 4.1 exact-upstream target 确实随 dense3d 变化
8 帧均值：

| arm | depth_nz | src_verified | target_conf_nz |
| --- | ---: | ---: | ---: |
| sparse | 0.01543 | 0.01543 | 0.01543 |
| q0.90 | 0.06054 | 0.06054 | 0.06054 |
| q0.80 | 0.12747 | 0.12747 | 0.12747 |
| q0.70 | 0.19078 | 0.19078 | 0.19078 |

逐像素差异（相对 sparse，8 帧平均）：
- q0.90：`depth_equal_ratio ≈ 0.9252`，`src_equal_ratio ≈ 0.9256`
- q0.80：`depth_equal_ratio ≈ 0.8596`，`src_equal_ratio ≈ 0.8606`
- q0.70：`depth_equal_ratio ≈ 0.7975`，`src_equal_ratio ≈ 0.7989`

也就是说，dense3d 下 `exact_upstream` 三件套发生了实质变化；不是“只有 `C_m` 变了，target depth 没变”。

### 4.2 但新增区域主要不是 both-branch supervision
8 帧均值：

| arm | `cm_both_ratio` | `cm_single_ratio` | valid_ratio |
| --- | ---: | ---: | ---: |
| sparse | 0.00738 | 0.00805 | 0.01543 |
| q0.90 | 0.01336 | 0.04718 | 0.06054 |
| q0.80 | 0.04068 | 0.08679 | 0.12747 |
| q0.70 | 0.06902 | 0.12176 | 0.19078 |

对 q0.70 相对 sparse 的新增 valid 区域做分解：
- 新增区域占全图约 `0.1873`
- 其中 `C_m = 1.0` 约 `35.9%`
- 其中 `C_m = 0.5` 约 `64.1%`
- source map 上约 `22.7% left-only / 41.5% right-only / 35.9% both-weighted`

因此 dense3d 扩出来的大头并不是双边稳健几何，而是单边支持。

### 4.3 exact_shared_cm_v1 真正送进 loss 的 effective mask 变大了，但单位质量变弱
8 帧均值：

| arm | `cm_mean` | `eff_mean_all` | `eff_mean_on_valid` |
| --- | ---: | ---: | ---: |
| sparse | 0.01141 | 0.00747 | 0.47957 |
| q0.90 | 0.03695 | 0.02130 | 0.34838 |
| q0.80 | 0.08408 | 0.04974 | 0.38706 |
| q0.70 | 0.12990 | 0.07724 | 0.40313 |

这里 `effective mask = C_m × valid_mask × target_confidence`。
结论是：dense3d 的总监督质量总量变大了，但每个有效像素上的平均 effective weight 低于 sparse，说明新增区域整体更弱、更不稳。

## 5. downstream evidence
`StageB120 + replay` compare：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260424_m3d_consumer_compare_stageB120_replay`

Replay：
- sparse：PSNR `24.0045` / SSIM `0.87177` / LPIPS `0.08247`
- q0.70：PSNR `23.6660` / SSIM `0.86585` / LPIPS `0.08577`
- q0.80：PSNR `23.5816` / SSIM `0.86296` / LPIPS `0.08702`

StageB loss（全程均值）：
- sparse：`loss_pseudo ≈ 0.04638`，`loss_depth ≈ 0.03861`
- q0.70：`loss_pseudo ≈ 0.05788`，`loss_depth ≈ 0.04759`
- q0.80：`loss_pseudo ≈ 0.05928`，`loss_depth ≈ 0.04794`

这和上面的组成分析一致：dense3d 的 supervision 数量增大，但下游真实吃进去之后，pseudo/depth loss 反而更高，最终 replay 更差。

## 6. answer to the key question
### target depth 会不会没有同步做 3D dense matching？
当前 grounded 答案：不会，至少不是这个层面的 bug。

更准确的说法是：
- dense3d 已经同步改动了 exact backend 输出
- signal builder 也已同步重建 exact-upstream target depth / target confidence / source map
- consumer 的 `exact_shared_cm_v1` 也确实在吃这些新产物
- 但 dense3d 新增 supervision 的大头来自单边支持与较弱 effective weight，因此 coverage 增长没有自动转成 replay 增益

## 7. next step
当前不应继续做同类 q-sweep，也不应再把排查重点放在“target depth 有没有同步”上。
更合理的下一步是回到 BRPO 原论文 method，对照 live 语义检查以下差异：
1. M~ 对 both / xor / unsupported 的原始定义与当前 exact `C_m` 是否仍有细小实现偏差
2. T~ 在 single-branch 区域是否应该像当前这样直接进入 target depth + shared loss
3. 原方法是否对 single-branch / low-confidence supervision 有更强的过滤、重权、或只在某些损失项中使用
4. 当前 `exact_shared_cm_v1` 的 shared-mask 设计，是否比原方法更容易把单边 dense 区域直接放大到 RGB+depth 双损失中

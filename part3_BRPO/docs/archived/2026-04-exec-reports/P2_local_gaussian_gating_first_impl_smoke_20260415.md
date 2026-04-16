# P2 local Gaussian gating 第一版实现 + smoke（2026-04-15）

## 1. 这次做了什么

本轮完成的是 `P2 local Gaussian gating` 第一版的工程接入与最小 smoke，不是 8-frame replay compare。

实际完成内容：
1. 在 `scripts/run_pseudo_refinement_v2.py` 中新增 `pseudo_local_gating_*` CLI；
2. 新增 `pseudo_branch/local_gating/`，把第一版 gating 逻辑拆成独立模块；
3. 在 `StageA.5` 上接入：`pseudo-only loss.backward() -> local gate mask Gaussian grad -> optimizer.step()`；
4. 在 `StageB` 上改成 split backward 结构：
   - `pseudo_backward(retain_graph=has_real_branch)`
   - 仅对 pseudo-side Gaussian grad 做 local gate mask
   - 若有 real branch，再做 `real_backward()`
   - 然后统一 step optimizer
5. 把 gating summary 写进 `stageA_history.json / stageB_history.json`。

## 2. 新增 / 修改代码

### 修改文件
- `scripts/run_pseudo_refinement_v2.py`

### 新增目录
- `pseudo_branch/local_gating/`
  - `__init__.py`
  - `gating_schema.py`
  - `signal_gate.py`
  - `visibility_union.py`
  - `grad_mask.py`
  - `gating_io.py`

## 3. 第一版固定边界（与执行计划保持一致）

当前第一版实际落地边界是：
- pseudo-side only
- hard/soft signal gate CLI 已预留，但这次主验证只做 hard gating
- `visibility_filter union`
- `xyz-only` 为第一默认参数
- StageA.5 先接通，StageB 也已补 split-backward 接口
- 不做 full SPGM
- 不做 per-pixel -> Gaussian 精细映射
- 不改 real branch 的全局纠偏路径

## 4. 新增 CLI

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

## 5. history 里现在会记录什么

每个 iteration 现在会写入：
- `pseudo_local_gating_mode`
- `pseudo_local_gating_params`
- `accepted_pseudo_sample_ids`
- `rejected_pseudo_sample_ids`
- `rejected_reasons`
- `accepted_signal_weights`
- `sample_signal_metrics`
- `visible_union_ratio`
- `visible_union_weight_mean`
- `accepted_visibility_count`
- `grad_keep_ratio_xyz`
- `grad_keep_ratio_opacity`
- `grad_weight_mean_xyz`
- `grad_weight_mean_opacity`
- `grad_norm_xyz_pre_mask / post_mask`
- `grad_norm_opacity_pre_mask / post_mask`

## 6. smoke run roots

### StageA.5 off / hard
- off:
  - `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_local_gating_smoke/stageA5_legacy_off_2iter`
- hard:
  - `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_local_gating_smoke/stageA5_legacy_hard_2iter`

### StageA.5 hard reject harness
- `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_local_gating_smoke/stageA5_legacy_hard_reject_finalcheck`

### StageB no-real smoke
- `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_local_gating_smoke/stageB_legacy_hard_noreal_1x1`

## 7. 关键 smoke 结果

### 7.1 StageA.5 off（2 iter）
- `mode=off`
- iter2 `accepted_last=[92, 57]`
- iter2 `visible_union_last=None`
- iter2 `grad_keep_ratio_xyz=1.0`
- iter2 `grad_norm_xyz_pre/post=0.06377 / 0.06377`

### 7.2 StageA.5 hard（2 iter，默认阈值）
- `mode=hard_visible_union_signal`
- iter2 `accepted_last=[92, 57]`
- iter2 `visible_union_last≈0.66872`
- iter2 `grad_keep_ratio_xyz≈0.66872`
- iter2 `grad_norm_xyz_pre/post=0.06377 / 0.06377`

这说明：
- gating summary 和 visible union 统计已经真实写进 history；
- 当 sampled pseudo views 全部通过 signal gate 时，第一版 `visibility_filter union` 与原始 pseudo-side visibility 支持基本一致，因此不会额外改变 xyz grad norm。

### 7.3 StageA.5 hard reject harness（1 iter）
- 设置：`min_rgb_mask_ratio=0.25`
- iter1 `accepted=0/2`
- `rejected_reasons={'196': ['rgb_mask_ratio'], '260': ['rgb_mask_ratio']}`
- `visible_union_ratio=0.0`
- `grad_keep_ratio_xyz=0.0`
- `grad_norm_xyz_pre/post=0.3790 -> 0.0`

这一步提供了关键验证：
- 不是只有 summary 在写，grad mask 本身确实能把 Gaussian xyz grad 实际裁成 0。

### 7.4 StageB no-real smoke（1+1 iter）
- `stage_mode=stageB`
- `lambda_real=0.0`
- StageB iter1 `accepted=0/2`
- `visible_union_ratio=0.0`
- `grad_keep_ratio_xyz=0.0`
- `grad_norm_xyz_pre/post=0.06378 -> 0.0`

这说明：
- StageB 新的 split-backward + pseudo-side gating 路径已经能真实执行；
- 这次没有 real branch，因此还不能把“real branch 保留全局纠偏”当作已 smoke 通过。

## 8. 这轮真正得到的工程判断

1. 第一版 local gating 已经接通，不再停留在计划层。
2. history / CLI / grad mask / StageB split-backward 都已经进入真实代码路径。
3. 一个重要设计细节已经更清楚：
   - 在 `StageA.5 pseudo-only` 下，如果 sampled pseudo views 全部通过 signal gate，那么 loss 本来就只来自这些 sampled views，Gaussian grad 本身已经局限在这些 views 的 visible subset；
   - 因而第一版 hard gating 的真正价值，主要体现在“拒绝 weak pseudo views 时阻断它们对 Gaussian 的更新”，而不是在“所有 sampled views 都通过时额外再缩一层 visible subset”。
4. 因此后续真正值得做的 compare，不是继续做超小 smoke，而是：
   - 在 8-frame `StageA.5` 上做 gated vs ungated short compare；
   - 主看 replay / held-out 是否更稳、以及 depth loss 是否没有被压没。

## 9. 当前尚未完成的部分

- 还没有做 8-frame `StageA.5` gated vs ungated short compare；
- 还没有做带 real branch 的 `StageB` smoke / compare；
- 还没有判断 gating 默认应该先挂在 `legacy` 还是 `RGB-only v2` 上；
- 还没有做 `xyz+opacity` 或 soft gating 的正式验证。

## 10. 下一步建议

下一步应直接进入：
1. `legacy` 背景下的 8-frame `StageA.5` gated vs ungated short compare；
2. 若通过，再接 `StageB` 的 real-branch smoke；
3. 之后才考虑 `RGB-only v2 gated`、`xyz+opacity`、soft gating 或更重的 SPGM。

# SIGNAL_AWARE_SELECTION_E15_COMPARE_20260414

## 1. 这轮 E1.5 在做什么

这轮不是再讨论 E1，而是把 `support-aware pseudo selection` 放进和 midpoint8 相同的正式链路里，做一次 apples-to-apples 验证。

support-aware pseudo selection 当前做的事情很具体：
1. 对每个 KF gap，不再默认固定取 midpoint；
2. 先枚举一组候选 pseudo 位置（当前是 `{1/3, 1/2, 2/3}`）；
3. 对每个候选，基于当前 stage PLY 与左右 anchor 做 lightweight verify；
4. 用 `support / verified depth / continuous confidence / correction magnitude / 左右平衡` 这些信号给候选打分；
5. 每个 gap 只保留 1 个分数更高的 pseudo frame。

它并没有：
- 增加 pseudo 数量；
- 增加 anchor 数量；
- 改动下游 refine 目标函数。

它只是在回答一个更基础的问题：同样只放 1 个 pseudo 时，midpoint 是否真的选对了位置。

## 2. 本轮正式验证设置

- 选中 pseudo ids：`[23, 57, 92, 127, 162, 196, 225, 260]`
- 对照 midpoint8：`[17, 51, 86, 121, 156, 190, 225, 260]`
- 新实验目录：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare`
- 正式链路：`select -> fusion -> verify -> pack -> M5 densify -> StageA-20iter`
- verify/pack 设置与 midpoint8 保持一致：
  - `verification_mode=fused_first`
  - `train_mask_mode=propagate`
  - `depth_fallback_mode=render_depth`
  - `depth_both_mode=average`
- densify 也复用了 midpoint8_compare 的同一组参数：
  - baseline：默认 M5 densify
  - conf-aware：`min_patch_confidence=0.12, both_relax=2, single_std_tighten=0.9`

## 3. signal layer：raw signal 对照

| metric | midpoint8 | signal-aware-8 | delta |
| --- | ---: | ---: | ---: |
| support_ratio_left | 0.01233 | 0.01148 | -0.00085 |
| support_ratio_right | 0.01019 | 0.01133 | 0.00114 |
| support_ratio_both | 0.00754 | 0.00738 | -0.00016 |
| verified_ratio | 0.01498 | 0.01543 | 0.00044 |
| continuous_confidence_mean_positive | 0.69049 | 0.69546 | 0.00496 |
| agreement_mean_positive | 0.75106 | 0.75282 | 0.00176 |
| render_fallback_ratio | 0.98502 | 0.98457 | -0.00044 |

解读：
- `support_ratio_both` 略降；
- 但 `verified_ratio`、`continuous confidence`、`agreement` 都有正向提升；
- 说明这次 selection 不是简单把 both-support 撑大，而是让有效 verified signal 更集中、更稳定一些。

## 4. densify layer：M5 densify 对照

### 4.1 baseline densify

| metric | midpoint8 | signal-aware-8 | delta |
| --- | ---: | ---: | ---: |
| mean_seed_valid_ratio | 0.01498 | 0.01543 | 0.00044 |
| mean_dense_valid_ratio | 0.02269 | 0.02961 | 0.00691 |
| mean_densified_only_ratio | 0.01990 | 0.02591 | 0.00601 |
| mean_render_fallback_ratio | 0.96511 | 0.95866 | -0.00645 |

### 4.2 conf-aware densify

| metric | midpoint8 | signal-aware-8 | delta |
| --- | ---: | ---: | ---: |
| mean_seed_valid_ratio | 0.01498 | 0.01543 | 0.00044 |
| mean_dense_valid_ratio | 0.01634 | 0.01831 | 0.00197 |
| mean_densified_only_ratio | 0.01265 | 0.01425 | 0.00159 |
| mean_render_fallback_ratio | 0.97236 | 0.97032 | -0.00204 |

解读：
- baseline densify 提升最明显，`mean_dense_valid_ratio` 从 `0.02269` 提到 `0.02961`；
- conf-aware densify 也有稳定提升，`mean_dense_valid_ratio` 从 `0.01634` 提到 `0.01831`；
- 说明 support-aware selection 的正向影响不只停留在 raw signal，而是能继续传导到 densify coverage。

## 5. short refine layer：8-frame StageA-20iter 对照

### 5.1 baseline StageA-20

| metric | midpoint8 | signal-aware-8 | delta |
| --- | ---: | ---: | ---: |
| loss_total | 0.03673 | 0.03659 | -0.00014 |
| loss_rgb | 0.01054 | 0.01029 | -0.00025 |
| loss_depth | 0.08524 | 0.08539 | 0.00015 |
| mean_trans_norm | 0.00116 | 0.00109 | -0.00007 |

### 5.2 conf-aware StageA-20

| metric | midpoint8 | signal-aware-8 | delta |
| --- | ---: | ---: | ---: |
| loss_total | 0.04392 | 0.04340 | -0.00052 |
| loss_rgb | 0.01745 | 0.01702 | -0.00043 |
| loss_depth | 0.09224 | 0.09138 | -0.00086 |
| mean_trans_norm | 0.00128 | 0.00120 | -0.00008 |

解读：
- baseline StageA-20 下，signal-aware-8 的 `loss_total` 略低于 midpoint8，`loss_rgb` 更低，`pose mean_trans` 也略低；
- conf-aware StageA-20 下，signal-aware-8 的 `loss_total / loss_rgb / loss_depth` 也都略低于 midpoint8；
- 提升幅度不大，但方向一致，而且没有出现“signal 更强但 short refine 更差”的反噬。

## 6. 当前结论

到 E1.5 为止，可以把 support-aware pseudo selection 的结论收敛为：
1. 这一步做的不是增加 pseudo 数量，而是把“每个 gap 唯一一个 pseudo 放在哪里”从固定 midpoint 改成 data-driven selection；
2. 在当前 re10k case 上，前 6 个 gap 的最佳位置都偏向 `2/3`，说明 midpoint 不是稳定最优点；
3. raw signal 上，selection 带来了更高的 verified ratio 与 continuous confidence；
4. densify 层上，selection 的收益更明显，dense_valid_ratio 有持续提升；
5. short StageA-20 上，baseline/confaware 两条线都出现了轻微但一致的正向改善。

因此，这一轮已经不只是“方向看起来对”，而是完成了正式 apples-to-apples 短对照验证。E1 可以视为通过，下一步可以进入 E2（dual-pseudo allocation）。

## 7. 产物路径

- `scripts/select_signal_aware_pseudos.py`
- `.../20260414_signal_enhancement_e1/reports/signal_aware_selection_report.json`
- `.../20260414_signal_enhancement_e15_compare/e15_compare_summary.json`
- `docs/SIGNAL_AWARE_SELECTION_E1_REPORT_20260414.md`
- `docs/SIGNAL_AWARE_SELECTION_E15_COMPARE_20260414.md`

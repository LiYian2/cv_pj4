# SIGNAL_AWARE_SELECTION_E1_REPORT_20260414

## 1. 目标

执行 `SIGNAL_ENHANCEMENT.md` 的 E1：先不改 pseudo 数量，只验证“固定 midpoint 是否确实选错了位置”。

## 2. 本轮设置

- 脚本：`scripts/select_signal_aware_pseudos.py`
- 输出目录：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e1`
- internal cache：`/data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache`
- stage：`after_opt`
- 候选位置：`{1/3, 1/2, 2/3}`
- 评分项：`both_ratio + verified_ratio + correction_magnitude + balance`

评分权重：
- `w_both=4.0`
- `w_verified=2.0`
- `w_correction=1.0`
- `w_balance=2.0`

## 3. 重要限制

这次 E1 是 lightweight selection run，不是 full fused-first apples-to-apples rerun。

具体来说：
1. 候选枚举覆盖 8 个 gap 的 `{1/3, 1/2, 2/3}`；
2. verify 直接使用 internal cache 的 `render_rgb/render_depth` 做轻量打分；
3. 没有为 24 个候选先补齐完整 fusion / pseudo_cache / StageA short run。

因此，这轮结果足够回答“midpoint 是否可能选错”，但还不足以单独宣告 E1 最终验收完成。

## 4. 结果概览

- midpoint8：`[17, 51, 86, 121, 156, 190, 225, 260]`
- signal-aware-8：`[23, 57, 92, 127, 162, 196, 225, 260]`
- 发生变化的 gap 数：`6 / 8`

逐 gap 结果：

| gap | midpoint | selected | selected_label | changed | midpoint_score | selected_score |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| [0, 34] | 17 | 23 | frac_667 | yes | 0.10587 | 0.11031 |
| [34, 69] | 51 | 57 | frac_667 | yes | 0.10191 | 0.10759 |
| [69, 104] | 86 | 92 | frac_667 | yes | 0.09720 | 0.09822 |
| [104, 139] | 121 | 127 | frac_667 | yes | 0.10325 | 0.10678 |
| [139, 173] | 156 | 162 | frac_667 | yes | 0.10371 | 0.10510 |
| [173, 208] | 190 | 196 | frac_667 | yes | 0.10231 | 0.10320 |
| [208, 243] | 225 | 225 | frac_500 | no | 0.09155 | 0.09155 |
| [243, 278] | 260 | 260 | frac_500 | no | 0.10227 | 0.10227 |

最明显的模式：
- 前 6 个 gap 全部从 `midpoint` 改成了 `2/3`；
- 最后 2 个 gap 保持 `midpoint`；
- 说明在这个 case 上，pseudo 最优位置明显偏向 gap 后段，而不是总在几何中心。

## 5. 与 midpoint8 的 summary 对照

| metric | midpoint8 | signal-aware-8 | delta |
| --- | ---: | ---: | ---: |
| support_ratio_both | 0.00754 | 0.00738 | -0.00016 |
| verified_ratio | 0.01498 | 0.01543 | 0.00044 |
| continuous_confidence_mean_positive | 0.69049 | 0.69546 | 0.00496 |
| agreement_mean_positive | 0.75106 | 0.75282 | 0.00176 |
| mean_abs_rel_correction_verified | 0.02051 | 0.02102 | 0.00051 |
| balance_ratio | 0.01019 | 0.01086 | 0.00067 |
| score | 0.10101 | 0.10313 | 0.00212 |

## 6. 解释

这轮结果支持两点：
1. `midpoint` 不是中性的默认最优点；至少在当前 re10k case 的前 6 个 gap 里，`2/3` 更容易拿到更高的 lightweight signal score；
2. 本轮提升主要来自 `verified_ratio / continuous confidence / balance / correction magnitude`，而不是 `both_ratio` 的同步上升。

同时也要明确一条保守结论：
- `support_ratio_both` 略低于 midpoint8；
- 所以当前不能把 E1 解释成“所有 signal 指标都一起更好”；
- 更准确的说法是：它给出了一个方向上更优、但仍需 apples-to-apples verify/pack + short refine 复核的 selection proposal。

## 7. 当前判断

E1 方向成立，可以继续往前走；但在进入 E2 之前，建议先补一个 E1.5 式验证：
1. 用 `signal-aware-8` 的 frame ids 生成新的 pseudo set；
2. 走与 midpoint8 相同的 verify/pack 链路；
3. 做一次 apples-to-apples 的 signal summary + 8-frame StageA short compare；
4. 如果结果仍稳定优于或接近 midpoint8，再进入 E2（dual-pseudo allocation）。

## 8. 产物路径

- `scripts/select_signal_aware_pseudos.py`
- `.../20260414_signal_enhancement_e1/reports/signal_aware_selection_report.json`
- `.../20260414_signal_enhancement_e1/manifests/signal_aware_selection_manifest.json`
- `.../20260414_signal_enhancement_e1/manifests/selection_summary.json`

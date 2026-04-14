# SIGNAL_DUAL_PSEUDO_E2_COMPARE_20260414.md

## 1. 这轮 E2 在做什么

这轮 E2 不再只给每个 gap 放 1 个 pseudo，而是把 E1 的 `{1/3, 1/2, 2/3}` 打分结果扩展成 `top2 per gap`。实际落地不是固定 `1/3 + 2/3`，而是按分数选前二，因此本 case 最终稳定选成了 `1/2 + 2/3`。

最终 16 帧为：
`[23, 17, 57, 51, 92, 86, 127, 121, 162, 156, 196, 190, 225, 231, 260, 266]`

相对 midpoint8：
`[17, 51, 86, 121, 156, 190, 225, 260]`

相对 E1 signal-aware-8：
`[23, 57, 92, 127, 162, 196, 225, 260]`

## 2. 本轮工程改动

代码改动：
- `scripts/select_signal_aware_pseudos.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`

主要变化：
1. `select_signal_aware_pseudos.py` 新增 `--topk-per-gap` 与 `--allocation-policy`，支持每个 gap 输出多于 1 个 pseudo；
2. selection manifest 现在会写入 `gap_index / allocation_rank / allocation_policy / candidate_fraction / selection_score / selection_label`；
3. `prepare_stage1_difix_dataset_s3po_internal.py` 新增 `--selection-manifest`，可直接消费 E1/E2 selection manifest；
4. pseudo-cache schema 升级到 `pseudo-cache-internal-v1.5`，sample/meta 层都保留 gap / allocation 元信息；
5. pack manifest 与 pseudo_cache manifest 都新增 `num_unique_gaps / allocation_policies`，便于后续区分单 pseudo 和多 pseudo 方案。

## 3. signal layer：E2 top2 vs midpoint8

比较口径：`selected_topk`（16 pseudo）对 `midpoint8`（8 pseudo）的 aggregate。

- `support_ratio_left`: `0.01233 -> 0.01161`
- `support_ratio_right`: `0.01019 -> 0.01099`
- `support_ratio_both`: `0.00754 -> 0.00741`
- `verified_ratio`: `0.01498 -> 0.01520`
- `continuous_confidence_mean_positive`: `0.69049 -> 0.69470`
- `agreement_mean_positive`: `0.75106 -> 0.75226`
- `mean_abs_rel_correction_verified`: `0.02051 -> 0.02089`

解读：
- E2 仍然保持了 E1 的方向：`verified_ratio / continuous confidence / agreement` 继续小幅正向；
- 但 `both-support` 仍未被抬高，说明“增加一个第二 pseudo”并没有把核心几何支持显著变强；
- 新增的第二 pseudo 多数是 midpoint，本质上更像“补一层中等质量 signal”，而不是再补一个强 signal。

## 4. densify layer：E2 vs E1.5

### 4.1 baseline densify
- midpoint8：`0.02269`
- E1 signal-aware-8：`0.02961`
- E2 dual-pseudo(top2)：`0.02621`

### 4.2 conf-aware densify
- midpoint8：`0.01634`
- E1 signal-aware-8：`0.01831`
- E2 dual-pseudo(top2)：`0.01737`

解读：
- E2 相比 midpoint8 仍然是正向的；
- 但相比 E1.5 已验收通过的 `signal-aware-8`，E2 在 baseline / conf-aware 两条 densify 线上都回落；
- 这说明“多一个 pseudo”带来的 coverage 增益，没有超过“第二个 pseudo 质量不如第一名”造成的稀释。

## 5. short refine layer：StageA-20

### baseline StageA-20
- E1 signal-aware-8：`loss_total 0.03659`, `loss_rgb 0.01029`, `loss_depth 0.08539`, `mean_trans 0.00109`
- E2 dual-pseudo(top2)：`loss_total 0.04555`, `loss_rgb 0.01328`, `loss_depth 0.10799`, `mean_trans 0.00118`

### conf-aware StageA-20
- E1 signal-aware-8：`loss_total 0.04340`, `loss_rgb 0.01702`, `loss_depth 0.09138`, `mean_trans 0.00120`
- E2 dual-pseudo(top2)：`loss_total 0.05815`, `loss_rgb 0.02589`, `loss_depth 0.11994`, `mean_trans 0.00127`

解读：
- 这轮 E2 没有把更多 pseudo 变成更好的 short refine；
- 相反，baseline / conf-aware 两条线的 loss 都明显高于 E1.5；
- 这说明当前 consumer 侧并没有从“每 gap 多一个中等质量 pseudo”里获得净收益，反而更像是被较弱样本拉低了整体 supervision 质量。

## 6. 当前结论

E2 的结论比较明确：
1. schema / manifest / pseudo_cache 的 multi-pseudo 支持已经打通；
2. `top2 per gap` 在当前 case 中稳定落成 `1/2 + 2/3`，而不是 `1/3 + 2/3`；
3. raw signal 层相对 midpoint8 仍有轻微正向，但幅度小；
4. densify 层相对 midpoint8 也有正向，但弱于 E1.5 的 `signal-aware-8`；
5. short refine 层则出现明显回落，说明这轮 E2 不能视为比 E1 更强的方案。

一句话压缩：**E2 把“多 pseudo”这条路工程上打通了，但当前这版 `top2 = 1/2 + 2/3` 并没有优于 E1 单 pseudo 最优选点；它更像增加了中等质量监督，而不是增加了高价值监督。**

## 7. 是否进入 E3（multi-anchor verify）

我的判断是：
- 不应把当前 E2 top2 方案直接当成新的默认 pseudo set；
- 如果马上做 E3，更合理的基底应是 **E1 通过正式对照的 `signal-aware-8`**，而不是 E2 的 16-pseudo set；
- E2 的主要价值是告诉我们：当前主矛盾不是“单纯 pseudo 数量还不够”，而更像是“高质量 pseudo 太少，低质量 pseudo 会稀释 supervision”。

因此下一步建议：
1. 默认 winner 仍保持 E1 `signal-aware-8`；
2. E3 若启动，应优先在 E1 winner 上做 `nearest2 -> nearest4` 的 verify 增强；
3. 若还想继续做 E2 分支，应改成更保守的变体，例如只在低把握 gap 上补第二个 pseudo，而不是所有 gap 一律 top2。

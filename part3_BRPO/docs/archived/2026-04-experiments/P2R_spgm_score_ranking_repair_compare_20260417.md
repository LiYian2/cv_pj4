# P2-R：SPGM score/ranking repair compare（2026-04-17）

> 目标：在 `P2-O` 已确认“selector-first 真实生效但 replay 变差”之后，按 `P2-Q` 的方案把 ranking 和 weighting 拆开，并回答：support-aware ranking 能否在不动 repair-A-style weighting 的前提下，把 far-only selector 拉回 toward repair A。

---

## 1. 一句话结论

ranking / weighting 拆分已经真实接通，而且 support-aware ranking 的方向是对的：
- far-only selector + 原始 ranking（v1）相对 control 仍然掉点；
- 把 ranking 改成 `support_blend` 后，结果有稳定小幅回升；
- 但这一轮回升还不够，最佳 support-blend arm 仍然低于 repair A control。

因此，当前结论不是“score/ranking repair 没用”，而是：
- 它已经比旧的 far-only v1 ranking 更好；
- 说明当前问题确实部分在 ranking；
- 但第一版 support-aware ranking 还不足以把 selector arm 拉回到 repair A 之上。

所以当前最合理的项目判断是：
- repair A 仍然是当前最好的 SPGM arm；
- score/ranking repair 是一条活路线，但现在还只是 partial repair；
- 下一步应继续沿“更保守 selector + support-aware ranking”往前走，而不是回去重新加大 selector 强度或跳到 stochastic。

---

## 2. 运行身份

### 2.1 运行根
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_p2r_spgm_score_ranking_repair_compare_e1`

### 2.2 代码身份
- repo commit：`6ad7bfd`
- 本轮运行在加入 ranking / weighting decouple 之后的工作树上执行

### 2.3 baseline / control / study arms
1. external baseline anchor（复用 `P2-L` / `P2-K`）
   - `canonical_baseline_post40_lr03_120`
2. in-tree control
   - `control_repair_a_dense_keep`
   - `dense_keep`
   - `ranking_mode=v1`
   - repair-A-style weighting：`support_eta=0.0 / weight_floor=0.25 / cluster_keep=(1,1,1)`
3. far-only selector, old ranking
   - `selector_far_only_v1_keep100_100_075`
   - `selector_quantile`
   - `ranking_mode=v1`
   - `selector_keep_ratio=(1.0, 1.0, 0.75)`
4. far-only selector, support-aware ranking
   - `selector_far_only_supportblend_l03_keep100_100_075`
   - `ranking_mode=support_blend`
   - `lambda_support_rank=0.3`
5. far-only selector, stronger support-aware ranking
   - `selector_far_only_supportblend_l05_keep100_100_075`
   - `ranking_mode=support_blend`
   - `lambda_support_rank=0.5`

---

## 3. 固定 protocol

本轮固定：
1. same canonical StageB protocol：`RGB-only v2 + gated_rgb0192 + post40_lr03_120`
2. same StageA.5 handoff：`stageA5_v2rgbonly_xyz_gated_rgb0192_80`
3. same sequential pseudo-state handoff：`init_pseudo_camera_states_json = pseudo_camera_states_final.json`
4. same pseudo cache / `signal_v2_root` / replay evaluator / real branch / seed
5. fixed repair-A-style weighting：`support_eta=0.0 / weight_floor=0.25 / cluster_keep=(1,1,1)`

study variable 只剩两个：
- selector 是否只裁 far cluster
- ranking 是否从旧 `v1` 换成 `support_blend`

---

## 4. 结果

### 4.1 主指标

| arm | PSNR | SSIM | LPIPS | ΔPSNR vs baseline | ΔPSNR vs control |
| --- | ---: | ---: | ---: | ---: | ---: |
| canonical baseline | `24.029824` | `0.872286` | `0.081774` | `0` | `+0.027748` |
| control repair A | `24.002077` | `0.871325` | `0.082440` | `-0.027748` | `0` |
| far-only v1 | `23.994081` | `0.871007` | `0.082650` | `-0.035743` | `-0.007996` |
| far-only support_blend λ=0.3 | `23.995370` | `0.871067` | `0.082617` | `-0.034454` | `-0.006707` |
| far-only support_blend λ=0.5 | `23.995545` | `0.871078` | `0.082606` | `-0.034279` | `-0.006532` |

### 4.2 最直接的结论

- far-only 本身仍然是负的：旧 ranking 下相对 control 掉 `-0.007996 PSNR`
- 但 support-aware ranking 确实把这个 gap 往回拉了：
  - λ=0.3：相对 far-only v1 回升 `+0.001289 PSNR`
  - λ=0.5：相对 far-only v1 回升 `+0.001464 PSNR`
- 最好 arm 是 `selector_far_only_supportblend_l05_keep100_100_075`
- 但它相对 control 仍然差 `-0.006532 PSNR`

所以：方向是对的，但还没到可替代 repair A 的程度。

---

## 5. 机制层观察

### 5.1 control 复现稳定

本轮 control 相对 `P2-O` 的 repair-A-style control 几乎完全重合：
- ΔPSNR = `+0.000129`
- ΔSSIM = `-3.06e-07`
- ΔLPIPS = `+9.42e-07`

因此本轮差异可以继续直接解释为 ranking / selector 变化，而不是 control drift。

### 5.2 ranking / weighting 解耦已经真实生效

control 的关键统计：
- `ranking_mode = v1`
- `selected_mean ≈ 0.729020`
- `ranking_mean ≈ 0.673999`

far-only selector 的关键统计：
- `selected_mean ≈ 0.668278`
- 说明这次确实只缩小了一部分 selected subset，而不是回到 mid+far 同时重砍的旧做法

support_blend 两臂的关键变化：
- `ranking_mean` 从 v1 的 `0.673987` 提到 `0.716731 / 0.745225`
- 但 `selected_mean` 基本保持不变（都约 `0.668278`）

这说明：
- 当前 improvement 不是因为 selector 又变弱了；
- 而是同样的 far-only keep ratio 下，排序本身变得更合理了。

### 5.3 这轮结果如何解读

`P2-R` 最关键的机制信息是：
- far-only selector 仍然会伤 replay，说明 selector 这条线还没完全站住；
- 但 support-aware ranking 已经能稳定追回一小部分损失；
- 所以当前问题不再像 `P2-O` 那样是“ranking 完全不 work”，而更像是“ranking repair 已开始起作用，但 selector 仍偏强 / 当前 score 仍不够好”。

换句话说，`P2-R` 把项目判断从：
- “先去修 ranking，看看是不是问题”
推进成了：
- “ranking 确实是问题的一部分，而且修它是有收益的；现在该继续做更保守的 selector + ranking 联合微调，而不是回头扩大 selector 强度。”

---

## 6. 当前可以下的工程判断

1. ranking / weighting decouple 已经成功落地，不是伪 plumbing。
2. `support_blend` ranking 比旧 v1 ranking 更好，这条 repair 路线有效。
3. 但当前最佳 support-blend far-only arm 仍低于 repair A control，因此 repair A 仍然是当前最优 SPGM anchor。
4. 当前不该回头做更强 selector，也不该直接跳 stochastic / `xyz+opacity`。
5. 下一步更合理的方向是：
   - 保留 `support_blend` ranking
   - 在更温和的 far keep ratio 上继续收敛（例如 `0.9 / 0.85 / 0.8`）
   - 必要时再小扫 `lambda_support_rank`

---

## 7. 建议的下一步

建议下一轮只做很小的 follow-up：
1. 固定 `ranking_mode=support_blend`
2. 固定当前较好的 `lambda_support_rank=0.5`
3. 不动 weighting 主体：仍然保持 repair-A-style weighting
4. 只扫 far keep ratio：
   - `far=0.90`
   - `far=0.85`
   - `far=0.80`

原因：
- `P2-R` 已经证明 ranking repair 是正方向；
- 当前更可能的问题是 far-only `0.75` 仍稍微过强；
- 所以最该先回答的是：support-aware ranking 配上更保守的 far-only selector，能不能把 gap 从 `-0.0065` 继续缩小到接近 0。

在这一步之前，不建议：
- 再回去扫 mid+far 联合更强 selector
- 直接上 stochastic
- 直接扩到 `xyz+opacity`
- 重新把主线切回 schedule / densify

---

## 8. 产物

- run root：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_p2r_spgm_score_ranking_repair_compare_e1`
- summary：`.../summary.json`
- markdown summary：`.../summary.md`
- arms：
  - `control_repair_a_dense_keep`
  - `selector_far_only_v1_keep100_100_075`
  - `selector_far_only_supportblend_l03_keep100_100_075`
  - `selector_far_only_supportblend_l05_keep100_100_075`

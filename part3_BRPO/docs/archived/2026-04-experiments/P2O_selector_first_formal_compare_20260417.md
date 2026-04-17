# P2-O：selector-first SPGM canonical StageB formal compare（2026-04-17）

> 目标：在完全复用 canonical `RGB-only v2 + gated_rgb0192 + post40_lr03_120` StageB protocol 的前提下，正式回答：在 `repair A` 之上引入 selector-first policy 后，replay 是否能继续优于 conservative deterministic control，并进一步逼近 canonical baseline。

---

## 1. 一句话结论

selector-first 的最小工程路径在 canonical StageB protocol 下已经被正式跑通，但当前 `selector_quantile` 结果并没有优于 repair A control。

更具体地说：
- control `repair A` 复现稳定：`24.00195 / 0.87133 / 0.08244`；
- selector S1（`keep=(1.0, 0.9, 0.75)`）降到 `23.99424 / 0.87099 / 0.08265`；
- selector S2（`keep=(1.0, 0.8, 0.6)`）进一步降到 `23.97587 / 0.87033 / 0.08304`；
- 两条 selector arm 都真实改变了 selected set，但 replay 随 selector 压力增强而单调变差。

因此，这一步的结论不是“selector-first 没接通”，而是：
- selector-first 机制已经真实生效；
- 但当前 importance ranking / selection criterion 还不够好；
- 在现阶段，`repair A` 仍然是当前最好的 SPGM anchor；
- 下一步不该继续加大 selector 强度，而应先修 score / ranking，再做更保守的 selective keep。

---

## 2. 运行身份

### 2.1 运行根
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_p2o_selector_first_formal_compare_e1`

### 2.2 代码身份
- repo commit：`6ad7bfd`
- 当前 compare 在已有 selector-capable 工作树上执行；summary 中已保存 `git status --short`

### 2.3 对照臂
1. external baseline anchor（复用 `P2-L` / `P2-K`）
   - `canonical_baseline_post40_lr03_120`
2. in-tree control（本轮重跑）
   - `control_repair_a_keep111_eta0_wf025`
   - `dense_keep`
   - `cluster_keep=(1.0, 1.0, 1.0)`
   - `support_eta=0.0`
   - `weight_floor=0.25`
3. selector S1
   - `selector_quantile_s1_keep100_090_075_eta0_wf025`
   - `selector_keep_ratio=(1.0, 0.9, 0.75)`
   - 其余 weighting 与 repair A 相同
4. selector S2
   - `selector_quantile_s2_keep100_080_060_eta0_wf025`
   - `selector_keep_ratio=(1.0, 0.8, 0.6)`
   - 其余 weighting 与 repair A 相同

---

## 3. 固定 protocol

本轮与 canonical bounded StageB baseline 对齐到以下层级：
1. same StageA.5 handoff：`stageA5_v2rgbonly_xyz_gated_rgb0192_80`
2. same sequential handoff：`init_pseudo_camera_states_json = pseudo_camera_states_final.json`
3. same StageB schedule：`stageB_iters=120`，`stageB_post_switch_iter=40`，`stageB_post_lr_scale_xyz=0.3`
4. same upstream signal/view gate：`signal_pipeline=brpo_v2`，same `signal_v2_root`，same `pseudo_cache`，`min_rgb_mask_ratio=0.0192`
5. same replay evaluator：`replay_internal_eval.py`
6. same real branch / seed：`lambda_real=lambda_pseudo=1.0`，`num_real_views=2`，`num_pseudo_views=4`，`seed=0`

study variable 只有一个：在 repair-A-style weighting 上，是否做 cluster-wise selector-first keep，以及 keep ratio 多强。

---

## 4. 结果

### 4.1 主指标

| arm | PSNR | SSIM | LPIPS | ΔPSNR vs baseline | ΔPSNR vs control |
| --- | ---: | ---: | ---: | ---: | ---: |
| canonical baseline | `24.029824` | `0.872286` | `0.081774` | `0` | `+0.027877` |
| control repair A | `24.001947` | `0.871325` | `0.082439` | `-0.027877` | `0` |
| selector S1 | `23.994236` | `0.870991` | `0.082654` | `-0.035588` | `-0.007711` |
| selector S2 | `23.975866` | `0.870325` | `0.083041` | `-0.053958` | `-0.026081` |

### 4.2 control 复现稳定性

本轮 control 与 `P2-M` 的旧 repair A 几乎完全重合：
- ΔPSNR = `-0.000172`
- ΔSSIM = `+0.00000157`
- ΔLPIPS = `-0.00000111`

因此这轮负结果不是 control 漂移造成的，selector arm 与 repair A 的差异可以直接按本轮结果解读。

---

## 5. 机制层观察

### 5.1 active set 与 rejection 没变，但 selected set 确实变了

三条新臂都保持：
- `iters_with_rejection = 96 / 120`
- `total_rejected_sample_evals = 116`
- `unique_rejected_ids = [225, 260]`
- `spgm_active_ratio_mean ≈ 0.7290`

但 selected ratio 已明显变化：
- control：`selected_mean = 0.729019`（与 active 一致，符合 `dense_keep`）
- selector S1：`selected_mean = 0.643992`
- selector S2：`selected_mean = 0.583241`

并且 `grad_keep_ratio_xyz_mean` 与 `spgm_selected_ratio_mean` 对齐，说明这次 selector-first 不是伪字段，而是真的把更新子集缩小了。

### 5.2 stronger selection → replay 单调变差

随着 keep ratio 变强：
- `selected_mean`：`0.7290 -> 0.6440 -> 0.5832`
- `grad_weight_mean_xyz_mean`：`0.5497 -> 0.4961 -> 0.4546`
- replay PSNR：`24.00195 -> 23.99424 -> 23.97587`

也就是说，这次不是“selector 没起作用所以结果不动”，而是 selector 起作用了，但当前 ranking 把有用更新削掉了。

### 5.3 selector 仍优于原始 SPGM v1，但不如 repair A

相对 `P2-L` 的原始 canonical `spgm_keep`（`23.94230 / 0.86956 / 0.08359`）：
- selector S1 仍有一定回升；
- selector S2 也没有退回到原始 SPGM 那么差；

但这两条都没有超过 repair A control，说明当前真正的瓶颈已经从“是否需要 selector plumbing”进一步转成“当前 score/ranking 是否足够可信，值得做 hard select”。

---

## 6. 当前可以下的工程判断

1. selector-first formal compare 已完成，且结论为负。
2. 这不是 plumbing failure：`selector_quantile` 已真实改变 selected set。
3. 这也不是 control drift：repair A 本轮重跑与 `P2-M` 几乎重合。
4. 当前最好的 SPGM arm 仍然是 `repair A`，不是 selector S1/S2。
5. 因而下一步不应继续直接加大 selector 强度，也不应立刻扩到 top-k / stochastic / `xyz+opacity`。
6. 更合理的下一步应转到：先修 importance score / ranking，再做更保守的 selective keep。

---

## 7. 建议的下一步

当前最小且最有诊断价值的下一步应是：
1. 保留 `repair A` 作为 SPGM control；
2. 不再动 weighting 主体（仍用 `support_eta=0.0 / weight_floor=0.25 / cluster_keep=(1,1,1)`）；
3. 把 selector 的改动收窄到 score/ranking 侧，而不是继续直接压 keep ratio；
4. 第一优先考虑更保守的 score-aware / far-cluster-only selector，而不是继续让 mid/far 同时大幅裁剪；
5. 在这一步站住之前，不推进 stochastic、`xyz+opacity`、更长 iter，仍不做 raw RGB densify。

---

## 8. 产物

- run root：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_p2o_selector_first_formal_compare_e1`
- summary：`.../summary.json`
- markdown summary：`.../summary.md`
- arms：
  - `control_repair_a_keep111_eta0_wf025`
  - `selector_quantile_s1_keep100_090_075_eta0_wf025`
  - `selector_quantile_s2_keep100_080_060_eta0_wf025`

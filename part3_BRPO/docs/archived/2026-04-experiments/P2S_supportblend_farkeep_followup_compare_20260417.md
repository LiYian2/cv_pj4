# P2-S：support_blend far-keep follow-up compare（2026-04-17）

> 目标：在 `P2-R` 已确认 `support_blend` ranking 有效、但 `far keep=0.75` 仍低于 repair A 之后，只固定一件事：`ranking_mode=support_blend, lambda_support_rank=0.5, repair-A-style weighting` 不变，只扫更保守的 far-only keep ratio，回答 selector-first 能否逼近或追平 repair A。

---

## 1. 一句话结论

这轮 follow-up 已把问题进一步收窄：
- `support_blend + far keep=0.90` 已把 selector-first arm 拉到与 repair A 基本持平；
- 它在 PSNR 上对 control 只有一个极小的正差（`+0.000069`），但 SSIM / LPIPS 仍是几乎可忽略的小负偏；
- `0.85 / 0.80` 继续显示出“selector 越强，replay 越差”的单调趋势，不过它们都明显好于 `P2-R` 的 `far=0.75`。

因此当前最准确的项目判断是：
- repair A 仍然是当前最稳的 SPGM anchor；
- 但 `support_blend + far keep=0.90` 已经是第一个真正达到“practical parity”的 selector arm；
- 现在不该再往更强 selector 方向走，而应围绕 `0.90` 做窄范围确认 / 精修，看这个近乎打平的结果是否稳定、以及是否还能在极保守区间里获得一致的小幅正增益。

---

## 2. 运行身份

### 2.1 运行根
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_p2s_supportblend_farkeep_followup_compare_e1`

### 2.2 代码身份
- repo commit：`4a1cc10`
- 本轮运行在 `P2-Q / P2-R` 的 decoupled ranking / weighting 工作树上继续执行

### 2.3 baseline / control / study arms
1. external baseline anchor（复用 `P2-L` / `P2-K`）
   - `canonical_baseline_post40_lr03_120`
2. in-run control
   - `control_repair_a_dense_keep`
   - `dense_keep`
   - repair-A-style weighting：`support_eta=0.0 / weight_floor=0.25 / cluster_keep=(1,1,1)`
3. fixed selector configuration
   - `policy_mode=selector_quantile`
   - `ranking_mode=support_blend`
   - `lambda_support_rank=0.5`
4. far-only keep sweep
   - `selector_far_only_supportblend_l05_keep100_100_090`
   - `selector_far_only_supportblend_l05_keep100_100_085`
   - `selector_far_only_supportblend_l05_keep100_100_080`
5. external selector reference（复用 `P2-R`）
   - `selector_far_only_supportblend_l05_keep100_100_075`

---

## 3. 固定 protocol

本轮固定：
1. same canonical StageB protocol：`RGB-only v2 + gated_rgb0192 + post40_lr03_120`
2. same StageA.5 handoff：`stageA5_v2rgbonly_xyz_gated_rgb0192_80`
3. same sequential pseudo-state handoff：`init_pseudo_camera_states_json = pseudo_camera_states_final.json`
4. same pseudo cache / `signal_v2_root` / replay evaluator / real branch / seed
5. same repair-A-style weighting：`support_eta=0.0 / weight_floor=0.25 / cluster_keep=(1,1,1)`
6. same ranking config：`ranking_mode=support_blend`, `lambda_support_rank=0.5`

study variable 只剩一个：
- far cluster keep ratio = `0.90 / 0.85 / 0.80`

---

## 4. 结果

### 4.1 主指标

| arm | PSNR | SSIM | LPIPS | ΔPSNR vs baseline | ΔPSNR vs control | ΔPSNR vs P2-R far=0.75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| canonical baseline | `24.029824` | `0.872286` | `0.081774` | `0` | `+0.027839` | `+0.034279` |
| control repair A | `24.001985` | `0.871321` | `0.082440` | `-0.027839` | `0` | `+0.006440` |
| far-only support_blend `0.90` | `24.002054` | `0.871316` | `0.082447` | `-0.027770` | `+0.000069` | `+0.006509` |
| far-only support_blend `0.85` | `24.000287` | `0.871261` | `0.082486` | `-0.029537` | `-0.001697` | `+0.004742` |
| far-only support_blend `0.80` | `23.998810` | `0.871191` | `0.082533` | `-0.031014` | `-0.003174` | `+0.003265` |
| P2-R best selector `0.75` | `23.995545` | `0.871078` | `0.082606` | `-0.034279` | `-0.006532` | `0` |

### 4.2 最直接的结论

- `far=0.90` 是当前最佳 selector arm；
- 它对 control 的差异已经缩到几乎可视为 parity：
  - `ΔPSNR = +0.000069`
  - `ΔSSIM = -0.000005`
  - `ΔLPIPS = +0.000007`
- `far=0.85 / 0.80` 仍低于 control，但都明显高于 `P2-R` 的 `far=0.75`：
  - `0.85` 相对 `0.75` 回升 `+0.004742 PSNR`
  - `0.80` 相对 `0.75` 回升 `+0.003265 PSNR`

所以：
- `support_blend` 方向不是只带来微弱修补，而是与更保守 selector 联动后，已经把 selector-first 从“明确负结果”推到了“几乎打平 repair A”的区间；
- 但当前还不能草率宣布 selector-first 已正式赢过 repair A，因为 `0.90` 的优势只出现在极小的 PSNR 正差上，还没有形成跨指标的一致正胜。

---

## 5. 机制层观察

### 5.1 control 继续稳定复现

本轮 control 与 `P2-R` 的 repair A control 几乎完全重合：
- `ΔPSNR = -0.000092`
- `ΔSSIM = -0.000003`
- `ΔLPIPS = -0.0000004`

所以本轮差异仍然可以继续解释为 selector keep ratio 的变化，而不是 control drift。

### 5.2 改善主要来自 selector 变得更保守，而不是 ranking 再漂移

从 `P2-R far=0.75` 到 `P2-S far=0.80 / 0.85 / 0.90`：
- `spgm_ranking_score_mean` 基本稳定在 `~0.74522`
- `spgm_support_norm_mean` 也基本稳定在 `~0.81646`
- 真正系统变化的是 `spgm_selected_ratio_mean`：
  - `0.75 -> 0.668278`
  - `0.80 -> 0.680431`
  - `0.85 -> 0.692585`
  - `0.90 -> 0.704733`

这说明：
- 当前 improvement 的主因不是 ranking 突然更强，而是同一套 support-aware ranking 在更保守 selector 下删掉了更少的 useful far Gaussians；
- 换句话说，`P2-R` 暴露出来的 residual gap，当前更像是“selector 强度仍过头”，而不是“ranking repair 本身失败”。

### 5.3 趋势非常清楚：keep 越保守，结果越好

在 `support_blend λ=0.5` 固定后，当前 replay 对 far keep ratio 的响应是单调的：
- `0.75 < 0.80 < 0.85 < 0.90`

这条趋势本身就很重要，因为它把下一步搜索方向从：
- “再往更强 selector 做更多花样”
收窄成了：
- “如果还要继续 selector-first，就只能往更温和、更接近 control 的区间里找”

---

## 6. 当前可以下的工程判断

1. `support_blend + conservative far-only selector` 已经不再是负结果集合；至少 `far=0.90` 已经把 selector arm 拉到与 repair A practical parity 的区间。
2. 但当前最佳 arm 还没有形成跨指标的一致领先，因此 repair A 仍应保留为当前最稳的 SPGM anchor。
3. `P2-S` 的最重要信息不是“selector-first 已经赢了”，而是：当前 selector-first 的可行区间确实存在，而且它明显位于非常保守的 far keep 一侧。
4. 因而下一步不应回头做更强 selector，也不应在当前节点直接跳 stochastic / `xyz+opacity` / 更长 iter。
5. 当前最合理的 follow-up 是：围绕 `far=0.90` 做窄范围确认 / 精修，而不是再回到 `0.85 / 0.80` 以下的更强区域。

---

## 7. 建议的下一步

建议下一轮只做一个很小的 confirmation/precision sweep：
1. 保留 `post40_lr03_120` 作为 canonical bounded StageB baseline；
2. 保留 repair A 作为 control；
3. 固定 `ranking_mode=support_blend`, `lambda_support_rank=0.5`；
4. 不再往更强方向扫；改为在 `far=0.90` 附近做窄范围确认，例如：
   - repeat `far=0.90` 一次确认稳定性；
   - 再试 `far=0.92 / 0.95`（必要时加 `0.88`）
5. 决策规则：只有当 selector arm 在重复运行里对 repair A 形成稳定、跨指标至少不劣的结果，才考虑把“当前最佳 selector arm”升级为新 anchor。

在这一步之前，不建议：
- 回去继续做更强 far-only selector；
- 重新启用 mid+far 联合强裁剪；
- 直接上 stochastic；
- 直接扩到 `xyz+opacity`。

---

## 8. 产物

- run root：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_p2s_supportblend_farkeep_followup_compare_e1`
- summary：`.../summary.json`
- markdown summary：`.../summary.md`
- arms：
  - `control_repair_a_dense_keep`
  - `selector_far_only_supportblend_l05_keep100_100_090`
  - `selector_far_only_supportblend_l05_keep100_100_085`
  - `selector_far_only_supportblend_l05_keep100_100_080`


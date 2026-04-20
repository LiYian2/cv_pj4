# P2-T：selector-first confirmation / precision sweep（2026-04-18）

> 目标：在 `P2-S` 已经把 `support_blend + far=0.90` 推到 practical parity 之后，只做一个很小的确认：固定 canonical StageB protocol、固定 repair-A-style weighting、固定 `support_blend, lambda=0.5`，只在 far-only 极保守区间做 precision sweep，回答 selector-first 是否已经足以升级为新 anchor。

---

## 1. 一句话结论
这轮 confirmation / precision sweep 没有把 selector-first 推成新 winner，但把结论进一步钉实了：
1. selector-first 的确已经进入 practical-parity 区，而不是回到明显负结果
2. 但在本轮 `0.88 / 0.90-repeat / 0.92 / 0.95` 微扫里，所有 selector arm 都仍略低于 in-run repair A control
3. 最好的 PSNR / LPIPS 仍在 `0.90-repeat`，最好的 SSIM 在 `0.92`，说明真正值得保留的只剩 `0.90 ~ 0.92` 这一极窄窗口
4. 因而当前还不能把 selector-first 升级为新 anchor；repair A 继续保留为主 control
5. 新主线不再继续磨 far-keep sweep，而是转去做 A1：unified RGB-D joint confidence

---

## 2. 运行身份

### 2.1 运行根
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260418_p2t_selector_confirmation_precision_compare_e1`

### 2.2 代码身份
- repo commit：`4a1cc10`

### 2.3 固定 protocol
1. same canonical StageB protocol：`RGB-only v2 + gated_rgb0192 + post40_lr03_120`
2. same StageA.5 handoff：`stageA5_v2rgbonly_xyz_gated_rgb0192_80`
3. same sequential pseudo-state handoff：`init_pseudo_camera_states_json = pseudo_camera_states_final.json`
4. same repair-A-style weighting：`support_eta=0.0 / weight_floor=0.25 / cluster_keep=(1,1,1)`
5. same selector ranking：`ranking_mode=support_blend`, `lambda_support_rank=0.5`

### 2.4 arms
1. `control_repair_a_dense_keep`
2. `selector_far_only_supportblend_l05_keep100_100_088`
3. `selector_far_only_supportblend_l05_keep100_100_090_repeat`
4. `selector_far_only_supportblend_l05_keep100_100_092`
5. `selector_far_only_supportblend_l05_keep100_100_095`

---

## 3. 结果

### 3.1 主指标

| arm | PSNR | SSIM | LPIPS | ΔPSNR vs control | ΔPSNR vs old 0.90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| control repair A | `24.001974` | `0.871323` | `0.082439` | `0` | `-0.000080` |
| far-only `0.88` | `24.001310` | `0.871291` | `0.082464` | `-0.000663` | `-0.000744` |
| far-only `0.90` repeat | `24.001879` | `0.871314` | `0.082446` | `-0.000095` | `-0.000176` |
| far-only `0.92` | `24.001872` | `0.871316` | `0.082449` | `-0.000102` | `-0.000182` |
| far-only `0.95` | `24.001584` | `0.871315` | `0.082448` | `-0.000390` | `-0.000471` |

### 3.2 最直接的结论
1. `0.90-repeat` 仍是当前最佳 PSNR / LPIPS selector arm，但它相对 control 仍是一个极小负差：
   - `ΔPSNR = -0.000095`
   - `ΔSSIM = -0.000009`
   - `ΔLPIPS = +0.000007`
2. `0.92` 的 SSIM 最好，但 PSNR / LPIPS 仍未超过 control
3. `0.88` 明显比 `0.90 / 0.92` 更差，说明往更强 selector 方向回退并不值得
4. `0.95` 也没有把结果再往上抬，说明“更保守就更好”的单调趋势到这里已经基本停住了

因此：
- selector-first 的可行窗口真实存在，但它没有在本轮里越过 repair A control
- 这条线可以保留为“已逼近”的参考臂，但还不够资格接管 anchor 身份

---

## 4. 机制层观察

### 4.1 selector 行为是真实的
- control `selected_ratio_mean ≈ 0.729018`
- `0.88 ≈ 0.699875`
- `0.90 ≈ 0.704732`
- `0.92 ≈ 0.709596`
- `0.95 ≈ 0.716881`

说明 selector-first 不是 no-op，selected set 在稳定变化。

### 4.2 ranking 仍然稳定
- 各 selector arm 的 `spgm_ranking_score_mean` 与 `spgm_support_norm_mean` 基本稳定
- 当前差异主要仍来自 far-only keep ratio 的微调，而不是 ranking 再次漂移

### 4.3 这轮 sweep 的真正信息量
这轮不是在寻找“大赢臂”，而是在回答：
- `0.90` 是偶然贴近，还是一个可复现微窗口？

答案更接近：
- `0.90 ~ 0.92` 的确是稳定微窗口
- 但它目前还是“贴近 control”，不是“越过 control”

---

## 5. 当前工程判断
1. repair A 继续保留为当前主 control / anchor
2. selector-first 继续保留为一个“已逼近但未越过”的候选方向
3. 如果以后回到 selector-first，只值得回到 `0.90 ~ 0.92` 的极窄窗口；不再继续扫更强 far-keep，也不再扫更宽区间
4. 当前更合理的主工程动作是转去做 A1：unified RGB-D joint confidence

---

## 6. 下一步
1. 开始 A1：`docs/A1_unified_rgbd_joint_confidence_engineering_plan.md`
2. A1 如果证明 unified observation 有方向感，再推进 A2
3. 在 A1/A2 之前，不继续做新的 far-keep sweep
4. stochastic / manager / deterministic state action 仍然排在 A1/A2 之后

---

## 7. 产物
1. 运行脚本：`tmp_run_p2t_selector_confirmation_precision_20260418.sh`
2. summary：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260418_p2t_selector_confirmation_precision_compare_e1/summary.json`
3. markdown summary：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260418_p2t_selector_confirmation_precision_compare_e1/summary.md`

# P2-D：signal gate 阈值 / 统计诊断（2026-04-16）

## 1. 目标

回答两个问题：
1. 为什么 current threshold 下 `StageA.5 / StageB` 都持续 `0 rejection`？
2. 下一步应该先调 gate 阈值，还是先把同一版 gating 挂到 `RGB-only v2` 上做短对照？

## 2. 诊断输入

代码侧检查：
- `pseudo_branch/local_gating/gating_schema.py`
- `pseudo_branch/local_gating/signal_gate.py`
- `scripts/run_pseudo_refinement_v2.py`

实验侧输入：
- `20260415_p2b_stageA5_local_gating_compare_e1/stageA5_legacy_xyz_gated_80/stageA_history.json`
- `20260415_p2c_stageB_local_gating_compare_e1/stageB_from_stageA5gated_gated_20/stageB_history.json`
- `20260415_signal_compare_stageAonly_e1/{stageA_legacy_80,stageA_v2_rgbonly_80,stageA_v2_full_80}/stageA_history.json`

## 3. 代码口径先确认

当前 hard gate 的判定来自 sampled pseudo view 上的静态 signal meta，而不是当前 iter 的在线 loss：
- `target_depth_verified_ratio`
- `rgb_confidence_nonzero_ratio`
- `target_depth_render_fallback_ratio`
- `mean_abs_rel_correction_verified`（当前默认阈值为 0，不参与）

当前默认阈值是：
- `min_verified_ratio = 0.01`
- `min_rgb_mask_ratio = 0.01`
- `max_fallback_ratio = 0.995`
- `min_correction = 0.0`

因此只要 sampled pseudo view 的静态 meta 始终高于这组阈值，就会持续 `0 rejection`；这不是 runtime bug，而是阈值本身过松。

## 4. 为什么 legacy StageA.5 / StageB 会持续 0 rejection

### 4.1 StageA.5 gated run（80 iter, 4 pseudo / iter）

统计自 `stageA_history.json` 的 `sample_signal_metrics`：
- `verified_ratio`: min `0.01890`, p50 `0.04469`, mean `0.04155`, max `0.05170`
- `rgb_mask_ratio`: min `0.14682`, p50 `0.18870`, mean `0.18989`, max `0.22046`
- `fallback_ratio`: min `0.94830`, p50 `0.95531`, mean `0.95845`, max `0.98110`

和当前阈值对比：
- `min_verified_ratio=0.01` 明显低于所有 sampled legacy view 的最小值 `0.01890`
- `min_rgb_mask_ratio=0.01` 更是远低于 legacy 的最小 RGB ratio `0.14682`
- `max_fallback_ratio=0.995` 又高于所有 sampled legacy view 的最大 fallback `0.98110`

所以 current threshold 在 legacy 上几乎等于 no-op，`0 rejection` 是结构上必然，不是偶然。

### 4.2 StageB gated run（20 iter, 4 pseudo / iter）

StageB 的 sampled 分布和 StageA.5 基本同一档：
- `verified_ratio`: min `0.01890`, mean `0.04116`, max `0.05170`
- `rgb_mask_ratio`: min `0.14682`, mean `0.18991`, max `0.22046`
- `fallback_ratio`: min `0.94830`, mean `0.95884`, max `0.98110`

这意味着：只要沿用 current threshold，StageB 一样会继续 `0 rejection`。

## 5. 如果真的想让 legacy gate 开始筛 weak pseudo，阈值应该在哪个区间

对 sampled legacy view 做阈值回放后，结果很清楚：
- 当 `min_verified_ratio <= 0.015`：仍然 `0 rejection`
- 当 `min_verified_ratio = 0.020`：开始只拒掉最弱那一档（主要是 sample `260`）
- 当 `min_verified_ratio = 0.030`：会稳定拒掉 `225 / 260` 这两档弱 view；StageA.5 中约 `74/320` sample eval 被拒，覆盖 `59/80` iter；StageB 中约 `22/80` sample eval 被拒，覆盖 `18/20` iter
- 当 `min_verified_ratio >= 0.040`：开始过强，几乎每轮都会拒掉较大比例 pseudo view

等价地看 fallback：
- `max_fallback_ratio = 0.995` 也几乎是 no-op
- legacy 上真正有区分度的是 `0.97 ~ 0.98` 这一带
- 但由于 fallback 与 verified 基本互补，legacy 上主要控制量其实还是 `verified_ratio`

另一个重要结论是：legacy 口径下 `rgb_mask_ratio` 根本不是有效判别轴。因为它全在 `0.1468 ~ 0.2205`，远离默认 `0.01`，即使把阈值提高到 `0.05`，仍然全通过。也就是说，如果继续在 legacy 上做第一轮 threshold calibration，真正该调的是 `verified/fallback`，不是 RGB 阈值。

## 6. 为什么不能直接把同一版 gating 挂到 RGB-only v2 上做短对照

从 `P1A` 的 `stageA_v2_rgbonly_80` 可以看到，`RGB-only v2` 的统计口径和 legacy 完全不是一个量级：
- legacy `rgb_confidence_nonzero_ratio`: mean 约 `0.1887`
- `v2-rgb-only` `rgb_confidence_nonzero_ratio`: mean 约 `0.01958`

但 `verified_ratio` 仍沿用 legacy depth target，所以还是 mean 约 `0.04134`。

这带来两个直接后果：
1. 如果直接沿用 current threshold（`0.01 / 0.01 / 0.995`），`RGB-only v2` 也大概率仍是 `0 rejection`；也就是说，先挂过去并不能回答更多问题。
2. 如果不做校准、只把 `min_rgb_mask_ratio` 从 `0.01` 往上抬，`RGB-only v2` 会立刻进入过敏区：
   - `min_rgb_mask_ratio = 0.020` 时已经会拒掉 `6/8` 个 pseudo view；
   - `0.025` 时会直接 `8/8` 全拒。

因此 `RGB-only v2` 不是不能挂 gating，而是它需要自己的阈值标定口径；把 legacy 阈值原样搬过去没有可解释性。

## 7. 当前最合理的决策

结论很明确：**下一步应该先做 signal gate 阈值重标定，而不是先把 current-threshold gating 挂到 `RGB-only v2` 上跑短对照。**

原因不是偏好 legacy，而是：
- current threshold 在 legacy 和 `RGB-only v2` 上都还没有进入“有筛选力”的工作区；
- legacy 的 `verified_ratio` 分布有足够动态范围，适合先做第一轮阈值标定；
- `RGB-only v2` 的 `rgb_confidence_nonzero_ratio` 比 legacy 缩小约一个数量级，必须 branch-specific 标定，否则 compare 没有解释力。

## 8. 推荐下一步实验顺序

### 8.1 先在 legacy 上做第一轮阈值标定

建议先只动一条主轴：
- 保持 `pseudo_local_gating=hard_visible_union_signal`
- 保持 `params=xyz`
- 保持 `min_rgb_mask_ratio=0.01`
- 保持 `min_correction=0.0`
- 优先扫 `min_verified_ratio ∈ {0.02, 0.03}`
- `max_fallback_ratio` 暂时保持宽松，或只做与 verified 等价的一组（例如 `0.98 / 0.97`）用于核对，不要一开始双轴大扫

原因：
- `0.02` 会先触碰最弱 pseudo，不至于一下把 gate 拉得太狠；
- `0.03` 能更稳定地区分 `225 / 260` 这一档弱 pseudo；
- 这两档足以回答“真正开始拒 weak pseudo 后，replay / loss 会不会出现可解释收益”。

### 8.2 等 legacy 阈值进入有效区后，再决定是否挂到 RGB-only v2

只有在这之后，再做下面的问题才有意义：
- 同一套“已进入有效区”的 gate 逻辑，在 `RGB-only v2` 上是否更匹配 BRPO-style mask 语义？
- 还是说 `RGB-only v2` 需要单独一套归一化 / branch-specific threshold？

## 9. 一句话结论

`0 rejection` 的原因已经找到：不是 P2 实现没生效，而是当前默认 gate 阈值对 legacy 来说基本等于 no-op；而 `RGB-only v2` 的 RGB ratio 又和 legacy 不同量纲，所以不能在没做阈值重标定前直接拿 current-threshold gating 去做可解释 compare。

# P2-P：post-P2O SPGM score / selector repair plan（2026-04-17）

> 目标：在 `P2-O` 已确认“selector-first 机制生效但 replay 变差”之后，把下一步从“继续加 selector 强度”收束成更小、更可诊断的 score/ranking repair。

---

## 1. 当前冻结结论

`P2-O` 已固定三件事：
1. repair A 仍是当前最好的 SPGM arm；
2. `selector_quantile` 已真实改变 selected set，不是 plumbing failure；
3. stronger selection 会让 replay 单调变差，因此当前瓶颈已经更像 score/ranking 不够好，而不是“还没做 hard select”。

因此，下一步不再继续直接扫更强 keep ratio，也不先上 stochastic / `xyz+opacity` / 更长 iter。

---

## 2. 下一步的核心问题

现在真正需要回答的是：

> 当前 quantile selector 的负结果，到底是因为“selection 本身就不该做”，还是因为“当前 importance score 还不够好，导致 hard select 选错了 Gaussian”？

这决定了后面是：
- 继续沿 selector 路线修 score；
- 还是把 SPGM 暂时收敛在 repair A，不再扩大 selector 支线。

---

## 3. 推荐执行顺序

### P0. 冻结当前 control
- control 固定为 `spgm_repair_a_keep111_eta0_wf025`
- baseline 继续固定为 canonical `post40_lr03_120 / gated_rgb0192`
- 在 score/ranking 问题没回答前，不再改 weighting 主体

### P1. 先做更保守的 selector 诊断，而不是更强 selector
第一优先建议：只在 far cluster 上做 select，near / mid 保持 dense keep。

建议第一组诊断臂：
1. `far_only_s1 = (1.0, 1.0, 0.75)`
2. `far_only_s2 = (1.0, 1.0, 0.60)`

理由：
- `P2-O` 已表明当前 `mid + far` 同时裁剪会伤 replay；
- 如果只裁 far 仍然伤 replay，就更能说明问题在 score/ranking，而不是 selector 施加位置不对；
- 如果 far-only 能接近或超过 repair A，说明之前的伤害主要来自 mid cluster 被误杀。

### P2. 若 far-only 仍为负，再进入 score repair
最小 score repair 应优先考虑：
- 把 support / visibility 强相关信息更直接地并入 ranking；
- 保持 selector policy 形状不变，只改 importance ordering；
- 先做 deterministic score mixing，不上 stochastic。

### P3. 只有 score repair 站住后，才考虑更宽 selector
只有在以下条件满足后，才值得继续谈：
- top-k / 二段式 support-aware select
- stochastic drop
- `xyz+opacity`
- 更长 iter

---

## 4. 验收标准

### 机制验收
1. `spgm_selected_ratio_mean < spgm_active_ratio_mean`
2. `grad_keep_ratio_xyz_mean` 与 `selected_ratio_mean` 对齐
3. rejection 统计可解释，不出现无意义的大幅漂移

### 结果验收
1. 新 arm 至少不能明显差于 repair A（建议门槛：PSNR 不低于 `-0.005`）
2. 若要算真正正向，需相对 repair A 出现明确 replay 增益
3. 若 stronger selector 仍单调更差，则停止继续加 selector 强度，转成 score-side 主修

---

## 5. 当前明确不该先做的事

- 不继续直接扫更强的 `(mid, far)` keep ratio
- 不在 selector-first 仍为负时直接上 stochastic
- 不把问题重新甩回 StageB schedule
- 不在 selector-first 尚未站住前推进 `xyz+opacity`
- 不因为 selector-first 为负就马上回头做 raw RGB densify

---

## 6. 一句话执行口径

`P2-O` 之后，最合理的下一步不是“继续加 selector”，而是：保留 repair A 为 control，先用 far-only 的更保守 selector 诊断 score/ranking 是否可信；若仍为负，再把主修点明确切到 importance score 本身。

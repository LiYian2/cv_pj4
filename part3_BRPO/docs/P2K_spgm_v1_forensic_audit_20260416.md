# P2-K：SPGM v1 深度审查与回归取证（2026-04-16）

> 审查目标：沿代码链路和实验链路确认这轮 SPGM v1 到底是“实现错了”，还是“方案/协议本身有问题”，并给出后续修正顺序。

---

## 1. 一句话结论

这轮 SPGM v1 不是“完全没做对”，而是出现了两层问题：

1. 代码层面：主 wiring 基本是对的，SPGM 的 grad modulation 确实真实生效；但仍有几个小实现坑/未完成项。
2. 实验层面：当前 `20260416_spgm_v1_*` compare 不是对 `post40_lr03_120 / gated_rgb0192` 这个 canonical bounded StageB baseline 的 apples-to-apples 对照，protocol 已经漂了。
3. 方案层面：即便只在这轮自有 protocol 内看，当前 SPGM v1 也更像“对同一 active set 做强连续压制”，而不是“更聪明地筛选/保护 Gaussian”，所以 replay 退化并不奇怪。

因此当前最合理的判断不是“直接宣判 SPGM 失败”，也不是“先去做 stochastic drop / 更长 iter”，而是：
- 先把 compare protocol 拉回 canonical baseline；
- 再做一轮 conservative SPGM（减弱压制、去掉固定 depth-tier 衰减）；
- 同时修掉代码里几个会误导后续判断的小坑。

---

## 2. 冻结这轮 run identity
### 2.1 代码版本
- repo commit：`6ad7bfd`

### 2.2 当前实际跑到的 SPGM 目录
- StageA.5 short compare：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_spgm_v1_stageA5_compare_e1/`
- StageB 40iter compare：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_spgm_v1_stageB_compare_e1/`
- StageB 120iter + replay：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_spgm_v1_stageB_compare_120iter/`

### 2.3 当前实际输入
- PLY：`after_opt/point_cloud/point_cloud.ply`
- pseudo_cache：`20260414_signal_enhancement_e15_compare/pseudo_cache_baseline`
- signal_v2：`20260414_signal_enhancement_e15_compare/signal_v2`
- pseudo ids：仍是 signal-aware-8（`23/57/92/127/162/196/225/260`）

### 2.4 当前 StageB-120 compare 的真实协议
- `stage_mode=stageB`
- `signal_pipeline=brpo_v2`
- `stageA_depth_loss_mode=legacy`
- `stageA5_trainable_params=xyz`
- `lambda_real=lambda_pseudo=1.0`
- `num_real_views=2`
- `num_pseudo_views=4`
- baseline：`hard_visible_union_signal`
- exp：`spgm_keep`
- `stageB_post_switch_iter=0`
- 没有 `init_pseudo_camera_states_json`

这点很关键：它不是当前文档里已经确认的 canonical bounded baseline（`post40_lr03_120 / gated_rgb0192`）协议。

---

## 3. 代码链路审查：哪些是对的
### 3.1 SPGM 插点是对的
当前 `scripts/run_pseudo_refinement_v2.py` 的链路是：
`pseudo_backward.backward(retain_graph=has_real_branch) -> maybe_apply_pseudo_local_gating(...) -> real_backward.backward() -> optimizer.step()`

这说明：
- SPGM 确实只作用在 pseudo-side backward 之后；
- real backward 仍在后面单独累积；
- StageB 的“只压 pseudo branch”这一点实现上是成立的。

### 3.2 SPGM 主调用链已接通
当前真实链路已经是：
`gate_results -> collect_spgm_stats -> build_spgm_importance_score -> build_spgm_grad_weights -> apply_gaussian_grad_mask`

也就是说：
- view-level signal gate 还在前面；
- per-Gaussian stats / score / policy 三段都实际执行了；
- grad 最终真实乘到了 Gaussian 参数上。

### 3.3 深度计算方向是对的
`pseudo_branch/spgm/stats.py` 没有误用 `render_pkg['depth']` 当 per-Gaussian depth，而是：
- 取 `gaussians._xyz`
- 配合 `current_w2c(vp)`
- 直接算 camera-space `z`

### 3.4 运行结果证明它不是 no-op
120iter compare 里：baseline `grad_weight_mean_xyz_mean ≈ 0.739`，exp `grad_weight_mean_xyz_mean ≈ 0.292`；baseline `grad_norm_xyz_mean ≈ 0.0808`，exp `grad_norm_xyz_mean ≈ 0.0691`。所以这不是“SPGM 没生效”，而是“SPGM 生效了，而且压得很明显”。

---

## 4. 当前确认的实现问题 / 未完成项
1. `spgm_density_mode` 现在是死参数：schema / CLI 有，但 `stats.py / score.py / policy.py` 没有实际消费。
2. `compute_density_entropy()` 的归一化写错了：注释是按 `log(B)`，当前实现实际按 `entropy_bins` 直接归一化。
3. SPGM 分支里的 `accepted_count` summary 直接写成 `len(sampled_views)`，当前 run 恰好全接受，所以没暴露；后续 reject-run 会误报。
4. `Noah.md / STATUS / DESIGN / CHANGELOG` 对 P2-K 的完成度表述过头：现在只能说 wiring 跑通、compare 跑完，但 protocol 没对齐 canonical baseline，且代码里仍有几个小缺口。

---

## 5. 当前 compare 为什么不算“正确完成”
这里的“不是正确完成”不是说 run 崩了，而是说它没有回答当前项目真正需要回答的问题。

1. 它没有对齐当前 canonical StageB baseline。文档已经把参考臂定成 `RGB-only v2 + gated_rgb0192 + post40_lr03_120`，但这轮 StageB-120 compare 实际跑的是 `hard_visible_union_signal + stageB_post_switch_iter=0`。
2. 它没有复用此前顺序 handoff 语义：当前 run 里没有 `init_pseudo_camera_states_json`，因此不是在既有 handoff 协议上继续比较。
3. 这轮 compare 的 signal gate 实际是 0 rejection：120iter 两臂都是 `rejected_nonempty_iters = 0`、`unique_rejected_ids = []`。所以 baseline 不是当前会稳定拒掉 `225/260` 的 `gated_rgb0192` 分支；SPGM 也不是建立在“先用 calibrated view gate 过滤坏 pseudo，再做 per-Gaussian management”的结构上。
4. 因而当前 replay regression 不能直接拿来宣判“SPGM 比当前项目 best baseline 差”。它目前只严格证明：在这条“hard_visible_union_signal raw 120iter”协议里，当前 SPGM v1 比自己的 hard-visible-union 对照更差。

当前 replay 的冻结结果是：
- baseline：`24.1360 / 0.87385 / 0.08189`
- exp：`23.5941 / 0.85926 / 0.08912`
- delta：`-0.542 PSNR / -0.01459 SSIM / +0.00724 LPIPS`

---

## 6. 如果暂时假设 protocol 没问题：方案本身哪里像有问题
1. 当前 SPGM 主要做的是“强压制”，不是“更聪明的选择”。120iter 统计里 baseline `grad_keep_ratio_xyz_mean ≈ 0.739`，exp `spgm_active_ratio_mean ≈ 0.739`，说明 active set 几乎没变；但 baseline `grad_weight_mean_xyz_mean ≈ 0.739`，exp `grad_weight_mean_xyz_mean ≈ 0.292`，说明主要变化是对同一 active set 做了较强连续衰减。
2. 结果上它换来了“real 更低、pseudo 更高、replay 更差”：120iter 末尾 baseline `loss_real_last = 0.0335`、exp `0.0319`；baseline `loss_pseudo_last = 0.0241`、exp `0.0293`。这很像 pseudo-side 更新被压住，real anchor 更容易维持，但地图没有被 pseudo supervision 充分改动。
3. 固定 depth-tier cluster keep 很可能太早、太硬。当前默认 `cluster_keep = (1.0, 0.8, 0.6)`、`support_eta = 0.5`、`weight_floor = 0.05`，再叠上 quantile 3-bin depth partition，更像“保守收缩 pseudo-side update”，而不像“补足当前 pipeline 缺的结构信息”。
4. 现在还不该先跳去 stochastic drop / 更长 iter。当前最先暴露的问题不是 deterministic keep 一定不如 stochastic，也不是 120 iter 一定不够长，而是 protocol 还没对齐、当前压制已经明显过强、代码里还有 entropy normalization / dead param 这类小坑没清掉。

---

## 6.5 审查后已立即修掉的 3 个小问题
1. `score.py` 的 density entropy 归一化已改为按 `log(B)`。
2. `spgm_density_mode` 已不再是 dead param；当前至少已真实接入 `opacity_support / support` 两种分支，并会记录 `density_mode_effective`。
3. SPGM 分支里的 accepted-count / history 统计已改成按真实 accepted views 记录，不再直接拿 `len(sampled_views)` 充数。

这三项现在已不再是后续 compare 的阻塞项；当前剩下的主问题是 protocol 对齐和方案本身的过强压制。

## 7. 下一步建议（按优先级排序）
1. 先把 compare protocol 拉回 canonical baseline：下一轮正式 compare 应固定成 `RGB-only v2 + gated_rgb0192 + post40_lr03_120` 作为 baseline，exp 只替换成 `spgm_keep`，其它保持同 pseudo set、同 handoff 语义、同 replay evaluator。
2. 先做 conservative SPGM，而不是更 aggressive 的 SPGM。建议优先两条臂：
   - A：`cluster_keep=(1.0,1.0,1.0)`，`support_eta=0.0`，`weight_floor=0.25`
   - B：`cluster_keep=(1.0,1.0,0.9)`，`support_eta=0.25`，`weight_floor=0.20`
3. 先修 3 个小代码坑，再继续扫参：`score.py` entropy normalization 改按 `log(B)`；`spgm_density_mode` 要么真正落地，要么先降级成预留接口；SPGM 分支的 accepted-count / history 统计按真实 accepted views 记录。
4. 只有在 protocol 已对齐、conservative deterministic SPGM 至少不再明显伤 replay 之后，才值得继续考虑 stochastic drop、`xyz_opacity`、更长 iter。

---

## 8. 当前建议写入项目主判断的话
> `P2-K` 当前确认的是“SPGM wiring 已跑通且确实调制了 pseudo-side grad”，但这轮 compare 还没有对齐 canonical `post40_lr03_120 / gated_rgb0192` baseline；同时当前 deterministic keep 默认超参明显更像强压制而不是更优筛选，因此 replay regression 目前更应被解读为“protocol drift + over-suppression”问题，而不是已经完成的 SPGM 结论。

---

## 9. 2026-04-17 follow-up：canonical formal compare 已完成
`P2-L` 已把这里提出的第 1 步正式补齐：[参见 `docs/P2L_spgm_canonical_stageB_compare_20260417.md`]。

对齐到 canonical `post40_lr03_120 / gated_rgb0192` protocol 之后，结果是：
- baseline：`24.02982 / 0.87229 / 0.08177`
- SPGM：`23.94230 / 0.86956 / 0.08359`
- delta：`-0.08753 PSNR / -0.00273 SSIM / +0.00182 LPIPS`

这一步说明：
1. `protocol drift` 不是当前负结果的唯一解释；即便 protocol 对齐，当前 deterministic `SPGM v1` 仍低于 canonical baseline。
2. 但这并没有推翻本审查的结构判断，反而进一步支持它：两臂 rejection 与 active set 几乎一致，真正变化仍主要是权重幅度被压低，说明当前 policy 依旧更像 suppressor 而不是 selector。
3. 因而本审查文档里的后续优先级现在可以收束成一句话：`protocol 对齐` 已完成，下一步正式进入 conservative / selector-first 的 SPGM repair。

# P2-N：selector-first SPGM P0/P1 plumbing + smoke（2026-04-17）

> 目标：先把 selector-first policy 的最小工程接口接通，并用一个极短的 `StageA.5` smoke 验证它不是只换了名字，而是真的让 active set 发生变化。

---

## 1. 一句话结论

P0/P1 已完成，selector-first 的最小代码路径已经接通，而且 smoke 证明它不再只是对同一 active set 做 soft suppress。

这轮 `StageA.5-5iter` smoke 下：
- `spgm_policy_mode_effective` 5/5 iter 都是 `selector_quantile`
- `spgm_selected_ratio` 5/5 iter 都严格小于 `spgm_active_ratio`
- `grad_keep_ratio_xyz` 与 `spgm_selected_ratio` 对齐，说明真正被更新的 Gaussian 子集已经缩小
- `refined_gaussians.ply / stageA_history.json / refinement_history.json` 都正常生成

因此现在可以把“先做 selector-first plumbing 再说”这一步划成已完成；下一个真正的问题已经变成：在 canonical StageB protocol 下，selector-first arm 能不能相对 repair A / canonical baseline 带来 replay 收益。

---

## 2. 本轮改动范围（P0）

### 2.1 改动文件
1. `pseudo_branch/spgm/policy.py`
2. `pseudo_branch/local_gating/gating_schema.py`
3. `pseudo_branch/local_gating/gating_io.py`
4. `scripts/run_pseudo_refinement_v2.py`

### 2.2 设计边界
这轮只做最小 plumbing，不改上游 signal、不改 SPGM score 主体、不改 StageB schedule。

保留的东西：
- `collect_spgm_stats(...)`
- `build_spgm_importance_score(...)`
- repair A 风格 soft weighting（`support_eta=0.0 / weight_floor=0.25 / cluster_keep=(1,1,1)`）

新增的东西：
- `spgm_policy_mode`
  - `dense_keep`：兼容当前 deterministic dense keep
  - `selector_quantile`：cluster-wise selector-first keep
- selector-first 相关 CLI / config
  - `--pseudo_local_gating_spgm_policy_mode`
  - `--pseudo_local_gating_spgm_selector_keep_ratio_near`
  - `--pseudo_local_gating_spgm_selector_keep_ratio_mid`
  - `--pseudo_local_gating_spgm_selector_keep_ratio_far`
  - `--pseudo_local_gating_spgm_selector_min_keep`
- history / summary 新字段
  - `spgm_selected_ratio`
  - `spgm_policy_mode_effective`
  - `spgm_selected_count_near/mid/far`

### 2.3 这轮没做什么
- 没做 top-k variant
- 没做 support 先 hard select 再 soft weight 的二段 score 重写
- 没做 StageB canonical formal compare
- 没做 replay compare

---

## 3. smoke 运行身份（P1）

### 3.1 运行根
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_p2n_selector_first_plumbing_smoke`
- out dir：`stageA5_selector_quantile_smoke5`

### 3.2 运行模式
- `stage_mode=stageA5`
- `stageA_iters=5`
- `stageA5_trainable_params=xyz`
- `pseudo_local_gating=spgm_keep`
- `spgm_policy_mode=selector_quantile`

### 3.3 selector 配置
- `support_eta=0.0`
- `weight_floor=0.25`
- `cluster_keep=(1.0, 1.0, 1.0)`
- `selector_keep_ratio=(1.0, 0.9, 0.75)`
- `selector_min_keep=1`

### 3.4 固定输入
- PLY：`after_opt/point_cloud/point_cloud.ply`
- pseudo cache：`20260414_signal_enhancement_e15_compare/pseudo_cache_baseline`
- signal_v2：`20260414_signal_enhancement_e15_compare/signal_v2`

---

## 4. smoke 结果

### 4.1 核心验收

1. `spgm_policy_mode_effective`
- 5/5 iter 都是 `selector_quantile`

2. `spgm_selected_ratio < spgm_active_ratio`
- active ratio：`[0.5870, 0.8234, 0.4929, 0.8210, 0.6685]`
- selected ratio：`[0.5186, 0.7273, 0.4354, 0.7252, 0.5905]`
- `selected_lt_active_all_iters = true`

3. `grad_keep_ratio_xyz` 已和 selected ratio 对齐
- `grad_keep_ratio_xyz = [0.5186, 0.7273, 0.4354, 0.7252, 0.5905]`

这说明当前 selector-first 实现不再是“active set 不变、只改 weight 幅度”；它已经真的把更新子集缩小了。

### 4.2 far cluster 也确实发生了筛选
- far cluster count：`[6545, 9181, 5496, 9154, 7453]`
- far selected count：`[4909, 6886, 4122, 6866, 5590]`

这和 `selector_keep_ratio_far=0.75` 的方向一致，说明 cluster-wise quantile keep 不是伪字段。

### 4.3 其它 smoke 信号
- `spgm_weight_p50`：`[0.7791, 0.7503, 0.7907, 0.7077, 0.7535]`
- `grad_norm_xyz_pre_mask mean = 0.1254`
- `grad_norm_xyz_post_mask mean = 0.1111`

### 4.4 产物
正常生成：
- `stageA_history.json`
- `refinement_history.json`
- `pseudo_camera_states_*.json`
- `refined_gaussians.ply`

---

## 5. 当前可以下的工程判断

1. selector-first 的最小 plumbing 已经接通，不再是计划项。
2. 这轮 smoke 已经真实证明：
   - `policy_mode` 被消费了；
   - selected set 会变化；
   - history 里能直接看出 `active -> selected` 的收缩。
3. 因而当前最合理的下一步不再是继续补 plumbing，而是：
   - 以 repair A 为 control
   - 在 canonical StageB protocol 下跑 selector-first formal compare
   - 先回答 replay 是否优于 repair A / 更接近 baseline

---

## 6. 边界与未完成项

这轮还不能声称：
- selector-first 已经优于 repair A
- selector-first 已经优于 canonical baseline
- top-k / stochastic / xyz+opacity 值得推进

因为这些都还没做 formal compare。

当前只能说：
- 代码接口已接通
- selector-first 分支被真实执行
- active set 的确被改动
- 可以进入正式 compare 阶段

# P2-Q：SPGM score/ranking repair implementation plan（2026-04-17）

> 目标：在 `P2-O` 已确认“selector-first 真实生效但 replay 变差”之后，基于当前 live code path，提出一版最小、可落地、可验证的 score/ranking 修补方案。

---

## 1. 先把 repair A 和 selector-first 的区别说清楚

当前两者的共同点是：
- 都走同一条 SPGM 主链：`signal gate -> collect_spgm_stats -> build_spgm_importance_score -> build_spgm_grad_weights -> apply_gaussian_grad_mask`
- 都复用同一个 canonical StageB protocol
- 都在 pseudo-side grad mask 这一层起作用，不改 StageB schedule、不改 upstream signal pipeline

真正的区别只有一层，但这一层很关键：

1. repair A 是“同一 active set 上的更温和加权”
- 在代码上就是 `policy_mode=dense_keep`
- 所有 active Gaussian 都还保留，只是 weight 不再像原始 SPGM v1 那样压得太狠
- 对应超参是：`support_eta=0.0 / weight_floor=0.25 / cluster_keep=(1,1,1)`
- 所以它修的是“压多重”，不是“选谁留下”

2. selector-first 是“先硬选子集，再对留下的子集做 repair-A-style weighting”
- 在代码上就是 `policy_mode=selector_quantile`
- 先按 cluster 内 score 排序，只保留 top quantile；未被选中的 active Gaussian 直接 weight=0
- 然后对被选中的 Gaussian 再用 repair-A-style soft weight
- 所以它修的是“选谁更新”

3. 为什么 `P2-O` 让我们把注意力从 policy 再往上移到 score/ranking
- `repair A` 的结果比原始 SPGM 好，说明“压制过强”这个问题是真的
- `selector-first` 的 `selected_ratio` 明显小于 `active_ratio`，说明 selection 也是真的发生了
- 但 replay 反而更差，且 selector 越强越差
- 这说明当前最主要的问题已经不是“policy 没接通”，而是“当前 ranking 用来 hard select 时还不够可靠”

一句话压缩：repair A = 不改 active set，只把同一批 Gaussian 压得更合理；selector-first = 真改 active set，但现在的排序标准还不够好，所以删错了人。

---

## 2. live code path：当前真正的 cut point 在哪里

### 2.1 当前 SPGM 调用链
`run_pseudo_refinement_v2.py` 里真实路径是：
1. `evaluate_sampled_views_for_local_gating` (`scripts/run_pseudo_refinement_v2.py:765`)
2. `collect_spgm_stats` (`run_pseudo_refinement_v2.py:806-812`)
3. `build_spgm_importance_score` (`run_pseudo_refinement_v2.py:813-825`)
4. `build_spgm_grad_weights` (`run_pseudo_refinement_v2.py:826-839`)
5. `apply_gaussian_grad_mask` (`run_pseudo_refinement_v2.py:840-844`)

所以当前 score/ranking 问题的最小修补面不是 schedule，也不是 signal gate，而是：
- 主修：`pseudo_branch/spgm/score.py`
- 次修：`pseudo_branch/spgm/policy.py`
- 配套 plumbing：`pseudo_branch/local_gating/gating_schema.py`、`pseudo_branch/local_gating/gating_io.py`、`scripts/run_pseudo_refinement_v2.py`

### 2.2 现有 stats 已经够做第一轮 score repair
`collect_spgm_stats()` 已经给了：
- `support_count`
- `depth_value`
- `density_proxy`
- `active_mask`

也就是说，第一轮不需要动 `stats.py` 才能做 score repair。

### 2.3 当前 ranking 失败的结构性原因
当前 `score.py` 的核心逻辑是：
- depth score：按 depth quantile 分 cluster，近处更高分
- density score：按 `density_proxy` 或 `support` 的归一化值 + entropy 修正
- importance raw：`alpha * depth + (1-alpha) * density`
- 最终 importance：`importance_raw * support_norm^eta`

但 `P2-O` 那条 selector-first compare 是在 repair-A-style config 上做的：
- `support_eta = 0.0`
- `cluster_keep=(1,1,1)`
- `policy_mode=selector_quantile`

这意味着：
- weighting 侧故意不让 support 再额外参与压制，这是对的，因为要保留 repair A 的正面结果；
- 但 selection 侧也继续沿用了同一个 `importance_score`；
- 当前这个 `importance_score` 更像“适合 soft weighting 的分数”，未必适合“做 hard select 的排序分数”。

因此，当前最自然的工程结论是：
不要直接继续改 selector 强度，而是先把“用于排序的 score”从“用于加权的 score”里拆出来。

---

## 3. 推荐的最小工程方案：先把 ranking 和 weighting 解耦

### 3.1 核心设计
第一版 score/ranking repair，不建议推翻现有 SPGM score 结构；建议只做一件事：

把当前单一的 `importance_score` 拆成两个角色：
1. `weight_score`：继续给 repair A / dense_keep / selected-subset soft weighting 用
2. `ranking_score`：只给 selector policy 做 cluster 内排序用

这样做的好处是：
- repair A 当前已经是正面 anchor，不要把它一起动坏
- selection 的负结果现在更像 ranking 问题，不是 weighting 问题
- 先把 ranking 单独修，因果更干净

### 3.2 第一版 ranking mode 的具体建议
先加一个最小新模式，不要一上来做太多分支：

`spgm_ranking_mode = v1 | support_blend`

含义：
- `v1`：完全复用当前 `importance_score`
- `support_blend`：在 cluster 内排序时，把 support 信息重新显式混入 ranking

建议公式：
- 先保留当前 `importance_raw`
- 另外计算 `support_norm`
- ranking score 用：
  - `ranking_score = (1 - lambda_support_rank) * importance_raw + lambda_support_rank * support_norm`

建议默认只给 ranking 用，不改 weight score。

为什么第一版用这个而不是更复杂的二段式门控：
- support_count 已在 `stats.py` 里现成可用
- selection 是 cluster 内排序，support blend 在 cluster 内就足够起作用
- 不需要先动更多模块，也不需要先引入 stochastic

### 3.3 为什么不建议第一版就把 far-only 逻辑硬编码进 score
far-only 是个很好的实验臂，但它更像 protocol/arm 设计，不应该直接焊死进 score 逻辑。

因此推荐分层：
- score 负责把 ranking 做得更可信
- policy 仍负责“每个 cluster 留多少”
- protocol 再决定是不是先只裁 far cluster

---

## 4. 具体工程改动面

### 4.1 `pseudo_branch/spgm/score.py`（主修改面）
新增内容：
1. 返回 `importance_raw`
2. 返回 `support_norm`
3. 返回 `ranking_score`
4. 新增 `ranking_mode`
5. 新增 `lambda_support_rank`

推荐新增接口参数：
- `ranking_mode: str = 'v1'`
- `lambda_support_rank: float = 0.0`

推荐返回字段：
- `importance_raw`
- `support_norm`
- `ranking_score`
- `ranking_mode_effective`
- `ranking_score_mean`
- `ranking_score_p50`

注意：
- `importance_score` 保留，继续作为 weight score
- `ranking_score` 只为 selector 排序服务

### 4.2 `pseudo_branch/spgm/policy.py`（次修改面）
当前 `build_spgm_grad_weights()` 只接 `importance_score`，并用它同时做：
- selector top-k / quantile 排序
- selected subset 的 soft weight 计算

建议改成：
- `weight_score`
- `ranking_score`

新的函数签名建议：
- `weight_score: torch.Tensor`
- `ranking_score: torch.Tensor | None = None`

规则：
- `dense_keep`：不需要 ranking，可忽略 `ranking_score`
- `selector_quantile`：如果 `ranking_score is None`，退回旧逻辑；否则 cluster 内 top-k 用 `ranking_score`
- selected subset 的 weight 继续用 `weight_score`

这一步是整个方案的关键：
“谁被选中”和“选中后压多重”终于被拆开了。

### 4.3 `scripts/run_pseudo_refinement_v2.py`（plumbing）
需要加两类东西：

1. CLI / config 透传
新增建议参数：
- `--pseudo_local_gating_spgm_ranking_mode`
- `--pseudo_local_gating_spgm_lambda_support_rank`

2. SPGM 调用改线
从：
- `build_spgm_importance_score(...) -> importance_score`
- `build_spgm_grad_weights(importance_score=...)`

改成：
- `build_spgm_importance_score(...) -> importance_score + ranking_score`
- `build_spgm_grad_weights(weight_score=importance_score, ranking_score=ranking_score, ...)`

### 4.4 `pseudo_branch/local_gating/gating_schema.py`
新增字段：
- `spgm_ranking_mode: str = 'v1'`
- `spgm_lambda_support_rank: float = 0.0`

### 4.5 `pseudo_branch/local_gating/gating_io.py`
新增 history / summary 字段：
- `spgm_ranking_mode_effective`
- `spgm_ranking_score_mean`
- `spgm_ranking_score_p50`
- 可选：`spgm_support_norm_mean`

目的不是为了好看，而是为了之后能直接回答：
- ranking 侧到底有没有变
- 变的是排序，还是只是 weight 在变

---

## 5. 第一轮实验设计：不要一上来做大 sweep

### 5.1 固定不变的东西
全部继续固定：
- same canonical baseline：`post40_lr03_120 / gated_rgb0192`
- same handoff
- same seed
- same replay evaluator
- same real branch
- same repair-A-style weighting：`support_eta=0.0 / weight_floor=0.25 / cluster_keep=(1,1,1)`

### 5.2 第一轮建议只跑 4 臂
1. `control_repair_a_dense_keep`
- `policy_mode=dense_keep`
- `ranking_mode=v1`

2. `selector_far_only_v1`
- `policy_mode=selector_quantile`
- `keep_ratio=(1.0, 1.0, 0.75)`
- `ranking_mode=v1`

3. `selector_far_only_support_blend_l03`
- `policy_mode=selector_quantile`
- `keep_ratio=(1.0, 1.0, 0.75)`
- `ranking_mode=support_blend`
- `lambda_support_rank=0.3`

4. `selector_far_only_support_blend_l05`
- `policy_mode=selector_quantile`
- `keep_ratio=(1.0, 1.0, 0.75)`
- `ranking_mode=support_blend`
- `lambda_support_rank=0.5`

为什么第一轮先 far-only：
- `P2-O` 说明 mid+far 同时裁剪已经明显伤 replay
- 如果只裁 far 仍然差，说明 ranking 问题更本质
- 如果 far-only + support-aware ranking 能接近或超过 repair A，说明之前主要是 mid cluster 被误杀，且 support-aware ranking 确实有用

### 5.3 暂时不要做的东西
第一轮不要碰：
- stochastic
- `xyz+opacity`
- 更长 iter
- top-k 新 policy mode
- 改 StageB schedule
- 改 signal gate
- raw RGB densify

---

## 6. 验收标准

### 6.1 机制验收
必须同时满足：
1. `spgm_selected_ratio_mean < spgm_active_ratio_mean`
2. `grad_keep_ratio_xyz_mean` 与 `selected_ratio_mean` 对齐
3. `spgm_ranking_mode_effective` 与 CLI 一致
4. ranking 日志能区分 `weight_score` 和 `ranking_score`

### 6.2 结果验收
第一轮的最低目标不是立刻超过 baseline，而是：
1. 新 arm 不应明显差于 repair A（建议 PSNR 门槛：不低于 `-0.005`）
2. 若 far-only + support-aware ranking 相对 far-only v1 回升，说明 score/ranking repair 路线是活的
3. 只有当 score-aware ranking 至少不再明显伤 replay，才值得继续谈更强 selector 或 top-k

---

## 7. 当前不建议的错误方向

1. 继续直接扫 `(mid, far)` 更强 keep ratio
2. 在 ranking 还没修好前，把 selector 失败解释成“selection 这条路不行”
3. 为了救 selector 结果，又回头同时改 weighting、schedule、opacity
4. 直接跳去 stochastic 或 `xyz+opacity`

---

## 8. 一句话执行口径

下一步最合理的不是“继续把 selector 调狠一点”，而是：
先把“用于排序的 score”从“用于加权的 score”里拆出来，做一版 support-aware ranking，并在 far-only selector 下做最小验证；只有 ranking 先站住，selector-first 这条线才值得继续扩。

# SIGNAL_ENHANCEMENT.md

## 1. 目标

当前主目标不是继续优化 consumer，而是优先解决 pseudo signal 过稀、raw seed 太弱的问题。

本阶段聚焦三类增强：
1. support-aware pseudo selection
2. multi-pseudo allocation（每 gap 不只 1 个 pseudo）
3. multi-anchor verify（不只左右最近 anchor）

阶段完成后，再回到：
- `docs/SIGNAL_SEMANTICS_AND_STABLE_REFINEMENT_PLAN_20260414.md`
继续看 StageB curriculum / local gating / depth reweight。

一句话：**先把 raw signal 做强，再谈下游怎么更稳地消费。**

---

## 2. 当前状态与为什么要开这一阶段

当前已经完成：
- continuous confidence + agreement-aware support
- RGB/raw confidence 与 depth/train-mask 语义分离
- confidence-aware densify
- 2-frame smoke
- 8-frame retuned short compare

当前结论：
1. 语义链路已经打通，但 raw seed 仍然稀；
2. continuous confidence 只能校准已有 support，不能创造新的 support；
3. densify 仍然高度依赖 seed 质量，因此阈值敏感；
4. 当前不适合直接进入 StageB 两段式 curriculum，因为 signal 还没收口。

因此现在应进入一个新的“signal enhancement”阶段，直接增强 signal 来源，而不是继续围绕当前弱 signal 做更细的消费修补。

---

## 3. 本阶段要做到什么

### 3.1 最低目标

在不改变整个 refine 主骨架的前提下，让 pseudo cache 的 raw signal 明显增强，至少体现在：
1. raw support coverage 上升；
2. both-support / verified-depth ratio 上升；
3. densify 后 non-fallback depth 不再主要受 seed 稀缺限制；
4. 8-frame short StageA 对照中，新 signal 方案至少能接近 baseline，最好优于 baseline。

### 3.2 成功标准

至少满足其中两条：
1. `seed_valid_ratio` 相比当前 midpoint8 baseline 明显提升；
2. `seed_both_ratio` 和 `continuous_confidence_mean_positive` 协同提升；
3. densify 后 `mean_dense_valid_ratio` 不低于当前 baseline 太多，同时 fallback ratio 不恶化；
4. 8-frame StageA 短跑下，`loss_total` / `loss_depth` / pose drift 不再同时差于 baseline。

---

## 4. 三个探索方向与优先级

## 4.1 A：support-aware pseudo selection（最高优先级）

### 目标

不要固定用 midpoint。对每个 KF gap，从一组候选 pseudo 位置中选出更可能产生高质量 signal 的帧。

### 为什么先做这个

这是最小改动、解释最干净的一步：
- 不增加太多后续复杂度；
- 直接测试“midpoint 是否选错了”；
- 如果这一步就有效，说明真正的问题首先在选点而不是后段 refine。

### 候选设计

对每个 gap，先定义候选位置集：
- 最小版：`{1/3, 1/2, 2/3}`
- 扩展版：`{1/4, 1/3, 1/2, 2/3, 3/4}`
- 如果 gap 足够大，可再考虑更密点，但第一版不建议太密

### 打分项（第一版）

每个候选 pseudo 位置，基于当前 stage PLY 和邻近 anchor 做轻量 verify，打出一个综合 score：
- `support_ratio_left`
- `support_ratio_right`
- `support_ratio_both`
- `verified_ratio`（projected valid depth / non-fallback ratio）
- `correction_magnitude`（target depth 相对 render depth 的平均修正量）
- 左右平衡项（避免一边很好一边很差）

建议第一版 score：

`score = w_both * both_ratio + w_verified * verified_ratio + w_corr * correction_mag + w_balance * min(left_ratio, right_ratio)`

其中 `w_both`、`w_verified` 权重大于 `w_corr`。

### 需要修改的文件

1. `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
- 当前 `pick_frames_between(...)` 只支持 `midpoint / tertile / both`
- 需要扩展成：
  - 候选模式（enumerate candidate pseudo ids）
  - 支持按评分自动选 pseudo

2. 新增脚本：`scripts/select_signal_aware_pseudos.py`
- 输入：
  - `internal_cache_root`
  - `stage_tag`
  - `kf_indices`
  - 候选位置集配置
- 输出：
  - 每个 gap 的候选列表
  - 每个候选的 score 明细
  - 最终选中的 pseudo frame ids
  - `selection_report.json`

3. `scripts/brpo_build_mask_from_internal_cache.py`
- 为候选 pseudo 轻量 verify 提供复用接口
- 最好能支持 “只算 summary，不全量导出大包” 的模式

### 输入 / 输出

输入：
- internal eval cache
- stage PLY
- kf indices
- 候选位置配置

输出：
- `manifests/signal_aware_selection_manifest.json`
- `reports/signal_aware_selection_report.json`
- 选中的 pseudo ids

### 验证方法

1. 与 midpoint8 对比：
- support ratio
- both ratio
- verified ratio
- correction magnitude

2. 如果只改 selection，不改后续 pipeline，再做一次 8-frame short StageA compare。

---

## 4.2 B：multi-pseudo allocation（第二优先级）

### 目标

对大 gap 不再只放 1 个 pseudo，而是放 2 个甚至更多，从而直接增加 raw support 机会。

### 为什么第二步再做

它会同时改变：
- pseudo 数量
- coverage
- 计算量

因此解释比 support-aware selection 更混。更适合作为 selection 之后的扩展。

### 第一版建议

分两档：
1. 双 pseudo：`1/3 + 2/3`
2. 更密探索：`1/4 + 1/2 + 3/4` 或 `1/5, 2/5, 3/5, 4/5`

不建议第一版直接上 4 个五分位，先从双 pseudo 开始。

### 需要修改的文件

1. `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
- `pick_frames_between(...)` 需要新增：
  - `placement=dual_tertile`
  - `placement=triple_quartile`
  - 以及显式候选列表模式

2. pseudo cache manifest 相关逻辑
- 保证同一个 gap 可以记录多个 pseudo
- sample meta 里要写清属于哪个 gap / 哪种 allocation policy

### 输入 / 输出

输入：
- gap 划分
- allocation policy

输出：
- 新 pseudo selection manifest
- 新 pseudo_cache
- allocation report（每 gap 放了几个 pseudo）

### 验证方法

1. 比较 raw seed ratio / both ratio 是否稳定上升；
2. 比较 densify 后 verified / fallback 是否改善；
3. 比较 StageA 短跑是否更容易优于 baseline。

---

## 4.3 C：multi-anchor verify（第三优先级）

### 目标

不只使用左右最近 keyframe 做 verify，而是允许额外邻近 anchor 提供支持 / 投票，增加 seed 的鲁棒性与覆盖。

### 为什么第三步做

这是更明显的方法变更：
- 计算量增大；
- verify 语义从双边变多边；
- 需要重新定义 both / single / agreement 的含义。

因此应该在 selection / allocation 之后推进。

### 第一版建议

不要直接上很复杂的多视角图优化，先做轻量版：
- 主 anchor：左右最近 keyframe
- 额外 anchor：左右再各取 1 个近邻（若存在）
- 最终规则：
  - 保留原左右最近 anchor 作为主语义层
  - 额外 anchor 只作为 bonus vote / support augmentation

也就是说，第一版不要完全推翻 current both/single 定义，而是：
- 先算 primary both/single
- 再用 extra anchor 的 vote 提高连续 confidence 或扩大 seed 支持

### 需要修改的文件

1. `pseudo_branch/brpo_reprojection_verify.py`
- 当前 `find_neighbor_kfs(...)` 只返回左右最近 kf
- 需要新增：
  - `find_neighbor_kfs_multi(...)`
  - 支持返回多 anchor 列表

2. `scripts/brpo_build_mask_from_internal_cache.py`
- 当前每个 pseudo 只跑 left/right 两分支
- 需要新增多 anchor verify 模式：
  - `--anchor-policy nearest2`（当前）
  - `--anchor-policy nearest4`
  - `--anchor-policy nearest2_plus_context`

3. `pseudo_branch/brpo_confidence_mask.py`
- 为多 anchor 增加 vote-aware / support-augmented confidence 汇总逻辑

### 输入 / 输出

输入：
- anchor policy
- verify thresholds

输出：
- 多 anchor verify summary
- 多 anchor confidence / support artifacts
- anchor-level provenance

### 验证方法

比较：
- support ratio
- both ratio
- continuous confidence summary
- densify 后 verified ratio

---

## 5. 建议的实施顺序

### Phase E1：support-aware pseudo selection

先只做 selection，不改 pseudo 数量。

可交付：
- `scripts/select_signal_aware_pseudos.py`
- selection manifest / report
- 一组 signal-aware 选中的 pseudo ids
- 与 midpoint8 的 summary 对照表

### Phase E2：dual-pseudo allocation

在 E1 基础上做 `1/3 + 2/3`。

可交付：
- dual-pseudo pseudo set
- 与 E1 / midpoint8 的 raw signal 对照

### Phase E3：multi-anchor verify

在 E1/E2 中效果更好的 pseudo set 上，加入多 anchor verify。

可交付：
- nearest2 vs nearest4 compare
- raw signal / densify / StageA short compare

---

## 6. 不建议当前优先做的事

1. 直接进入 `StageB 两段式 curriculum`
- 当前 signal enhancement 阶段尚未收口，过早进入会混淆判断

2. 直接做 `pseudo branch local Gaussian gating`
- 它更像 signal 已经够用后的 consumer 约束器

3. 直接做 `active-support-aware depth reweight`
- 这仍属于消费端增强，不能替代 raw signal 提升

4. 直接做 full SPGM
- 这是更后段的 stabilizer，不是当前主线

---

## 7. 每个阶段要交付什么

### E1 交付物
- `scripts/select_signal_aware_pseudos.py`
- `reports/signal_aware_selection_report.json`
- `manifests/signal_aware_selection_manifest.json`
- 1 份简洁的 compare 表（midpoint8 vs signal-aware-8）

### E2 交付物
- dual-pseudo selection manifest
- dual-pseudo pseudo cache
- compare 表（midpoint8 vs signal-aware-8 vs dual-pseudo）

### E3 交付物
- multi-anchor verify artifacts
- compare 表（nearest2 vs nearest4）
- 8-frame short StageA 对照结果

---

## 8. 测试与验收方法

每个阶段都固定看三层：

### signal layer
- `support_ratio_left/right/both`
- `seed_valid_ratio`
- `continuous_confidence_summary`
- `agreement_summary`
- `correction_magnitude`

### densify layer
- `dense_valid_ratio`
- `densified_only_ratio`
- `render_fallback_ratio`
- accepted / rejected patch counts

### short refine layer
- 8-frame StageA short run
- `loss_total / loss_rgb / loss_depth`
- true pose delta aggregate

验收原则：
- 如果 signal layer 没明显改善，就不要进入更后面的 StageB consumer 调整；
- 只有当至少一条 signal enhancement 路线在 8-frame short compare 上接近或优于 baseline，才回到 `SIGNAL_SEMANTICS_AND_STABLE_REFINEMENT_PLAN_20260414.md` 继续第 4 步之后的内容。

---

## 9. 当前推荐执行顺序

我建议直接按下面顺序推进：
1. 先做 E1：support-aware pseudo selection
2. 再做 E2：dual-pseudo allocation
3. 然后做 E3：multi-anchor verify
4. 做完 signal enhancement 阶段后，再回到：
   - `docs/SIGNAL_SEMANTICS_AND_STABLE_REFINEMENT_PLAN_20260414.md`
   继续看 StageB curriculum / local gating / depth reweight

一句话压缩：
**现在最该做的不是继续修 consumer，而是先把 pseudo 从哪选、选几个、拿哪些 anchor 去 verify 这三件事做强。**

## 10. E1 第一轮执行结果（2026-04-14 夜）

### 已完成
- `scripts/select_signal_aware_pseudos.py` 已落地
- 已对 8 个 gap 的 `{1/3, 1/2, 2/3}` 候选完成 lightweight scoring
- 已生成：
  - `.../20260414_signal_enhancement_e1/reports/signal_aware_selection_report.json`
  - `.../20260414_signal_enhancement_e1/manifests/signal_aware_selection_manifest.json`
  - `docs/SIGNAL_AWARE_SELECTION_E1_REPORT_20260414.md`

### 当前结果
- midpoint8：`[17, 51, 86, 121, 156, 190, 225, 260]`
- signal-aware-8：`[23, 57, 92, 127, 162, 196, 225, 260]`
- `6/8` 个 gap 改选，且前 6 个 gap 全部偏向 `2/3`

aggregate 对照：
- `support_ratio_both`: `0.00754 -> 0.00738`
- `verified_ratio`: `0.01498 -> 0.01543`
- `continuous_confidence_mean_positive`: `0.69049 -> 0.69546`
- `agreement_mean_positive`: `0.75106 -> 0.75282`
- `mean_abs_rel_correction_verified`: `0.02051 -> 0.02102`
- `score`: `0.10101 -> 0.10313`

### 当前判断
E1 方向成立：`midpoint` 不是当前 case 的稳定最优 pseudo 位置。

但需要明确：这轮还是 lightweight render-based verify，不是 full fused-first apples-to-apples rerun。因此它已经足够作为“继续走 E1”的证据，但还不应直接当成最终验收。

### 建议的下一步
先不要直接进入 E2。

更稳的顺序是：
1. 用 `signal-aware-8` 生成新的 pseudo set；
2. 走与 midpoint8 相同的 verify/pack；
3. 做一次 apples-to-apples 的 8-frame StageA short compare；
4. 若结果仍成立，再进入 E2（dual-pseudo allocation）。

## 11. E1.5 正式短对照结果（2026-04-14 深夜）

### 已完成
- 已对 `signal-aware-8` 重跑正式 `fusion -> verify -> pack`
- 已完成与 midpoint8 apples-to-apples 的 M5 densify 对照
- 已完成与 midpoint8 apples-to-apples 的 8-frame StageA-20iter 对照
- 总结报告：`docs/SIGNAL_AWARE_SELECTION_E15_COMPARE_20260414.md`

### 当前结果
raw signal：
- `verified_ratio`: `0.01498 -> 0.01543`
- `continuous_confidence_mean_positive`: `0.69049 -> 0.69546`
- `support_ratio_both`: `0.00754 -> 0.00738`

baseline densify：
- `mean_dense_valid_ratio`: `0.02269 -> 0.02961`

conf-aware densify：
- `mean_dense_valid_ratio`: `0.01634 -> 0.01831`

StageA-20：
- baseline `loss_total`: `0.03673 -> 0.03659`
- conf-aware `loss_total`: `0.04392 -> 0.04340`

### 当前判断
E1 已可视为通过：
- raw signal 层有小幅正向改善；
- densify 层改善更明显；
- short refine 层也未出现反噬，反而在 baseline / conf-aware 两条线都略优于 midpoint8。

因此下一步可以进入 E2（dual-pseudo allocation），不需要再把 E1 卡在“midpoint 到底够不够好”的争论上。


## 12. E2 正式对照结果（2026-04-14 深夜）

### 已完成
- 已将 E2 从规划推进到正式落地：selection manifest / schema / pseudo_cache 已支持每 gap 多 pseudo
- 已完成 `top2 per gap` 的正式 `select -> fusion -> verify -> pack`
- 已完成 baseline / conf-aware densify 对照
- 已完成 16-pseudo 的 StageA-20iter 对照
- 总结报告：`docs/SIGNAL_DUAL_PSEUDO_E2_COMPARE_20260414.md`

### 当前结果
本轮实际 pseudo set：
- `top2 = [23, 17, 57, 51, 92, 86, 127, 121, 162, 156, 196, 190, 225, 231, 260, 266]`
- 在当前 case 中，`top2` 稳定落成 `1/2 + 2/3`

raw signal（vs midpoint8）：
- `verified_ratio`: `0.01498 -> 0.01520`
- `continuous_confidence_mean_positive`: `0.69049 -> 0.69470`
- `support_ratio_both`: `0.00754 -> 0.00741`

baseline densify：
- midpoint8：`0.02269`
- E1 signal-aware-8：`0.02961`
- E2 dual-pseudo(top2)：`0.02621`

conf-aware densify：
- midpoint8：`0.01634`
- E1 signal-aware-8：`0.01831`
- E2 dual-pseudo(top2)：`0.01737`

StageA-20：
- E1 baseline `loss_total`: `0.03659`
- E2 baseline `loss_total`: `0.04555`
- E1 conf-aware `loss_total`: `0.04340`
- E2 conf-aware `loss_total`: `0.05815`

### 当前判断
E2 工程上是成功的，但方法上还没有赢：
- 它已经证明 multi-pseudo allocation 可以无缝接入当前 pipeline；
- 也说明在本 case 中“第二个 pseudo”更像 midpoint，而不是更偏左的 `1/3`；
- 但当前这版 E2 没有优于 E1 的单 pseudo winner，尤其在 short refine 层出现了明显回落。

因此当前结论不是“更多 pseudo 一定更好”，而是：
- E2 相比 midpoint8 仍有轻微正向；
- 但它没有超过 E1 `signal-aware-8`；
- 当前 default winner 仍应保持 E1。

### 关于是否进入 E3
可以进入 E3，但不建议把当前 E2 16-pseudo set 当作基底。

更合理的做法是：
1. 保持 E1 `signal-aware-8` 为当前 winner；
2. 在 E1 winner 上推进 E3 `multi-anchor verify`；
3. 若之后还要继续做 E2，应考虑更保守的 conditional top2，而不是所有 gap 一律 top2。

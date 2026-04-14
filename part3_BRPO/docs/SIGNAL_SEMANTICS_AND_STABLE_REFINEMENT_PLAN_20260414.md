# Signal semantics and stable refinement plan（2026-04-14）

## 1. 目标

基于 2026-04-13 夜的 `MIDPOINT8_M5_P1_BOTTLENECK_REVIEW_20260413.md`、更新后的 `DEBUG_CONVERSATION.md`，以及当前真实代码链路，给出下一阶段可直接执行的工程方案。

本阶段的总目标不是继续在现有 midpoint8 + 全局 `xyz/opacity` 上扫同一组 refine 超参，而是先解决两个更根本的问题：
1. pseudo supervision 的语义还不够接近 BRPO，且当前有效几何 signal 太稀、太弱；
2. StageB 的消费方式缺少后段稳定器，导致弱局部 supervision 去驱动全局 Gaussian 扰动。

一句话压缩：**先增强/校准 signal，再把 refine 变成更稳定、更局部的 curriculum。**

---

## 2. 这轮分析后固定下来的判断

### 2.1 应该优先做 A 层，B 层可以后置，C 层必须配套但不能喧宾夺主

- A 层（continuous confidence、agreement-aware support、confidence-aware densify）是当前第一优先级，因为它直接作用于“signal 本身”。
- B 层（从固定 midpoint 扩展到 1/3、2/3 或更密 pseudo）本质上主要增加计算量与 coverage，不是当前最先要动的架构问题；可以放在 A 层跑通之后。
- C 层（下游消费方式修正）不能跳过，因为就算 A 层把 signal 稍微增强，若仍然让全局 Gaussian 无约束地吃这份 supervision，后段依旧容易退化。

### 2.2 和 BRPO 的真正差异，不在“有没有双向”，而在“语义是否闭合”

从当前代码和更新后的分析看：
1. `pseudo_fusion.py` 已经不是简单平均，它已经有 residual fusion、agreement、view weight；所以 fusion 不是从 0 到 1 的缺失项。
2. 但 `brpo_confidence_mask.py` 仍然是离散 `1.0 / 0.5 / 0.0`；
3. `brpo_train_mask.py` 又把 seed 传播成更大的 train region；
4. `pseudo_loss_v2.py` 里 RGB 和 depth 都吃同一个 `confidence_mask`；
5. `run_pseudo_refinement_v2.py` 的 StageB 仍然是单段式 joint optimize，且 Gaussian 参数组直接是全局 `gaussians._xyz` / `gaussians._opacity`。

因此当前最该补的不是“再追一个更大的 pseudo 模型”，而是把下面三件事连起来：
1. raw confidence semantics 更接近 BRPO；
2. propagation / densify 只主要服务于 depth，而不是把 RGB 监督也一起放松；
3. StageB 在后段对 map-side update 做 gating / freeze / schedule。

### 2.3 对 C 层的总判断

C 层不应被理解成“再加一个大正则”或者“直接上 full SPGM”。
当前更合理的 C 层是三步：
1. loss reweight：让 depth 在 active support 上恢复可比存在感；
2. optimizer scope gating：把 pseudo 分支引起的 Gaussian 更新收缩到支持更充分的局部可见子集；
3. stage curriculum：把 StageB 改成前强后稳，而不是 120iter 和 300iter 只差“多跑了更久”。

---

## 3. 当前代码链路上，最值得改的具体位置

### 3.1 pseudo signal 生成侧

#### (a) `pseudo_branch/brpo_confidence_mask.py`
当前问题：
- 仍然是离散 both/single/none 三值语义；
- 没把 left/right reproj / rel-depth margin、双侧 agreement 真正映射到连续权重。

下一步：
- 保留 `support_both / support_single / support_left / support_right` 这些离散拓扑层，作为解释层；
- 但新增 continuous confidence 输出，不再只保存离散版 `confidence_mask_brpo_*`；
- both-support 需额外乘 agreement-aware 因子，而不是一律 1.0。

建议新增输出：
- `confidence_mask_brpo_cont_fused.npy`
- `confidence_mask_brpo_cont_left.npy`
- `confidence_mask_brpo_cont_right.npy`
- `confidence_mask_brpo_agreement.npy`

连续权重建议先采用保守版本：
- `w_reproj = exp(- reproj_error / tau_reproj)`
- `w_depth = exp(- rel_depth_error / tau_depth)`
- `w_agree = exp(- |log z_left - log z_right| / tau_agree)`（仅 both-support 有定义）
- single 区只用单边 `w_reproj * w_depth`
- both 区用 `sqrt(w_left * w_right) * w_agree`

注意：第一版不要引入太多新自由度；先把 continuous confidence 做成“离散语义上的连续化校准”，而不是完全重写验证语义。

#### (b) `pseudo_branch/brpo_train_mask.py`
当前问题：
- propagation 后 train region 直接继承为训练 confidence，导致 RGB 和 depth 共享同一张扩张后的 mask。

下一步：
- train propagation 继续保留，但把它明确定义成 **depth candidate region**，不要再默认等同于 RGB main supervision mask；
- 新增两层概念：
  1. `rgb_confidence_mask_*`：更接近 raw / seed confidence semantics
  2. `depth_train_mask_*`：propagation 后候选区域

也就是说，传播层保留，但语义上要从“训练置信图”改回“depth 可 densify / 可训练区域”。

#### (c) `pseudo_branch/brpo_depth_densify.py`
当前问题：
- densify 主要按 candidate region + patch seed consistency 决定，没有显式继承 continuous confidence / both-vs-single quality。

下一步：
- densify patch acceptance 增加 confidence-aware 条件：
  - patch mean continuous confidence >= `tau_conf_patch`
  - both-rich patch 可适当放宽 `min_seed_count`
  - single-dominant patch 收紧 `max_seed_delta_std`
- 第一版不改 densify 的核心 patchwise 机制，只补 decision rule。

### 3.2 refine consumer / loss 侧

#### (a) `pseudo_branch/pseudo_loss_v2.py`
当前问题：
- RGB 和 depth 共用同一张 `confidence_mask`；
- depth 虽分 source-aware seed/dense/fallback，但总量仍然被 `(1 - beta_rgb)` 压得很低；
- 还没有显式记录 active support mass，难以做“有效质量”归一化。

下一步：
- 新增“RGB mask”和“depth mask”分离接口，不再强制共用一张 mask；
- 新增 active-support-aware reweight，目标不是暴力增大 depth，而是让 depth 在有效区域里恢复到可比存在感。

建议实现：
1. 保留 source-aware depth 三分支；
2. 新增 per-view 统计：
   - `rgb_active_ratio`
   - `depth_seed_ratio`
   - `depth_dense_ratio`
   - `depth_verified_ratio`
   - `depth_effective_mass`
3. 新增可选 reweight 模式：
   - `none`
   - `normalize_by_active_mass`
   - `target_rgb_depth_ratio`

第一版推荐做 `normalize_by_active_mass`：
- 让 depth loss 先按 active verified mass 归一到“每单位有效像素”的尺度；
- 再乘一个上限裁剪系数，例如 `clip(verified_target_ratio / verified_ratio, 1, max_gain)`；
- `max_gain` 初始建议保守设为 `3.0 ~ 5.0`，避免直接炸掉。

关键原则：
- 不追求让 depth 总权重压过 RGB；
- 追求的是“在当前只有 3.49% verified depth 的情况下，不要让 depth 实际存在感只剩 RGB 的几分之一”。

#### (b) `scripts/run_pseudo_refinement_v2.py`
当前问题：
- 只有一个 `stageA_mask_mode`，没有 RGB / depth mask 语义分离；
- StageB 是单段式 joint backward + step；
- 没利用 `render(...)` 已返回的 `visibility_filter` 做 local Gaussian gating。

下一步需要三类改动：

1. mask / target CLI 分离
建议新增：
- `--stageA_rgb_mask_mode {raw_confidence, seed_support_only, train_mask, legacy}`
- `--stageA_depth_mask_mode {seed_support_only, train_mask}`
- `--stageA_confidence_variant {discrete, continuous}`

其中默认建议改成：
- RGB: `raw_confidence`
- depth: `train_mask`
- confidence variant: `continuous`

2. StageB 两段式 curriculum
建议新增：
- `--stageB_phase1_iters 120`
- `--stageB_phase2_iters 180`
- `--stageB_phase2_pose_lr_scale 0.0`（直接 freeze）或 `0.1`
- `--stageB_phase2_xyz_lr_scale 0.3`
- `--stageB_phase2_opacity_lr_scale 0.3`
- `--stageB_replay_checkpoints 80,120,160,200,300`

推荐默认：
- phase1: 保持现有 conservative joint
- phase2: freeze pseudo pose；gaussian lr 下调到 0.1~0.3；real anchor 保留

3. pseudo 分支的 local Gaussian gating
利用 `pkg["visibility_filter"]`，对 sampled pseudo views 形成 union mask；
但不是所有 pseudo views 都无条件记入，而是只对“signal gate 通过”的 pseudo views 累加 mask：
- `confidence_nonzero_ratio >= tau_support`
- `verified_ratio >= tau_verified`
- `correction_magnitude >= tau_corr`

然后把 StageB backward 改成两段：
- pseudo backward：先回传 pseudo loss，再对 Gaussian grad 做 local mask，只保留 union visibility 内的梯度；
- real backward：再回传 real loss，不做同样的 pseudo-local mask，让 real anchor 继续提供全局纠偏。

这可以直接复用 v1 里 split backward 的思路，只是把“冻结哪些参数组”换成“对 `_xyz` / `_opacity` 的 grad 做 visibility-gated masking”。

建议新增一个 helper 文件：
- `pseudo_branch/gaussian_grad_gate.py`

里面最小先实现：
- `build_union_visibility_mask(...)`
- `apply_gaussian_grad_mask_(gaussians, active_mask, mask_xyz=True, mask_opacity=True)`

### 3.3 不建议现在优先做的事

1. 立刻做 full SPGM（1D OT depth partition + density entropy + cluster-aware stochastic masking）
   - 原因：它更像 stabilizer，不会凭空创造 signal。
2. 立刻把 pseudo 数量扩到很密
   - 原因：当前语义和消费逻辑还没统一，直接加量只会把问题放大或让结论更混。
3. 立刻追完整 pseudo-view UNet
   - 原因：当前更缺的是 semantics + stable consumer，而不是更复杂的生成器。

---

## 4. 下一阶段的执行方案（按优先级）

## Phase S1：统一 pseudo supervision semantics（优先级最高）

### S1.1 continuous confidence + agreement-aware support [done 2026-04-14]

目标：先把离散 1/0.5/0 改成更细粒度的连续 confidence，但不破坏当前 seed topology。

文件：
- `pseudo_branch/brpo_confidence_mask.py`
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`

完成标志：
- [x] pseudo cache sample 内可同时看到 discrete 与 continuous confidence 文件。
- [x] meta/summary 中新增 both/single 分布与 continuous confidence 统计。
- 实现位置：`pseudo_branch/brpo_confidence_mask.py`、`scripts/brpo_build_mask_from_internal_cache.py`、`scripts/prepare_stage1_difix_dataset_s3po_internal.py`。

### S1.2 RGB / depth mask 语义分离 [done 2026-04-14]

目标：
- RGB 回到更接近 BRPO raw confidence semantics；
- depth 继续使用 propagation + M5 densify 的扩展区域。

文件：
- `pseudo_branch/brpo_train_mask.py`
- `scripts/run_pseudo_refinement_v2.py`
- `pseudo_branch/pseudo_loss_v2.py`

完成标志：
- [x] StageA/A.5/StageB 可以分别指定 RGB mask 与 depth mask。
- [x] history 中显式记录两种 mask 的 coverage。
- 实现位置：`scripts/run_pseudo_refinement_v2.py`、`pseudo_branch/pseudo_loss_v2.py`。

### S1.3 confidence-aware densify [done 2026-04-14]

目标：让 M5 densify 真正继承 seed quality，而不是只在 candidate region 内 patch 投票。

文件：
- `pseudo_branch/brpo_depth_densify.py`
- `scripts/materialize_m5_depth_targets.py`

完成标志：
- [x] densify summary 里新增 patch-level confidence 统计。
- [x] dense_valid_ratio 变化现在可由 confidence gate / both-vs-single rule解释。
- 备注：首轮 smoke 参数 (`min_patch_confidence=0.25, both_relax=2, single_std_tighten=0.75`) 明显偏保守，后续需回调。

---

## Phase S2：把 StageB 改成稳定 curriculum（第二优先级）

### S2.1 active-support-aware depth reweight

目标：恢复 depth 在 active support 上的可比存在感。

文件：
- `pseudo_branch/pseudo_loss_v2.py`
- `scripts/run_pseudo_refinement_v2.py`

建议最小新增参数：
- `--stageA_depth_reweight_mode {none, normalize_by_active_mass}`
- `--stageA_depth_reweight_max_gain`
- `--stageA_depth_target_verified_ratio`

### S2.2 StageB 两段式训练

目标：解决“120 好、300 坏”不是靠多扫几组超参，而是让后段机制变稳。

文件：
- `scripts/run_pseudo_refinement_v2.py`

phase2 推荐默认：
- freeze pseudo pose
- xyz / opacity lr 下调
- real anchor 保留
- 固定 replay checkpoints

### S2.3 local Gaussian gradient gating

目标：让 pseudo branch 只改 support-visible 的 Gaussian 子集，而不是把 pseudo-side loss 直接传给全局 map。

文件：
- `pseudo_branch/gaussian_grad_gate.py`（新）
- `scripts/run_pseudo_refinement_v2.py`

第一版 gating 只做：
- sampled pseudo views 的 union `visibility_filter`
- 按 pseudo sample 的 signal gate 决定是否纳入 union
- 对 `_xyz` / `_opacity` grad 做布尔 mask

---

## Phase S3：稳定后再做的增强（第三优先级）

1. pseudo 选点从固定 midpoint 扩展到 `1/3 + 2/3` 或更密配置
2. joint optimize 后补一个 BRPO 风格的 Gaussian refinement stage
3. 最后才考虑最小版 SPGM，再往 full SPGM 靠

这里的原则是：先把语义统一和消费稳定器补上，再扩大 pseudo 数量或 map-side 管理复杂度。

---

## 5. 具体实验矩阵（建议顺序）

### E0：cache-only 语义审计（不跑 refine） [smoke done 2026-04-14 / 2-frame]

目标：确认新 continuous confidence 是否真的提供了比离散版更有信息量的排序。

检查项：
1. continuous confidence 直方图
2. both vs single 的 agreement 分布
3. raw confidence coverage vs propagated depth train region coverage
4. densify 前后 verified/dense/fallback source ratio

通过条件：
- continuous confidence 不是退化成几乎全 0 或全 1；
- both-support 平均权重高于 single，但有 margin 区分；
- propagation 不再被误当成 RGB confidence semantics。

### E1：StageA / A.5 supervision semantics ablation [smoke done 2026-04-14 / 2-frame StageA-10iter]

对照组建议：
1. baseline：当前 discrete + shared train_mask
2. A1：continuous confidence + RGB(raw) / depth(train)
3. A2：A1 + confidence-aware densify

看三类指标：
- signal gate：verified ratio / correction magnitude / confidence stats
- pseudo-side alignment：loss_rgb / loss_depth
- replay gate：270-frame replay non-regression

### E2：StageB curriculum ablation

在 E1 最优 cache semantics 上继续：
1. B0：当前单段式 StageB120 / 300（对照）
2. B1：两段式，但不做 local gating
3. B2：两段式 + local Gaussian gating
4. B3：B2 + active-support-aware depth reweight

主要判断：
- 能否保住 120iter 的正益并减少 300iter 退化；
- `xyz drift` 的受影响范围是否明显收缩；
- replay 是否能从“后段掉头”变成“后段持平或小幅继续提升”。

---

## 6. 每一轮实验都必须固定记录的 gate

以后不要再让“pseudo loss 降了”替代总体结论。
每次长跑前后固定记录三层 gate：

1. signal gate
- `confidence_nonzero_ratio`
- `verified_ratio`
- `seed_ratio`
- `dense_ratio`
- `correction_magnitude`
- continuous confidence 的均值 / 分位数

2. pseudo alignment gate
- masked RGB loss
- seed/dense/fallback depth loss
- abs pose components

3. replay gate
- 270-frame replay vs baseline
- A.5 / StageB vs 前一阶段
- xyz drift 覆盖率（如 `>1e-3` 比例）

任何一轮如果 signal gate 本身太弱，就不要直接进入更长的 StageB。

---

## 7. 文档与归档约定

1. 本文档作为“下一阶段的主执行计划”，暂时放在 `docs/` 根目录。
2. 等这一阶段被执行并被新的计划取代后，再移动到：
   - `docs/archived/2026-04-early-stage/` 或
   - `docs/archived/2026-04-exec-reports/`
   具体按内容性质决定。
3. 每完成一个子阶段后，同步更新：
   - `docs/STATUS.md`
   - `docs/DESIGN.md`
   - `docs/CHANGELOG.md`
   - `docs/hermes.md`

---

## 8. 推荐的实际开工顺序

如果只允许按最小风险推进，我建议严格按下面顺序做：
1. continuous confidence + agreement-aware support
2. RGB/raw confidence 与 depth/train-mask 语义分离
3. confidence-aware densify
4. StageB 两段式 curriculum
5. pseudo branch 的 local Gaussian gating
6. active-support-aware depth reweight
7. 再考虑更密 pseudo / Gaussian refinement / 最小版 SPGM

一句话压缩：
**先把 pseudo supervision 语义做对，再把 StageB 变稳，再谈更强的 map-side 管理。**





补充进展（2026-04-14 晚）：已完成 8-frame StageA-20iter baseline vs retuned-confaware 小对照。当前 retuned conf-aware densify 已从首轮过严状态回调到可用区间（dense_valid_ratio 约 `2.27% -> 1.63%`，不再塌到 `0.1%` 以下），但 conf-aware 组在 20iter StageA 上 loss 仍高于 baseline，因此还不能直接进入 StageB curriculum；下一步应先做一轮更系统的 8-frame E1 短跑/小网格回调，再决定是否进入 plan 第4步。

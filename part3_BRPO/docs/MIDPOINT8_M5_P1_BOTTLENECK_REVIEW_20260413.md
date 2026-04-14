# Midpoint8 M5 P1 bottleneck review（2026-04-13 夜）

## 1. 复查目标
在 P0 修复确认 `StageA -> A.5 -> B` 顺序 handoff 已正确之后，继续沿 `pseudo cache -> source-aware loss -> A.5 / B refine -> replay` 的真实代码链路复查：为什么这条 pipeline 仍然整体为负，以及当前主要瓶颈到底在“方法”还是“信号”。

## 2. 复查范围
- 代码链路：
  - `pseudo_branch/brpo_train_mask.py`
  - `pseudo_branch/brpo_depth_densify.py`
  - `pseudo_branch/brpo_depth_target.py`
  - `pseudo_branch/pseudo_loss_v2.py`
  - `pseudo_branch/pseudo_refine_scheduler.py`
  - `pseudo_branch/gaussian_param_groups.py`
  - `scripts/run_pseudo_refinement_v2.py`
- 数据/实验链路：
  - midpoint8 M5 pseudo cache
  - repaired sequential rerun (`repair_seq_rerun/`)

## 3. 代码链路上的关键事实
### 3.1 pseudo supervision 只在 train mask 内生效
`build_stageA_loss_source_aware(...)` 中：
- RGB loss 使用 `masked_rgb_loss(render_rgb, target_rgb, confidence_mask, ...)`
- depth loss 使用 `masked_depth_loss_by_source(..., confidence_mask, depth_source_map, ...)`

因此 pseudo RGB 和 pseudo depth 都只在 `train_confidence_mask_brpo_fused.npy` 的非零区域生效。mask 外整张图对 pseudo supervision 完全无约束。

### 3.2 source-aware depth 的 fallback 默认不参与优化
当前配置：
- `lambda_depth_seed = 1.0`
- `lambda_depth_dense = 0.35`
- `lambda_depth_fallback = 0.0`

也就是说，即使 `target_depth_for_refine_v2.npy` 在 fallback 区域回退成 `render_depth`，这些像素默认不提供 depth 梯度。

### 3.3 A.5 / B 会直接打开全局 Gaussian 参数
`build_micro_gaussian_param_groups(...)` 直接把：
- `gaussians._xyz`
- `gaussians._opacity`

作为训练参数组加入 optimizer，没有额外的 per-view / per-region gating。虽然真实梯度仍依赖可见点，但从优化对象角度看，这是对整个 Gaussian 集合做全局可训练，而 supervision 只来自 8 个 pseudo views + 少量 real views。

## 4. midpoint8 M5 pseudo signal 的量化结果
统计对象：
`.../re10k1__internal_afteropt__midpoint_proto_v1/pseudo_cache/samples/*`

8 个 pseudo frames 的平均统计：
- `mean_confidence_nonzero_ratio = 0.1870`
- `mean_target_depth_seed_ratio = 0.0150`
- `mean_target_depth_dense_ratio = 0.0199`
- `mean_target_depth_verified_ratio = 0.0349`
- `mean_target_depth_render_fallback_ratio = 0.9651`

解释：
1. pseudo RGB supervision 只覆盖约 18.7% 像素；
2. 真正具备 non-fallback depth 的像素只有约 3.49%；
3. 其余约 96.5% 像素在 depth target 上都是 render fallback。

进一步看 train-mask 内部 source 构成：
- mask 内 seed 像素约占 `8.02%`
- mask 内 dense 像素约占 `10.40%`
- mask 内 fallback 像素约占 `81.58%`

由于 `lambda_depth_fallback=0`，depth loss 真正吃到的有效区域只剩 train-mask 中的前两部分。

## 5. 一个更关键的数学事实：depth 信号不仅稀，而且很弱
对所有 midpoint8 pseudo samples，统计 `target_depth_for_refine_v2` 相对 `render_depth` 的变化量：
- 在有非 fallback support 的区域内，平均相对深度改变量：`mean ≈ 1.53%`
- support 区域平均只占全图：`3.49%`
- 因此折算到整张图，平均“全图相对修正量”只有：
  - `0.0349 × 0.0153 ≈ 0.00053`
  - 也就是约 `0.053%` 的全图平均相对修正量

这说明当前 midpoint8 M5 target 对“现有 render geometry”的修正其实非常小：
- 不是只有 coverage 小；
- 就连有 supervision 的区域，target 也大多只是在 render depth 周围做很小的校正。

所以从优化角度看，现在的 pseudo depth 更像是“微弱 nudging”，而不是足以驱动明显几何重排的强信号。

## 6. repaired sequential rerun 反映出的 refine 瓶颈
### 6.1 StageA 的 pose 确实在动，但量级很小
`stageA80/stageA_history.json`：
- `mean_stageA_scene_scale = 3.1961`
- `mean_trans_norm = 0.00376`
- 相对 scene scale 约 `0.00376 / 3.1961 = 0.00118`

即 StageA 平均平移改变量约为 scene scale 的 `0.118%`，属于很小的微调。

### 6.2 A.5 / B 会对大量 Gaussian 产生全局扰动
对 replay baseline PLY 做逐点比对：

A.5(80, from StageA) 相对 baseline：
- mean xyz drift = `0.00394`
- `74.13%` 的 gaussians 有 `> 1e-3` 的 xyz 变化
- `35.14%` 的 gaussians 有 `> 5e-3` 的 xyz 变化
- mean |opacity drift| = `0.00770`

StageB120(from A.5) 相对 baseline：
- mean xyz drift = `0.00734`
- `86.24%` 的 gaussians 有 `> 1e-3` 的 xyz 变化
- `63.40%` 的 gaussians 有 `> 5e-3` 的 xyz 变化
- mean |opacity drift| = `0.01560`

这和 supervision 的稀疏性形成明显不对称：
- supervision：只覆盖 8 个 pseudo views，且 depth 有效区域仅 3.49%；
- 可训练对象：却是一个包含 33452 个点的全局 Gaussian 集合。

所以当前 refine 更像是在用非常局部、非常弱的信号去推动广泛的全局参数扰动，这很容易造成：
1. pseudo-side loss 下降；
2. replay / held-out 视角退化。

### 6.3 loss 下降与 replay 改善已经被实证解耦
修复 handoff 之后的 sequential rerun：
- StageA.5 pseudo loss 明显下降，但 replay 仍低于 baseline；
- StageB 总 loss / pseudo depth 继续下降，但 replay 进一步退化。

因此现在可以更确定地说：
当前 pipeline 的核心问题不是“loss 没降”，而是“loss 在当前 supervision 口径下并不对应我们真正关心的几何/重建泛化目标”。

## 7. 当前最可能的主瓶颈排序
### 瓶颈 A：pseudo depth 信号强度不足（最核心）
表现：
- valid depth support 仅 `3.49%`
- support 区域相对修正量平均仅 `1.53%`
- fallback 占 `96.51%`

判断：
当前 midpoint8 M5 在这个 re10k case 上更像“轻微修补 render depth”，不足以承担强几何监督的角色。

### 瓶颈 B：监督作用域与优化作用域严重不匹配
表现：
- pseudo supervision 只看 mask 内 18.7% 区域；
- 但 A.5 / B 打开的是全局 xyz / opacity 参数；
- 结果是大比例 Gaussian 被推动，而 held-out replay 退化。

判断：
这是一个典型的 underconstrained global refinement 问题。

### 瓶颈 C：当前 StageB 的 real anchor 仍然太弱
表现：
- `num_real_views = 2`
- sparse real RGB anchor 存在，但不能阻止全局退化

判断：
它能提供一些 regularization，但不足以与 pseudo-side 局部过拟合形成强对抗，更谈不上 paper-style 的 joint geometry anchoring。

## 8. 这轮 P1 复查后的方法判断
目前更接近下面这个结论，而不是“再扫几个 lr 就行”：

1. handoff bug 确实已经修掉，但不是主终局瓶颈；
2. 当前 midpoint8 M5 pseudo signal 在这个 case 上“覆盖率低 + 修正幅度小”；
3. 现有 A.5 / B refine 把这种弱信号施加到全局 Gaussian 上，导致 pseudo loss 能降，但 replay 容易掉；
4. 因此现在首先要审视的是“监督设计与优化作用域是否匹配”，而不是继续在同一设置上做大量参数扫描。

## 9. 建议的后续验证顺序
### 9.1 先做 signal-side 验证，而不是直接继续调 refine lr
优先做两类验证：
1. pseudo 选点策略复查：
   - midpoint 之外，尝试更靠近高视差/高不确定区的 pseudo 选点
   - 不只看每 gap 取中点，也看 gap 内双点 / tertile / visibility-aware 方案
2. pseudo signal 质量分桶：
   - 逐 pseudo frame 统计 support ratio、relative correction magnitude、replay 邻域收益
   - 看是否存在“少数 pseudo 有效，大多数 pseudo 只是弱噪声”的情况

### 9.2 如果继续 refine，先缩小可训练作用域
比起继续放大迭代数，更值得优先试：
1. 只允许 pseudo 可见区域对应的 Gaussian 子集参与 A.5；
2. 或做更强的 xyz/update gating，而不是直接全局开 `gaussians._xyz`；
3. StageB 中增强 real-anchor 密度/频率，再观察是否能抑制 replay 退化。

### 9.3 把“signal sanity check”变成长跑前固定 gate
以后每轮新 pseudo cache / 新 staged design，在长跑前至少先过：
1. support ratio gate
2. target-vs-render correction magnitude gate
3. stage handoff continuity gate
4. evaluation-artifact continuity gate
5. short replay non-regression gate

否则很容易再次出现：实验跑了很久，最后才发现 supervision 本身几乎不提供有效几何驱动力。

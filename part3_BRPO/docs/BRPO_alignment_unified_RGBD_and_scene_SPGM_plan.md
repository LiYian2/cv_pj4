# BRPO 对齐分析与中期落地方案（Unified RGB-D + Scene-level SPGM）

## 0. 文档目的

这份文档面向后续 agent 的工程分析与执行，回答两个问题：

1. **沿 BRPO 原理看，我当前系统与 BRPO 真正差在哪里**，而且这种差异不是“工程写法不同”，而是**优化对象、约束形式、信息流角色**上的理论差异。  
2. **如果要继续向 BRPO 靠拢，应该如何在当前 repo 的代码流上逐步落地**，并明确指出是**替换哪个模块**、还是**新增到哪个模块之后**。

本文重点围绕两个中期方向：

- **方向 A：把 current RGB-first weak-depth refine 往统一的 confidence-weighted RGB-D joint refinement 靠拢**
- **方向 B：把 SPGM 从“更新抑制器”推进成“场景级 Gaussian 管理器”**

---

## 0.5 额外内容：此部分内容是文档生成后加入，无分析信息或落地规划，需要你针对这些内容做理论分析并评估可行性，是否值得落地。

###0.5.1 Supervision 密度差距是最根本的分歧。 BRPO 用 Difix3D 在像素级别补全整张伪帧，再经 UNet 提炼，最终送入 diffusion；confidence mask 过滤掉几何不一致的像素后，仍然有相当密度的有效像素参与 loss。你的链路中即使经过 M5 densify，train-mask 内有效深度覆盖也只到 ~14\%，而 RGB mask 的 verified 比例更低（~2\%）。这个差距不是工程落地的细节，而是 supervision signal 本身的量级差——BRPO 的 confidence mask 是在 pixel-complete 伪帧上做过滤，你的 signal_v2 是从极稀疏的 verified correspondences 出发往外扩。起点就不同。探索：重新审视 depth supervision 的获取策略。 当前 M5 densify 从 ~2\% 扩到 ~14\% 是通过空间传播，但你的 design 原则里明确"不对 raw RGB mask 直接 densify，应优先几何约束的 support/depth expand"。这个判断是对的——但"几何约束的扩张"具体指什么还没有落地。一个与 BRPO 更接近的方向是：利用 MASt3R（你已经用于 pose estimation）的 dense match 在伪帧和相邻真实帧之间建立更密集的 correspondence，这可以把 verified depth 的覆盖率提高一个量级，同时保持几何一致性的约束语义。
###0.5.2 Loss 归一化逻辑。 BRPO 的 loss 写成 ||Cm ⊙ (It - Ît)||₁ / ||Cm||₁，分母归一化是专门设计来处理稀疏 mask 的——当 Cm 非零区域极少时，归一化使这些区域的梯度贡献保持与全像素 loss 相当的量级，不会被淹没。如果你的实现没有这个归一化（或者归一化方式不同），在 verified≈2\% 的情况下，整个 pseudo supervision 对 loss 的贡献几乎为零，这直接解释了为什么改善是噪音级别。 首先要解决的是 loss 归一化。 验证你的实现是否做了 / ||Cm||₁ 归一化。如果没有，在 verified≈2\% 的 mask 下，pseudo supervision 贡献的梯度在数值上几乎可以忽略，这是最便宜、最直接的修复。
###0.5.3 SPGM 的设计语义完全不同。 BRPO 的 SPGM 是 stochastic pruning：通过 Bernoulli 采样直接将低分 Gaussian 的 opacity 置零，从而从物理上减少这些 Gaussian 参与渲染的概率。这是一个 during-training 的结构筛选机制，目的是让有限的 pseudo supervision 集中作用于最需要修正的区域。你当前的 repair A (1,1,1) 实际上 keep ratio=100\%，weight_floor=0.25 只是对梯度做了软性下限，不存在任何结构筛选。即使 selector-first 路线激活，你走的也是梯度 gating 而非 opacity stochastic drop——梯度门控和 Bernoulli 不等价，前者让所有 Gaussian 都能收到梯度（只是不同强度），后者在采样层面切断了低质 Gaussian 的更新通路。方案：向 BRPO 的 stochastic drop 靠拢。 selector-first 方向上，你已经发现 far=0.90 进入了 parity 区间，说明极保守的筛选是有效的，但"伤 replay"的问题来自 ranking 质量不足导致误删了有用的 Gaussian。下一步可以考虑不做 hard keep/drop，而是实现 BRPO 式的 Bernoulli opacity masking：不把 Gaussian 从训练集里物理删除，而是在前向时按 drop probability 随机屏蔽 opacity，这样既有结构筛选效果，又保留了梯度回传路径，能减少误删导致的 replay 退化。
###0.5.4 伪帧在优化中的角色定位不同。 BRPO 把 pseudo frame 当作"普通训练帧"直接插入 joint optimization，confidence mask 在 loss 里统一过滤；整个优化过程是连续的，不区分"pseudo-only"和"joint"阶段。你的 StageA.5（pseudo-only backward，local gating）+ StageB（joint refine）这个两阶段结构在设计上有其合理性（先不污染 real branch），但实际效果是：当 A.5 的 pseudo signal 太弱（2\% coverage + 短 iter），传递到 StageB 的 Gaussian 状态几乎没有变化，StageB 本身只有 40 iter 也不足以在此基础上继续收益。两个阶段相互依赖但各自都不够强，比 BRPO 一体化的 joint optimization 脆弱得多。方案：关于两阶段结构的取舍。 如果 A.5 的 pseudo-only backward 产生的 Gaussian 变化量在 replay 里不可测（这很可能是现状），可以考虑把 A.5 的迭代预算合并到 StageB 的 joint optimization 里，而在 joint 阶段用 local gating 替代 A.5 的 scope 限制——即在 joint refine 里给 pseudo loss 加 local Gaussian mask（visibility 约束），而不是用单独的 pseudo-only 阶段。这样 pseudo 和 real 的 supervision 在同一个优化循环里相互正则化，避免了弱信号在短窗口里无法积累的问题。

## 1. 当前代码流与两个中期方向的插点

### 1.1 当前主线代码流

按当前 repo 与文档，主线已经稳定为：

```text
part2 internal cache
↓
internal prepare: select → difix → fusion → verify → pack
↓
pseudo_cache + signal_v2
↓
StageA (pose / exposure only)
↓
StageA.5 (local Gaussian micro-tune + local gating / SPGM)
↓
StageB (joint refine + bounded schedule)
↓
refined_gaussians.ply + replay eval
```

这条总线在文档中已被固定，且明确了：
- **StageA 不更新 Gaussian**，因此 StageA-only replay 不是有效判优指标。
- 真正和 Gaussian 相关的 refine 都在 **StageA.5 / StageB**。  
- 当前 canonical StageB baseline 是 **RGB-only v2 + gated_rgb0192 + post40_lr03_120**。  
- 当前 SPGM 已经不是“没接上”，而是进入了 **repair A / selector-first / support_blend** 的细调阶段。  

### 1.2 当前信号构建位置

`part3_BRPO/scripts/build_brpo_v2_signal_from_internal_cache.py` 已经是 `signal_v2` 的核心 builder。它当前做两件事：

1. 通过 `build_rgb_mask_from_correspondences(...)` 从 fused RGB 与左右 reference 做 correspondence，写出 RGB confidence / summary。  
2. 通过 `build_depth_supervision_v2(...)` 使用 `projected_depth_left/right`、`fusion_weight_left/right`、`overlap_mask_left/right` 以及 `raw_rgb_confidence_v2` 构建 depth supervision，并写出 depth summary。  

也就是说，你现在已经有了一个**共同上游**：  
**fused RGB → RGB mask** 与 **fused geometry artifacts → depth supervision** 是同源的，但 downstream 还没有真正变成统一约束。

### 1.3 当前 refine / gating / SPGM 的插点

`part3_BRPO/scripts/run_pseudo_refinement_v2.py` 当前的关键点是：

- StageA / StageA.5 / StageB 都由它消费 pseudo_cache / signal_v2。
- 在 pseudo loss backward 之后，会进入 `maybe_apply_pseudo_local_gating(...)`。
- 当前这个函数已经分成两条分支：
  - **legacy visibility-union path**：`build_visibility_weight_map(...) -> apply_gaussian_grad_mask(...)`
  - **SPGM path**：`collect_spgm_stats(...) -> build_spgm_importance_score(...) -> build_spgm_grad_weights(...) -> apply_gaussian_grad_mask(...)`

也就是说：

- **方向 A（统一 RGB-D refine）** 的插点应该主要在 **signal builder + loss consumer** 之间。  
- **方向 B（scene-level SPGM）** 的插点应该主要在 **run_pseudo_refinement_v2.py 的 pseudo backward 之后、optimizer.step 之前**，并在必要时增加一步 **state-level Gaussian management**。  

---

## 2. 方向 A：向统一 confidence-weighted RGB-D joint refinement 靠拢

### 2.1 BRPO 的理论目标是什么

BRPO 在 joint refinement 阶段的核心形式不是“RGB 一路、depth 一路，各自弱约束进来”，而是：

\[
L = \beta L_{rgb} + (1-\beta)L_d + \lambda_s L_s
\]

其中：

\[
L_{rgb} = \frac{\| C_m \odot (I_t - \hat I_t)\|_1}{\|C_m\|_1}
\]

\[
L_d = \frac{\| C_m \odot (D_t - \hat D_t)\|_1}{\|C_m\|_1}
\]

这里最关键的不是公式表面，而是**同一个 confidence mask \(C_m\)** 同时定义了：

- 哪些像素的 RGB 残差值得信
- 哪些像素的 depth 残差值得信
- 因而哪些局部几何/外观更新可以被写回 Gaussian 与 pose

这意味着 BRPO 不是“有 RGB loss 再加一个 depth loss”，而是在解一个更强的问题：

> **在同一个可信观测集合上，同时对 photometric inconsistency 和 geometric inconsistency 施加约束。**

这在理论上会带来两个重要后果：

#### （1）同一像素支持的光度梯度与几何梯度是对齐的
如果 RGB 认为某个像素可靠、depth 却认为它不可靠，那么两个梯度的可信支持域就不一致。  
BRPO 的做法是让 \(C_m\) 统一定义这个域，因此优化器看到的是一个**一致的“可信观测子集”**。

#### （2）depth 不再是 sidecar，而是和 RGB 共享同一观测语义
如果 depth 只是一个额外的 narrow side loss，那么它更像 regularizer。  
而在 BRPO 里，depth 被放在与 RGB 同一 confidence-weighted 观测集合里，因此它在 joint optimization 中扮演的是**几何主约束的一部分**，不是边缘辅助项。

### 2.2 你当前系统与 BRPO 在这个方向上的真正理论差异

#### 差异 1：你当前最优链路仍然是 RGB-first，而不是 unified RGB-D
当前文档已经固化：

- **RGB-only v2 可保留**
- **full v2 depth verified 约 2%，过窄**
- 若扩张，应优先做几何约束的 support/depth expand，而不是 raw RGB densify

这意味着你当前最优系统实际上在解的是：

> **“先相信一条较干净但很窄的 RGB correspondence 路，再让 depth 以很弱、很局部的方式附着进去。”**

这和 BRPO 的理论对象不同。  
BRPO 解的是：

> **“在同一 confidence 定义下，同时更新 appearance 与 geometry。”**

这不是工程细节差异，而是**优化问题本身不同**。

#### 差异 2：你当前的 confidence 还没有升级成统一观测测度
当前 `signal_v2` 虽然已经把 RGB mask 与 depth supervision 都从 fused outputs 中单独构建出来，但 downstream 仍是：

- RGB mask 有自己的使用口径
- depth target / depth supervision 有自己的使用口径
- StageB 的 canonical best 事实上还是 `RGB-only v2`

理论上，这意味着你当前系统里还不存在一个真正的：

\[
\mu_{\text{trusted}}(p)
\]

也就是“像素 \(p\) 属于可信观测集的程度”的统一定义。  
而 BRPO 的 \(C_m\) 就是在扮演这个统一测度。

当前你的系统更像：

\[
\mu_{\text{rgb}}(p),\quad \mu_{\text{depth}}(p)
\]

但还没有：

\[
\mu_{\text{joint}}(p)
\]

这会导致一个后果：  
同一 pseudo frame 里，RGB 和 depth 对“哪些地方值得写回几何”并没有完全共享语义。

#### 差异 3：你现在的 depth 更像局部 regularizer，不像 geometry-defining observation
当 verified≈2% 时，depth 对优化器的角色通常会退化成：

- “如果某些地方刚好有很强几何证据，就给一点修正”
- 但它不足以改变大部分几何更新的主方向

在这种情况下，depth 对系统的作用更像一个 **sparse local regularizer**，而不是 **共同定义可信几何观测集的主项**。  
这也是为什么你当前会看到：

- 改善很轻
- RGB-only v2 经常仍是 best branch
- depth 扩不起来时，系统大多只会做轻微保守修正

这不是简单的“coverage 低”，而是**geometry information 还没有进入系统的主优化坐标系**。

### 2.3 要怎么向 BRPO 靠拢：理论上应该改成什么

目标不是“粗暴扩大 raw RGB coverage”，而是把当前系统改写成：

\[
C_{\text{joint}}(p) = g\big(C_{\text{rgb}}(p),\ C_{\text{geo}}(p),\ S_{\text{support}}(p)\big)
\]

其中：

- \(C_{\text{rgb}}(p)\)：当前 fused-RGB correspondence confidence  
- \(C_{\text{geo}}(p)\)：左右 projected depth / overlap / reprojection consistency 给出的几何一致性  
- \(S_{\text{support}}(p)\)：该像素是否处于多视角有效支持区域  

然后在 loss 里改成：

\[
L_{rgb}^{joint} = \frac{\| C_{joint} \odot (I_t - \hat I_t)\|_1}{\|C_{joint}\|_1}
\]

\[
L_{d}^{joint} = \frac{\| C_{joint} \odot (D_t - \hat D_t)\|_1}{\|C_{joint}\|_1}
\]

也就是说：

- RGB 不再只用自己的窄 mask
- depth 不再只靠自己的窄 verified 区域
- 而是由一个**共享的 joint confidence** 定义可信观测集

这样做的理论意义在于：

1. **把 depth 从 sidecar 拉回主约束**
2. **让 photometric 与 geometric residual 共享可信支持域**
3. **让后续 StageB 的 pseudo branch 更新方向更接近“几何+外观共同可信”的区域**

### 2.4 按当前 repo 的工程落地：应该改哪里

#### A-1. 新增 joint confidence builder，而不是推翻 signal_v2

**不要推翻现在的 `build_brpo_v2_signal_from_internal_cache.py`。**  
它已经是最自然的接入口。

建议新增模块：

```text
part3_BRPO/pseudo_branch/brpo_v2_signal/joint_confidence.py
```

建议在里面实现：

- `build_joint_confidence_from_rgb_and_depth(...)`
- `build_joint_depth_target(...)`
- `write_joint_signal_outputs(...)`

#### A-2. 在 `build_brpo_v2_signal_from_internal_cache.py` 里增加 joint 分支

当前流程已经有：

1. `rgb_result = build_rgb_mask_from_correspondences(...)`
2. `depth_result = build_depth_supervision_v2(...)`

建议在这两步之后加第三步：

3. `joint_result = build_joint_confidence_from_rgb_and_depth(rgb_result, depth_result, fusion artifacts, overlap masks, projected validity, ...)`

输出到：

```text
signal_v2/frame_xxx/
├── raw_rgb_confidence_v2.npy
├── target_depth_for_refine_v2_brpo.npy
├── depth_supervision_mask_v2_brpo.npy
├── joint_confidence_v2.npy
├── joint_confidence_cont_v2.npy
├── joint_depth_target_v2.npy
└── joint_meta_v2.json
```

注意：

- `joint_confidence_v2.npy` 不是简单取 min/max，而应体现：
  - RGB correspondence 可靠性
  - 左右投影深度一致性
  - overlap / projected validity
  - 必要时对 single-side / both-side 支持做分级
- `joint_depth_target_v2.npy` 应是 **joint confidence 支持域下的最终 depth target**

#### A-3. 在 loss consumer 里新增 “joint mode”

当前 refine consumer 主要是：

```text
part3_BRPO/pseudo_branch/pseudo_loss_v2.py
part3_BRPO/scripts/run_pseudo_refinement_v2.py
```

这里要加的不是“再多一个 legacy/depth mode”，而是显式支持：

- `stageA_rgb_mask_mode = joint_confidence_v2`
- `stageA_depth_mask_mode = joint_confidence_v2`
- `stageA_target_depth_mode = joint_depth_v2`
- 或更明确的：
  - `stage_joint_confidence_mode = brpo_joint_v1`

更好的方式是：  
在 `pseudo_loss_v2.py` 里增加一个明确的 joint path，例如：

```python
if joint_confidence_mode == "brpo_joint_v1":
    rgb_mask = load_joint_confidence(...)
    depth_mask = rgb_mask
    target_depth = load_joint_depth_target(...)
```

这样 RGB 与 depth 真正复用同一个 support 域。

#### A-4. support/depth expand 应该做在哪里

如果要扩张，不应先 densify raw RGB。  
应新增一个**几何约束下的 support expand 模块**，建议放在：

```text
part3_BRPO/pseudo_branch/brpo_v2_signal/support_expand.py
```

它的输入应来自现有 artifacts：

- `projected_depth_left.npy`
- `projected_depth_right.npy`
- `fusion_weight_left/right.npy`
- `overlap_mask_left/right.npy`
- `raw_rgb_confidence_v2.npy`

它的角色是：

- 先由高精度 RGB+geometry seed 定义初始 support
- 再在几何一致区域内做有限扩张
- 让 depth 的 usable support 不再只有 verified≈2%

这一步是为了让 depth 真正进入主约束，而不是只在 extremely sparse 区域里当 sidecar。

### 2.5 这个方向的分阶段落地顺序

#### Phase A1：只新增 joint confidence，不改 SPGM
目标：先验证 unified RGB-D observation 是否比 RGB-first 更稳定。

- 保留当前 StageA.5 / StageB schedule
- 保留 current SPGM / repair A 不变
- 只替换 pseudo loss 的 mask / target 入口
- 看：
  - depth non-fallback ratio 是否提高
  - `loss_depth_dense` 是否更可用
  - StageB-40 / StageB-120 是否比 RGB-only v2 更稳

#### Phase A2：做 support/depth expand
目标：让 `joint_confidence` 从“统一但仍过窄”推进到“统一且可用”。

- 不 densify raw RGB
- 只在 geometry-consistent 区域扩 depth support
- 先在 Re10k 做小 compare，再考虑 DL3DV

#### Phase A3：再决定是否需要 full RGB-D StageB
只有当 A1/A2 证明 unified joint confidence 真正提供额外有效几何信息，才值得把它升级为 canonical StageB branch。

---

## 3. 方向 B：把 SPGM 从更新抑制器推进成场景级 Gaussian 管理器

### 3.1 BRPO 的理论目标是什么

BRPO 的 SPGM 不是“给当前 active set 多乘一个权重”。  
它在理论上分成四步：

#### （1）按深度分布做 1D quantile partition
设 Gaussian 深度为 \(z_i\)，将经验分布按 quantile 分成 \(K\) 个 cluster。  
论文用的是 1D optimal-transport / Wasserstein 视角来说明：  
在一维上，quantile split 对 contiguous clustering 是合理近似。

#### （2）计算 depth score
\[
\hat s_i^{(z)} = 1 - \frac{d_i - d_{min}}{d_{max} - d_{min} + \delta}
\]

#### （3）计算 density entropy 与 density score
\[
H(\rho) = - \frac{1}{\log B}\sum_{b=1}^{B} p_b \log(p_b + \varepsilon)
\]

\[
\hat s_i^{(\rho)} = \tilde \rho_i (1 - \beta \bar H) + \gamma \bar H
\]

#### （4）融合成 unified importance，再做 cluster-aware attenuation
\[
S_i = \alpha \hat s_i^{(z)} + (1-\alpha)\hat s_i^{(\rho)}
\]

然后：

\[
p_i^{drop} = r \cdot w_{cluster(i)} \cdot S_i
\]

并对 opacity 做随机衰减：

\[
\alpha_i \leftarrow \alpha_i \cdot m_i,\quad m_i \sim \text{Bernoulli}(1-p_i^{drop})
\]

这套机制的理论角色不是“帮 loss 更平滑”，而是：

> **对整张 Gaussian population 进行 scene-aware population control。**

也就是说，SPGM 决定的不是“这个 pseudo view 要不要训”，而是：

- 哪些 Gaussian 更值得保留
- 哪些 Gaussian 该被弱化
- 哪些 depth cluster 该更保守
- 哪些 density pattern 说明它可能是 floating / under-supported Gaussian

### 3.2 你当前系统与 BRPO 在 SPGM 上的真正理论差异

#### 差异 1：你当前 SPGM 仍主要是 pseudo-branch grad modulation
当前 `run_pseudo_refinement_v2.py` 的 SPGM 路径已经是：

```text
gate_results
→ collect_spgm_stats
→ build_spgm_importance_score
→ build_spgm_grad_weights
→ apply_gaussian_grad_mask
```

也就是说，你当前 SPGM 的最终动作仍然是：

> **对 pseudo backward 之后的 Gaussian grad 做 weighted mask。**

这说明当前 SPGM 的本体角色仍然是：

- 对 pseudo branch 的更新进行限制
- 属于 optimizer-side grad manager

而 BRPO 的 SPGM 理论角色更强：

- 它是 scene-level Gaussian population manager
- 决定的是 Gaussian 的保留 / 弱化 / cluster-wise attenuation

所以差异不在于“你有没有 stats/score/policy 三件套”，而在于：

> **你当前 SPGM 的输出主要还是一个 pseudo-branch grad weight，而不是对 scene Gaussian population 的直接管理。**

#### 差异 2：你当前 SPGM 仍然是 active-set 后处理，不是 scene-wide Gaussian prior
从 `P2-L/P2-M/P2-R/P2-S` 的实验结论看：

- 原始 deterministic SPGM 更像 suppressor
- repair A 把 suppressive policy 变温和
- selector-first 与 support_blend 逐步逼近 parity
- 当前最好的 selector-first 也只是 `far=0.90` practical parity

这说明当前 SPGM 的主要自由度仍然是：

- 在已有 active set 上压多重
- 或者在 far cluster 上删多少

换句话说，它主要是在做：

\[
\text{given selected active set} \Rightarrow \text{how much to update}
\]

而 BRPO 的 SPGM 更像：

\[
\text{scene Gaussian state} \Rightarrow \text{which Gaussian population structure should be preserved / weakened}
\]

这两者的对象不同。

#### 差异 3：你当前的 density 还是 proxy，且没有真正进入 Gaussian state management
当前 `score.py` 已支持：

- `density_mode='opacity_support'`
- `density_mode='support'`

而 `importance_raw` 由 depth score 和 density score 融合得到。  
这已经是一个不错的第一步，但它与 BRPO 论文的理论 still 有差距：

1. 当前 density 仍是 proxy（opacity/support），不是真正面向 Gaussian scene structure 的 population statistic。  
2. 它的输出目前主要还是给 grad weighting / selector ranking 用。  
3. 还没有升级成**改变 Gaussian population state**的控制变量。  

因此当前的 `score.py` 已经有 BRPO 的影子，但它还没有从“score for weighting”变成“score for scene management”。

#### 差异 4：你当前还没有 BRPO 式的 stochastic attenuation，也没有真正的 \(L_s\) 角色
当前 docs 已明确：

- 暂不推进 stochastic
- 暂不推进 xyz+opacity
- 继续以 repair A 为 anchor，围绕 selector-first 做极保守 compare

这在工程上是对的，但理论上也说明：

- 你当前还没有进入 BRPO 的 stochastic attenuation 语义
- 也没有一个和 BRPO \(L_s\) 同层级的 Gaussian scale regularization 主项

于是当前 SPGM 仍然主要影响“这一步 pseudo branch 的梯度”，而不是“Gaussian 作为场景组成单元的结构状态”。

### 3.3 要怎么向 BRPO 靠拢：理论上应该改成什么

目标不是立刻把 stochastic drop 硬搬进来，而是把 SPGM 拆成两层角色：

#### 层 1：Update Permission Layer（谁允许被更新）
这层仍然是你当前已有的：

- `active_mask`
- `selector_quantile`
- `support_blend`
- `grad weighting`

它回答的是：

> **这一轮 pseudo refinement，哪些 Gaussian 可以被写入梯度？**

#### 层 2：Scene State Management Layer（谁该被保留/弱化）
这才是向 BRPO 靠拢的关键新增层。  
它回答的是：

> **从整张图的 depth-density 统计看，哪些 Gaussian 在场景层面更不可信、更该衰减或更低 lr？**

理论上，这一层至少应提供两个输出：

1. **update weight / lr scale**
   \[
   \eta_i^{update} = f(S_i, cluster(i), support_i)
   \]

2. **state attenuation / population control**
   \[
   a_i^{state} = g(S_i, cluster(i))
   \]

其中：

- \(\eta_i^{update}\) 作用在 grad / lr
- \(a_i^{state}\) 作用在 opacity / scale / prune priority / keep priority

这样，SPGM 才不再只是“这一轮梯度压一下”，而是变成“场景级 Gaussian 管理器”。

### 3.4 按当前 repo 的工程落地：应该改哪里

#### B-1. 保留现有 `spgm/{stats,score,policy}.py`，但加一个 manager 层

当前已有：

```text
part3_BRPO/pseudo_branch/spgm/
├── stats.py
├── score.py
└── policy.py
```

建议新增：

```text
part3_BRPO/pseudo_branch/spgm/manager.py
```

它负责把当前已有的 `score.py` 从“打分器”升级成“状态管理器”。

建议新增两个函数：

- `build_spgm_update_policy(...)`
- `apply_spgm_state_management(...)`

其中：

- `build_spgm_update_policy(...)` 仍输出当前 grad 权重 / selector mask  
- `apply_spgm_state_management(...)` 额外输出或直接施加：
  - opacity attenuation
  - per-cluster lr scaling
  - prune / keep priority adjustment
  - optional scale regularization target

#### B-2. 在 `run_pseudo_refinement_v2.py` 中把 SPGM 分成两步

当前逻辑是：

```text
pseudo backward
→ collect_spgm_stats
→ build_spgm_importance_score
→ build_spgm_grad_weights
→ apply_gaussian_grad_mask
→ optimizer.step
```

建议改成：

```text
pseudo backward
→ collect_spgm_stats
→ build_spgm_importance_score
→ build_spgm_update_policy
→ apply_gaussian_grad_mask
→ apply_spgm_state_management
→ optimizer.step
```

这里的 `apply_spgm_state_management(...)` 第一版不一定真的直接改参数，也可以先做：

- per-cluster effective lr scaling
- low-score far cluster 的 opacity decay factor 记录
- prune priority / keep priority 标记写 history

也就是先把 manager 层插进来，再逐步增强它。

#### B-3. `stats.py` 要从 accepted pseudo view 扩到 scene-aware stats

当前 `collect_spgm_stats(...)` 是从 accepted pseudo views + render packages 提取：

- support_count
- depth_value
- density_proxy
- active_mask

第一版是合理的，但如果要更接近 BRPO 的 scene-level SPGM，建议把输入扩成：

- accepted pseudo views
- real anchor views
- current train window / local window
- 可选的 scene-wide visibility summary

目标是让 `support_count / depth_value / density_proxy` 不再只是“pseudo branch 的局部统计”，而是更接近：

> **当前 scene / local window 里的 Gaussian population statistics**

你不需要一步做到 full-scene global，  
但至少要从“pseudo accepted subset”往“current train window population”走。

#### B-4. `score.py` 需要补两类 scene-level statistic

当前 `score.py` 已有：

- depth partition
- density entropy
- `importance_score`
- `ranking_score`
- `support_blend`

下一步最值得补的是：

##### （a）更像 scene-density 的 density proxy
当前的 `opacity_support` 太像 branch-local proxy。  
如果 Gaussian state 中能稳定取到 scale / covariance 体积，可以引入：

\[
\rho_i \propto \frac{\alpha_i}{\text{vol}_i + \epsilon} \times support_i
\]

让 density 更接近“结构密度”而不是“support count 的变形”。

##### （b）state score 与 ranking score 的真正分离
当前 `importance_score` 与 `ranking_score` 已经做了 decouple，这是很好的起点。  
下一步要进一步明确：

- `ranking_score`：给 selector / active-set 用
- `state_score`：给 opacity attenuation / lr scaling / prune priority 用

也就是说，`score.py` 最终不应只输出：

- `importance_score`
- `ranking_score`

还应输出：

- `state_score`

这样 policy 层和 manager 层才能真正分开。

#### B-5. `policy.py` 应继续保留 selector-first，但不要把它误当成完整 SPGM

当前 `policy.py` 的 `selector_quantile` + `selector_keep_ratio` 是必要的。  
它应该继续保留，因为它承担的是：

- 从 repair A 往 selector-first 演进
- 把 active set 从 dense_keep 推到保守 far-only keep

但它不是完整的 BRPO SPGM，只能算其中一个子头。

因此建议：

- `policy.py` 继续负责 **update selection / update weighting**
- `manager.py` 新增负责 **state attenuation / cluster-wise management**

这样理论上更清楚，工程上也不容易继续把 SPGM误解成“只是 selector”。

### 3.5 这个方向的分阶段落地顺序

#### Phase B1：保持 deterministic，但引入 state manager 空壳
目标：先把“SPGM = selector/weighting + state management”这件事在代码结构上明确下来。

- 不引入 stochastic
- 不引入 xyz+opacity
- 只新增 `manager.py`
- 先让它产出 history，不直接改参数

#### Phase B2：让 state manager 控制 per-cluster lr / opacity decay
目标：从“只改 grad”推进到“开始改 Gaussian state behavior”。

第一版建议：
- near / mid / far 三个 cluster 用不同 effective lr scale
- far & low-score cluster 做 very mild opacity decay
- 不直接 prune，不直接 stochastic

#### Phase B3：再考虑 stochastic
只有当：
- ranking 站稳
- selector-first 形成稳定可行区
- state manager 的 deterministic 版本确实带来正收益

才考虑把 BRPO 的 stochastic attenuation 搬进来。  
否则 stochastic 只会放大当前 ranking 不稳的问题。

---

## 4. 两个方向如何一起接到你的 pipeline

### 4.1 推荐的整体顺序

**先做方向 A，再做方向 B。**

原因不是工程方便，而是理论依赖关系：

- 如果 observation 还不是 unified RGB-D，那么 SPGM 即使做成 scene-level manager，也是在管理一个仍然“RGB-first / weak-depth”的系统。
- 只有当 pseudo branch 的观测本身更接近 BRPO 的 unified confidence-weighted RGB-D，SPGM 才有条件真正管理“几何+外观共同可信”的 Gaussian 更新。

所以更合理的顺序是：

```text
A1. joint confidence / joint depth target
→ A2. geometry-constrained support/depth expand
→ B1. add scene-state manager shell
→ B2. deterministic state management
→ B3. optional stochastic
```

### 4.2 你当前 pipeline 中哪些模块替换、哪些模块新增

#### 保留不动
- `pseudo_fusion.py`
- `build_pseudo_cache.py`
- `local_gating/signal_gate.py`
- `gaussian_param_groups.py`
- 当前 `repair A / selector-first` 的基本实验 protocol

#### 新增
- `pseudo_branch/brpo_v2_signal/joint_confidence.py`
- `pseudo_branch/brpo_v2_signal/support_expand.py`
- `pseudo_branch/spgm/manager.py`

#### 修改
- `scripts/build_brpo_v2_signal_from_internal_cache.py`
- `pseudo_branch/pseudo_loss_v2.py`
- `scripts/run_pseudo_refinement_v2.py`
- `pseudo_branch/spgm/stats.py`
- `pseudo_branch/spgm/score.py`
- `pseudo_branch/spgm/policy.py`

---

## 5. 给 agent 的直接任务拆分

### Task A1：实现 unified RGB-D joint confidence
- 新增 `joint_confidence.py`
- 在 `build_brpo_v2_signal_from_internal_cache.py` 中写出 `joint_confidence_v2.npy` 与 `joint_depth_target_v2.npy`
- 在 `pseudo_loss_v2.py` 中支持 `joint_confidence_v2` 作为 RGB / depth 共同 mask

### Task A2：实现 geometry-constrained support/depth expand
- 新增 `support_expand.py`
- 仅基于 projected depth / overlap / fusion weight 做几何一致性扩张
- 不 densify raw RGB

### Task B1：SPGM 结构升级
- 新增 `spgm/manager.py`
- 在 `run_pseudo_refinement_v2.py` 中把 SPGM 分成 `update policy` 与 `state management` 两层

### Task B2：scene-aware density proxy
- 在 `stats.py / score.py` 中引入更结构化的 density proxy
- 明确拆分 `ranking_score` 与 `state_score`

### Task B3：deterministic state management
- 先做 per-cluster lr scaling
- 再做 mild opacity attenuation
- 先不做 stochastic

---

## 6. 最后压缩成一句判断

### 方向 A 的本质
你当前和 BRPO 的真正差异，不是“depth coverage 低”这么简单，而是：

> **你现在在解一个 RGB-first、depth-sidecar 的 refine 问题；BRPO 在解一个 unified confidence-weighted RGB-D joint refinement 问题。**

要靠拢 BRPO，必须先把 observation space 统一起来。

### 方向 B 的本质
你当前和 BRPO 的真正差异，不是“有没有 SPGM 文件夹”这么简单，而是：

> **你现在的 SPGM 仍主要是 pseudo-branch grad modulator；BRPO 的 SPGM 是 scene-level Gaussian population manager。**

要靠拢 BRPO，必须让 SPGM 不只决定“这一轮梯度怎么乘”，还要决定“这张图里哪些 Gaussian 更该保留、哪些更该弱化、哪些 cluster 的更新更该稀疏”。

---

## 7. 参考来源（给 agent 用）

### Repo 当前状态与设计
- `part3_BRPO/docs/STATUS.md`
- `part3_BRPO/docs/DESIGN.md`
- `part3_BRPO/docs/archived/2026-04-experiments/P2I_stageB_window_localization_20260416.md`
- `part3_BRPO/docs/archived/2026-04-experiments/P2L_spgm_canonical_stageB_compare_20260417.md`
- `part3_BRPO/docs/archived/2026-04-experiments/P2M_spgm_conservative_repair_compare_20260417.md`
- `part3_BRPO/docs/archived/2026-04-experiments/P2R_spgm_score_ranking_repair_compare_20260417.md`
- `part3_BRPO/docs/archived/2026-04-experiments/P2S_supportblend_farkeep_followup_compare_20260417.md`

### 当前核心代码入口
- `part3_BRPO/scripts/build_brpo_v2_signal_from_internal_cache.py`
- `part3_BRPO/scripts/run_pseudo_refinement_v2.py`
- `part3_BRPO/pseudo_branch/pseudo_loss_v2.py`
- `part3_BRPO/pseudo_branch/spgm/stats.py`
- `part3_BRPO/pseudo_branch/spgm/score.py`
- `part3_BRPO/pseudo_branch/spgm/policy.py`

### BRPO 理论依据
- BRPO paper:
  - confidence mask inference: Eq. (9)
  - SPGM depth partition: Eq. (10)–(11)
  - density entropy: Eq. (12)–(13)
  - unified importance: Eq. (14)
  - cluster-aware attenuation: Eq. (15)–(16)
  - joint RGB-D refinement: Eq. (18)–(20)

### 可借鉴的外部方向
- RegGS: 结构一致性优先于纯像素 supervision 的 registration 思路
- DIFIX3D+: progressive 3D update / 渐进式 pseudo 注入思路

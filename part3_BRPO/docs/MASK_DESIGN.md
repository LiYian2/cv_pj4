# MASK_DESIGN.md - A1 / mask 设计细化文档

> 更新时间：2026-04-20 02:56 (Asia/Shanghai)

> **书写规范**：
> 1. 这份文档只讲 A1 / mask：信息从哪里来、怎么变成信号、怎么被下游消费
> 2. 优先讲机制与代码流，不重复过程性时间线（过程去 CHANGELOG）
> 3. 需要比较时，统一比较 BRPO / old A1 / new A1 / current brpo_style_v1
> 4. 允许写必要的实验结论，但重点始终是“为什么”
> 5. 更新后修改文档顶部时间戳

---

## 1. 先说结论

A1 这部分真正的核心，不是“mask 长什么样”，而是三件事：**信息从哪里来，target 怎么生成，confidence/mask 怎么决定下游该信什么。**

BRPO、old A1、current new A1、current `brpo_style_v1` 的根本区别，正是在这三件事上的信息流完全不同。

- **BRPO A1**：先有 pseudo-frame，再由真实 reference 对 pseudo-frame 做事后验证，得到统一 `C_m`，再让 RGB/depth 共用这个 `C_m`。
- **old A1**：RGB confidence 和 depth source 先分别构造，再在 downstream 求共同 trusted support；target 基本不重建。
- **current new A1**：先做 4-candidate competition / fusion，再从同一套 competition 里同时派生 target 和 confidence。
- **current `brpo_style_v1`**：已经把 shared `C_m` 和 decoupled target/confidence 的方向改对了，但 verifier backend 和 target builder 仍是第一版工程近似，所以还没完全超过 old A1。

---

## 2. BRPO 的 A1：信息从哪里来，怎么变成信号，怎么被消费

### 2.1 信息从哪里来

按 `part3_BRPO_A1_B3_vs_BRPO_detailed_analysis.md` 的分析，BRPO 的 A1 不是“先给 depth candidate 打分，再顺手产一个 confidence”。它先有一个 pseudo-frame，文档里记作 `I_t^{fix}`。然后围绕这个 pseudo-frame 去看它和左右真实 reference 的 correspondence 关系。

分析文档明确把 verifier 统计对象写成两个集合：

- `M_k`：pseudo-frame 与前一侧真实 reference 的 mutual nearest-neighbor correspondence set
- `M_{k+1}`：pseudo-frame 与后一侧真实 reference 的 mutual nearest-neighbor correspondence set

所以 BRPO A1 最早进入系统的信息，不是 candidate score，而是：**这个 pseudo-frame 像素能不能在左右 reference 里找到双向一致的真实对应。**

### 2.2 这些信息怎么变成后续信号

BRPO 直接用集合关系定义统一 confidence mask `C_m`：

- `M_k ∩ M_{k+1}` → `1.0`
- `M_k ⊕ M_{k+1}` → `0.5`
- `outside union` → `0.0`

也就是说，BRPO 的 confidence 不是“哪个解释更像对的”，而是“**这个 pseudo-frame 像素有没有被真实多视角验证**”。

这里最关键的结构性特点是：

1. pseudo content 先生成；
2. confidence 再由 reference 事后验证生成；
3. target / content 与 confidence 的来源是分开的。

### 2.3 这些信号怎么被下游消费

按分析文档，BRPO 最终不是给 RGB 一张 mask、给 depth 再单独一张 mask，而是让同一个 `C_m` 去同时作用 RGB 和 depth supervision。也就是：**统一 verifier signal 下沉到 joint optimization。**

所以 BRPO A1 的本体可以概括成一句话：

> **先生成 pseudo-frame，再由真实 reference 验证 pseudo-frame 像素，最后把同一个 verification mask 下沉到 RGB/depth。**

---

## 3. old A1：信息从哪里来，怎么变成信号，怎么被消费

当前 old A1 主要落在：
- `pseudo_branch/brpo_v2_signal/rgb_mask_inference.py`
- `pseudo_branch/brpo_v2_signal/depth_supervision_v2.py`
- `pseudo_branch/brpo_v2_signal/joint_confidence.py`
- `scripts/run_pseudo_refinement_v2.py`

### 3.1 信息从哪里来

old A1 的上游信息是两条分开的链：

第一条是 RGB 侧。`rgb_mask_inference.py` 先拿 `target_rgb_fused.png` 去和左右 reference RGB 做 matcher correspondence，得到：

- `support_left`
- `support_right`
- `support_both`
- `support_single`
- `raw_rgb_confidence_v2`
- `raw_rgb_confidence_cont_v2`

第二条是 depth 侧。`depth_supervision_v2.py` 会基于：

- `projected_depth_left/right`
- `fusion_weight_left/right`
- `overlap_mask_left/right`
- `render_depth`

构造：

- `target_depth_for_refine_v2_brpo`
- `depth_supervision_mask_v2_brpo`
- `target_depth_source_map_v2_brpo`

所以 old A1 最开始进入系统的信息，其实是：**一条 RGB support/confidence 链 + 一条 depth supervision/verified source 链。**

### 3.2 这些信息怎么变成后续信号

old A1 在 `joint_confidence.py` 里做的事情非常朴素：

- 从 `target_depth_source_map_v2_brpo` 压出一个 `geometry_tier`
  - single verified → `0.5`
  - both verified → `1.0`
  - 其他 → `0`
- 然后直接做：
  - `joint_confidence = min(raw_rgb_confidence, geometry_tier)`
  - `joint_confidence_cont = raw_rgb_confidence_cont * geometry_tier`
- 同时 `joint_depth_target_v2` 只是把 `target_depth_for_refine_v2_brpo` copy 出来继续用

所以 old A1 不是真正的 joint builder。它的信号形成逻辑更像：

> **RGB 侧先说“我有多相信这个像素”，depth 侧先说“这个像素在几何上属于 single 还是 both 支持”，最后 downstream 取两者共同的 trusted support。**

这里最重要的一点是：**old A1 不重写 target 本体。** 它只是把已有 target 套上一层更保守的 joint support filter。

### 3.3 这些信号怎么被下游消费

在 `run_pseudo_refinement_v2.py` 里，old A1 对应的消费方式是：

- `pseudo_observation_mode=off`
- `stageA_rgb_mask_mode=joint_confidence_v2`
- `stageA_depth_mask_mode=joint_confidence_v2`
- `stageA_target_depth_mode=joint_depth_v2`

也就是说，consumer 还是旧 consumer，只是 RGB/depth mask 和 target_depth_mode 指向了 old A1 这条 joint-support 语义。

所以 old A1 的强项在于：**它不激进，但稳定。** 它没有把 observation object 整体重写，只是把监督域压到一个更可信的共同支持域里。

---

## 4. current new A1：信息从哪里来，怎么变成信号，怎么被消费

current new A1 主要落在：
- `pseudo_branch/brpo_v2_signal/joint_observation.py`
- `scripts/build_brpo_v2_signal_from_internal_cache.py`
- `scripts/run_pseudo_refinement_v2.py`

### 4.1 信息从哪里来

current new A1 不再沿用 old A1 那种“RGB 一条链、depth 一条链，最后取 min”的逻辑，而是先收集四类 depth/source candidate：

- `left_depth`
- `right_depth`
- `both_depth`
- `render_depth`

同时它还有四类 evidence：

- appearance
- geometry
- support
- prior

这些信息在 `joint_observation.py` 中被压成四路 `score_map`，也就是每个像素上四个 candidate 的解释分数。

### 4.2 这些信息怎么变成后续信号

current new A1 的动作链条在代码里很清楚：

1. `score_stack` 上取 `best_idx / best_score`
2. 对 `score_stack` 做 softmax-like 归一化，得到 `score_prob`
3. 用 `score_prob` 对 `depth_stack` 做加权融合，得到 `fused_depth`
4. 从同一条链里再得到：
   - `confidence_depth = best_score`
   - `confidence_rgb = selected appearance`
   - `confidence_joint = sqrt(conf_rgb * conf_depth)`
5. 再由 `confidence_joint > min_joint_confidence` 得到 `valid_mask`

然后打包成：

- `pseudo_depth_target_joint_v1`
- `pseudo_confidence_joint_v1`
- `pseudo_confidence_rgb_joint_v1`
- `pseudo_confidence_depth_joint_v1`
- `pseudo_uncertainty_joint_v1`
- `pseudo_source_map_joint_v1`
- `pseudo_valid_mask_joint_v1`

所以 current new A1 的真正语义是：

> **先做 candidate competition / fusion，再从同一套 competition 中同时派生 target 和 confidence。**

### 4.3 这些信号怎么被下游消费

在 `run_pseudo_refinement_v2.py` 里，`pseudo_observation_mode=brpo_joint_v1` 时，consumer 会直接锁到这一整包 observation bundle：

- `rgb_conf = pseudo_confidence_joint_v1`
- `depth_conf = pseudo_confidence_joint_v1`
- `depth_target = pseudo_depth_target_joint_v1`
- `source_map = pseudo_source_map_joint_v1`

这就是为什么我一直说它不是 old A1 换壳，而是真正的 observation object rewrite。

### 4.4 它和 BRPO 的关键差别

current new A1 虽然已经是 upstream builder，但它和 BRPO 的根差还在：

> **它的 target 和 confidence 仍然同源。**

`score_stack` 同时生成 `fused_depth` 和 `confidence_joint`。如果 candidate ranking 本身错了，就会出现“错的 target + 偏高的 confidence”一起出现的情况，也就是 analysis 文档里说的那种 **smooth but self-consistent error**。

这就是为什么它虽然比 control 强，但仍然不稳定地输给 old A1。

---

## 5. verify proxy v1：信息从哪里来，怎么变成信号，为什么失败

这条线现在已经不是主研究线了，但它解释了很多边界。

### 5.1 信息从哪里来

`pseudo_observation_verifier.py` 的输入不是独立 pseudo-frame，而是：

- `pseudo_depth_target_joint_v1`
- `projected_depth_left/right`
- `overlap_mask_left/right`
- `render_depth`（只做诊断）

也就是说，它验证的是 **current new A1 已经生成好的 target**。

### 5.2 这些信息怎么变成信号

它先做：

- `verify_left`
- `verify_right`
- `verify_both`
- `verify_xor`
- `verify_union`

然后把 confidence 写成：

- both → `1.0`
- xor → `0.5`
- none → `0.0`

### 5.3 为什么它失败

它失败的根因不是“hard mask 不好”，而是：

1. **只改 confidence，不改 target**。如果 target builder 本身就有问题，后挂 verifier 无法修正 target；
2. **独立性还不够强**。target 是从 `projected_depth_left/right` 融出来的，verifier 又拿 `projected_depth_left/right` 回来审这个 target，本质上还是“同一家信息源回头自证”；
3. **信号更硬但收益不够强**。原本 continuous 的 `confidence_joint` 被换成 `{1, 0.5, 0}`，但 target 质量没有同步提升，结果就是 supervision 变少、错误 target 还在。

这就是为什么它会显著差于 old A1 和 current new A1。

---

## 6. current brpo_style_v1：信息从哪里来，怎么变成信号，怎么被消费

这是当前真正该看的 A1 版本。主要代码路径是：
- `pseudo_branch/brpo_v2_signal/rgb_mask_inference.py`
- `pseudo_branch/brpo_v2_signal/pseudo_observation_brpo_style.py`
- `scripts/build_brpo_v2_signal_from_internal_cache.py`
- `scripts/run_pseudo_refinement_v2.py`

### 6.1 信息从哪里来

当前 `brpo_style_v1` 的信息源有两类。

第一类是 pseudo-frame side：
- `fusion/samples/<frame_id>/target_rgb_fused.png`

它在工程上充当 pseudo-frame 代理。

第二类是 verifier / depth side：
- `rgb_support_left_v2 / rgb_support_right_v2`
- `projected_depth_left / projected_depth_right`
- `fusion_weight_left / fusion_weight_right`
- `overlap_mask_left / overlap_mask_right`

所以它不再从 4-candidate score 开始，而是从：

> **fused pseudo-frame 的左右 reference support + 两侧 projected depth**

开始。

### 6.2 这些信息怎么变成后续信号

`pseudo_observation_brpo_style.py` 里的逻辑是：

先定义：
- `valid_left = support_left & overlap_mask_left & (projected_depth_left > 0)`
- `valid_right = support_right & overlap_mask_right & (projected_depth_right > 0)`

再定义：
- `verify_both = valid_left & valid_right`
- `verify_left_only = valid_left & ~valid_right`
- `verify_right_only = valid_right & ~valid_left`
- `verify_xor = left_only | right_only`

然后统一 confidence：
- `verify_both -> 1.0`
- `verify_xor -> 0.5`
- 其余 -> `0.0`

这就是当前版的 shared `C_m`。

depth target 则改成：
- both verified：按 `fusion_weight_left/right` 对 `projected_depth_left/right` 做 composition
- left only：直接取 left projected depth
- right only：直接取 right projected depth
- neither：不监督

所以 current `brpo_style_v1` 已经不再是“candidate score -> target + confidence”，而是：

> **support sets -> shared `C_m`; verified projected depth -> target**

这一步的结构已经比 current new A1 更接近 BRPO。

### 6.3 这些信号怎么被下游消费

在 `run_pseudo_refinement_v2.py` 中，`pseudo_observation_mode=brpo_style_v1` 时，consumer 读的是：

- `pseudo_confidence_brpo_style_v1.npy`
- `pseudo_depth_target_brpo_style_v1.npy`
- `pseudo_source_map_brpo_style_v1.npy`
- `pseudo_valid_mask_brpo_style_v1.npy`

也就是说，RGB / depth 确实共用同一个 `C_m`。

---

## 7. BRPO / old A1 / new A1 / current brpo_style_v1 的真正差别

如果只看“mask 长什么样”，很容易把四者讲混。真正该比的是信息流。

### 7.1 BRPO

- **信息源**：pseudo-frame 与 reference 的 direct verification
- **target**：pseudo-frame / verified target builder 语义
- **confidence**：事后 verifier `C_m`
- **消费方式**：RGB / depth 共用同一个 `C_m`

### 7.2 old A1

- **信息源**：RGB confidence + depth source map
- **target**：基本复用既有 `target_depth_for_refine_v2_brpo`
- **confidence**：`min(raw_rgb_confidence, geometry_tier)`
- **消费方式**：旧 consumer 读取更保守的 joint support filter

### 7.3 current new A1

- **信息源**：4-candidate depth/source + appearance/geometry/support/prior 四类 evidence
- **target**：`score_stack -> score_prob -> fused_depth`
- **confidence**：同一个 `score_stack` 再派生 `confidence_joint`
- **消费方式**：整包 `pseudo_*_joint_v1` 被 consumer 锁定

### 7.4 current brpo_style_v1

- **信息源**：fused pseudo-frame 对左右 reference 的 support sets + 两侧 projected depth
- **target**：verified projected depth composition
- **confidence**：shared `C_m`（both=1.0, xor=0.5, none=0.0）
- **消费方式**：整包 `pseudo_*_brpo_style_v1` 被 consumer 锁定，RGB / depth 共用同一个 `C_m`

### 7.5 当前 brpo_style_v1 和真正 BRPO 还有什么不同

当前仍有三层差异：

第一，**pseudo-frame 还不是完整 BRPO 版 `I_t^{fix}`**。我们现在用的是 `target_rgb_fused.png` 作为工程代理。  
第二，**`M_left / M_right` 还不是完整 direct verifier backend**。当前更多是 support map 语义，而不是完整 BRPO-style mutual NN / geometric verification backend。  
第三，**target builder 还只有第一版**。现在只是 verified projected depth composition，还不是更完整的 BRPO verified target builder。

所以当前可以说“语义方向已经对了”，但还不能说“已经完全等价于 BRPO A1”。

---

## 8. 为什么 current brpo_style_v1 仍略差于 old A1

这一点要结合代码流看，不能只看结果数字。

### 8.1 先说结果事实

固定 `new T1 + summary_only` compare 下：

- `oldA1_newT1_summary_only = 24.187772 / 0.875594 / 0.080362`
- `newA1_newT1_summary_only = 24.149355 / 0.875578 / 0.079962`
- `brpoStyleA1_newT1_summary_only = 24.175377 / 0.875338 / 0.080344`

所以 current `brpo_style_v1` 的位置很清楚：

- 明显优于 current new A1 的 PSNR
- 已经逼近 old A1
- 但还没稳定超过 old A1

### 8.2 为什么它比 current new A1 强

因为它把最关键的结构问题改对了：

- current new A1：`score_stack` 同时决定 target 和 confidence
- current brpo_style_v1：`C_m` 与 target builder 已经不再从同一套 candidate score 派生

所以它不再那么容易产生“平滑但自洽的错误”，这就是为什么 PSNR 能明显回升。

### 8.3 为什么它还没超过 old A1

主要有三条代码层原因。

第一，**当前 `C_m` 还不够强。**  
现在 `C_m` 的上游是 `rgb_mask_inference.py` 输出的 `support_left/right`。这条线本质上还是 matcher support occupancy map，不是更完整的 BRPO-style direct verification backend。也就是说，我们已经拿到了“shared `C_m` 的语义壳”，但 verifier 强度本身还不够。

第二，**当前 target builder 还太简单。**  
`brpo_style_v1` 的 target 现在只是：在 support sets 上，把 `projected_depth_left/right` 做 weighted composition。这个逻辑方向对，但工程成熟度还比不上 old A1 背后那条已经更稳的 depth pipeline。old A1 虽然语义不如 BRPO，但它依赖的是一条更成熟的 target_depth 生成链，因此当前仍然更稳。

第三，**当前 `C_m` 更硬、更稀，但 target 稳定性还没一起补齐。**  
old A1 的 `joint_confidence_v2` 虽然不如 BRPO 那么“语义正确”，但它是 `raw_rgb_confidence` 与 `geometry_tier` 的连续/半连续裁剪，工程上更平滑。当前 brpo_style 的 `C_m` 已经变成更硬的 `{1, 0.5, 0}`，如果 verifier 和 target builder 还不够强，这种更硬的监督未必立刻占优。

所以当前差于 old A1，不是因为“shared `C_m` 这个方向错了”，而是因为：

> **shared `C_m` 的语义已经对了，但 verifier backend 和 target builder 还没有把 old A1 的工程稳定性接住。**

### 8.4 这意味着下一步该做什么

这次结果其实把后续方向压缩得很清楚了。

不该做的是：
- 回头继续打磨 `verify proxy v1`
- 回到 current new A1 的 4-candidate score 线上继续调权重
- 把 current `brpo_style_v1` 误判成“已经够了”或者“已经错了”

该做的是：
- 继续加强 `M_left / M_right` 的 direct verifier backend
- 继续加强 BRPO-style target builder
- 保持固定 `new T1 + summary_only` compare，不把 A1 / topology / B3 混着改

---

## 9. 一句话总结

> **old A1 胜在工程稳定，current new A1 败在 target/confidence 同源，current `brpo_style_v1` 已经把语义方向改对并明显优于 current new A1，但 verifier backend 与 target builder 还不够强，所以目前仍略低于 old A1。**

# A1 工程落地方案：BRPO-style Joint Observation Rewrite

## 1. 文档地位
本文件**覆盖**旧版 `A1_unified_rgbd_joint_confidence_engineering_plan.md`。旧版 A1 的方法地位降级为：

> **joint support filter / unified consumer semantics**

也就是：它只是在现有 `RGB mask + depth target` 之上定义了一个共同 trusted support，让 RGB / depth 在 loss 里共享同一消费域；它**不是** BRPO 风格的 joint observation construction。

新版 A1 的唯一目标是：

> **把 pseudo branch 的输入，从“分头构建后再求共同过滤域”改写成“先构建一个统一的 pseudo observation object，再在其上做 confidence-weighted optimization”。**

一句话目标：**不再继续修补 `joint_confidence_v2` 这条 support-filter 路线，而是直接把 pseudo observation builder 改成更接近 BRPO 方法的 joint builder。**

---

## 2. 先说清楚：这次要停掉什么
这次 A1 重写不是继续在原逻辑上缝补，而是明确停止以下做法：

1. **停止把 RGB 和 depth 当作两条先独立成立、最后再求交的分支。**
   当前 `raw_rgb_confidence_v2 -> depth_supervision_v2 -> joint_confidence_v2` 的结构，本质上仍是 RGB-first / depth-sidecar / downstream joint filter。

2. **停止把 `joint_confidence = min(raw_rgb_confidence, geometry_tier)` 当作“joint observation”本身。**
   这个形式只能表达“共同可信域”，不能表达“共同构造出的观测对象”。

3. **停止把 `joint_depth_target_v2` 写成 `target_depth_for_refine_v2_brpo` 的直接复用。**
   新版 A1 必须重新定义 `joint depth target` 的构造逻辑，而不是给旧 target 换一个 joint 名字。

4. **停止在新 joint 模式下继续暴露独立的 `rgb_mask_mode / depth_mask_mode / target_depth_mode` 自由拼装。**
   只要 consumer 还能把三者随意组合，系统语义就仍然是 decoupled，而不是 joint observation。

5. **停止把“support 统一”误当成“observation 已经统一”。**
   旧 A1 的工作成果保留，但它只作为历史对照臂，不再作为主线目标定义。

---

## 3. 这次真正要对齐 BRPO 的是什么
这次不是追求“coverage 看起来更大”，而是追求**方法对象一致**。

更接近 BRPO 的核心不是“有一张 mask”，而是：

> **先得到一份统一的 pseudo observation，再由 confidence / uncertainty 决定其中哪些像素更强、哪些更弱、哪些应被过滤。**

也就是说，系统里真正应该存在的对象不是：
- 一张 `raw_rgb_confidence`
- 一张 `depth_supervision_mask`
- 一张 `target_depth`
- 最后再求一张 `joint_confidence`

而应该是每个像素对应一个统一 observation tuple：

\[
O(p) = \{ I_t(p),\ D_t(p),\ C_{joint}(p),\ C_{rgb}(p),\ C_{depth}(p),\ U(p),\ S(p) \}
\]

其中：
- `I_t(p)`：pseudo RGB target
- `D_t(p)`：pseudo depth target
- `C_joint(p)`：联合 confidence
- `C_rgb(p)`：appearance confidence
- `C_depth(p)`：geometry confidence
- `U(p)`：不确定性或弱观测程度
- `S(p)`：来源标签（both / single / inferred / fallback / render prior / dense-complete 等）

这个对象的意义是：
- **先构造 observation，再做 loss**
- confidence 是 observation 的属性，不只是一个下游裁剪器
- depth target 必须是 joint semantics 下重新生成的结果，而不是旧 target 的 alias

---

## 4. 对当前代码事实的重新判断
当前 live code 的问题不在于 wiring 没通，而在于 builder 的职责定义就不对。

### 4.1 当前 RGB builder 的方法角色
`pseudo_branch/brpo_v2_signal/rgb_mask_inference.py` 当前做的是：
- 从 left/right matcher 结果回填 fused image 上的 sparse support
- 生成 `support_left/right/both/single`
- 生成 `raw_rgb_confidence_v2` 与 continuous 版 `raw_rgb_confidence_cont_v2`

这一步的本质是：

> **从 correspondence 恢复 support map / confidence map**

它还不是 “joint pseudo frame builder”。

### 4.2 当前 depth builder 的方法角色
`pseudo_branch/brpo_v2_signal/depth_supervision_v2.py` 当前做的是：
- 先定义 `rgb_active = raw_rgb_confidence >= threshold`
- 再在 `rgb_active` 区域里检查 left/right projected depth 的有效性
- 再生成 `target_depth_for_refine_v2_brpo` 与 `source_map`

这一步的本质是：

> **用 RGB support 激活一个 depth supervision target**

它不是“joint depth observation builder”，而是“RGB-gated depth supervision builder”。

### 4.3 当前 joint builder 的方法角色
`pseudo_branch/brpo_v2_signal/joint_confidence.py` 当前做的是：
- 用 `geometry_tier` 去裁 `raw_rgb_confidence`
- 得到 `joint_confidence_v2`
- 然后把 `target_depth_for_refine_v2_brpo` 直接 copy 成 `joint_depth_target_v2`

这一步本质是：

> **对已有 RGB / depth 分支定义一个共同消费域**

而不是生成新的 joint observation。

所以当前整个 A1 旧版链路的问题不是“做得不够强”，而是：

> **它只统一了 downstream consumer semantics，没有统一 upstream observation construction semantics。**

---

## 5. 新版 A1 的设计原则（必须严格遵守）

### 5.1 先构造 observation，再构造 mask
在新版 A1 中，mask / confidence 不再是先验裁剪器，而是 observation object 的组成部分。

### 5.2 depth target 不能再由 `rgb_active` 硬触发
一个像素是否有 depth observation，不能再由 `raw_rgb_confidence >= threshold` 单独决定。

更合理的顺序应是：
1. 先收集该像素的 geometry candidates
2. 再结合 appearance / geometry / support 一起打 joint evidence
3. 再选择或融合出该像素的最终 depth target 与 confidence

### 5.3 joint depth target 必须重新生成，不能复用旧 target
`joint_depth_target_v3` 必须是新版 joint builder 的输出，而不是：
- `target_depth_for_refine_v2_brpo` 的直接 copy
- 或 `target_depth_for_refine_v2_brpo` 只换一个 joint 文件名

### 5.4 新 joint 模式必须是“不可拆分消费”的
在新模式下，consumer 不允许再自由组合：
- 一个 RGB mask
- 一个 depth mask
- 一个 depth target

新版 joint observation mode 必须是一整包消费：
- RGB target
- depth target
- joint confidence
- source map / uncertainty

### 5.5 优先接近 BRPO 的方法语义，不优先照顾旧接口习惯
这次重写允许引入新 artifact 命名、新 mode、新 builder 模块；不要求继续伪装成旧 A1 的小修小补。

---

## 6. 新版 joint observation 的方法定义

### 6.1 observation 的组成
新版 A1 的 builder 产物应围绕一个统一对象展开：

```text
joint_observation_v3/frame_xxxx/
├── pseudo_depth_target_joint_v1.npy
├── pseudo_confidence_joint_v1.npy
├── pseudo_confidence_rgb_joint_v1.npy
├── pseudo_confidence_depth_joint_v1.npy
├── pseudo_uncertainty_joint_v1.npy
├── pseudo_source_map_joint_v1.npy
├── pseudo_valid_mask_joint_v1.npy
└── joint_observation_meta_v1.json
```

说明：
- `pseudo_rgb_target` 不额外新造一份图像，继续复用现有 `samples/<frame>/target_rgb_fused.png`
- 但在 `joint_observation_meta_v1.json` 中必须把它写成 observation object 的组成部分，而不只是 consumer 的默认输入

### 6.2 核心思路：从 candidate competition / fusion 生成 depth observation
对于每个像素，不再先问“它是不是 RGB active”，而是先收集 depth candidates：
- `d_left`: 左投影深度（若有效）
- `d_right`: 右投影深度（若有效）
- `d_both`: 左右一致时的融合深度
- `d_render`: 当前 render depth prior
- `d_dense`（可选预留）: 来自更稠密 correspondence / MASt3R / future completion 的候选

对每个 candidate 计算 joint evidence：

\[
score_k(p) = w_a A(p) + w_g G_k(p) + w_s S_k(p) + w_p P_k(p)
\]

其中：
- `A(p)`: appearance evidence（来自 matcher confidence / fused RGB support / reference consistency）
- `G_k(p)`: 几何一致性（左右深度一致性、reprojection residual、局部平滑/邻域合理性）
- `S_k(p)`: support strength（both-side > single-side > inferred > fallback）
- `P_k(p)`: source prior（更偏方法先验）

然后：
1. 若存在高分 candidate，选取最高分或做分数加权融合，生成 `pseudo_depth_target_joint_v1(p)`
2. 把对应得分转成 `pseudo_confidence_depth_joint_v1(p)`
3. 把 appearance evidence 转成 `pseudo_confidence_rgb_joint_v1(p)`
4. 再由两者融合得到 `pseudo_confidence_joint_v1(p)`
5. 同时写出 `source_map` 和 `uncertainty`

### 6.3 joint confidence 的定义
新版 joint confidence 不能再是简单的：
- `min(raw_rgb_confidence, geometry_tier)`
- 或 `raw_rgb_confidence * geometry_tier`

推荐第一版用下面这种更稳定的形式：

\[
C_{joint}(p) = \sqrt{C_{rgb}(p) \cdot C_{depth}(p)}
\]

或者等价的保守融合：

\[
C_{joint}(p) = \frac{2 C_{rgb}(p) C_{depth}(p)}{C_{rgb}(p) + C_{depth}(p) + \epsilon}
\]

要求：
- `C_rgb` 和 `C_depth` 必须都是**joint builder 内部重新定义**出来的连续置信度
- 不允许直接把旧 discrete tier 当 joint confidence 最终值

### 6.4 uncertainty 的角色
`pseudo_uncertainty_joint_v1` 不是装饰品。它用于区分：
- strong observation
- weak-but-usable observation
- fallback-like observation

第一版可以简单定义为：

\[
U(p) = 1 - C_{joint}(p)
\]

但 meta 中必须同时保留更细的来源信息，例如：
- `both_weighted`
- `single_left`
- `single_right`
- `render_prior`
- `dense_completed`
- `invalid`

这样 consumer 才能在后续精细地区分不同 observation 质量，而不是只靠一个 hard gate。

---

## 7. builder 层怎么改：明确的新模块与职责

### 7.1 新增文件
建议新增：

1. `pseudo_branch/brpo_v2_signal/joint_observation.py`
2. 必要时新增辅助函数到：
   - `pseudo_branch/brpo_v2_signal/__init__.py`

第一版不建议一开始就拆太多文件，先把 builder 真正写对。

### 7.2 新模块应包含的函数
`joint_observation.py` 至少应包含：

1. `collect_joint_observation_candidates(...)`
   - 汇总每个像素的 depth candidates / source priors / validity

2. `score_joint_observation_candidates(...)`
   - 计算 `A / G / S / P`
   - 输出 per-candidate score 与 per-pixel best candidate index

3. `build_joint_observation_from_candidates(...)`
   - 生成：
     - `pseudo_depth_target_joint_v1`
     - `pseudo_confidence_rgb_joint_v1`
     - `pseudo_confidence_depth_joint_v1`
     - `pseudo_confidence_joint_v1`
     - `pseudo_uncertainty_joint_v1`
     - `pseudo_source_map_joint_v1`
     - `pseudo_valid_mask_joint_v1`

4. `write_joint_observation_outputs(...)`
   - 写文件与 meta

### 7.3 现有 builder 的处理原则
现有：
- `rgb_mask_inference.py`
- `depth_supervision_v2.py`

不删除，但在新版 A1 中角色变成：
- `rgb_mask_inference.py`：提供 appearance evidence / support prior 的输入资产
- `depth_supervision_v2.py`：提供 geometry candidates 的一部分输入资产或对照基线

它们不再被当成“最终 pseudo observation 的直接产物”。

---

## 8. consumer 层怎么改：禁止继续拆着吃

### 8.1 新 mode 设计
在 `scripts/run_pseudo_refinement_v2.py` 中新增统一 mode：

```text
--pseudo_observation_mode brpo_joint_v1
```

当这个 mode 打开时：
- 不再允许用户独立指定 `stageA_rgb_mask_mode / stageA_depth_mask_mode / stageA_target_depth_mode`
- 或者即便保留 CLI，也必须在日志中明确这些参数被 joint mode 覆盖

### 8.2 新 mode 的消费规则
在 `brpo_joint_v1` 下：
- RGB target：`sample_dir/target_rgb_fused.png`
- depth target：`signal_v2/frame_xxxx/pseudo_depth_target_joint_v1.npy`
- RGB loss 权重：`pseudo_confidence_joint_v1` 或 `pseudo_confidence_rgb_joint_v1`
- depth loss 权重：`pseudo_confidence_joint_v1` 或 `pseudo_confidence_depth_joint_v1`
- source-aware 辅助项：来自 `pseudo_source_map_joint_v1`
- uncertainty-aware reweight：来自 `pseudo_uncertainty_joint_v1`

### 8.3 `pseudo_loss_v2.py` 的改法
第一版不追求新公式花样，先追求语义正确：
- 新增 joint observation path（可以是单独 helper）
- 它直接消费一整包 observation object，而不是外部分别喂 RGB mask / depth mask / depth target
- `masked_rgb_loss` / `masked_depth_loss` 继续复用可以，但调用侧必须由 joint mode 统一组织

重点不是把 loss 公式写得花，而是：

> **consumer 不能再把“joint”退化回三个旧旋钮。**

---

## 9. 输入 / 输出重新定义

### 9.1 输入
新版 joint builder 的输入资产：
1. `target_rgb_fused.png`
2. `raw_rgb_confidence_v2.npy`
3. `raw_rgb_confidence_cont_v2.npy`
4. `projected_depth_left.npy`
5. `projected_depth_right.npy`
6. `fusion_weight_left.npy`
7. `fusion_weight_right.npy`
8. `overlap_mask_left.npy`
9. `overlap_mask_right.npy`
10. `render_depth.npy`
11. 可选：future dense match / MASt3R-derived dense cues

### 9.2 输出
新版 observation artifacts：
1. `pseudo_depth_target_joint_v1.npy`
2. `pseudo_confidence_joint_v1.npy`
3. `pseudo_confidence_rgb_joint_v1.npy`
4. `pseudo_confidence_depth_joint_v1.npy`
5. `pseudo_uncertainty_joint_v1.npy`
6. `pseudo_source_map_joint_v1.npy`
7. `pseudo_valid_mask_joint_v1.npy`
8. `joint_observation_meta_v1.json`

### 9.3 必须在 meta 中明确写出的事实
meta 必须至少包含：
- 当前使用的 candidate 列表
- score 组成（appearance / geometry / support / prior）
- depth target 是否由旧 `target_depth_for_refine_v2_brpo` 直接复用（新版必须是 `false`）
- joint mode 下 consumer 是否禁止独立 mask/depth 组合（新版必须是 `true`）

---

## 10. 实施步骤（严格按这个顺序）

### Step 1：冻结旧 A1 语义，不再在其上追加 patch
- 旧 `joint_confidence_v2 / joint_depth_target_v2` 保留为历史对照臂
- 不再继续给旧 A1 增加新规则、新阈值、新 compare

### Step 2：写新 builder，不复用旧 `joint_confidence.py`
- 新增 `joint_observation.py`
- 先完成 candidate collection / scoring / writing
- 第一版就必须输出完整 observation bundle，而不是只输出 confidence

### Step 3：修改 `build_brpo_v2_signal_from_internal_cache.py`
- 在现有 builder 之后追加新版 joint observation builder
- 写出新 artifacts
- 但不要覆盖旧 artifacts，避免失去对照臂

### Step 4：修改 `run_pseudo_refinement_v2.py`
- 新增 `--pseudo_observation_mode brpo_joint_v1`
- joint mode 下锁死 consumer 逻辑
- 明确禁止再靠旧 mask/target 三件套自由组合

### Step 5：修改 `pseudo_loss_v2.py`
- 新增 joint observation ingestion path
- 保持现有归一化逻辑，但 joint mode 的输入必须来自 observation bundle

### Step 6：做机制 smoke，不先谈指标
第一轮 smoke 的验收重点不是 PSNR，而是证明方法对象确实换了：
1. `pseudo_depth_target_joint_v1` 不是旧 target 的直接 copy
2. `pseudo_confidence_joint_v1` 不是 `min(raw_rgb_confidence, geometry_tier)` 的重命名
3. joint mode 下 old-style `rgb_mask_mode / depth_mask_mode / target_depth_mode` 不再主导行为
4. meta 能明确解释每类 source 的来源和比例

### Step 7：做 compare
第一轮 compare 必须至少包含三臂：
1. canonical control：`RGB-only v2 + depth-sidecar`
2. old A1：`joint_support_filter_v1`（即当前 `joint_confidence_v2` 路线）
3. new A1：`brpo_joint_v1`

这轮 compare 的目的不是证明“更大 coverage 一定更好”，而是验证：

> **把 observation object 写对之后，是否能比“只统一 support 语义”更稳定。**

---

## 11. 验收标准

### 11.1 方法验收（比结果更重要）
只有满足以下全部条件，才算 A1 真正改成“更靠近 BRPO 方法”：

1. 新版 joint builder 产出的是完整 observation bundle，而不是只有 joint mask
2. `pseudo_depth_target_joint_v1` 由 joint candidate competition / fusion 重新生成
3. joint confidence 由 joint builder 内部定义，不是旧 RGB confidence 与 geometry tier 的后验裁剪
4. joint mode 下 consumer 不允许再分拆为独立 RGB / depth / target 旋钮
5. observation bundle 中显式保留 uncertainty 和 source map

### 11.2 结果验收
在方法验收通过之后，再看：
1. 相比 old A1，StageB-40 至少产生更清晰的训练侧差异
2. replay 不应明显更差
3. 如果结果仍负，优先回看 candidate scoring / source prior，而不是再把它退化回 old A1 的 support-filter 思路

---

## 12. 明确不做的事
1. 不再继续把 `joint_confidence_v2` 当主线 patch 下去
2. 不把新版 A1 继续包装成“只是在旧 signal_v2 上加几个新文件”
3. 不在新版 A1 尚未完成前继续推进 B3 / opacity / stochastic
4. 不在 joint mode 下保留过多兼容分支，导致新语义再次被旧接口稀释
5. 不把“coverage 数字更大”当成方法是否更靠近 BRPO 的唯一判断

---

## 13. 第一轮执行优先级（重启后直接照这个做）
1. **先写新 doc 对应的 builder 和 consumer mode，不再补旧 A1。**
2. **先做方法机制 smoke，再做 compare。**
3. **compare 必须包含 old A1，对照“support 统一”和“observation 重写”到底差在哪。**
4. **在 observation rewrite 没站住前，冻结 B 线。**

---

## 14. 最后压成一句话
旧 A1 做的是：

> **把已有 RGB / depth 分支用一个更严格的共同 trusted support 重新消费。**

新版 A1 要做的是：

> **把 pseudo branch 的输入直接重写成一个 BRPO-style joint observation object，让 RGB、depth、confidence、uncertainty、source 从一开始就是同一个观测实体，而不是几条分支最后再求交。**

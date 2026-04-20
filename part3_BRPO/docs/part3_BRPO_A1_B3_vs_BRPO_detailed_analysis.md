# Part3 BRPO：A1 / B3 与 BRPO 原方法的拆解式对照分析

## 文档目的

这份文档只做一件事：

- **把 BRPO 在 A1（pseudo observation / confidence）和 B3（SPGM / population management）上到底“统计了什么量、这些量如何变成动作”讲清楚**；
- **把我们当前方法在 A1 和 B3 上到底“统计了什么量、这些量如何变成动作”讲清楚**；
- 然后在“统计对象 / 动作对象 / 动作时序 / 控制作用域 / 误差模式”这些维度上，给出 **BRPO vs 我们 current implementation** 的细粒度差异分析。

这份文档不再只说“更像 / 不像 BRPO”这种空话，而是把对象、公式、顺序、动作全部拆开。

---

# Part I. A1：BRPO 的 confidence / joint optimization 到底是什么，我们 current A1 到底是什么

## 1. BRPO 的 A1 先在做什么：不是“先有 joint mask”，而是“先有 pseudo frame，再做验证”

BRPO 在 pseudo frame 这条线上，顺序是：

1. 先有当前 Gaussian render 的图像 `I_t^{gs}`；
2. 用 pseudo-view deblur UNet，结合前后参考帧，得到 refined image `\hat I_t`；
3. 再分别以左/右 reference 为条件，做 diffusion restoration，得到两个 restoration candidate；
4. 用 overlap score fusion，把它们融合成最终的 `I_t^{fix}`；
5. 然后才对 `I_t^{fix}` 做 confidence mask inference，得到 `C_m`；
6. 最后 joint optimization 用这个 `C_m` 同时作用在 RGB loss 和 depth loss 上。

所以 BRPO 不是“先造一张 joint confidence，再配 depth target”，而是：

> **先构造 pseudo-frame（`I_t^{fix}`），再问这个 pseudo-frame 的像素能不能被真实多视角验证。**

---

## 2. BRPO 的 overlap score fusion 在统计什么量

BRPO 对左右两条 restoration 分支的融合，不是随便平均，而是先算 overlap / geometry / pose 的可信度。

### 2.1 它先统计的量

对于两台相机 `C_a, C_b`，paper 先把 `C_a` 的 depth map 投影到 `C_b`：

$$
p_b = \Pi\left(K, T_b T_a^{-1} \Pi^{-1}(p_a, d_a)\right)
$$

然后定义 overlap mask：

$$
O_{ab}(p) = [p_b \in \Omega_b] \land [d_b(p_b) > \epsilon]
$$

再定义 depth consistency score：

$$
s_d(p) = \exp\left(- \frac{|d_a(p) - d_b(p_b)|}{(d_a(p)+d_b(p_b))/2 + \epsilon}\right)
$$

再加入 pose-consistency：

$$
s_t = \exp(-\|t_a - t_b\|_2)
$$

最后得到 combined overlap confidence：

$$
C_{ab}(p) = s_d(p) s_t
$$

### 2.2 这些量怎么变成动作

这些量不是日志，它们直接决定 fusion residual 的权重：

$$
I_t^{fix} = I_t + W_1 \odot r^{(1)} + W_2 \odot r^{(2)},
$$

其中

$$
W_i(p) = \frac{C_{t,i}(p)}{C_{t,1}(p)+C_{t,2}(p)+\epsilon}
$$

也就是说，**BRPO 在这一步统计的量，是“每个分支对当前 pseudo-frame 修复到底有多可信”；动作是“它们对最终 `I_t^{fix}` 的 residual 融合权重”。**

注意：这一步还不是最终 `C_m`，这一步只是为了构造 `I_t^{fix}`。

---

## 3. BRPO 的 confidence mask inference 在统计什么量

这一步是 BRPO A1 的关键。

在 paper 里，`C_m` 不是从 depth candidate score 推出来的，而是从 **fused pseudo-frame 与两侧真实参考帧的 correspondence 关系** 推出来的。

### 3.1 它统计的量是什么

paper 取：

- `M_k`：`I_t^{fix}` 和前一侧真实 reference `I_k^{rf}` 的 mutual nearest-neighbor correspondence set
- `M_{k+1}`：`I_t^{fix}` 和后一侧真实 reference `I_{k+1}^{rf}` 的 mutual nearest-neighbor correspondence set

这两个集合的含义是：

> **这个 pseudo-frame 的像素，是否能在真实参考帧里找到双向一致的几何对应。**

### 3.2 这些量怎么变成动作

paper 直接定义：

$$
C_m(p)=
\begin{cases}
1.0 & p \in M_k \cap M_{k+1} \\
0.5 & p \in M_k \oplus M_{k+1} \\
0.0 & p \notin M_k \cup M_{k+1}
\end{cases}
$$

也就是说，BRPO 的 confidence 在这一步回答的是：

> **“这个 pseudo-frame 像素，事后能不能被两侧真实帧验证？”**

这里的“验证”不是口头词，而是 **集合关系**：

- 两边都支持 → 1.0
- 只一边支持 → 0.5
- 两边都不支持 → 0.0

这就是 BRPO 的 confidence mask 本体。

---

## 4. BRPO 的 A1 最后怎么进入 loss

BRPO 最后不是“RGB 一张 mask、depth 一张 mask、target depth 再单独拼装”。

它在 joint optimization 里直接用同一个 `C_m` 去权重 RGB 和 depth：

$$
L = \beta L_{rgb} + (1-\beta)L_d + \lambda_s L_s
$$

$$
L_{rgb} = \frac{\|C_m \odot (I_t - \hat I_t)\|_1}{\|C_m\|_1}
$$

$$
L_d = \frac{\|C_m \odot (D_t - \hat D_t)\|_1}{\|C_m\|_1}
$$

这里真正重要的不是“有归一化”，而是：

> **RGB 和 depth 使用的是同一个观测测度 `C_m`。**

也就是说，BRPO 里 A1 的 joint semantics 是：

- 先得到一个 pseudo-frame；
- 再从真实 reference 对这个 pseudo-frame 的验证关系中，定义一个统一 confidence；
- 再让 RGB / depth 一起共用这个 confidence。

---

## 5. 我们的 old A1 到底统计了什么量，它怎么变成动作

old A1 不是 BRPO-style joint observation，它本质上是 **joint support filter**。

### 5.1 old A1 在统计什么量

它先有：

- `raw_rgb_confidence`
- `depth_source_map`
- 从 `depth_source_map` 推出来的 `geometry_tier`

其中：

- 双侧 depth 验证 → `geometry_tier = 1.0`
- 单侧 depth 验证 → `geometry_tier = 0.5`
- fallback / 无效 → `geometry_tier = 0.0`

### 5.2 这些量怎么变成动作

old A1 直接定义：

$$
joint\_confidence = \min(raw\_rgb\_confidence, geometry\_tier)
$$

连续版则是：

$$
joint\_confidence\_{cont} = raw\_rgb\_confidence\_{cont} \cdot geometry\_tier
$$

然后 `joint_depth_target_v2` 甚至不是新生成的，只是旧 `target_depth_for_refine_v2_brpo` 的 copy。

所以 old A1 的逻辑不是：

> 先构造 joint observation，再由其属性决定监督

而是：

> 先分别构造 RGB confidence 和 depth source，再在 downstream 用一个更严格的共同 support 去过滤。

也就是说，old A1 统计的量是：

- RGB 分支的 confidence
- depth 分支的 geometry tier

动作是：

- 用 `min(...)` 或乘法，把它们压成一个共同 trusted support

这是 **统一消费域**，不是 **统一构造观测对象**。

---

## 6. 我们的 new A1 到底统计了什么量，它怎么变成动作

new A1 已经不是 old A1 换壳。它确实做了真正的 joint builder。

### 6.1 它先统计什么量

`joint_observation.py` 里，你们先对每个像素收集四类 depth/source candidate：

- `left_depth`
- `right_depth`
- `both_depth`
- `render_depth`

然后每个 candidate 都有一套打分证据：

- `A(p)`：appearance evidence，来自 `raw_rgb_confidence` / `raw_rgb_confidence_cont`
- `G_k(p)`：geometry consistency，来自 left-right 或 candidate-render consistency
- `S_k(p)`：support strength，来自 overlap / both vs single 等
- `P_k(p)`：source prior，手工 source prior

然后构造每个 candidate 的分数：

$$
score_k(p)=0.35 A(p) + 0.35 G_k(p) + 0.20 S_k(p) + 0.10 P_k(p)
$$

这里先说清楚：这不是 BRPO 的 `C_m`，这是 **candidate competition score**。

### 6.2 这些量怎么变成动作

你们 current new A1 的动作链条是：

1. 用 `score_stack` 取 `best_idx` 和 `best_score`
2. 用 softmax over score stack 得到 `score_prob`
3. 用 `score_prob` 对 `depth_stack` 做加权融合，得到 `fused_depth`
4. 定义：
   - `confidence_depth = best_score`
   - `confidence_rgb = selected appearance`
   - `confidence_joint = sqrt(confidence_rgb * confidence_depth)`
5. 再用 `confidence_joint > min_joint_confidence` 得到 `valid_mask`

所以 new A1 的动作不是“先有 pseudo-frame 后做验证”，而是：

> **先做 candidate competition / depth-source fusion，再从同一套 competition 里派生出 target 和 confidence。**

这一步你们已经做成了完整 observation bundle：

- `pseudo_depth_target_joint_v1`
- `pseudo_confidence_joint_v1`
- `pseudo_confidence_rgb_joint_v1`
- `pseudo_confidence_depth_joint_v1`
- `pseudo_uncertainty_joint_v1`
- `pseudo_source_map_joint_v1`
- `pseudo_valid_mask_joint_v1`

consumer 端也确实有 `pseudo_observation_mode=brpo_joint_v1` 去锁定这套 bundle 的消费方式。

所以 new A1 的进步是非常真实的：

- 不再只是 old A1 那种 downstream joint filter
- 而是真的变成了 upstream joint builder + downstream joint consumer

但是它和 BRPO 的关键差别也很明确：

> **你们 current new A1 的 confidence 主来源是 candidate score；BRPO 的 confidence 主来源是 pseudo-frame 与 reference 的事后几何验证。**

---

## 7. A1 的高层区别：BRPO 更生硬，我们 current new A1 更平滑

这个判断是对的，而且不是模糊感觉，而是机制层面的事实。

### 7.1 BRPO 为什么更硬

BRPO 的 `C_m` 是三值：

- 1.0
- 0.5
- 0.0

而且来自集合关系：

- `M_k ∩ M_{k+1}`
- `M_k ⊕ M_{k+1}`
- `outside union`

它本质是一个 **verification mask**。

### 7.2 我们 current new A1 为什么更平滑

你们是：

- candidate score 是连续值
- `fused_depth` 是 softmax over candidates 的平滑融合
- `confidence_joint = sqrt(conf_rgb * conf_depth)` 也是连续值
- `render_prior` 也作为一个 candidate 合法存在

所以 current new A1 更像：

> **smooth candidate-scoring confidence + smooth target fusion**

### 7.3 为什么“更平滑”反而可能更差

这里不是说平滑天然错，而是 current new A1 的平滑有一个结构性问题：

> **target 和 confidence 来自同一套 candidate score。**

也就是说，如果某个像素的 candidate ranking 本身错了，就会出现：

- 错的 depth target
- 但同时又给它一个偏高的 confidence

于是形成一种“平滑但自洽的错误”。

BRPO 虽然硬，但它在这里更解耦：

- pseudo content 先生成
- confidence 再由真实 reference 去验证

也就是：**生成结果和 confidence 的来源是分开的。**

所以从训练稳定性上说，当前 new A1 可能输给 old A1 或 BRPO-style hard verifier，不一定是因为“平滑”这个方向错，而是因为：

> **你们 current 的平滑 confidence 和 target 同源，缺少独立 verifier。**

---

## 8. A1 的最终总结

### BRPO 在 A1 上统计什么量

- overlap / depth consistency / pose consistency → 用于 fusion 权重
- `I_t^{fix}` 与两侧真实 reference 的 mutual NN correspondence sets `M_k, M_{k+1}` → 用于 confidence mask

### BRPO 这些量怎么变成动作

- overlap confidence → 融合 residual，生成 `I_t^{fix}`
- `M_k, M_{k+1}` → 生成三值 `C_m`
- `C_m` → 同时作用到 RGB 和 depth loss

### 我们 old A1 统计什么量

- `raw_rgb_confidence`
- `depth_source_map -> geometry_tier`

### old A1 这些量怎么变成动作

- `joint_confidence = min(raw_rgb_confidence, geometry_tier)`
- `joint_depth_target` 只是旧 target 复用

### 我们 current new A1 统计什么量

- four depth/source candidates
- appearance / geometry / support / prior 四类 evidence
- candidate score `score_k`

### current new A1 这些量怎么变成动作

- `score_k` → `best_score`, `score_prob`, `fused_depth`
- `best_score` + `confidence_rgb` → `confidence_joint`
- `confidence_joint` → `valid_mask`
- joint bundle → downstream RGB/depth supervision

### A1 的本质差别

- **BRPO**：先生成 pseudo-frame，再由真实 reference 事后验证 pseudo-frame 像素
- **old A1**：分支构造后，再求共同 trusted support
- **current new A1**：先做 candidate competition / fusion，再由同一 competition 派生 target 和 confidence

也就是说：

> **old A1 问的是“共同支持域在哪里”；new A1 问的是“候选解释哪个更可信”；BRPO 问的是“这个 pseudo-frame 像素到底有没有被真实多视角验证”。**

---

# Part II. B3：BRPO 的 SPGM 到底是什么，我们 current B3 到底是什么

## 9. BRPO 的 B3 第一步在做什么：depth partition 不是个名字，它是在按深度顺序给 Gaussian 分层

BRPO 先取所有 Gaussian 的 depth 值 `z_i`，把它们看作一维分布，然后按 quantile 做分段。

paper 用了一维 Wasserstein / quantile clustering 的表述，但实际想法很朴素：

> **把 Gaussian 在深度轴上从近到远排序，再切成 near / mid / far 三段。**

这就是 depth partition。

然后它给每个 Gaussian 一个 depth score，直觉上是：

- 越近，score 越高
- 越远，score 越低

所以 BRPO 的 depth partition 在统计的是：

> **这个 Gaussian 在场景深度结构里属于哪一层，以及它的深度优先级是多少。**

它不是动作，只是 scene structure encoding。

---

## 10. BRPO 的 density entropy 在做什么：它在判断“density 这个指标此时值不值得信”

BRPO 第二步对 density `\rho_i` 做 histogram，得到 `p_b`，再算归一化 Shannon entropy：

$$
H(\rho) = -\frac{1}{\log B} \sum_b p_b \log(p_b+\epsilon)
$$

这个量的含义是：

- entropy 低：density 分布集中，说明 density 很有辨别力
- entropy 高：density 分布平均，说明 density 区分谁重要这件事不太可靠

然后 paper 定义 entropy-aware density score：

$$
\hat s_i^{(\rho)} = \tilde\rho_i (1-\beta \bar H) + \bar H \gamma
$$

这一步的本质不是再给 density 起个名字，而是在做一个**全局校准**：

> **当前场景里，density 这个信号本轮应不应该被重用、该信多少。**

---

## 11. BRPO 的 unified score 在做什么：它把 depth / density 合成一个 per-Gaussian scene score

paper 定义：

$$
S_i = \alpha \hat s_i^{(z)} + (1-\alpha)\hat s_i^{(\rho)}
$$

这一步的含义是：

- depth 告诉你它在几何结构上更近还是更远
- density score 告诉你它在当前 density distribution 下是否更有辨别力
- `S_i` 把这两者揉成一个 scene-level score

也就是说，到这里 BRPO 仍然还没动作，它还只是在做：

> **对每个 Gaussian 统计一个 scene perception score。**

---

## 12. BRPO 的动作是怎么来的：它不是改 gradient，而是改 opacity participation

paper 接下来定义：

$$
p_i^{drop} = r \cdot w_{cluster(i)} \cdot S_i
$$

然后：

$$
m_i \sim Bernoulli(1-p_i^{drop})
$$

最后：

$$
\alpha_i \leftarrow \alpha_i \cdot m_i
$$

这里要抓住两个关键点：

### 12.1 动作对象是 opacity

它乘的是 `\alpha_i`，不是 `_xyz.grad`。  
这意味着它直接改变的是 **当前 render 中 Gaussian 参与成像的强度**。

### 12.2 动作规律是 stochastic per-Gaussian masking

每个 Gaussian 都有自己的 `p_i^{drop}`，然后做 Bernoulli 采样。  
它不是 deterministic keep-top-k，也不是一个全局阈值。

所以 BRPO 的 B3 不是：

> 先渲染完，再把某些点的 gradient 乘小

而是：

> **先统计每点 score，再把这个 score 直接变成当前 forward 里的 opacity masking。**

---

## 13. 我们 old B3 到底统计什么量，怎么变成动作

old B3 的地位就是你们文档里说的：`deterministic grad/state probe`。

### 13.1 它统计什么量

它也会算一些 state-related score / cluster diagnostics，但其核心问题不是统计不统计，而是：

> **统计完以后，动作只发生在 backward 之后。**

### 13.2 它怎么变成动作

old B3 的主动作是给 `_xyz.grad` 乘一个 state-dependent scale。也就是：

- render 已经做完
- pseudo loss 已经 backward 完
- 再把 `_xyz.grad` 缩一下
- 再 optimizer.step

所以 old B3 的动作对象是：

- `_xyz.grad`

不是：

- `opacity`
- `forward participation`

这和 BRPO 显然不是一类 operator。

---

## 14. 我们 current new B3 的第一层：stats.py 到底统计什么量

current new B3 已经不再只是 old B3。

`collect_spgm_stats(...)` 现在统计的是：

- `support_count`：accepted pseudo support
- `population_support_count`：current train-window support
- `depth_value`：accepted pseudo weighted median depth
- `density_proxy`
- `struct_density_proxy`
- `active_mask`
- `population_active_mask`

但这里必须把一句话讲死：

> **current 主逻辑的 active universe 仍然强绑在 `support_count > 0` 上。**

因为 `stats.py` 顶部文档已经明确说了：

- `active_mask` 仍然 derived from accepted pseudo support
- `population_active_mask` 是新加的 scene-aware summary，但当前主逻辑保留 backward compatibility

所以 current new B3 的 stats 不是纯 scene population stats，而是：

> **accepted pseudo active set + 一些 population-aware 补充统计。**

---

## 15. 我们 current new B3 的第二层：score.py 到底统计什么量

你们现在不是一个 score，而是三个：

- `importance_score`
- `ranking_score`
- `state_score`

### 15.1 你们自己的 depth partition 在做什么

`build_depth_partition(...)`：

- 取 active Gaussian 的 `depth_value`
- 排序
- 分成 near / mid / far 三段
- 定义
$$
depth\_score = 1 - \frac{z-z_{\min}}{z_{\max}-z_{\min}}
$$

这一步和 BRPO 的高层意图是接近的：

> **都在按深度顺序，把 Gaussian 分层。**

### 15.2 你们自己的 density entropy 在做什么

`compute_density_entropy(...)`：

- 对 active Gaussian 的某种 density value 做 histogram
- 算 normalized Shannon entropy

这一点和 BRPO 的高层想法也接近：

> **都在判断当前 density distribution 是否有辨别力。**

### 15.3 真正关键的差别：你们 action 用的不是 paper 那个 unified score

你们先算：

$$
importance\_raw = \alpha \cdot depth\_score + (1-\alpha)\cdot density\_score
$$

但真正给 manager action 用的，不是这个，而是：

$$
state\_score = 0.45\cdot state\_density\_norm + 0.35\cdot population\_support\_norm + 0.20\cdot depth\_score
$$

所以你们 current new B3 的关键事实是：

> **最终控制 participation 的，不是 BRPO 的 `S_i`，而是你们工程化重写后的 `state_score`。**

这个 `state_score` 比 BRPO 更偏：

- support / visibility
- population window summary
- struct density

而不是仅仅 depth + entropy-aware density。

---

## 16. 我们 current new B3 的第三层：manager.py 到底怎么把 score 变成动作

这一步一定要讲清楚，因为这里才是真差异。

### 16.1 `_build_state_candidate_mask(...)` 在做什么

它不是 drop probability。它做的是：

- 对每个 cluster 单独看 `state_score`
- 取 cluster 内某个 quantile 以下的点
- 标成 `candidate_mask`

也就是说，这一步在回答：

> **这一层（near/mid/far）里，哪些点属于低状态分数子集。**

### 16.2 `_build_participation_render_mask(...)` 在做什么

这一步不是 per-Gaussian Bernoulli，而是：

- 在每个 cluster 的 candidate 子集里
- 按 `keep_ratio` 算出要保留多少个
- 取 top-k `state_score`
- 其余全部 drop
- 得到一个 boolean `render_mask`

所以它的动作规律是：

> **cluster-wise bottom-quantile candidate selection + deterministic keep-top-k + boolean render mask**

这和 BRPO 的：

> **per-Gaussian probability → Bernoulli masking → opacity attenuation**

不是一个 action law。

### 16.3 `apply_spgm_state_management(...)` 真正做了什么

current manager 有三个 mode：

- `summary_only`
- `xyz_lr_scale`
- `deterministic_participation`

其中 `deterministic_participation` 的意义是：

> **不再只在 backward 后缩 grad，而是为下一轮 pseudo render 预备一个 participation mask。**

所以 current new B3 最大的真实进步是：

> **动作位置从 post-backward grad scaling，推进到了 pre-render participation control。**

这一步必须承认，它不是旧 B3 换壳。

---

## 17. 我们 current new B3 的动作什么时候生效

还有一个没有完全跨过去的 gap：你们现在的 B3 participation 还是“滞后一拍”的。
DESIGN 里写得非常明确：当前 B3 第一版的接入语义是
iter t: render pseudo / real -> assemble joint loss -> backward once -> local gating / SPGM summary -> build participation mask for iter t+1 -> optimizer.step，然后 iter t+1 的 pseudo render 才消费这个 participation_render_mask。这意味着当前 population control 对当前这一步 residual landscape 的作用不是同步的，而是 t 统计、t+1 生效。BRPO 则不是这样：它的 confidence 和 Gaussian management 都是直接参与当前伪观测如何影响当前优化的。数学上，这会带来一个差别：你们现在的参与控制更像对优化轨迹做 delayed operator，而不是对当前 forward model 的即时改写。这个差别在弱监督场景里会很伤，因为 pseudo signal 本来就小，再滞后一拍，等效耦合会更弱。

`run_pseudo_refinement_v2.py` 里，StageB 的时序是：

1. 当前 iteration 开始，先读取 `current_pseudo_render_mask`
2. 用它跑当前 pseudo render
3. 当前 iteration 结束后，再通过 `maybe_apply_pseudo_local_gating(...)` 生成下一轮的 `_spgm_participation_render_mask`
4. 把它存起来，供下一轮 pseudo render 使用

所以 current new B3 是：

$$
\text{iter } t \text{ 统计 / 决策} \rightarrow \text{iter } t+1 \text{ 生效}
$$

也就是一个：

> **one-step delayed pre-render participation controller**

这和 BRPO 更像“当前 score 直接作用于当前 opacity participation”不同。

---

## 18. B3 的本质差别到底是什么

现在把 BRPO / old B3 / current new B3 摊平讲。

### 18.1 BRPO 在 B3 上统计什么量

- Gaussian depth distribution → near / mid / far partition
- density distribution entropy → 判断 density 指标本轮是否有辨别力
- per-Gaussian unified score `S_i`

### 18.2 BRPO 这些量怎么变成动作

- `S_i` + cluster weight → `p_i^{drop}`
- `p_i^{drop}` → Bernoulli sample `m_i`
- `m_i` → 当前优化中的 `opacity participation`

### 18.3 我们 old B3 统计什么量

- 一些 state / cluster related diagnostics

### 18.4 old B3 这些量怎么变成动作

- backward 后对 `_xyz.grad` 做 deterministic scaling

### 18.5 我们 current new B3 统计什么量

- accepted pseudo support
- population window support
- weighted median depth
- density proxy / struct density proxy
- `importance_score / ranking_score / state_score`

### 18.6 current new B3 这些量怎么变成动作

- `state_score` → cluster 内 quantile candidate
- candidate → keep-top-k selector
- selector → boolean `participation_render_mask`
- `participation_render_mask` → 下一轮 pseudo render participation

---

## 19. B3 的综合总结：我们和 BRPO 的真正差别，不是“有没有 depth partition / entropy 这些名词”，而是以下六点

### 差别 1：控制的 universe 不一样

- **BRPO**：更像 scene-level Gaussian population
- **我们 current**：仍然主要锚在 accepted pseudo active set，上面再加 population summary

### 差别 2：最终 action 用的 score 不一样

- **BRPO**：paper 里的 unified score `S_i`
- **我们 current**：工程化重写的 `state_score`

### 差别 3：动作变量不一样

- **BRPO**：opacity `\alpha_i`
- **我们 current**：boolean `participation_render_mask`

### 差别 4：动作规律不一样

- **BRPO**：per-Gaussian stochastic Bernoulli masking
- **我们 current**：cluster-wise quantile + deterministic keep-top-k

### 差别 5：动作时序不一样

- **BRPO**：当前统计当前作用的 paper-style写法
- **我们 current**：iter `t` 统计，iter `t+1` 生效的 delayed controller

### 差别 6：作用域不一样

- **BRPO**：更像 scene-level population manager
- **我们 current**：pseudo-branch scoped participation controller

---

## 20. 为什么 current new B3 可能 weak-negative

### 20.1 它现在的动作比 BRPO 更硬

你们不是 `opacity attenuation`，而是 boolean render mask。  
这意味着一旦被 drop，下一轮 pseudo render 里它直接不参与，而不是“参与得更弱”。

### 20.2 它的 action 还太受 pseudo active set 波动影响

因为 `active_mask` 仍然强依赖 accepted pseudo support。  
所以 current new B3 更容易抑制“这轮没被看好”的点，而不一定是 scene-level 真坏点。

### 20.3 它的 score 更像 visibility/state controller，而不是 BRPO-style importance controller

`state_score` 里 `population_support_norm` 占比较高，所以动作可能更多在响应“这一轮谁更常被看见”，而不是“谁在 scene population 里更该被弱化”。

### 20.4 一拍滞后会导致控制振荡

当前 iteration 用上一轮 mask，当前 iteration 结束才更新下一轮 mask。  
如果 pseudo view / support 波动大，boolean controller 很容易 oscillate。

---

## 21. 如果要继续把 current B3 往 BRPO 靠，最值得做的不是继续扫 keep ratio，而是这三步

### 第一步：把动作变量从 boolean render mask 改成 deterministic opacity attenuation

先做：

$$
\alpha_i^{eff} = \alpha_i \cdot s_i^{part}, \qquad s_i^{part}\in[0,1]
$$

而不是继续：

- candidate → keep-top-k → boolean drop

这会把 current B3 从“硬 selector”往 BRPO 的“opacity participation manager”推近。

### 第二步：把 participation action 用的主 score 从 `state_score` 往 BRPO-like importance score 收

- `ranking_score` 继续给 update permission / selector
- `state_score` 继续给 diagnostics
- participation action 尽量改成基于更接近 `S_i` 的 unified importance score

### 第三步：把控制 universe 从 `active_mask` 往 `population_active_mask` 推

不要让 current B3 永远停留在 “accepted pseudo active set controller”。  
否则它永远更像 local visibility controller，而不是 scene population manager。

### 第四步：deterministic opacity attenuation 站稳后，再进 stochastic Bernoulli masking

最后才做：

$$
m_i \sim Bernoulli(1-p_i^{drop}), \qquad \alpha_i^{eff} = \alpha_i m_i
$$

而不是现在这种 hard top-k 直接拿来代理 paper 的 stochastic masking。

---

# 最终一句话总结

## A1

- **BRPO**：先生成 pseudo-frame `I_t^{fix}`，再由两侧真实 reference 对 pseudo-frame 像素做事后几何验证，得到统一 confidence `C_m`，最后 RGB/depth 共用这个 `C_m`。  
- **old A1**：先各自构造 RGB / depth signal，再 downstream 求共同 trusted support。  
- **current new A1**：先做 candidate competition / source-depth fusion，再从同一套 competition 中同时派生 target 和 confidence。  

所以 A1 的核心差别是：

> **BRPO 的 confidence 是 pseudo-frame verification；我们的 current new A1 的 confidence 是 candidate-scoring confidence。**

## B3

- **BRPO**：先按深度顺序对 Gaussian 分层，再用 density entropy 校准 density 是否可信，合成 unified score `S_i`，最后把 `S_i` 直接变成当前 forward 中的 opacity stochastic masking。  
- **old B3**：backward 后给 `_xyz.grad` 乘缩放。  
- **current new B3**：基于 accepted pseudo active set + population summary 统计 state score，再做 cluster-wise candidate selection + keep-top-k，生成下一轮 pseudo render 用的 boolean participation mask。  

所以 B3 的核心差别是：

> **BRPO：score → current-step opacity masking**  
> **我们 current：state_score → next-step boolean render selection**


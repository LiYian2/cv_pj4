# BRPO Fusion / Mask / SPGM / Gaussian-Subset Refine 理论分析（更新版）

## 0. 说明

这份文档只讨论**理论层面的区别、数学语义、以及如果要更接近 BRPO 应该怎样理解与实现**。

不写工程排期，不写实验执行顺序，不写 agent 任务拆分。  
重点回答六个问题：

1. BRPO 的 fusion 在理论上到底是什么；
2. 你们当前 fusion 和 BRPO 除了 UNet 之外还差什么；
3. BRPO 的 confidence mask 原始语义是什么，它和 depth 的关系是什么；
4. 你们当前 `seed_support / train_mask / target_depth_for_refine / v2` 这套分层语义为什么会偏离 BRPO；
5. 为什么当前更应该优先考虑 **Gaussian 子集 refine / local gating**，而不是直接上 full SPGM；
6. 如果继续朝 BRPO 方向改（尤其是 fuse 和 mask），是否会让 depth signal 太少的问题变好。

---

## 1. BRPO 的 fusion：它真正要解决的是什么

BRPO 的 fusion 不是“把左右两张修复图按某种 confidence 混合”，而是：

> 对同一个 target pseudo frame，给定两张来自不同 reference 条件的候选修复图，判断**每一张候选图在几何上相对真实 reference 有多可信**，然后按这种几何可信度来做残差融合。

因此，BRPO fusion 的核心语义是：

- 比较对象不是“left 分支和 right 分支彼此是否一致”；
- 而是“candidate 1 相对 reference 1 是否几何可信”“candidate 2 相对 reference 2 是否几何可信”。

这是一个 **target ↔ reference** 的几何 overlap 问题，不是一个 **left ↔ right** 的互相比对问题。

---

## 2. BRPO fusion 的数学定义

设 target pseudo frame 为 \(t\)，两个参考帧为 \(a,b\)。  
给定 target 视图上的像素 \(p_t\) 和深度 \(d_t\)，把它投影到参考视图：

$$
p_a = \Pi\!\bigl(K,\; T_a T_t^{-1}\, \Pi^{-1}(p_t, d_t)\bigr)
$$

其中：

- \(\Pi^{-1}(p,d)\)：把像素与深度反投影到 3D；
- \(T_a T_t^{-1}\)：把 target 坐标系下的 3D 点变到 reference 坐标系；
- \(\Pi(\cdot)\)：再投影到参考图像平面。

然后定义 overlap mask：

$$
O_{ta}(p_t)= \mathbf{1}[p_a \in \Omega_a] \cdot \mathbf{1}[d_a(p_a) > \epsilon]
$$

也就是 target 像素经深度投影后能落在 reference 图像内，且 reference 对应位置存在有效深度。

接着定义深度一致性项：

$$
s_d^{(a)}(p_t)=
\exp\!\left(
-\frac{|d_t(p_t)-d_a(p_a)|}{(d_t(p_t)+d_a(p_a))/2+\epsilon}
\right)
$$

再定义相机平移一致性项（视图基线惩罚）：

$$
s_t^{(a)} = \exp(-\|t_t - t_a\|_2)
$$

于是 combined overlap confidence 写成：

$$
C_{t,a}(p_t)= O_{ta}(p_t)\, s_d^{(a)}(p_t)\, s_t^{(a)}
$$

对另一侧 reference \(b\) 同理得到 \(C_{t,b}(p_t)\)。

然后 BRPO 用它们做归一化权重：

$$
W_a(p_t)=\frac{C_{t,a}(p_t)}{C_{t,a}(p_t)+C_{t,b}(p_t)+\epsilon}, \qquad
W_b(p_t)=\frac{C_{t,b}(p_t)}{C_{t,a}(p_t)+C_{t,b}(p_t)+\epsilon}
$$

最后对候选图残差做融合。设两张候选修复图为 \(\tilde I_t^{(a)}, \tilde I_t^{(b)}\)，原始 render 为 \(I_t\)，定义残差：

$$
r^{(a)}=\tilde I_t^{(a)}-I_t, \qquad
r^{(b)}=\tilde I_t^{(b)}-I_t
$$

则最终 fused pseudo frame 为：

$$
I_t^{fix}= I_t + W_a \odot r^{(a)} + W_b \odot r^{(b)}
$$

### 这一套式子的核心含义

它不是在问：

- 左右分支谁更清晰；
- 左右分支彼此谁更像；

而是在问：

- 在这个 target 像素位置上，哪一个候选分支**更能被真实 reference 的几何 overlap 解释**。

这就是 BRPO fusion 最本质的语义。

---

## 3. 你们当前 fusion 的数学形式

你们当前 `pseudo_fusion.py` 的逻辑可以抽象成：

### 3.1 分支分数

对左右分支分别定义：

$$
S_L = C_L \cdot G_L \cdot V_L,\qquad
S_R = C_R \cdot G_R \cdot V_R
$$

其中：

- \(C_L, C_R\)：branch confidence；
- \(G\)：render depth 与 branch depth 的一致性 gate；
- \(V\)：整张图级别的 view-level scalar。

具体地，depth gate 写成：

$$
G_L =
\exp\!\left(
-\frac{|D_r - D_L|}{\tau_d D_r + \epsilon}
\right),\qquad
G_R =
\exp\!\left(
-\frac{|D_r - D_R|}{\tau_d D_r + \epsilon}
\right)
$$

view-level weight 是一个全局标量：

$$
V = \alpha_1 \cdot \text{mean\_confidence} + \alpha_2 \cdot \text{valid\_ratio}
$$

### 3.2 左右分支之间的 agreement

你们再定义：

- RGB agreement：
$$
A_{rgb}(p)=\exp\!\left(-\frac{\|I_L(p)-I_R(p)\|_1}{\tau_{rgb}}\right)
$$

- 可选 depth agreement：
$$
A_{depth}(p)=\exp\!\left(
-\frac{|D_L(p)-D_R(p)|}{\tau_{ld} D_r(p)+\epsilon}
\right)
$$

最终：
$$
A(p)=A_{rgb}(p) \quad \text{或}\quad A(p)=A_{rgb}(p)A_{depth}(p)
$$

### 3.3 融合权重与残差融合

你们先构造未归一化权重：

$$
\widetilde W_L = A \cdot S_L,\qquad
\widetilde W_R = A \cdot S_R
$$

再归一化：

$$
\hat W_L = \frac{\widetilde W_L}{\widetilde W_L+\widetilde W_R+\epsilon},\qquad
\hat W_R = \frac{\widetilde W_R}{\widetilde W_L+\widetilde W_R+\epsilon}
$$

最后做 residual fusion：

$$
I_F = I_r + \hat W_L (I_L-I_r) + \hat W_R (I_R-I_r)
$$

你们的 fused confidence 不是单独重新验证得到，而是从融合权重反推：

$$
C_F = \min\!\left(1,\frac{\widetilde W_L+\widetilde W_R}{\tau_c}\right)\cdot A
$$

---

## 4. 你们当前 fusion 和 BRPO fusion 的关键区别

除了“你们没训练 UNet”之外，核心区别至少有五条。

### 4.1 区别一：权重语义不同

BRPO：

$$
W_i \propto C_{t,i}(p)
$$

这里的 \(C_{t,i}(p)\) 是 **target ↔ reference** 的几何 overlap confidence。

你们当前：

$$
\hat W_i \propto C_i \cdot G_i \cdot V_i
$$

本质是 **branch 自己的 confidence × render-centered depth gate × 全局分支权**。

也就是说：

- 你们的权重更像“哪边看起来更靠谱”；
- BRPO 的权重更像“哪边相对真实 reference 更几何可信”。

### 4.2 区别二：你们的 agreement 不是 BRPO 的主语义

BRPO 需要的是：

- candidate \(1\) 相对 reference \(1\) 的 overlap confidence；
- candidate \(2\) 相对 reference \(2\) 的 overlap confidence。

你们现在主要额外看的是：

- left repaired 和 right repaired **彼此之间**的 RGB / depth agreement。

这两者不是一回事。

左右两张 repaired 图彼此一致，不代表它们几何上对；  
它们可能只是一起 hallucinate 得很一致。

### 4.3 区别三：你们的 agreement 对 fused image 本身几乎不起主导作用

因为你们代码里：

$$
\widetilde W_L = A S_L,\qquad \widetilde W_R = A S_R
$$

归一化后：

$$
\hat W_L=\frac{AS_L}{AS_L+AS_R+\epsilon},\qquad
\hat W_R=\frac{AS_R}{AS_L+AS_R+\epsilon}
$$

当 \(A\) 对左右两边是同一张图时，它在归一化中基本抵消掉。

所以：

- 当前 agreement 对 fused image 的左右分配权重影响有限；
- 它更主要影响的是你们后面构造出来的 `C_fused`。

而 BRPO 中 overlap confidence 是**直接决定 fusion weight** 的。

### 4.4 区别四：你们有一个 BRPO 没有的全局 heuristic：view-level scalar \(V\)

你们的 \(V_L,V_R\) 来自整张图的 `mean_confidence + valid_ratio`。  
这是一个全局 heuristic，会把整张图一侧“整体偏强/偏弱”编码进所有像素。

BRPO 没有这种形式的全局 branch heuristic。  
它只有一个 pose-consistency scalar \(s_t\)，但那是写在 overlap confidence 语义里的，不是外加一个整图 scalar 去偏置整侧分支。

### 4.5 区别五：你们把 fusion 与最终训练 confidence 的语义部分揉在一起了

你们现在：

- 先做 fusion；
- 再用 fusion 权重直接构造 `C_fused`；
- 再让这个 `C_fused` 影响下游。

BRPO 不是这样。  
BRPO 的顺序是：

1. 用 overlap confidence 做 fusion，得到 \(I_t^{fix}\)；
2. 再在 \(I_t^{fix}\) 上做几何/对应验证，得到最终训练用 confidence mask \(C_m\)。

也就是说：

- **fusion**：回答“哪一张候选图应该贡献更多残差”；
- **mask**：回答“这个 fused pixel 适不适合进入后续训练”。

BRPO 这两件事是分开的。

---

## 5. 如果想实现更接近 BRPO 的 fusion，从理论上应该怎么做

这里先不谈工程排期，只谈理论实现思路。

### 5.1 暂时保留你们现有的 left/right repaired 图来源

也就是先把：

$$
\tilde I_t^{(1)} \leftarrow I_L,\qquad
\tilde I_t^{(2)} \leftarrow I_R
$$

先不强求它们必须经过 BRPO 原论文里的 pseudo-view deblur UNet。

换句话说，先只替换 **fusion 语义**，不替换 candidate repaired images 的来源。

### 5.2 不再使用当前 \(S=C\cdot G\cdot V\)

改成直接构造：

$$
C_{t,1}(p),\qquad C_{t,2}(p)
$$

这里每个 \(C_{t,i}\) 都来自：

- target pseudo view 到对应 reference 的几何投影 overlap；
- target/reference 的 depth-consistency；
- target/reference 的 pose-consistency。

即：

$$
C_{t,i}(p)= O_{t,i}(p)\, s_d^{(i)}(p)\, s_t^{(i)}
$$

### 5.3 用 BRPO 的方式直接归一化成 fusion 权重

$$
W_i(p)=\frac{C_{t,i}(p)}{C_{t,1}(p)+C_{t,2}(p)+\epsilon}
$$

### 5.4 再做 residual fusion

$$
I_t^{fix}=I_t+W_1\odot (\tilde I_t^{(1)}-I_t)+W_2\odot (\tilde I_t^{(2)}-I_t)
$$

### 5.5 最后再单独做 confidence mask inference

不要再把 fusion 权重直接当训练 confidence。  
而是：

1. 先得到 \(I_t^{fix}\)；
2. 再在 \(I_t^{fix}\) 与前后真实 reference 之间做 robust feature / correspondence 验证；
3. 生成 BRPO 语义的 confidence mask。

这一步是关键，因为它把：

- “哪张 candidate repaired image 更应该被混进 fused pseudo”  
和
- “哪一个 fused pseudo pixel 更应该被训练时信任”

重新分开了。

---

## 6. BRPO 的 mask 原始语义：它本质上是什么

BRPO 的 confidence mask \(C_m\) 是定义在 fused pseudo frame \(I_t^{fix}\) 上的。  
核心问题是：

> 这个 fused pseudo pixel，能不能在真实前后 reference 里找到几何一致的对应点？

论文通过 robust correspondence network 建立两组 correspondence：

- \(M_k\)：\(I_t^{fix}\) 与前一 reference 的对应；
- \(M_{k+1}\)：\(I_t^{fix}\) 与后一 reference 的对应。

然后定义：

$$
C_m(p)=
\begin{cases}
1.0,& p\in M_k\cap M_{k+1}\\
0.5,& p\in M_k\oplus M_{k+1}\\
0.0,& p\notin M_k\cup M_{k+1}
\end{cases}
$$

所以，BRPO 的 confidence mask 主体语义是：

- **高置信**：双边都能几何对应；
- **中置信**：单边能对应；
- **低置信**：都不能对应。

注意，这不是“直接由 depth threshold 生成”的 mask。  
它是 **feature / correspondence 驱动的几何验证结果**。

---

## 7. BRPO 的 mask 和 depth，到底是什么关系

这是一个很容易混淆但很关键的点。

### 7.1 BRPO 的最终 mask 不是直接由 depth 生成的

更准确地说：

- **mask 的直接生成机制**：基于 fused RGB / feature correspondence；
- **不是** 直接对 depth map 做阈值判断得到最终 confidence mask。

### 7.2 但 depth 仍然间接影响 mask

因为 depth 先参与了 fusion。

也就是说：

1. depth 通过 overlap confidence 影响 \(I_t^{fix}\) 的生成；
2. \(I_t^{fix}\) 的质量又会影响后续 correspondence 与 \(C_m\)。

所以不能说 mask 和 depth “完全没关系”；  
但可以说：

- **最终 mask 的主语义不是 depth threshold**
- **depth 主要在 fusion、refine、SPGM 中扮演角色**

这就是 BRPO 的原始语义分工。

---

## 8. 你们现在为什么会偏离 BRPO：`seed_support / train_mask / target_depth_for_refine / v2`

你们现在的分层语义，本质上是被 **coverage 太低** 逼出来的。

因为在你们当前 case 下：

- raw verified support 很少；
- fallback 极高；
- 就连 support 区域里的 correction magnitude 也不大。

在这种情况下，如果完全照 BRPO 那种“只在 \(I_t^{fix}\) 上做一个离散 confidence mask，然后 RGBD 都吃它”的简单语义，depth supervision 很可能会稀到几乎训不动。

于是你们才被迫拆成：

1. `seed_support`：原始高置信 seed；
2. `train_mask`：传播后的可训练区域；
3. `target_depth_for_refine`：M3 blended target；
4. `target_depth_for_refine_v2`：M5 densified target。

这套拆分的初衷是合理的。  
问题不在于“拆复杂了”，而在于：

> 复杂化之后，最好不要再让这一整套增强语义同时支配 RGB 和 depth 两个分支。

---

## 9. 更合理的语义分工：RGB / depth 应该怎么分

这里给出一个纯理论判断，不谈具体工程接口。

### 9.1 RGB 分支：尽量接近 BRPO 原始语义

也就是：

- RGB supervision 的主 mask 应该尽量来自 fused pseudo 上的 raw confidence / correspondence semantics；
- 不要默认继承 propagation 后的大 train region。

因为 propagation 本质上更像：

- 为了 depth 可训练性而扩张出来的 candidate region；
- 而不是“这个 RGB pseudo pixel 本身就很可信”。

### 9.2 Depth 分支：允许保留你们现在的工程增强

depth 当前确实太稀，所以：

- propagation
- M3 blended target
- M5 densify
- confidence-aware densify

这些都继续保留是合理的。

所以，最合理的不是“完全回到 BRPO 简单版”，而是：

- **RGB：尽量回归 BRPO 的简单原始语义**
- **Depth：保留你们为 coverage 被迫做出来的复杂增强**

这样做不是后退，而是把复杂化重新限定到它真正需要服务的地方。

---

## 10. 继续朝 BRPO 改 fuse 和 mask，会不会让 depth signal 变强？

这是最关键的边界判断。

### 10.1 会让信号更“干净”、更“闭合”

改成更 BRPO 风格的 fuse + mask，最可能带来的改善是：

1. fusion 的权重语义更正确；
2. fused pseudo image 的来源更可解释；
3. confidence mask 的定义更闭合；
4. RGB supervision coverage 和可信度可能提高；
5. densify 的输入语义也会更干净。

也就是说，它更可能改善的是：

- signal purity
- signal semantics
- signal utilization

### 10.2 但它不太会单独把 depth signal 从“少”变成“很多”

原因很简单：

当前问题不只是 mask 语义不对，而是：

1. raw verified support 本身就少；
2. support 区域里的 correction magnitude 也小；
3. 当前 pseudo depth 更像微弱 nudging，而不是强几何驱动力。

所以：

- 改 fuse / mask 更像“提高现有信号的利用率与可信度”；
- 不像“凭空制造大量强 depth supervision”。

### 10.3 因此，对 depth scarcity 的准确判断应该是

如果继续朝 BRPO 的方向改：

- **mask coverage** 大概率会变好一些；
- **RGB/raw confidence semantics** 会更合理；
- **下游 refine 的输入质量** 会更高；

但：

- **verified depth ratio 不一定会大幅增加**
- **correction magnitude 也不一定会突然变强**

所以它是一个**必要但不充分**的改进方向。

---

## 11. 为什么当前更应该优先考虑 Gaussian 子集 refine / local gating

你们当前最核心的结构问题之一不是 loss 不降，而是：

> 弱、局部、稀疏的 pseudo supervision 正在推动全局 Gaussian 参数。

这会产生典型现象：

- pseudo-side loss 能降；
- replay / held-out 视角却可能退化。

因此当前最需要的不是“再加一个大正则”，而是让：

$$
\text{supervision scope} \quad \leftrightarrow \quad \text{optimization scope}
$$

重新匹配。

---

## 12. Gaussian 子集 refine 的理论意义

高斯子集 refine 的本质，不是“换个正则”，而是：

- pseudo 分支只允许更新 **pseudo 可见且 signal gate 通过** 的 Gaussian 子集；
- real branch 继续保留更全局的纠偏能力。

设每个 Gaussian 的 pseudo-side update mask 为 \(m_j \in \{0,1\}\)，则 pseudo 分支对某个参数 \(\theta_j\) 的更新变成：

$$
\nabla_{\theta_j}^{pseudo} \leftarrow m_j \cdot \nabla_{\theta_j}^{pseudo}
$$

其中 \(\theta_j\) 可以是：

- 只对 \(\mu_j\)（xyz）；
- 或对 \((\mu_j,\eta_j)\)（xyz + opacity）。

这个 \(m_j\) 的来源可以被理论化为：

- sampled pseudo views 的 visibility union；
- 再与 signal gate（support ratio / verified ratio / correction magnitude）相交。

### 它解决的是什么

1. 降低 underconstrained global drift；
2. 让 pseudo supervision 的影响局部化；
3. 比 full SPGM 更直接命中你们当前主矛盾。

---

## 13. SPGM 在理论上是什么，为什么现在不是最高优先级

BRPO 的 Scene Perception Gaussian Management（SPGM）本质上是：

- 用 joint depth-density information 去识别 floating / under-optimized / unstable Gaussians；
- 再通过 importance score 与 stochastic masking / management 去稳定 joint optimization。

所以：

- SPGM 更像 **stabilizer**
- 不像 **signal amplifier**

### 为什么这很重要

如果上游 signal 本身很弱，那么 SPGM 不会凭空创造更多监督。  
它更多解决的是：

- 有监督时别把 map 带坏；
- 有错误 pseudo 时降低副作用。

因此在你们当前阶段：

- full SPGM 不是不重要；
- 但它不应先于“更接近 BRPO 的 fuse / mask 语义”和“Gaussian 子集 refine / local gating”。

---

## 14. 最终判断：当前最合理的理论结论

### 14.1 关于 fusion
你们当前 fusion 和 BRPO 的真正差异，不只是 UNet，而是：

- 你们现在的权重语义是 `branch confidence × render-depth gate × global view scalar`
- BRPO 的权重语义是 **target ↔ reference overlap confidence**

因此，如果要更接近 BRPO，最值得改的是：

> 把 fused image 的权重定义，从 “left ↔ right 彼此 agreement” 的辅助语义，换回 “target ↔ reference 的几何 overlap confidence” 这个主语义。

### 14.2 关于 mask
BRPO 的最终 confidence mask 更接近：

- fused RGB / feature correspondence 上的几何验证结果

而不是：

- 由 depth support / densify / propagation 直接定义的训练 mask。

所以，如果你们继续朝 BRPO 方向改：

- RGB mask 应尽量回归 raw confidence / correspondence semantics；
- depth 继续保留你们现在更复杂的 coverage-enhancement 机制。

### 14.3 关于 depth scarcity
改 fuse 与 mask：

- 会让 depth / mask / confidence 语义更干净、更闭合；
- 可能会让 mask coverage 更好；
- 可能会提高 densify 输入质量；

但：

- 不太可能单独把 depth signal 从“很少”直接变成“很多”。

因此这类改动更像：

- **提高 signal purity 与 utilization 的必要条件**
- 而不是 **单独解决 signal scarcity 的充分条件**

### 14.4 关于 refine
在当前阶段，最值得优先考虑的 map-side 方向不是 full SPGM，而是：

- **Gaussian 子集 refine / local gating**

因为它直接对准当前主矛盾：

> 弱局部监督去推动全局 Gaussian 扰动。

---

## 15. 一句话压缩

**如果继续朝 BRPO 方向改，最值得优先做的是：把 fusion 权重换成“target ↔ reference”的几何 overlap confidence，把最终 mask 回到 fused RGB / correspondence 语义，同时让 pseudo 分支只作用于局部 Gaussian 子集。这样会让 signal 更干净、更闭合，也更不容易把弱监督放大成全局退化；但它大概率不会单独解决 depth signal 本身过少的问题。**

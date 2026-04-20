# EXTERNAL_ANALYSIS.md - Mask 与 Gaussian Gating 深度分析

> 更新时间：2026-04-20 15:42 (Asia/Shanghai)
> 分析者：Noah (OpenClaw assistant)

本文档整合 MASK_DESIGN.md 与 REFINE_DESIGN.md 的代码级分析，聚焦两个核心问题：
1. 四种 mask/observation 模式的信息流差异，及为什么 brpo_style_v1 暂时弱于 old A1
2. 三种 B3 gaussian gating 模式的机制差异，及为什么效果几乎无区别

---

## Part 1: Mask/Observation 模式分析

### 1.1 四种模式概览

| 模式 | Confidence 来源 | Target 来源 | Confidence 形态 | 稳定性 |
|------|----------------|-------------|----------------|--------|
| BRPO (论文) | $M_{left} \cap M_{right}$ → mutual NN verification | BRPO verified target builder | 离散三档 | 语义正确，工程未知 |
| old A1 | $\min(\text{rgb\_conf}, \text{geometry\_tier})$ | 复用 $\text{target\_depth\_v2\_brpo}$ | 半连续 | **稳**（保守但成熟） |
| new A1 | $\text{score\_stack} \to \text{confidence\_joint}$ | $\text{score\_stack} \to \text{fused\_depth}$ | 连续 | 不稳（target/confidence 同源） |
| brpo_style_v1 | $\text{support\_sets} \to C_m$ | verified projected depth composition | 离散三档 | 比新 A1 强但弱于 old A1 |

---

### 1.2 信息流差异详解

#### BRPO (论文原始方案)

**信息源**：pseudo-frame $I_t^{fix}$ 与左右 reference 的 **mutual nearest-neighbor correspondence**

**验证逻辑**：
$$M_{left} = \text{mutual\_NN}(I_t^{fix}, \text{ref}_{left})$$
$$M_{right} = \text{mutual\_NN}(I_t^{fix}, \text{ref}_{right})$$
$$C_m = \begin{cases} 1.0 & \text{if } i \in M_{left} \cap M_{right} \\ 0.5 & \text{if } i \in M_{left} \oplus M_{right} \\ 0.0 & \text{otherwise} \end{cases}$$

**关键结构特点**：
1. pseudo content 先生成
2. confidence 由 reference 事后验证生成
3. target 与 confidence 来源分离
4. RGB/depth 共用同一个 $C_m$

---

#### old A1 (当前工程最稳)

**代码路径**：
- `rgb_mask_inference.py` → `support_left/right` + `raw_rgb_confidence`
- `depth_supervision_v2.py` → `target_depth_for_refine_v2_brpo` + `source_map`
- `joint_confidence.py` → `joint_confidence = min(rgb_conf, geometry_tier)`

**信息流**：

RGB 链：
$$\text{support}_{left/right} = \text{matcher\_found\_correspondence}$$
$$\text{rgb\_conf} = \text{continuous\_score\_from\_matcher\_confidence}$$

Depth 链：
$$\text{geometry\_tier} = \begin{cases} 1.0 & \text{if source\_map == BOTH} \\ 0.5 & \text{if source\_map == LEFT or RIGHT} \\ 0.0 & \text{otherwise} \end{cases}$$

Joint：
$$\text{joint\_confidence} = \min(\text{rgb\_conf}, \text{geometry\_tier})$$
$$\text{joint\_confidence\_cont} = \text{rgb\_conf\_cont} \times \text{geometry\_tier}$$

**关键特点**：
- **不重写 target**：`joint_depth_target_v2` 直接复用 `target_depth_for_refine_v2_brpo`
- **confidence 是取 min**：RGB 说"我信不信" × geometry 说"能不能信" → 共同 trusted support
- **confidence 是半连续**：不是纯 $\{1, 0.5, 0\}$，而是连续值乘以离散 tier

**稳定性来源**：
- RGB confidence 是连续值，允许更平滑的监督强度调制
- depth pipeline 经过长期验证，`target_depth_for_refine_v2_brpo` 有成熟的 fusion logic
- 取 min 策略保守但稳健，不会激进引入新错误

---

#### new A1 (current joint observation)

**代码路径**：`joint_observation.py`

**信息流**：

候选集：
$$\text{depth\_stack} = [\text{left\_depth}, \text{right\_depth}, \text{both\_weighted}, \text{render\_prior}]$$

证据评估：
$$\text{score\_stack} = \text{evaluate\_4\_evidence\_types}(\text{appearance}, \text{geometry}, \text{support}, \text{prior})$$

Target 生成：
$$\text{score\_prob} = \text{softmax\_like}(\text{score\_stack})$$
$$\text{fused\_depth} = \sum_{c} \text{score\_prob}_c \times \text{depth\_stack}_c$$

Confidence 生成（同源）：
$$\text{confidence\_joint} = \sqrt{\text{conf\_rgb} \times \text{conf\_depth}}$$

**核心问题**：target 和 confidence 都从同一个 `score_stack` 派生。如果 candidate ranking 错了，会出现：
$$\text{wrong\_target} + \text{inflated\_confidence} \to \text{smooth\_but\_self\_consistent\_error}$$

这就是为什么 new A1 不稳定。

---

#### brpo_style_v1 (当前尝试)

**代码路径**：`pseudo_observation_brpo_style.py`

**信息流**：

验证集定义：
$$\text{valid}_{left} = \text{support}_{left} \land \text{overlap}_{left} \land (\text{projected\_depth}_{left} > 0)$$
$$\text{valid}_{right} = \text{support}_{right} \land \text{overlap}_{right} \land (\text{projected\_depth}_{right} > 0)$$

Shared $C_m$：
$$C_m = \begin{cases} 1.0 & \text{if valid}_{left} \land \text{valid}_{right} \\ 0.5 & \text{if valid}_{left} \oplus \text{valid}_{right} \\ 0.0 & \text{otherwise} \end{cases}$$

Depth target：
$$\text{fused\_depth}[\text{verify\_both}] = \frac{w_l \cdot d_l + w_r \cdot d_r}{w_l + w_r}$$
$$\text{fused\_depth}[\text{left\_only}] = d_l$$
$$\text{fused\_depth}[\text{right\_only}] = d_r$$

**关键特点**：
- shared $C_m$：RGB/depth 共用同一个 confidence（语义正确）
- target/confidence 分离：confidence 来自 support sets，target 来自 verified projected depth
- 但 verifier backend 还不够强

---

### 1.3 为什么 brpo_style_v1 弱于 old A1？

**代码层三重原因**：

#### (1) Verifier backend 不够强

`rgb_mask_inference.py` 的 support 生成：
```
support_left = left_maps['support_mask'] > 0.5  # matcher 找到匹配的点
```

这只是"能不能找到匹配"，不是 BRPO 的"双向 mutual NN + geometric verification"。

BRPO 的 $M_{left}/M_{right}$ 要验证：
- pseudo-frame 像素能不能在 reference 里找到对应点
- reference 像素能不能反过来在 pseudo-frame 里找到对应点（双向一致性）
- 几何约束是否满足

当前 brpo_style 只有单向 matcher support mask，缺少双向验证和几何约束。

**后果**：$C_m$ 的 $\{1, 0.5, 0\}$ 判断不够精准，可能：
- 把本该可信的像素误判为 0（丢失监督）
- 把本该不可信的像素误判为 1（引入错误监督）

---

#### (2) Target builder 太简单

brpo_style 的 depth target：
```
fused_depth[verify_both] = weighted_avg(projected_depth_left, projected_depth_right)
fused_depth[left_only] = projected_depth_left
fused_depth[right_only] = projected_depth_right
```

这只是 projected depth 的简单加权组合。

old A1 的 `target_depth_for_refine_v2_brpo` 虽然逻辑类似，但它背后有一条更成熟的 pipeline：
- `projected_valid_left/right` 经过更严格的验证
- `fusion_weight` 经过更细致的计算
- 有 render_depth fallback 作为兜底
- 经过长期实验验证，稳定性更好

---

#### (3) $C_m$ 更硬，但 target 稳定性没跟上

old A1 的 confidence（半连续）：
```
joint_confidence = min(raw_rgb_confidence, geometry_tier)
# raw_rgb_confidence 是连续值，结果是半连续，更平滑
```

brpo_style 的 confidence（硬三档）：
```
confidence[verify_both] = 1.0  # 硬
confidence[verify_xor] = 0.5   # 硬
其余 = 0.0                     # 硬
```

如果 verifier 和 target 都很强，硬 confidence 可以更精准地过滤噪声。但如果 verifier 不够强 + target 不够稳，硬 confidence 反而会：
- 监督域收缩过度（丢失有效监督）
- 或监督域扩张错误（引入噪声监督）

---

### 1.4 数值对比

固定 new T1 + summary_only compare：

| 模式 | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| old A1 | 24.187772 | 0.875594 | 0.080362 |
| new A1 | 24.149355 | 0.875578 | 0.079962 |
| brpo_style | 24.175377 | 0.875338 | 0.080344 |

brpo_style 的位置：
- 明显优于 new A1（+0.026 dB）
- 逼近 old A1（-0.012 dB）
- 但还没稳定超过 old A1

---

### 1.5 Part 1 总结

brpo_style_v1 比 old A1 弱，不是因为"shared $C_m$ 方向错了"，而是：

| 问题层 | brpo_style | old A1 | 差距 |
|--------|-----------|--------|------|
| Verifier backend | 单向 matcher mask | 双向验证 + geometry tier | 弱 |
| Target builder | 简单 weighted fusion | 成熟 depth pipeline | 弱 |
| Confidence 形态 | 硬三档 $\{1, 0.5, 0\}$ | 半连续取 min | 稳 |

**下一步方向**：
- 加强 verifier backend：引入双向 mutual NN + geometric verification
- 加强 target builder：让 verified projected depth 更稳健
- 保持 fixed new T1 + summary_only compare，不把 A1/topology/B3 混着改

---

## Part 2: Gaussian Gating (B3) 分析

### 2.1 三种模式概览

| 模式 | 动作类型 | 输出 | 生效时机 | 动作域 |
|------|---------|------|---------|--------|
| Summary only | 无动作 | 只统计 | N/A | N/A |
| Boolean | 二值 mask | $m_i \in \{0, 1\}$ | delayed | far cluster |
| Opacity | 连续 scale | $\text{scale}_i \in [0.9, 1.0]$ | delayed | far cluster |
| BRPO (论文) | stochastic mask | $m_i \sim \text{Bernoulli}(1-p_i^{drop})$ | current-step | 全域 |

---

### 2.2 信息流差异详解

#### 六层链路结构

B3 不是单个函数，而是六层：

1. **stats 层**：`stats.py` — 每个 Gaussian 的 support/depth/density 统计
2. **score 层**：`score.py` — ranking/state/participation 三种分数
3. **policy 层**：`policy.py` — update 权重（非 B3 主语义）
4. **manager 层**：`manager.py` — score → participation action
5. **renderer 层**：`gaussian_renderer/__init__.py` — action → forward
6. **loop 层**：`run_pseudo_refinement_v2.py` — timing (delayed vs current)

当前瓶颈主要在 score/manager/loop 三层。

---

#### Boolean path

**score.py 定义**：
$$\text{state\_score}_i = 0.45 \cdot \text{struct\_density}_i + 0.35 \cdot \text{pop\_support}_i + 0.20 \cdot \text{depth}_i$$

**manager.py 动作**：
$$\mathcal{C}_c = \{i \in \mathcal{A} \cap c : \text{state}_i \le Q_{0.5}(\text{state}|c)\}$$
$$\mathcal{K}_c = \text{TopK}(\text{state}, \mathcal{C}_c, \text{keep}_c)$$
$$m_i = \mathbb{1}[i \in \mathcal{K}_c \cup (\mathcal{A} \setminus \mathcal{C}_c)]$$

**renderer 消费**：
```
if mask is not None:
    rendered_image = rasterizer(
        means3D=means3D[mask],
        opacities=opacity[mask],
        ...
    )
```

---

#### Opacity path

**score.py 定义**：
$$\text{participation\_score}_i = 0.80 \cdot \text{importance}_i + 0.20 \cdot \text{pop\_support}_i$$

**manager.py 动作**：
$$\text{scale}_i = \begin{cases} \text{floor}_c + (1-\text{floor}_c) \times \text{participation}_i & \text{if } i \in \text{candidate}_c \\ 1.0 & \text{otherwise} \end{cases}$$

当前配置：$\text{floor}_{far} = 0.9$

**renderer 消费**：
```
opacity = pc.get_opacity
if opacity_scale is not None:
    opacity = opacity * opacity_scale
```

---

#### BRPO (论文原始)

$$p_i^{drop} = r \cdot w_{c(i)} \cdot S_i$$
$$m_i \sim \text{Bernoulli}(1 - p_i^{drop})$$
$$\alpha_i^{eff} = \alpha_i \times m_i$$

关键区别：
- stochastic（每个 Gaussian 独立掷骰子）
- current-step 生效（不是 delayed）
- unified score（不是分离的 state/participation）

---

### 2.3 为什么效果几乎无区别？

**C1 实验结果**：

| 模式 | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| Summary only | 24.187304 | 0.875587 | 0.080364 |
| Boolean (far=0.9) | 24.186847 | 0.875591 | 0.080387 |
| Opacity (floor=0.9) | 24.186731 | 0.875586 | 0.080398 |

Delta：
- Boolean vs Summary: $-0.000457$ PSNR
- Opacity vs Summary: $-0.000573$ PSNR
- Opacity vs Boolean: $-0.000116$ PSNR

**三种模式的表现几乎无区别。**

---

### 2.4 原因分析

#### (1) 分数高度相关 → candidate 重合

**C0 诊断数据**：
- $\text{corr}(\text{state}, \text{participation}) = 0.83$（高）
- $\text{corr}(\text{state}, \text{pop\_support}) = 0.92$（极高）
- $\text{corr}(\text{participation}, \text{pop\_support}) = 0.80$（高）

两个分数都依赖 pop_support：
- $\text{state\_score}$：pop_support 占 35%
- $\text{participation\_score}$：pop_support 占 20%

**Candidate 重合**（quantile=0.5）：
- Jaccard = 0.67（重合 67%）
- $\text{both} + \text{neither}$ 合计占 active set 80%

**后果**：两套分数选出的 candidate 几乎是同一个群体，boolean 和 opacity 动作的对象高度重合。

---

#### (2) 动作幅度太轻

**Boolean**：
- $\text{keep\_ratio}_{far} = 0.9$ → candidate 内只 drop 10%
- 实测 $\text{far\_boolean\_keep\_ratio\_mean} = 0.95$（只 drop 5%）

**Opacity**：
- $\text{floor}_{far} = 0.9$ → 最小衰减到 0.9
- 实测 $\text{far\_opacity\_scale\_mean} = 0.98$（只衰减 2%）

**后果**：真正被调制的 Gaussian 只有 $\text{active} \times \text{far} \times \text{candidate\_diff} \approx 10\%$ 左右，且幅度极轻。

---

#### (3) Delayed 生效

当前实现：iter $t$ 统计/决策，iter $t+1$ 才生效。

**后果**：打不准当前 residual landscape。BRPO 是 current-step 生效，能精准作用到当前优化状态。

---

### 2.5 四群体 probe 分析

把 active set 拆成四群：

| 群体 | 定义 | 占比 | 特征 |
|------|------|------|------|
| state_only | $\text{state\_candidate} \setminus \text{part\_candidate}$ | 10.26% | state 低但 participation 高 |
| part_only | $\text{part\_candidate} \setminus \text{state\_candidate}$ | 10.26% | participation 低但 state 高 |
| both | $\text{state\_candidate} \cap \text{part\_candidate}$ | 39.74% | 两套分数都认为该处理 |
| neither | $\mathcal{A} \setminus (\text{state} \cup \text{part})$ | 39.73% | 两套分数都认为不该处理 |

**关键发现**：
- $\text{state\_only}$ 和 $\text{part\_only}$ 确实证明两套分数不是换名
- 但 $\text{both} + \text{neither}$ 合计 80%，说明真正有差异的群体太小
- 这也是为什么 opacity 路径虽然"概念更对"，但指标上还没体现

---

### 2.6 Part 2 总结

三种 B3 模式效果无区别，不是因为 gating 没生效，而是：

| 问题层 | 现状 | 影响 |
|--------|------|------|
| 分数定义 | state/participation 高度相关（0.83） | candidate 重合 67% |
| 动作幅度 | boolean drop 5%，opacity 衰减 2% | 调制太轻 |
| 生效时机 | delayed (t→t+1) | 打不准当前 residual |
| 动作域 | active × far × candidate_diff ≈ 10% | 真正被影响的太少 |

**类比**：一个班级 100 人，boolean 惩罚 5 人（不让参与），opacity 惩罚 10 人（每人扣 2 分），其余 85 人正常。全班平均分几乎没区别。

**下一步建议**：
- 不急着进 O2a/b
- 先在 C0 层把分数拉开（改 candidate quantile 或 floor 映射）
- 让 boolean vs opacity 真正作用到不同群体

---

## 附录：核心代码位置

### Mask/Observation 相关

| 文件 | 功能 |
|------|------|
| `pseudo_branch/brpo_v2_signal/rgb_mask_inference.py` | RGB support/confidence 生成 |
| `pseudo_branch/brpo_v2_signal/depth_supervision_v2.py` | Depth target/source_map 生成 |
| `pseudo_branch/brpo_v2_signal/joint_confidence.py` | old A1 joint confidence 合成 |
| `pseudo_branch/brpo_v2_signal/joint_observation.py` | new A1 candidate competition |
| `pseudo_branch/brpo_v2_signal/pseudo_observation_brpo_style.py` | brpo_style_v1 shared $C_m$ |

### B3 Gaussian Gating 相关

| 文件 | 功能 |
|------|------|
| `pseudo_branch/spgm/stats.py` | Gaussian 统计 |
| `pseudo_branch/spgm/score.py` | ranking/state/participation 分数 |
| `pseudo_branch/spgm/manager.py` | mask/scale 生成 |
| `third_party/S3PO-GS/gaussian_splatting/gaussian_renderer/__init__.py` | renderer 消费 mask/opacity_scale |
| `scripts/run_pseudo_refinement_v2.py` | timing (delayed 生效) |

---

> 文档由 Noah 生成，基于 REFINE_DESIGN.md 与 MASK_DESIGN.md 的代码流追踪分析。
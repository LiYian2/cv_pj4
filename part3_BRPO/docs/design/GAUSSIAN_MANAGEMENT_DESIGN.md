# GAUSSIAN_MANAGEMENT_DESIGN.md - G~ Gaussian Management 设计文档

> 更新时间：2026-04-20 19:00 (Asia/Shanghai)

> **书写规范**：
> 1. 只讲 G~（Gaussian Management）：信息从哪里来、怎么转换成 gating action、怎么被下游消费
> 2. 遵循"信息源 → 信号转换 → 下游消费"链式分析
> 3. 需要比较时，统一比较 BRPO / boolean G~ / opacity G~ / summary control
> 4. 数学公式用 `$...$` 或 `$$...$$` 包裹
> 5. 更新后修改文档顶部时间戳

---

## 1. 先说结论

G~（Gaussian Management）决定"哪些 Gaussian 参与 pseudo render、以什么强度参与"。当前三种模式（boolean / opacity / summary）效果几乎无区别，原因是：
- 三条分数（ranking/state/participation）高度相关（corr≈0.83）
- 动作幅度太轻（boolean drop 5%，opacity 衰减 2%）
- delayed 生效（iter t 决策，iter t+1 消费）
- 动作域太窄（active × far × candidate_diff ≈ 10%）

下一步不是推进 stochastic，而是先在 C0 层把分数拉开。

---

## 2. G~ 的信息源

G~ 的上游信息来自每个 Gaussian 的 scene statistics：

### 2.1 Stats 层（`stats.py`）

**输入**：
- 渲染过程中每个 Gaussian 的 visibility record
- 当前 train window 的帧范围

**产物**：
- $s_i$：`support_count` — accepted pseudo support 次数
- $\tilde{s}_i$：`population_support_count` — current train window support 次数
- $z_i$：`depth_value` — 平均深度
- $\rho_i$：`density_proxy` — 密度代理
- $\mathcal{A}$：`active_mask = [s_i > 0]`
- $\mathcal{P}$：`population_active_mask = [\tilde{s}_i > 0]`

当前主控制域仍是 $\mathcal{A}$（pseudo-side active），不是 $\mathcal{P}$（population active）。

---

## 3. Score 层（`score.py`）

### 3.1 三条分数定义

**Importance raw**：
$$
\text{importance}_i = \alpha \cdot \text{depth}_i + (1-\alpha) \cdot \text{density}_i
$$

**Ranking score**（update 权重语义）：
$$
\text{ranking}_i \approx \text{importance}_i \cdot \text{support}_i^\eta
$$

**State score**（状态诊断语义）：
$$
\text{state}_i = 0.45 \cdot \text{structDensity}_i + 0.35 \cdot \text{popSupport}_i + 0.20 \cdot \text{depth}_i
$$

**Participation score**（参与控制语义）：
$$
\text{part}_i = 0.80 \cdot \text{importance}_i + 0.20 \cdot \text{popSupport}_i
$$

### 3.2 分数相关性（C0 数据）

- $\text{corr}(\text{state}, \text{participation}) = 0.83$（高）
- $\text{corr}(\text{state}, \text{pop\_support}) = 0.92$（极高）
- $\text{corr}(\text{participation}, \text{pop\_support}) = 0.80$（高）

两条分数都依赖 pop_support，导致 candidate 重合度高（Jaccard=0.67）。

---

## 4. Manager 层（`manager.py`）— 各版本 G~

### 4.1 Summary control

**动作**：无动作，只统计。

**下游消费**：pseudo render 不受 G~ 额外控制。

---

### 4.2 Boolean G~

**信息源**：state_score + depth cluster partition

**信号转换**：
1. Depth partition：按深度排序分 near/mid/far 三 cluster
2. Candidate 定义：
$$
\mathcal{C}_c = \{i \in \mathcal{A} \cap c : \text{state}_i \le Q_{0.5}(\text{state}|c)\}
$$
3. Keep-top-k：
$$
\mathcal{K}_c = \text{TopK}(\text{state}, \mathcal{C}_c, \text{keep}_c)
$$
4. Boolean mask：
$$
m_i = \mathbb{1}[i \in \mathcal{K}_c \cup (\mathcal{A} \setminus \mathcal{C}_c)]
$$

**当前配置**：$\text{keep}_{far} = 0.9$

**实测效果**：far 只 drop 5%（$\text{keep\_ratio} \approx 0.95$）

---

### 4.3 Opacity G~

**信息源**：participation_score + depth cluster partition

**信号转换**：
1. Candidate 定义：
$$
\mathcal{C}_c = \{i \in \mathcal{A} \cap c : \text{part}_i \le Q_{0.5}(\text{part}|c)\}
$$
2. Opacity scale：
$$
\text{scale}_i = \begin{cases}
\text{floor}_c + (1-\text{floor}_c) \cdot \text{part}_i & \text{if } i \in \mathcal{C}_c \\
1.0 & \text{otherwise}
\end{cases}
$$

**当前配置**：$\text{floor}_{far} = 0.9$

**实测效果**：far 平均 scale ≈ 0.98（只衰减 2%）

---

### 4.4 BRPO G~（论文原始）

**信息源**：unified score $S_i$

**信号转换**：
$$
p_i^{drop} = r \cdot w_{c(i)} \cdot S_i
$$
$$
m_i \sim \text{Bernoulli}(1 - p_i^{drop})
$$
$$
\alpha_i^{eff} = \alpha_i \times m_i
$$

**关键差异**：
- Stochastic（每个 Gaussian 独立掷骰子）
- Current-step 生效（不是 delayed）
- Per-Gaussian probability，不是 cluster-level quantile

---

## 5. Renderer 消费（`gaussian_renderer/__init__.py`）

### 5.1 Boolean mask 路径

```python
if mask is not None:
    rendered_image = rasterizer(
        means3D=means3D[mask],
        opacities=opacity[mask],
        ...
    )
```

直接过滤 Gaussian，不参与 rasterizer。

### 5.2 Opacity scale 路径

```python
opacity = pc.get_opacity
if opacity_scale is not None:
    opacity = opacity * opacity_scale
```

Opacity 衰减后全部 Gaussian 参与 rasterizer，但低 score Gaussian 的渲染贡献降低。

---

## 6. Loop timing（`run_pseudo_refinement_v2.py`）

当前实现：
- Iter $t$：collect stats → build scores → generate action（mask/scale）
- Iter $t+1$：pseudo render 消费 action

这是 **delayed** 模式，与 BRPO 的 current-step 不同。

---

## 7. 为什么三种模式无区别

### 7.1 C1 数值证据

| 模式 | PSNR | Delta vs summary |
|------|------|------------------|
| Summary | 24.187304 | 0 |
| Boolean | 24.186847 | -0.000457 |
| Opacity | 24.186731 | -0.000573 |

三种模式差异 < 0.001 PSNR。

### 7.2 四重原因

**1. 分数高度相关**：
- state vs participation corr = 0.83
- 两套 candidate 重合 67%
- both + neither 合计占 active set 80%

**2. 动作幅度太轻**：
- Boolean：只 drop 5%
- Opacity：只衰减 2%

**3. Delayed 生效**：
- Iter t 决策，iter t+1 消费
- 打不准当前 residual landscape

**4. 动作域太窄**：
- 只有 far cluster 有动作
- $\text{active} \times \text{far} \times \text{candidate\_diff} \approx 10\%$

---

## 8. 四群体 Probe 分析

把 active set 拆成四群：

| 群体 | 占比 | 特征 |
|------|------|------|
| state_only | 10.26% | state 低，participation 高 |
| part_only | 10.26% | participation 低，state 高 |
| both | 39.74% | 两分数都认为该处理 |
| neither | 39.73% | 两分数都认为不该处理 |

关键发现：
- state_only / part_only 确实证明两分数不同
- 但 both + neither 合计 80%，差异群体太小

---

## 9. 下一步建议

### 9.1 不该做的
- 不直接推进 O2a/b（stochastic）
- 不回退到 xyz grad scale
- 不在 delayed opacity 仍弱负时继续加新变量

### 9.2 该做的
- C0-1：固定记录四群体统计
- C0-2：小步改 candidate quantile 或 floor 映射
- 只重跑新 opacity arm，复用旧 controls
- 只有当 delayed opacity 不再弱负时，才进入 O2a/b

---

## 10. 代码位置

| 文件 | 功能 |
|------|------|
| `pseudo_branch/spgm/stats.py` | Gaussian statistics |
| `pseudo_branch/spgm/score.py` | 三条分数定义 |
| `pseudo_branch/spgm/manager.py` | Boolean/Opacity action generation |
| `third_party/S3PO-GS/gaussian_renderer/__init__.py` | Renderer mask/opacity_scale 消费 |
| `scripts/run_pseudo_refinement_v2.py` | Delayed timing loop |

---

## 11. 六层链路总结

G~ 不是单个函数，而是六层：

1. **Stats 层**：每个 Gaussian 的 support/depth/density
2. **Score 层**：ranking/state/participation 三条分数
3. **Policy 层**：update 权重（非 G~ 主语义）
4. **Manager 层**：score → boolean mask 或 opacity scale
5. **Renderer 层**：mask/opacity_scale 进入 forward
6. **Loop 层**：delayed vs current-step timing

当前瓶颈在 2/4/6：score 没拉开、action 太保守、timing delayed。

---

> 文档口径：G~ = Gaussian Management（per-Gaussian gating）。原 B3。
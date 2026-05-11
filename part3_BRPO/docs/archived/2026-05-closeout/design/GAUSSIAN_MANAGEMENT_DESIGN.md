# GAUSSIAN_MANAGEMENT_DESIGN.md - G~ Gaussian Management 设计文档

> 更新时间：2026-04-22 04:22 (Asia/Shanghai)

> **书写规范**：
> 1. 只讲 G~（Gaussian Management）：信息从哪里来、怎么转换成 gating action、怎么被下游消费
> 2. 遵循"信息源 → 信号转换 → 下游消费"链式分析
> 3. 数学公式用 `$...$` 或 `$$...$$` 包裹
> 4. 更新后修改文档顶部时间戳

---

## 1. 概览

G~（Gaussian Management）决定"哪些 Gaussian 参与 pseudo render、以什么强度参与"。当前系统支持 **5 种 manager_mode + 2 种 score_semantics**。

**关键结论**：
- G~ clean compare 已完成：direct BRPO current-step 仅 +0.00538 PSNR
- 三种 deterministic mode（boolean/opacity/summary）效果差异 < 0.001 PSNR
- G~ 已冻结为 side branch，不再是主线突破口

---

## 2. 所有 G~ 变种一览

### 2.1 Manager modes

| Mode | 动作类型 | 分数来源 | Timing | BRPO 对齐度 |
|------|---------|---------|--------|------------|
| `summary_only` | 无动作（只统计） | - | - | control baseline |
| `xyz_lr_scale` | XYZ grad scale | state_score | delayed | legacy |
| `deterministic_participation` | Boolean mask | state_score | delayed | 低 |
| `deterministic_opacity_participation` | Opacity scale | participation_score | delayed | 低 |
| `stochastic_bernoulli_opacity` | Bernoulli sampling | unified_score | **current-step** | ✅ 高（direct BRPO） |

### 2.2 Score semantics

| Score semantics | 分数定义 | 用途 |
|----------------|---------|------|
| `legacy_v1` | ranking/state/participation 三条分数 | deterministic modes |
| `brpo_unified_v1` | unified score $S_i$ | stochastic_bernoulli_opacity |

---

## 3. 信息源层（Stats 层）

G~ 的上游信息来自每个 Gaussian 的 **scene statistics**：

### 3.1 Stats 定义

**代码位置**：`stats.py`

**核心 stats**：

$$
\begin{aligned}
s_i &= 	ext{support\_count} — 	ext{accepted pseudo support 次数} \
\tilde{s}_i &= 	ext{population\_support\_count} — 	ext{current train window support 次数} \
z_i &= 	ext{depth\_value} — 	ext{平均深度} \

ho_i &= 	ext{density\_proxy} — 	ext{密度代理} \
\mathcal{A} &= 	ext{active\_mask} = [s_i > 0] \
\mathcal{P} &= 	ext{population\_active\_mask} = [	ilde{s}_i > 0]
\end{aligned}
$$

**当前控制域分层**：
- legacy deterministic / summary 路径主要仍围绕 $\mathcal{A}$（pseudo-side active）做统计与诊断；
- direct BRPO clean compare 的 `control_universe` 已切到 $\mathcal{P}$（population active），这也是当前与 BRPO 论文对齐的主语义。

**BRPO 论文控制域**：$\mathcal{P}$（population active）。

---

## 4. 信号转换层（Score 层）

### 4.1 Legacy 三条分数

**代码位置**：`score.py` → `build_spgm_importance_score()`

**Ranking score**（update 权重语义）：
$$
	ext{ranking}_i pprox 	ext{importance}_i \cdot 	ext{support}_i^\eta
$$

**State score**（状态诊断语义）：
$$
	ext{state}_i = 0.45 \cdot 	ext{structDensity}_i + 0.35 \cdot 	ext{popSupport}_i + 0.20 \cdot 	ext{depth}_i
$$

**Participation score**（参与控制语义）：
$$
	ext{part}_i = 0.80 \cdot 	ext{importance}_i + 0.20 \cdot 	ext{popSupport}_i
$$

---

### 4.2 BRPO unified score

**Phase G-BRPO-0 新增**

**代码位置**：`score.py` → `build_spgm_brpo_unified_score()`

**Unified score 定义**：

$$
S_i = lpha \cdot \hat{s}_i^{(z)} + (1-lpha) \cdot \hat{s}_i^{(
ho)}
$$

其中：
- $\hat{s}_i^{(z)}$：normalized depth score（closer → higher score）
- $\hat{s}_i^{(
ho)}$：normalized density score
- $lpha$：depth vs density weight（默认 0.5）

**深度 score normalization**：
$$
\hat{s}_i^{(z)} = 1 - rac{z_i - z_{min}}{z_{max} - z_{min}}
$$

**Drop probability（BRPO 公式）**：
$$
p_i^{drop} = r \cdot w_{cluster(i)} \cdot S_i
$$

其中：
- $r$：global drop rate
- $w_{cluster(i)}$：per-cluster weight multiplier

---

### 4.3 分数相关性分析（C0 数据）

**关键发现**：
- $	ext{corr}(	ext{state}, 	ext{participation}) = 0.83$（高）
- $	ext{corr}(	ext{state}, 	ext{pop\_support}) = 0.92$（极高）
- 两套 candidate 重合度高（Jaccard = 0.67）

**结论**：三条分数高度相关，导致不同模式效果接近。

---

## 5. 动作生成层（Manager 层）

### 5.1 Summary only（control baseline）

**动作**：无动作，只统计。

**输出**：
- `spgm_state_participation_ratio = 1.0`
- `grad_weight_mean_xyz = 1.0`

---

### 5.2 Deterministic boolean mask

**代码位置**：`manager.py` → `_build_participation_render_mask()`

**Candidate 定义**：
$$
\mathcal{C}_c = \{i \in \mathcal{A} \cap c : 	ext{state}_i \le Q_{0.5}(	ext{state}|c)\}
$$

**Keep-top-k**：
$$
\mathcal{K}_c = 	ext{TopK}(	ext{state}, \mathcal{C}_c, 	ext{keep}_c)
$$

**Boolean mask**：
$$
m_i = \mathbb{1}[i \in \mathcal{K}_c \cup (\mathcal{A} \setminus \mathcal{C}_c)]
$$

**当前配置**：$	ext{keep}_{far} = 0.75$

**实测效果**：far 只 drop ~5%

---

### 5.3 Deterministic opacity scale

**代码位置**：`manager.py` → `_build_participation_opacity_scale()`

**Opacity scale**：
$$
	ext{scale}_i = egin{cases}
	ext{floor}_c + (1-	ext{floor}_c) \cdot 	ext{part}_i & 	ext{if } i \in \mathcal{C}_c \
1.0 & 	ext{otherwise}
\end{cases}
$$

**当前配置**：$	ext{floor}_{far} = 0.92$

**实测效果**：far 平均 scale ≈ 0.98（只衰减 2%）

---

### 5.4 Stochastic Bernoulli opacity（direct BRPO）

**Phase G-BRPO-0 新增**

**代码位置**：`manager.py` → `_build_stochastic_bernoulli_opacity_action()`

**BRPO 公式**：

$$
egin{aligned}
p_i^{drop} &= r \cdot w_{cluster(i)} \cdot S_i \
m_i &= 	ext{Bernoulli}(1 - p_i^{drop}) \
lpha_i^{eff} &= lpha_i 	imes m_i
\end{aligned}
$$

**关键特点**：
- **Stochastic**：每个 Gaussian 独立掷骰子
- **Current-step**：统计 → mask → render → backward 同一轮
- **Per-Gaussian probability**：不是 cluster-level quantile

---

## 6. 下游消费层（Renderer）

### 6.1 Boolean mask 路径

```python
if mask is not None:
    rendered_image = rasterizer(
        means3D=means3D[mask],
        opacities=opacity[mask],
        ...
    )
```

直接过滤 Gaussian，不参与 rasterizer。

---

### 6.2 Opacity scale 路径

```python
opacity = pc.get_opacity
if opacity_scale is not None:
    opacity = opacity * opacity_scale
```

Opacity 衰减后全部 Gaussian 参与 rasterizer，但低 score Gaussian 的渲染贡献降低。

---

## 7. Timing 层（Loop 层）

### 7.1 Delayed timing（current deterministic modes）

```
iter t:
  pseudo render (G~ action from iter t-1)
  real render
  assemble joint loss
  backward once
  collect G~ stats → generate G~ action for iter t+1
  optimizer.step

iter t+1:
  pseudo render (G~ action from iter t)
  ...
```

G~ 是 delayed：iter t 统计 → iter t+1 消费。

---

### 7.2 Current-step timing（stochastic_bernoulli_opacity）

```
iter t:
  collect G~ stats from iter t-1
  build unified score
  generate stochastic mask (current-step)
  pseudo render (current-step G~ action)
  pseudo backward
  real render
  real backward
  optimizer.step
```

G~ 是 current-step：统计 → mask → render → backward 同一轮。

---

## 8. Clean Compare 结果（G~-C3）

**实验**：`20260421_g_brpo_clean_compare_v1`

**结果**：

| Mode | PSNR | Delta vs summary |
|------|------|------------------|
| `summary_only` | 24.021169 | 0（baseline） |
| `legacy_delayed_opacity` | 24.004704 | -0.016466（负向） |
| `direct_brpo_current_step` | 24.026548 | +0.005379（小幅正向） |

**结论**：
- 三种 deterministic mode 效果接近（差异 < 0.001 PSNR）
- Direct BRPO current-step 仅小幅正向
- G~ 不是主瓶颈

---

## 9. 为什么三种 deterministic mode 无区别

### 9.1 四重原因

**1. 分数高度相关**：
- state vs participation corr = 0.83
- candidate 重合 67%

**2. 动作幅度太轻**：
- Boolean：只 drop 5%
- Opacity：只衰减 2%

**3. Delayed 生效**：
- Iter t 决策，iter t+1 消费
- 打不准当前 residual landscape

**4. 动作域太窄**：
- 只有 far cluster 有动作
- $	ext{active} 	imes 	ext{far} 	imes 	ext{candidate\_diff} pprox 10\%$

---

## 10. G~ 与其他模块的关系

```
G~ 输入:
  - Stats: support_count, depth_value, density_proxy
  - Score: unified_score (BRPO) or state/participation (legacy)

G~ 输出:
  - render_mask (boolean) or opacity_scale (continuous)
  - gating summary stats

下游消费:
  - Renderer: mask/opacity_scale 进入 forward
  - Loss: 不直接消费 G~
  - M~/T~/R~: 不直接消费 G~
```

---

## 11. 代码位置索引

| 文件 | 功能 | 关键函数 |
|------|------|---------|
| `pseudo_branch/gaussian_management/spgm/stats.py` | Stats 提取 | `accumulate_support`, `_weighted_median_vectorized` |
| `pseudo_branch/gaussian_management/spgm/score.py` | Score 计算 | `build_spgm_importance_score`, `build_spgm_brpo_unified_score` |
| `pseudo_branch/gaussian_management/spgm/manager.py` | Manager 动作生成 | `apply_spgm_state_management`, `_build_stochastic_bernoulli_opacity_action` |
| `third_party/S3PO-GS/gaussian_renderer/__init__.py` | Renderer 消费 | mask/opacity_scale path |
| `scripts/run_pseudo_refinement_v2.py` | CLI 参数 | `--pseudo_local_gating_spgm_manager_mode`, `--pseudo_local_gating_params` |

---

## 12. 当前状态与下一步

### 12.1 当前状态

G~ 已冻结为 side branch：
- Clean compare 完成
- Direct BRPO semantics 对齐
- 收益有限（+0.00538 PSNR）

### 12.2 不做的事

- 不继续打磨 deterministic mode 参数
- 不再把 delayed opacity / stochastic current-step 当新的主推进路线
- 不回退到 xyz_lr_scale

### 12.3 下一步

主线转向 T~ upstream backend，不在 G~ 上继续打转。

---

> 文档口径：G~ = Gaussian Management（per-Gaussian gating）。六层链路：Stats → Score → Policy → Manager → Renderer → Loop。

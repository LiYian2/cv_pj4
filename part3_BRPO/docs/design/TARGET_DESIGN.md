# TARGET_DESIGN.md - T~ Target 设计文档

> 更新时间：2026-04-20 19:00 (Asia/Shanghai)

> **书写规范**：
> 1. 只讲 T~（target）：信息从哪里来、怎么转换成 target 数值、怎么被下游消费
> 2. 遵循"信息源 → 信号转换 → 下游消费"链式分析
> 3. 需要比较时，统一比较 BRPO / old T~ / new T~ / hybrid T~ / stable T~ / exact T~
> 4. 数学公式用 `$...$` 或 `$$...$$` 包裹
> 5. 更新后修改文档顶部时间戳

---

## 1. 先说结论

T~（target）与 M~（mask）最初实现是耦合的：两者都从同一套 correspondence / projected depth 信息派生。但语义上它们是独立组件：
- **M~**：决定"哪些像素被监督、监督强度如何"
- **T~**：决定"监督目标的数值是什么"

当前最佳组合是 `exact M~ + old T~`（≈ old A1），因为 old T~ 的 depth pipeline 更成熟稳定。pure exact T~ 在当前 proxy backend 下仍弱于 old T~ 约 -0.013 PSNR。

---

## 2. T~ 的信息源

T~ 的上游信息来自 depth 侧：

**主要输入**：
- `projected_depth_left`：左侧 reference 投影到 pseudo-frame 的 depth
- `projected_depth_right`：右侧 reference 投影到 pseudo-frame 的 depth
- `fusion_weight_left`：左侧 fusion 权重
- `fusion_weight_right`：右侧 fusion 权重
- `overlap_mask_left`：左侧 overlap 有效域
- `overlap_mask_right`：右侧 overlap 有效域
- `render_depth`：当前 render depth（fallback 用）

**辅助输入**（来自 M~ 的 verification 结果）：
- `verify_both`：双侧都验证通过的区域
- `verify_left_only`：只左侧验证通过
- `verify_right_only`：只右侧验证通过
- `verify_xor`：单侧验证区域

---

## 3. 各版本 T~ 的信号转换

### 3.1 BRPO T~（论文原始）

BRPO 论文中的 target 来自 pseudo-frame `I_t^{fix}$ 对应的 verified depth field。语义上：
- 在 `M_left ∩ M_right` 区域：双侧验证通过的 projected depth composition
- 在 `M_left ⊕ M_right` 区域：单侧验证通过的 projected depth
- 在 outside union 区域：不监督

数学形态：
$$
d_{target}[i] = \begin{cases}
\text{composition}(d_{left}[i], d_{right}[i]) & \text{if } i \in M_{left} \cap M_{right} \\
d_{left}[i] \text{ or } d_{right}[i] & \text{if } i \in M_{left} \oplus M_{right} \\
\text{no supervision} & \text{otherwise}
\end{cases}
$$

关键点：BRPO 的 target 与 confidence 来源分离，target 由 independent verifier 决定。

---

### 3.2 Old T~（当前最稳）

**代码路径**：`depth_supervision_v2.py`

**信息源**：projected depth + fusion weight + rgb confidence gate

**信号转换**：
1. 验证域定义：
$$
\text{valid}_{left} = (\text{projected\_depth}_{left} > 0) \land (\text{rgb\_confidence} \ge 0.5)
$$
$$
\text{valid}_{right} = (\text{projected\_depth}_{right} > 0) \land (\text{rgb\_confidence} \ge 0.5)
$$

2. Target 区域划分：
$$
\text{both} = \text{valid}_{left} \land \text{valid}_{right}
$$
$$
\text{left\_only} = \text{valid}_{left} \land \neg \text{valid}_{right}
$$
$$
\text{right\_only} = \text{valid}_{right} \land \neg \text{valid}_{left}
$$
$$
\text{fallback} = \neg \text{valid}_{left} \land \neg \text{valid}_{right} \land (\text{rgb\_confidence} \ge 0.5)
$$

3. Target 数值生成：
$$
d_{target}[\text{both}] = \frac{w_l \cdot d_l + w_r \cdot d_r}{w_l + w_r}
$$
$$
d_{target}[\text{left\_only}] = d_l
$$
$$
d_{target}[\text{right\_only}] = d_r
$$
$$
d_{target}[\text{fallback}] = \text{render\_depth} \text{ 或 } 0
$$

**输出**：`target_depth_for_refine_v2_brpo` + `target_depth_source_map_v2_brpo`

**稳定性来源**：
- RGB confidence gate 提供前置过滤
- fusion weight 经过长期验证
- render_depth fallback 作为兜底
- 整条 pipeline 经过大量实验打磨

---

### 3.3 New T~（candidate competition）

**代码路径**：`joint_observation.py`

**信息源**：4-candidate depth stack + 4-evidence score stack

**候选集**：
$$
\text{depth\_stack} = [d_{left}, d_{right}, d_{both\_weighted}, d_{render}]
$$

**证据评估**：
$$
\text{score\_stack} = f(\text{appearance}, \text{geometry}, \text{support}, \text{prior})
$$

**Target 生成**：
$$
\text{score\_prob} = \text{softmax}(\text{score\_stack})
$$
$$
d_{target} = \sum_{c} \text{score\_prob}_c \cdot d_c
$$

**关键问题**：target 与 confidence（M~）同源。如果 score ranking 错误：
$$
\text{wrong\_target} + \text{inflated\_confidence} \to \text{smooth\_but\_self\_consistent\_error}
$$

这就是为什么 new T~ 不稳定。

---

### 3.4 Hybrid T~（brpo_style）

**代码路径**：`pseudo_observation_brpo_style.py`

**信息源**：verified projected depth + M~ verification result

**信号转换**：
$$
d_{target}[\text{verify\_both}] = \frac{w_l \cdot d_l + w_r \cdot d_r}{w_l + w_r}
$$
$$
d_{target}[\text{verify\_left\_only}] = d_l
$$
$$
d_{target}[\text{verify\_right\_only}] = d_r
$$
$$
d_{target}[\text{neither}] = 0
$$

**特点**：
- target 与 M~ 已分离（不再同源）
- 但 target builder 仍是第一版工程近似
- 简单的 weighted composition，没有 old T~ 的成熟 fallback 机制

---

### 3.5 Stable T~

**信息源**：hybrid T~ + old T~ + render_depth blend

**信号转换**：
$$
d_{target} = \alpha \cdot d_{hybrid} + (1-\alpha) \cdot d_{stable}
$$

其中 $d_{stable}$ 来自 old T~ 或 render_depth fallback。

**目的**：用 stable target 做兜底，减少 hybrid T~ 的不稳定。

**效果**：比 hybrid T~ 略好，但仍弱于 old T~ 约 -0.012 PSNR。

---

### 3.6 Exact T~（pure BRPO semantics）

**代码路径**：`pseudo_observation_brpo_style.py` → `exact_brpo_full_target_v1`

**信息源**：strict M~ verification result + projected depth

**信号转换**：
- 严格按 BRPO 论文语义：在 exact $C_m$ 覆盖域内用 verified projected depth
- 强制 RGB/depth 共用同一个 $C_m$（`stageA_depth_loss_mode=legacy`）
- 无 source-aware fallback tiers

**数学形态**：
$$
d_{target}[i] = \begin{cases}
\text{composition}(d_l, d_r) & \text{if } C_m[i] = 1.0 \\
d_l \text{ or } d_r & \text{if } C_m[i] = 0.5 \\
\text{no supervision} & \text{if } C_m[i] = 0.0
\end{cases}
$$

**效果**：PSNR 24.174488，仍弱于 old T~ 约 -0.01325 PSNR，与 hybrid T~ 几乎等价。

---

## 4. 下游消费方式

T~ 的输出被 depth loss 消费：

### 4.1 Standard depth loss

$$
L_{depth} = \text{mask} \cdot |d_{render} - d_{target}|
$$

其中 mask 来自 M~。

### 4.2 Shared-$C_m$ depth loss（exact BRPO）

$$
L_{depth} = C_m \cdot |d_{render} - d_{target}|
$$

RGB loss 也用同一个 $C_m$：
$$
L_{rgb} = C_m \cdot |I_{render} - I_{target}|
$$

---

## 5. 各版本与 BRPO 的差异

| 版本 | Target 来源 | 与 M~ 关系 | 与 BRPO 差距 |
|------|------------|-----------|-------------|
| BRPO | verified projected depth composition | 分离 | 0（定义） |
| Old T~ | projected depth + rgb gate + fallback | 分离（取 min 策略） | 工程成熟但语义不完全对齐 |
| New T~ | score_prob weighted fusion | 同源（candidate competition） | 语义偏离大 |
| Hybrid T~ | verified projected depth composition | 分离 | builder 工程不够强 |
| Stable T~ | hybrid + stable blend | 分离 | fallback 有帮助但不够 |
| Exact T~ | strict verified composition | 分离 + shared $C_m$ | 语义对齐但 backend 不够强 |

---

## 6. 为什么 Exact T~ 没赢

**数值证据**：exact T~ 与 hybrid T~ 的 target array 差异只有浮点噪声（max abs diff ~1.4e-06），source map 相同。

**原因分析**：
- 当前 proxy `I_t^{fix}` / projected-depth backend 本身不够丰富
- exact BRPO target semantics 在当前 backend 下与 hybrid T~ 几乎等价
- 真正的瓶颈在上游（Layer B proxy），而不是 target contract 本身

---

## 7. 当前最佳组合与下一步

### 7.1 最佳组合
- `exact M~ + old T~` ≈ old A1（24.1877 PSNR）
- old T~ 的 depth pipeline 更成熟，提供稳定兜底

### 7.2 下一步方向
若坚持 exact BRPO：
- 上移到 Layer B proxy：改进 `I_t^{fix}$` 质量、verifier backend、projected-depth supervision field
- 不继续只在 stable/fallback 权重上打转

若走 replay-first：
- 承认 hybrid/stable T~ 是工程分支
- 在该标签下继续优化，不再表述为 pure BRPO

---

## 8. 代码位置

| 文件 | 功能 |
|------|------|
| `pseudo_branch/brpo_v2_signal/depth_supervision_v2.py` | Old T~ 生成 |
| `pseudo_branch/brpo_v2_signal/joint_observation.py` | New T~（candidate competition） |
| `pseudo_branch/brpo_v2_signal/pseudo_observation_brpo_style.py` | Hybrid T~ / Exact T~ |
| `scripts/run_pseudo_refinement_v2.py` | T~ 消费（depth loss） |

---

> 文档口径：T~ = Target 模块。与 M~（Mask）最初耦合实现，但语义独立。
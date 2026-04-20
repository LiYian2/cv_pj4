# MASK_DESIGN.md - M~ Mask 设计文档

> 更新时间：2026-04-20 19:00 (Asia/Shanghai)

> **书写规范**：
> 1. 只讲 M~（mask/confidence）：信息从哪里来、怎么转换成 confidence、怎么被下游消费
> 2. 遵循"信息源 → 信号转换 → 下游消费"链式分析
> 3. 需要比较时，统一比较 BRPO / old M~ / new M~ / hybrid M~ / exact M~
> 4. 数学公式用 `$...$` 或 `$$...$$` 包裹
> 5. 更新后修改文档顶部时间戳

---

## 1. 先说结论

M~（mask/confidence）决定"哪些像素被监督、监督强度如何"。T~（target）决定"监督目标的数值"，两者最初实现耦合但语义独立。

当前最佳组合是 `exact M~ + old T~`（≈ old A1），因为 exact M~ 已基本对齐 BRPO semantics，而 old T~ 的 depth pipeline 更成熟。pure exact M~ + exact T~ 在当前 proxy backend 下仍弱于 old A1 约 -0.013 PSNR。

---

## 2. M~ 的信息源

M~ 的上游信息来自 pseudo-frame 与 reference 的 correspondence verification：

**主要输入**：
- `target_rgb_fused.png`：pseudo-frame proxy（工程代理）
- `rgb_support_left`：左侧 matcher correspondence support
- `rgb_support_right`：右侧 matcher correspondence support
- `projected_depth_left`：左侧 projected depth
- `projected_depth_right`：右侧 projected depth
- `overlap_mask_left`：左侧 overlap 有效域
- `overlap_mask_right`：右侧 overlap 有效域

---

## 3. 各版本 M~ 的信号转换

### 3.1 BRPO M~（论文原始）

**信息源**：pseudo-frame $I_t^{fix}$ 与左右 reference 的 **mutual nearest-neighbor correspondence**

**定义**：
- $M_{left}$：pseudo-frame 与左侧 reference 的双向匹配点集
- $M_{right}$：pseudo-frame 与右侧 reference 的双向匹配点集

**Confidence $C_m$**：
$$
C_m[i] = \begin{cases}
1.0 & \text{if } i \in M_{left} \cap M_{right} \\
0.5 & \text{if } i \in M_{left} \oplus M_{right} \\
0.0 & \text{otherwise}
\end{cases}
$$

**关键特点**：
- Pseudo content 先生成
- Confidence 由 reference 事后验证生成
- Target 与 confidence 来源分离
- RGB/depth 共用同一个 $C_m$

---

### 3.2 Old M~（当前最稳）

**代码路径**：
- `rgb_mask_inference.py` → RGB support/confidence
- `depth_supervision_v2.py` → depth source map
- `joint_confidence.py` → joint confidence

**RGB 链信息流**：
1. Matcher 找 correspondence：
$$
\text{support}_{left/right} = \text{matcher\_found\_correspondence}
$$
2. 生成 continuous confidence：
$$
\text{rgb\_conf} = \text{continuous\_score\_from\_matcher\_confidence}
$$

**Depth 链信息流**：
1. Projected depth fusion 得到 source_map
2. 压出 geometry tier：
$$
\text{geometry\_tier} = \begin{cases}
1.0 & \text{if source\_map == BOTH} \\
0.5 & \text{if source\_map == LEFT or RIGHT} \\
0.0 & \text{otherwise}
\end{cases}
$$

**Joint confidence**：
$$
\text{joint\_confidence} = \min(\text{rgb\_conf}, \text{geometry\_tier})
$$
$$
\text{joint\_confidence\_cont} = \text{rgb\_conf\_cont} \times \text{geometry\_tier}
$$

**特点**：
- 半连续值（不是硬三档）
- RGB/depth 分别过滤，取共同 trusted support
- 不重写 target，只加 confidence filter

---

### 3.3 New M~（candidate competition）

**代码路径**：`joint_observation.py`

**信息源**：4-candidate depth stack + 4-evidence score stack

**Confidence 生成**：
$$
\text{confidence}_{joint} = \sqrt{\text{conf}_{rgb} \times \text{conf}_{depth}}
$$

其中 conf_rgb / conf_depth 都从同一个 score_stack 派生。

**关键问题**：confidence 与 T~（target）同源。如果 score ranking 错误，"wrong target + inflated confidence" 同时出现。

---

### 3.4 Hybrid M~（brpo_style）

**代码路径**：`pseudo_observation_brpo_style.py`

**信息源**：support sets + overlap mask + projected depth validity

**验证域定义**：
$$
\text{valid}_{left} = \text{support}_{left} \land \text{overlap}_{left} \land (d_{left} > 0)
$$
$$
\text{valid}_{right} = \text{support}_{right} \land \text{overlap}_{right} \land (d_{right} > 0)
$$

**Shared $C_m$**：
$$
C_m = \begin{cases}
1.0 & \text{if valid}_{left} \land \text{valid}_{right} \\
0.5 & \text{if valid}_{left} \oplus \text{valid}_{right} \\
0.0 & \text{otherwise}
\end{cases}
$$

**特点**：
- Shared $C_m$（RGB/depth 共用）
- Confidence 与 T~ 分离
- 但 verifier backend 不够强（单向 matcher mask）

---

### 3.5 Exact M~（strict BRPO semantics）

**代码路径**：`pseudo_observation_brpo_style.py` → exact branches

**信息源**：strict mutual NN verification（工程近似）

**Confidence**：同 hybrid M~，但语义严格定位为 BRPO-style

**与 old M~ 对比**：
- `exact M~ + old T~` ≈ old A1（差 < 1e-5 PSNR）
- 说明 exact M~ 已基本对齐 BRPO semantics

---

## 4. 下游消费方式

M~ 的输出被 RGB/depth loss 消费：

### 4.1 Standard loss

$$
L_{rgb} = M~ \cdot |I_{render} - I_{target}|
$$
$$
L_{depth} = M~ \cdot |d_{render} - T~|
$$

M~ 决定监督域和监督强度。

### 4.2 Shared-$C_m$ loss（exact BRPO）

$$
L_{rgb} = C_m \cdot |I_{render} - I_{target}|
$$
$$
L_{depth} = C_m \cdot |d_{render} - T~|
$$

RGB/depth 共用同一个 $C_m$。

---

## 5. 各版本与 BRPO 的差异

| 版本 | Confidence 来源 | 形态 | 与 BRPO 差距 |
|------|----------------|------|-------------|
| BRPO | mutual NN verification | 离散三档 | 0（定义） |
| Old M~ | RGB matcher + geometry tier | 半连续 | 工程稳但语义不完全对齐 |
| New M~ | score_stack 派生 | 连续 | 与 T~ 同源，偏离大 |
| Hybrid M~ | support sets verification | 离散三档 | verifier 不够强 |
| Exact M~ | strict BRPO semantics | 离散三档 | 语义对齐，已基本 parity |

---

## 6. 为什么 exact M~ + exact T~ 没赢

**数值证据**：
- `exact M~ + old T~` ≈ 24.1877（与 old A1 等价）
- `exact M~ + exact T~` ≈ 24.1744（弱 -0.013 PSNR）
- exact T~ 与 hybrid T~ target array 差异只有浮点噪声

**原因**：
- M~ 已基本对齐
- T~ 的 backend 不够强（proxy $I_t^{fix}$ / projected-depth field）
- 真正瓶颈在上游 Layer B，不是 M~ contract

---

## 7. 当前最佳组合与下一步

### 7.1 最佳组合
- `exact M~ + old T~` ≈ old A1（当前最稳）
- exact M~ 的 semantics 已对齐 BRPO
- old T~ 的 depth pipeline 提供稳定兜底

### 7.2 下一步方向
若坚持 exact BRPO：
- 上移到 Layer B proxy：改进 `I_t^{fix}` 质量、verifier backend
- 不继续在 M~ contract 上微调

若走 replay-first：
- 承认 hybrid/stable T~ 是工程分支
- 在该标签下继续优化

---

## 8. 代码位置

| 文件 | 功能 |
|------|------|
| `pseudo_branch/brpo_v2_signal/rgb_mask_inference.py` | RGB support/confidence |
| `pseudo_branch/brpo_v2_signal/joint_confidence.py` | Old M~ joint confidence |
| `pseudo_branch/brpo_v2_signal/joint_observation.py` | New M~（与 T~ 同源） |
| `pseudo_branch/brpo_v2_signal/pseudo_observation_brpo_style.py` | Hybrid M~ / Exact M~ |
| `scripts/run_pseudo_refinement_v2.py` | M~ 消费（RGB/depth loss） |

---

## 9. M~ 与 T~ 的最初耦合实现

**耦合点**：
- old A1：RGB 链和 depth 链分别构造，但最终 joint confidence 取 min，target 复用已有 depth pipeline
- new A1：score_stack 同时派生 confidence（M~）和 target（T~）

**语义分离**：
- M~：监督域 + 监督强度
- T~：监督目标数值

当前工程趋势：M~ 和 T~ 越来越分离，各自有独立设计文档。

---

> 文档口径：M~ = Mask（confidence）模块。与 T~（Target）最初耦合实现，但语义独立。T~ 的详细设计见 TARGET_DESIGN.md。
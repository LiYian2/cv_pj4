# MASK_DESIGN.md - M~ Mask 设计文档

> 更新时间：2026-04-22 02:50 (Asia/Shanghai)

> **书写规范**：
> 1. 只讲 M~（mask/confidence）：信息从哪里来、怎么转换成 confidence、怎么被下游消费
> 2. 遵循"信息源 → 信号转换 → 下游消费"链式分析
> 3. 数学公式用 `$...$` 或 `$$...$$` 包裹
> 4. M~ 与 T~ 语义分离，但工程实现可能有组合命名约定
> 5. 更新后修改文档顶部时间戳

---

## 1. 概览

M~（Mask/Confidence）决定"哪些像素被监督、监督强度如何"。**M~ 只有 3 大类**：

| 类别 | C_m 来源 | 与 T~ 关系 | 关键特征 |
|------|---------|-----------|---------|
| **M1: Legacy Joint** | rgb_conf × geometry_tier（取 min） | 分离 | 半连续值，工程最稳 |
| **M2: BRPO-style Support Sets** | verify_both=1.0, verify_xor=0.5, neither=0.0 | 分离（同验证域） | 离散三档，与 BRPO 论文 C_m 形态一致 |
| **M3: Hybrid Geometry-gated** | geometry + candidate competition | **同源** | confidence 与 target 同一 score_stack 派生 |

**关键结论**：
- M2 (BRPO-style) 已对齐 BRPO 论文 C_m semantics
- `exact_brpo_cm_old_target_v1 ≈ old A1`（差 < 1e-5 PSNR），说明 M2 已不是主瓶颈
- 主瓶颈在 T~ 的 verifier backend（Layer B）

---

## 2. M1: Legacy Joint Confidence

### 2.1 信息源

**代码位置**：`joint_confidence.py` + `rgb_mask_inference.py`

**输入**：
- RGB matcher confidence：`rgb_conf_cont`（连续值）
- Geometry tier：来自 source_map（BOTH=1.0, LEFT/RIGHT=0.5, NONE=0）

### 2.2 信号转换

**RGB 链**：
$$
	ext{rgb\_conf} = 	ext{continuous\_score\_from\_matcher\_confidence}
$$

**Depth 链（geometry tier）**：
$$
	ext{geometry\_tier}[i] = egin{cases}
1.0 & 	ext{if source\_map}[i] = 	ext{BOTH} \
0.5 & 	ext{if source\_map}[i] = 	ext{LEFT or RIGHT} \
0.0 & 	ext{otherwise}
\end{cases}
$$

**Joint confidence**：
$$
	ext{joint\_confidence} = \min(	ext{rgb\_conf}, 	ext{geometry\_tier})
$$
$$
	ext{joint\_confidence\_cont} = 	ext{rgb\_conf\_cont} 	imes 	ext{geometry\_tier}
$$

### 2.3 下游消费

被 `build_stageA_loss()` 和 `build_stageA_loss_source_aware()` 消费：
- RGB loss: `rgb_mask = confidence_mask` 或 `rgb_confidence_mask`
- Depth loss: `depth_mask = rgb_mask` 或 `depth_confidence_mask`

### 2.4 特点

- **半连续值**：不是硬三档 $\{1, 0.5, 0\}$，而是连续值 capped by geometry tier
- **RGB/depth 分离过滤**：各自有 mask，取共同 trusted support
- **工程稳**：经过大量实验验证，fallback 机制成熟

---

## 3. M2: BRPO-style Support Sets

### 3.1 信息源

**代码位置**：`pseudo_observation_brpo_style.py`

**输入**：
- `support_left`：左侧 matcher correspondence support
- `support_right`：右侧 matcher correspondence support
- `overlap_mask_left/right`：overlap 有效域
- `projected_depth_left/right`：投影 depth validity

### 3.2 信号转换

**验证域定义**：
$$
	ext{valid}_{left} = 	ext{support}_{left} \land 	ext{overlap}_{left} \land (d_{left} > 0)
$$
$$
	ext{valid}_{right} = 	ext{support}_{right} \land 	ext{overlap}_{right} \land (d_{right} > 0)
$$

**三档 C_m 生成（BRPO 论文形态）**：
$$
	ext{verify\_both} = 	ext{valid}_{left} \land 	ext{valid}_{right}
$$
$$
	ext{verify\_xor} = 	ext{valid}_{left} \oplus 	ext{valid}_{right}
$$

$$
C_m[i] = egin{cases}
1.0 & 	ext{if } i \in 	ext{verify\_both} \
0.5 & 	ext{if } i \in 	ext{verify\_xor} \
0.0 & 	ext{otherwise}
\end{cases}
$$

### 3.3 下游消费

被 `build_stageA_loss()` 和 `build_stageA_loss_exact_shared_cm()` 消费：
- 作为 shared C_m，RGB/depth 共用同一 mask

### 3.4 特点

- **离散三档**：与 BRPO 论文 C_m 形态一致
- **与 T~ 分离**：confidence 与 target 来源不同（虽然共用同一验证域）
- **但 verifier backend 不够强**：当前是 proxy backend（单向 matcher mask），不是 BRPO 论文要求的双向验证

### 3.5 Exact C_m（M2 的 exact instantiation）

`exact_brpo_cm_*` 系列：
- **信息源**：`verify_single_branch_exact()` 输出
- **C_m 生成**：同 M2 三档逻辑，但用 exact backend support
- **Provenance tracked**：记录每个像素来自哪个 reference

**数值证据**：
- `exact_brpo_cm_old_target_v1 ≈ old A1`（差 < 1e-5 PSNR）
- 说明 M2 (BRPO-style C_m) 已基本对齐 BRPO semantics

---

## 4. M3: Hybrid Geometry-gated

### 4.1 信息源

**代码位置**：`joint_observation.py` + `brpo_direct_v1` path

**输入**：
- 4-candidate depth stack
- 4-evidence score stack

### 4.2 信号转换

**Score stack 派生**：
$$
	ext{score\_prob} = 	ext{softmax}(	ext{score\_stack})
$$
$$
	ext{confidence} = \sqrt{	ext{conf}_{rgb} 	imes 	ext{conf}_{depth}}
$$

其中 conf_rgb / conf_depth 都从同一个 score_stack 派生。

### 4.3 特点

- **M~ 与 T~ 同源**：confidence 和 target 都从 score_stack 派生
- **同源问题**：如果 score ranking 错误：
  $$	ext{wrong\_target} + 	ext{inflated\_confidence} 	o 	ext{smooth\_but\_self\_consistent\_error}$$
- 这就是为什么 M3 不稳定

---

## 5. M~ 与 BRPO 论文对齐分析

### 5.1 BRPO 论文 M~ 定义

$$
C_m[i] = egin{cases}
1.0 & 	ext{if } i \in M_{left} \cap M_{right} \
0.5 & 	ext{if } i \in M_{left} \oplus M_{right} \
0.0 & 	ext{otherwise}
\end{cases}
$$

其中 $M_{left/right}$ 来自 **mutual nearest-neighbor correspondence**。

### 5.2 各类对齐度

| 类别 | C_m 形态 | Verifier Backend | 与 BRPO 论文对齐度 |
|------|---------|-----------------|------------------|
| M1 Legacy | 半连续 | RGB gate + depth pipeline | 低（形态不同） |
| M2 BRPO-style | 离散三档 ✅ | Proxy（单向 matcher）⚠️ | **形态一致，backend 不够强** |
| M2 Exact | 离散三档 ✅ | Exact（mutual NN + geometric）✅ | **完全对齐** |
| M3 Hybrid | 连续（同源）| Candidate competition | 低（偏离大） |

---

## 6. M~ 与 T~ 的命名约定

`pseudo_observation_mode` 命名反映了 M~ + T~ 组合：

| 命名 pattern | M~ 部分 | T~ 部分 |
|-------------|--------|--------|
| `brpo_style_v1` | M2 (BRPO-style) | T2 (BRPO-style Proxy) |
| `exact_brpo_cm_old_target_v1` | M2 Exact | T1 (Old) |
| `exact_brpo_cm_full_target_v1` | M2 Exact | T2 (BRPO-style Proxy) |
| `exact_brpo_cm_stable_target_v1` | M2 Exact | T3 (Stable) |
| `exact_brpo_upstream_target_v1` | M2 Exact | T4 (Exact Upstream) |
| `hybrid_brpo_cm_geo_v1` | M3 (Hybrid) | T3 (Hybrid) |

**关键**：`cm` = confidence mask（M~），`target` 后缀 = T~ variant。

---

## 7. 下游消费层总结

| Loss mode | RGB mask | Depth mask | M~ 类型 |
|-----------|---------|-----------|---------|
| `legacy` | `confidence_mask` 或 `rgb_confidence_mask` | `confidence_mask` 或 `depth_confidence_mask` | M1/M2/M3 |
| `source_aware` | `rgb_confidence_mask` | `depth_confidence_mask` + source_map tier | M1 |
| `exact_shared_cm_v1` | **shared C_m** | **shared C_m** | M2 Exact |

---

## 8. 代码位置索引

| 文件 | 功能 | M~ 类别 |
|------|------|--------|
| `pseudo_branch/brpo_v2_signal/joint_confidence.py` | Legacy joint confidence | M1 |
| `pseudo_branch/brpo_v2_signal/rgb_mask_inference.py` | RGB confidence | M1 |
| `pseudo_branch/brpo_v2_signal/pseudo_observation_brpo_style.py` | BRPO-style M~ + Exact C_m | M2 |
| `pseudo_branch/brpo_reprojection_verify.py` | Exact backend verifier | M2 Exact |
| `pseudo_branch/brpo_v2_signal/joint_observation.py` | Hybrid geometry-gated | M3 |
| `pseudo_branch/pseudo_loss_v2.py` | Loss 消费 | 所有 |

---

## 9. 当前状态

- **M2 已对齐**：`exact_brpo_cm_old_target_v1 ≈ old A1`
- **不是主瓶颈**：M~ 已完成对齐，下一步在 T~ upstream backend

---

> 文档口径：M~ = Mask（confidence）模块。与 T~（Target）语义分离。组合实验见 M_T_COMBINATIONS.csv。

# TARGET_DESIGN.md - T~ Target 设计文档

> 更新时间：2026-05-04 14:20 (Asia/Shanghai)

> **书写规范**：
> 1. 只讲 T~（target）：信息从哪里来、怎么转换成 target 数值、怎么被下游消费
> 2. 遵循"信息源 → 信号转换 → 下游消费"链式分析
> 3. 数学公式用 `$...$` 或 `$$...$$` 包裹
> 4. T~ 与 M~ 语义分离，但工程实现可能有组合命名约定
> 5. 更新后修改文档顶部时间戳

---

## 1. 概览

T~（Target）决定"监督目标的数值是什么"。**T~ 有 5 大类**：

| 类别 | Verifier Backend | Composition 权重 | Fallback | 与 BRPO 论文对齐度 |
|------|-----------------|-----------------|---------|------------------|
| **T1: Old Pipeline** | RGB gate + depth pipeline | fusion_weight | render_depth fallback | 低（工程稳） |
| **T2: BRPO-style Proxy** | Proxy backend（单向 matcher） | fusion_weight | 无（隐式有问题） | 中（形态对，backend 不够强） |
| **T3: Stable Blend** | 同 T2 + stable fallback | stable blend | stable fallback | 低（工程 hybrid） |
| **T4: Exact Upstream** | Exact backend（mutual NN + geometric） | continuous confidence | **no_render_fallback=true** | **最高** |
| **T5: Paper-realign Depth-only** | light geometry-only bidirectional projection | fusion_weight + depth-only confidence | **no_render_fallback=true** | 中高（producer 解耦，但比 T4 更轻） |

**关键结论**：
- T4 不只是最接近 BRPO 论文 semantics 的 T~ variant，**它也已经在 fixed clean G~ / fixed T1 formal compare 中成为当前 winner**
- T2 / `exact_brpo_full_target_v1` 的负结果说明：命名 `brpo_style` 或 consumer-side exact 化本身不等于真正对齐 BRPO target path
- 真正决定 replay 转正的增益来自 T~ 的 verifier backend（Layer B）与 projected-depth / target field 的 upstream 对齐
- 当前 T~ 主线应固定为：`exact_brpo_upstream_target_v1 + exact_shared_cm_v1`
- `paper_brpo_target_v1` 已作为 paper-realign compare branch 落地：它验证了“去掉 RGB gate / 去掉 render fallback、把 T~ 拆成 depth-only producer”这条结构能跑通，但 first full9 compare 还不足以让它替代 T4 mainline。

---

## 2. T1: Old Pipeline

### 2.1 信息源

**代码位置**：`pseudo_branch/target/depth_supervision_v2.py` → `build_depth_supervision_v2()`

**输入**：
- `projected_depth_left/right`：reference 投影到 pseudo-frame 的 depth
- `fusion_weight_left/right`：fusion 权重（来自 proxy matcher）
- `rgb_confidence`：RGB gate（前置过滤）
- `render_depth`：当前 render depth（fallback 用）

### 2.2 信号转换

**验证域定义**：
$$
	ext{valid}_{left} = (d_{left} > 0) \land (	ext{rgb\_confidence} \ge 0.5)
$$
$$
	ext{valid}_{right} = (d_{right} > 0) \land (	ext{rgb\_confidence} \ge 0.5)
$$

**Target 区域划分**：
$$
\begin{aligned}
	ext{both} &= 	ext{valid}_{left} \land 	ext{valid}_{right} \
	ext{left\_only} &= 	ext{valid}_{left} \land 
eg 	ext{valid}_{right} \
	ext{right\_only} &= 	ext{valid}_{right} \land 
eg 	ext{valid}_{left} \
	ext{fallback} &= 
eg 	ext{valid}_{left} \land 
eg 	ext{valid}_{right} \land (	ext{rgb\_confidence} \ge 0.5)
\end{aligned}
$$

**Target 数值生成**：
$$
d_{target}[	ext{both}] = \frac{w_l \cdot d_l + w_r \cdot d_r}{w_l + w_r}
$$
$$
d_{target}[	ext{left\_only}] = d_l
$$
$$
d_{target}[	ext{right\_only}] = d_r
$$
$$
d_{target}[	ext{fallback}] = 	ext{render\_depth} 	ext{ 或 } 0
$$

### 2.3 下游消费

被 `build_stageA_loss()` 和 `build_stageA_loss_source_aware()` 消费：
- Legacy: 单一 depth loss term
- Source-aware: seed/dense/fallback tier 分层

### 2.4 特点

- **工程 maturity 最高**：RGB gate + fusion weight + fallback 都经过长期验证
- **但语义不完全对齐 BRPO**：
  - Verifier 是 RGB gate（不是 mutual NN）
  - Composition 权重是 fusion_weight（不是 verifier-driven confidence）
  - 有 fallback（BRPO 论文无 fallback）

---

## 3. T2: BRPO-style Proxy

### 3.1 信息源

**代码位置**：`pseudo_branch/observation/pseudo_observation_brpo_style.py` → `build_brpo_style_observation()`

**输入**：
- `support_left/right`：matcher correspondence support
- `projected_depth_left/right`：projected depth
- `fusion_weight_left/right`：fusion 权重
- `overlap_mask_left/right`：overlap 有效域

### 3.2 信号转换

**验证域定义**（与 M2 共用）：
$$
	ext{valid}_{left} = 	ext{support}_{left} \land 	ext{overlap}_{left} \land (d_{left} > 0)
$$
$$
	ext{valid}_{right} = 	ext{support}_{right} \land 	ext{overlap}_{right} \land (d_{right} > 0)
$$

**Target 数值生成**：
$$
d_{target}[	ext{verify\_both}] = \frac{w_l \cdot d_l + w_r \cdot d_r}{w_l + w_r}
$$
$$
d_{target}[	ext{verify\_left\_only}] = d_l
$$
$$
d_{target}[	ext{verify\_right\_only}] = d_r
$$
$$
d_{target}[	ext{neither}] = 0
$$

### 3.3 特点

- **命名 `brpo_style` 只反映形态**：
  - C_m 形态正确（离散三档）
  - Target composition 与 C_m 共用验证域
  
- **但 verifier backend 是 proxy**：
  - `support_left/right` 来自单向 matcher mask
  - 不是 BRPO 论文要求的 mutual NN + geometric verification
  
- **Composition 权重是 fusion_weight**：
  - 来自 proxy matcher confidence
  - 不是 BRPO 论文的 verifier-driven confidence

- **无 explicit fallback**，但 backend 不够强时隐式有问题

### 3.4 为什么 T2 不是最接近 BRPO semantics

**BRPO 论文 T~ 要求**：
- Verifier: mutual nearest-neighbor + geometric verification
- Composition 权重: verifier-driven confidence
- Fallback: 无

**T2 实际**：
- Verifier: proxy backend（单向 matcher）⚠️
- Composition 权重: fusion_weight（proxy）⚠️
- Fallback: 无 ✅

只有 fallback 符合，verifier 和 composition 权重都不够强。

---

## 4. T3: Stable Blend

### 4.1 信息源

**代码位置**：`pseudo_branch/observation/pseudo_observation_brpo_style.py` → `build_brpo_style_observation_v2()`

**输入**：
- T2 的所有输入
- `stable_depth_target`：稳定 depth target（来自 old T~ 或 render_depth）
- `render_depth`：当前 render depth

### 4.2 信号转换

$$
d_{target} = \alpha \cdot d_{T2} + (1-\alpha) \cdot d_{stable}
$$

其中 $d_{stable}$ 来自 old T~ 或 render_depth fallback。

### 4.3 特点

- **工程 hybrid**：用 stable target 做兜底
- **减少极端 case**：T2 不稳定时，stable blend 提供保护
- **但偏离 BRPO semantics**：引入 fallback，不是 strict BRPO

---

## 5. T4: Exact Upstream（Phase T2 新增）

### 5.1 信息源

**代码位置**：
- `pseudo_branch/target/depth_supervision_v2.py` → `build_exact_upstream_depth_target()`
- `pseudo_branch/observation/brpo_reprojection_verify.py` → `verify_single_branch_exact()`

**输入**：
- `support_left_exact/right_exact`：exact backend verified support
- `projected_depth_left_exact/right_exact`：exact backend projected depth
- `confidence_left_exact/right_exact`：continuous confidence（verifier-driven）
- `provenance_left/right`：branch provenance tracking

### 5.2 信号转换

**Exact backend verification**：
$$
	ext{reproj\_err} = \|u_{reproj} - u_{pseudo}\|_2
$$
$$
	ext{rel\_depth\_err} = \frac{|z_{reproj} - z_{pseudo}|}{z_{pseudo}}
$$

**Binary support**：
$$
	ext{support}^{	ext{exact}} = (	ext{reproj\_err} < 	au_{px}) \land (	ext{rel\_depth\_err} < 	au_d) \land (z_{reproj} > 0)
$$

**Continuous confidence（verifier-driven）**：
$$
C_{side} = \exp left(-\frac{	ext{reproj\_err}}{	au_{px}}
ight) \cdot \exp left(-\frac{	ext{rel\_depth\_err}}{	au_d}
ight)
$$

**Target 数值生成（verifier-driven weighted composition）**：
$$
d_{target}[i] = \begin{cases}
\frac{C_l \cdot d_l + C_r \cdot d_r}{C_l + C_r} & \text{if both sides verified} \\
d_l & \text{if only left verified} \\
d_r & \text{if only right verified} \\
0 & \text{otherwise}
\end{cases}
$$

**关键**：权重 $C_l, C_r$ 是 **continuous confidence（verifier-driven）**，不是 fusion_weight（proxy）。

### 5.3 下游消费

被 `build_stageA_loss_exact_shared_cm()` 消费：
- Shared C_m（M2 Exact）
- Continuous confidence weighting
- `no_render_fallback=true`

### 5.4 特点

- **Verifier backend 是 exact**：mutual NN + geometric verification ✅
- **Composition 权重是 verifier-driven**：continuous confidence ✅
- **Fallback 是 no_render_fallback=true**：符合 BRPO semantics ✅
- **Provenance tracked**：记录每个像素来自哪个 reference

### 5.5 为什么 T4 是最接近 BRPO semantics

**BRPO 论文 T~ 要求**：
- Verifier: mutual nearest-neighbor + geometric verification
- Composition 权重: verifier-driven confidence
- Fallback: 无

**T4 实际**：
- Verifier: exact backend（mutual NN + geometric）✅
- Composition 权重: continuous confidence（verifier-driven）✅
- Fallback: no_render_fallback=true ✅

**完全对齐**，这是 T4 的语义基础。

### 5.6 T4 formal compare verdict

固定 protocol：
- `joint_topology_mode=brpo_joint_v1`
- G~ = clean `summary_only` no-action control
- 4-arm replay compare：`oldA1` / `exactBrpoCm_oldTarget_v1` / `exactBrpoFullTarget_v1` / `exactBrpoUpstreamTarget_v1`

结果判断：
- `exactBrpoUpstreamTarget_v1_newT1_summary_only` 明确高于 `oldA1_newT1_summary_only`
- 同时高于 `exactBrpoCm_oldTarget_v1_newT1_summary_only`
- 也高于 `exactBrpoFullTarget_v1_newT1_summary_only`

设计结论：
- `exact_brpo_full_target_v1` 证明：**只把 proxy backend 下的 consumer-side target/loss contract 写得更 exact，不足以赢 old A1**
- `exact_brpo_upstream_target_v1` 证明：**把 verifier backend / projected-depth / target field 整体改成 exact upstream 之后，strict BRPO T~ 才真正转成正向 winner**
- 因此当前 T~ 主线不是 old T~，也不是 exact-full-target proxy variant，而是 **T4 exact upstream**

---

## 6. T5: Paper-realign Depth-only（2026-05-04 新增）

### 6.1 信息源

**代码位置**：`pseudo_branch/target/depth_supervision_v2.py` → `build_paper_target_depth_v1()`

**输入**：
- `projected_depth_left/right`：左右 reference 投影到 fused pseudo-frame 的 depth
- `projected_valid_left/right`：轻量几何有效域
- `fusion_weight_left/right`：双侧融合权重
- `tau_rel_depth`：双侧轻量一致性阈值

### 6.2 信号转换

T5 的核心不是再造一个 heavy verifier，而是把 T~ 拆成 **depth-only producer**：
- `both`：左右都 valid 时，用 `fusion_weight_left/right` 做加权融合
- `left_only / right_only`：只保留单边投影深度
- `none`：直接 invalid，不再走 `render_depth` fallback
- `target_confidence`：只来自 depth-side 双边一致性，不再读取 `raw_rgb_confidence`

### 6.3 下游消费

被 `build_paper_brpo_target_observation()` 包成 `paper_brpo_target_v1`：
- `pseudo_depth_target_paper_brpo_target_v1.npy`
- `pseudo_source_map_paper_brpo_target_v1.npy`
- `pseudo_valid_mask_paper_brpo_target_v1.npy`
- `pseudo_target_confidence_paper_brpo_target_v1.npy`

再被 `paper_brpo_split_v1` / `paper_brpo_split_depthconf_v1` 消费：
- `paper_brpo_split_v1`：RGB 只用 `C_m`，depth 也只用 `C_m`
- `paper_brpo_split_depthconf_v1`：RGB 只用 `C_m`，depth 用 `C_m × depth-only target_confidence`

### 6.4 特点

- **优点**：producer 语义更接近“RGB mask 与 depth target 解耦”的 paper-realign 目标；T~ 不再被 RGB gate 提前裁死
- **缺点**：它比 T4 更轻，没有 exact backend 的 mutual NN + reprojection verification，因此 target field 更宽但也更容易引入 single-side / weakly-supported supervision
- **当前判断**：T5 已完成 live code、full signal 与 full9 compare，证明这条 depth-only producer 结构可运行；但 current verdict 仍是 **T4 exact-upstream 继续做 mainline，T5 保留为 compare branch**。


## 7. T~ 与 M~ 的命名约定

`pseudo_observation_mode` 命名反映了 M~ + T~ 组合：

| 命名 pattern | M~ 部分 | T~ 部分 |
|-------------|--------|--------|
| `brpo_style_v1` | M2 | T2 |
| `brpo_style_v2` | M2 | T3 |
| `paper_brpo_cm_old_target_v1` | M2 Paper | T1 |
| `paper_brpo_target_v1` | M2 Paper | T5 |
| `exact_brpo_cm_old_target_v1` | M2 Exact | T1 |
| `exact_brpo_cm_full_target_v1` | M2 Exact | T2 |
| `exact_brpo_cm_stable_target_v1` | M2 Exact | T3 |
| `exact_brpo_cm_hybrid_target_v1` | M2 Exact | T3 (hybrid) |
| `exact_brpo_upstream_target_v1` | M2 Exact | T4 |
| `hybrid_brpo_cm_geo_v1` | M3 | T3 (hybrid) |

---

## 8. 各类与 BRPO 论文对齐分析

### 8.1 BRPO 论文 T~ 定义

- **Verifier**：pseudo-frame 与 reference 的 mutual nearest-neighbor + geometric verification
- **Composition 权重**：verifier-driven confidence
- **Fallback**：无（不监督区域保持 invalid）

### 8.2 各类对齐度

| 类别 | Verifier Backend | Composition 权重 | Fallback | 与 BRPO 论文对齐度 |
|------|-----------------|-----------------|---------|------------------|
| T1 | RGB gate + depth pipeline | fusion_weight | render_depth fallback | 低 |
| T2 | Proxy（单向 matcher）⚠️ | fusion_weight（proxy）⚠️ | 无 ✅ | 中 |
| T3 | 同 T2 | stable blend | stable fallback | 低 |
| T4 | Exact（mutual NN + geometric）✅ | continuous confidence（verifier-driven）✅ | no_render_fallback=true ✅ | **最高** |
| T5 | light geometry-only bidirectional projection | fusion_weight + depth-only confidence | no_render_fallback=true ✅ | 中高 |

---

## 9. 代码位置索引

| 文件 | 功能 | T~ 类别 |
|------|------|--------|
| `pseudo_branch/target/depth_supervision_v2.py` | Old Pipeline + Exact Upstream + Paper-realign depth-only | T1, T4, T5 |
| `pseudo_branch/observation/pseudo_observation_brpo_style.py` | BRPO-style Proxy + Stable Blend + Paper target observation | T2, T3, T5 |
| `pseudo_branch/observation/brpo_reprojection_verify.py` | Exact backend verifier | T4 |
| `pseudo_branch/refine/pseudo_loss_v2.py` | Loss 消费 | 所有 |

---

## 10. 当前状态

- **T4 semantic path 已完全落地**：exact backend / exact target field / exact loss contract / consumer path 都已打通
- **T4 formal compare 已完成**：`exact_brpo_upstream_target_v1` 在 fixed clean G~ / fixed T1 replay compare 中赢过 old A1、exact-oldtarget、exact-fulltarget
- **当前 T~ 主线已更新**：`exact_brpo_upstream_target_v1 + exact_shared_cm_v1`
- **T5 compare branch 已落地**：`paper_brpo_target_v1` 与 `paper_brpo_split_*` 已完成 full9 compare，但当前仍停留在 compare/diagnostics 身份
- **结构性教训已固化**：T~ 的主瓶颈确实在 Layer B verifier/backend / projected-depth field，而不是继续做 proxy-backend 下的 consumer-side exact 化

---

> 文档口径：T~ = Target 模块。与 M~（Mask）语义分离。组合实验见 M_T_COMBINATIONS.csv。

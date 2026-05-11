# MASK_DESIGN.md - M~ Mask 设计文档

> 更新时间：2026-05-04 14:20 (Asia/Shanghai)

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
- 2026-05-04 新增的 `paper_cm_only` 已把 M2 再拆出一个更轻的 compare variant：直接在 fused pseudo domain 上和 left/right GT 做 support-set matching，不再让几何 verifier 参与定义 `C_m`。这条线已经可复现，但 first full9 compare 还不足以取代 current exact M~。

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
	ext{geometry\_tier}[i] = \begin{cases}
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

**代码位置**：`pseudo_branch/observation/pseudo_observation_brpo_style.py`

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

### 3.6 Paper C_m-only（M2 的轻量变体，2026-05-04 新增）

**代码位置**：`scripts/brpo_build_mask_from_internal_cache.py --verification-mode paper_cm_only`

**信息源**：
- fused pseudo RGB
- left / right GT ref RGB
- reciprocal matcher（`sparse_desc_2d` 或 `dense_pts3d_3d`）

**信号转换**：
- 直接在 fused pseudo image domain 上分别得到 `M_left`、`M_right`
- 不再调用 branch-native geometry verifier 去决定 support 是否成立
- 仍然 rasterize 成同一个离散三档 `C_m`：`both -> 1.0`，`xor -> 0.5`，`none -> 0.0`

**特点**：
- 与论文里的集合语义更近：`C_m` 更像纯 matching support set，而不是 matching + heavy geometry contract 的混合物
- 但它也更轻：缺少 exact backend 的 reprojection/depth consistency 约束，因此 current 定位是 compare branch，不是已经转正的 mainline M~

---

## 4. M3: Hybrid Geometry-gated

### 4.1 信息源

**代码位置**：`pseudo_branch/observation/joint_observation.py` + `brpo_direct_v1` path

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
| `paper_brpo_cm_old_target_v1` | M2 Paper（fused-domain support sets） | T1 (Old) |
| `paper_brpo_target_v1` | M2 Paper（fused-domain support sets） | T5 (Paper-realign depth-only) |
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
| `exact_shared_cm_v1` | **shared C_m × target_confidence** | **shared C_m × target_confidence** | M2 Exact |
| `paper_brpo_split_v1` | **shared C_m** | **shared C_m** | M2 Paper / M2 Exact |
| `paper_brpo_split_depthconf_v1` | **shared C_m** | **shared C_m × depth-only target_confidence** | M2 Paper / M2 Exact |

---

## 8. 代码位置索引

| 文件 | 功能 | M~ 类别 |
|------|------|--------|
| `pseudo_branch/mask/joint_confidence.py` | Legacy joint confidence | M1 |
| `pseudo_branch/mask/rgb_mask_inference.py` | RGB confidence | M1 |
| `pseudo_branch/mask/brpo_confidence_mask.py` | BRPO-style support-set confidence / discrete+continuous C_m | M2 |
| `pseudo_branch/mask/brpo_train_mask.py` | train-time propagated support / confidence mask | M2 |
| `pseudo_branch/mask/confidence_builder.py` | simplified confidence helper / legacy utility | M1 |
| `pseudo_branch/observation/pseudo_observation_brpo_style.py` | BRPO-style M~ + Exact C_m bundle | M2 |
| `pseudo_branch/observation/brpo_reprojection_verify.py` | Exact backend verifier | M2 Exact |
| `pseudo_branch/observation/joint_observation.py` | Hybrid geometry-gated | M3 |
| `pseudo_branch/refine/pseudo_loss_v2.py` | Loss 消费 | 所有 |

---

## 9. 当前状态

- **M2 已对齐**：`exact_brpo_cm_old_target_v1 ≈ old A1`
- **不是主瓶颈**：M~ contract 本身已完成对齐，当前剩余变量主要在 support seed / matching layer
- 当前 live exact `C_m` support 入口已从单一 `pseudo_branch/common/flow_matcher.py` 扩成可切换 matcher factory：`sparse_desc_2d` 继续走旧 `FlowMatcher`，`dense_pts3d_3d` 走新 `Dense3DMatcher`
- 为后续 M~ matching upgrade，step1/step2 代码已落地到 `pseudo_branch/common/`：
  - `mast3r_pair_forward.py`：shared MASt3R pair forward helper
  - `mast3r_matchers.py`：reusable matcher layer，当前已实现 `Dense3DMatcher` 与 `build_pair_matcher()`
- 2026-04-24 已完成 live wiring：`scripts/brpo_build_mask_from_internal_cache.py` 与 `scripts/build_brpo_v2_signal_from_internal_cache.py` 都已支持 `--matcher-mode` / `--dense3d-conf-quantile`，并把 matcher config / meta 落盘
- full mechanism validation 与后续 `StageB120 + replay` compare 已完成。结果已经很明确：dense3d 的 coverage 提升是真的，q0.70 也确实优于 q0.80，但在真正的长程 replay compare 里，sparse 仍然更好（PSNR 24.0045 > 23.6660 > 23.5816；SSIM 同样 sparse 最优）
- 进一步的 structural forensic 表明：旧 live exact `C_m` 低 coverage 的直接原因就是 sparse 2D reciprocal matching，而不是额外 bug；同时 dense3d 接通后，`exact_brpo_upstream_target_v1` 的 target depth / target confidence / source map 也确实同步变化，因此问题不在于 target depth 没切过去。
- 当前 M~ 的真实症结更像是新增 support 的组成：dense3d 新增 valid 区域主要是 single-branch，而不是 both-branch。以 q0.70 为例，新增 valid 区域里约 `64.1%` 属于 `C_m=0.5`，只有约 `35.9%` 属于 `C_m=1.0`；而 `exact_shared_cm_v1` 真正进 loss 的 effective mask 还会再乘 `target_confidence`，使新增区域单位质量偏弱。
- 针对“是否应退回到只用裸 `C_m` 做 supervision”这一点，新的 `cm_only` StageB120+replay ablation 已给出否定结果：移除 `valid_mask / target_confidence` 后，sparse 与 q070 都小幅退化；而当前 live 导出中 `valid_mask` 并未额外裁掉 `C_m` 区域，因此真正不能简单移除的是 `target_confidence`。也就是说，当前问题不是 `target_confidence` 把好信号压坏了，而更像是它在替 single-heavy supervision 做必要抑制。
- `rgb_only` StageB120+replay ablation 也已完成：在 fixed route 下用 `--stageA_disable_depth` 完全移除 pseudo depth loss 后，sparse 明显退化，而 q070 只得到极小表面变化且仍明显落后 sparse。因此 dense3d 的主要 gap 也不能简单归因于 depth 项本身。
- 因此 MASt3R 3D 路线并非“没起作用”，而是“当前新增 coverage 没有转成更好的 downstream replay 结果”。真正的决策点已经从“3D 有没有用”变成“single/both contract、target depth composition 与 confidence weighting 是否与原方法存在偏差”。
- 这意味着后续若继续推进 M~，应优先回到 BRPO 原始 method，对照检查 dense support 的有效性、single/both 组成、target depth composition 与 confidence weighting，而不是继续做同类 quantile 扫描

---

> 文档口径：M~ = Mask（confidence）模块。与 T~（Target）语义分离。组合实验见 M_T_COMBINATIONS.csv。

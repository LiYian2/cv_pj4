# A1 BRPO-style pseudo observation 工程落地文档

> 上位文档：`BRPO_A1_B3_reexploration_master_plan.md`
> 理论约束：`part3_BRPO_A1_B3_vs_BRPO_detailed_analysis.md`
> 目标：不再继续打磨 `verify proxy v1`，而是直接把 A1 推到更接近 BRPO 的语义：**pseudo-frame first, direct verification second, shared confidence for RGB/depth**。

---

## 0. 结论先说

上一轮 `verify proxy v1` 已经回答了它该回答的问题：

> **只做 confidence decoupling，但继续复用 current new A1 target，不足以把 A1 推到 BRPO。**

所以这份文档不再围绕 `proxy verifier` 做局部修补，而是直接切换到新的主问题：

> **如何把 fused pseudo-frame 变成 BRPO-style observation object，并用 direct correspondence verification 生成统一 `C_m`。**

这里的“BRPO-style”在当前工程里明确指三件事：

1. `confidence / mask` 不再来自 4-candidate score，而来自 **fused pseudo-frame 与左右 reference 的直接 correspondence support**；
2. `RGB` 和 `depth` 共用同一个 verifier-derived `C_m`；
3. `depth target` 不再来自 candidate score competition，而来自 **verified projected depth composition**。

---

## 1. 当前代码基础上，什么可以直接复用

当前仓库里已经有两块足够关键的积木，不需要再从零开始：

### 1.1 pseudo-frame side

当前 `prepare_root/fusion/samples/<frame_id>/target_rgb_fused.png` 已经给了 fused pseudo RGB。  
它虽然不等于论文里全部的 `I_t^{fix}` 推导细节，但在工程上已经能扮演 pseudo-frame 的角色。

### 1.2 direct verification side

当前仓库里已经有：

- `pseudo_branch/brpo_v2_signal/rgb_mask_inference.py`
  - 用 matcher 在 `fused_rgb` 与 `left/right reference RGB` 之间做 correspondence
  - 已经能得到 `support_left / support_right / support_both / support_single`
- `pseudo_branch/brpo_reprojection_verify.py`
  - 已经有单分支 reprojection verification 的实现与接口定义

这说明当前并不缺“BRPO-style verifier”所需的全部基础件；缺的是把这些件真正组织成新的 A1 producer / consumer contract。

---

## 2. 新 A1 的目标信息流

新的 BRPO-style A1 要明确按下面的信息流实现：

```text
fused pseudo RGB (pseudo-frame proxy)
+ left/right reference RGB correspondence
+ left/right projected depth
→ direct correspondence support sets M_left / M_right
→ shared confidence mask C_m
→ verified projected depth composition
→ observation bundle
→ StageB consumer uses the same C_m for RGB/depth
```

核心不是“再加一个 diagnostics 分支”，而是把 observation object 重新定义成：

- `pseudo_frame`：当前 fused RGB
- `C_m`：来自 correspondence verification
- `depth_target`：来自 verified projected depth

---

## 3. 第一版 BRPO-style A1 的工程定义

当前先落第一版可运行的 `brpo_style_v1`，定义如下。

### 3.1 confidence / mask

直接用 correspondence support 集合定义：

- `M_left`：fused pseudo RGB 能被左 reference 支持的像素
- `M_right`：fused pseudo RGB 能被右 reference 支持的像素

然后：

- `C_m = 1.0` on `M_left ∩ M_right`
- `C_m = 0.5` on `M_left ⊕ M_right`
- `C_m = 0.0` elsewhere

这一步不再经过 `score_k`、`best_score`、`confidence_joint = sqrt(conf_rgb * conf_depth)`。

### 3.2 depth target

depth target 不再来自 4-candidate score competition，而改成：

- 两侧都 verified：用 `projected_depth_left/right` 做 verified weighted composition
- 仅左侧 verified：用 `projected_depth_left`
- 仅右侧 verified：用 `projected_depth_right`
- 两侧都不 verified：不监督（target=0）

### 3.3 consumer contract

新的 consumer mode：

- `pseudo_observation_mode=brpo_style_v1`

其消费规则：

- `rgb_conf = pseudo_confidence_brpo_style_v1.npy`
- `depth_conf = pseudo_confidence_brpo_style_v1.npy`
- `depth_for_refine = pseudo_depth_target_brpo_style_v1.npy`
- `source_map = pseudo_source_map_brpo_style_v1.npy`

也就是 RGB / depth 共用同一个 `C_m`。

---

## 4. 为什么这版比 proxy v1 更接近 BRPO

### 4.1 它不再把 confidence 建立在 current new A1 target 上

`proxy v1` 的问题是：

- target 已经由 `brpo_joint_v1` candidate competition 决定；
- verifier 只是事后去审这个 target。

而 `brpo_style_v1` 直接把 confidence 的来源前移到：

- fused pseudo-frame 与 reference 的 direct support

这一步已经不再是“current new A1 target 的复查版 confidence”。

### 4.2 它不再从 `score_stack` 里同时派生 target 和 confidence

current new A1 的结构性问题是：

- `score_stack -> fused_depth`
- `score_stack -> confidence_joint`

这版新 A1 则改成：

- `support sets -> C_m`
- `verified projected depths -> depth_target`

所以 target / confidence 的来源关系比 current new A1 更接近 BRPO。

---

## 5. 这版仍然保留的工程近似

这版虽然更像 BRPO，但仍有两处需要诚实记录的近似：

1. pseudo-frame 目前仍用现有 `target_rgb_fused.png` 作为工程代理，而不是完整重走论文里所有 `I_t^{fix}` 细节；
2. correspondence backend 仍建立在当前 matcher / support map 管线之上，不等于论文实现本身。

但这两处近似都比 `proxy verifier v1` 更靠近真正的 BRPO A1。

---

## 6. 实施顺序

### Phase 1：producer 重写

新增 `pseudo_branch/brpo_v2_signal/pseudo_observation_brpo_style.py`：

- 输入：`support_left/right`、`projected_depth_left/right`、`fusion_weight_left/right`、`overlap_mask_left/right`
- 输出：`pseudo_confidence_brpo_style_v1`、`pseudo_depth_target_brpo_style_v1`、`pseudo_source_map_brpo_style_v1`、`pseudo_valid_mask_brpo_style_v1`

### Phase 2：builder 接线

修改 `scripts/build_brpo_v2_signal_from_internal_cache.py`：

- 在 `rgb_mask_inference` 与现有 depth artifacts 之后，写出新的 BRPO-style bundle
- 旧 `brpo_joint_v1` 保留
- `proxy verify v1` 不再作为主执行线继续扩展

### Phase 3：consumer 接线

修改 `scripts/run_pseudo_refinement_v2.py`：

- 新增 `pseudo_observation_mode=brpo_style_v1`
- 保证 RGB / depth 共用同一个 `C_m`

### Phase 4：smoke + formal compare

先做：

- signal build smoke
- 1-iter consumer smoke

再做固定 `new T1 + summary_only` compare：

- `old A1 + new T1`
- `current new A1 + new T1`
- `brpo_style_v1 + new T1`

---

## 7. 当前明确不做的事

1. 不继续打磨 `proxy verify v1` 的阈值或权重；
2. 不把 current new A1 的 4-candidate score 再解释成 BRPO-style verifier；
3. 不在 observation compare 里同时改 topology 或 B3。

---

## 8. 一句话版执行目标

> **直接把 A1 改成：fused pseudo-frame 的 direct correspondence support 生成统一 `C_m`，再用 verified projected depth 生成 depth target，并让 RGB / depth 在 consumer 端共用这个 `C_m`。**

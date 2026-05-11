# C_m Dense Match Densify (Gaussian blur + normalize, no reprojection/depth overlap) Plan — 2026-05-10

> 目标：评估并规划一个新的、可参数开关控制的 C_m 生成模块：只基于 pseudo↔ref 的 mutual matches 做 densify + Gaussian blur + normalize，不引入 reprojection / depth overlap 约束。该模块的目的不是直接替换主线，而是回答一个更具体的问题：当前 online mapping 几乎无增益，是否主要因为 reciprocal support 过于点状，导致 pseudo supervision 有效覆盖太小。

---

## 0. 结论先说

结论：可行，而且工程落地成本不高；但只适合作为“并行 compare branch”，不适合直接替换当前默认 C_m。

原因分两部分：

1. 工程上可行
- 当前代码已经把“C_m 的 RGB-only 生成”和“depth target 的 exact backend 生成”解耦。
- 实际入口非常清楚：`runtime_exact_backend.py` 在 `rgb_only_verification=true` 时，先用 `_accumulate_match_maps(...)` 生成 `support_mask/confidence_map` 给 C_m，再单独跑 `verify_single_branch_exact(...)` 生成 `projected_depth_map` 给 depth target。
- 所以现在要加的 densify 模块，本质上只是替换/扩展 `_accumulate_match_maps(...)` 之后的 branch support construction，不必碰 depth target builder。

2. 科学上值得做，但风险明确
- 值得做：它能直接测试“当前 pseudo 没作用，是不是主要卡在 C_m coverage 太稀疏”这个假设。
- 风险：因为不加 reprojection/depth overlap，这个新 mask 会显著比当前 raw reciprocal support 更大，但增加的是“形态学扩张后的 correspondence evidence”，不是新增几何证据；所以很可能让 pseudo loss 变大、变化更明显，但不一定更好。
- 因此建议：第一版只做成 compare-only 模块，默认关闭，所有产物单独保存，不覆盖 raw reciprocal support。

一句话判断：
这是一个“很值得做的可控 ablation”，不是“安全可转正的默认升级”。

---

## 1. 当前代码事实：为什么这件事现在容易插进去

### 1.1 当前 C_m / depth target 已经分路

实际代码位置：
- `pseudo_branch/integration/runtime_exact_backend.py`
- `pseudo_branch/mask/rgb_mask_inference.py`
- `pseudo_branch/observation/pseudo_observation_brpo_style.py`

当前 `rgb_only_verification=true` 时，代码链是：

```text
pseudo+ref images
  -> matcher.match_pair(...)
  -> _accumulate_match_maps(...)           # 只负责 RGB-only support/conf
  -> verify_single_branch_exact(...)       # 只负责 projected depth / provenance / exact debug
  -> left_result/right_result merge
  -> build_exact_brpo_upstream_target_observation(...)
```

关键事实：
- `_accumulate_match_maps(...)` 当前只是把 match 点 round 到像素：
  - `support[yi, xi] = 1`
  - `conf_map[yi, xi] = max(conf)`
- 它不会做 point dilation、Gaussian blur、normalize，也不会做 dense support reconstruction。
- 所以当前 raw support 本质上还是 one-pixel reciprocal seed。

### 1.2 当前 observation builder 已经允许 soft override，但第一版不一定要用

`build_exact_brpo_upstream_target_observation(...)` 已支持：
- `confidence_cm_override`

这意味着：
- 如果后续要把 Gaussian blur 后的 soft 值直接喂进 final `C_m`，已经有现成入口。
- 但第一版不建议直接这么做，因为这会把“coverage 变化”和“loss 权重语义变化”混在一起。

因此更推荐：
- 第一阶段只改 branch support coverage；
- final C_m 仍先保持 `both->1.0 / xor->0.5 / none->0.0` 的离散 contract；
- 等确认 coverage 变化确实是关键瓶颈，再考虑 soft override 第二阶段。

---

## 2. 这个新模块和现有 `cm_local_expansion` 的本质区别

当前已有模块：
- `pseudo_branch/mask/cm_local_expansion.py`

它的语义是：
- 从 raw reciprocal seed 出发；
- 依赖 pseudo RGB + pseudo render depth 的局部连续性 gate；
- 做小半径局部扩张；
- 可以输出 soft C_m；
- 默认不扩 depth target scope。

而这次想做的新模块语义是：
- 直接对 matcher 的 sparse reciprocal points 做 densify；
- 使用 radius + Gaussian blur + normalize 形成 dense correspondence evidence；
- 暂时不使用 reprojection / depth overlap；
- 先把它当成“更大的 RGB-only branch support builder”，不是 local geometric expansion。

所以它不应该塞进 `cm_expansion_mode=local_soft_v1` 里硬改，而应该作为一个新的、平行的 support-generation mode。

这是为了避免两个概念混淆：
- `local_soft_v1`：seed 周围局部 gated expansion
- `dense_match_v1`：对 match evidence 自身先 densify 再阈值化

---

## 3. 推荐设计：先做 binary densified support，不先做 soft final C_m

### 3.1 为什么第一版不要直接上 soft C_m override

如果直接把 blur+normalize 的 soft map 作为 final `confidence_cm_override`，会同时引入两件事：

1. C_m coverage 变大
2. C_m 权重语义从离散三档变成连续 soft

这样一旦结果变化，很难判断：
- 是 coverage 增加起作用，还是 soft weighting 起作用；
- 或者是不是又出现了类似 E8 的 normalized objective 稀释问题。

所以第一版建议：
- `soft_map` 只作为 branch 内部 evidence；
- 先 threshold 成 binary `support_mask`；
- final `C_m` 仍由 observation builder 走离散 `both/xor/none`；
- `confidence_map` 可以保存为 soft map，供后续分析，但先不强依赖它进入 final loss。

### 3.2 第一版推荐 contract

对每个 branch：

```python
soft_map = dense_match_score_from_points(...)
support_mask = (soft_map >= corr_threshold).astype(np.float32)
confidence_map = soft_map.astype(np.float32)
```

然后仍然走现有合成：
- left/right binary support 进入 final discrete C_m
- projected depth 仍来自 `verify_single_branch_exact(...)`
- depth target scope 不变

这样第一版改动最小，变量也最单纯：
只改 C_m coverage，不改 depth consumer contract。

---

## 4. 可行落地方案（推荐 Phase D1）

### 4.1 新增模块文件

建议新增：

```text
pseudo_branch/mask/dense_match_densify.py
```

建议提供两个纯函数：

```python
def points_to_soft_mask(
    points_xy: np.ndarray,
    h: int,
    w: int,
    *,
    radius: int = 2,
    seed_values: np.ndarray | None = None,
    seed_mode: str = "binary",  # binary | confidence_weighted
) -> np.ndarray:
    ...
```

```python
def build_dense_match_maps(
    image_shape: tuple[int, int],
    pts_fused: np.ndarray,
    conf: np.ndarray,
    *,
    point_radius: int = 2,
    blur_sigma: float = 2.0,
    blur_kernel: int | None = None,
    normalize_mode: str = "max",
    corr_threshold: float = 0.15,
    seed_mode: str = "binary",  # binary | confidence_weighted
) -> dict[str, np.ndarray]:
    # returns:
    #   raw_support_mask
    #   raw_conf_map
    #   dense_soft_map
    #   dense_support_mask
    #   match_density
    #   summary
```

实现原则：
- `binary` seed_mode：每个点先写成 1.0，再 blur；最接近你室友当前流程。
- `confidence_weighted` seed_mode：每个点按归一化 matcher conf 写入，再 blur；作为后续可选 arm，不是第一优先。
- `normalize_mode=max`：blur 后除以全图最大值；先不要做更复杂的 percentile normalize。

### 4.2 runtime 接入点

修改文件：

```text
pseudo_branch/integration/runtime_exact_backend.py
```

在 `rgb_only_verification=true` 分支里，当前是：

```python
left_rgb_maps = _accumulate_match_maps(...)
right_rgb_maps = _accumulate_match_maps(...)
```

推荐改成：

```python
if cfg.rgb_only_support_mode == "reciprocal_seed":
    left_rgb_maps = _accumulate_match_maps(...)
    right_rgb_maps = _accumulate_match_maps(...)
elif cfg.rgb_only_support_mode == "dense_match_v1":
    left_rgb_maps = build_dense_match_maps(...)
    right_rgb_maps = build_dense_match_maps(...)
else:
    raise ValueError(...)
```

注意：
- `verify_single_branch_exact(...)` 这条 depth target 路线保持完全不变。
- 这样新模块只影响 C_m coverage，不碰 exact projected depth consumer。

### 4.3 配置项建议

在 `RuntimeExactBackendConfig` 新增：

```python
rgb_only_support_mode: str = "reciprocal_seed"  # reciprocal_seed | dense_match_v1
cm_dense_point_radius: int = 2
cm_dense_blur_sigma: float = 2.0
cm_dense_blur_kernel: int = 0   # 0 => auto from radius/sigma
cm_dense_corr_threshold: float = 0.15
cm_dense_seed_mode: str = "binary"  # binary | confidence_weighted
cm_dense_normalize_mode: str = "max"
```

推荐默认值：
- `rgb_only_support_mode="reciprocal_seed"`
- `cm_dense_point_radius=2`
- `cm_dense_blur_sigma=2.0`
- `cm_dense_corr_threshold=0.15`
- `cm_dense_seed_mode="binary"`

理由：
- 这组值最接近你室友当前做法；
- 足够产生显著 coverage 变化；
- 也方便和你室友结果做对照。

### 4.4 产物保存建议

必须保留 raw reciprocal support，不要覆盖：

```text
exact_backend_v1/
  support_left_raw_reciprocal.npy
  support_right_raw_reciprocal.npy
  support_left_dense_match_v1.npy
  support_right_dense_match_v1.npy
  confidence_left_dense_match_v1.npy
  confidence_right_dense_match_v1.npy
  dense_match_left_soft_v1.npy
  dense_match_right_soft_v1.npy
  dense_match_meta.json
```

metadata 至少记录：
- `rgb_only_support_mode`
- `point_radius`
- `blur_sigma`
- `blur_kernel`
- `corr_threshold`
- `seed_mode`
- `normalize_mode`
- `raw_union_ratio`
- `dense_union_ratio`
- `raw_both_ratio`
- `dense_both_ratio`
- `depth_target_filled_ratio`（确认不变）

---

## 5. 为什么我判断它“值得做”

这个方案对你当前主问题有直接诊断价值。

你现在真正的疑问不是“更大 mask 一定更好”，而是：

```text
为什么 online mapping 明明多加了 pseudo supervision，结果几乎和 baseline 一样？
```

这个 densify 模块可以直接测试其中一个非常具体的假设：

```text
H1: 当前 pseudo supervision 的有效影响太弱，不是因为链条没接上，而是因为 reciprocal support 太 sparse，loss 实际看到的 pseudo 区域太小。
```

如果加上 dense_match_v1 后：
- C_m union/both 显著增大；
- pseudo_loss / gradients 明显变化；
- 最终结果仍几乎不变；

那说明“瓶颈不只是 support sparsity”，而更可能在：
- loss contract
- pseudo RGB 质量
- depth target 语义
- optimization schedule / weight balance

反过来，如果 dense_match_v1 后结果明显变化，至少能说明：
- 当前 raw reciprocal seed coverage 确实过保守；
- online mapping 的 no-effect 有一部分来自 support construction，而不是链条断裂。

所以这件事是一个很好的诊断分叉点。

---

## 6. 主要风险与我对风险的判断

### 6.1 风险一：没有几何约束，mask 会明显变大，但未必更准

这是最核心风险。

因为第一版刻意不加 reprojection/depth overlap，所以新增像素只是“被某个 match 点局部扩散覆盖到”，而不是“有额外几何验证”。

这意味着：
- coverage 会更大；
- pseudo loss 会更有存在感；
- 但新增区域的真实性不如 raw reciprocal seed。

我的判断：
- 这不妨碍它作为 ablation；
- 但它绝不应该直接上默认主线。

### 6.2 风险二：如果直接 soft consume，可能重演 E8 的 objective dilution

E8 已经说明：
- 只要把更多无强锚点区域纳进 normalized masked RGB objective，原 seed 的有效梯度质量就可能被稀释。

因此我不建议第一版直接：
- `confidence_cm_override = dense_soft_map`

而是先：
- `dense_soft_map -> threshold -> binary support`
- final `C_m` 仍维持离散三档。

### 6.3 风险三：如果结果还是没变化，可能让人误以为模块没接上

这个需要靠产物流验证避免。

必须明确保存并检查：
- raw vs dense support 覆盖率
- final signal 的 C_m ratio
- pseudo loss history 是否真的变化
- depth target filled ratio 是否没变

只要这些产物在，结果就算“没提升”，也能区分是：
- 接线没生效
- 还是确实 coverage 变大了但没有带来收益

---

## 7. 推荐实施顺序

### Phase D0：sidecar 离线诊断（先不碰 runtime）

先基于已有 `brpo_debug` 目录做离线 materialize：
- 输入 raw reciprocal support 点集 / 或当前保存的 raw support
- 输出 dense soft map、dense support、summary
- 目标是先把 coverage 变化量看清楚

建议新增：

```text
scripts/diagnostics/materialize_dense_match_support.py
```

用途：
- 对 E5c / E7-depthoff 的现有 debug root 跑离线 coverage compare
- 先看 union/both 大概会放大到什么程度
- 先确认参数不要一开始过猛

### Phase D1：runtime 最小接线

接入 `runtime_exact_backend.py` 的 `rgb_only_verification=true` 分支：
- 只替换 C_m branch support builder
- exact depth target 路线不动
- final C_m 仍离散三档
- 默认关闭

这是我认为最合适的第一版正式落地。

### Phase D2：短程 compare run

只做短程 compare：
- smoke 级别：2~3 个 pseudo events
- short compare：8-frame 或少量 keyframe
- 看 coverage / pseudo loss / metrics 是否有任何实质变化

### Phase D3：只有在 D1/D2 证明 coverage 真是瓶颈后，再考虑 soft override

如果 D1/D2 显示：
- densified binary support 明显有正作用；
- 但仍觉得 blur soft 值本身有信息；

再考虑第二阶段：
- 用 `confidence_cm_override` 消费 soft map
- 但这时要明确标成另一个 mode，例如 `dense_match_soft_cm_v1`
- 不能和 D1 混在一起

---

## 8. 推荐实验臂（只做规划，不是现在就跑）

### A0: baseline
- `rgb_only_support_mode = reciprocal_seed`
- 当前 E5c / E7-depthoff 行为

### A1: peer-like densify first arm
- `rgb_only_support_mode = dense_match_v1`
- `cm_dense_point_radius = 2`
- `cm_dense_blur_sigma = 2.0`
- `cm_dense_corr_threshold = 0.15`
- `cm_dense_seed_mode = binary`

用途：最接近你室友现在的“点扩张 + blur + normalize + threshold”路线。

### A2: safer densify arm
- `point_radius = 1`
- `blur_sigma = 1.0`
- `corr_threshold = 0.20`

用途：更保守，避免 coverage 一下子膨胀太大。

### A3: confidence-weighted seed arm
- `seed_mode = confidence_weighted`

用途：测试“扩张时保留 matcher conf 强弱”是否比 binary disk 更稳。

我建议真正开始时只做：
- A0 vs A1
- 如果 A1 太猛，再补 A2

---

## 9. 成功/失败判据

### 9.1 接线成功的最低判据

不是看最终 PSNR，而是先看这几个产物事实：

1. `dense_union_ratio > raw_union_ratio`
2. `dense_both_ratio > raw_both_ratio`
3. `depth_target_filled_ratio` 基本不变
4. `final signal C_m` 确实来自 densified support，而不是 raw fallback
5. pseudo loss history 相比 baseline 有可见变化

如果这 5 条都满足，说明模块已经真的接上。

### 9.2 这个模块“值得继续”的判据

以下任意一种都算有价值：
- online mapping 指标终于和 baseline 拉开差距
- 即使指标没提升，但 pseudo loss / pose behavior 明显变化，证明 coverage 不是虚接
- 它帮助排除“当前 no-effect 只是因为 mask 太小”这个假设

### 9.3 这个模块“不该继续扩”的信号

- coverage 大幅变大，但结果持续更差
- pseudo loss 显著下降却最终渲染/ATE 更差
- 与 E8 一样出现明显 dilution pattern

这时结论应该是：
- 不是不生效；
- 而是“无几何约束的大 coverage”本身不适合作为当前 loss contract 的监督信号。

---

## 10. 具体文件改动清单（建议）

### 必改

1. `pseudo_branch/mask/dense_match_densify.py`
- 新增
- 实现 point disk -> Gaussian blur -> normalize -> threshold

2. `pseudo_branch/integration/runtime_exact_backend.py`
- 新增 config 字段
- 在 `rgb_only_verification=true` 下切换 support builder
- 写 dense-match sidecar/artifacts

### 可选但推荐

3. `scripts/diagnostics/materialize_dense_match_support.py`
- 先离线 compare raw vs dense coverage
- 降低 runtime 试错成本

### 第一版不建议动

4. `pseudo_branch/observation/pseudo_observation_brpo_style.py`
- 第一版不需要改 final loss contract
- 除非后续明确要做 soft `confidence_cm_override`

---

## 11. 最终建议

我的建议是：做，而且按下面这个方式做最稳。

推荐决策：
1. 做成新的 parallel mode，不覆盖当前 `local_soft_v1` 和 raw reciprocal support。
2. 第一版只做“densified binary branch support”，不要直接 soft final `C_m`。
3. 继续保持 `depth target = exact backend` 不变，这样这个实验只测 C_m coverage 的影响。
4. 先 sidecar，再 runtime，再短程 compare，不要直接 full run。

如果只用一句话概括：

```text
可行，且很值得做成一个受控 compare branch；
但第一版必须只测试“coverage 变大”本身，不要把 soft weighting、depth overlap、projection overlap 全部一次性混进去。
```

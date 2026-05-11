# C_m Controlled Local Expansion Plan (r1/r2 soft) — 2026-05-08

> 目标：在不放弃 BRPO paper 的 mutual-nearest-neighbor / reciprocal seed 语义前提下，做一个可控、可回滚、可度量的局部 C_m 扩张实验。核心不是把 pseudo supervision 变成全图，而是把当前过于点状的 reciprocal support 在局部一致区域内小幅扩开，并给扩张区域更低权重。

---

## 0. 当前事实与动机

当前 E5c 产物显示：

- `matcher_mode = dense_pts3d_3d`
- `dense3d_conf_quantile = 0.15`
- artifact meta 中 `candidate_ratio_query ≈ 1.0`，说明 MASt3R/pointmap 候选阶段已经接近全图，不是 quantile 阈值把像素筛没了。
- 但最终 `C_m` 仍是 reciprocal match seed 的像素集合：
  - C_m union median ≈ 26.2%
  - C_m both median ≈ 4.0%
  - C_m single median ≈ 22.2%
  - projected depth union median ≈ 11.3%

解释：MASt3R 前向和候选是 dense 的，但我们消费方式是：dense candidate → 3D reciprocal NN → 单像素 support。这个支持域仍然偏“seed-like”，不是 dense supervision mask。

本方案要解决的是：reciprocal seed 过于点状、both 逐像素交集过低、pseudo RGB/depth supervision 的有效区域不足。

---

## 1. 设计原则

1. reciprocal seed 不删除。它仍然是 paper-aligned 的高可信证据源。扩张只能围绕 seed 发生。
2. 默认关闭，不影响 E5/E6 等现有配置与运行目录。
3. 先做 sidecar 离线诊断，再接 runtime。不能一上来改主 pipeline。
4. 扩张区域必须低权重，不允许直接把扩张像素当成原始 both=1.0 seed。
5. 必须保留 provenance：raw seed 与 expanded pixel 分开保存、分开统计。
6. 扩张只允许在局部 RGB/depth 连续区域内发生，避免跨物体边缘和明显错误区域。

---

## 2. 推荐方案：r1/r2 local soft expansion

### 2.1 输入

对每个 pseudo frame、每个 branch(left/right)：

- `support_left_exact.npy` / `support_right_exact.npy`：raw reciprocal support seed
- `confidence_left_exact.npy` / `confidence_right_exact.npy`：seed confidence map；非 seed 区域通常为 0
- `runtime_inputs/pseudo_fused_rgb.png` 或 `pseudo_render_rgb_runtime.png`：实际用于 matching/supervision 的 pseudo RGB
- `runtime_inputs/pseudo_render_depth_runtime.npy`：当前 map 在 pseudo view 的 render depth，用作局部表面连续性 gate
- optional：`fusion/confidence_mask_fused.npy` 或 overlap confidence，作为额外局部可靠性 gate；第一版可不强依赖

注意：当前 matcher 没有把 full MASt3R dense confidence map 直接输出给 C_m expansion。因此第一版不要假设可在非 seed 像素上读到 MASt3R confidence；扩张 confidence 应从 seed confidence + 局部一致性衰减得到。若后续需要 full conf，可另加 matcher 输出字段。

### 2.2 单 branch 扩张规则

对每个 raw seed 像素 `p`，扫描半径 `r ∈ {1, 2}` 的邻域候选 `q`：

- `r=1` 对应 3x3，是首选 safe arm。
- `r=2` 对应 5x5，是 stronger diagnostic arm。

候选 `q` 被接受需要同时满足：

1. seed confidence 有效：`seed_conf[p] >= seed_conf_min`。建议第一版 `seed_conf_min=0.0`，因为当前 confidence 已只在 match seed 上有值；如果噪声明显，再提高。
2. RGB 连续：`mean_abs(rgb[q] - rgb[p]) <= tau_rgb_l1`。建议初值 `tau_rgb_l1=0.08`，RGB 在 `[0,1]`。
3. depth 连续：`abs(depth[q] - depth[p]) / max(depth[p], depth[q], eps) <= tau_depth_rel`。建议初值 `tau_depth_rel=0.05`；如果 depth render 噪声大，可放宽到 `0.08`。
4. depth 有效：`depth[p] > 1e-6` 且 `depth[q] > 1e-6`。如果 depth render 在某些区域缺失，第一版不要跨缺失扩张。

候选分数：

```python
spatial = exp(- dist(p, q) / max(radius, 1))
rgb_score = max(0, 1 - rgb_l1 / tau_rgb_l1)
depth_score = max(0, 1 - rel_depth / tau_depth_rel)
expanded_branch_conf[q] = max(
    expanded_branch_conf[q],
    seed_conf[p] * spatial * rgb_score * depth_score * expansion_weight
)
```

建议初值：

```yaml
cm_expansion_radius: 1
cm_expansion_weight: 0.5
cm_expansion_tau_rgb_l1: 0.08
cm_expansion_tau_depth_rel: 0.05
cm_expansion_min_expanded_conf: 0.05
```

branch final support：

```python
branch_support_expanded = raw_support | (expanded_branch_conf >= min_expanded_conf)
```

但 final C_m 不能只用 binary support 直接 both=1/xor=0.5，否则扩张像素被过度信任。必须走 soft C_m composition。

---

## 3. Soft C_m composition

### 3.1 区分 raw 与 expanded

对 left/right branch 分别有：

- `raw_left`, `raw_right`
- `expanded_left`, `expanded_right`，包含 raw seed + local expansion
- `new_left = expanded_left & ~raw_left`
- `new_right = expanded_right & ~raw_right`

### 3.2 推荐 final confidence 规则

保留 raw BRPO 三档：

```python
C_m[raw_left & raw_right] = 1.0
C_m[raw_left ^ raw_right] = 0.5
```

对扩张区域低权重：

```python
expanded_both = expanded_left & expanded_right & ~(raw_left & raw_right)
expanded_single = (expanded_left ^ expanded_right) & ~(raw_left | raw_right)
raw_plus_expanded_single = ((raw_left & new_right) | (raw_right & new_left))

C_m[expanded_both] = max(C_m, cm_expanded_both_weight)      # 建议 0.6 或 0.75
C_m[raw_plus_expanded_single] = max(C_m, cm_raw_exp_agree_weight)  # 建议 0.5 或 0.6
C_m[expanded_single] = max(C_m, cm_expanded_single_weight)  # 建议 0.25
```

建议第一版参数：

```yaml
cm_expanded_both_weight: 0.6
cm_raw_exp_agree_weight: 0.5
cm_expanded_single_weight: 0.25
```

解释：

- raw both 仍是最强 1.0。
- raw single 仍是 paper 原始 0.5。
- expanded both 低于 raw both，因为它不是真实 reciprocal seed 逐像素重合。
- expanded single 只给弱监督，避免“多了但错了”。

### 3.3 为什么不能只用现有 support_left/right 传给旧 builder

现有 `build_exact_brpo_upstream_target_observation()` 会把 support sets 硬编码成：

- both → 1.0
- xor → 0.5
- none → 0.0

如果直接把 expanded support 传进去，扩张区域会被当成原始 paper seed，风险过高。因此落地时需要新增一个 optional `confidence_cm_override` / `cm_override` 入口，或新增 soft expansion observation builder。

---

## 4. 工程落地文件

### 4.1 新增模块

建议新增：

```text
pseudo_branch/mask/cm_local_expansion.py
```

包含纯函数：

```python
def expand_branch_support_local(
    raw_support: np.ndarray,
    seed_confidence: np.ndarray,
    pseudo_rgb: np.ndarray,
    pseudo_depth: np.ndarray,
    *,
    radius: int = 1,
    expansion_weight: float = 0.5,
    tau_rgb_l1: float = 0.08,
    tau_depth_rel: float = 0.05,
    min_seed_conf: float = 0.0,
    min_expanded_conf: float = 0.05,
) -> dict:
    # returns raw_support, expanded_support, expanded_only, expanded_confidence, summary
```

```python
def compose_soft_cm_from_expanded_branches(
    raw_left: np.ndarray,
    raw_right: np.ndarray,
    expanded_left: np.ndarray,
    expanded_right: np.ndarray,
    *,
    raw_both_weight: float = 1.0,
    raw_single_weight: float = 0.5,
    expanded_both_weight: float = 0.6,
    raw_exp_agree_weight: float = 0.5,
    expanded_single_weight: float = 0.25,
) -> dict:
    # returns confidence_cm, support/provenance maps, summary
```

```python
def apply_cm_local_expansion(...):
    # convenience wrapper for left/right branch expansion + final C_m composition
```

### 4.2 新增 sidecar 诊断脚本

建议新增：

```text
scripts/diagnostics/materialize_cm_local_expansion.py
```

输入：已有 `brpo_debug` root，例如：

```bash
/home/bzhang512/miniconda3/envs/s3po-gs/bin/python \
  scripts/diagnostics/materialize_cm_local_expansion.py \
  --debug-root /data3/bzhang512/part3_online_mapping_experiments/E5c_jointprimary_maskedcolor_rgbonly_cm_difix/brpo_debug \
  --radius 1 \
  --tau-rgb-l1 0.08 \
  --tau-depth-rel 0.05 \
  --out-name cm_local_expand_r1_v1
```

输出到每个 frame root 下的 sidecar 目录，不覆盖旧产物：

```text
frame_xxxx/cm_local_expand_r1_v1/
  cm_raw.npy
  cm_expanded_soft.npy
  support_left_raw.npy
  support_right_raw.npy
  support_left_expanded.npy
  support_right_expanded.npy
  support_left_expanded_only.npy
  support_right_expanded_only.npy
  expansion_provenance.npy
  summary.json
```

全局汇总：

```text
<brpo_debug>/cm_local_expand_r1_v1_summary.json
```

summary 必须包含：

- raw cm union / both / single ratio
- expanded cm nonzero ratio
- expanded both / raw+expanded agree / expanded single ratio
- mean positive C_m before/after
- effective mask weight before/after：`C_m.sum() / num_pixels`
- left/right expanded-only ratio
- expansion reject reason counts：rgb fail / depth fail / invalid depth / low conf
- projected depth target filled ratio 不变，用于证明这个 arm 没改 depth target

---

## 5. Runtime 接入点

### 5.1 配置开关

在 `Results.brpo_online_mapping` 下新增，默认关闭：

```yaml
cm_expansion_mode: none   # none | local_soft_v1
cm_expansion_radius: 1
cm_expansion_weight: 0.5
cm_expansion_tau_rgb_l1: 0.08
cm_expansion_tau_depth_rel: 0.05
cm_expansion_min_seed_conf: 0.0
cm_expansion_min_expanded_conf: 0.05
cm_expanded_both_weight: 0.6
cm_raw_exp_agree_weight: 0.5
cm_expanded_single_weight: 0.25
cm_expansion_apply_to_depth_scope: false
```

`cm_expansion_apply_to_depth_scope=false` 是第一版推荐值：先只改变 final `confidence_mask/C_m` 的 soft 权重；depth target 仍由 projected depth 是否存在决定。若后续配合 2img+PAIR calibrated depth，再开一个明确 arm 测试 depth scope expansion。

### 5.2 配置解析

修改：

```text
/home/bzhang512/CV_Project/third_party/S3PO-GS/utils/slam_backend.py
```

在 `_resolve_brpo_online_mapping_cfg()` 里读取上述字段，并传入 `RuntimeExactBackendConfig`。

### 5.3 RuntimeExactBackendConfig

修改：

```text
pseudo_branch/integration/runtime_exact_backend.py
```

在 dataclass 中加入 `cm_expansion_*` 字段。

插入位置：

- 已完成 MASt3R matching
- 已得到 `left_result/right_result` 的 raw `support_mask` 与 `confidence_map`
- 写 `exact_backend_meta.json` 前

不要覆盖 raw 文件。建议在 `exact_backend_v1/` 里额外保存：

```text
support_left_raw_reciprocal.npy
support_right_raw_reciprocal.npy
support_left_cm_expanded.npy
support_right_cm_expanded.npy
confidence_cm_local_soft_v1.npy
cm_expansion_meta.json
```

### 5.4 Signal builder / observation builder

当前 `build_exact_brpo_upstream_target_observation()` 内部固定由 support sets 生成 discrete C_m。为了 soft expansion，需要新增可选参数：

```python
confidence_cm_override: np.ndarray | None = None
cm_override_semantics: str | None = None
```

逻辑：

```python
if confidence_cm_override is not None:
    confidence_cm = np.asarray(confidence_cm_override, dtype=np.float32)
else:
    confidence_cm = discrete both/xor from support_left/support_right
```

注意：depth target builder 仍可使用 raw support 或 expanded support。第一版推荐：

- C_m / RGB confidence 使用 soft expanded `confidence_cm_override`
- depth target scope 仍使用 raw support unless `cm_expansion_apply_to_depth_scope=true`

这需要在 `runtime_signal_builder.py` 里明确传参：

- `support_left/right_for_depth`: raw support
- `confidence_cm_override`: expanded soft C_m

如果为了最小改动，也可以第一版先只在 sidecar 里做统计，不接 runtime；接 runtime 时不要偷懒把 expanded support 直接塞进旧 builder。

---

## 6. 实验 arms

### 6.1 Sidecar-only 诊断

先在 E5c 已有 9 个 frame 上跑 sidecar：

- `r1_soft`: radius=1
- 可选 `r2_soft`: radius=2

通过条件：

- expanded effective weight 增加但不过大；建议 first arm 增加 1.2x–1.8x，而不是 3x+。
- expanded single 不应压倒 raw single 太多。
- 扩张主要贴近 raw seed，不应出现大面积块状外溢。
- projected depth target filled ratio 必须不变。

### 6.2 Runtime smoke

新 config，不改 E5c：

```text
configs/e7c_cm_local_expand_r1_soft.yaml
```

新 save_dir：

```text
/data3/bzhang512/part3_online_mapping_experiments/E7c_cm_local_expand_r1_soft
```

短 smoke 只跑少量 keyframe，检查：

- `cm_expansion_meta.json` 存在
- raw/expanded ratio 正确
- `exact_backend_meta.json` 记录 `cm_expansion_mode=local_soft_v1`
- signal meta 记录 `confidence_cm_override=true`
- `pseudo_confidence_exact_brpo_upstream_target_v1.npy` 的非零比例和 sum/weight 确实改变
- projected depth target filled ratio 未因这个 arm 被悄悄扩大

### 6.3 Formal compare

若 smoke 正常，再排正式 compare：

- Baseline：E5c 或同等 no-expansion config
- E7c：r1 soft expansion
- E7d：r2 soft expansion，仅作 diagnostic，不一定作为主线

如果同时接 2img+PAIR calibrated depth，应另开 arm，不能和 C_m expansion 第一次混在一起，否则无法解释增益来源。

---

## 7. 关键风险与防护

### 风险 1：扩张区域“多了但错了”

防护：

- r1 优先，不先上 r2/r3。
- RGB + depth 双 gate。
- 扩张区域低权重。
- summary 中必须报告 expanded-only 比例和 effective mask weight。

### 风险 2：扩张像素被旧 builder 当成 raw both=1.0

防护：

- 不允许直接把 expanded support 塞给旧 `both/xor` builder。
- 必须使用 `confidence_cm_override` 或新 soft builder。

### 风险 3：不小心扩大 depth loss scope

防护：

- 第一版 `cm_expansion_apply_to_depth_scope=false`。
- projected depth target filled ratio 必须保持 baseline 相同。
- 若后续配合 2img+PAIR depth，单独命名为 `cm_expand_plus_twoimg_depth_v1`。

### 风险 4：当前 pseudo RGB 质量差导致扩张跟着 hallucination 外溢

防护：

- 和 E6 GT-pseudo/no-Difix upper-bound 对照；如果 GT pseudo 下扩张更健康，说明 RGB target 质量仍是关键瓶颈。
- 可选增加 `fusion_confidence` gate。

---

## 8. Claude 执行顺序

1. 不改现有 E5/E6 配置和输出目录。
2. 新增 `pseudo_branch/mask/cm_local_expansion.py`，写纯函数与小单元 smoke。
3. 新增 `scripts/diagnostics/materialize_cm_local_expansion.py`，先 sidecar 跑 E5c 已有 frames。
4. 读 sidecar summary，确认 r1 expansion 比例合理。
5. 只有 sidecar 合格后，才修改 runtime：
   - `RuntimeExactBackendConfig`
   - `slam_backend.py` config resolver
   - `runtime_exact_backend.py` 产出 expansion side products
   - `runtime_signal_builder.py` / `pseudo_observation_brpo_style.py` 支持 `confidence_cm_override`
6. 新建 config/save_dir 做短 smoke。
7. smoke 通过后再排 formal compare。

---

## 9. 最小验收标准

实现后必须能回答这几个问题：

1. raw C_m union/both/single 是多少？
2. expanded C_m nonzero ratio 是多少？
3. effective mask weight 增加了多少倍？
4. expanded-only 的 pixel 占比是多少？
5. projected depth target filled ratio 是否保持不变？
6. final loss 入口读到的是 raw C_m 还是 expanded soft C_m？
7. 扩张区域是低权重，还是被错误当成 both=1.0？

如果这些不能从 metadata/summary 直接读出，就视为落地不合格。

---

## 10. 一句话结论

推荐先做 `reciprocal seed + r1 local RGB/depth-gated soft expansion`。它保留 BRPO paper 的 mutual NN 可信 seed 语义，同时以低权重、可统计、可回滚的方式缓解当前 C_m 过于点状和 both 过低的问题。第一版不要直接全图 dense，也不要只调 MASt3R quantile。

# BRPO_MASK_MAST3R_3D_MATCHING_PLAN_20260424

> 更新时间：2026-04-24 04:38 CST
> 目标：把当前 exact BRPO mask 的匹配入口升级为 `MASt3R 3D reciprocal matching`，但仍然保持最终 mask 语义是 exact BRPO 的离散三档 `1.0 / 0.5 / 0.0`。

---

## 1. 当前 live 路径与为什么要做 3D matching

当前 live 路径虽然已经用了 MASt3R，但只是把它当作 descriptor backbone：
- `pseudo_branch/common/flow_matcher.py` 只取 `desc` / `desc_conf`
- 再用 `fast_reciprocal_NNs(..., subsample=8)` 取 sparse 2D matches
- 下游 exact verify 再用 rendered depth 做几何筛选

这条路的问题不是 verify 规则错，而是匹配入口太稀，而且仍然是 2D descriptor-level reciprocal matching。MASt3R 本身还给了更强的东西：
- `pred1['pts3d']`
- `pred2['pts3d_in_other_view']`
- `pred1['conf']`
- `pred2['conf']`

其中 `pred1['pts3d']` 和 `pred2['pts3d_in_other_view']` 已经被放在同一参考坐标系里，仓库内 `dust3r.utils.geometry.find_reciprocal_matches()` 与 `mast3r/colmap/database.py` 也已经提供了现成的 3D reciprocal matching 参考。也就是说，这条路线并不是“另起炉灶”，而是把当前 repo 已经具备、但还没接到 part3_BRPO live pipeline 的能力接进来。

---

## 2. 方案选择结论

这一版 3D matching 的具体方案建议是：

- 模型：仍用现有 `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`
- 3D表征：
  - query 侧：`pred1['pts3d']`
  - ref 侧：`pred2['pts3d_in_other_view']`
- 候选筛选：使用 `pred['conf']` 做 quantile filter
- reciprocal 匹配：`dust3r.utils.geometry.find_reciprocal_matches(P1, P2)`
- 最终输出仍然回到 2D 像素点对 `pts_query_xy / pts_ref_xy`
- 这些点对再送入现有 `verify_single_branch_exact()`
- 再由 `build_brpo_confidence_mask()` 融成离散 BRPO `C_m`

这条路线的关键优点是：它把 “match generator” 从 2D descriptor 提升到 MASt3R 自身输出的 3D pointmap，但保留了 part3 现有 exact verify 和 BRPO 三档 `C_m` contract，因此不会把整个 pipeline 一次性推倒重来。

---

## 3. 目标语义：哪些不改，哪些要改

### 3.1 不改的部分

以下部分全部保持：
- `verify_single_branch_exact()` 的几何验证规则
- `build_brpo_confidence_mask()` 的 BRPO 三档离散规则
- `exact_brpo_upstream_target_v1` / `exact_shared_cm_v1` 的 consumer contract
- `both=1.0 / xor=0.5 / none=0.0`

### 3.2 要改的部分

只替换 branch support 的匹配入口：
- 旧：稀疏 2D descriptor reciprocal matches
- 新：MASt3R 3D reciprocal matches，经 2D 像素索引回映射后再进 exact verify

---

## 4. 推荐实现形态

### 4.1 与 2D dense 方案共享同一 forward helper

建议复用同一层公共 helper，不要给 3D 方案再开第二套 model wrapper：

1. `pseudo_branch/common/mast3r_pair_forward.py`
2. `pseudo_branch/common/mast3r_matchers.py`

原因：
- 2D dense 和 3D matching 都需要 MASt3R pair forward
- `desc` / `desc_conf` / `pts3d` / `conf` 都应来自同一个 bundle
- 这样后续可以通过 `--matcher-mode` 在 live script 内切换，而不是写两套 parallel pipeline

### 4.2 matcher 类建议

在 `pseudo_branch/common/mast3r_matchers.py` 中加入：

```python
class Dense3DMatcher(BasePairMatcher):
    def match_pair(self, img1_path: str, img2_path: str, size: int = 512):
        ...
```

接口保持和旧 matcher 一致：

```python
pts1_xy, pts2_xy, match_conf
```

这样下游的 `rgb_mask_inference.py` 和 `verify_single_branch_exact()` 都能直接复用。

---

## 5. MASt3R 3D matching 的具体算法

### 5.1 pair forward

对 `(pseudo_rgb, ref_rgb)` 做一次 MASt3R inference，取出：
- `pts3d_1 = pred1['pts3d']`
- `pts3d_2_in_1 = pred2['pts3d_in_other_view']`
- `conf1 = pred1['conf']`
- `conf2 = pred2['conf']`

这一步的关键理解是：
- `pts3d_1` 与 `pts3d_2_in_1` 已处于同一坐标系
- 因此可以直接在 3D 空间做 nearest-neighbor reciprocal matching

### 5.2 候选筛选

第一版建议用 `conf` 的 quantile filter，而不是绝对阈值：

```text
dense3d_conf_mode = quantile
dense3d_conf_quantile = 0.90
```

对两个视角分别：
- `mask1 = conf1 >= quantile(conf1, q)`
- `mask2 = conf2 >= quantile(conf2, q)`

然后：
- 取出被保留像素的 2D 坐标 `xy1_sel / xy2_sel`
- 取出对应 3D 点 `pts3d_1_sel / pts3d_2_sel`

### 5.3 reciprocal 3D matching

直接调用现有 DUSt3R 函数：

```python
reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(
    pts3d_1_sel,
    pts3d_2_sel,
)
```

然后映射回 2D 坐标：

```python
pts_query_xy = xy1_sel[nn2_in_P1][reciprocal_in_P2]
pts_ref_xy = xy2_sel[reciprocal_in_P2]
```

### 5.4 match confidence

第一版建议直接用 3D confidence 的几何平均：

```python
match_conf = sqrt(conf1_sel[nn2_in_P1][reciprocal_in_P2] * conf2_sel[reciprocal_in_P2])
```

原因：
- `verify_single_branch_exact()` 真正决定 support 的还是 reprojection / depth thresholds
- 这一层只需要提供一个可解释的 match-quality 诊断量

---

## 6. 具体代码落点

### 6.1 新文件

1. `pseudo_branch/common/mast3r_pair_forward.py`
   - 与 dense2d 方案共享
   - 负责 model load / inference / bundle cache

2. `pseudo_branch/common/mast3r_matchers.py`
   - 新增 `Dense3DMatcher`
   - 实现 3D candidate selection + reciprocal 3D match + 2D coordinate back-mapping

### 6.2 修改文件

1. `pseudo_branch/common/__init__.py`
   - 导出 `Dense3DMatcher` 与 `build_pair_matcher`

2. `scripts/build_brpo_v2_signal_from_internal_cache.py`
   - 把 `FlowMatcher()` 换成 `build_pair_matcher(...)`
   - 新增 CLI：
     - `--matcher-mode {sparse_desc_2d,dense_desc_2d,dense_pts3d_3d}`
     - `--dense3d-conf-quantile`

3. `scripts/brpo_build_mask_from_internal_cache.py`
   - 同步替换 matcher
   - 同步接入 CLI 参数
   - `run_branch()` 下游 verify 不动

4. `pseudo_branch/mask/rgb_mask_inference.py`
   - 保持 `matcher.match_pair()` 接口兼容
   - metadata 增加：
     - `matcher_mode`
     - `dense3d_conf_quantile`
     - `num_candidate_pixels_left/right`
     - `num_reciprocal_matches_left/right`

5. `pseudo_branch/common/flow_matcher.py`
   - 保留为 legacy sparse matcher，不删

---

## 7. Pipeline 中的调用位置

3D matching 的调用位置必须仍然在 per-branch support 生成层，而不是在 left/right 融合之后。

### 7.1 Signal v2 路径

位置：`scripts/build_brpo_v2_signal_from_internal_cache.py`

顺序：
1. `fused_rgb` 对 `left_ref_rgb` 做 MASt3R 3D reciprocal matching
2. `fused_rgb` 对 `right_ref_rgb` 做 MASt3R 3D reciprocal matching
3. 得到 per-branch support seed
4. 再进入当前 `build_brpo_style_observation()` / `build_exact_brpo_upstream_target_observation()` 逻辑

### 7.2 exact backend / old verification 路径

位置：`scripts/brpo_build_mask_from_internal_cache.py`

顺序：
1. `pseudo_rgb` 或 branch pseudo 对 ref RGB 做 MASt3R 3D reciprocal matching
2. 生成 `pts_pseudo / pts_ref`
3. 进入 `verify_single_branch_exact()`
4. 输出 `support_mask`
5. 再用 `build_brpo_confidence_mask()` 融成离散 `C_m`

关键点：3D matcher 只替换 pair match generator；exact verify 和 BRPO 三档 `C_m` 不改。

---

## 8. 输入 / 输出规划

### 8.1 输入

保持现有 pipeline 输入不变：
- pseudo/fused RGB
- left/right ref RGB
- pseudo depth
- ref depth render
- camera states / stage ply / fusion weight / overlap mask

### 8.2 新增配置输入

建议新增：

```text
--matcher-mode sparse_desc_2d|dense_desc_2d|dense_pts3d_3d
--matcher-size 512
--dense3d-conf-quantile 0.90
```

第一版不建议加太多次级 knob；先把 quantile 跑通，再看是否需要：
- `--dense3d-max-points-per-image`
- `--dense3d-min-confidence`

### 8.3 输出

主输出文件名不改，避免打断 consumer：
- `pseudo_confidence_exact_brpo_upstream_target_v1.npy`
- `pseudo_depth_target_exact_brpo_upstream_target_v1.npy`
- `support_left/right`
- `confidence_mask_brpo*`

metadata 必须新增：
- `matcher_mode=dense_pts3d_3d`
- `dense3d_conf_quantile`
- `num_candidate_points_left/right`
- `num_reciprocal_matches_left/right`
- `num_support_left/right`

建议额外写入 diag：
- `diag/dense3d_conf_left.png`
- `diag/dense3d_conf_right.png`
- `diag/dense3d_candidate_mask_left.png`
- `diag/dense3d_candidate_mask_right.png`

---

## 9. 单帧 smoke 结论（用于决定优先级）

基于 frame 23 的 live smoke：
- 当前 sparse exact：`cm_nonzero_ratio ≈ 0.01644`
- MASt3R 3D reciprocal matching + `conf q=0.90`：`cm_nonzero_ratio ≈ 0.05763`

同一帧上，3D matching 对 coverage 的提升明显高于 dense 2D。

对比同一帧 smoke：
- sparse：`~1.64%`
- dense2d：`~3.71%`
- dense3d：`~5.76%`

需要特别说明：这里的 `~5%` 不是“MASt3R 3D matching 只能覆盖 5%”，也不是调用错误导致的假低值；它表示的是：
- 先按 `conf q=0.90` 只保留每张图 top 10% 的 3D高置信像素
- 再在这个 candidate 子集上做 3D reciprocal matching
- 再经过 `verify_single_branch_exact()` 的 reprojection / relative-depth 严格筛选
- 最后再把 left/right branch 融成 exact BRPO 的 `both=1.0 / xor=0.5 / none=0.0`

因此 `q=0.90` 本身就是一个保守 smoke 配置，而不是 full-dense coverage。后续补充 sweep 显示，在 Re10k frame 23 / 57 上，fused exact `cm_nonzero_ratio` 大致会随 quantile 放宽而单调上升：
- `q=0.95`：约 `0.02–0.03`
- `q=0.90`：约 `0.05–0.06`
- `q=0.80`：约 `0.12`
- `q=0.70`：约 `0.18–0.19`
- `q=0.50`：约 `0.30`

这说明当前主要变量不是 MASt3R 调用坏了，而是 candidate-pruning 的保守程度；如果目标是正式实验而不是 wiring smoke，必须把 `dense3d_conf_quantile` 当作一等实验轴。

所以如果目标是优先突破“raw exact BRPO support 域过小”，3D 方案是更值得优先做的主线。

---

## 10. 为什么 3D 方案更适合作为主线

原因不是“更复杂所以更高级”，而是它更接近当前问题本身：
- 当前瓶颈是 support domain 太小，同时 2D descriptor reciprocal match 对 viewpoint / texture / occlusion 更敏感
- MASt3R 本身已经给出 3D pointmap 与 confidence
- 3D reciprocal matching 更接近“几何一致性先行”的 match generator
- 后面再接 `verify_single_branch_exact()`，就形成了“3D reciprocal candidate + exact reprojection/depth verify”的双重过滤

也就是说，这一版虽然最终仍是 exact BRPO 三档 `C_m`，但 support seed 已经从“2D sparse descriptor”升级到了“MASt3R 3D pointmap”。

---

## 11. 建议执行顺序

### Phase M3D-1：shared pair forward helper
- 先落 `mast3r_pair_forward.py`
- 这一步与 dense2d 共用

### Phase M3D-2：dense3d matcher 接线
- 在 `mast3r_matchers.py` 加入 `Dense3DMatcher`
- 两个 live script 接入 `--matcher-mode=dense_pts3d_3d`

### Phase M3D-3：single-frame exact smoke
- 先只跑 frame 23
- 第一轮不要只测一个 `q=0.90`，而是至少测 `q ∈ {0.95, 0.90, 0.80, 0.70}`
- 记录：
  - `support_ratio_left/right`
  - `support_ratio_both/single`
  - `cm_nonzero_ratio`
  - `mean_reproj_error`
  - `mean_rel_depth_error`
- 决策规则：如果 coverage 提升主要来自 single 爆炸、而 both 与误差质量明显变坏，则不要直接把更低 quantile 带进正式 compare

### Phase M3D-4：8-frame exact signal smoke
- 跑当前 T4 full root 的 8 帧
- 检查 coverage 提升是否稳定
- 检查是否有某些帧出现 massive false positive 或 support collapse
- 把 `dense3d_conf_quantile` 作为正式 sweep 轴保留下来，至少比较 `0.90 / 0.80 / 0.70` 三档，而不是一上来锁死 `0.90`

### Phase M3D-5：tiny consumer smoke
- 用当前 `exact_brpo_upstream_target_v1 + exact_shared_cm_v1` 做极小训练 smoke
- 先确认 consumer 不炸，再进正式 compare

### Phase M3D-6：正式 compare
- `dense_pts3d_3d` 对当前 sparse exact winner
- 正式 compare 不应只跑一个 quantile；至少把 `dense3d_conf_quantile` 当作主实验轴之一
- compare 时固定：
  - same frame ids
  - same clean summary G~
  - same T1
  - same exact_shared_cm_v1
- 推荐第一轮 formal arms：`sparse control` + `dense3d_q0.90` + `dense3d_q0.80` + `dense3d_q0.70`

---

## 12. 风险与控制

主要风险：

1. 3D pointmap reciprocal match 对 scale / outlier 较敏感
- 控制：先用 `conf quantile` 做 candidate pruning
- 之后仍必须经过 `verify_single_branch_exact()`

2. union coverage 上升，但 both 区域不一定同步上升
- 控制：compare 时不能只看 `cm_nonzero_ratio`，必须同时看 `both/single` 拆分

3. 3D reciprocal match 可能在某些重复结构区域引入错误 3D nearest neighbors
- 控制：保留 exact reprojection + relative depth threshold 的二次验证，不允许“3D reciprocal 命中即通过”

---

## 13. 本方案的定位

MASt3R 3D matching 方案应被视为当前 mask 升级的主线候选，而不是远期附加实验。

原因：
- 它直接利用了当前环境里已经存在但未接入 live pipeline 的 MASt3R 3D输出
- 与 dense2d 相比，它在单帧 smoke 上给出了更大的 support 域提升
- 同时又不破坏 exact BRPO 的离散 `C_m` 语义

建议口径：
- dense2d 是低风险升级 control / side option
- dense3d 是当前更值得优先落地的主路线

---

## 14. 下一步执行口径

如果继续执行本方案，严格按下面顺序做：

1. 落 shared pair forward helper
2. 落 `dense_pts3d_3d` matcher
3. 接到两个 live script
4. 先过 single-frame / 8-frame exact smoke
5. 再做 tiny consumer smoke
6. 最后才进入 formal compare

在这之前，不要改下游 `C_m` 三档规则，不要改 `exact_shared_cm_v1` consumer contract，也不要把 3D matcher 直接与新的 continuous mask 方案绑在一起。第一版只解决“更好的 support seed”，不混入第二个变量。

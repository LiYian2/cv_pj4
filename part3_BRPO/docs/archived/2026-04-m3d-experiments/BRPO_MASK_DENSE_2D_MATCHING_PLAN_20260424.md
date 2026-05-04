# BRPO_MASK_DENSE_2D_MATCHING_PLAN_20260424

> 更新时间：2026-04-24 04:38 CST
> 目标：把当前 exact BRPO mask 的匹配入口从 `sparse 2D MASt3R reciprocal matches` 升级为 `dense 2D descriptor matching`，但保持下游 `C_m ∈ {1.0, 0.5, 0.0}` 的 exact BRPO 三档离散语义不变。

---

## 1. 当前 live 路径与问题定位

当前 live mask / signal 路径仍然是：

1. `scripts/build_brpo_v2_signal_from_internal_cache.py`
   - 在 `build_rgb_mask_from_correspondences()` 前实例化 `FlowMatcher()`
   - 用 `matcher.match_pair(fused_rgb, left_ref)` 和 `matcher.match_pair(fused_rgb, right_ref)` 生成 `support_left/right`
2. `scripts/brpo_build_mask_from_internal_cache.py`
   - `run_branch()` 内部直接 `matcher.match_pair(pseudo_rgb, ref_rgb)`
   - 输出 `pts_pseudo / pts_ref` 后交给 `verify_single_branch()` 或 `verify_single_branch_exact()`
3. `pseudo_branch/common/flow_matcher.py`
   - 用 MASt3R descriptor + `fast_reciprocal_NNs(..., subsample_or_initxy1=8)`
   - 本质是稀疏 2D reciprocal descriptor matching
4. `pseudo_branch/mask/brpo_confidence_mask.py`
   - 只负责把 `support_left/right` 融成 BRPO 三档 `both=1.0 / xor=0.5 / none=0.0`
   - 这里的离散语义本身不用改

问题不在 `C_m` 三档规则，而在 support seed 太稀。当前 `size=512 + subsample=8` 时，每个 branch 的 seed 上限接近 `64×64=4096`，所以 raw exact `C_m` 覆盖率自然落在 `~1.5%–2%`。

---

## 2. 方案选择结论

这一版 2D dense matching 不建议引入新库。建议直接复用仓库里已经存在的 MASt3R / DUSt3R 能力，采用：

- 模型：`naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`
- 特征：`pred['desc']`
- 置信度：`pred['desc_conf']`
- reciprocal matching：`mast3r.fast_nn.bruteforce_reciprocal_nns(..., dist="dot", block_size=2**13)`

原因很直接：
- 现有环境已经有 `mast3r` / `dust3r` / `roma`，没有现成 `LoFTR` / `LightGlue` / `Kornia`
- 当前 sparse 路径本来就基于 MASt3R，继续沿用同一模型最容易平滑接入现有 pipeline
- `mast3r/colmap/database.py` 已经给出了 non-sparse descriptor matching 的现成参考写法
- 这样升级的是“采样策略与筛选策略”，不是换整套匹配栈，工程风险最低

不选 OpenCV dense optical flow 作为这一版主方案。原因不是不能做，而是它更像 motion field，不是 reciprocal correspondence + confidence-aware matching，跟当前 BRPO support-set 语义对接不够自然。

---

## 3. 目标语义：哪些不改，哪些要改

### 3.1 不改的部分

以下语义保持原样：
- per-branch exact verify 仍走 `verify_single_branch_exact()`
- left/right support 仍先各自生成，再用 `build_brpo_confidence_mask()` 融合
- 最终 `C_m` 仍然严格是：
  - `both -> 1.0`
  - `xor -> 0.5`
  - `none -> 0.0`
- `exact_brpo_upstream_target_v1` 与 `exact_shared_cm_v1` 的 consumer contract 不改

### 3.2 要改的部分

只替换“match pair 的产生方式”：
- 旧：`sparse_desc_2d`，固定 subsample=8
- 新：`dense_desc_2d`，先按 `desc_conf` 过滤，再在保留像素上做 dense reciprocal descriptor matching

---

## 4. 推荐实现形态

### 4.1 新增公共匹配层

建议新增两个文件，不再把 dense 逻辑硬塞回旧 `FlowMatcher`：

1. `pseudo_branch/common/mast3r_pair_forward.py`
2. `pseudo_branch/common/mast3r_matchers.py`

原因：
- 2D dense 和后续 3D matching 都需要同一份 MASt3R forward 输出
- 如果继续把所有逻辑塞进 `flow_matcher.py`，会把当前 simple sparse path 和后续 v2 path 混在一起
- 现在 repo 已完成目录整理，新的通用 runtime helper 放在 `pseudo_branch/common/` 最合适

### 4.2 `mast3r_pair_forward.py` 负责什么

提供一个共享 forward wrapper，例如：

```python
@dataclass
class MASt3RPairBundle:
    desc1: np.ndarray
    desc2: np.ndarray
    desc_conf1: np.ndarray
    desc_conf2: np.ndarray
    pts3d_1: np.ndarray | None
    pts3d_2_in_1: np.ndarray | None
    conf1: np.ndarray | None
    conf2: np.ndarray | None
    image_size: tuple[int, int]
    meta: dict
```

核心接口建议：

```python
def run_mast3r_pair(
    img1_path: str,
    img2_path: str,
    size: int = 512,
    device: str = "cuda",
) -> MASt3RPairBundle:
    ...
```

实现要点：
- 内部复用当前 MASt3R model load 逻辑
- 返回 `desc` / `desc_conf`
- 同时保留 `pts3d` / `pts3d_in_other_view` / `conf`，给后续 3D plan 复用
- 在 matcher 实例内做轻量 LRU cache，key 用 `(img1_path, img2_path, size)`
- 只做运行时内存缓存；第一版先不做跨脚本磁盘缓存

### 4.3 `mast3r_matchers.py` 负责什么

建议引入一个统一接口，而不是到处散写：

```python
class BasePairMatcher:
    def match_pair(self, img1_path: str, img2_path: str, size: int = 512):
        raise NotImplementedError

class SparseDescMatcher(BasePairMatcher):
    ...

class DenseDescMatcher(BasePairMatcher):
    ...
```

`DenseDescMatcher.match_pair()` 建议输出继续保持旧接口兼容：

```python
pts1_xy, pts2_xy, match_conf
```

这样 `rgb_mask_inference.py`、`brpo_build_mask_from_internal_cache.py` 的下游 verify 逻辑基本都不用动。

---

## 5. Dense 2D matching 的具体算法

### 5.1 候选像素选择

对两个视角分别做：
- 读取 `desc_conf`
- 用 quantile 而不是固定阈值过滤候选像素

建议第一版默认配置：
- `dense2d_conf_mode = quantile`
- `dense2d_conf_quantile = 0.90`

理由：
- 单帧 smoke 上，`q=0.90` 比 `q=0.95` 更有机会真正扩大 support
- quantile 比绝对阈值更稳，不容易被不同帧的 confidence scale 漂移影响

生成候选 2D 坐标时直接用：
- `dust3r.utils.geometry.xy_grid(W, H)`
- 按 mask 选出候选坐标

### 5.2 reciprocal descriptor matching

在候选 descriptor 子集上做：

```python
nn0, nn1 = bruteforce_reciprocal_nns(
    desc1_sel,
    desc2_sel,
    device="cuda",
    dist="dot",
    block_size=2**13,
)
reciprocal = (nn1[nn0] == np.arange(len(nn0)))
```

然后把 reciprocal index 映射回候选 2D 坐标：
- `pts1 = xy1_sel[reciprocal]`
- `pts2 = xy2_sel[nn0][reciprocal]`

### 5.3 match confidence

第一版不引入复杂新分数，直接定义为：

```python
match_conf = sqrt(desc_conf1_sel[reciprocal] * desc_conf2_sel[nn0][reciprocal])
```

这与当前 sparse path 使用 `desc_conf` 的风格一致，且后续若需要做 top-k 或额外筛选，可直接沿用。

---

## 6. 具体代码落点

### 6.1 新文件

1. `pseudo_branch/common/mast3r_pair_forward.py`
   - model load
   - image load
   - MASt3R inference
   - pair bundle cache

2. `pseudo_branch/common/mast3r_matchers.py`
   - `SparseDescMatcher`
   - `DenseDescMatcher`
   - `build_pair_matcher()` factory

### 6.2 修改文件

1. `pseudo_branch/common/__init__.py`
   - 导出 `build_pair_matcher` / `DenseDescMatcher`

2. `scripts/build_brpo_v2_signal_from_internal_cache.py`
   - 旧：`matcher = FlowMatcher()`
   - 新：`matcher = build_pair_matcher(...)`
   - 新增 CLI：
     - `--matcher-mode {sparse_desc_2d,dense_desc_2d}`
     - `--dense2d-conf-quantile`
     - `--dense2d-max-points-per-image`（可选）

3. `scripts/brpo_build_mask_from_internal_cache.py`
   - 同样替换 `FlowMatcher()`
   - 同样接入 CLI 参数
   - `run_branch()` 不改 verify contract，只改 matcher source

4. `pseudo_branch/mask/rgb_mask_inference.py`
   - 函数签名可以保持不变，因为它只依赖 `matcher.match_pair()`
   - 但要在 `matcher_meta` 中追加：
     - `matcher_mode`
     - `conf_quantile`
     - `num_candidate_pixels_left/right`
     - `num_reciprocal_matches_left/right`

5. `pseudo_branch/common/flow_matcher.py`
   - 第一版不删除，保留为 legacy sparse matcher
   - 若接入统一 factory，则只在 factory 内部按 `sparse_desc_2d` 复用它

---

## 7. Pipeline 中的调用位置

Dense 2D matcher 的调用时机必须放在“per-branch support 生成”这一步，而不是放到 left/right 融合之后。

### 7.1 Signal v2 路径

位置：`scripts/build_brpo_v2_signal_from_internal_cache.py`

顺序应为：
1. `fused_rgb` 对 `left_ref_rgb` 做 dense 2D reciprocal matching
2. `fused_rgb` 对 `right_ref_rgb` 做 dense 2D reciprocal matching
3. 得到 `support_left/right`
4. 再走现有 `build_brpo_style_observation()` / `build_exact_brpo_*()` 等逻辑

### 7.2 exact backend / old verification 路径

位置：`scripts/brpo_build_mask_from_internal_cache.py`

顺序应为：
1. `pseudo_rgb`（或 branch pseudo / fused pseudo）对 ref RGB 做 dense 2D reciprocal matching
2. 把 `pts_pseudo / pts_ref` 送入 `verify_single_branch_exact()`
3. 得到 per-branch exact support
4. 再用 `build_brpo_confidence_mask()` 生成离散 `C_m`

关键点：dense 2D 方案只替换 match generator，不改 verify 和 `C_m` 融合层。

---

## 8. 输入 / 输出规划

### 8.1 输入

保持现有输入不变：
- pseudo/fused RGB 路径
- left/right ref RGB 路径
- pseudo depth
- rendered ref depth
- stage PLY / camera states / fusion weights 等

### 8.2 新增配置输入

建议新增 CLI 参数：

```text
--matcher-mode sparse_desc_2d|dense_desc_2d
--matcher-size 512
--dense2d-conf-quantile 0.90
--dense2d-max-points-per-image 30000   # 可选；0 表示不截断
```

### 8.3 输出

主输出文件名先不改，避免打断 consumer：
- `pseudo_confidence_exact_brpo_upstream_target_v1.npy`
- `pseudo_depth_target_exact_brpo_upstream_target_v1.npy`
- `support_left/right`
- `confidence_mask_brpo*`

但 metadata 要明确记录 matcher 配置，至少包括：
- `matcher_mode`
- `matcher_size`
- `dense2d_conf_quantile`
- `num_candidate_pixels_{left,right}`
- `num_matches_{left,right}`
- `num_support_{left,right}`

必要时可额外写入 diag：
- `diag/dense2d_desc_conf_left.png`
- `diag/dense2d_desc_conf_right.png`
- `diag/dense2d_candidate_mask_left.png`
- `diag/dense2d_candidate_mask_right.png`

---

## 9. 单帧 smoke 结论（用于决定是否优先做）

基于 frame 23 的 live smoke：
- 当前 sparse exact：`cm_nonzero_ratio ≈ 0.01644`
- dense 2D descriptor + `desc_conf q=0.90`：`cm_nonzero_ratio ≈ 0.03709`

这说明 dense 2D 方案在不改 BRPO 三档语义的前提下，确实能把 raw exact `C_m` 覆盖率从 `~1.6%` 拉到 `~3.7%` 左右。它不是最终最强方案，但足够值得作为低风险升级路线。

---

## 10. 建议执行顺序

### Phase D2-1：matcher 抽象与 dense2d 接线
- 新增 `mast3r_pair_forward.py`
- 新增 `mast3r_matchers.py`
- 两个 live script 接入 `--matcher-mode`
- 默认仍保留 `sparse_desc_2d`

### Phase D2-2：single-frame exact smoke
- 先只跑 frame 23
- 对比 sparse vs dense2d 的：
  - `support_ratio_left/right`
  - `support_ratio_both/single`
  - `cm_nonzero_ratio`
  - `mean_reproj_error`
  - `mean_rel_depth_error`

### Phase D2-3：8-frame signal build smoke
- 在当前 T4 full root 的 8 帧上跑完整 signal build
- 检查 summary 中 coverage 是否稳定高于 sparse
- 确认没有某些帧大面积塌陷

### Phase D2-4：tiny consumer smoke
- 只做 1-iter 或极小 StageA smoke
- 确认 `exact_brpo_upstream_target_v1 + exact_shared_cm_v1` consumer 不因 coverage 提高而直接炸掉

### Phase D2-5：正式 compare
- dense2d arm 与当前 sparse exact winner 正面对照
- compare 时固定：
  - same pseudo cache / same frame ids
  - same clean summary G~
  - same T1
  - same exact_shared_cm_v1 loss contract

---

## 11. 风险与控制

主要风险有三个：

1. dense candidate 太多，错误匹配变多
- 控制：先用 `desc_conf quantile`，而不是全图全量
- 第一版默认 `q=0.90`

2. left/right support 域不一致，both 区域反而下降
- 控制：summary 必须同时看 `both` 和 `single`，不能只看 union coverage

3. consumer 端虽然 coverage 上升，但 supervision 质量变差
- 控制：必须保留 exact verify；不能直接把 reciprocal match 命中的像素视作 valid support

---

## 12. 本方案的定位

Dense 2D 方案的定位是：
- 作为当前 sparse 2D path 的低风险升级版
- 优先验证“只是把匹配密度拉高，exact BRPO support 是否就能明显改善”
- 如果它能稳定提升 signal coverage 且不破坏质量，可以成为一条成本更低的主线或强 control
- 但它不是最终最贴近 MASt3R 本体能力的方案；更强的路线仍然是 3D matching 文档里的 MASt3R 3D reciprocal matching

---

## 13. 下一步执行口径

如果继续执行本方案，下一步不要先改下游 loss 或 `C_m` 规则，先严格按下面顺序落地：

1. 做 shared pair forward helper
2. 做 `dense_desc_2d` matcher
3. 接到两个 live script
4. 先过 exact single-frame / 8-frame smoke
5. 再决定是否进入正式 compare

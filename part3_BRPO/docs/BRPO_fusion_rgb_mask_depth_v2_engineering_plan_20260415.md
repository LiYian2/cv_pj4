# BRPO-style Fusion + RGB Mask / Depth Supervision v2 工程落地方案（2026-04-15）

## 1. 这份文档解决什么问题

这份方案不是继续在现有 `seed_support -> train_mask -> target_depth_for_refine -> v2 densify` 链路上局部补丁，而是定义一条新的、与旧版 mask/depth 逻辑隔离的 signal pipeline。

这条新链路的核心判断只有一句：

- fusion 用 depth/geometry 做 target↔reference 的权重；
- mask 用 fused RGB 上的 correspondence / reprojection validity 来决定训练可信度；
- depth supervision 单独生成，不再反过来充当 RGB mask 的主语义来源。

这和当前实现的差别，不只是“fusion 算错了”，而是把当前过度耦合的三件事重新拆开：
1. 哪个 branch 更该贡献 fused residual；
2. 哪个 fused pixel 适合做 RGB supervision；
3. depth target 应该在什么区域、以什么来源进入 refine。

---

## 2. 当前代码链路里，哪些地方必须承认已经不适合继续沿用

基于当前 repo 实现，现状是：

1. `scripts/prepare_stage1_difix_dataset_s3po_internal.py:388-466` 的 `stage_fusion()` 现在本质还是 RGB-only placeholder fusion：
   - `target_left/target_right` 只是 DiFix 图或 render RGB；
   - `depth_left/depth_right/render_depth` 全是 `None`；
   - `conf_left/conf_right` 是全 1；
   - `pseudo_branch/pseudo_fusion.py` 虽然支持 depth gate / agreement，但当前 pipeline 实际没有喂真实 branch depth / render depth。

2. `scripts/brpo_build_mask_from_internal_cache.py` 当前的 verify / mask 路径虽然比更早版本强，但它仍然把：
   - support seed
   - train mask propagation
   - continuous confidence
   - depth target 的起点
   放在一条强耦合链上。

3. `pseudo_branch/brpo_confidence_mask.py` 里的 `build_brpo_confidence_mask()`，当前本质仍是：
   - 先由 left/right support 离散得到 `both / single / none`；
   - 再用 projected depth agreement 做 continuous confidence；
   - 然后 pack 阶段继续把 train mask / target depth 都建立在这条 support 语义上。

4. `scripts/prepare_stage1_difix_dataset_s3po_internal.py:759-812` 的 pack 阶段仍然会从 `projected_depth_left/right` 直接生成：
   - `target_depth_for_refine.npy`
   - `target_depth_for_refine_source_map.npy`
   - `verified_depth_mask.npy`
   这条路径本质仍是“render depth 生成 seed，再扩成 mask，再产出 target depth”的旧思路，只是后来叠加了 propagation / densify / confidence-aware patch acceptance。

因此这次不应该再在旧文件上继续叠语义。正确做法是：

- fusion：允许直接覆盖当前 `pseudo_branch/pseudo_fusion.py` 的实现；
- mask / depth：不要覆盖旧文件，不要直接改写原来的 `brpo_confidence_mask.py / brpo_train_mask.py / brpo_depth_target.py / brpo_depth_densify.py` 语义；
- 新版 signal path 单独建目录、单独建脚本、单独建输出 schema，再由 refine 侧显式选择读取新版。

---

## 3. 目标边界

## 3.1 本次要落地的新主线

推荐的新主线：

`select -> difix -> BRPO v2 fusion -> BRPO v2 RGB-mask inference -> BRPO v2 depth supervision build -> pseudo_cache_v2 pack -> refine consumer(v2)`

其中：
- `select`：可继续复用当前 `scripts/select_signal_aware_pseudos.py` 与 E1 winner；
- `difix`：可继续复用当前 `prepare_stage1_difix_dataset_s3po_internal.py` 的 `stage_difix()`；
- `fusion`：覆盖当前 `pseudo_branch/pseudo_fusion.py` 的实现；
- `mask/depth/pack`：建立新的 v2 路径；
- `refine`：在 `run_pseudo_refinement_v2.py` 中新增一套显式读取 v2 signal 的入口，不要默认 fallback 到旧链路。

## 3.2 本次明确不做的事

1. 不把 full SPGM 一起塞进来；
2. 不把旧的 `train_confidence_mask_brpo_fused.npy` 语义悄悄改掉；
3. 不让新版 depth supervision 直接复写 `target_depth_for_refine.npy` 旧文件；
4. 不在第一版里把所有最近几天的新模块强绑进来。

可以复用的，只限于：
- pseudo selection（E1 winner / selection manifest）；
- current reprojection / projected depth helpers；
- current refine consumer 的 RGB/depth mask 分离接口。

---

## 4. 推荐的新文件组织

## 4.1 目录与文件

推荐新增目录：

```text
part3_BRPO/
├── pseudo_branch/
│   ├── pseudo_fusion.py                       # 直接覆盖为 BRPO-style fusion v2 主实现
│   └── brpo_v2_signal/
│       ├── __init__.py
│       ├── fusion_io.py                       # fusion 输入解析、branch path 解析、meta/schema
│       ├── rgb_mask_inference.py              # fused RGB -> mask/correspondence inference
│       ├── depth_supervision_v2.py            # 新版 depth target / source map / summary
│       ├── signal_pack_v2.py                  # frame 级导出、manifest 写入
│       └── signal_metrics.py                  # verified ratio / correction / overlap 等统计
├── scripts/
│   ├── build_brpo_v2_signal_from_internal_cache.py
│   └── prepare_stage1_brpo_v2_dataset_s3po_internal.py
└── docs/
    └── BRPO_fusion_rgb_mask_depth_v2_engineering_plan_20260415.md
```

## 4.2 为什么这样拆

1. `pseudo_fusion.py` 直接覆盖，是因为当前 fusion 确实应被视为“实现错位”，不值得维护旧语义；
2. `brpo_v2_signal/` 单独成目录，是为了把：
   - fused RGB mask 语义
   - depth supervision 语义
   - export schema
   和旧版 `brpo_confidence_mask / train_mask / depth_target / densify` 隔离开；
3. `prepare_stage1_brpo_v2_dataset_s3po_internal.py` 单独建脚本，是为了避免把旧 prepare 的 `verify/pack` 阶段继续污染成双语义怪物。

---

## 5. 当前 pipeline 与目标 pipeline 的一一对应

## 5.1 当前链路

```text
internal_cache
-> select pseudo ids
-> difix left/right
-> stage_fusion(rgb-only / uniform conf)
-> brpo_build_mask_from_internal_cache.py
   -> FlowMatcher + reprojection support
   -> support_left/right
   -> discrete/continuous confidence
   -> optional train mask propagation
-> pack
   -> projected_depth_left/right
   -> target_depth_for_refine.npy
-> M5 densify / source-aware refine
```

## 5.2 新链路

```text
internal_cache
-> select pseudo ids
-> difix left/right
-> BRPO-style fusion v2
   -> target↔reference overlap confidence
   -> target_rgb_fused.png
   -> fusion_weight_left/right.npy
   -> overlap_conf_left/right.npy
-> fused RGB mask inference v2
   -> raw_rgb_confidence_v2.npy/png
   -> raw_rgb_confidence_cont_v2.npy/png
   -> rgb_mask_meta_v2.json
-> depth supervision v2
   -> target_depth_for_refine_v2_brpo.npy
   -> target_depth_source_map_v2_brpo.npy
   -> depth_supervision_mask_v2_brpo.npy
   -> depth_meta_v2_brpo.json
-> pseudo_cache_v2 pack
-> refine consumer(v2) 显式读取 rgb-mask-v2 + depth-v2
```

关键变化是：
- RGB mask 先在 fused RGB 上独立定义；
- depth supervision 再参考该 RGB mask、branch projected depth、render depth 去生成；
- 不再让“depth support seed”反过来决定 RGB 可不可信。

---

## 6. Fusion：这次允许直接覆盖的部分

## 6.1 需要直接覆盖的文件

直接覆盖：`pseudo_branch/pseudo_fusion.py`

但建议保留现有外部函数名，以减少调用方改动：
- `run_fusion_for_sample(...)`
- `get_fusion_diag_images(...)`

外部接口尽量保持兼容，内部语义整体换掉。

## 6.2 当前 fusion 的问题

当前 `pseudo_fusion.py` 的主语义是：

- 分支分数 `S = C * G * V`
- left-right agreement `A`
- 归一化后 residual fusion

而且在当前 `stage_fusion()` 实际调用里：
- `C` 是 uniform 1；
- `G` 没真正启用；
- `V` 没真实 branch-level几何含义；
- agreement 主要是左右 repaired 图互相比较。

这和 BRPO 的关键区别是：
- BRPO 不是 left/right 互相比谁像；
- 而是分别看 left candidate 相对 left ref、right candidate 相对 right ref 的几何 overlap confidence。

## 6.3 fusion v2 的推荐输入

新的 `run_fusion_for_sample()` 推荐扩成下面这类输入：

```python
stats = run_fusion_for_sample(
    render_rgb_path: str,
    render_depth_path: str,
    pseudo_state: dict,
    left_ref_state: dict,
    right_ref_state: dict,
    target_rgb_left_path: str,
    target_rgb_right_path: str,
    stage_ply_path: str,
    pipeline_params,
    background,
    output_dir: str,
    depth_consistency_tau: float = 0.15,
    overlap_valid_eps: float = 1e-4,
    translation_scale_tau: float = 1.0,
)
```

其中需要在函数内部自行完成：
1. 读取 pseudo render depth；
2. 从 `stage_ply` 渲染 left/right ref depth；
3. 计算 target→ref overlap confidence；
4. 基于 left/right overlap confidence 做归一化权重；
5. 对 DiFix 候选残差做 fused residual；
6. 导出 fused image 与诊断图。

## 6.4 fusion v2 的输出

每个 frame 的 fusion 目录至少导出：

```text
fusion_v2/frame_XXXX/
├── target_rgb_fused.png
├── fusion_weight_left.npy
├── fusion_weight_right.npy
├── overlap_conf_left.npy
├── overlap_conf_right.npy
├── overlap_mask_left.npy
├── overlap_mask_right.npy
├── ref_depth_left_render.npy
├── ref_depth_right_render.npy
├── fusion_meta.json
└── diag/
    ├── fusion_weight_left.png
    ├── fusion_weight_right.png
    ├── overlap_conf_left.png
    ├── overlap_conf_right.png
    ├── overlap_mask_left.png
    └── overlap_mask_right.png
```

## 6.5 fusion v2 的核心计算

建议按下面的工程定义实现：

1. 对 pseudo pixel，以 pseudo render depth 反投影到世界；
2. 投到 left/right ref；
3. 判断是否在 ref 图内，且 ref depth 有效；
4. 比较 pseudo depth 与 ref rendered depth 的相对误差；
5. 引入一个 branch-level baseline / translation 一致性项；
6. 得到：
   - `overlap_conf_left`
   - `overlap_conf_right`
7. 用它们直接归一化：
   - `fusion_weight_left`
   - `fusion_weight_right`
8. 对两张 repaired 图做 residual fusion。

这里不要再引入当前 `mean_confidence + valid_ratio` 那种全局 heuristic；也不要把 left-right agreement 当 fusion 主语义。

---

## 7. 新版 RGB mask：完全独立于旧 train_mask 语义

## 7.1 必须新建文件，不覆盖旧 mask 逻辑

新增：`pseudo_branch/brpo_v2_signal/rgb_mask_inference.py`

不要去覆盖：
- `pseudo_branch/brpo_confidence_mask.py`
- `pseudo_branch/brpo_train_mask.py`

因为这两个文件已经承载旧版 `support -> train mask` 语义，会把新版目标混淆掉。

## 7.2 新版 RGB mask 的输入

`infer_rgb_confidence_mask_v2(...)` 推荐输入：

```python
def infer_rgb_confidence_mask_v2(
    fused_rgb_path: str,
    pseudo_state: dict,
    left_ref_state: dict,
    right_ref_state: dict,
    left_ref_rgb_path: str,
    right_ref_rgb_path: str,
    left_ref_depth: np.ndarray,
    right_ref_depth: np.ndarray,
    pseudo_render_depth: np.ndarray,
    matcher,
    tau_reproj_px: float = 4.0,
    tau_rel_depth: float = 0.15,
    tau_rgb_consistency: float = 0.10,
) -> dict:
```

## 7.3 新版 RGB mask 的输出

每个 frame 导出：

```text
signal_v2/frame_XXXX/
├── raw_rgb_confidence_v2.npy
├── raw_rgb_confidence_v2.png
├── raw_rgb_confidence_cont_v2.npy
├── raw_rgb_confidence_cont_v2.png
├── rgb_support_left_v2.npy
├── rgb_support_right_v2.npy
├── rgb_support_both_v2.npy
├── rgb_support_single_v2.npy
├── rgb_mask_meta_v2.json
└── diag/
    ├── reproj_error_left_rgbmask.png
    ├── reproj_error_right_rgbmask.png
    ├── rel_depth_error_left_rgbmask.png
    ├── rel_depth_error_right_rgbmask.png
    └── match_density_rgbmask_*.png
```

## 7.4 新版 RGB mask 的推荐语义

推荐保留 BRPO 风格的离散主语义：
- both = 1.0
- single = 0.5
- none = 0.0

但同时输出 continuous 版本：
- `raw_rgb_confidence_cont_v2`

continuous 的来源应是：
- reprojection residual
- relative depth consistency
- optional RGB feature agreement

注意这里的 RGB mask 是“fused RGB 这个 pixel 是否在 ref 里可被几何对应解释”，而不是“depth propagation 后这个位置有没有资格训练 depth”。

## 7.5 为什么这条线能提高 coverage

重点不在于“把阈值放宽”；而在于：
- 先用 depth/geometry 做更合理的 fusion；
- fused RGB 质量更闭合后，再做 correspondence mask；
- 这样单个 pixel 更可能在至少一侧 reference 找到稳定对应。

也就是说，它更可能提升的是“可被 RGB 监督消费的有效区域”，而不是直接把 verified depth 比例魔法般放大。

---

## 8. 新版 depth supervision：单独一条线，不再复用旧 seed/train-mask path

## 8.1 新增文件

新增：`pseudo_branch/brpo_v2_signal/depth_supervision_v2.py`

不要直接 import 旧：
- `brpo_train_mask.py`
- `brpo_depth_target.py`
- `brpo_depth_densify.py`

如果要复用数学细节，可以复制逻辑到新文件并改名，避免在实验阶段搞不清楚究竟走了旧语义还是新语义。

## 8.2 新版 depth supervision 的输入

```python
def build_depth_supervision_v2(
    render_depth: np.ndarray,
    projected_depth_left: np.ndarray,
    projected_depth_right: np.ndarray,
    projected_valid_left: np.ndarray,
    projected_valid_right: np.ndarray,
    raw_rgb_confidence: np.ndarray,
    raw_rgb_confidence_cont: np.ndarray | None,
    fusion_weight_left: np.ndarray,
    fusion_weight_right: np.ndarray,
    fallback_mode: str = 'render_depth',
    both_mode: str = 'weighted_by_fusion',
    single_mode: str = 'single_branch_projected',
    min_rgb_conf_for_depth: float = 0.5,
    use_continuous_reweight: bool = True,
) -> dict:
```

## 8.3 depth supervision v2 的主原则

第一版推荐坚持三条：

1. depth 仍允许 fallback 到 render depth；
2. 但 depth 的“可训练区域”不再来自旧 propagation train-mask，而来自：
   - raw RGB confidence 主区域
   - branch projected depth valid 区域
   - 两者的规则化组合
3. both 区域的 depth 合成，不再简单平均，而是优先用 fusion weight 做同语义加权。

## 8.4 depth supervision v2 的输出

每个 frame 导出：

```text
signal_v2/frame_XXXX/
├── target_depth_for_refine_v2_brpo.npy
├── target_depth_source_map_v2_brpo.npy
├── depth_supervision_mask_v2_brpo.npy
├── depth_supervision_mask_seed_v2_brpo.npy
├── depth_supervision_mask_dense_v2_brpo.npy
├── depth_meta_v2_brpo.json
└── diag/
    ├── target_depth_for_refine_v2_brpo.png
    ├── depth_supervision_mask_v2_brpo.png
    ├── depth_source_map_v2_brpo.png
    └── depth_rel_correction_v2_brpo.png
```

`target_depth_source_map_v2_brpo.npy` 仍建议保持 source-aware 训练友好：
- 1 = verified_left
- 2 = verified_right
- 3 = verified_both_weighted
- 4 = render_fallback
- 5 = optional dense-expanded-from-v2

但这一次的 dense-expanded-from-v2 不应再沿用旧 `seed_support -> train_mask propagation` 名字；否则语义会再次混乱。

## 8.5 depth supervision v2 的设计取舍

第一版不要急着上 patchwise densify。

更稳的顺序是：
1. 先做 fused-weighted verified target；
2. 先让 refine consumer 真正吃到 `RGB mask v2 + depth target v2`；
3. 若 coverage 仍不够，再在 `depth_supervision_v2.py` 内加一个轻量 `expand_depth_support_v2()`；
4. 这个扩张函数也必须输出独立文件名，不与旧 `target_depth_for_refine_v2` 混用。

---

## 9. 新 prepare / pack 脚本应该怎么组织

## 9.1 新脚本

新增：`scripts/prepare_stage1_brpo_v2_dataset_s3po_internal.py`

建议不要在旧脚本里再加新的 `--verification-mode brpo_v2` 分支。旧脚本已经太重，而且它天然会把新旧 schema 混在一个 pack 目录下。

## 9.2 新脚本的 stage 设计

推荐阶段：

```text
--stage {select,difix,fusion_v2,signal_v2,pack_v2,all}
```

其中：
- `select`：可直接复用旧 `stage_select()` 或 selection manifest；
- `difix`：可直接复用旧 `stage_difix()`；
- `fusion_v2`：调用覆盖后的 `pseudo_fusion.py`；
- `signal_v2`：调用 `build_brpo_v2_signal_from_internal_cache.py` 或等价函数；
- `pack_v2`：把 `signal_v2` 产物打进新的 pseudo_cache schema。

## 9.3 新 pseudo cache schema

推荐新的 schema_version：

```text
pseudo-cache-internal-v2-brpo-signal
```

不要继续沿用 `pseudo-cache-internal-v1.5`，因为这次不只是字段增加，而是 supervision 语义真的换了。

## 9.4 pack_v2 输出目录

```text
<run_root>/pseudo_cache_v2/
├── manifest.json
└── samples/<frame_id>/
    ├── camera.json
    ├── refs.json
    ├── render_rgb.png
    ├── render_depth.npy
    ├── target_rgb_left.png
    ├── target_rgb_right.png
    ├── target_rgb_fused.png
    ├── raw_rgb_confidence_v2.npy
    ├── raw_rgb_confidence_cont_v2.npy
    ├── target_depth_for_refine_v2_brpo.npy
    ├── target_depth_source_map_v2_brpo.npy
    ├── depth_supervision_mask_v2_brpo.npy
    ├── source_meta_v2.json
    └── signal_meta_v2.json
```

## 9.5 manifest.json 必须新增的信息

至少新增：
- `schema_version`
- `signal_pipeline = brpo_v2`
- `fusion_version`
- `rgb_mask_version`
- `depth_supervision_version`
- `selection_manifest_path`
- `signal_root`
- `sample_fields`（显式列出 RGB/depth 新字段名）

这样 refine loader 才能不靠猜测读取。

---

## 10. refine consumer：这次必须同步改的地方

## 10.1 当前可复用的部分

`run_pseudo_refinement_v2.py` 已经有两块可以直接利用：
1. RGB / depth mask 分离：`--stageA_rgb_mask_mode` / `--stageA_depth_mask_mode`
2. depth target 独立加载：`resolve_depth_target_path()`

这说明 refine 侧的接口层已经够用了，不需要再重做一遍训练主循环。

## 10.2 需要新增的 args

建议新增：

```text
--signal_pipeline {legacy,brpo_v2}
--stageA_rgb_mask_mode {auto,raw_confidence,raw_confidence_cont,seed_support_only,train_mask,legacy,brpo_v2_raw,brpo_v2_cont}
--stageA_depth_mask_mode {auto,seed_support_only,train_mask,legacy,brpo_v2_depth}
--stageA_target_depth_mode {auto,render_depth,target_depth_for_refine,target_depth_for_refine_v2,brpo_v2}
```

语义上：
- `brpo_v2_raw` -> `raw_rgb_confidence_v2.npy`
- `brpo_v2_cont` -> `raw_rgb_confidence_cont_v2.npy`
- `brpo_v2_depth` -> `depth_supervision_mask_v2_brpo.npy`
- `brpo_v2` -> `target_depth_for_refine_v2_brpo.npy`

## 10.3 需要新增的 loader

建议新增：

```text
pseudo_branch/brpo_v2_signal/fusion_io.py
```

内含：
- `resolve_stageA_rgb_mask_v2(sample_dir, mode)`
- `resolve_stageA_depth_mask_v2(sample_dir, mode)`
- `resolve_stageA_target_depth_v2(sample_dir, mode)`
- `load_signal_meta_v2(sample_dir)`

`run_pseudo_refinement_v2.py` 只负责：
- 判断 `--signal_pipeline`
- 调用对应 resolver
- 把结果塞回 `view['conf_rgb'] / view['conf_depth'] / view['depth_for_refine']`

不要把太多路径判断硬编码回主脚本。

## 10.4 refine history 也要更新

在 `stageA_history.json / stageB_history.json` 中新增：
- `signal_pipeline`
- `rgb_mask_v2_nonzero_ratio`
- `rgb_mask_v2_mean_positive`
- `depth_mask_v2_nonzero_ratio`
- `target_depth_v2_verified_ratio`
- `target_depth_v2_render_fallback_ratio`
- `target_depth_v2_weighted_both_ratio`

否则后面又会回到“只看 loss，不知道这次到底吃的是哪套 supervision”。

---

## 11. 建议的最小实现顺序

## Phase F1：先把 fusion 彻底改对

文件：
- `pseudo_branch/pseudo_fusion.py`
- `scripts/prepare_stage1_brpo_v2_dataset_s3po_internal.py` 的 `fusion_v2`

目标：
- 当前 E1 winner pseudo set 上，先生成新的 `target_rgb_fused.png` 与 `fusion_weight_left/right.npy`
- 只做 cache-level 诊断，不接 refine

验收：
- 每帧都有 overlap_conf / fusion_weight 导出
- 诊断图能看出不是 uniform / agreement-only 权重

## Phase F2：做 fused RGB mask v2

文件：
- `pseudo_branch/brpo_v2_signal/rgb_mask_inference.py`
- `scripts/build_brpo_v2_signal_from_internal_cache.py`

目标：
- 让 `raw_rgb_confidence_v2` 成为新版 RGB supervision 主入口

验收：
- `raw_rgb_confidence_v2` 的 nonzero ratio、both/single 占比可统计
- 与旧 discrete / continuous confidence 做 apples-to-apples compare

## Phase F3：接 depth supervision v2，但先不做 densify

文件：
- `pseudo_branch/brpo_v2_signal/depth_supervision_v2.py`
- `signal_pack_v2.py`

目标：
- 生成 `target_depth_for_refine_v2_brpo.npy`
- 生成 `depth_supervision_mask_v2_brpo.npy`

验收：
- 能读到 verified_ratio / fallback_ratio / correction magnitude
- 与旧 `target_depth_for_refine_v2` 对照

## Phase F4：把 refine consumer 接上 v2 signal

文件：
- `scripts/run_pseudo_refinement_v2.py`
- `pseudo_branch/brpo_v2_signal/fusion_io.py`

目标：
- 在不破坏 legacy 的前提下，支持 `--signal_pipeline brpo_v2`

验收：
- 2-frame smoke
- 8-frame short compare
- history 里能明确看出读的是 v2 还是 legacy

## Phase F5：必要时再决定是否加 depth-support expand v2

只有当：
- RGB mask v2 已经带来更干净的 supervision
- 但 depth coverage 仍明显不够

再加 `expand_depth_support_v2()`。

不要一开始就把旧 M5 densify 全量搬过来。

---

## 12. 推荐的实验顺序

## Exp-F0：cache-only 对照（不跑 refine）

对象：E1 `signal-aware-8`

对照：
- old fusion + old mask summary
- new fusion + new rgb mask v2 summary

关注：
- `raw_rgb_confidence_v2 nonzero ratio`
- `support_both / support_single`
- `fusion_weight` 的左右偏置分布

## Exp-F1：只换 RGB mask，不换 depth target

目的：先验证“depth监督 fuse，但 rgb 做 mask”这件事本身有没有正向作用。

做法：
- target RGB 用 `target_rgb_fused.png`
- RGB mask 用 `raw_rgb_confidence_v2`
- depth 仍暂用当前 `target_depth_for_refine_v2`

如果这一步 short compare 已有改善，说明 RGB 语义重排本身是有效的。

## Exp-F2：换完整 v2 signal

做法：
- RGB mask 用 `raw_rgb_confidence_v2`
- depth target 用 `target_depth_for_refine_v2_brpo.npy`
- depth mask 用 `depth_supervision_mask_v2_brpo.npy`

关注：
- short StageA 结果是否至少不差于 Exp-F1
- verified_ratio / fallback_ratio 是否更合理

## Exp-F3：只在 Exp-F2 通过后，再考虑引入最近几天模块

只允许二选一地小步引入：
1. E1 selection manifest 作为默认 pseudo set
2. confidence-aware expand v2 的轻量版本

不要把 E1 + E3 + v2 depth expand 一次性混跑。

---

## 13. 哪些最近几天的模块可以接，哪些先别接

## 13.1 推荐直接接入

1. `scripts/select_signal_aware_pseudos.py`
   - 当前 default winner 就是 E1 `signal-aware-8`
   - 它是新 signal pipeline 的最佳上游输入，不需要再回 midpoint8

2. `run_pseudo_refinement_v2.py` 现有的 RGB/depth 分离接口
   - 这是接 v2 signal 最省改动的入口

3. `brpo_reprojection_verify.py` 里的几何 helper
   - 投影 / 采样 / 相机状态转换已经够用

## 13.2 建议先不要直接接入

1. 旧 `brpo_train_mask.py`
2. 旧 `brpo_depth_target.py`
3. 旧 `brpo_depth_densify.py`
4. 旧 `materialize_m5_depth_targets.py`

原因不是这些模块无用，而是它们名字和输出已经强绑定旧语义。第一版 v2 必须先证明：
- RGB mask 语义重排是否有效；
- weighted verified depth target 是否有效。

等这两点过了，再决定要不要把某些 densify 机制迁回 v2。

---

## 14. 风险点与规避

1. 最大风险不是代码写不出来，而是新旧 artifact 名字混用。
   - 规避：所有 v2 文件名都带 `_v2` 或 `_v2_brpo`

2. 第二个风险是 pack 脚本偷偷继续写旧 `target_depth_for_refine.npy`。
   - 规避：v2 pack 写到 `pseudo_cache_v2/`，不写回旧 `pseudo_cache/`

3. 第三个风险是 refine loader 默默 fallback 到 legacy。
   - 规避：`--signal_pipeline brpo_v2` 下如果缺少 v2 文件，直接报错，不允许 silent fallback

4. 第四个风险是把“depth coverage 不够”又理解成“继续扩 train_mask”。
   - 规避：v2 第一版先不叫 train_mask，改叫 `depth_supervision_mask_v2_brpo`

---

## 15. 一句话执行判断

如果按现在的 repo 继续做，这条线最合理的工程策略是：

- 直接重写 `pseudo_fusion.py`，把 fusion 主语义换成 target↔reference overlap confidence；
- 新建 `brpo_v2_signal/`，把 fused RGB mask 和 depth supervision 完全从旧 `seed_support -> train_mask -> target_depth` 路径里剥离出来；
- 用新的 `pseudo_cache_v2` 和 `--signal_pipeline brpo_v2` 去做 short compare；
- 先证明“depth监督 fuse、RGB做mask”这件事本身有没有带来更高质量的可消费覆盖，再决定哪些最近几天的模块值得迁入这条新主线。

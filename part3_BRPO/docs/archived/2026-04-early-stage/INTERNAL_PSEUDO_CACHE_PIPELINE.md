# INTERNAL_PSEUDO_CACHE_PIPELINE.md

> 最后更新：2026-04-12 02:10
> 目的：说明 Re10k-1 internal route 下，`internal pseudo cache` 从最上游 `internal_eval_cache` 一直到 `pseudo_cache` 中各类产物（RGB target / seed support / train mask / projected depth / blended target depth）的完整生成链路。

---

## 1. 一句话总览

当前 internal pseudo cache 不是“从 render 一步变出训练样本”，而是分成下面几层：

```text
internal_eval_cache
  ↓
select pseudo frames + left/right refs
  ↓
Difix left/right
  ↓
fusion -> fused pseudo RGB target
  ↓
verify (left branch / right branch)
  ↓
seed_support + projected_depth
  ├─→ propagation -> train_mask
  └─→ blend with render_depth -> target_depth_for_refine
  ↓
pack -> pseudo_cache
```

最关键的是：
- `train_mask` 和 `verified depth` 都来自 verify 结果；
- 但它们在 verify 之后**分叉**成两条不同的派生链；
- 这就是为什么当前 `train_mask coverage` 能到 `~19%`，而 `verified depth coverage` 仍只有 `~1.56%`。

---

## 2. 最上游：internal_eval_cache 是 source of truth

路径示例：

```text
/home/bzhang512/CV_Project/output/part2_s3po/re10k-1/
  s3po_re10k-1_full_internal_cache/
  Re10k-1_part2_s3po/2026-04-11-05-33-58/
  internal_eval_cache/
```

关键文件：

```text
internal_eval_cache/
├── camera_states.json
├── manifest.json
├── after_opt/
│   ├── render_rgb/<frame>_pred.png
│   ├── render_depth_npy/<frame>_pred.npy
│   └── point_cloud/point_cloud.ply
└── before_opt/...
```

这里最重要的四类源数据是：

1. `camera_states.json`
   - 保存所有 frame 的位姿、内参、图像尺寸、image path 等
   - **当前 internal route 的 pseudo / left_ref / right_ref 都是从这里取 camera state**

2. `after_opt/render_rgb/<frame>_pred.png`
   - 当前地图从 pseudo frame 视角渲出来的 RGB
   - 这是 pseudo frame 的最原始 RGB source

3. `after_opt/render_depth_npy/<frame>_pred.npy`
   - 当前地图从 pseudo frame 视角渲出来的 depth
   - 这是 pseudo frame 的最原始 depth source

4. `after_opt/point_cloud/point_cloud.ply`
   - 当前地图 PLY
   - verify 阶段会拿它在 reference 视角现渲 `ref_depth`

---

## 3. select：先确定哪些 frame 要进入 pseudo cache

脚本：
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py --stage select`

作用：
- 从 `camera_states.json` 和 `stage_meta.json` 里拿到所有 non-KF frame
- 给每个 pseudo frame 找最近的左 / 右 keyframe
- 生成 pseudo selection manifest

这一层只是定义：
- pseudo sample 是谁
- 它对应的 `left_ref` / `right_ref` 是谁

这一步之后，每个 pseudo sample 已经有：
- `frame_id`
- `left_ref_frame_id`
- `right_ref_frame_id`
- `render_rgb_path`
- `render_depth_path`

---

## 4. Difix：同一个 pseudo frame 做左右两路修复

脚本：
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py --stage difix`

输出：

```text
difix/
├── left_fixed/<image_name>.png
└── right_fixed/<image_name>.png
```

含义：
- `left_fixed`：更偏向利用 left reference 约束修复的 pseudo RGB
- `right_fixed`：更偏向利用 right reference 约束修复的 pseudo RGB

到这里，一张 pseudo frame 已经有三种 RGB 形态：
1. 原始 render RGB
2. `left_fixed`
3. `right_fixed`

---

## 5. fusion：把 left/right repaired 图融合成 fused pseudo target

脚本：
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py --stage fusion`

输出：

```text
fusion/samples/<frame_id>/
├── target_rgb_fused.png
├── confidence_mask_fused.npy
└── fusion_meta.json
```

含义：
- `target_rgb_fused.png`：当前主线下最接近 BRPO 语义的 pseudo RGB target
- 当前 `fused_first verification` 会优先拿这张图作为 pseudo image

同时，pack 阶段也会保留：
- `target_rgb_left.png`
- `target_rgb_right.png`
- `target_rgb_fused.png`

---

## 6. verify：左右 branch 分开做几何验证

脚本：
- `scripts/brpo_build_mask_from_internal_cache.py`
- 被 `prepare_stage1_difix_dataset_s3po_internal.py --stage verify` 调起

### 6.1 这里的 “branch” 是什么意思？

这里的 `branch` 不是指模型分支，而是指：
- **left branch**：pseudo frame 与 `left_ref` 的那一路配对验证
- **right branch**：pseudo frame 与 `right_ref` 的那一路配对验证

也就是同一个 pseudo frame，会分别和左右 reference 各做一套 matcher + geometry check。

所以：
- left branch 的 `ref_state` = left keyframe 的 camera state
- right branch 的 `ref_state` = right keyframe 的 camera state

### 6.2 verify 用的 camera state 是哪里来的？

**是 `internal_eval_cache/camera_states.json` 里的。**

当前代码路径就是：
- `brpo_build_mask_from_internal_cache.py` 读取 `internal_eval_cache/camera_states.json`
- 用 `states_by_id[frame_id]` 取 `pseudo_state`
- 用最近邻 keyframe 取 `left_state` / `right_state`

所以你说的这个理解是对的：

**我们现在的 internal 路径，verify 里用的 pseudo / reference 相机，确实就是 internal eval 存下来的 camera states。**

### 6.3 verify 里的 `ref_depth` 是怎么来的？

这个点非常关键。

对每个 branch：
- 不直接用 external depth
- 而是拿：
  - `reference 视角的 camera state`
  - `当前 stage PLY`
- 调 `render_depth_from_state(...)`
- 在 **reference view** 重新渲一张 `ref_depth`

也就是说：

```text
ref_depth = current stage PLY rendered at reference camera
```

而不是从 pseudo view 的 render depth 里裁出来的。

这样做的原因是：
- verify 的几何起点在 reference view
- 要先在 reference 像素上 backproject 到 3D
- 所以需要 reference view 对应的 depth

### 6.4 verify 里到底做了什么

对每个 branch：
1. matcher 在 `pseudo_rgb` 和 `ref_rgb` 之间找匹配点
2. 在 `ref_depth` 上取参考点深度，backproject 到 3D world
3. 再把 3D 点 project 回 pseudo view
4. 检查：
   - reprojection error
   - relative depth error（和 pseudo render depth 对比）
5. 满足阈值的匹配点，才算 support

---

## 7. verify 的直接产物：两类，不是一类

verify 完之后，当前会直接吐出两类东西。

### 7.1 第一类：seed support

```text
seed_support_left.npy
seed_support_right.npy
seed_support_both.npy
seed_support_single.npy
```

含义：
- `seed_support_left`：left branch 通过验证的 pseudo 像素
- `seed_support_right`：right branch 通过验证的 pseudo 像素
- `seed_support_both`：左右都通过
- `seed_support_single`：只一边通过

它记录的是：

**哪些 pseudo 像素通过了 BRPO-style 几何验证。**

这是一张 **支持区域/置信区域图**，本质是 mask，不是 depth。

### 7.2 第二类：projected depth

```text
projected_depth_left.npy
projected_depth_right.npy
projected_depth_valid_left.npy
projected_depth_valid_right.npy
```

它记录的不是“是否通过”，而是：

**通过验证的那些 support 像素，在 pseudo view 下对应的几何深度值是多少。**

也就是：
- `seed_support_*` 只告诉你：这个像素是否可信
- `projected_depth_*` 进一步告诉你：这个可信像素的 depth 值是多少

所以两者不是重复的：
- support 是 **where**
- projected depth 是 **what depth value**

---

## 8. 为什么会分成 mask 和 depth 两条派生链？

这一步是当前最关键的理解点。

因为 verify 之后，后续要服务两个不同目的：

### 8.1 目的 A：训练时“哪些区域该被监督”

这就是 **mask** 的作用。

训练里需要知道：
- 哪些区域该算 RGB loss
- 哪些区域可信度更高
- 哪些区域完全不要管

这时候需要的是一张 **可训练 coverage 更大的 supervision mask**。

所以从 `seed_support` 出发，会走：

```text
seed_support -> propagation -> train_mask
```

这条路允许把高精度 seed 扩成更大覆盖的训练区域。

### 8.2 目的 B：训练时“这些可信区域的深度值是多少”

这就是 **depth** 的作用。

如果要给 Stage A / Stage B depth supervision，就不只要知道：
- 哪些区域可信

还要知道：
- 这些区域对应的深度目标值具体是多少

所以需要另一条路：

```text
projected_depth + render_depth fallback -> target_depth_for_refine
```

这条路关心的是 **数值 depth target**，不是 coverage mask 本身。

---

## 9. 为什么 mask 可以扩散，但 depth 没有一起扩散？

因为它们的风险不一样。

### 9.1 mask 扩散相对安全

mask 扩散做的是：
- 让附近那些在颜色 / 深度上相近的区域，也能参与监督

这本质上是在扩大“**可以被监督的区域**”。

即使扩错一点，它的后果通常是：
- 监督范围变宽了
- 但不一定直接把几何值写错

### 9.2 depth 扩散风险更大

depth 不只是“这个区域能不能用”，而是“这个区域的深度值到底是多少”。

如果把 sparse verified depth 直接像 mask 那样扩散：
- 你不只是扩大 coverage
- 你是在把一个像素的 depth 数值传播到周围一片区域

这很容易把几何值传播错。

所以当前实现里采取的是更保守的设计：
- mask 可以 propagation
- depth 先只保留 verify 直接给出的 sparse projected depth
- 其他区域先 fallback 到 render depth

这就是为什么当前：
- `train_mask coverage ≈ 19.4%`
- `verified depth coverage ≈ 1.56%`

两者会明显不一致。

---

## 10. 当前 pseudo cache 里每类文件到底从哪来

### 10.1 直接来自 internal eval cache 的

```text
render_rgb.png        <- internal_eval_cache/<stage>/render_rgb/<frame>_pred.png
render_depth.npy      <- internal_eval_cache/<stage>/render_depth_npy/<frame>_pred.npy
camera.json           <- camera_states.json 中的 pseudo_state
refs.json             <- camera_states.json 中的 left/right ref states
```

### 10.2 来自 Difix / fusion 的 RGB target

```text
target_rgb_left.png   <- difix/left_fixed/<image_name>.png
target_rgb_right.png  <- difix/right_fixed/<image_name>.png
target_rgb_fused.png  <- fusion/samples/<frame>/target_rgb_fused.png
```

### 10.3 来自 verify 的 seed / projected depth

```text
seed_support_*.npy
projected_depth_left.npy
projected_depth_right.npy
projected_depth_valid_left.npy
projected_depth_valid_right.npy
verification_meta.json
```

### 10.4 来自 propagation 的 train mask

```text
train_confidence_mask_brpo_left.npy
train_confidence_mask_brpo_right.npy
train_confidence_mask_brpo_fused.npy
train_support_*.npy
```

### 10.5 来自 M3 pack 的 blended depth target

```text
target_depth_for_refine.npy
target_depth_for_refine_source_map.npy
target_depth_for_refine_meta.json
verified_depth_mask.npy
```

### 10.6 alias / compatibility 层

```text
confidence_mask_brpo_*.npy   -> alias 到当前训练真正消费的 train mask
support_*.npy                -> alias 到 seed support
```

---

## 11. 当前最重要的现实结论

### 11.1 train_mask 已经不是最原始 seed 了

当前：
- `seed_support_union_coverage ≈ 1.56%`
- `train_mask_coverage ≈ 19.4%`

说明 train mask 这条链已经完成了“从 seed 到 training supervision”的扩张。

### 11.2 verified depth 仍然基本停留在 seed/support 级别

当前：
- `verified_depth_coverage ≈ 1.56%`

并且它与 `seed_support_union_coverage` 基本一致。

说明当前的 verified depth 还没有像 train mask 一样被“扩成大 coverage supervision”，而是仍然只保留 verify 直接支持的 sparse depth。

### 11.3 这就是当前新的主问题来源

所以现在的真实状态是：
- RGB supervision 用的是已经变大的 train mask
- depth supervision 里真正来自 BRPO correction 的区域仍然很小

于是 `blended_depth` 在 Stage A 中：
- 已经不是没接上
- 但因为 verified depth 还是太 sparse，当前作用还偏弱

---

## 12. 最后用一句最短的话总结

当前 internal pseudo cache 的核心逻辑不是：
- “从 render 直接做出一个 mask/depth”

而是：

**render + camera states 给出 pseudo sample；verify 先产出 sparse 几何 seed；然后 seed 分成两条路：一条扩成 train mask，一条保留为 sparse projected depth 并与 render depth 融成 target depth。**

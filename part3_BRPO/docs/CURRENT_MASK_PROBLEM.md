# CURRENT_MASK_PROBLEM.md

> 最后更新：2026-04-12 02:20
> 主题：mask problem 已从“BRPO support 太稀，不能当训练 mask”推进到“train mask 已基本可用，但 verified depth 仍过稀，导致 blended depth 在 Stage A 中接上了却几乎不动”。

---

## 1. 当前结论

现在不能再笼统说“mask 还太稀”。更准确的说法是：

1. **train mask 这一层已经基本可用。**
   经过 `M1 → M2 → M2.5`，当前 `train_confidence_mask_brpo_fused` 的 coverage 已经从 seed-level 的 `~1.5%` 抬到大约 `18% ~ 20%`，落在此前判断的合理训练区间 `10% ~ 25%` 内。

2. **verified depth 这一层仍然过稀。**
   当前 `target_depth_for_refine` 中，真正来自 BRPO verified depth 的区域仍只有 `~1.56%`；其余 `~98.4%` 都是 `render_depth` fallback。

3. **两者不是同一层产物。**
   - `train_mask` 是从 `seed_support` 经过 propagation 扩出来的训练 supervision mask；
   - `verified depth` 当前仍直接来自 verify 阶段的 sparse projected depth，没有跟着 propagation 一起变大。

一句话说：

**mask problem 的第一阶段已经基本解决（train mask 不再只是 seed）；但新的主问题已经转成：verified depth 过稀，导致 M3/M4 的 blended depth 虽然接上了，却还没在 Stage A 里真正“动起来”。**

---

## 2. 当前精确统计：coverage 到底是多少

当前统计基于：

```text
/home/bzhang512/CV_Project/output/part2_s3po/re10k-1/
  s3po_re10k-1_full_internal_cache/
  Re10k-1_part2_s3po/2026-04-11-05-33-58/
  internal_prepare/re10k1__internal_afteropt__brpo_proto_v4_stage3/pseudo_cache/samples/
```

样本帧：`10 / 50 / 120`

### 2.1 seed support coverage

平均：
- `seed_support_union_coverage ≈ 1.5608%`
- `seed_support_both_coverage ≈ 0.7975%`

逐帧：
- frame 10: union `1.6285%`, both `0.7496%`
- frame 50: union `1.5076%`, both `0.7389%`
- frame 120: union `1.5465%`, both `0.9041%`

这说明 verify 的几何 seed 仍然是典型的 **高精度 / 低覆盖** 信号。

### 2.2 train mask coverage

平均：
- `train_mask_coverage ≈ 19.4126%`

逐帧：
- frame 10: `20.1538%`
- frame 50: `18.0340%`
- frame 120: `20.0500%`

这说明当前 `train_mask` 已经不再是“几乎没 coverage”的状态，而是已经进入此前 M2.5 认为更合理的区间（大约 `10% ~ 25%`）。

### 2.3 verified depth coverage

平均：
- `verified_depth_coverage ≈ 1.5608%`

逐帧：
- frame 10: `1.6285%`
- frame 50: `1.5076%`
- frame 120: `1.5465%`

注意这里一个关键事实：

**当前 verified depth coverage 与 seed support union coverage 是一样的。**

这不是巧合，而是当前实现逻辑决定的：
- 只有通过 verify 的支持点，才会写入 `projected_depth_left/right`；
- 后续 `target_depth_for_refine` 的 verified 区域，正是这些 projected depth 的有效并集。

### 2.4 当前 source map 说明了什么

当前 `target_depth_for_refine_source_map.npy` 统计大致是：
- `both_fused`: `~0.74% ~ 0.90%`
- `left_only + right_only`: `~0.61% ~ 0.88%`
- `render_fallback`: `~98.4%`

因此当前 M3 的语义非常明确：

**它不是大覆盖 depth supervision，而是“少量 BRPO verified depth correction + 大部分 render depth fallback”。**

注意：
- `target_depth_for_refine.npy` 本身因为有 fallback，所以几乎是整张图都有值；
- 当前 `~1.56%` 指的不是它“有没有值”，而是其中有多少像素是**真正来自 BRPO verified depth 的新信息**。

---

## 3. 逻辑链：train mask 和 verified depth 到底是什么关系

当前链路可以概括为：

```text
left/right verify
  ↓
seed_support_left/right/both/single
  ├─→ propagation → train_confidence_mask_brpo_*
  └─→ projected_depth_left/right (仅 support 像素)
                      ↓
                build_blended_target_depth
                      ↓
              target_depth_for_refine.npy
```

也就是说，当前有两条分叉：

1. **mask 分支**
   - `seed_support` → propagation → `train_mask`
   - 这条分支会把 coverage 从 `~1.5%` 扩到 `~19%`

2. **depth 分支**
   - `seed_support / verify support` → `projected_depth_*`
   - 再组装 `target_depth_for_refine`
   - 这条分支当前**没有**跟着 propagation 一起扩张

因此当前不能说：
- “verified depth coverage 变大了，是因为 train mask 变大了”

更应该说：
- `train_mask` 已经被 propagation 放大；
- `verified depth` 还停留在 seed / support 级别，所以仍然很稀。

---

## 4. mask 和 depth 各自的用处

这是当前最需要分清楚的地方。

### 4.1 mask 的用处

mask 回答的是：

**哪些像素应该参与监督，参与监督时权重多大。**

当前 `train_mask` 主要服务：
- masked RGB loss
- masked depth loss 的支持区域

它本质上是一个 **where / confidence** 问题。

### 4.2 depth 的用处

depth 回答的是：

**这些可信像素的目标深度值到底是多少。**

当前 `projected_depth_*` 和 `target_depth_for_refine.npy` 主要服务：
- 给 Stage A / 后续 Stage B 提供 depth target

它本质上是一个 **what depth value** 问题。

### 4.3 为什么 mask 可以扩散，但 depth 没跟着扩散

因为两者风险不同：
- 扩 mask，本质上是在扩大“允许监督的区域”；
- 扩 depth，本质上是在把一个 sparse depth 数值传播到周围区域。

后者风险更大，因为一旦传播错，就不是单纯“监督范围变大”，而是会把错误几何值写进 target。

所以当前实现选择了更保守的做法：
- `seed_support -> propagation -> train_mask`
- `projected_depth` 先保持 sparse
- 再和 `render_depth` 混成 `target_depth_for_refine`

也正因此，当前会出现：
- `train_mask ≈ 19.4%`
- `verified depth ≈ 1.56%`

这种明显不对称。

---

## 5. 当前 loss 结构，以及浅 verified depth 对 loss 的影响

### 5.1 旧的 v1 逻辑是什么

最早的 `run_pseudo_refinement.py`（v1）本质上是：
- pseudo pose 固定
- 直接更新 Gaussian / PLY
- 主要用 confidence-weighted RGB loss
- 更像 **fixed-pose appearance tuning**

所以你说的“最早不就是 RGB 加权 loss 且 PLY 会更新”，这个理解是对的，指的是 **v1**。

### 5.2 现在的 Stage A 在优化什么

当前 `run_pseudo_refinement_v2.py` 的 **Stage A** 不是 final refine，它做的是 warm-up / alignment：
- **更新 pseudo pose delta**
- **更新 exposure**
- **不更新 Gaussian / PLY**

当前总 loss 大致是：

```text
L = β * L_rgb
  + (1-β) * L_depth
  + λ_pose * L_pose_reg
  + λ_exp * L_exp_reg
```

其中：
- `L_rgb`：在 `train_mask` 上算，所以当前有效训练区域约 `19.4%`
- `L_depth`：也乘同一个 mask，但其中真正来自 BRPO verified depth 的新信号只有 `~1.56%`
- `L_pose_reg`：防止 pseudo pose delta 乱飘
- `L_exp_reg`：防止 exposure 乱飘

所以 Stage A 的作用不是“改地图”，而是：

**先把 pseudo supervision 和当前地图对齐一点，再决定要不要进入会真正改 Gaussian 的 Stage B。**

### 5.3 现在这个浅 verified depth 会怎么影响 loss

当前最关键的影响是：
- RGB 分支用的是扩大后的 `train_mask`，因此有一块相对大的有效监督区域；
- depth 分支虽然也在算，但真正携带“新几何信息”的区域只有 `~1.56%`；
- 剩下 `~98.4%` 的区域只是 `render_depth fallback`，更多是保底完整图，而不是新约束。

因此当前会出现：
- `blended_depth` 的 depth loss **非零**，说明它确实接进来了；
- 但 depth loss 在 Stage A 中几乎不下降，说明它对当前优化的牵引还不够强。

### 5.4 这对 Stage B 的意义是什么

原本 two-stage 的规划是：
- **Stage A**：先调 pseudo pose / exposure，把 pseudo supervision 对齐
- **Stage B**：再让 Gaussian + pseudo pose 做 joint refinement，也就是开始真正改地图 / PLY

所以现在如果 depth 还是这么稀，就会有一个风险：
- 如果直接进入 Stage B，地图更新更可能被 RGB 主导；
- sparse depth correction 太弱，不足以稳定地约束几何；
- 这样 Stage B 可能会把“对齐不够清楚的 supervision”直接写进 Gaussian。

这就是为什么当前不适合直接自信地进 Stage B。

---

## 6. 当前问题已经演变成什么

到现在，问题应该重命名成更准确的表述：

### 6.1 已经基本解决的部分

- `fused-first verification` 已打通
- `seed_support` 与 `train_mask` 已明确拆层
- `train_mask` coverage 已进入可训练区间
- `blended target_depth_for_refine` 已落地
- Stage A consumer 已能显式区分：
  - `train_mask`
  - `seed_support_only`
  - `blended_depth`
  - `render_depth_only`

### 6.2 当前新的主问题

当前新的主问题不再是：
- `BRPO mask 有没有做出来`
- 或 `train_mask coverage 能不能从 seed 变大`

而是：

**verified depth 仍停留在 seed/support 级别，coverage 约 `1.56%`，导致 `blended_depth` 在 Stage A 中虽然已接通，但 depth signal 基本不动。**

因此当前问题已经更像：

**depth-flatness problem on top of solved train-mask problem**

---

## 7. 我现在的判断：下一步应该做什么

我觉得当前最合理的下一步不是直接进 Stage B，而是先做一轮 **M4.6 / depth-flatness diagnosis**。

重点不是再堆更多长跑，而是把当前问题拆清楚：

1. 现在 `L_depth` 不动，主要是因为 verified depth 太稀，还是因为当前 loss / optimizer 对它不敏感？
2. 当前 `L_depth` 里，真正来自 `verified region` 的贡献占多少？是不是被大面积 fallback 区域“稀释”了？
3. 后面如果要增强 depth 作用，应该先：
   - 调 Stage A 的 loss / 权重 / 诊断口径
   - 还是回 upstream 做更受控的 depth-valid coverage 扩张

我现在不建议的事是：
- 直接进入 Stage B
- 或者在没搞清楚前，重新回到 `50% ~ 70%` 的宽 propagation

当前最稳的推进顺序应该是：

```text
先做 M4.6：解释为什么 depth 不动
  ↓
如果是 loss/sensitivity 问题，就先改 Stage A
  ↓
如果是 verified depth 太稀，就研究更受控的 depth-valid 扩张
  ↓
只有这两件事更清楚后，再决定是否进入 Stage B
```

---

## 8. 当前结论（供后续引用）

当前结论固定为：

1. **train mask 问题已经基本解决。**
   当前 `train_mask coverage ≈ 19.4%`，已不再只是 seed-level support。

2. **verified depth 稀疏问题仍未解决。**
   当前 `verified_depth coverage ≈ 1.56%`，与 `seed_support_union` 基本一致，说明它还停留在 verify/support 级别。

3. **当前实现下，verified depth 不会被 train mask 直接放大。**
   两者共享上游 seed，但分叉后：
   - `train_mask` 走 propagation
   - `verified depth` 走 sparse projected depth

4. **当前 Stage A 不是 final refine，而是 pose/exposure alignment warm-up。**
   它现在还不更新 Gaussian / PLY，真正改地图的是后续 Stage B。

5. **当前新的主问题是：**
   `train_mask 已经够大，但 verified depth 仍太稀，导致 blended depth 在 Stage A 中已接通却几乎不动。`

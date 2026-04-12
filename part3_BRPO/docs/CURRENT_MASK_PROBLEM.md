# CURRENT_MASK_PROBLEM.md

> 最后更新：2026-04-12 13:35
> 主题：mask problem 的第一阶段已经基本解决；当前新的主问题不是 train mask 不够，而是 **verified/densified depth 虽然已经变强，但 Stage A 的 pose/render 连通性存在异常，导致 depth 与 pose refine 都不可信地“发力失败”。**

---

## 1. 当前一句话结论

现在不能再把问题简单表述成“BRPO mask 太稀”。更准确的说法是：

- **train mask 这一层已经基本可用**：`train_mask coverage ≈ 19.4%`，处于此前判断更合理的 `10% ~ 25%` 区间；
- **原始 verified depth 的确太稀**：M3 下真正来自 verified projection 的区域只有 `~1.56%`；
- **M5 densify 已经把 depth 新信息区域显著抬高**：在 selected M5 参数下，`train_mask` 内非 fallback 区域已经到 `~81.2%`；
- **但即使做了 densify 和 source-aware depth loss，Stage A 的 depth loss 依然基本不动**；
- 进一步诊断发现：**当前 renderer 前向对 pose delta 看起来几乎不敏感，但反向却返回非零 pose 梯度**，这说明当前 Stage A 的 pose refine 路径本身就存在很大可疑点。

一句话说：

**mask problem 已经从“train mask 太稀”演变成“depth target 已增强，但 Stage A pose/render 路径可能有 forward/backward 不一致，导致 refine 过程本身不可信”。**

---

## 2. 当前关键统计

当前统计基于：

```text
/home/bzhang512/CV_Project/output/part2_s3po/re10k-1/
  s3po_re10k-1_full_internal_cache/
  Re10k-1_part2_s3po/2026-04-11-05-33-58/
  internal_prepare/re10k1__internal_afteropt__brpo_proto_v4_stage3/pseudo_cache/samples/
```

样本帧：`10 / 50 / 120`

### 2.1 M3 之前已经确认的 coverage

- `seed_support_union_coverage ≈ 1.5608%`
- `train_mask_coverage ≈ 19.4126%`
- `verified_depth_coverage ≈ 1.5608%`

这说明：
- `seed_support -> propagation -> train_mask` 这条链已经把训练 mask 从 seed-level 扩成了可训练覆盖；
- 但 `verified depth` 在 M3 里仍然停留在 seed/support 级别。

### 2.2 M5-0：当前 depth signal 被覆盖结构严重限制

M5-0 的诊断重点不是再看整图，而是看 **train_mask 内部**。

结果：
- `mean_train_mask_coverage ≈ 19.41%`
- `mean_verified_depth_coverage ≈ 1.56%`
- `mean_verified_within_train_mask_ratio ≈ 8.05%`
- `mean_render_fallback_within_train_mask_ratio ≈ 91.95%`

解释：
- 在训练真正消费的 mask 区域里，只有大约 `8%` 的像素落在 verified depth 区域；
- 剩下大约 `92%` 虽然也在 mask 里，但 depth target 仍然只是 fallback。

所以 M4.5 的“depth 不动”不只是因为 depth loss 值小，而是因为：

**当前 train_mask 内真正携带新几何信息的区域比例太小。**

### 2.3 M5-1：densify 之后的 depth coverage 已经明显增强

默认参数（`patch=11, stride=5, min_seed=6, max_std=0.08`）过于保守，只把 densified 区域抬到 `~2.08%`，总 non-fallback 也只有 `~3.64%`，不够。

经过小范围 sweep 后，当前选中的一组参数是：

```text
patch_size = 11
stride = 5
min_seed_count = 4
max_seed_delta_std = 0.08
```

在这组参数下：
- `mean_seed_ratio ≈ 1.56%`
- `mean_densified_ratio ≈ 14.21%`
- `mean_total_nonfallback_ratio ≈ 15.77%`
- `mean_render_fallback_ratio ≈ 84.23%`

更关键的是，在 `train_mask` 内部：
- `nonfallback_within_train_mask_ratio ≈ 81.22%`
- `seed_within_train_mask_ratio ≈ 8.05%`
- `dense_within_train_mask_ratio ≈ 73.17%`

这说明 M5-1 已经把“真正有新几何信息的区域”从 seed 级抬到了**train_mask 的大部分区域**。

所以到这一步，问题已经不再是“depth 还是太 sparse，什么都没法做”。

---

## 3. 逻辑链：train mask、verified depth、densified depth 到底是什么关系

当前链路应理解为：

```text
left/right verify
  ↓
seed_support_left/right/both/single
  ├─→ propagation → train_confidence_mask_brpo_*
  └─→ projected_depth_left/right (仅 support 像素)
                      ↓
                M3: blended target_depth_for_refine
                      ↓
                M5: densify correction field
                      ↓
              target_depth_for_refine_v2.npy
```

这意味着当前有三层不同的概念：

1. **train mask**
   - 回答“哪些像素参与监督、权重多大”
   - 来自 `seed_support -> propagation`

2. **verified depth**
   - 回答“哪些 support 像素有高置信深度值”
   - 来自 `projected_depth_*`
   - M3 下只覆盖 `~1.56%`

3. **densified depth**
   - 回答“如何把 sparse correction 受控地扩进 train-mask 区域”
   - 来自 `render_depth + sparse log-depth correction + patch-wise densify`
   - M5-1 下已经能把非 fallback 提到 `~15.8%` 整图、`~81.2%` 的 train-mask 内部区域

因此当前不能再说：
- verified depth 会随着 train mask 变大而自动变大；
- 或 M5 之后的 dense 区域还只是“和以前一样太 sparse”。

到 M5-1 为止，**upstream depth target 已经从“太 sparse 无法讨论”提升到“足够强，应该能测试 downstream 是否真的在用”。**

---

## 4. 当前 loss 结构，以及为什么新的问题不再只是 coverage

### 4.1 旧的 Stage A 结构

当前 Stage A 的总 loss 仍然大致是：

```text
L = β * L_rgb
  + (1-β) * L_depth
  + λ_pose * L_pose_reg
  + λ_exp * L_exp_reg
```

这里：
- `L_rgb` 在 `train_mask` 上算；
- `L_depth` 在当前 target depth 上算；
- Stage A 只更新 pseudo pose delta 和 exposure，不改 Gaussian / PLY。

### 4.2 M5-2 之后的结果

第三阶段（M5-2）我已经把 consumer 接成两组：

1. `M5 densified target + old depth loss`
2. `M5 densified target + source-aware depth loss`

结果很关键：
- `M5 + old depth loss`：`loss_depth ≈ 0.02770`，300 iter 中基本不动；
- `M5 + source-aware depth loss`：`loss_depth ≈ 0.06867`，300 iter 中也基本不动；
- 其中 source-aware 已经显式拆成：
  - `loss_depth_seed ≈ 0.05796`
  - `loss_depth_dense ≈ 0.03062`
  - `loss_depth_fallback ≈ 0`

这意味着：
- 当前问题已经**不再只是 fallback 稀释**；
- 因为就算把 fallback 基本去掉，`seed` 和 `dense` 两项还是不动。

所以到这一步，问题从“depth 太 sparse”继续推进成：

**depth target 已经增强，但 Stage A 仍然没有把它转化成有效的 pose/render 对齐。**

---

## 5. 新的关键诊断：Stage A pose/render 路径本身很可疑

这是当前最重要的新发现。

### 5.1 代码审计结果

- `cam_rot_delta / cam_trans_delta` 确实被传进 `gaussian_renderer`；
- `run_pseudo_refinement_v2.py` 的 optimizer 也确实在更新这两个参数；
- 所以“参数没接上”不是主问题。

### 5.2 但前向 sensitivity probe 的结果非常异常

我做了一个直接 probe：
- 固定同一张 pseudo view
- 人工把 `cam_rot_delta / cam_trans_delta` 设到：
  - `(0.01, 0.01)`
  - `(0.05, 0.05)`
  - `(0.1, 0.1)`
  - `(0.2, 0.2)`
- 重新 render RGB / depth
- 比较与 base render 的差异

结果是：

```text
rgb_mean_abs_change = 0.0
depth_mean_abs_change = 0.0
```

即使到 `0.2 / 0.2` 这种量级，前向 render 结果仍然完全不变。

这说明一个非常严重的问题：

**当前 forward 看起来对 pose delta 根本不敏感。**

### 5.3 但 backward 却给出了非零 pose 梯度

更诡异的是，autograd 诊断里：
- `grad_rgb_only` 对 pose 的梯度非零；
- `grad_depth_legacy_only` 对 pose 的梯度非零；
- `grad_depth_source_aware_only` 对 pose 的梯度也非零，而且更大。

summary 大致是：
- `mean_grad_rgb_rot ≈ 0.265`
- `mean_grad_rgb_trans ≈ 0.0786`
- `mean_grad_depth_legacy_rot ≈ 0.504`
- `mean_grad_depth_legacy_trans ≈ 0.197`
- `mean_grad_depth_src_rot ≈ 1.732`
- `mean_grad_depth_src_trans ≈ 0.550`

这和前向 probe 组合起来，非常像：

**renderer 的 pose-delta forward / backward 存在不一致，或者当前 theta/rho 路径在 forward 中没有真正生效，但 backward 仍返回了梯度。**

### 5.4 这解释了为什么当前现象那么怪

这可以同时解释几件之前看起来很怪的事：

1. **loss 基本不下降**
   - 因为 forward render 对 pose 变化几乎没有响应

2. **pose delta 却一直在涨**
   - 因为 backward 仍然在给非零梯度，optimizer 照样更新

3. **初始 loss 看起来偏低**
   - 因为当前 target 本来就建立在当前 render 基础上（尤其 depth）；
   - 再加上 pose 变化对 forward 基本不起作用，loss 自然很难大幅变化

4. **RGB loss 还有一点下降**
   - 更像是 exposure 在起作用，而不是 pose 真正改动了 render

### 5.5 regularization 当前也不能真正兜底

另一个细节是：
- `pose_reg_loss = ||rot|| + ||trans||`
- 在初始化为 0 的时候，它的梯度也是 0

所以在最开始：
- regularization 不能提供任何“拉回去”的力；
- 一旦 supervision 分支给了不可靠的 pose 梯度，pose 就会直接被带走。

---

## 6. 现在应该如何判断问题

到当前阶段，问题已经不再是单一的“depth coverage 少”。更准确地说，有两个层次：

### 6.1 Upstream 问题：基本已经推进到可用程度

- train mask 已基本可用；
- M5 densify 也已把 non-fallback depth 区域抬到了有训练意义的量级；
- 因此 upstream 现在**不是最先该怀疑的地方**。

### 6.2 Downstream 问题：Stage A pose/render 连通性需要优先审计

当前最优先的问题已经变成：

**Stage A 中，pose delta 对 renderer 的前向影响看起来失效，但 backward 却仍返回了非零梯度。**

在这个问题没查清之前：
- 不宜继续靠看 loss 曲线判断 depth target 是否有效；
- 更不宜直接进入 Stage B；
- 否则可能会把一个 forward/backward 不一致的 pose path 继续带入 joint refine。

---

## 7. 当前结论（供后续引用）

当前结论固定为：

1. **train mask 问题已经基本解决。**
   当前 `train_mask coverage ≈ 19.4%`，已不再只是 seed-level support。

2. **verified depth 原始版本确实太稀，但 M5 densify 已显著改善。**
   当前 M5 selected 参数下，整图 `non-fallback ratio ≈ 15.8%`，在 `train_mask` 内部约 `81.2%` 的区域已是非 fallback depth。

3. **M5-2 说明仅靠 densify + source-aware loss 仍不足以让 Stage A 的 depth loss 动起来。**
   这意味着问题不再只是 fallback 稀释。

4. **当前最大的异常是 Stage A pose/render 路径可能存在 forward/backward 不一致。**
   前向 probe 下，即使较大的 pose delta 也几乎不改变 render；但 backward 仍返回非零 pose 梯度。

5. **因此当前不适合直接进入 Stage B。**
   在修清 Stage A pose path 之前，任何继续往 joint refine 推进的实验都会带着不可信的 pose 更新机制。

6. **当前最合理的下一步是：**
   先把 renderer / pose-delta 的前向与反向连通性审计清楚，再决定是否继续推进第 4 阶段。

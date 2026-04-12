# DESIGN.md - Part3 设计文档

> 本文档记录**当前实际采用**的设计决策、接口定义与默认实验方案。
> 最后更新：2026-04-12 14:25

---

## 1. 系统边界

### 1.1 当前主线边界

当前 Part3 Stage1 主线仍然是 **mask-problem route on top of internal route**，但当前的设计判断已经更新：
- pseudo 不进入 S3PO frontend tracking；
- Stage1 仍然是 standalone refine，不接 `slam.py` 主队列；
- internal route 的主线仍是：
  `part2 full rerun → internal cache → prepare → difix → fuse → verify → train_mask/depth target → Stage A consumer → replay/eval`；
- 旧 EDP 线继续保留，但与 BRPO 新线文件级隔离，不混写；
- **当前最优先的问题不再是根因定位，而是：在补回 S3PO 原始 residual pose 闭环之后，这套 M5 supervision 是否已经足够强。**

也就是说，现在的主线已经不是继续盲目扩 depth loss 或直接进 Stage B，而是：

**先基于已恢复的 `tau -> update_pose -> R/T` 闭环做修复后验证，再决定后续 Stage A / Stage B 的设计。**

### 1.2 当前阶段定义

| 阶段 | 状态 | 说明 |
|------|------|------|
| Phase 1 | ✅ | internal cache 导出 |
| Phase 2 | ✅ | same-ply replay consistency |
| Phase 3 | ✅ | internal prepare + Difix + BRPO verification 并入 |
| Phase 4 | ✅ | mask-only ablation（legacy vs brpo） |
| Phase 5 | ✅ | auditability / provenance 修补 |
| Phase 6 | ✅ | canonical schema 打通：Difix 双侧 + fuse + side-aware/fused BRPO mask + sample pack |
| Phase M1 | ✅ | `fused-first verification` 与 `branch-first` 双模式并行 |
| Phase M2 | ✅ | `seed_support → train_mask` propagation 接入 upstream |
| Phase M2.5 | ✅ | propagation 合理区间收紧到约 `10% ~ 25%` coverage |
| Phase M3 | ✅ | pseudo-view sparse verified depth + blended `target_depth_for_refine` |
| Phase M4 | ✅ | Stage A consumer 显式消费 `train_mask + blended_depth` |
| Phase M4.5 | ✅ | `blended_depth vs render_depth_only` 的 Stage A long eval |
| Phase M5-0 | ✅ | 诊断 M3 depth signal 在 train-mask 内部的结构 |
| Phase M5-1 | ✅ | densify correction field，生成 `target_depth_for_refine_v2` |
| Phase M5-2 | ✅ | source-aware depth loss 接入 Stage A |
| Phase M5-3 | ✅ 最小修复完成 | 已按 S3PO 原始机制补回 `tau -> update_pose -> R/T` 闭环，并完成最小验证 |
| Phase 7B | ❌ 暂缓 | 在 Stage A pose path 修清前，不进入 joint refine |

---

## 2. 当前实现与结论

### 2.1 当前 v1 refine 的性质

当前主入口：`scripts/run_pseudo_refinement.py`

当前 v1 实际做的是：
- 读一张已有 PLY；
- 读 pseudo cache；
- pseudo 相机固定；
- 用 confidence-weighted RGB loss 优化 Gaussian；
- 支持 `legacy|brpo` 两类 confidence mask 来源；
- depth 不作为 v1 的正式几何监督主线。

因此 v1 本质上仍然是 **fixed-pose appearance tuning**。

### 2.2 M3 / M4 / M4.5 之后的判断

到 M4.5 为止，已经确认：
- M3 upstream 已经接通；
- M4 consumer 已经接通；
- M4.5 表明 `blended_depth` 在 Stage A 中是“已接通但偏平”的。

也就是说，M4.5 结束时最合理的判断是：

**depth supervision 已接入，但在当前 Stage A 配置下，对优化的实际牵引还偏弱。**

### 2.3 M5-0 / M5-1 / M5-2 之后的判断

M5 的结果进一步收紧了问题范围：

1. **M5-0 表明旧问题确实存在。**
   在旧 M3 结构下，`train_mask` 内真正落在 verified depth 区域的比例只有约 `8%`，说明 depth signal 结构确实被 coverage 限制得很厉害。

2. **M5-1 表明 upstream densify 是可行的。**
   在选中的参数下：
   - `seed_ratio ≈ 1.56%`
   - `densified_ratio ≈ 14.21%`
   - `non-fallback within train_mask ≈ 81.2%`

   这说明 upstream 已经能把真正有新几何信息的区域扩进大部分 train-mask。

3. **M5-2 表明仅靠 densify + source-aware depth loss 仍不足以让 Stage A 动起来。**
   结果上：
   - `M5 + legacy depth loss`：`loss_depth ≈ 0.0277`，300 iter 中基本不动；
   - `M5 + source-aware depth loss`：`loss_depth ≈ 0.0687`，300 iter 中也基本不动；
   - 但这次已经能明确看到：`loss_depth_seed` 与 `loss_depth_dense` 都被显式接通了。

因此问题已经不再只是“fallback 稀释”或“depth target 太 sparse”。

### 2.4 当前最新判断：Stage A 已补回 S3PO residual-pose 闭环，但效果仍需验证

最新 diagnosis 的关键结论是：

- `gaussian_renderer/__init__.py` 的确把 `cam_rot_delta / cam_trans_delta` 以 `theta / rho` 传进了 Python wrapper；
- 但 `diff_gaussian_rasterization` 的 forward 路径实际只用 `viewmatrix / projmatrix`，不直接消费 `theta / rho`；
- 这一点和 S3PO 原始代码并不矛盾，因为它本来的设计就是：backward 求 `grad_tau`，随后由 `update_pose()` 把 tau 折回 `R/T`；
- 我们当前 Stage A 的问题在于：**没有调用 `update_pose()`，却把 residual 当成会直接改变 forward render 的长期状态来使用。**

更具体地说，probe 下即使到：
- `rot = 0.2`
- `trans = 0.2`

前向 render 的平均与最大变化仍然是 `0.0`。

这和 backward 里的非零梯度一起看，当前最合理的工程判断是：

**pose-delta 路径在 renderer / rasterizer 中很可能存在 forward / backward 不一致。**

所以当前最准确的结论已经变成：

**当前 Stage A 的 loss 不下降，不一定主要是 depth target 不好；更大的嫌疑是 pose refine 本身就没有通过一个可信的 forward path 影响 render。**

---

## 3. 当前锁定的 internal route 设计

### 3.1 Source of truth

正式源数据统一来自：

```text
<run_root>/internal_eval_cache/
```

而不是：
- S3PO 原生顶层空 `render_rgb/`
- 旧 external eval render 目录
- `third_party` 下的临时输出

### 3.2 训练输入的正式层级

正式训练输入统一来自：

```text
<run_root>/internal_prepare/<prepare_key>/pseudo_cache/
```

当前 sample 侧已经有两代 depth target：
- `target_depth_for_refine.npy`：M3 blended depth target
- `target_depth_for_refine_v2.npy`：M5 densified depth target

当前需要区分四层：
- `seed_support_*`
- `train_confidence_mask_*`
- `target_depth_for_refine.npy`
- `target_depth_for_refine_v2.npy`

### 3.3 当前 prepare 流程定义

当前 `prepare_stage1_difix_dataset_s3po_internal.py` 已经支持：

```text
select → difix(left/right) → fuse → verify → pack
```

其中 `verify / pack` 当前已经能够产出：
- `seed_support_*`
- `train_confidence_mask_*`
- `projected_depth_left/right`
- `target_depth_for_refine.npy`
- `target_depth_for_refine_v2.npy`
- `*_source_map.npy`

所以当前 prepare 已经从“verification-ready prototype”推进到“能给 Stage A 提供两代 depth target 的 canonical input layer”。

---

## 4. Canonical schema（当前锁定内容）

### 4.1 sample 级输出

当前 sample schema 的关键部分是：

```text
samples/<frame_id>/
├── render_rgb.png
├── render_depth.npy
├── target_rgb_left.png
├── target_rgb_right.png
├── target_rgb_fused.png
├── seed_support_*.npy
├── train_confidence_mask_brpo_*.npy
├── projected_depth_left.npy
├── projected_depth_right.npy
├── target_depth_for_refine.npy
├── target_depth_for_refine_source_map.npy
├── target_depth_for_refine_v2.npy
├── target_depth_dense_source_map.npy
├── depth_correction_seed.npy
├── depth_correction_dense.npy
└── metadata json files
```

### 4.2 三层 supervision 语义

当前必须明确分开：

1. `seed_support_*`
   - 高精度、低覆盖几何 seed

2. `train_confidence_mask_brpo_*`
   - 训练真正消费的 supervision mask

3. `target_depth_for_refine{,_v2}`
   - depth target
   - `v1` 是 M3 的 sparse-correction + fallback
   - `v2` 是 M5 的 densified correction + fallback

### 4.3 当前 depth target 的设计判断

当前设计上仍坚持：
- 不直接 densify absolute projected depth；
- 以 `render_depth` 作为 dense base；
- densify 的对象是 **log-depth correction field**。

这条设计目前仍然是合理的。M5-1 的结果已经说明：
- correction-field densify 可以把 non-fallback depth 扩到明显更有训练意义的量级；
- upstream 这一侧没有出现“完全不受控”的扩散。

因此当前 upstream 设计判断不需要推翻。

---

## 5. Stage A / Stage B 的当前设计判断

### 5.1 现在是否还适合继续停在 Stage A？

是，而且比之前更确定。

原因不是“depth 还不够好”，而是：

**当前 Stage A 的 pose/render 前向都还没审计清楚，根本不适合把这条不可信的 pose path 再带进 Stage B。**

### 5.2 为什么当前明确不适合进入 Stage B

如果现在直接上 Stage B，会把几个还没解开的风险一起耦合进去：
- pose delta 前向是否真的影响 render？
- backward pose gradient 是否可信？
- 当前 Stage A 的 pose drift 是否只是伪梯度导致？
- densified depth signal 是否真的在驱动 geometry，而不是只在 loss 里“看起来有值”？

在这些问题没拆清楚前，直接进入 Stage B 等于把可疑的 pose 路径直接带入 joint refine。

### 5.3 当前推荐的 Stage A 口径

当前推荐的 Stage A 口径已经分成两层：

**upstream / target 层：**
- `target_side=fused`
- `confidence_mask_source=brpo`
- `stageA_mask_mode=train_mask`
- `stageA_target_depth_mode=blended_depth_m5`

**diagnosis / loss 层：**
- `stageA_depth_loss_mode=source_aware`
- `lambda_depth_seed=1.0`
- `lambda_depth_dense=0.35`
- `lambda_depth_fallback=0.0`

但是在 pose path 修清之前，这套口径更多是 **diagnosis mode**，不是 final training default。

---

## 6. 当前推荐的下一步

### 6.1 当前最优先

当前最优先的下一步不是：
- 继续堆更多 Stage A long run；
- 或者直接进入 Stage B。

而是：

```text
M5 upstream 已证明可行
  ↓
M5 source-aware loss 已接通
  ↓
当前最优先：审计 Stage A pose/render 连通性
  ↓
确认 theta/rho 在 renderer/rasterizer 的 forward/backward 是否一致
  ↓
修清后再重跑 M5 Stage A
  ↓
只有这一步正常后，再讨论是否进入 Stage B
```

### 6.2 下一阶段预计修改的代码

1. `gaussian_splatting/gaussian_renderer/__init__.py`
   - 审计 `theta / rho` 参数在 forward 中是否真的影响 render 结果

2. 底层 rasterizer 绑定 / CUDA 实现
   - 确认 pose-delta 的 forward 与 backward 是否匹配

3. `run_pseudo_refinement_v2.py`
   - 在 pose path 修清前，不继续作为 Stage B 的前置合理性证据

4. `pseudo_loss_v2.py`
   - 当前 source-aware 设计保留，但后续解释要建立在可信的 pose path 上

### 6.3 当前不建议做的事

- 在 Stage A pose forward 都还不敏感时直接进 Stage B
- 在没修清 pose path 前继续用 loss 不下降去反推“depth supervision 无效”
- 在没排除 forward/backward 不一致前继续调大量 loss 权重

---

## 7. 当前总设计判断

当前可以把设计判断固定成一句话：

**M5 upstream 路线是可行的，densified correction target 也已经成形；但 downstream 当前最大的阻塞不是 target 设计，而是 Stage A 的 pose/render 路径存在 forward/backward 不一致嫌疑，必须先修这个问题，之后才值得继续往 Stage B 推进。**

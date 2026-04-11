# DESIGN.md - Part3 设计文档

> 本文档记录**当前实际采用**的设计决策、接口定义与默认实验方案。
> 最后更新：2026-04-12 01:45

---

## 1. 系统边界

### 1.1 当前主线边界

当前 Part3 Stage1 主线已经明确切到 **mask-problem route on top of internal route**，而不是继续把 sparse external GT 或旧 v1 对照当作默认主线。

当前边界是：
- pseudo 不进入 S3PO frontend tracking；
- Stage1 仍然是 standalone refine，不接 `slam.py` 主队列；
- internal route 的目标先固定为：
  `part2 full rerun → internal cache → prepare → difix → fuse → verify → train_mask/depth target → Stage A consumer → replay/eval`；
- 旧 EDP 线继续保留，但与 BRPO 新线文件级隔离，不混写；
- 当前重点已经不是“有没有 BRPO mask”，而是 **如何让 upstream 输出以正确形式被 Stage A 消费**。

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
| Phase M4.6 | 当前主线 | 诊断为什么 depth signal 已接通但在 Stage A 中基本不动 |
| Phase 7B | 后续主线 | Gaussian + pseudo pose joint refinement |

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
- depth 当前不作为 v1 的正式几何监督主线。

因此 v1 更像 **fixed-pose appearance tuning**，不适合作为 M3/M4 的最终承载器。

### 2.2 Phase 4 之后的关键判断

已完成的 3-frame internal ablation 说明：
- `legacy mask` 的 replay 结果优于 `brpo mask`；
- 但这不等价于 BRPO verification 无效；
- 更合理的解释是：**当前 BRPO verification 产出的信号，在旧 v1 消费方式下没有被以最合适方式使用。**

因此主线判断切换为：
1. 先把 verify 输出拆成 `seed_support / train_mask / depth target` 三层；
2. 再让 `run_pseudo_refinement_v2.py` 明确消费它们；
3. 最后才判断 BRPO-style supervision 本身是否有效。

### 2.3 M3 / M4 / M4.5 之后的最新判断

当前已经可以确认三件事：

1. **M3 upstream 已经接通。**
   `target_depth_for_refine.npy` 现在是 blended target，而不是单纯 render-depth fallback 占位符。

2. **M4 consumer 已经接通。**
   `run_pseudo_refinement_v2.py` 现在已能显式区分：
   - `train_mask`
   - `seed_support_only`
   - `blended_depth`
   - `render_depth_only`
   并把实际使用来源写进 `stageA_history.json`。

3. **M4.5 表明 depth signal 是“接通但偏平”的。**
   在 `blended_depth_long` 与 `render_depth_only_long` 的 300-iter Stage A 对照中：
   - `blended_depth` 的 depth loss 非零；
   - `render_depth_only` 的 depth loss 近似为零；
   - 但 `blended_depth` 的 depth loss 在当前设置下几乎不下降。

所以当前最准确的结论不是“depth 没有被接入”，而是：

**depth supervision 已接入，但在当前 Stage A 配置下，对优化的实际牵引还偏弱。**

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

当前需要明确分成三层：
- `seed_support_*`：高精度几何种子
- `train_confidence_mask_*`：训练真正消费的 mask
- `target_depth_for_refine.npy`：M3 的 blended depth target

也就是说：
- `internal_eval_cache/` 是长期复用的 source layer；
- `internal_prepare/` 是 canonical training-input layer；
- `part3 output/` 是实验输出层。

### 3.3 当前 prepare 流程定义

当前 `prepare_stage1_difix_dataset_s3po_internal.py` 已经支持：

```text
select → difix(left/right) → fuse → verify → pack
```

其中 `verify / pack` 当前已不再只产单层 support，而是进入：
- `verification_mode=branch_first | fused_first`
- `seed_support_*`
- `train_confidence_mask_*`
- `projected_depth_left/right`
- `target_depth_for_refine.npy`
- `target_depth_for_refine_source_map.npy`

这意味着当前 prepare 已经从“verification-ready prototype”演进到“consumer-ready canonical input layer”。

---

## 4. Canonical schema（当前锁定内容）

### 4.1 sample 级输出

当前 sample schema 已锁定为：

```text
samples/<frame_id>/
├── camera.json
├── refs.json
├── source_meta.json
├── render_rgb.png
├── render_depth.npy
├── ref_rgb_left.png
├── ref_rgb_right.png
├── target_rgb_left.png
├── target_rgb_right.png
├── target_rgb_fused.png
├── target_depth.npy
├── target_depth_for_refine.npy
├── target_depth_for_refine_source_map.npy
├── target_depth_for_refine_meta.json
├── verified_depth_mask.npy
├── projected_depth_left.npy
├── projected_depth_right.npy
├── projected_depth_valid_left.npy
├── projected_depth_valid_right.npy
├── seed_support_left.npy
├── seed_support_right.npy
├── seed_support_both.npy
├── seed_support_single.npy
├── train_confidence_mask_brpo_left.npy
├── train_confidence_mask_brpo_right.npy
├── train_confidence_mask_brpo_fused.npy
├── train_support_left.npy
├── train_support_right.npy
├── train_support_both.npy
├── train_support_single.npy
├── confidence_mask_brpo_left.npy       # alias -> train mask
├── confidence_mask_brpo_right.npy      # alias -> train mask
├── confidence_mask_brpo_fused.npy      # alias -> train mask
├── support_left.npy                    # alias -> seed support
├── support_right.npy                   # alias -> seed support
├── support_both.npy                    # alias -> seed support
├── support_single.npy                  # alias -> seed support
├── fusion_meta.json
├── verification_meta.json
└── diag/
```

### 4.2 三种 target_side 的消费语义

推荐并已实现的消费语义：
- `target_side=left`：主要消费 `train_confidence_mask_brpo_left`
- `target_side=right`：主要消费 `train_confidence_mask_brpo_right`
- `target_side=fused`：主要消费 `train_confidence_mask_brpo_fused`

当前规则是：
- **不要再让一个 mask 无差别服务所有 target mode**；
- `fused target` 应优先配对 `fused confidence`；
- `seed_support_only` 只作为 debug / ablation 模式，不作为默认训练层。

### 4.3 depth 字段的设计定位

当前 depth 字段分三层：
1. `render_depth.npy`：当前地图渲染深度，用于诊断 / consistency / fallback
2. `target_depth.npy`：兼容字段，当前不作为主线 depth supervision
3. `target_depth_for_refine.npy`：当前正式主线使用的 **blended depth target**

当前规则是：
- `support_both`：使用双边 verified depth（当前 `average` 组装）
- `support_left`：使用 left verified depth
- `support_right`：使用 right verified depth
- unsupported：回退到 `render_depth`

因此当前 `target_depth_for_refine.npy` 的语义是：

**verified depth as correction signal + render depth as fallback**

而不是“纯 sparse BRPO depth 替换整张图 depth”。

---

## 5. fuse 与 train-mask 的设计判断

### 5.1 为什么当前需要 fuse + train-mask + blended depth 三层同时存在

因为当前问题不是单一环节坏掉，而是三个语义层原本混在了一起：
- verification 需要高精度几何 evidence；
- training 需要更可用的 supervision coverage；
- depth target 需要明确区分 verified correction 与 fallback。

因此当前设计不再尝试“一个 support / 一个 mask 解决所有问题”，而是显式拆层。

### 5.2 当前 coverage 的设计判断

这里要明确区分两种 coverage：

1. **train_mask coverage**
   - 当前目标区间：**约 `10% ~ 25%`**
   - 当前主线 prototype：约 `18% ~ 20%`
   - 这是训练真正消费的 mask 覆盖

2. **verified depth coverage**
   - 当前量级：**约 `1.5% ~ 1.6%`**
   - 这是 source map 中真正来自 BRPO verified depth 的区域
   - 不应把它和 train_mask coverage 混为一谈

因此当前判断是：
- `~1.5%` 如果当 train_mask，确实太稀；
- 但如果当“depth correction source”，它可以接受，只是当前显得偏保守。

---

## 6. Stage A / two-stage refine 的当前设计判断

### 6.1 为什么当前值得继续停在 Stage A

当前更缺的是：
- depth signal 为什么不动的诊断；
- upstream coverage 与 downstream optimizer sensitivity 的分离判断；
- replay / eval 层面对 blended depth 是否真有收益的证据。

这些都比“尽快进入 Stage B”更优先。

### 6.2 为什么当前不适合直接上 full Stage B

因为现在如果直接上 full Stage B，会把几个未回答问题混在一起：
- verified depth 太稀，还是足够？
- depth loss 设计不够敏感，还是权重/正则问题？
- 当前 pose delta 是否被 RGB 主导？
- blended depth 的收益能否在 replay / eval 里真正体现？

在这些问题没拆清楚前，直接上 Stage B 会增加调试耦合。

### 6.3 当前推荐的 Stage A 口径

当前推荐默认口径：
- `target_side=fused`
- `confidence_mask_source=brpo`
- `stageA_mask_mode=train_mask`
- `stageA_target_depth_mode=blended_depth`

同时保留 debug / ablation 开关：
- `seed_support_only`
- `legacy`
- `render_depth_only`

当前 Stage A 的最关键要求不是“马上拉高指标”，而是：
- **history 必须可审计**；
- **不同 source mode 必须能真实区分**；
- **不要再让 consumer 悄悄 fallback 到隐式默认值。**

---

## 7. 下一步最适合做什么

### 7.1 当前最优先

当前最优先的下一步不是：
- 直接进 Stage B；
- 或继续堆更多 `blended vs render` 长跑而不解释原因。

而是：

```text
M3 已完成
projected_depth + blended target_depth_for_refine
  ↓
M4 已完成
Stage A consumer 显式接通
  ↓
M4.5 已完成
blended_depth vs render_depth_only long eval
  ↓
M4.6（下一步）
depth-flatness diagnosis
  ↓
决定是改 Stage A sensitivity
还是回 upstream 做受控扩张
```

### 7.2 下一阶段预计修改的代码

1. `scripts/run_pseudo_refinement_v2.py`
   - 增强 history / diagnostics
   - 必要时增加更细的 per-view / per-mode 统计

2. `pseudo_branch/pseudo_loss_v2.py`
   - 检查当前 depth loss 的有效梯度和归一化方式
   - 评估是否需要更显式的 verified-region emphasis

3. `pseudo_branch/brpo_depth_target.py` / `brpo_train_mask.py`
   - 如果诊断表明 current verified depth 太弱，做更受控的覆盖扩张研究

4. `scripts/replay_internal_eval.py`
   - 如继续放大 Stage A 实验，补 replay 对照闭环

### 7.3 当前不建议优先做的事

- 在 M4.6 前直接进入 full Stage B
- 在没有新证据前，把 verified depth 覆盖从 `~1.5%` 粗暴拉到高覆盖区
- 在没有区分 train_mask 与 verified depth 语义前，再次混用“coverage”这一指标

---

## 8. 评估口径说明

当前需要明确区分四层结论：
- `external GT`：固定相机 appearance diagnostic
- `internal replay`：当前真正的 map/protocol 闭环口径
- `M3 upstream`：verify / train-mask / blended depth target 是否正确定义
- `M4/M4.5 Stage A`：consumer 是否真正读到并使用这些 target

当前 M4.5 的结论是：
- `blended_depth` 与 `render_depth_only` 在 Stage A 中已经**可区分**；
- 但区分主要体现在 depth loss 是否非零，而不是当前优化曲线是否大幅拉开；
- 因此下一步应优先解释“为什么 depth 不动”，而不是匆忙宣布 depth supervision 已经有效。

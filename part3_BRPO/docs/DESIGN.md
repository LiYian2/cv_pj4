# DESIGN.md - Part3 设计文档

> 本文档记录**当前实际采用**的设计决策、接口定义与默认实验方案。
> 最后更新：2026-04-13 13:25

---

## 1. 系统边界

### 1.1 当前主线边界

当前 Part3 Stage1 主线仍然是 **mask-problem route on top of internal route**，但当前的设计判断已经更新：
- pseudo 不进入 S3PO frontend tracking；
- Stage1 仍然是 standalone refine，不接 `slam.py` 主队列；
- internal route 的主线仍是：
  `part2 full rerun → internal cache → prepare → difix → fuse → verify → train_mask/depth target → Stage A consumer → replay/eval`；
- 旧 EDP 线继续保留，但与 BRPO 新线文件级隔离，不混写；
- **当前最优先的问题不再是“链路是否断开”，而是：在补回 S3PO residual pose 闭环之后，depth 只表现为弱下降，且 absolute pose prior 的权重尺度尚未标定。**

也就是说，现在的主线已经不是继续做 pose 路径真伪排查，而是：

**先基于已恢复的闭环，完成 absolute pose prior 的尺度化/权重标定，并完成 A.5(minimal micro-joint) 的可行性筛查，再决定是否推进 Stage B。**

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
| Phase M5-4 | ✅ 已接入待标定 | absolute pose prior（`SE3_log + R0/T0 + lambda_abs_pose`）已接入并完成 smoke 对照 |
| Phase 7B | ❌ 暂缓 | 在 Stage A 仍是弱有效且 abs prior 未标定前，不进入 joint refine |

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

### 2.4 当前最新判断：闭环已恢复，当前瓶颈是 absolute pose prior 的尺度标定
### 2.5 A.5(midpoint) 最小验证结论（分口径修正）
- 旧记录中的 midpoint A.5 正向结果只对应旧口径/非严格顺序实验；
- 严格 `midpoint8 + M5 + source-aware + StageA/A.5/B` 新执行表明：StageA replay 基本不变，A.5(`xyz+opacity`) replay 退化，StageB120 进一步退化；
- 因此当前不能再把 `A.5(midpoint)` 视为可靠 StageB 回退基线，必须先修正 pipeline handoff 与评估逻辑。


- 伪帧策略从 3 帧扩展为 `kf gap midpoint` 的 8 帧覆盖；
- 在 80 iter 快速对照下：
  - `StageA5(xyz)` 相比 `StageA` 未显示改进；
  - `StageA5(xyz+opacity)` 的 `loss_total/loss_depth` 均优于 `StageA`；
  - `StageA5(xyz)+abs prior(1.0/0.1)` 在该口径下未优于 baseline。
- 当前设计决策：A.5 继续以 `xyz+opacity` 作为 next-default 候选，`xyz-only` 不作为优先推进路径。


最新工程与实验结论是：

- `apply_pose_residual_()` 已把 S3PO 的 residual pose 闭环补回，Stage A 不再是假优化；
- 300 iter 下 `loss_depth` 会下降，但幅度仍小，属于“弱有效”；
- 由于 residual 每步清零，`pose_reg` 基本为 0，无法约束累计 drift；
- 因此 absolute pose prior 是必要项，但当前权重尺度未标定：
  - `lambda_abs_pose=0.1` 与 noabs 几乎重合；
  - `lambda_abs_pose=100` 能显著压 drift（`abs_pose_norm mean 0.00154 -> 0.00034`），但会轻微压制 depth 下降。

所以当前最准确的设计判断是：

**Stage A 的主矛盾已经从“pose path 是否可信”转为“如何在 drift 抑制与 depth 对齐之间找到可用权重区间”。**

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

原因不是“pose path 不可信”，而是：

**闭环已修后仍然只有弱下降，且 absolute pose prior 的有效权重区间还没标定；现在直接进 Stage B 会把这个未解耦问题放大。**

### 5.2 为什么当前明确不适合进入 Stage B

如果现在直接上 Stage B，会把几个还没解开的权重耦合问题一起带进去：
- absolute pose prior 太弱时，累计 drift 约束几乎无效；
- absolute pose prior 太强时，会压制 depth 对齐；
- 当前尚未得到“既抑制 drift 又不伤 depth”的稳定默认配置。

在这个权衡没标定清楚前，直接进入 Stage B 会把可调问题变成更难解释的 joint 问题。

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

### 6.1 已完成的 A+B（本轮）

- A：split+scaled abs prior 已落地：`lambda_abs_t/r + robust + scene_scale`；
- B：梯度贡献诊断已落地：在线 `--stageA_log_grad_contrib` + 独立脚本 `diagnose_stageA_loss_contrib.py`；
- 扫描已完成：`6组×80iter` 小网格 + `top3×300iter` 深跑。

### 6.2 当前结论（来自 A+B 实测）

1. `lambda_abs_t` 增大可以稳定压低 `abs_pose_rho_norm/theta_norm`（drift 抑制有效）；
2. depth 末值在各组差异很小（当前没有“同时显著压 drift 且显著提升 depth”的组）；
3. 因此 abs prior 当前角色更接近“稳定器”，不是“depth 改善器”。
4. depth-heavy 复跑后仍成立：best-depth 与 best-drift 组不重合。

### 6.3 下一步（仍停留在 Stage A）

- depth-heavy 口径复跑（同 top3 组）已完成；
- 已固化双默认：
  - drift-prioritized：`beta_rgb=0.7, lambda_abs_t=3.0, lambda_abs_r=0.1`
  - depth-prioritized：`beta_rgb=0.3, lambda_abs_t=1.0, lambda_abs_r=0.1`
- 下游价值验证结论：在 270 帧 replay 上三组指标近似重合（差异 1e-4~1e-6），当前不能宣称 Stage A 口径已带来下游提升；
- 下一步应优先扩大受影响帧范围或进入 A.5 最小验证，再决定是否推进 Stage B。

### 6.4 当前不建议做的事

- 把 legacy `lambda_abs_pose` 继续当主入口；
- 把 `lambda_abs_t=3.0` 直接视作“最终默认”（它只证明 drift 抑制有效）；
- 在 tradeoff 未固化时推进 Stage B。

## 7. 当前总设计判断

当前可以把设计判断固定成一句话：

**M5 upstream 路线可用，Stage A pose 闭环也已修回；A+B 与 value-eval 已确认当前 abs prior 主要抑制 drift，而下游指标尚无可见收益。仅靠当前 Stage A 三帧位姿更新不足以产生 270 帧 replay 提升，下一步应转向“扩大影响范围/最小A.5验证”后再决策 Stage B。**


## 8. Internal cache -> StageA/A.5 -> StageB 当前方法与 pipeline（2026-04-13）

### 8.1 当前已落地 pipeline

```text
part2 internal cache + pseudo_cache(midpoint 8 frames)
    -> StageA / StageA.5 refine (v2)
    -> StageB conservative joint refine (v2, real anchor + pseudo RGBD)
    -> refined_gaussians.ply + stageA/stageB history + replay eval
```

关键实现点：
1. StageA.5 已支持 `xyz` / `xyz_opacity` 微调；
2. StageB 已支持 `--stage_mode stageB` 与 real branch (`--train_manifest`, `--num_real_views`, `--lambda_real`)；
3. StageB 当前采用 conservative 口径：`xyz_opacity`、不引入 densify/prune、以 replay gate 决策扩展。

### 8.2 当前方法判断（来自最新实测）

1. 短程 StageB（120iter）相对 A.5 baseline 为正向（gate PASS）；
2. 长程 StageB（300iter）相对 A.5 baseline 退化（gate FAIL）；
3. 说明当前 StageB 不是“不可用”，而是“后段稳定性不足”，存在长程漂移/权重失衡风险。

### 8.3 设计层面的下一步优化方向

1. 在方法层面引入“短程 gate + 早停窗口”作为默认执行策略；
2. 在训练层面加入分段调度（后段降 lr / 分段冻结 pose 或 opacity）；
3. 在损失层面扫描 `lambda_real : lambda_pseudo` 平衡，避免后段 real/pseudo 目标冲突；
4. 在工程层面保留 A.5 作为硬回退基线，StageB 必须满足 non-regression gate 才可晋级。


## 9. Strict midpoint8 M5 pipeline 复盘后，当前设计层面的新判断（2026-04-13 夜）

### 9.1 StageA 当前在设计上是“诊断分支”，不是“能被 replay 直接验证的有效阶段”

严格 M5 跑完之后，一个关键事实被确认：
- StageA 只优化 pseudo pose + exposure；
- Gaussian 完全冻结；
- 导出的 `refined_gaussians.ply` 与输入 PLY byte-identical；
- `replay_internal_eval.py` 只消费 PLY，不消费 StageA 导出的 pseudo camera states。

因此，**当前设计下用 270 replay 去判断 StageA 是否有效，在结构上就是失真的**。即使 StageA 真让 pseudo pose 稍微变好，只要 Gaussian 没更新、相机状态又没传入下游，replay 就不会反映出来。

### 9.2 当前真正的 stage handoff 缺了一段：pseudo camera states 没有跨阶段继承

`run_pseudo_refinement_v2.py` 当前流程是每次重新从 pseudo cache 载入 viewpoint，然后调用 `make_viewpoint_trainable(...)`。A.5 / StageB 没有读取前一阶段导出的 `pseudo_camera_states_final.json`。

这意味着：
1. StageA 不能真正初始化 A.5；
2. A.5 也不能真正初始化 StageB 的相机侧；
3. 当前所谓 `A -> A.5 -> B`，本质上更像：
   - Gaussian 侧部分 warm start；
   - 相机侧每阶段重新开局。

这与论文中“先稳 pose，再进 joint stage”的顺序语义存在实质差异。

### 9.3 当前 midpoint8 M5 的几何监督强度，不足以支持“靠 StageA 单独把 pose 拉起来”

本轮 midpoint8 M5 实测：
- 平均 `train_mask ≈ 18.7%`
- 平均 `seed + dense ≈ 3.49%`
- 即 train-mask 内只有约 `18.65%` 是 non-fallback 几何监督

所以当前 source-aware depth loss 在这组 midpoint pseudo 上并不是“train-mask 内大部分区域都有可用几何”，而是“RGB mask 覆盖 > depth 真监督覆盖”。这会导致：
- pose 梯度连通，但 leverage 不够大；
- StageA 相机更新量很小；
- A.5 / B 更容易走向 pseudo-side 局部拟合，而不是全局几何改善。

### 9.4 A.5 / StageB 当前更像“局部补偿器”，而不是 paper-style joint optimizer

在严格跑法里：
- A.5(`xyz_opacity`) 明显降低 pseudo-side loss；
- 但 replay 退化；
- StageB120 再进一步退化。

这说明当前 joint 部分虽然在优化，但优化方向不对。核心原因更接近：
1. joint 自由度太窄（只开 `xyz/opacity`，没开 scaling/rotation/SH 等更完整的 joint parameter space）；
2. 缺少论文里的 `scene perception Gaussian management`；
3. StageB real branch 只是 sparse real RGB anchor，没有足够强的几何纠偏能力。

### 9.5 现在更像“结构模块问题优先于参数问题”

目前不是说参数完全不重要，但优先级已经变了：
- `beta_rgb`、lr、`lambda_real/pseudo` 这些都可以再调；
- 但如果不先修正 `StageA评估口径 + 相机状态handoff + StageB几何锚点不足 + 缺失paper关键joint机制`，继续扫参数大概率只是在错误结构上做局部最优。


## 10. 修复 handoff 之后的设计更新（2026-04-13 夜）

### 10.1 之前的结构诊断被验证为真问题，而不是误判

通过 sequential smoke 与 short rerun 已明确：
- 旧版 `run_pseudo_refinement_v2.py` 的确没有把前一阶段的 pseudo camera states 传给下一阶段；
- 修复后，`StageA final == StageA.5 init`、`StageA.5 final == StageB init` 已被逐项验证；
- 因此前面的“pipeline 不是严格顺序 handoff”判断是正确的。

### 10.2 handoff 修复后，pipeline 设计更接近原始意图，但仍未达到 paper-style强joint

修复后，当前设计至少满足：
1. StageA 的 pseudo pose 状态可以真实初始化 A.5；
2. A.5 的 pseudo pose 状态可以真实初始化 B；
3. 因此 `A -> A.5 -> B` 终于具备了相机状态层面的顺序语义。

但修复后 rerun 仍然为负，说明剩下的设计问题不只是 handoff：
- 几何信号强度仍不足；
- current joint space 仍太窄（`xyz/opacity only`）；
- real-anchor StageB 仍缺强几何约束；
- paper中的 `scene perception Gaussian management` 仍未进入 refine loop。

### 10.3 未来 staged pipeline 的默认设计规范应该升级

这次事件说明，以后设计 staged experiment 时，必须把下面四项当成前置验证，而不是事后排查：
1. `stageN final -> stageN+1 init` 是否逐项对齐；
2. 评估指标是否真的消费了本阶段会变的 artifact；
3. 状态汇总是否记录“真实变化”而不是仅记录 post-update residual buffer；
4. 在长跑之前必须做 0-iter / short smoke continuity check。

这不只是工程习惯，而应视为 pipeline design 的一部分。

## 11. P1 bottleneck review：当前 refine 设计的主矛盾（2026-04-13 夜）

### 11.1 当前 source-aware refine 的真实监督作用域非常窄
从代码上看，`pseudo_loss_v2.py` 中的 RGB 和 depth 都通过同一个 `confidence_mask` 做 masked supervision，因此：
- pseudo RGB 只在 train-mask 内生效；
- pseudo depth 还要再受 `depth_source_map` 与 `lambda_depth_*` 约束；
- 在当前设置 `lambda_depth_fallback=0` 下，fallback depth 区域完全不提供 depth 梯度。

因此，这条链路不是“整张 pseudo 图都在监督”，而是只在一个较小候选区域内做 supervision，而其中大部分区域还是 RGB-only / depth-fallback。

### 11.2 当前 midpoint8 M5 在这个 case 上更像弱修正，而不是强几何驱动
量化结果表明：
- train-mask 平均覆盖约 `18.7%`；
- 真正 non-fallback depth 平均仅约 `3.49%`；
- support 区域内 `target_depth_for_refine_v2` 相对 `render_depth` 的平均修正仅约 `1.53%`。

这意味着当前 M5 target 更多是在已有 render depth 周围做微弱 nudging，而不是提供大范围、强幅度的几何校正信号。

### 11.3 当前 A.5 / B 的优化对象却是全局 Gaussian 参数
`scheduler + gaussian_param_groups` 当前直接打开全局：
- `gaussians._xyz`
- `gaussians._opacity`

即使真实梯度仍受 visibility 影响，优化语义上依然是“全局点云参数可训练”。当 supervision 只有 8 个 pseudo views 且 depth 有效区域仅 3.49% 时，这种设计天然容易出现 underconstrained global refinement：
- pseudo-side loss 降；
- held-out replay 退化。

### 11.4 设计层面的核心矛盾
所以目前真正的设计矛盾不是单一 bug，而是：
`弱而局部的 pseudo signal` 作用在 `全局 Gaussian refine` 上。

这比“某个 lr 没调对”更根本，也解释了为什么：
1. 修复 handoff 后结果虽略好，但仍然整体为负；
2. A.5 / B 在优化日志上可以看起来正常，但 replay 仍掉。

### 11.5 后续设计应优先遵守的原则
1. 先验证 pseudo signal 是否足够强，再决定是否打开全局 Gaussian；
2. 如果 signal 只在局部可靠，则优化对象也应局部化 / gated；
3. 新 pseudo cache 必须在长跑前过 `support ratio + correction magnitude` 的 signal gate；
4. replay 指标必须被视为独立 gate，不能再由 pseudo-side loss 代理。

## 12. 下一阶段设计更新：signal semantics first, stable consumer second（2026-04-14）

基于更新后的 BRPO 差异分析，当前设计层面新增三点判断：

1. 你们和 BRPO 的核心差异，不在“有没有双向 fusion”，而在“confidence / train-mask / depth target / refine consumer”这套语义是否闭合；
2. A 层应先于 B 层推进：先做 `continuous confidence + agreement-aware support + confidence-aware densify`，再考虑更密 pseudo；
3. C 层当前最合理的形态不是 full SPGM，而是：
   - loss side：active-support-aware depth reweight；
   - optimizer side：pseudo 分支 local Gaussian gating；
   - curriculum side：StageB 从单段式改为前强后稳的两段式。

对应的执行化方案已单独写入：
- `docs/SIGNAL_SEMANTICS_AND_STABLE_REFINEMENT_PLAN_20260414.md`
## 13. S1.2 设计落地补充（2026-04-14）

当前 consumer 侧已完成第一版语义分离：
1. RGB 分支可显式选择 `raw_confidence`；
2. depth 分支可显式选择 `train_mask` 或 `seed_support_only`；
3. loss 侧不再强制 RGB 与 depth 共用一张 confidence mask。

这一步的意义是先把 BRPO 风格的 raw confidence semantics 和你们后续为 coverage 扩大的 depth train region 解耦，避免 propagation 后的大 mask 继续无差别地喂给 RGB supervision。
## 14. S1.3 smoke 后的设计判断（2026-04-14）

2-frame smoke 说明三件事：
1. semantic wiring 已经打通：continuous confidence、RGB/depth mask 分离、confidence-aware densify 都能真实产出并被 consumer 读取；
2. 但 S1.3 首轮参数过于保守，densified coverage 从约 `2.20%` 被压到约 `0.07%` 量级，明显过头；
3. 在这种过窄 supervision 下，StageA 短跑出现更小 pose drift 但更高 loss，这更像“约束过严”而不是“更优信号”。

因此下一步不是直接扩到长跑，而是先把 confidence-aware densify 阈值调回到“能抑制低质 patch、但不把 dense support 压没”的区间，再做 8-frame 短跑语义对照。
## 15. 8-frame retuned compare 后的设计判断（2026-04-14 晚）

和 2-frame 首轮 smoke 相比，retuned confidence-aware densify 已明显回调到更合理区间：
- dense_valid_ratio 不再塌到 `0.1%` 量级；
- 但它仍比 baseline 更保守，且 StageA-20iter 上 conf-aware 组 loss 仍偏高。

这说明当前最合理的解释不是“语义方案错了”，而是：
1. raw-confidence + train-mask split 已经打通；
2. confidence-aware densify 方向是对的；
3. 但当前阈值/权重仍未调到能稳定胜过 baseline 的区间。

因此现在进入 StageB curriculum 仍然偏早。更合理的顺序是：先把 E1（8-frame 短跑 semantics compare）收口，再决定是否进入第4步。


## 16. E1 之后的设计判断：midpoint 只是先验，不是最优性保证（2026-04-14 夜）

E1 的 lightweight selection run 给出一个新的设计事实：在当前 re10k case 的 8 个 gap 中，前 6 个 gap 的最优候选都从 `midpoint` 偏向了 `2/3`。

这说明：
1. pseudo 位置本身就是 signal quality 的一阶变量，而不是固定后可忽略的采样细节；
2. `midpoint` 更像一个方便的默认先验，而不是受当前 geometry / overlap / visibility 自动保证的最优点；
3. 当前 signal 改善不一定表现为 `both_ratio` 单独上升，更可能体现为 `verified_ratio`、`continuous confidence`、左右平衡项与 correction magnitude 的联合改善。

因此，后续设计上应把 pseudo selection 单独视为一个可学习/可搜索层，而不是默认内嵌在 pipeline 常量里的固定规则。

同时，这轮结果也提示一个方法学边界：
- 当前 E1 仍是 render-based lightweight verify；
- 它证明了“选点值得改”，但还没有证明“改完选点后，完整 pseudo pipeline + short refine 一定更优”；
- 所以下一步应先做 `signal-aware-8` 的 apples-to-apples verify/pack + short StageA compare，再决定是否直接推进 E2。

## 17. E1.5 之后的设计判断：selection 的收益先体现在 verified-to-dense 传导（2026-04-14 深夜）

E1.5 的正式短对照说明，support-aware pseudo selection 的收益模式不是“raw both-support 全面暴涨”，而是更细一点：
1. `support_ratio_both` 可以不升反降；
2. 但 `verified_ratio / continuous confidence / agreement` 仍可同步上升；
3. 更关键的是，这种提升会继续传导到 densify，使 `dense_valid_ratio` 和 `densified_only_ratio` 上升、fallback 下降；
4. 到 short StageA-20 这一层时，收益已经变小，但仍保持同方向的轻微改善。

这说明 selection 的真正作用不是粗暴扩大 support 面积，而是把“更可消费的 verified geometry”前置到 pseudo set 中。

因此，后续设计上应把 E1 看成 signal pipeline 的上游结构改动，而不是单纯的数据采样细节。E2 / E3 也应继续沿这个思路推进：先增强 raw signal 形成，再谈更复杂的 consumer 稳定化。


## 18. E2 之后的设计判断：数量扩张不能替代高质量 pseudo（2026-04-14 深夜）

E2 的正式对照把 multi-pseudo 这条路工程上打通了，但也给出一个很明确的设计事实：当前 case 下，把每个 gap 从 1 个 pseudo 扩到 2 个 pseudo，并没有继续优于 E1 的单 pseudo 最优选点。

更具体地说：
1. `top2 per gap` 最终稳定落成 `1/2 + 2/3`，说明第二名候选确实几乎总是 midpoint，而不是 `1/3`；
2. raw signal 层相对 midpoint8 仍有轻微正向，但 `both-support` 并未变强；
3. densify 层相对 midpoint8 仍提升，但已经弱于 E1 的 `signal-aware-8`；
4. short StageA-20 则出现明显回落，说明“多一个中等质量 pseudo”并没有带来更优的可消费监督，反而在 consumer 侧稀释了整体 signal 质量。

这轮结果的重要含义是：
- 当前主矛盾不是“pseudo 还不够多”；
- 而是“高质量 pseudo 太少，低于 winner 质量的补充样本会把 supervision 拉回到中等水平”；
- 因此，后续更值得推进的不是继续无差别加 pseudo 数量，而是增强 winner pseudo 的 verified robustness。

据此，E3 的合理启动方式也更清楚了：
- 若继续做 multi-anchor verify，应优先作用在 E1 通过正式对照的 `signal-aware-8` 上；
- 不应把当前 E2 16-pseudo set 直接当作新的默认基底；
- 如果未来还要继续做 E2 分支，更合理的方向应是 conditional / gated allocation，而不是所有 gap 统一 top2。


## 19. 对 mask/depth/confidence pipeline 的设计判断：它更像 signal gate，而不是持续增益引擎（2026-04-14 更深夜）

E1/E1.5/E2 连起来之后，当前设计层面的最重要新判断是：

这条 `mask / depth / confidence` pipeline 的真实角色，更像一个上游 signal gate / ranking / verification pipeline，而不是一个还能持续制造大量新增监督的主引擎。

原因有三点：
1. E1 证明“选点”能带来真实提升，说明这条线不是伪方向；
2. E2 证明“加数量”不能替代“保 winner”，说明其主增益不是来自 pseudo 数量扩张；
3. 当前 consumer 采用固定 `num_pseudo_views=4` 的随机采样机制，意味着当 pseudo 池变大但质量分层明显时，次优样本会稀释 strong winner 的训练曝光。

这带来一个关键设计结论：
- 对这条 pipeline，继续加 pseudo、加 mask 规则、加 confidence 细节，并不必然等于更强 supervision；
- 当 raw verified signal 仍然只在低百分比量级时，这条线更擅长做的是“筛出更可消费的少量 winner”，而不是把弱信号自动放大成强监督。

因此后续设计优先级应调整为：
- 把 E3 当作这条线的 final probe；
- 如果 E3 也只带来很小收益，则应明确把主矛盾转回 `weak local supervision -> global refinement` 的结构失配，而不是继续在 mask/depth/confidence 支线内部做高强度增量优化。

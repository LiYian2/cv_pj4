# charles.md

> 记录时间：2026-04-12 15:20
> 主题：Part3 当前已完成 M5 upstream、修回 Stage A 的 S3PO residual pose 闭环；现在进入“修复后效果是否足够强”的评估阶段。

## 当前到哪一步了

当前主线仍然是 **Re10k-1 full internal route + mask-problem route**。

已经完成：
1. `part2 full rerun -> internal_eval_cache`
2. same-ply `replay_internal_eval` 一致性验证
3. `prepare_stage1_difix_dataset_s3po_internal.py` 打通 `select -> difix -> fusion -> verify -> pack`
4. `M1 / M2 / M2.5`：`seed_support -> train_mask`，当前 `train_mask coverage ≈ 19.4%`
5. `M3`：`projected_depth_*` + `target_depth_for_refine.npy`
6. `M5-0 / M5-1`：depth signal diagnosis + `target_depth_for_refine_v2.npy` densify target
7. `M5-2`：`run_pseudo_refinement_v2.py` 已支持 `blended_depth_m5 + source_aware depth loss`
8. **关键修复已完成**：Stage A 现在会在每次 `optimizer.step()` 后执行 `apply_pose_residual_()`，把 `tau -> R/T` 折回，恢复 S3PO 原始 residual pose 闭环

## 当前最重要的结论

1. **之前的“loss 不动”不只是 depth sparse，也有 Stage A pose 闭环断掉的问题。这个闭环现在已经修回。**
2. 修复后，Stage A 不再是假优化：render 会随 pose 更新而变化，loss 也不再完全平。
3. 但当前 **depth 只表现为弱下降**，还没有强到足以直接进入下一阶段。
4. 当前新的结构问题是：**`pose_reg` 只约束 residual，本轮 step 后 residual 会清零，所以它不能约束累计 pose drift。**
5. 现在不要直接进 Stage B。

## 下一次来先看什么

按顺序看：
1. `/home/bzhang512/CV_Project/part3_BRPO/docs/CURRENT_MASK_PROBLEM.md`
2. `/home/bzhang512/CV_Project/part3_BRPO/docs/STATUS.md`
3. `/home/bzhang512/CV_Project/part3_BRPO/docs/DESIGN.md`
4. `/home/bzhang512/CV_Project/part3_BRPO/docs/CHANGELOG.md`
5. 这一份 `charles.md`

## 关键路径

### 当前 pseudo cache prototype
- `/home/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_prepare/re10k1__internal_afteropt__brpo_proto_v4_stage3/pseudo_cache/`

### M5-0 / M5-1 分析与 densify
- `/home/bzhang512/CV_Project/output/part3_stage1_internal/re10k-1/full/2026-04-12_m50_m51_eval/analysis/`

### M5-2 source-aware loss 对照
- `/home/bzhang512/CV_Project/output/part3_stage1_internal/re10k-1/full/2026-04-12_m52_stageA_loss_eval/`

### pose 闭环最小修复 smoke
- `/home/bzhang512/CV_Project/output/part3_stage1_internal/re10k-1/full/2026-04-12_m54_pose_fix_smoke/`

### 修复后的 300-iter 参数/规模验证
- `/home/bzhang512/CV_Project/output/part3_stage1_internal/re10k-1/full/2026-04-12_m55_pose_fix_scale_eval/`
- 重点看：`analysis/m55_summary.json`

## 下一次优先做什么

只做一件事：

**继续做修复后的 Stage A 结构评估，不进 Stage B。**

优先方向：
1. 给 Stage A 补一个 **absolute pose prior**（约束当前 `R/T` 相对初始 pose 的偏移，而不是只约束 residual）
2. 在这个前提下重跑一轮对照：
   - `default + absolute pose prior`
   - `depth_heavy + absolute pose prior`
3. 判断 depth 是否能从“弱下降”变成更明显的下降；如果仍然很弱，再看是否要继续调训练长度 / 权重 / densify 幅度

## 给下一个 Charles 的一句话

**不要再回头找“假优化”的根因了，那个已经修掉了。现在的任务是：在修复后的真实闭环上，判断 M5 depth supervision 到底只是弱有效，还是通过补 absolute pose prior 之后能真正变强。**

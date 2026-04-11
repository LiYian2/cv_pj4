# Charles.md

> 记录时间：2026-04-11 17:20
> 主题：Part3 Stage1 已完成 Phase 6 第一轮；Phase 7A / Stage A 最小版已落地并通过 smoke

## 这次已经做到哪一步了

当前主线仍然是 **Re10k-1 full internal route**。

已完成：
1. `part2 full rerun → internal_eval_cache` 导出
2. same-ply `replay_internal_eval` 一致性验证
3. `prepare_stage1_difix_dataset_s3po_internal.py` 打通 `select → difix → fusion → verify → pack`
4. BRPO verification 已能吃 `Difix left/right` repaired 图并输出 `left/right/fused` confidence
5. `pseudo_cache/` 已接入：
   - `target_rgb_left/right/fused`
   - `confidence_mask_brpo_left/right/fused`
   - `target_depth_for_refine`
6. `run_pseudo_refinement.py` 已按 `target_side` 正式消费这些新 schema
7. 已完成 `fused + brpo` 的 v1 2-iter smoke
8. **Phase 7A 最小版已完成**：
   - 新增 `run_pseudo_refinement_v2.py`
   - 新增 `pseudo_camera_state.py`
   - 新增 `pseudo_loss_v2.py`
   - 新增 `pseudo_refine_scheduler.py`
   - Stage A 已能优化 pseudo pose delta + exposure
   - 已跑通 `fused + brpo` 的 2-iter Stage A smoke

## 现在最重要的结论

1. Phase 6 的 schema plumbing 已经完成，不要再把 `fusion / side-aware mask / canonical pack` 当成待规划事项。
2. Phase 7A 也已经从文档规划进入代码与 smoke 阶段。
3. 当前更合理的下一步是：
   - 把 Stage A 从 smoke 扩成可评估版本
   - 或在新 schema 上做最小 `left/right/fused` 对照
4. 还不该立刻跳到 full Stage B。

## 下一次来先看什么

按顺序看：
1. `/home/bzhang512/CV_Project/part3_BRPO/docs/STATUS.md`
2. `/home/bzhang512/CV_Project/part3_BRPO/docs/DESIGN.md`
3. `/home/bzhang512/CV_Project/part3_BRPO/docs/CHANGELOG.md`
4. `/home/bzhang512/CV_Project/part3_BRPO/docs/STAGE1_INTERNAL_CACHE_REPLAY_RUNBOOK.md`
5. 这一份 `charles.md`

## 下一次优先做什么

### A. 把 Stage A 扩成可评估版本
- 收紧 CLI / history / state export
- 明确 `target_depth_for_refine` 的真正生成策略（当前还是 render-depth fallback）
- 决定 Stage A 默认是否保留 depth 项，还是先做 RGB-only 再逐步加 depth

### B. 二选一推进
1. 做最小对照：
   - `left + brpo_left`
   - `right + brpo_right`
   - `fused + brpo_fused`
2. 或继续往 `Stage B` 的最小 joint refine 骨架推进

## 关键路径

### Phase 6 prototype root
- `/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_prepare/re10k1__internal_afteropt__brpo_proto_v3/`

### Phase 6 v1 smoke
- `/home/bzhang512/my_storage_500G/CV_Project/output/part3_stage1_internal/re10k-1/full/2026-04-11_phase6_schema_smoke/fused_brpo_2iter/`

### Phase 7A v2 smoke
- `/home/bzhang512/my_storage_500G/CV_Project/output/part3_stage1_internal/re10k-1/full/2026-04-11_stageA_v2_smoke/fused_brpo_stageA_2iter/`

## 给下一个 Charles 的一句话

Phase 6 已经打通，Phase 7A 也已经起跑。现在别再回头重做 schema 设计，优先把 **Stage A 从 smoke 推到可评估版本**。

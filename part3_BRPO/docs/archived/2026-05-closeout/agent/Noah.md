# Noah.md - Part3 BRPO next-step handoff

> 2026-04-21 22:30 CST
> 当前用途：告诉 Noah 下一步要做 T~ formal compare（Phase T4）。
> 重要说明：这份文件是执行入口，不是 authority。当前 authority 仍是 `docs/current/STATUS.md`、`docs/current/DESIGN.md`、`docs/current/CHANGELOG.md`。

---

## 1. 先看什么，再开工

如果刚接手，推荐阅读顺序：
1. `docs/current/STATUS.md`：确认 T~ Phase T1-T3 已完成，Phase T4 待执行。
2. `docs/current/CHANGELOG.md`：看今天 T~ Phase T1/T2/T3 的改动记录。
3. `docs/current/DESIGN.md`：确认 T~ 模块状态与语义定义。
4. `docs/T_direct_brpo_alignment_engineering_plan.md`：Phase T4 的具体执行计划。
5. 如需细节，再看 `docs/design/T_EXACT_UPSTREAM_DESIGN.md`（Phase T1 设计文档）。

---

## 2. 当前可信结论

### 2.1 M~
- `exact_brpo_cm_old_target_v1 ≈ old A1`，M~ 已对齐。
- T~ compare 时，默认用 `exact_brpo_cm_old_target_v1` 作为 control。

### 2.2 G~
- G~ clean compare 已完成，direct BRPO current-step 仅 `+0.00538 PSNR`。
- G~ 已冻结为 side branch，不再是主线突破口。

### 2.3 T~
- **Phase T1-T3 已完成**：
  - T1：exact backend bundle + fusion branch-native ✅
  - T2：exact target field builder ✅
  - T3：exact loss contract + 训练端集成 ✅
- **Phase T4 待执行**：formal compare 验证 `exact_brpo_upstream_target_v1` 效果

### 2.4 R~
- `brpo_joint_v1` 仍是当前固定 topology 主线。

---

## 3. 下一步任务：Phase T4 formal compare

### 3.1 目标

验证 `exact_brpo_upstream_target_v1` replay eval 效果，对比 `old A1` baseline。

### 3.2 执行步骤

1. **确认 compare 固定输入**：
   - PLY：`baseline_summary_only` 的 final PLY
   - Signal root：`20260414_signal_enhancement_e15_compare`
   - Frame ids：使用相同 frame set

2. **生成 exact upstream signal**：
   - 用 `build_brpo_v2_signal_from_internal_cache.py` 生成 `exact_brpo_upstream_target_v1` signal
   - 参数：`--pseudo-observation-mode exact_brpo_upstream_target_v1 --stageA-depth-loss-mode exact_shared_cm_v1 --use-exact-backend --exact-backend-root <path>`

3. **跑 replay eval**：
   - 用 `run_pseudo_refinement_v2.py` replay 模式
   - 记录 PSNR/SSIM/LPIPS 对比

4. **汇报 verdict**：
   - 如果 `exact_brpo_upstream_target_v1 > old A1`：T~ upstream 路线成功
   - 如果 `exact_brpo_upstream_target_v1 ≈ old A1`：T~ upstream 对齐但不增益，需进一步分析
   - 如果 `exact_brpo_upstream_target_v1 < old A1`：需回溯上游问题

### 3.3 预期时间

- Signal build：~30min
- Replay eval：~1h
- 总计：~1.5h

---

## 4. 其他可能任务

如果 T4 compare 结果不理想，可能需要：
- 回溯 T~ upstream backend 问题
- 分析 exact backend confidence vs proxy fusion weight 差异
- 检查 depth target builder 的 fallback 行为

---

## 5. 不做的事

- 不再扩 G~（已冻结）
- 不再调 M~（已对齐）
- 不改动 R~ topology（`brpo_joint_v1` 固定）
- 不混入其他实验路线

---

## 6. 联系

如果遇到阻塞，联系 Boyi ZHANG 确认方向。

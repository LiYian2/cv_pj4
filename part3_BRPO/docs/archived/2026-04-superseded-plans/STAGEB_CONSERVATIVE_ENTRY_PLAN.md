# Stage B 保守入口执行计划（基于 A.5 midpoint 验证）

更新时间：2026-04-13
适用项目：`/home/bzhang512/CV_Project/part3_BRPO`

---

## 1. 文档目的

本文件用于记录 Part3 从 Stage A / A.5 进入 Stage B 的执行规划，作为后续实验的统一路线图。

核心原则：
1. 框架遵循 `BRPO_two_stage_refine_engineering_plan.md` 的 two-stage 设计；
2. 入口策略采用当前实测更稳的 A.5（midpoint + xyz_opacity）结果，而非直接从旧 Stage A 盲进；
3. 以 replay gate 为准，不满足 gate 则回退，不做无界扩跑。

---

## 2. Stage B 在本项目中的作用（简述）

Stage B 的目标不是单独调 appearance，而是做 **joint refinement**：
- 一侧优化 Gaussian 参数；
- 一侧优化 pseudo camera pose residual（可选 exposure）；
- 同时受 real anchor 分支约束，避免 pseudo 分支把误差全部“塞进外观参数”。

建议损失形态（与总计划一致）：

`L_total = λ_real * L_real + λ_pseudo * (β * L_rgb_masked + (1-β) * L_depth_masked) + λ_s * L_scale_reg + λ_pose * L_pose_reg + λ_exp * L_exp_reg`

其中关键点：
- pseudo RGB/depth 使用 confidence mask；
- depth 监督使用 `target_depth_for_refine` 体系；
- real 分支用于稳定全局几何与外观。

---

## 3. 与原 BRPO 计划的一致与差异

### 3.1 一致项（保持不变）

1. 仍按 two-stage 主线：Stage A(或A.5 warm start) -> Stage B joint。
2. 仍在 standalone v2 形态先验证，不先并回 full backend。
3. 仍保留 confidence-mask 加权、real anchor 稳定器、RGBD 联合监督。

### 3.2 调整项（基于当前实测）

1. Stage B 的默认入口改为：
   - 先使用 A.5 `xyz_opacity`（midpoint 8帧）作为 warm start。
2. Stage B 第一轮采用 conservative schedule：
   - 低学习率；
   - 先禁 densify/prune；
   - 先小参数集联合，再按 gate 决定是否放开。
3. 强制 replay gate：
   - 未优于或至少不劣于 A.5 基线时，不进入扩展长跑。

---

## 4. 已有事实依据（当前基线）

当前已验证结论（300iter + replay, 270帧）：
- A.5 `xyz_opacity` 相对 StageA baseline：
  - PSNR +0.02135
  - SSIM +0.000348
  - LPIPS -0.000062

结论：A.5 `xyz_opacity` 可作为 Stage B 的默认 warm start 基线。

---

## 5. Stage B 执行分阶段计划

## Phase 0：预检与基线冻结

目标：确保 Stage B 的输入稳定且可回滚。

检查项：
1. 固定 warm start 产物路径（A.5 最优 run）；
2. 确认 pseudo cache（midpoint 8帧）与 manifest 一致；
3. 确认 `target_depth_for_refine`、confidence mask 可读且覆盖率正常；
4. 记录回滚基线：A.5 replay 指标 + 对应 refined ply。

通过标准：
- 输入完整、指标可复现、回滚点明确。

## Phase 1：Stage B 最小可跑版（conservative）

目标：先验证 joint 优化链路可稳定收敛。

建议设置：
1. Gaussian 参数：先开 `xyz + opacity`（必要时再开有限 appearance）。
2. Camera 参数：开启 `rot/trans delta`；exposure 可关闭或极低 lr。
3. 优化策略：低 lr、小步更新。
4. 稳定项：提高 `λ_real` 权重，pose 正则不为 0。
5. 结构项：densify/prune 关闭。

通过标准：
- 训练无发散；
- pose delta 不异常；
- loss 曲线平稳下降或至少稳定。

## Phase 2：短程 gate 对照

目标：决定是否允许进入长程 Stage B。

实验建议：
1. Stage B conservative 短程（如 100~150 iter）；
2. 与 A.5 baseline 做 replay 对照（同评估脚本、同 cache）。

gate 条件（建议）：
1. 质量：PSNR/SSIM 不低于 A.5，LPIPS 不高于 A.5；
2. 稳定：无明显漂移或退化；
3. 可解释：关键监控（pose/grad/mask ratio）不异常。

未通过动作：
- 回退到 A.5；
- 仅调整学习率与权重后重试，不直接拉长迭代。

## Phase 3：长程 Stage B 与逐步放开

目标：在 gate 通过后评估真实增益上限。

策略：
1. 扩到 300 iter 级别；
2. 逐步放开更多 Gaussian 参数组；
3. 最后再评估 densify/prune 的必要性。

通过标准：
- replay 相比 A.5 继续有增益，且稳定可复现。

---

## 6. 回滚与决策规则

1. 任何阶段只要 replay 指标劣化且无可解释收益，立即回退 A.5 基线。
2. 如果 Stage B 多次调参仍无法稳定超过 A.5：
   - 结论应写明“当前数据条件下 A.5 更优”；
   - 暂缓复杂 Stage B 扩展项（如 densify/prune 联动）。
3. 所有阶段必须保留：
   - 运行命令；
   - 配置快照；
   - replay json；
   - 汇总对比表。

---

## 7. 后续执行清单（按顺序）

1. 固化 Phase 0 输入与回滚基线；
2. 实现/打开 Stage B conservative 最小链路；
3. 跑 Phase 2 短程 gate；
4. gate 通过后再进 Phase 3 长程；
5. 每阶段更新 `STATUS.md` 与 `CHANGELOG.md`。

---

## 8. 备注

本文件是 `BRPO_two_stage_refine_engineering_plan.md` 在当前实验事实上的执行化版本，不替代原理论设计文档。
理论主线看原文；实际跑法和 gate 以本文件为准。

# CURRENT_MASK_PROBLEM.md

> 最后更新：2026-04-13 13:25
> 主题：mask 问题主线仍是 Stage A drift-depth tradeoff；A.5 midpoint 已给出第一轮 xyz vs xyz+opacity 方向性结论。

---

## 1. 当前一句话结论

mask 主问题（coverage/接线）已经基本解决；当前真正问题是：
- abs prior 能抑制 drift，但目前没有带来明显 depth 改善；
- 所以现在的核心不是“再修 mask”，而是“在 Stage A/A.5 内把 drift-depth 口径固化出来（目前 `xyz+opacity` 已在 replay 上给出正向增益）”。

---

## 2. 现在 depth 的现状（你关心的重点）

1. depth 不是断开的：M5 后 `train_mask` 内 non-fallback 比例已约 `81.2%`，不是过去的“几乎全 fallback”。
2. depth 也不是强牵引：在 A 小网格（6x80）和 top3 深跑（3x300）里，`loss_depth_last` 仅小幅波动（约 `0.06837~0.06854`），没有随 abs prior 权重出现明显收益带。
3. `lambda_abs_t/r` 增大主要体现在 drift 压缩（`rho/theta` 下降），不是 depth 提升。
4. 因此“100/强约束也不算有效最终方案”的判断成立：它解决的是漂移稳定性，不是 depth 质量改进。

---

## 3. 和昨天“刚修好 mask 但发现 depth 有问题”相比，有什么不同

1. 昨天：主矛盾是链路问题（有没有 depth 信号、pose 闭环是否真实生效）。
2. 今天：链路已确认可用，主矛盾是参数化问题（如何在 drift 与 depth 之间选可解释折中）。
3. 昨天：重点在 upstream 与 wiring；今天：重点在 Stage A 的 abs prior 口径与梯度贡献解释。
4. 昨天不进 Stage B 是“链路未完全确认”；今天不进 Stage B 是“tradeoff 还没固化”。

---

## 4. 当前已完成的证据链（A+B）

A（结构落地）：
- split+scaled abs prior 已接入（`lambda_abs_t/r + scene_scale + robust`）；
- history 已可直接读 `loss_abs_pose_trans/rot` 与 `abs_pose_rho/theta_norm`。

B（可解释诊断）：
- 单步 loss-term 梯度诊断脚本已可用；
- 训练中可按间隔记录 grad contribution，避免只看 loss 数值。

实验结果：
- 小网格 `6组×80iter`：drift 随 `lambda_abs_t` 增大显著收紧；depth 几乎不变。
- 深跑 `3组×300iter`：
  - drift 最优：`lt3.0_lr0.1`
  - depth 最优：`lt1.0_lr0.1`
  - 两者不重合。

---

## 5. 下一步准备做什么（仍在 Stage A，不进下一阶段实现）

1. depth-heavy 口径复跑已完成：tradeoff 未平移为单一最优。
2. grad contrib 汇总已完成：rot/trans 主要由 rgb 与 depth_total 主导，abs prior 主要是边界约束。
3. 已固化两套临时默认：
   - drift-prioritized：`beta_rgb=0.7, lambda_abs_t=3.0, lambda_abs_r=0.1`
   - depth-prioritized：`beta_rgb=0.3, lambda_abs_t=1.0, lambda_abs_r=0.1`
4. 下游价值验证已完成：270帧 replay 指标近似重合，当前不能宣称已有下游收益。
5. 下一步不是重复同类网格，而是扩大受影响帧范围或做 A.5 最小验证。


## 7. 最新 gate 建议（基于 300iter + replay）

1. `stageA5(xyz+opacity)` 已在 270 帧 replay 上超过 baseline（PSNR +0.02135，SSIM +0.000348，LPIPS -6.23e-05）。
2. 因此不再建议无限期停留在 A/A.5；可以进入 StageB。
3. 但进入方式应保守：先 geometry-first 或低学习率 warmup，首轮关闭 densify/prune，且保留 A.5 baseline 作为回退线。

# Charles.md - Next Session Hand-off

> 作用：给下一次 `new/reset` 之后来到这里的 Charles 一个最短可执行的接手说明。
> 最后更新：2026-04-08 23:01

## 0. 先看什么（按顺序）

先读这四个文档，不要先重翻旧笔记：
1. `STATUS.md` —— 当前现状与主线配置
2. `DESIGN.md` —— 为什么这么设计、下一步该试什么
3. `CHANGELOG.md` —— 最近一次做了什么、结果是什么
4. `RENDER_EVAL_PROTOCOL.md` —— 当前 internal / external render-eval 协议差异，以及如果切换到 full/internal protocol 该怎么改 pipeline

如果只需要快速进入实验现场，再额外看：
- `2026-04-08_joint_refine_freezegeom_ablation/summary.json`
- `2026-04-08_joint_refine_tertile_freezegeom_lambda0p5/`
- `E_joint_realdensify_freezegeom_lambda0p5_tertile/external_eval_gt/`

## 1. 当前最重要的实验结论

当前 part3 的默认起点已经不是 B，也不是 C，而是 **E**。

当前最重要的已验证结论是：
- **joint refine 比 pure pseudo 更合理**；
- **pseudo 默认不应直接更新几何**；
- **densify 应优先走 `real`**；
- **`lambda_pseudo=0.5` 优于 `1.0`**；
- **在保持 C 策略不变时，把 pseudo 从 midpoint 扩展到 tertile，三项指标继续同步提升**。

当前 E 组结果：
- `PSNR 21.7658 / SSIM 0.7879 / LPIPS 0.2110`

## 2. 现在存在一个更上游的问题

当前已经确认：Part2 里 **internal render-eval protocol** 与 **external eval protocol** 不是同一个协议，分数不能直接混比。

这件事已经单独整理到：
- `RENDER_EVAL_PROTOCOL.md`

在用户给出老师反馈或额外决策之前，**不要主动继续沿着这个问题扩展探索**，更不要自行拍板把项目协议切到 internal 或 external 任何一边。

当前只做两件事：
1. 记住这个协议问题已经存在；
2. 需要继续讨论时，先回到 `RENDER_EVAL_PROTOCOL.md`。

## 3. 当前不要做什么

在用户明确下一步之前，先不要主动推进：
- backend 集成 pseudo refine
- internal protocol 代码大改
- full/internal pseudo cache 重建
- 围绕协议问题继续做额外审计实验

先等老师意见或用户明确指定。

## 4. 一句话提醒

**实验主线当前以 E 为默认起点；协议主线当前以 `RENDER_EVAL_PROTOCOL.md` 为统一参考，但在用户提供老师反馈前，不要继续自行扩展。**

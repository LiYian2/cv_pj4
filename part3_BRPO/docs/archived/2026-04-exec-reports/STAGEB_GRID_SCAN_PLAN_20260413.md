# StageB 网格扫描计划（针对120好/300坏）

更新时间：2026-04-13

## 1. 背景与目标

当前事实：
- StageB 120iter 相对 A.5 baseline：PASS（短程提升）
- StageB 300iter 相对 A.5 baseline：FAIL（长程退化）

目标：
1) 找到导致后段退化的主因（权重、学习率、变量放开节奏）；
2) 找到“可稳定到300iter”的最小配置；
3) 若无稳定配置，明确回退策略（保留 A.5 或 StageB early-stop）。

## 2. 更深层方法问题（pipeline/method）

### 2.1 当前 pipeline（已落地）

```text
internal cache / pseudo cache(midpoint 8)
 -> StageA/A.5 warm start
 -> StageB conservative joint (real + pseudo RGBD)
 -> replay gate
```

### 2.2 当前风险判断

- StageB 在短程可带来增益，但长程存在目标冲突或过优化；
- 可能不是“StageB方向错”，而是“StageB后段调度与权重不合理”。

## 3. 扫描假设（先简单分析参数组合）

H1. lambda_real 过强会在后段压制 pseudo 几何收益，导致 replay 回落。  
H2. xyz/opacity 学习率在后段过大，累计漂移导致质量回退。  
H3. 后段变量应分阶段冻结（先 freeze pose 或 freeze opacity）以稳定收敛。  

## 4. 扫描维度与实验矩阵（第一轮）

固定不变：
- warm start: A.5 xyz+opacity refined ply
- pseudo cache: midpoint 8
- stage_mode=stageB
- stageB_iters=300
- num_real_views=2, num_pseudo_views=4
- densify/prune: off

### 4.1 维度A：real/pseudo权重
- A1: lambda_real=1.0, lambda_pseudo=1.0（当前基线）
- A2: lambda_real=0.5, lambda_pseudo=1.0
- A3: lambda_real=0.2, lambda_pseudo=1.0

### 4.2 维度B：后段学习率调度（120iter后）
- B1: 不降（当前基线）
- B2: xyz/opacity lr 乘 0.3
- B3: xyz/opacity lr 乘 0.1

### 4.3 第一轮最小网格（6组，先控成本）
1. G00: A1 + B1（复现实验）
2. G01: A2 + B1
3. G02: A3 + B1
4. G03: A1 + B2
5. G04: A2 + B2
6. G05: A2 + B3

说明：优先覆盖“权重”与“后段降lr”两个最可能因素，不在第一轮引入过多变量。

## 5. 评估协议与门槛

主指标：replay PSNR / SSIM / LPIPS（对比 A.5 baseline）

门槛：
1) 非退化门槛：PSNR>=A.5, SSIM>=A.5, LPIPS<=A.5
2) 稳定性门槛：stageB_history 后段 loss 不出现持续恶化；grad_norm_xyz 无爆冲

额外记录：
- 80/120/160/200/300 的中间快照 replay（用于定位最佳 early-stop 窗口）

## 6. 决策规则

- 若某组在300iter通过门槛：进入 Phase4（更细调 + 可选变量放开）
- 若无组通过但120~200存在稳定优势：采用“StageB早停策略”作为默认
- 若全线不优于A.5：回退 A.5 为主线，StageB 仅保留研究分支

## 7. 第二轮预案（仅在第一轮后决定）

仅当第一轮仍不稳定时，再引入：
1) 后段冻结策略（120后 freeze pose 或 freeze opacity）
2) num_real_views 扫描（2 -> 1）
3) num_pseudo_views 扫描（4 -> 6）

## 8. 产物规范

每组输出：
- run目录 + replay json + stageB_history.json
- summary json（含对A.5 delta）
- 汇总表：`docs/STAGEB_GRID_SCAN_RESULTS_20260413.md`（扫描后生成）

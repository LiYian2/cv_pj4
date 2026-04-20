# PIPELINE.md

> Purpose: a compact source-of-truth for drawing the current Part3 BRPO system.
> Scope: all pipeline branches as of 2026-04-19, including current mainline, alternate branches, and deprecated/landed branches.
> Update: 2026-04-19 — 模块详解重写，基于代码级实现与数学原理。

## 1. One-sentence view

当前默认候选主线：**old A1 + new T1**，即 `joint_confidence_v2 + joint_depth_v2` observation + `brpo_joint_v1` topology + repair A `dense_keep` + bounded StageB。

主数据流：part2 S3PO full rerun → internal_eval_cache → internal prepare(select / Difix / fusion / verify / pack) → signal branch → refine → replay eval

---

## 2. 全链路总览图

```
+---------------------------------------------------------------------+
|               Part3 BRPO 全链路总览 (2026-04-19)                     |
+---------------------------------------------------------------------+

                       [ Dataset split ]
                              |
                              v
                  [ Part2 S3PO full rerun ]
                              |
                              v
              [ internal_eval_cache: before/after_opt PLY ]
                              |
                              v
+----------------------------------------------------------+
|                  Internal Prepare                        |
|  select -> Difix(left/right) -> fusion -> verify -> pack |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                    Signal Branch                         |
|                                                          |
|  [ Legacy ]      [ signal_v2 ]     [ A1 joint_v2 ]       |
|  depth mask      RGB-only         joint_confidence_v2    |
|  [废弃]         [对照]           [当前主线old A1]        |
|                                                          |
|                  [ new A1 joint_v1 ]                     |
|                  pseudo_*_joint_v1                       |
|                  [已实现暂不landing]                     |
|                                                          |
|                  [ A2 geometry expand ]                  |
|                  [废弃 widening失败]                     |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                    Refine Stage                          |
|                                                          |
|  [ StageA ] pose/exposure only, 不更新Gaussian [闭环]   |
|       |                                                  |
|       v                                                  |
|  [ StageA.5 ] local micro [降级为optional warmup]       |
|       |                                                  |
|       v                                                  |
|  [ StageB ]                                              |
|    |-- old topology (两阶段串联) [对照]                  |
|    |-- new topology (T1: brpo_joint_v1) [当前拓扑主线]   |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|            Gradient Management / SPGM                    |
|                                                          |
|  [ No extra manage ] baseline [对照]                    |
|  [ Local gating ] view-conditioned gate [可用]          |
|  [ SPGM repair A ] dense_keep [当前anchor]              |
|  [ selector-first ] support_blend far=0.90 [near-parity]|
|  [ B1/B2 shell ] manager shell + diagnostics [已接通]   |
|  [ B3旧版 ] xyz_lr_scale [降级为probe]                  |
|  [ B3新版 ] deterministic_participation [已接入weak负]  |
+----------------------------------------------------------+
                              |
                              v
              [ Replay Eval: PSNR/SSIM/LPIPS ]
```

---

## 3. 上游信号链路详解（按代码实现）

### 3.1 Link-1: Legacy depth-first branch [废弃]
状态: 废弃，仅作历史对照。

实现位置: 早期 `pseudo_branch/brpo_depth_target.py` 等旧链路。

方法: depth 先行构 mask/target，RGB supervision 用 propagated semantics。

废弃原因: 被 signal_v2 取代，verified coverage 过窄（~2%）。

---

### 3.2 Link-2: signal_v2 RGB-only branch [对照]
状态: 对照臂，不再作为主线。

代码位置: `pseudo_branch/brpo_v2_signal/rgb_mask_inference.py` + `depth_supervision_v2.py`

#### (A) RGB confidence 构建
1) 双分支匹配：fused→left/ref, fused→right/ref。  
2) 把 match 映射到像素，构建 `support_mask/conf_map/match_density`。  
3) support 分类：
- `support_both = support_left & support_right`
- `support_single = support_left ^ support_right`

离散置信度：
\[
C_{rgb}(p)=
\begin{cases}
1.0,& p\in support_{both}\\
0.5,& p\in support_{single}\\
0,& otherwise
\end{cases}
\]

连续置信度（双边几何平均）：
\[
C^{cont}_{rgb}(p)=\sqrt{C_L(p)\cdot C_R(p)}\quad (p\in support_{both})
\]
单边区域直接继承对应分支 confidence。

#### (B) depth target 构建
`rgb_active = raw_rgb_confidence >= min_rgb_conf_for_depth`（默认 0.5）

双边有效时加权融合：
\[
D_{target}(p)=\frac{w_L(p)D_L(p)+w_R(p)D_R(p)}{w_L(p)+w_R(p)}
\]

单边有效时直接取对应投影深度；双边都无效且 fallback=`render_depth` 时，取 `render_depth(p)`。

source_map 编码：
- `SOURCE_LEFT=1`
- `SOURCE_RIGHT=2`
- `SOURCE_BOTH_WEIGHTED=3`
- `SOURCE_RENDER_FALLBACK=4`
- `SOURCE_NONE=0`

输出: `raw_rgb_confidence_v2(.npy/.png)`, `target_depth_for_refine_v2_brpo`, `target_depth_source_map_v2_brpo`。

角色: canonical control arm。

---

### 3.3 Link-3: old A1 (joint_confidence_v2) [当前 observation 主线]
状态: 当前 observation 主线。

代码位置: `pseudo_branch/brpo_v2_signal/joint_confidence.py`

先把 geometry source 转成 tier：
\[
C_{geom}(p)=
\begin{cases}
1.0,& source=both\\
0.5,& source=left/right\\
0,& else
\end{cases}
\]

核心 joint 规则：
\[
C_{joint}(p)=\min(C_{rgb}(p), C_{geom}(p))
\]

连续版：
\[
C^{cont}_{joint}(p)=C^{cont}_{rgb}(p)\cdot C_{geom}(p)
\]

`joint_depth_target_v2` 目前直接复用 `target_depth_for_refine_v2_brpo`（不重建）。

定位: **joint support filter**（统一消费域），而不是完整 observation rewrite。  
覆盖率约 1.96%，但在 Re10k 明确优于 sidecar control。

---

### 3.4 Link-4: new A1 (brpo_joint_v1) [已实现暂不landing]
状态: builder + consumer lock 完成；首轮 compare 不占优。

代码位置: `pseudo_branch/brpo_v2_signal/joint_observation.py`

#### 核心思想
不是 min-rule，而是「候选竞争 + 软融合」：
候选集合 `k ∈ {left, right, both_weighted, render_prior}`。

每候选 score：
\[
s_k(p)=w_aA_k(p)+w_gG_k(p)+w_sS_k(p)+w_pP_k
\]
默认权重：`w_a=0.35, w_g=0.35, w_s=0.20, w_p=0.10`。

其中：
- `A_k`: appearance evidence（来自 `raw_rgb_confidence_cont`）
- `G_k`: geometry consistency（如 left-right 相对一致性）
- `S_k`: support strength（both > single > prior）
- `P_k`: source prior（both_weighted 高，render_prior 低）

softmax 融合深度（温度 τ=0.12）：
\[
\pi_k(p)=\frac{\exp(s_k(p)/\tau)\cdot\mathbb{1}_{valid_k(p)}}{\sum_j\exp(s_j(p)/\tau)\cdot\mathbb{1}_{valid_j(p)}}
\]
\[
D_{joint}(p)=\sum_k \pi_k(p) D_k(p)
\]

joint confidence：
\[
C_{joint}(p)=\sqrt{C_{rgb}(p)\cdot C_{depth}(p)}
\]
其中 `C_depth` 由 best candidate score 给出。

输出 bundle：
- `pseudo_depth_target_joint_v1`
- `pseudo_confidence_joint_v1`
- `pseudo_confidence_rgb_joint_v1`
- `pseudo_confidence_depth_joint_v1`
- `pseudo_uncertainty_joint_v1 = 1 - C_joint`
- `pseudo_source_map_joint_v1`

与 old A1 差异：old A1 是 support filter；new A1 是 observation object rewrite。

首轮结果：`new A1 + new T1 = 24.135512` vs `old A1 + new T1 = 24.185846`。

---

### 3.5 Link-5: A2 geometry-constrained expand [废弃]
状态: widening 方案失败，不作为主线。

代码位置: `pseudo_branch/brpo_v2_signal/support_expand.py`

步骤：
1) 从 A1 提取高置信 seed（默认阈值 0.7）。
2) 迭代膨胀（binary dilation）找候选邻域。
3) 几何一致性判定：
\[
\frac{|D_{cand}(p)-D_{seed-near}(p)|}{\max(D_{seed-near}(p),\epsilon)} < \delta
\]
默认 `δ=0.05`。
4) 按来源赋扩张置信度：both=0.8, single=0.6。

结果：coverage 1.96% → 6.05%，但 PSNR 下降 -0.286。  
结论：扩张引入了过多低质量像素 supervision，负收益。

---

## 4. Refine Stage 链路详解（按代码实现）

### 4.1 StageA [闭环可用]
代码位置: `pseudo_branch/pseudo_loss_v2.py`

总损失（StageA）：
\[
L = \beta L_{rgb} + (1-\beta)L_{depth} + \lambda_{pose}L_{pose} + L_{abs} + \lambda_{exp}L_{exp}
\]

其中：
- `L_rgb`: masked L1（先做 exposure 校正）
- `L_depth`: masked L1（可 source-aware 分解）
- `L_pose`: `||cam_rot_delta|| + t_w ||cam_trans_delta||`
- `L_exp`: `|exposure_a| + |exposure_b|`

Absolute pose prior（SE(3)）:
\[
\Delta T = T_{current}T_0^{-1},\quad \tau=\log_{SE(3)}(\Delta T)=[\rho,\theta]
\]
\[
L_{abs}=\lambda_t\,\rho_{robust}(\|\rho\|/s_{scene}) + \lambda_r\,\rho_{robust}(\|\theta\|)
\]
默认参数：`lambda_abs_t=3.0`, `lambda_abs_r=0.1`。

StageA 作用是 pose/exposure 收敛闭环，不是最终 replay 判优主战场。

---

### 4.2 StageA.5 [已降级]
状态: optional warmup / control。

旧职责: pseudo-only local micro-tune。  
降级原因: 新 topology（T1）中 pseudo+real 同 loop 更稳定，不再依赖 StageA.5 先单独蓄能。

---

### 4.3 StageB old topology [对照]
结构: `StageA -> StageA.5 -> StageB`。

问题: 弱 pseudo 信号在 StageA.5 难积累；进入 StageB 后易被 real anchor 稀释。

---

### 4.4 StageB new topology (T1: brpo_joint_v1) [当前拓扑主线]
方法: joint loop 中同迭代融合 pseudo + real。

每轮语义：
1) sample real/pseudo views  
2) forward real + pseudo  
3) apply gating/SPGM（仅控制 pseudo scope）  
4) assemble joint loss  
5) backward once + step once

比较：`new topology=24.149837` vs `old topology=24.116956`（+0.032881 PSNR）。

关键结论: topology 收益跨 observation 稳定存在。

---

## 5. Gradient Management / SPGM 链路详解（按代码实现）

### 5.1 No extra Gaussian management [对照]
方法: plain refine baseline。

---

### 5.2 Local gating [可用]
代码位置: `pseudo_branch/local_gating/signal_gate.py`

硬门控（hard）：按阈值筛 view：
- `verified_ratio >= min_verified_ratio`
- `rgb_mask_ratio >= min_rgb_mask_ratio`
- `fallback_ratio <= max_fallback_ratio`
- `min_correction >= min_correction`

软门控（soft）：
\[
w = \left(\frac{c_1+c_2+c_3(+c_4)}{N}\right)^{\gamma}
\]
其中 `c_i` 是各指标归一化分量，`γ=soft_power`。

角色: pseudo-view scope controller。

---

### 5.3 SPGM repair A (dense_keep) [当前anchor]
代码位置: `pseudo_branch/spgm/policy.py`

`policy_mode=dense_keep`：所有 active Gaussian 保留，只做软加权。

\[
w_i = (w_{floor} + (1-w_{floor})s_i^{weight})\cdot f_{cluster(i)}
\]

默认 anchor: `weight_floor=0.25`, `keep=(1,1,1)`。

---

### 5.4 SPGM selector-first [near-parity]
代码位置: `policy.py` 的 `selector_quantile`

cluster 内按 `ranking_score` 取 top-k：
\[
Selected_k = TopK\big(r_k\cdot |C_k|,\ s^{ranking}|_{C_k}\big)
\]

当前 near-parity（far_keep≈0.90），但 ranking 误删风险仍在。

---

### 5.5 B1/B2 [已接通]
代码位置: `spgm/stats.py`, `spgm/score.py`, `spgm/manager.py`

B1：结构拆分为 `stats -> score -> policy -> manager`。  
B2：引入 scene-aware proxy 与三分数解耦：
- `weight_score`（权重）
- `ranking_score`（排序）
- `state_score`（状态管理）

#### 关键统计量
- `support_count`: accepted pseudo support
- `population_support_count`: current window support
- `depth_value`: weighted median camera depth（向量化实现）
- `density_proxy = opacity_norm * support_norm`
- `struct_density_proxy = (opacity/volume) * (1 + population_support_norm)`

`state_score`（当前实现）:
\[
s_i^{state}=0.45\hat\rho_i^{struct}+0.35\hat S_i^{pop}+0.20s_i^{depth}
\]

---

### 5.6 B3 旧版 (xyz_lr_scale) [降级为diagnostic probe]
代码位置: `spgm/manager.py`, `manager_mode=xyz_lr_scale`

post-backward 动作：
\[
g_i^{xyz}\leftarrow g_i^{xyz}\cdot f_{cluster(i)}\cdot(0.85+0.15s_i^{state})
\]

问题：动作发生在 backward 后，不改变 render participation，本质仍是 grad modulator。

---

### 5.7 B3 新版 (deterministic_participation) [已接入首轮weak-negative]
代码位置: `spgm/manager.py`, `manager_mode=deterministic_participation`

两步：
1) 每 cluster 选低分候选（state_score 低于分位数阈值）。
2) 候选中再按 keep ratio 保留 top state_score，其余 drop，形成 `participation_render_mask`。

形式化：
\[
Candidate_k = \{i\in C_k\mid s_i^{state}\le Q_q(s^{state}|C_k)\}
\]
\[
Drop_k = Candidate_k \setminus TopK(r_k\cdot|Candidate_k|, s^{state}|_{Candidate_k})
\]

首轮参数：`near=1.0, mid=0.9, far=0.75`。  
结果：`24.182511 vs 24.185744`（-0.00323 PSNR）。

判断：方法对象已切到 pre-render participation control，但当前强度略大，需收缩。

---

## 6. 当前推荐主线配置

### 6.1 固定参照线
RGB-only v2 + gated_rgb0192 + post40_lr03_120

### 6.2 当前默认候选主线
old A1 (`joint_confidence_v2 + joint_depth_v2`)
+ new T1 (`joint_topology_mode=brpo_joint_v1`)
+ repair A `dense_keep`
+ bounded StageB (`post40_lr03_120`)

Re10k 最佳结果: `PSNR 24.185846 / SSIM 0.875423 / LPIPS 0.080379`

---

## 7. 废弃/降级链路汇总

| 链路 | 状态 | 废弃/降级原因 |
|------|------|---------------|
| Legacy depth-first | 废弃 | 被 signal_v2 取代 |
| full v2 depth branch | 废弃 | verified 约 2%，过窄 |
| A2 geometry expand | 废弃 | widening 策略失败，-0.286 PSNR |
| StageA.5 as mainline | 降级 | T1 joint topology 更稳 |
| old topology | 对照 | new topology 有稳定正增益 |
| new A1 (joint v1) | 暂不landing | old A1 + new T1 更优 |
| B3 xyz_lr_scale | 降级 | diagnostic probe，weak-negative |
| selector-first | near-parity | 极窄窗口接近 parity，不升级 |

---

## 8. 数据集 case 状态

| 数据集 | 当前状态 | 当前主线 |
|--------|----------|----------|
| Re10k-1 | A1/T1 收敛，B3 第一版 compare 已完成 | old A1 + new T1 |
| DL3DV-2 | canonical baseline + repair A 已打通 | 暂未平移新 topology |

---

## 9. 实验历史关键节点

| 日期 | 里程碑 | 结果 |
|------|--------|------|
| 2026-04-16 | P2-J bounded StageB schedule | post40_lr03_120 最佳 |
| 2026-04-17 | P2-S support_blend far-keep | far=0.90 接近 parity |
| 2026-04-17 | A1 joint confidence compare | A1 优于 control |
| 2026-04-17 | A1+A2 geometry expand | widening 失败 |
| 2026-04-18 | B1/B2 manager shell | 已接通 |
| 2026-04-18 | T1 topology compare | new topology +0.033 PSNR |
| 2026-04-18 | T1-R3 2x2 factor experiment | old A1 + new T1 最优 |
| 2026-04-19 | B3-R1 deterministic participation | 已接入，weak-negative |

---

## 10. 参考文档

- 当前状态: `docs/STATUS.md`
- 设计原则: `docs/DESIGN.md`
- 过程记录: `docs/CHANGELOG.md`
- 总规划: `docs/BRPO_alignment_unified_RGBD_and_scene_SPGM_plan.md`
- A1 计划: `docs/A1_unified_rgbd_joint_confidence_engineering_plan.md`
- T1 计划: `docs/T1_brpo_joint_optimization_topology_engineering_plan.md`
- B3 计划: `docs/B3_deterministic_state_management_engineering_plan.md`
- archived: `docs/archived/`

---

## 11. 关键数学方法速查

| 模块 | 核心公式 | 代码位置 |
|------|----------|----------|
| RGB continuous confidence | $\sqrt{C_L\cdot C_R}$ | `rgb_mask_inference.py` |
| Depth weighted fusion | $\frac{w_LD_L+w_RD_R}{w_L+w_R}$ | `depth_supervision_v2.py` |
| old A1 joint | $\min(C_{rgb},C_{geom})$ | `joint_confidence.py` |
| new A1 depth fusion | $D=\sum_k\pi_k D_k$ | `joint_observation.py` |
| StageA abs pose prior | $\lambda_t\rho(\|\rho\|/s)+\lambda_r\rho(\|\theta\|)$ | `pseudo_loss_v2.py` |
| SPGM weight score | $(\alpha s^{depth}+(1-\alpha)s^{density})\hat S^\eta$ | `score.py` |
| SPGM state score | $0.45\hat\rho^{struct}+0.35\hat S^{pop}+0.20s^{depth}$ | `score.py` |
| B3 old | $g\leftarrow g\cdot f_{cluster}(0.85+0.15s^{state})$ | `manager.py` |
| B3 new | candidate quantile + keep-ratio top-k | `manager.py` |

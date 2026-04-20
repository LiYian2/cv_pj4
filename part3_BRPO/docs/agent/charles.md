# charles.md

> 记录时间：2026-04-20 06:02
> 主题：B3 的 C0/C1 已补齐到可诊断状态；**下一 session 不要推进 O2a/b**，先继续做 C0-2 的小步 action-law 改动。

## 这次新增完成了什么

围绕 B3，我已经把当前阶段该做的几件事都落下来了：

1. **O0 / O1 已完成**
   - `pseudo_branch/spgm/score.py` 已显式拆出 `participation_score`
   - `pseudo_branch/spgm/manager.py` 已新增 `deterministic_opacity_participation`
   - `scripts/run_pseudo_refinement_v2.py`、`pseudo_branch/local_gating/*`、`gaussian_renderer/__init__.py` 已接通 next-step `participation_opacity_scale`

2. **C1 formal compare 已完成**（且按用户要求没有重跑全部对照组）
   - 复用旧 control：
     - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260419_b3_det_participation_compare_e2/compare_summary.json`
   - 只新跑 opacity 臂：
     - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_b3_opacity_participation_compare_e1/oldA1_newT1_det_opacity_participation_floor090_mid100`
   - 汇总：
     - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_b3_opacity_participation_compare_e1/compare_summary.json`

3. **C0 diagnosis 已完成**
   - 主诊断：
     - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_b3_c0_diagnosis/diagnosis_summary.json`
   - 四分组 probe：
     - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_b3_c0_diagnosis/partition_probe.json`
     - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_b3_c0_diagnosis/C0_DIAGNOSIS_REPORT.md`

4. **C0-1 instrumentation 已接到 StageB history 里**
   - `run_pseudo_refinement_v2.py` 现在会把以下字段直接写进 `stageB_history.json`：
     - `spgm_c0_state_only_*`
     - `spgm_c0_part_only_*`
     - `spgm_c0_both_*`
     - `spgm_c0_neither_*`
   - smoke 验证目录：
     - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_b3_c0_history_smoke/oldA1_newT1_opacity_c0hist_i3`

5. **REFINE_DESIGN 已写好**
   - 路径：`/home/bzhang512/CV_Project/part3_BRPO/docs/REFINE_DESIGN.md`
   - 作用：像 `MASK_DESIGN.md` 那样，把 BRPO B3、old B3、current boolean B3、current opacity B3 的信息来源 / 信号构建 / 下游消费 / 数学形式讲清楚。

---

## 当前硬结论（给下个 session 的约束）

### 1. 现在**不要推进 O2a/b**

原因不是 O2a/b 永远不做，而是它的前提还没满足：

- C1 结果仍是 weak-negative：
  - summary_only：`24.187304 / 0.875587 / 0.080364`
  - old boolean far090_mid100：`24.186847 / 0.875591 / 0.080387`
  - new opacity floor090_mid100：`24.186731 / 0.875586 / 0.080398`
- opacity 相对 summary 是 `-0.000573 PSNR`
- opacity 相对 old boolean 也还是略差

所以当前还不能说：
> “动作变量已经站稳，可以继续切 control universe。”

### 2. 当前真正的问题在 **score / candidate / action law**，不在 O2a/b

C0 diagnosis 的结论是：

- `ranking_score`、`state_score`、`participation_score` **不是一回事**，但还没有拉开 enough
- 当前 `participation_score` 仍明显受 population support 牵引
- `state_only / part_only` 虽然存在，但比例还不大
- 当前 far 的平均 opacity scale 只有大约 `0.981`，action 仍偏轻
- 当前 loop 仍是 delayed：iter `t` 统计，iter `t+1` 生效

一句话：
> **现在已经有诊断显微镜了，显微镜看到的问题仍在 O1/C1 这一层，不在 O2。**

---

## 下个 session 先读什么（严格顺序）

1. `docs/charles.md`（本文件）
2. `docs/STATUS.md`
3. `docs/DESIGN.md`
4. `docs/CHANGELOG.md`
5. `docs/REFINE_DESIGN.md`
6. `docs/B3_opacity_participation_population_manager_engineering_plan.md`
7. `pseudo_branch/spgm/score.py`
8. `pseudo_branch/spgm/manager.py`
9. `scripts/run_pseudo_refinement_v2.py`
10. `third_party/S3PO-GS/gaussian_splatting/gaussian_renderer/__init__.py`

如果想快速确认当前 C0 的定量结论，再看：
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_b3_c0_diagnosis/diagnosis_summary.json`
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_b3_c0_diagnosis/partition_probe.json`

---

## 下个 session 先做什么

### Step 1：继续做 **C0-2**，不要跳到 O2

只允许做**一小步** action-law 改动，不要同时改太多。

首选方向二选一：

#### 方案 A：改 candidate law
只改 far 的 candidate quantile / rule，例如：
- far 用更激进或更保守的 quantile
- near / mid 不动
- 不改 floor 映射

#### 方案 B：改 opacity mapping
保持 candidate 不变，只改 opacity scale 映射，例如：
- far 的 linear floor 映射改成更陡的非线性映射
- near / mid 不动
- 不改 quantile

**一次只做 A 或 B 之一，不要同一 patch 里两者一起改。**

### Step 2：formal compare 时继续只重跑**新 opacity 臂**
旧 control 直接复用，不要重复烧算力：
- 复用 `summary_only`
- 复用 `old boolean far090_mid100`
- 只新跑修改后的 opacity 臂

### Step 3：看新的 C0 history 字段有没有真的变好
关键要看：
- `spgm_c0_state_only_ratio`
- `spgm_c0_part_only_ratio`
- `spgm_c0_both_ratio`
- `spgm_c0_neither_ratio`
- `spgm_c0_*_ranking_mean / state_mean / participation_mean`

目标不是“字段写出来了”，而是：
> 修改后，`part_only` 群体是否更像真正该被 opacity 路径命中的群体，并且 action 强度是否真正拉开。

---

## 明确不要做的事

1. **不要推进 O2a/b**
2. 不要同时改 quantile + floor mapping + timing
3. 不要重跑全部 control arms
4. 不要因为已经写了 C0 history 就误以为“这层已经解决了”

---

## 一句话 handoff

> **下一步不是 O2a/b，而是继续把 O1/C1 这一层做对：用现在已经接进 StageB history 的 C0 字段，指导一小步 candidate-law 或 opacity-law 修改，然后只重跑新 opacity 臂。**

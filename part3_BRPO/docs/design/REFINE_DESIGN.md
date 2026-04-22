# REFINE_DESIGN.md - R~ Joint Refine 设计文档

> 更新时间：2026-04-22 04:22 (Asia/Shanghai)

> **书写规范**：
> 1. 只讲 R~（Joint Refine）：信息从哪里来、怎么组装成 joint loss、backward timing 如何
> 2. 遵循"信息源 → 信号转换 → 下游消费"链式分析
> 3. 数学公式用 `$...$` 或 `$$...$$` 包裹
> 4. 更新后修改文档顶部时间戳

---

## 1. 概览

R~（Joint Refine）决定"pseudo loss 和 real loss 怎么在同一 iteration 内组装、backward、更新"。当前系统支持 **2 种 topology mode + 3 种 depth_loss_mode**。

**关键结论**：
- `brpo_joint_v1`（T1）是当前 topology 主线
- Stage 协议固定为 `post40_lr03_120`
- G~ timing 是 delayed（与 BRPO 论文不同）

---

## 2. 所有 R~ 变种一览

### 2.1 Topology modes

| Mode | Pseudo/Real 组装方式 | Backward timing | G~ timing |
|------|---------------------|-----------------|-----------|
| `off` | Sequential（pseudo → real） | Separate backward | Delayed |
| `brpo_joint_v1` | Joint（pseudo + real → joint loss） | Single backward | Delayed |

### 2.2 Depth loss modes

| Mode | RGB/Depth mask 关系 | Fallback | 适用 observation mode |
|------|---------------------|---------|---------------------|
| `legacy` | RGB/depth 可分离 mask | render_depth fallback | 通用 |
| `source_aware` | RGB/depth 可分离 mask + tier | seed/dense/fallback tier | `old T~` |
| `exact_shared_cm_v1` | **RGB/depth 共用 C_m** | **无 fallback** | `exact_brpo_upstream_target_v1` |

---

## 3. 信息源层

R~ 的上游信息来自两条 branch：

### 3.1 Pseudo branch

**输入**：
- Pseudo frames（来自 prepare stage 的 fusion output）
- M~（mask/confidence）
- T~（depth target）
- Rendered pseudo RGB/depth（当前 Gaussian state）

### 3.2 Real branch

**输入**：
- Real frames（training dataset）
- Rendered real RGB/depth

---

## 4. 信号转换层

### 4.1 StageA（pose/exposure only）

**目标**：只更新 pseudo pose/exposure，不更新 Gaussian。

**Loss 组装**：

$$
L_{pose} = L_{rgb}^{pseudo} + \lambda_{pose} \cdot L_{pose\_residual}
$$

**约束**：
- Gaussian 参数 frozen
- Replay 不是判优指标（只 pose/exposure 变化）

---

### 4.2 StageB T1（`brpo_joint_v1`）

**代码位置**：`run_pseudo_refinement_v2.py` → StageB loop

**Loss 组装**：

$$
L_{joint} = \lambda_{pseudo} \cdot L_{pseudo} + \lambda_{real} \cdot L_{real} + L_{abs\_prior}
$$

其中 pseudo loss：

$$
L_{pseudo} = eta_{rgb} \cdot L_{rgb} + (1-eta_{rgb}) \cdot L_{depth} + \lambda_{pose} \cdot L_{pose\_reg} + \lambda_{exp} \cdot L_{exp\_reg}
$$

---

### 4.3 Legacy depth loss（`stageA_depth_loss_mode=legacy`）

$$
L_{depth} = C_m \cdot |d_{render} - d_{target}|
$$

RGB 和 depth 使用同一 $C_m$（但可分别有 rgb_confidence_mask / depth_confidence_mask override）。

---

### 4.4 Source-aware depth loss

$$
L_{depth} = \lambda_{seed} \cdot L_{seed} + \lambda_{dense} \cdot L_{dense} + \lambda_{fallback} \cdot L_{fallback}
$$

其中每个 tier 对应不同 source_ids：
- Seed: {1, 2, 3}（双侧 + stable）
- Dense: {4}（single-side）
- Fallback: {0}（render_depth fallback）

---

### 4.5 Exact shared-C_m loss（Phase T3 新增）

$$
L_{rgb} = C_m \cdot C_{target} \cdot |I_{render} - I_{target}|
$$
$$
L_{depth} = C_m \cdot C_{target} \cdot |d_{render} - d_{target}|
$$

**关键特点**：
- RGB/depth **强制共用** 同一 $C_m$
- $C_{target}$ 是 continuous confidence（来自 exact backend）
- 无 fallback tier（`no_render_fallback=true`）

**代码位置**：`pseudo_loss_v2.py` → `build_stageA_loss_exact_shared_cm()`

---

## 5. Backward timing 层

### 5.1 T1 timing（`brpo_joint_v1`）

```
iter t:
  1. pseudo render (G~ action from iter t-1)
  2. real render
  3. assemble joint loss: L_joint = λ_pseudo * L_pseudo + λ_real * L_real
  4. backward once (joint backward)
  5. collect G~ stats → generate G~ action for iter t+1
  6. optimizer.step

iter t+1:
  1. pseudo render (G~ action from iter t)
  ...
```

**关键特点**：
- Single backward（不是 separate pseudo → real backward）
- G~ delayed（iter t 统计 → iter t+1 消费）

---

### 5.2 BRPO 论文 timing

```
iter t:
  1. collect G~ stats
  2. generate G~ stochastic mask (current-step)
  3. pseudo render (current-step G~ action)
  4. pseudo backward
  5. real render
  6. real backward
  7. optimizer.step
```

**与 T1 的差异**：
- BRPO 是 sequential（pseudo → real）
- T1 是 joint（pseudo + real → joint backward）
- G~ timing：BRPO current-step，T1 delayed

---

## 6. Abs prior（固定背景约束）

**定义**：

$$
L_{abs\_prior} = \lambda_t \cdot |t_{pseudo} - t_{prior}| + \lambda_r \cdot |r_{pseudo} - r_{prior}| + \lambda_{abs\_pose} \cdot L_{abs\_scaled}
$$

**当前配置**：
- $\lambda_t = 3.0$
- $\lambda_r = 0.1$
- $\lambda_{abs\_pose} = 0.0$（默认不启用）

**Robust type**：`charbonnier`

---

## 7. Stage 协议

当前固定协议：`post40_lr03_120`

- `post40`：iteration 40 之后开始 StageB
- `lr03`：StageB learning rate = 0.3
- `120`：StageB iteration 数 = 120

---

## 8. R~ 与其他模块的关系

```
R~ 输入:
  - M~: pseudo RGB/depth mask
  - T~: pseudo depth target
  - G~: pseudo render Gaussian gating (delayed action)

R~ 输出:
  - joint loss
  - optimizer updates (pose/exposure/Gaussian)

Pipeline:
  M~ + T~ 定义 pseudo supervision scope 和 target
  G~ 决定哪些 Gaussian 参与 pseudo render
  R~ 决定 pseudo + real 怎么组装成 joint loss
  Backward + optimizer.step
```

---

## 9. 代码位置索引

| 文件 | 功能 | 关键函数 |
|------|------|---------|
| `scripts/run_pseudo_refinement_v2.py` | 主 loop | StageA loop, StageB loop, G~ timing |
| `pseudo_branch/refine/pseudo_loss_v2.py` | Loss 组装 | `build_stageA_loss`, `build_stageA_loss_source_aware`, `build_stageA_loss_exact_shared_cm` |
| `pseudo_branch/refine/pseudo_refine_scheduler.py` | Stage 协议 | iteration scheduling |
| `pseudo_branch/gaussian_management/local_gating/*` | G~ timing 介入点 | `maybe_apply_pseudo_local_gating` |

---

## 10. 当前状态与下一步

### 10.1 当前状态

- `brpo_joint_v1` 是固定 topology 主线
- Stage 协议 `post40_lr03_120` 已稳定
- G~ timing 保持 delayed（与 BRPO 论文不同）

### 10.2 不做的事

- 不改 topology（`brpo_joint_v1` 固定）
- 不改 Stage 协议
- 不推进 G~ current-step timing（G~ 已冻结）

### 10.3 下一步

主线不再继续扫 standalone topology compare；下一阶段是把固定的 `brpo_joint_v1 + exact_shared_cm_v1 + clean summary G~` 这套 consumer 壳带进 backend-only integration。

---

> 文档口径：R~ = Joint Refine（topology）。与 M~/T~/G~ 协同工作，决定 loss 组装和 backward timing。

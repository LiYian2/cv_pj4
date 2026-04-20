# REFINE_DESIGN.md - R~ Joint Refine 设计文档

> 更新时间：2026-04-20 19:00 (Asia/Shanghai)

> **书写规范**：
> 1. 只讲 R~（Joint Refine）：信息从哪里来、怎么组装成 joint loss、backward timing 如何
> 2. 遵循"信息源 → 信号转换 → 下游消费"链式分析
> 3. 需要比较时，统一比较 StageA / StageA.5 / StageB / BRPO topology
> 4. 数学公式用 `$...$` 或 `$$...$$` 包裹
> 5. 更新后修改文档顶部时间戳

---

## 1. 先说结论

R~（Joint Refine）决定"pseudo loss 和 real loss 怎么在同一 iteration 内组装、backward、更新"。当前主线是 `brpo_joint_v1`（new T1），核心特点是：
- 同一 iteration 内 assemble joint loss
- Backward once
- G~（Gaussian Management）在 backward 前介入作为 pseudo scope controller

这是对传统 sequential topology（pseudo first → real second）的重写。

---

## 2. R~ 的信息源

R~ 的上游信息来自两条 branch：

**Pseudo branch**：
- Pseudo frames（来自 prepare stage 的 fusion output）
- M~（mask/confidence）
- T~（depth target）
- Rendered pseudo RGB/depth

**Real branch**：
- Real frames（training dataset）
- Rendered real RGB/depth

---

## 3. 各阶段 R~ 的信号转换

### 3.1 StageA

**目标**：只更新 pseudo pose/exposure，不更新 Gaussian。

**信息流**：
$$
L_{pose} = L_{rgb}^{pseudo} + \lambda_{pose} \cdot L_{pose\_residual}
$$

**约束**：
- Gaussian 参数 frozen
- Replay 不是判优指标（只 pose/exposure 变化）

---

### 3.2 StageA.5（optional）

**定位**：optional warmup / control / orchestration anchor

**角色**：不是主线必经阶段，只是可选的 micro-tune。

---

### 3.3 StageB / T1（当前主线）

**代码路径**：`run_pseudo_refinement_v2.py` → `joint_topology_mode=brpo_joint_v1`

**信息流**：
1. Pseudo render（受 M~ mask 和 G~ gating 控制）
2. Real render
3. Assemble joint loss：
$$
L_{joint} = L_{pseudo} + L_{real} + L_{abs\_prior}
$$
4. Backward once
5. G~ 在 backward 后生成 next iteration 的 gating action
6. Optimizer.step

**关键特点**：
- 同一 iteration 内 pseudo/real 同时参与
- Backward 只一次，不是先 pseudo backward 再 real backward
- G~ 作为 pseudo scope controller，决定哪些 Gaussian 参与 pseudo render

---

### 3.4 BRPO 论文拓扑

**信息流**：
- Pseudo render → pseudo loss → pseudo backward
- Real render → real loss → real backward
- G~ stochastic masking 在 pseudo render 前生效（current-step）

**与当前 T1 的差异**：
- BRPO 是 sequential（pseudo → real）
- T1 是 joint（pseudo + real → joint backward）
- G~ timing：BRPO current-step，T1 delayed

---

## 4. Joint Loss 组装

### 4.1 Pseudo loss components

**RGB loss**：
$$
L_{rgb}^{pseudo} = M~ \cdot |I_{render}^{pseudo} - I_{target}^{pseudo}|
$$

**Depth loss**：
$$
L_{depth}^{pseudo} = M~ \cdot |d_{render}^{pseudo} - T~|
$$

**Pose prior**（可选）：
$$
L_{pose} = \lambda_t \cdot L_{trans} + \lambda_r \cdot L_{rot}
$$

### 4.2 Real loss components

$$
L_{real} = L_{rgb}^{real} + L_{depth}^{real}
$$

### 4.3 Abs prior（固定背景）

$$
L_{abs\_prior} = \lambda_t \cdot |t_{pseudo} - t_{prior}| + \lambda_r \cdot |r_{pseudo} - r_{prior}|
$$

当前配置：$\lambda_t = 3.0, \lambda_r = 0.1$

---

## 5. Backward timing 与 G~ 介入

### 5.1 T1 timing

```
iter t:
  pseudo render (G~ action from iter t-1)
  real render
  assemble joint loss
  backward once
  collect G~ stats
  generate G~ action for iter t+1
  optimizer.step

iter t+1:
  pseudo render (G~ action from iter t)
  ...
```

G~ 是 delayed：iter t 统计 → iter t+1 消费。

### 5.2 BRPO timing

```
iter t:
  collect G~ stats
  generate G~ stochastic mask
  pseudo render (current-step G~ action)
  pseudo backward
  real render
  real backward
  optimizer.step
```

G~ 是 current-step：统计 → mask → render → backward 同一轮。

---

## 6. Stage 协议

当前固定协议：`post40_lr03_120`

- `post40`：iteration 40 之后开始 StageB
- `lr03`：learning rate 0.3
- `120`：StageB iteration 数 120

---

## 7. 各阶段状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| StageA | ✅ | pose/exposure 更新，Gaussian frozen |
| StageA.5 | ⚠️ | optional warmup，非主线 |
| StageB T1 | ✅ | 当前 topology 主线 |
| BRPO sequential | 参考 | 论文原始拓扑，与 T1 结构不同 |

---

## 8. 代码位置

| 文件 | 功能 |
|------|------|
| `scripts/run_pseudo_refinement_v2.py` | 主 loop，topology 实现 |
| `pseudo_branch/pseudo_loss_v2.py` | Loss 组装 |
| `pseudo_branch/pseudo_refine_scheduler.py` | Stage 协议 |
| `pseudo_branch/local_gating.py` | G~ timing 介入点 |

---

## 9. R~ 与其他模块的关系

```
M~ → pseudo RGB/depth mask
T~ → pseudo depth target
G~ → pseudo render Gaussian gating
R~ → joint loss assembly + backward timing

Pipeline:
  M~ + T~ 定义 pseudo supervision
  G~ 决定哪些 Gaussian 参与 pseudo render
  R~ 决定 pseudo + real 怎么组装成 joint loss
  Backward + optimizer.step
```

---

> 文档口径：R~ = Joint Refine（topology）。原 T1。
# DESIGN.md - Part3 BRPO 设计文档

> 更新时间：2026-04-20 19:10 (Asia/Shanghai)

> **书写规范**：
> 1. 只记录"设计原则、架构决策、接口定义"，不记录实验数据
> 2. 覆盖式更新，直接修改对应版块，不追加
> 3. 设计判断用一句话固化，不展开过程
> 4. 引用格式：`[参见 STATUS §X]`
> 5. 更新后修改文档顶部时间戳

---

## 1. 系统边界与模块口径

### 1.1 四大模块

| 模块 | 口径 | 功能 | 详细文档 |
|------|------|------|---------|
| Mask | M~ | 监督域 + 监督强度 | MASK_DESIGN.md |
| Target | T~ | 监督目标数值 | TARGET_DESIGN.md |
| Gaussian Management | G~ | Per-Gaussian gating | GAUSSIAN_MANAGEMENT_DESIGN.md |
| Joint Refine | R~ | Topology joint loop | REFINE_DESIGN.md |

**注**：Fusion 已定，不纳入设计文档范围。

### 1.2 模块关系

```
M~ → pseudo RGB/depth mask（监督域）
T~ → pseudo depth target（监督目标）
G~ → pseudo render Gaussian gating（参与控制）
R~ → joint loss assembly + backward timing（拓扑）

Pipeline:
  Prepare → Fusion → M~/T~ builder → G~ gating → R~ topology → Backward → Optimize
```

---

## 2. 当前主线

### 2.1 固定参照线

**RGB-only v2 + gated_rgb0192 + post40_lr03_120**（canonical StageB protocol）

### 2.2 当前候选主线

| 模块 | 当前状态 | 说明 |
|------|---------|------|
| M~ | exact M~ | 已基本对齐 BRPO semantics |
| T~ | old T~ | 最稳，depth pipeline 成熟 |
| G~ | summary control | Boolean/Opacity 仍弱负，暂不 landing |
| R~ | T1 (brpo_joint_v1) | 当前 topology 主线 |

**最佳组合**：`exact M~ + old T~ + summary G~ + T1` ≈ old A1 + new T1

---

## 3. 设计判断（固化）

### 3.1 M~ 结论
- exact M~ 与 old M~ 基本等价（差 < 1e-5 PSNR）
- strict BRPO $C_m$ 已基本对齐
- 剩余 gap 不在 M~

### 3.2 T~ 结论
- old T~ 最稳（工程成熟）
- exact T~ 在当前 proxy backend 下仍弱 -0.013 PSNR
- 真正瓶颈在上游 Layer B proxy

### 3.3 G~ 结论
- Boolean/Opacity/Summary 三模式无区别
- 原因：分数相关、动作轻、delayed、动作域窄
- 下一步：C0 层拉开分数，不进 O2a/b

### 3.4 R~ 结论
- T1 (brpo_joint_v1) 是稳定 topology 主线
- 与 BRPO sequential topology 结构不同但有效
- G~ timing：T1 delayed，BRPO current-step

---

## 4. 接口定义

### 4.1 M~ 接口

| mode | confidence 来源 | 输出文件 |
|------|----------------|---------|
| old | rgb + geometry tier min | joint_confidence_v2.npy |
| new | score_stack 派生 | pseudo_confidence_joint_v1.npy |
| hybrid | support sets | pseudo_confidence_brpo_style_v1.npy |
| exact | strict BRPO semantics | pseudo_confidence_exact_brpo_*.npy |

### 4.2 T~ 接口

| mode | target 来源 | 输出文件 |
|------|------------|---------|
| old | projected depth + rgb gate | target_depth_for_refine_v2_brpo.npy |
| new | score_prob weighted | pseudo_depth_target_joint_v1.npy |
| hybrid | verified composition | pseudo_depth_target_brpo_style_v1.npy |
| exact | strict BRPO target | pseudo_depth_target_exact_brpo_full_target_v1.npy |

### 4.3 G~ 接口

| mode | action 类型 | 输出 |
|------|------------|------|
| summary | 无动作 | 只统计 |
| boolean | mask | participation_render_mask |
| opacity | scale | participation_opacity_scale |

### 4.4 R~ 接口

| mode | topology | 说明 |
|------|---------|------|
| off | sequential | pseudo → real sequential backward |
| brpo_joint_v1 | joint | pseudo + real → joint backward |

---

## 5. 不做的事

- 不继续打磨 verify proxy（已完成 negative proof）
- 不在 M~ contract 上继续微调（已对齐）
- 不在 G~ delayed opacity 仍弱负时推进 O2a/b
- 不把 T~ 剩余 gap 简化成单侧问题
- 不在 observation compare 里同时改 topology 或 G~

---

## 6. 参考

- 状态：[STATUS.md]
- 过程：[CHANGELOG.md]
- M~ 详细：[MASK_DESIGN.md]
- T~ 详细：[TARGET_DESIGN.md]
- G~ 详细：[GAUSSIAN_MANAGEMENT_DESIGN.md]
- R~ 详细：[REFINE_DESIGN.md]
# PIPELINE.md

> Purpose: compact source-of-truth for Part3 BRPO system overview.
> Update: 2026-04-20 19:20 — 口径统一为 M~/T~/G~/R~，详细设计指向 design/ 文件夹。

---

## 口径说明

- **M~** = Mask（confidence），监督域 + 监督强度
- **T~** = Target，监督目标数值
- **G~** = Gaussian Management，per-Gaussian gating
- **R~** = Topology，joint loop

历史：A1 → M~ + T~，B3 → G~，T1 → R~。

详细设计文档：
- `docs/design/MASK_DESIGN.md`
- `docs/design/TARGET_DESIGN.md`
- `docs/design/GAUSSIAN_MANAGEMENT_DESIGN.md`
- `docs/design/REFINE_DESIGN.md`

---

## 1. One-sentence view

当前最佳组合：**exact M~ + old T~ + summary G~ + T1 R~** ≈ old A1 + new T1

主数据流：part2 S3PO full rerun → internal_eval_cache → prepare → signal branch → refine → replay

---

## 2. 全链路总览图

```
+---------------------------------------------------------------------+
|               Part3 BRPO 全链路总览 (2026-04-20)                     |
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
|  [ M~ ] Mask/confidence                                  |
|    |-- old M~ (joint_confidence_v2) [当前主线]           |
|    |-- new M~ (pseudo_*_joint_v1) [暂不landing]          |
|    |-- hybrid M~ (brpo_style_v1)                        |
|    |-- exact M~ (exact_brpo_cm) [已对齐BRPO semantics]   |
|                                                          |
|  [ T~ ] Target                                           |
|    |-- old T~ (target_depth_v2_brpo) [当前最稳]          |
|    |-- new T~ (candidate competition)                    |
|    |-- hybrid T~ (verified composition)                  |
|    |-- exact T~ (pure BRPO semantics) [弱于old -0.013]   |
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
|    |-- R~ T1 (brpo_joint_v1) [当前拓扑主线]              |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|            Gaussian Management (G~)                      |
|                                                          |
|  [ summary ] 只统计，不动作 [对照]                       |
|  [ boolean ] deterministic mask [weak-negative]          |
|  [ opacity ] deterministic scale [weak-negative]         |
|  [ BRPO ] stochastic current-step [未实现]               |
+----------------------------------------------------------+
                              |
                              v
              [ Replay Eval: PSNR/SSIM/LPIPS ]
```

---

## 3. 模块详解（指向设计文档）

### 3.1 M~（Mask）

详见 `docs/design/MASK_DESIGN.md`

核心信息流：
- 信息源：pseudo-frame correspondence verification
- 信号转换：support sets → $C_m$（both=1.0, xor=0.5, none=0.0）
- 下游消费：RGB/depth loss mask

当前状态：exact M~ 已对齐 BRPO semantics

---

### 3.2 T~（Target）

详见 `docs/design/TARGET_DESIGN.md`

核心信息流：
- 信息源：projected depth + fusion weight
- 信号转换：verified composition → target 数值
- 下游消费：depth loss target

当前状态：old T~ 最稳，exact T~ 在当前 proxy backend 下弱 -0.013

---

### 3.3 G~（Gaussian Management）

详见 `docs/design/GAUSSIAN_MANAGEMENT_DESIGN.md`

核心信息流：
- 信息源：Gaussian stats（support/depth/density）
- 信号转换：scores → action（mask/scale）
- 下游消费：renderer participation

当前状态：三模式无区别，原因：分数相关、动作轻、delayed、动作域窄

---

### 3.4 R~（Topology）

详见 `docs/design/REFINE_DESIGN.md`

核心信息流：
- 信息源：pseudo loss + real loss
- 信号转换：joint assembly + backward timing
- 下游消费：optimizer.step

当前状态：T1 (brpo_joint_v1) 稳定主线

---

## 4. 当前推荐配置

### 4.1 最佳组合

`exact M~ + old T~ + summary G~ + T1 R~`

### 4.2 固定参照线

RGB-only v2 + gated_rgb0192 + post40_lr03_120

---

## 5. 实验历史关键节点

| 日期 | 里程碑 | 结果 |
|------|--------|------|
| 2026-04-19 | M~ exact BRPO compare | exact M~ ≈ old M~ |
| 2026-04-20 | T~ exact BRPO compare | exact T~ 弱于 old -0.013 |
| 2026-04-20 | G~ opacity C1 compare | weak-negative |
| 2026-04-20 | 口径统一 | M~/T~/G~/R~ |

---

## 6. 参考文档

### 经常更新（docs/current/）
- `STATUS.md` — 当前状态
- `DESIGN.md` — 设计原则
- `CHANGELOG.md` — 过程记录

### 详细设计（docs/design/）
- `MASK_DESIGN.md` — M~ 详细
- `TARGET_DESIGN.md` — T~ 详细
- `GAUSSIAN_MANAGEMENT_DESIGN.md` — G~ 详细
- `REFINE_DESIGN.md` — R~ 详细

### Agent 入口
- `docs/agent/hermes.md` — Hermes 入口

### 分析文档
- `docs/EXTERNAL_ANALYSIS.md` — Noah 外部分析
- `docs/part3_BRPO_A1_B3_vs_BRPO_detailed_analysis.md` — detailed BRPO analysis

### 归档
- `docs/archived/experiments/` — 实验记录
- `docs/archived/plans/` — 历史计划

---

## 7. 数据集状态

| 数据集 | 当前状态 | 主线 |
|--------|----------|------|
| Re10k-1 | M~/T~/G~/R~ 收敛 | exact M~ + old T~ + summary G~ + T1 |
| DL3DV-2 | canonical baseline | 暂未平移新模块 |

---

> 文档口径：M~/T~/G~/R~。详细设计见 docs/design/。
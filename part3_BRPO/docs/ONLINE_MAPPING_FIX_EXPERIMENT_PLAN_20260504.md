# Online Mapping 修复实验规划

> 创建时间：2026-05-04 22:00 (Asia/Shanghai)
> 状态：规划阶段，待执行

---

## 1. 核心问题分析

基于 Phase 3 conservative single-gap smoke 的诊断结果，用户分析的四个根本原因：

### 1.1 迭代次数严重不足
- Standalone: ~200 iterations 累积优化
- Online: 每个 keyframe event 只做 2 iterations
- 8 个 events 累计只有 ~16 iterations
- 这不足以让 Gaussians 收敛到新的 supervision

### 1.2 Supervision 稀疏性问题
- pseudo_effective_mask_nonzero_ratio: 0.68% - 7.8%
- both_only 模式进一步限制了有效区域
- 稀疏的 supervision 在短迭代下几乎无效

### 1.3 Optimization dynamics 不匹配
- S3PO backend 的 real mapping iterations = 300 per keyframe
- Pseudo mapping iterations = 2 per keyframe event
- Real vs Pseudo 比例 = 150:1，pseudo 几乎没有机会影响 scene

### 1.4 Loss balance 问题
- loss_real: 0.077-0.142
- loss_pseudo: 0.086-0.340 (iteration 2 时翻倍)
- Pseudo loss 在 iteration 2 时显著上升，说明 scene 被 pseudo 推向不稳定方向

---

## 2. 实验优先级排序

按影响量级和可行性排序：

| 优先级 | 实验轴 | 预期影响 | 依赖基础设施 | 变量范围 |
|-------|-------|---------|-------------|---------|
| **P0** | 增加 pseudo_map_iters | 直接解决迭代不足 | ✅ 已有 Phase 3 shell | 2 → 20 → 50 |
| **P1** | 放宽 pseudo_scene_mask_mode | 扩大 supervision 覆盖 | ✅ 已有 `both_only / valid_only / all_valid` mode | both_only → valid_only → all_valid |
| **P2** | 增加 pseudo views per event | 提升 pseudo mass | ❌ 需要 runtime multi-slot selector | 1 → 2 → 3 |
| **P3** | 切换到 paper-realign contract | rgb/depth 解耦 + no verifier | ✅ 已有 paper_* 分支完整落地 | exact → paper |

---

## 3. 实验序列设计

### Phase A: 单变量迭代次数测试（P0）

**目标**: 验证"迭代次数不足"是否是主瓶颈

**实验设计**:

| arm | pseudo_map_iters | pseudo_scene_mask_mode | pseudo_views_per_event | contract |
|------|------------------|------------------------|------------------------|----------|
| A0_control | 2 | both_only | 1 | exact |
| A1_iter20 | 20 | both_only | 1 | exact |
| A2_iter50 | 50 | both_only | 1 | exact |

**测试场景**: 选择一个代表性 gap（例如 `current_window=[0,34]` midpoint `frame_id=17`），单次 event，记录：
- `loss_real / loss_pseudo / loss_pseudo_pose / loss_pseudo_scene` 曲线
- `gaussian_xyz_max_abs_delta / gaussian_opacity_max_abs_delta`
- 最终 replay PSNR（如果 event 后能做 local replay）

**成功判据**: 
- A1 相对 A0 有明显正向趋势（loss 更稳定、delta 更合理）
- A2 进一步验证或找到最优 iter 数区间

**预期问题**: 
- 如果 iter=50 仍不稳定，说明不仅是迭代次数问题，需要结合 P1/P2

---

### Phase B: mask mode 放宽测试（P1）

**目标**: 验证"supervision 稀疏性"是否是主瓶颈

**实验设计**:

| arm | pseudo_map_iters | pseudo_scene_mask_mode | pseudo_views_per_event | contract |
|------|------------------|------------------------|------------------------|----------|
| B0_control | 20 | both_only | 1 | exact |
| B1_valid_only | 20 | valid_only | 1 | exact |
| B2_all_valid | 20 | all_valid | 1 | exact |

**依赖**: Phase A 确定合理的 iter baseline（假设选 iter=20）

**测试场景**: 同上单 gap

**关键指标**:
- `pseudo_effective_mask_nonzero_ratio`（预期: both_only ~1-2% → valid_only ~5-10% → all_valid ~15-20%）
- `loss_pseudo` 曲线是否更平滑
- replay PSNR

**成功判据**:
- B1 或 B2 相对 B0 有正向改善
- 若 B2 反而更差，说明过度放宽 supervision 引入噪声，需要回头收紧

---

### Phase C: multi pseudo views per event（P2）

**目标**: 验证"pseudo mass 不够"是否是次级瓶颈

**实验设计**:

| arm | pseudo_map_iters | pseudo_scene_mask_mode | pseudo_views_per_event | contract |
|------|------------------|------------------------|------------------------|----------|
| C0_control | 20 | valid_only | 1 | exact |
| C1_2views | 20 | valid_only | 2 | exact |
| C2_3views | 20 | valid_only | 3 | exact |

**依赖**: Phase B 确定 mask mode baseline（假设选 valid_only）

**基础设施需求**: 
- 当前 runtime slot selector 只选 single midpoint pseudo
- 需要扩展为 multi-slot selector（例如 `[17, 19]` 或 `[17, 19, 21]`）
- 这是小量代码改动，但需要先落地再跑实验

**成功判据**:
- C1 相对 C0 有正向改善
- 若 C1 更差，说明单 pseudo 已经足够，多 pseudo 引入额外 mismatch

---

### Phase D: paper-realign contract 切换（P3）

**目标**: 验证"rgb/depth 解耦 + mask 不用 verifier"是否更适合 online setting

**实验设计**:

| arm | pseudo_map_iters | pseudo_scene_mask_mode | pseudo_views_per_event | contract |
|------|------------------|------------------------|------------------------|----------|
| D0_control | 20 | valid_only | 1 | exact |
| D1_paper_exact_iter | 20 | valid_only | 1 | paper |

**paper contract 组成**:
- M~ = `paper_cm_only`（fused-domain support sets, **no geometry verifier**）
- T~ = `paper_brpo_target_v1`（depth-only bidirectional projection, no RGB gating）
- R~ = `paper_brpo_split_v1`（RGB 只用 `C_m`, depth 再乘 depth-only `target_confidence`）

**关键差异**:
- exact contract: RGB/depth 共用 verifier-gated `C_m`，depth 使用 `exact_brpo_upstream_target_v1`
- paper contract: 取消 verifier gating，supervision 覆盖更大，但可能引入更多噪声

**成功判据**:
- D1 相对 D0 有明显正向改善
- 若 D1 更差，说明 verifier gating 在 online setting 下仍有价值

---

### Phase E: 组合优化（跨轴组合）

**目标**: 找到最优组合

**实验设计**: 基于前面四个 phase 的 winner，组合成最终候选

| arm | pseudo_map_iters | pseudo_scene_mask_mode | pseudo_views_per_event | contract |
|------|------------------|------------------------|------------------------|----------|
| E0_baseline | 2 | both_only | 1 | exact |
| E1_best_single_axis | (P0 winner) | (P1 winner) | 1 | exact |
| E2_multi_view | (P0 winner) | (P1 winner) | (P2 winner) | exact |
| E3_paper_combo | (P0 winner) | (P1 winner) | 1 | paper |
| E4_full_combo | (P0 winner) | (P1 winner) | (P2 winner) | paper |

**测试场景扩展**: 
- 单 gap → 成功后扩展到 multi-gap representative sequence（例如 5 keyframe events）
- 最终做 full sequence online compare vs standalone baseline

---

## 4. 基础设施需求清单

| 需求 | 当前状态 | 改动量 | 优先级 |
|------|---------|-------|-------|
| pseudo_map_iters 可配置 | ✅ 已有 | 无 | P0 即可跑 |
| pseudo_scene_mask_mode 三档 | ✅ 已有 `both_only / valid_only / all_valid` | 无 | P1 即可跑 |
| multi pseudo views per event | ❌ 需要 runtime multi-slot selector | 小量代码 | P2 前落地 |
| paper-realign contract 接入 online | ✅ 已有 paper_* producer/loss | 需配置 glue | P3 可跑 |

---

## 5. 预期时间线

| Phase | 预估工作量 | 预估时间 |
|-------|----------|---------|
| A (iter test) | 3 arms × single-gap smoke | 1-2 天 |
| B (mask mode test) | 3 arms × single-gap | 1 天 |
| P2 infra (multi-slot selector) | 代码落地 + smoke | 1-2 天 |
| C (multi view test) | 3 arms × single-gap | 1 天 |
| D (paper contract test) | 2 arms × single-gap | 1 天 |
| E (combo test) | 5 arms × multi-gap | 2-3 天 |

---

## 6. 执行策略

### 6.1 推荐执行顺序

1. **先跑 Phase A**: 验证迭代次数是否是主瓶颈，这是最小变量测试
2. **同时准备 P2 infra**: multi-slot selector 可在 Phase A/B 期间并行开发
3. **Phase A 结果决定后续策略**:
   - 若 iter=50 仍不稳定 → 优先结合 P1 (mask mode) 再做 iter
   - 若 iter=20 已经稳定 → 直接进入 P1 mask mode 测试
4. **Phase D paper contract**: 作为对照分支，帮助理解 exact contract 的 verifier gating 在 online setting 下是否仍然必要

### 6.2 测试场景选择

**推荐测试 gap**: `current_window=[0,34]`，midpoint pseudo `frame_id=17`
- 这是一个已经验证过的 smoke scenario
- Left/right refs 为 `(0,34)`，pseudo slot 为 `17`
- Phase 3 smoke 已在此场景成功执行

**扩展场景**: 5 keyframe events 的 representative sequence
- 待单 gap 验证成功后选择
- 需要覆盖更多 pseudo slots 和更大的 Gaussian 变化范围

---

## 7. 相关文档

- Phase 3 smoke 记录: `/data/bzhang512/tmp/brpo_online_mapping_phase3_smoke/event_kf_0034/`
- Paper-realign 分支落地: `docs/current/CHANGELOG.md` (2026-05-04 section)
- S3PO integration plan: `docs/S3PO_PIPELINE_MAPPING_INTEGRATION_PLAN_20260503.md`
- Refine forensic master: `docs/REFINE_FORENSICS_MASTER_20260425.md`
- M3D experiments archived: `docs/archived/2026-04-m3d-experiments/`

---

## 8. 状态跟踪

| Phase | 状态 | 开始时间 | 完成时间 | 结果摘要 |
|-------|------|---------|---------|---------|
| A | 待执行 | — | — | — |
| B | 待执行 | — | — | — |
| P2 infra | 待执行 | — | — | — |
| C | 待执行 | — | — | — |
| D | 待执行 | — | — | — |
| E | 待执行 | — | — | — |

---

> **注**: 本文档记录规划阶段，实际执行结果将更新到 CHANGELOG.md 和 STATUS.md
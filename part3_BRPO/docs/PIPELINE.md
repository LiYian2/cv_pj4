# PIPELINE.md

> Purpose: compact source-of-truth for the current Part3 BRPO pipeline.
> Update: 2026-04-23 00:38 CST — 与 `docs/current/STATUS.md` / `docs/current/DESIGN.md` 对齐到当前 standalone winner 口径。

---

## 1. 口径说明

- **M~** = Mask（confidence），定义监督域与监督强度
- **T~** = Target，定义监督目标数值
- **G~** = Gaussian Management，定义 per-Gaussian 参与控制
- **R~** = Topology，定义 joint refine 的 loss assembly / backward timing

历史映射：A1 → M~ + T~，B3 → G~，T1 → R~。

---

## 2. 一句话总览

当前 standalone winner 已固定为：**exact M~ + exact upstream T~ + clean summary G~ + T1**。

更具体地说：
- M~：`exact_brpo_cm_old_target_v1` 已证明与 `old A1` 基本等价，M~ 不再是主瓶颈
- T~：`exact_brpo_upstream_target_v1 + exact_shared_cm_v1` 已在 fixed clean G~ / fixed T1 protocol 下赢过 `old A1`、`exact_brpo_cm_old_target_v1`、`exact_brpo_full_target_v1`
- G~：当前保持 `summary_only` / clean summary control，不再把 G~ 当 standalone 主突破口
- R~：`joint_topology_mode=brpo_joint_v1` 仍是固定 topology 主线

当前固定参照线仍是：**RGB-only v2 + gated_rgb0192 + post40_lr03_120**。

---

## 3. 当前主数据流

`part2 S3PO full rerun`
→ `internal_eval_cache`
→ `prepare_stage1_difix_dataset_s3po_internal.py`
→ exact verifier/backend + exact upstream target build
→ `run_pseudo_refinement_v2.py`（`exact_shared_cm_v1` + `brpo_joint_v1` + clean summary G~）
→ replay eval

一句话理解：当前真正赢下 formal compare 的关键，不是继续做 consumer-side 小修补，而是把 verifier backend / projected-depth / target field 一起推进到 exact-upstream 语义。

---

## 4. 四大模块当前口径

### 4.1 M~
- 语义：监督域 + 监督强度
- 当前判断：`exact_brpo_cm_old_target_v1 ≈ old A1`
- 结论：M~ 已完成语义对齐，但增益不来自覆盖率变大，而来自 mask semantics 被收干净
- 详细：`docs/design/MASK_DESIGN.md`

### 4.2 T~
- 语义：depth target / target contract
- 当前主线：`exact_brpo_upstream_target_v1 + exact_shared_cm_v1`
- 结论：strict BRPO T~ 的正向 winner 来自更 upstream 的 exact backend / projected-depth / target field 对齐
- 详细：`docs/design/TARGET_DESIGN.md`

### 4.3 G~
- 语义：per-Gaussian gating / participation control
- 当前主线：clean summary control（`summary_only`）
- 结论：G~ clean compare 已完成；当前收益有限，保留干净 summary 作为 control 即可
- 详细：`docs/design/GAUSSIAN_MANAGEMENT_DESIGN.md`

### 4.4 R~
- 语义：joint loss assembly + backward timing
- 当前主线：`joint_topology_mode=brpo_joint_v1`
- 结论：T1 已固定，不再继续扫 standalone topology compare
- 详细：`docs/design/REFINE_DESIGN.md`

---

## 5. 当前推荐组合与 control

- 当前默认主线：`exact_brpo_upstream_target_v1 + exact_shared_cm_v1 + clean summary G~ + brpo_joint_v1`
- 当前 semantics-clean control：`exact_brpo_cm_old_target_v1 + brpo_joint_v1 + clean summary G~`
- 历史强 control：`old A1 + new T1`

---

## 6. 数据集状态

| 数据集 | 当前状态 | 备注 |
|--------|----------|------|
| Re10k-1 | standalone winner 已冻结到 `exact M~ + exact upstream T~ + clean summary G~ + T1` | 下一步重点转 backend-only integration |
| DL3DV-2 | canonical baseline / repair A 已打通 | 暂未平移 strict BRPO A1/T~ 主线 |

---

## 7. 推荐阅读顺序

1. `docs/current/STATUS.md` — 当前状态与结论
2. `docs/current/DESIGN.md` — 系统边界与模块关系
3. `docs/current/CHANGELOG.md` — 已落地改动与时间线
4. `docs/design/MASK_DESIGN.md`
5. `docs/design/TARGET_DESIGN.md`
6. `docs/design/GAUSSIAN_MANAGEMENT_DESIGN.md`
7. `docs/design/REFINE_DESIGN.md`

历史执行计划与长文分析若还需要，再按需回看对应 archived / root 文档，不再把它们当作当前 authority。

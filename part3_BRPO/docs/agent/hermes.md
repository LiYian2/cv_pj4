# hermes.md

> 用途：Part3 BRPO 压缩/重启后的第一入口。先看这份，再按这里列的顺序继续。
> 维护原则：只保留当前真实状态、当前执行顺序、关键文档入口和固定环境信息。
> 更新时间：2026-04-24 08:28 (Asia/Shanghai)

---

## 1. 先看什么

如果用户让我“先回忆一下现在做到哪了”，按这个顺序：
1. 本文件 `docs/agent/hermes.md`
2. `docs/current/STATUS.md`
3. `docs/current/DESIGN.md`
4. `docs/current/CHANGELOG.md`
5. 如果要继续工程整理，先看 `docs/SCRIPTS_FINAL_AUDIT_STAGE34_20260422.md`，再看 `docs/SCRIPTS_FINAL_AUDIT_STAGE12_20260422.md`、`docs/PSEUDO_BRANCH_T_RESIDUAL_CLEANUP_PHASE6_20260422.md` 与 `docs/design/PSEUDO_BRANCH_LAYOUT.md`
6. 如果要继续做 M~ matching 升级，先看 `docs/BRPO_MASK_MAST3R_3D_MATCHING_PLAN_20260424.md`，再看 `docs/BRPO_MASK_DENSE_2D_MATCHING_PLAN_20260424.md`
7. 需要长文证据时再看：
   - `docs/archived/2026-04-experiments/G_BRPO_CLEAN_COMPARE_20260421.md`
   - `docs/SCRIPTS_FINAL_AUDIT_STAGE34_20260422.md`
   - `docs/SCRIPTS_FINAL_AUDIT_STAGE12_20260422.md`
   - `docs/PSEUDO_BRANCH_T_RESIDUAL_CLEANUP_PHASE6_20260422.md`
   - `docs/PSEUDO_BRANCH_COMMON_MIGRATION_PHASE5_20260422.md`
   - `docs/PSEUDO_BRANCH_M_MIGRATION_PHASE4_20260422.md`
   - `docs/PSEUDO_BRANCH_T_OBSERVATION_MIGRATION_PHASE3_20260422.md`
   - `docs/PSEUDO_BRANCH_R_MIGRATION_PHASE2_20260422.md`
   - `docs/PSEUDO_BRANCH_G_MIGRATION_PHASE1_20260422.md`
   - `docs/T4_EXACT_UPSTREAM_COMPARE_PLAN_20260421.md`
   - `docs/design/` 下四个模块设计文档

---

## 2. 当前固定结论

- 当前 standalone winner 已更新为：`exact M~ + exact upstream T~ + clean summary G~ + T1`
- `exact_brpo_upstream_target_v1 + exact_shared_cm_v1` 已在 fixed clean G~ / fixed T1 compare 中赢过 `old A1`、`exact_brpo_cm_old_target_v1`、`exact_brpo_full_target_v1`
- `old A1 + new T1` 与 `exact_brpo_cm_old_target_v1 + clean summary G~ + T1` 现在都只保留为 control
- G~ clean compare 结论不变：direct BRPO current-step 只有小幅正向；legacy delayed opacity 明确负向；G~ 冻结为 side branch

---

## 3. 当前工程主线

主线已经从 standalone compare 转向 **S3PO backend-only integration**：
1. 冻结 `exact M~ + exact upstream T~ + clean summary G~ + T1` 这套 winner
2. 若用户要求继续做 M~ matching upgrade，当前优先级应是：先做 `MASt3R 3D matching` 主线，再保留 `dense2d` 作为低风险 control / side option；两条都保持 exact BRPO 离散 `C_m` 语义，不要先改成 continuous mask
3. M~ 3D 路线现在已经完成 live wiring：`scripts/brpo_build_mask_from_internal_cache.py` 与 `scripts/build_brpo_v2_signal_from_internal_cache.py` 已接到 matcher factory。grounded 结果见 `docs/M3D_LIVE_WIRING_AND_SMOKE_20260424.md`：frame 23 单帧 live sweep 下，backend exact `cm_nonzero_ratio` 为 sparse `0.0164`、dense3d `q0.90=0.0576` / `q0.80=0.1261` / `q0.70=0.1921`；signal `joint_nonzero_ratio` 为 `0.0200 / 0.0754 / 0.1531 / 0.2271`。8 帧 full smoke 中，dense3d `q0.80` 的 backend / signal mean 覆盖率已到 `0.1275 / 0.1591`，q0.80 tiny consumer smoke 也已跑通。若继续往下做，优先补 q0.70 / q0.90 的 full compare，再决定是否扩成多 quantile consumer compare
4. 把 builder / verifier backend / loss contract 抽成 backend-only 可复用模块
5. 保持 pseudo supervision 只进 backend refine，不回灌 tracking / frontend
6. 工程整理同步推进：
   - `scripts/` 顶层现只保留 8 个 live core + 1 个外部 CLI wrapper；内部 compatibility boundary 已收进 `scripts/compat/`，non-live diagnostics 已收进 `scripts/diagnostics/`，legacy prepare 已收进 `scripts/legacy_prepare/`，历史 compare runner 归档在 `scripts/archive_experiments/`
   - `pseudo_branch/` 已完成 G~ Phase 1 + R~ Phase 2 + Phase 3 T~/observation + Phase 4 M~ + Phase 5 common + Phase 6 residual T~ cleanup：G~ 已收进 `pseudo_branch/gaussian_management/`，R~ 已收进 `pseudo_branch/refine/`，observation 主入口已收进 `pseudo_branch/observation/`，mask 主入口已收进 `pseudo_branch/mask/`，target 主入口与 residual T~ builder 已全部收进 `pseudo_branch/target/`，common 主入口已收进 `pseudo_branch/common/`
   - `pseudo_branch/` 顶层现在只剩 `__init__.py`；scripts final audit 的 Stage 1 / 2 / 3 / 4 也已完成，代码路径整理本身可以视为结束；若还要继续工程收尾，重点应转向 backend-only integration 与 working-tree/commit hygiene，而不是再继续改布局；详见 `docs/SCRIPTS_FINAL_AUDIT_STAGE34_20260422.md`

---

## 4. 当前不要做的事

1. 不再引用旧 `+0.0114` G~ baseline
2. 不再把 delayed opacity / O2a-b 当 G~ 主推进路线
3. 不在 observation compare 里同时改 topology 或 G~
4. 不再把 proxy-backend exact target 包装成“已经对齐的 strict BRPO winner”
5. 不继续维持 `pseudo_branch/` 平铺加长期兼容壳的目录状态

---

## 5. 一句话 handoff

当前已经完成 T4 exact-upstream formal compare，standalone winner 固定为 `exact M~ + exact upstream T~ + clean summary G~ + T1`；G~ 只保留 side branch。工程整理这边 pseudo_branch 第二轮整理与 scripts final audit Stage 1 / 2 / 3 / 4 都已落地；代码路径整理现在可以视为完成，下一步回到 backend-only integration，并在最后统一处理 working-tree/commit 收尾。

---

## 6. 云服务器环境信息

### SSH alias
- Group8DDY

### Python 环境
- Conda env: s3po-gs
- 路径: /home/bzhang512/miniconda3/envs/s3po-gs/bin/python

### PYTHONPATH
- /home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO

### 项目路径
- Part3 root: /home/bzhang512/CV_Project/part3_BRPO
- 输出: /data2/bzhang512/CV_Project/output/part3_BRPO/experiments

### 执行模板

```bash
ssh Group8DDY "cd /home/bzhang512/CV_Project/part3_BRPO && export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO && /home/bzhang512/miniconda3/envs/s3po-gs/bin/python scripts/xxx.py --args"
```

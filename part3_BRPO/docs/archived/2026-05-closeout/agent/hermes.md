# hermes.md

> 用途：Part3 BRPO 压缩/重启后的第一入口。先看这份，再按这里列的顺序继续。
> 维护原则：只保留当前真实状态、当前执行顺序、关键文档入口和固定环境信息。
> 更新时间：2026-04-30 20:18 (Asia/Shanghai)

---

## 1. 先看什么

如果用户让我“先回忆一下现在做到哪了”，按这个顺序：
1. 本文件 `docs/agent/hermes.md`
2. `docs/current/STATUS.md`
3. `docs/current/DESIGN.md`
4. `docs/current/CHANGELOG.md`
5. 如果要继续工程整理，先看 `docs/archived/2026-04-cleanup-records/SCRIPTS_FINAL_AUDIT_STAGE34_20260422.md`，再看 `docs/archived/2026-04-cleanup-records/SCRIPTS_FINAL_AUDIT_STAGE12_20260422.md`、`docs/archived/2026-04-cleanup-records/PSEUDO_BRANCH_T_RESIDUAL_CLEANUP_PHASE6_20260422.md` 与 `docs/design/PSEUDO_BRANCH_LAYOUT.md`
6. 如果要继续做 refine 结构重构，先看 `docs/REFINE_SLAM_STYLE_AUDIT_20260430.md` 与 `docs/REFINE_SLAM_STYLE_EXEC_PLAN_20260430.md`；如果要继续 forensic，再看 `docs/REFINE_FORENSICS_MASTER_20260425.md`；如果要补 matching 升级设计，再看 `docs/archived/2026-04-m3d-experiments/BRPO_MASK_MAST3R_3D_MATCHING_PLAN_20260424.md` 与 `docs/archived/2026-04-m3d-experiments/BRPO_MASK_DENSE_2D_MATCHING_PLAN_20260424.md`
7. 需要长文证据时再看：
   - `docs/archived/2026-04-experiments/G_BRPO_CLEAN_COMPARE_20260421.md`
   - `docs/archived/2026-04-cleanup-records/SCRIPTS_FINAL_AUDIT_STAGE34_20260422.md`
   - `docs/archived/2026-04-cleanup-records/SCRIPTS_FINAL_AUDIT_STAGE12_20260422.md`
   - `docs/archived/2026-04-cleanup-records/PSEUDO_BRANCH_T_RESIDUAL_CLEANUP_PHASE6_20260422.md`
   - `docs/archived/2026-04-cleanup-records/PSEUDO_BRANCH_COMMON_MIGRATION_PHASE5_20260422.md`
   - `docs/archived/2026-04-cleanup-records/PSEUDO_BRANCH_M_MIGRATION_PHASE4_20260422.md`
   - `docs/archived/2026-04-cleanup-records/PSEUDO_BRANCH_T_OBSERVATION_MIGRATION_PHASE3_20260422.md`
   - `docs/archived/2026-04-cleanup-records/PSEUDO_BRANCH_R_MIGRATION_PHASE2_20260422.md`
   - `docs/archived/2026-04-cleanup-records/PSEUDO_BRANCH_G_MIGRATION_PHASE1_20260422.md`
   - `docs/archived/2026-04-plans-landed/T4_EXACT_UPSTREAM_COMPARE_PLAN_20260421.md`
   - `docs/design/` 下四个模块设计文档

---

## 2. 当前固定结论

- 当前 standalone winner 已更新为：`exact M~ + exact upstream T~ + clean summary G~ + T1`
- refine 的工程主线已经从 standalone consumer shell 切到 **S3PO backend continuation**；`run_pseudo_refinement_v2.py` 只保留 reference 身份。
- Phase 1 / Phase 2 已经 landed：pseudo bundle / pseudo loss backend 模块已加入 live repo，`slam_backend_brpo.py` 与 `slam.py` hook 已接通。
- 已完成 actual hook smoke，产物根为 `/data/bzhang512/tmp/s3po_brpo_hook_manual_smoke/`；当前验证边界是 hook-level after_opt smoke，不是 full frontend rerun compare。
- `exact_brpo_upstream_target_v1 + exact_shared_cm_v1` 已在 fixed clean G~ / fixed T1 compare 中赢过 `old A1`、`exact_brpo_cm_old_target_v1`、`exact_brpo_full_target_v1`
- `old A1 + new T1` 与 `exact_brpo_cm_old_target_v1 + clean summary G~ + T1` 现在都只保留为 control
- G~ clean compare 结论不变：direct BRPO current-step 只有小幅正向；legacy delayed opacity 明确负向；G~ 冻结为 side branch
- `cm_only` 与 `rgb_only` 两个 consumer-side ablation 都已完成：去掉 `target_confidence` 不能修复 dense，完全关掉 pseudo depth loss 也不能修复 dense；后者还会明显伤 sparse。因此 dense3d 当前主问题不能简化成“confidence 多余”或“depth loss 质量差”。
- 当前 refine forensic 主结论已固定：旧 live M~ 的低 coverage 确实来自 sparse 2D reciprocal matching，但 replay 真正恶化的根因不是 coverage 本身，而是 stronger pseudo route 下更大的 target mismatch、更高的 single-branch 占比、弱 real anchor，以及 joint pseudo pose + Gaussian feedback 的渐进式失稳。
- 新的 `4real+2pseudo` dense q070 follow-up 已完成：只改 batch ratio 并不能救回 current exact dense route；`current_4r2p_exact` 与 `rgb_only_4r2p_exact` 都比旧 `2+4` q070 controls 更差。唯一强阳性的 arm 是 `allones_rgb_only_4r2p`（`24.1126 PSNR`），因此接下来应优先怀疑 exact pseudo contract，而不是把 minibatch ratio 当成主修复方向。
- 同步补做的 `continuous-confidence + RGB-only` dense q070 arm 也没有翻盘：`contconf_rgb_only_4r2p = 23.5406 PSNR`，只比离散 exact RGB-only 高 `+0.0210`，但显著低于 `allones_rgb_only_4r2p`。这与更早连续/hybrid M 线历史偏弱的记录一致。

---

## 3. 当前工程主线

主线已经从 standalone compare 转向 **S3PO backend-only integration**：
1. 冻结 `exact M~ + exact upstream T~ + clean summary G~ + T1` 这套 winner
2. 若用户要求继续做 M~ matching upgrade，当前优先级应是：先做 `MASt3R 3D matching` 主线，再保留 `dense2d` 作为低风险 control / side option；两条都保持 exact BRPO 离散 `C_m` 语义，不要先改成 continuous mask
3. M~ dense3d 的长程 `StageB120 + replay` compare 与后续 structural forensic 已完成，详见 `docs/archived/2026-04-m3d-experiments/M3D_STAGEB120_REPLAY_COMPARE_20260424.md` 与 `docs/archived/2026-04-m3d-experiments/M3D_STRUCTURAL_FORENSICS_20260424.md`：虽然 dense3d 在 mechanism 层显著增密，且 q0.70 仍是 dense 内部最优候选，但真正的 replay winner 仍然是 sparse（PSNR 24.0045 > 23.6660 > 23.5816）。更关键的是，forensic 已确认旧 live M~ 确实是 sparse 2D reciprocal matching，而 dense3d 接通后 exact-upstream target depth 也确实同步变化；当前更像是新增 supervision 的 single/both 组成与 effective weighting 有问题，而不是 target depth 没切过去。因此不要继续做同类 q-sweep，也不要现在把 dense3d 默认化；下一步若继续推进 M~，应优先回到 BRPO 原始 method 对照 M~/T~ 语义，而不是继续扫 quantile。
4. 新的 consumer-side `cm_only` ablation（`docs/archived/2026-04-m3d-experiments/M3D_CM_ONLY_STAGEB120_REPLAY_COMPARE_20260424.md`）已经验证：把 `exact_shared_cm_v1` 改成只用裸 `C_m`、不乘 `valid_mask / target_confidence` 后，sparse 与 q070 都更差。当前 live exact-upstream 里 `valid_mask` 与 `C_m` 区域重合，因此真正不能简单拿掉的是 `target_confidence`；它至少在现有实现中起到了必要抑制，而不是纯多余组件。
4. 把 builder / verifier backend / loss contract 抽成 backend-only 可复用模块
5. 保持 pseudo supervision 只进 backend refine，不回灌 tracking / frontend
6. 工程整理同步推进：
   - `scripts/` 顶层现只保留 8 个 live core + 1 个外部 CLI wrapper；内部 compatibility boundary 已收进 `scripts/compat/`，non-live diagnostics 已收进 `scripts/diagnostics/`，legacy prepare 已收进 `scripts/legacy_prepare/`，历史 compare runner 归档在 `scripts/archive_experiments/`
   - `pseudo_branch/` 已完成 G~ Phase 1 + R~ Phase 2 + Phase 3 T~/observation + Phase 4 M~ + Phase 5 common + Phase 6 residual T~ cleanup：G~ 已收进 `pseudo_branch/gaussian_management/`，R~ 已收进 `pseudo_branch/refine/`，observation 主入口已收进 `pseudo_branch/observation/`，mask 主入口已收进 `pseudo_branch/mask/`，target 主入口与 residual T~ builder 已全部收进 `pseudo_branch/target/`，common 主入口已收进 `pseudo_branch/common/`
   - `pseudo_branch/` 顶层现在只剩 `__init__.py`；scripts final audit 的 Stage 1 / 2 / 3 / 4 也已完成，代码路径整理本身可以视为结束；若还要继续工程收尾，重点应转向 backend-only integration 与 working-tree/commit hygiene，而不是再继续改布局；详见 `docs/archived/2026-04-cleanup-records/SCRIPTS_FINAL_AUDIT_STAGE34_20260422.md`

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


## 2026-05-09 handoff — E7a 2IMG+PAIR binary cap
- Current E7a path was patched and rerun in-place. Code: pseudo_branch/common/twoimg_pair_proxy_depth.py::apply_cm_cap now uses binary (C_m > 0) support; do not restore depth * C_m.
- Rerun output: /data3/bzhang512/part3_online_mapping_experiments/E7a_jointprimary_twoimg_pair_proxy/DL3DV-2_part2_s3po/2026-05-09-15-51-46.
- Verified P0 fix in artifacts, but result is mixed: PSNR improved to 20.597 yet still below E5c 21.201; ATE/stats_final is bad at 0.329. Next reentry should investigate post-fix 2img depth value quality and pose/geometry drift, not re-open the old 0.5 cap bug.


## 2026-05-09 handoff — E7a depth-off winner
- Do not conclude E7a failed because pseudo online mapping is useless. Clean ablation shows the opposite: E7a_binarycap_depthoff reaches after_opt PSNR 21.633 and stats_final RMSE 0.0619, better than E5c PSNR 21.201 / RMSE 0.0645.
- The harmful component is current 2IMG dense depth loss. Post-binarycap depth-enabled E7a has PSNR 20.597 and RMSE 0.329, with spikes around frame 264 and frames 302/303/305.
- Artifact diagnosis: no global scale drift on shared E5c support (ratio about 1.01), but abs-rel about 0.20; right PAIR anchor disagreement is large (mean median abs-rel about 0.31, late events around 0.63); added support is mostly single-support.
- If re-entering: preserve apply_cm_cap binary support; for ablations remember lambda_depth=0 requires match_real_loss_weights=false, otherwise resolver resets lambda_depth to 0.025 from Training.alpha. Next depth repair should test both-only / anchor-valid / right-left-consistency gated depth, not full dense 2IMG depth.


## 2026-05-10 E8 C_m local expansion audit

- Fixed C_m expansion metadata/stat reporting: observation summaries now separate raw reciprocal C_m stats from consumed soft-C_m stats; diagnostic sidecar dry-run no longer writes frame outputs; sidecar summaries include depth target filled before/after and complete reject counters.
- Audited E8 at /home/bzhang512/my_storage2_1T/part3_online_mapping_experiments/E8_cm_local_expand_r1_soft. The run is protocol-aligned with E5c except cm_expansion_mode=local_soft_v1 and cm_expansion_apply_to_depth_scope=false.
- Final metrics: E8 after_opt PSNR 20.5222, SSIM 0.6629, LPIPS 0.2684, stats_final RMSE 0.0875. E5c reference: PSNR 21.2012, RMSE 0.0645. E8 degraded.
- Code/production-flow check: raw support is preserved; signal confidence equals cm_expanded_soft and not cm_raw; depth target/valid scope stays raw/projected. No current hard wiring bug was found in the audited E8 artifacts.
- Mechanistic diagnosis: local expansion adds about 8-16 percent image area as RGB-only soft C_m; depth_in_added is 0 for all audited events because apply_to_depth_scope=false and projected depth target remains raw. Since paper_brpo_split_v1 RGB loss normalizes by confidence_mask.sum, added easy/weak pixels dilute raw reciprocal seed RGB gradients by about 10-25 percent while adding no geometric anchor. This is the leading explanation for lower training pseudo losses but worse final PSNR/ATE.
- Next recommendation: do not continue full local_soft_v1 as-is. Test conservative variants: both-only/near-depth-valid expansion, or budget-preserving reweighting that keeps raw seed gradient mass constant; optionally pair with depth-off if isolating pure RGB expansion.


## 2026-05-10 handoff — dense_match_v1 landed
- Added standalone peer-style RGB-only support mode without geometry overlap gating: `rgb_only_support_mode=dense_match_v1`.
- Implementation files: `pseudo_branch/mask/dense_match_densify.py`, `pseudo_branch/integration/runtime_exact_backend.py`, `third_party/S3PO-GS/utils/slam_backend.py`.
- Semantics: reciprocal match points -> disk rasterization -> Gaussian blur -> normalize -> threshold; this only changes RGB-only branch support/C_m coverage. Exact projected depth target path is unchanged.
- Important non-overlap rule: runtime rejects `dense_match_v1` together with `cm_expansion_mode != none`; keep dense-match experiments separate from E8/local_soft_v1 identity.
- New YAML knobs: `rgb_only_support_mode`, `cm_dense_point_radius`, `cm_dense_blur_sigma`, `cm_dense_blur_kernel`, `cm_dense_corr_threshold`, `cm_dense_seed_mode`, `cm_dense_normalize_mode`.
- Debug artifacts: raw reciprocal support still saved separately; dense outputs live under `exact_backend_v1/dense_match_v1/` with `dense_match_meta.json`.
- Verification already done: py_compile/import smoke; direct toy smoke showed dense support ratio > raw support ratio; BackEnd resolver smoke confirmed YAML -> runtime propagation.

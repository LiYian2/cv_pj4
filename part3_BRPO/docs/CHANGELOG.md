# CHANGELOG.md - Part3 Stage1 过程记录

> 本文件记录每次工作的过程、发现和结果。
> 更新规则：以日期为单位记录；允许为修正文档一致性而整理旧条目，但不要把“现状”写进本文件。

---

## 2026-04-17

### P2-M：SPGM conservative deterministic repair compare 完成

运行根：
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_p2m_spgm_conservative_repair_e1`

新增文档：
- `docs/P2M_spgm_conservative_repair_compare_20260417.md`

协议：
1. 完全复用 `P2-L` 的 canonical StageB protocol；
2. 不改 handoff / schedule / upstream gate，只做 deterministic SPGM policy repair；
3. 新增两条 conservative repair 臂：
   - A: `keep=(1,1,1), support_eta=0.0, weight_floor=0.25`
   - B: `keep=(1,1,0.9), support_eta=0.25, weight_floor=0.20`

关键结果：
- previous SPGM：`23.94230 / 0.86956 / 0.08359`
- repair A：`24.00212 / 0.87132 / 0.08244`
- repair B：`23.99595 / 0.87115 / 0.08259`
- canonical baseline：`24.02982 / 0.87229 / 0.08177`

机制层结论：
- repair A / B 都明显优于 previous SPGM，说明 current policy 的主要问题确实包含 suppress 过强；
- 但两臂 rejection 与 active ratio 仍几乎不变，真正回升的是 `grad_weight_mean_xyz_mean`（`0.3586 -> 0.5497 / 0.4977`），说明这一步仍主要是在修 suppress 强度，而不是改 selection；
- best arm 为 A，但它仍略低于 canonical baseline。

新的项目判断：
- conservative deterministic repair 已证明这条方向能部分追回 replay；
- 但只做更温和的 soft suppress 已接近边际，下一步应从 A 臂出发做 selector-first rewrite；
- 在 selector-first 之前，不推进 stochastic drop、`xyz+opacity` 或更长 iter。

---

### P2-L：SPGM canonical StageB formal compare 完成

运行根：
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_p2k_canonical_stageB_compare_e1`

新增文档：
- `docs/P2L_spgm_canonical_stageB_compare_20260417.md`

协议：
1. 完全复用 canonical bounded StageB baseline：`RGB-only v2 + gated_rgb0192 + post40_lr03_120`；
2. 完全复用顺序 handoff：`init_pseudo_camera_states_json = stageA5_v2rgbonly_xyz_gated_rgb0192_80/pseudo_camera_states_final.json`；
3. 只替换 Gaussian-side pseudo grad manager：baseline=`hard_visible_union_signal`，exp=`spgm_keep`；
4. 其余设置保持一致：同 pseudo cache、同 `signal_v2`、同 real branch、同 seed、同 replay evaluator。

关键结果：
- canonical baseline：`24.02982 / 0.87229 / 0.08177`
- canonical SPGM：`23.94230 / 0.86956 / 0.08359`
- delta（SPGM - baseline）：`-0.08753 PSNR / -0.00273 SSIM / +0.00182 LPIPS`

机制层结论：
- aligned compare 下，两臂 rejection 完全一致：`96/120` iter、拒绝 `225 / 260`；
- `grad_keep_ratio_xyz_mean` 几乎不变：baseline `0.7290`，SPGM `0.7290`；
- `grad_weight_mean_xyz_mean` 明显下降：baseline `0.7290`，SPGM `0.3586`；
- `loss_real_last` 更低，但 `loss_pseudo_last` 更高，继续指向“强 suppressor 而不是 selector”。

新的项目判断：
- protocol drift 这一层已经被正式排除：即便对齐到 canonical baseline，当前 deterministic `SPGM v1` 仍低于 canonical baseline；
- 因而下一步不再是“先补 formal compare”，而是进入 `SPGM repair`：优先 conservative deterministic / selector-first 改写；
- 在此之前，不推进 stochastic drop、`xyz+opacity` 或更长 iter 放大。

---

## 2026-04-16


### P2-K：SPGM v1 forensic audit 完成

### P2-K：forensic follow-up code hygiene fixes + protocol confirmation

代码修补：
- `pseudo_branch/spgm/score.py`：修正 density entropy 归一化为 `H / log(B)`；同时让 `density_mode` 真正参与 density-side score / entropy 分支，而不再是 dead param
- `scripts/run_pseudo_refinement_v2.py`：SPGM 分支的 `accepted_count` 改为真实 accepted views 计数；并把 `density_mode_effective` 传入 summary/history
- `pseudo_branch/local_gating/gating_io.py`：新增 `spgm_density_mode_effective` summary 字段

验证：
- 本地与远端 `py_compile` 已通过
- 直接 smoke 已确认：
  - uniform 2-bin entropy 返回 `1.0`（归一化标尺已正确）
  - `density_mode='support'` 已被真实消费并写入 summary
  - `maybe_apply_pseudo_local_gating()` 在 1 accepted / 1 rejected 的 stub 场景下，`accepted_visibility_count` 与 `spgm_accepted_view_count` 都正确记录为 `1`

实验口径再确认：
- 当前 canonical reference baseline 仍应是 `post40_lr03_120 / gated_rgb0192`
- 已跑的 `20260416_spgm_v1_stageB_compare_120iter` 实际 protocol 为：`stageB_post_switch_iter=0`、`stageB_post_lr_scale_xyz=1.0`、`min_rgb_mask_ratio=0.01`、`spgm_keep/raw hard-visible-union compare`
- 因而它不是对 canonical bounded baseline 的正式对照

---


审查文档：
- `docs/P2K_spgm_v1_forensic_audit_20260416.md`

审查范围：
1. `STATUS / DESIGN / CHANGELOG / Noah.md / hermes.md`
2. `scripts/run_pseudo_refinement_v2.py` 与 `pseudo_branch/spgm/*`
3. `20260416_spgm_v1_stageA5_compare_e1`
4. `20260416_spgm_v1_stageB_compare_e1`
5. `20260416_spgm_v1_stageB_compare_120iter`

冻结事实：
- 代码主 wiring 是对的：SPGM 确实接在 `pseudo_backward -> maybe_apply_pseudo_local_gating -> real_backward` 上，grad 真实被调制，不是 no-op；
- 但当前 StageB-120 compare 不是对 canonical `post40_lr03_120 / gated_rgb0192` bounded baseline 的 apples-to-apples 对照；实际 run 使用的是 `hard_visible_union_signal`、`stageB_post_switch_iter=0`，且未复用既有 handoff；
- 120iter replay 结果确实为负：baseline `24.136 / 0.87385 / 0.08189`，exp `23.594 / 0.85926 / 0.08912`；
- 这轮自有 protocol 下，SPGM 更像是在几乎同一 active set 上做强连续衰减：`grad_keep_ratio_xyz_mean≈0.739` 基本不变，但 `grad_weight_mean_xyz_mean` 从 baseline 的约 `0.739` 压到 exp 的约 `0.292`。

确认的实现问题 / 未完成项：
- `score.py` 的 density entropy 归一化按 `entropy_bins` 而不是 `log(B)`；
- `spgm_density_mode` 当前只是暴露接口，尚未被 stats/score/policy 实际消费；
- SPGM 分支里的 accepted-count/history 统计写法对未来 reject-run 不严谨。

新的项目判断：
- 当前不能把 P2-K 写成“SPGM 已完成验证”；
- 下一步主线应改为 `SPGM forensic repair`：先把 compare protocol 拉回 canonical bounded baseline，再做 conservative deterministic SPGM（更高 `weight_floor`、更弱 `support_eta`、先去掉固定 cluster decay）；
- 在此之前，不直接跳去 stochastic drop / xyz+opacity / 更长 iter。

---

### P2-J：bounded StageB schedule compare 完成

运行根：
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2j_stageB_bounded_schedule_compare_e1`

新增文档：
- `docs/P2J_bounded_stageB_schedule_compare_20260416.md`

代码更新：
- `scripts/run_pseudo_refinement_v2.py` 已新增 StageB post-switch 调度参数：
  - `--stageB_post_switch_iter`
  - `--stageB_post_lr_scale_xyz`
  - `--stageB_post_lr_scale_opacity`
  - `--stageB_post_lambda_real`
  - `--stageB_post_lambda_pseudo`
- `stageB_history.json` 现会同步记录：
  - `lambda_real_effective`
  - `lambda_pseudo_effective`
  - `gaussian_lr_xyz_effective`
  - `post_switch_applied*`

协议：
1. 固定当前 winner `RGB-only v2 + gated_rgb0192`；
2. 固定总 budget 仍为 `120iter`；
3. 只做三组 bounded late-stage compare：
   - `post40_lr03_120`
   - `post80_lr03_120`
   - `post80_lr03_real05_120`

关键结果：
- `post40_lr03_120`: `24.03019 / 0.87229 / 0.08177`
- `post80_lr03_120`: `23.99358 / 0.87099 / 0.08307`
- `post80_lr03_real05_120`: `24.01828 / 0.87167 / 0.08255`
- 最佳臂为：`post40_lr03_120`

新的项目判断：
- bounded late-stage schedule 足以把原始 `120iter` cliff 明显拉回；
- 但这个结果更像是“给出一个足够好的 bounded StageB baseline”，而不是“证明 StageB 仍值得继续深调”；
- 因此下一步不再继续开新的 StageB schedule 网格，而是保留 `post40_lr03_120` 作为对照臂，同时把主工程重心转向 `SPGM` 第一版落地。

---

### P2-I：gated winner 的 `StageB` 回落窗口定位 + SPGM 方向激活

运行根：
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2i_stageB_window_localization_e1`

新增文档：
- `docs/P2I_stageB_window_localization_20260416.md`

协议：
1. 固定当前最佳分支 `RGB-only v2 + gated_rgb0192`；
2. 从同一个 `P2-F` gated StageA.5 handoff 起跑；
3. 做 `stageB_iters = 20 / 40 / 80 / 120` 的窗口定位；
4. 其余设置保持与 `P2-H` gated arm 一致。

关键结果：
- `20`: `24.01061 / 0.87302 / 0.08036`
- `40`: `24.02235 / 0.87253 / 0.08122`
- `80`: `24.01980 / 0.87179 / 0.08247`
- `120`: `23.91443 / 0.86873 / 0.08484`
- `40` 是当前 PSNR 最佳点；`20` 是当前 SSIM / LPIPS 最佳点；`80 -> 120` 出现明显 cliff。

机制层结论：
- gating 在整个 sweep 里都持续生效；`225 / 260` 持续被拒，`grad_keep_ratio_xyz_mean` 稳定在约 `0.73 ~ 0.74`；
- `120` 的 regression 不是 gate no-op，而是 late-stage 训练动力学失稳；最直接的伴随信号是 `loss_real` 在 `80` 之后明显反弹。

新的项目判断：
- `StageB` 不是完全没价值；当前 winner 的可用窗口至少延续到了 `40`，在 PSNR 口径下甚至能延续到 `80`；
- 但它也不值得继续做无边界长跑调参；如果还给 `StageB` 一次机会，也只值得做一轮 bounded 的后段 schedule compare；
- 用户补的 `docs/SPGM_landing_plan_for_part3_BRPO.md` 现在应正式升级成活跃主线文档：主工程方向开始从“继续磨 view gate + 长跑 schedule”转向“让 SPGM 接管 per-Gaussian 更新权限管理”。

---

### P2-H：RGB-only v2 `StageB-120iter` verify 完成

运行根：
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2h_stageB_v2rgbonly_verify120_e1`

新增文档：
- `docs/P2H_stageB_v2rgbonly_verify120_20260416.md`

协议：
1. 完全复用 `P2-G` 的 `RGB-only v2 StageB` 协议；
2. 只把 `stageB_iters` 从 `20` 提到 `120`；
3. 仍从 `P2-F` 的两条 `StageA.5` handoff（ungated / gated_rgb0192）起跑；
4. 继续保持 `xyz-only`、`lambda_real=lambda_pseudo=1.0`、`num_real_views=2`、`num_pseudo_views=4`。

关键结果：
- `after_opt baseline`: `23.94891 / 0.87349 / 0.07878`
- `ungated_120`: `23.91042 / 0.86866 / 0.08485`
- `gated_rgb0192_120`: `23.91463 / 0.86873 / 0.08485`
- `gated - ungated`: `+0.00421 PSNR / +7.77e-05 SSIM / +7.22e-07 LPIPS`

机制层结论：
- gating 这次不是 no-op：`gated_rgb0192` 在 `120iter` 中有 `96/120` iter rejection，累计拒绝 `116` 次 sample eval，持续拒掉 `225 / 260`；
- `grad_keep_ratio_xyz_mean≈0.729`，说明 pseudo-side Gaussian 更新范围仍在被真实裁剪；
- `loss_real_last` 与 pose aggregate 和 ungated 基本同量级，说明 real branch 依旧没有被明显误伤。

新的项目判断：
- 当前主候选仍然是 `RGB-only v2 + gated_rgb0192`，但它还不能视为已经稳定的中预算 `StageB` 主线；
- `P2-H` 已把问题从“还要不要继续做更长一点的 verify”收束成“如何稳住 `StageB` 后段”；
- 因此下一步优先级应切到 `StageB stabilization`（回落窗口定位 + 后段 `lr / lambda_real:lambda_pseudo` 调度），而不是先做 raw RGB densify 或 support/depth expand。

---


### P2-K：SPGM Phase 0 plumbing 完成

新增产物：
- 、、、 骨架模块
-  扩展 SPGM 字段和方法
-  扩展 SPGM summary 字段
-  新增 SPGM CLI、三路分支逻辑、history 字段

备份文件：
- 
- 
- 
- 

新增 CLI 参数：
-  choices 扩展为 
- 
- 
- 
- 
- 
- 
- 
- 
- 

新增 SPGM history 字段：
- 
- 
- 
- 
- 
- 
- 
- 

当前 wiring 状态：
-  已通过
-  CLI 已正确显示 SPGM 参数
- 三路分支逻辑（visibility_union / spgm / off）已接入
- Phase 0 placeholder 实现返回零值/占位数据

下一步（Phase 1）：
- 实现 ：collect_spgm_stats 的真实统计量提取
- 验证 StageA.5 smoke 能跑通 SPGM mode


## 2026-04-15

### Fusion：BRPO-style `target ↔ reference overlap confidence` 第一版落地

代码变更：
- `pseudo_branch/pseudo_fusion.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`

本轮实现：
1. `pseudo_fusion.py` 不再以旧的 `branch confidence × render-depth gate × global view scalar` 为主语义；
2. 新版 fusion 会基于：
   - pseudo render depth
   - pseudo / left-ref / right-ref camera states
   - left / right reference rendered depth
   计算每一侧的 `target ↔ reference overlap confidence`；
3. 左右分支权重现在由 `overlap_conf_left/right` 直接归一化得到，再对 DiFix 候选图做 residual fusion；
4. `stage_fusion()` 现已真正喂入几何输入，而不是旧的 RGB-only / uniform-confidence placeholder 调用。

新增导出：
- `fusion_weight_left.npy`
- `fusion_weight_right.npy`
- `overlap_conf_left.npy`
- `overlap_conf_right.npy`
- `overlap_mask_left.npy`
- `overlap_mask_right.npy`
- `projected_depth_left.npy`
- `projected_depth_right.npy`
- `ref_depth_left_render.npy`
- `ref_depth_right_render.npy`
- 对应 `diag/` 可视化图

兼容性处理：
- 若调用方没有提供 pseudo/ref states 或 reference depth，新版 `pseudo_fusion.py` 仍保留一个 legacy fallback，避免旧脚本直接崩掉；
- 但 `prepare_stage1_difix_dataset_s3po_internal.py::stage_fusion()` 主路径已切到新的几何版 fusion。

验证：
- 已通过远端 `py_compile`：
  - `pseudo_branch/pseudo_fusion.py`
  - `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
- 已完成 `stage_fusion()` 直连 smoke：
  - 输出目录：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_stage_fusion_direct_smoke/`
- 已完成一次带真实 DiFix 左右图的 direct fusion smoke：
  - 输出目录：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_fusion_direct_difix_50/`

当前判断：
- 这一步已经和 `docs/BRPO_fusion_mask_spgm_subset_refine.md` 中关于 fusion 的主判断对齐：
  - fused image 的权重定义已从“left ↔ right agreement 的辅助语义”切回“target ↔ reference overlap confidence”主语义；
- 但当前只完成了 BRPO 口径中的 fusion 部分；
- 最终训练 mask 仍不是 `post-fusion RGB / correspondence` 的独立推理结果；
- 因此接下来的主工作不应回到旧 `seed_support -> train_mask -> target_depth` 路径补丁，而应继续完成：
  1. fused RGB mask v2
  2. depth supervision v2
  3. local Gaussian gating / subset refine

### Signal v2：fused RGB mask + isolated depth supervision 第一版落地

代码变更：
- `pseudo_branch/brpo_v2_signal/__init__.py`
- `pseudo_branch/brpo_v2_signal/rgb_mask_inference.py`
- `pseudo_branch/brpo_v2_signal/depth_supervision_v2.py`
- `scripts/build_brpo_v2_signal_from_internal_cache.py`

本轮实现：
1. 新增 `signal_v2` 独立路径，不覆盖旧 `brpo_confidence_mask.py / brpo_train_mask.py / brpo_depth_target.py` 主线；
2. `rgb_mask_inference.py` 现在会直接在 fused RGB 上，与 left/right reference RGB 做 reciprocal correspondence 匹配；
3. 新版 `raw_rgb_confidence_v2` / `raw_rgb_confidence_cont_v2` 来自 fused RGB ↔ reference RGB correspondence，本轮不依赖旧 depth seed / train-mask；
4. `depth_supervision_v2.py` 会基于：
   - `projected_depth_left/right`
   - `fusion_weight_left/right`
   - `raw_rgb_confidence_v2`
   单独生成：
   - `target_depth_for_refine_v2_brpo.npy`
   - `target_depth_source_map_v2_brpo.npy`
   - `depth_supervision_mask_v2_brpo.npy`
5. 以上新产物全部写到 `signal_v2/` 下，不改写 legacy `pseudo_cache/` 样本中的旧文件名。

新增导出：
- `signal_v2/frame_<id>/raw_rgb_confidence_v2.npy`
- `signal_v2/frame_<id>/raw_rgb_confidence_cont_v2.npy`
- `signal_v2/frame_<id>/rgb_support_{left,right,both,single}_v2.npy`
- `signal_v2/frame_<id>/target_depth_for_refine_v2_brpo.npy`
- `signal_v2/frame_<id>/target_depth_source_map_v2_brpo.npy`
- `signal_v2/frame_<id>/depth_supervision_mask_v2_brpo.npy`
- `signal_v2/frame_<id>/rgb_mask_meta_v2.json`
- `signal_v2/frame_<id>/depth_meta_v2_brpo.json`
- 对应 `diag/` 可视化图与 `summary.json`

验证：
- 已通过远端 `py_compile`：
  - `pseudo_branch/brpo_v2_signal/__init__.py`
  - `pseudo_branch/brpo_v2_signal/rgb_mask_inference.py`
  - `pseudo_branch/brpo_v2_signal/depth_supervision_v2.py`
  - `scripts/build_brpo_v2_signal_from_internal_cache.py`
- 已完成 1-frame smoke：
  - prepare root：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_stage_fusion_direct_smoke/`
  - 输出目录：`.../signal_v2/`
- frame 50 的 smoke 结果：
  - `raw_rgb_confidence_nonzero_ratio ≈ 0.01865`
  - `support_ratio_both ≈ 0.00904`
  - `depth verified_ratio ≈ 0.01865`
  - `mean_abs_rel_correction_verified ≈ 0.03556`

当前判断：
- 这一步和分析文档的 `mask` / `depth` 语义分工是一致的：
  - mask 主语义已回到 fused RGB / correspondence；
  - depth supervision 已独立成服务 refine 的另一条线；
- 当时这个版本还没有接入 `run_pseudo_refinement_v2.py`；
- 同日后续更新见下一小节：refine consumer 已在 `signal_pipeline=brpo_v2` 路径下完成最小接入并跑通 3-frame smoke。

### Signal v2：3-frame coverage eval + refine consumer smoke 接通

代码变更：
- `scripts/run_pseudo_refinement_v2.py`

本轮接入：
1. 新增 `--signal_pipeline {legacy, brpo_v2}`
2. 新增 `--signal_v2_root`
3. RGB mask 已支持 `brpo_v2_raw / brpo_v2_cont`
4. depth mask 已支持 `brpo_v2_depth`
5. depth target 已支持 `brpo_v2 / target_depth_for_refine_v2_brpo`

3-frame coverage eval：
- 目录：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_signal_v2_coverage_tmp/signal_v2/`
- frame `10 / 50 / 120` 平均：
  - `rgb support left ≈ 1.56%`
  - `rgb support right ≈ 1.20%`
  - `rgb support both ≈ 0.93%`
  - `raw_rgb_confidence_nonzero_ratio ≈ 1.82%`
  - `depth verified_ratio ≈ 1.82%`
  - `both_weighted_ratio ≈ 1.40%`
  - `left_only_ratio ≈ 0.42%`
  - `render_fallback_ratio = 0`
  - `mean_abs_rel_correction_verified ≈ 3.15%`
- 解释：
  1. v2 mask 现在确实是一个更“窄但干净”的 fused-RGB correspondence mask；
  2. depth supervision 已不再靠旧 `train_mask` 扩覆盖，而是严格跟着新的 RGB 可信区域和 projected depth 走。

StageA refine smoke：
- 输出目录：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_signal_v2_refine_smoke`
- 关键结果：
  - `signal_pipeline = brpo_v2`
  - first sample RGB mask mode = `brpo_v2_raw`
  - first sample depth mask mode = `brpo_v2_depth`
  - first sample depth kind = `target_depth_for_refine_v2_brpo`
  - `mean_confidence_nonzero_ratio ≈ 1.82%`
  - `mean_target_depth_verified_ratio ≈ 1.82%`
  - StageA `3 iter` 成功跑完，并产出：
    - `stageA_history.json / refinement_history.json`
    - `pseudo_camera_states_{stageA,final}.json`
    - `refined_gaussians.ply`

当前判断：
- 当前状态已经不是“文件存在但没接上”，而是 `fusion + mask v2 + depth supervision v2 + refine consumer` 四段都已打通；
- 但这还只是 wiring / smoke 成功，不等于已经证明这版窄 coverage 优于 legacy；
- 下一步应该直接做一轮 apples-to-apples short compare：`legacy vs signal_v2`，使用同一组 pseudo、同一组 iter、同一组 refine 超参；
- local Gaussian gating 仍是后续必须补的结构项，不应因为 smoke 通过就后移得过远。

### P0：legacy E1 root 上的 split abs prior 标定 + compare 汇总脚本补齐

代码/脚本变更：
- `scripts/summarize_stageA_compare.py`
- `scripts/run_p0_absprior.sh`
- `docs/P0_absprior_and_P1A_stageA_signal_compare_20260415.md`

本轮执行：
1. 在 `20260414_signal_enhancement_e15_compare/pseudo_cache_baseline` 上，以 `legacy + target_depth_for_refine_v2 + source_aware` 口径完成 6 组 `StageA-80iter` split abs prior 标定；
2. 同时补齐统一汇总脚本 `summarize_stageA_compare.py`，直接读取 `stageA_history.json` / `replay_eval.json` 并输出 `summary.json + compare_table.csv + compare_table.md`；
3. 当前固定背景配置收敛为：`lambda_abs_t=3.0`、`lambda_abs_r=0.1`。

关键结论：
- noabs 的 drift 明显更大；
- `(3.0, 0.1)` 已进入有效区间；
- 继续把 `lambda_abs_t` 从 `3.0` 拉到 `6.0` 没有换来足够明确的额外收益；
- 因此后续 compare 默认先固定在 `(3.0, 0.1)`，不再继续大范围粗扫。

### P0：确认纯 `StageA` 不更新 Gaussian，StageA-only replay 只是 identity sanity check

本轮验证：
- 已直接比对：
  - `BASE_PLY`
  - `stageA_noabs_80/refined_gaussians.ply`
  - `stageA_abs_t3_r0p1_80/refined_gaussians.ply`
- 三者 `sha256sum` 完全一致。

因此当前必须修正的设计/实验口径是：
- 纯 `stage_mode=stageA` 只更新 pseudo camera / exposure，不更新 Gaussian；
- 所以 replay-on-PLY 在 `StageA-only` compare 中不是判优指标；
- 后续真正依赖 replay 的 compare，必须移到 `StageA.5 / StageB` 或其他会真实更新 Gaussian 的阶段。

### P1A：canonical E1 root 补齐 8-frame `signal_v2`，并完成 `legacy / v2-rgb-only / v2-full` 的 StageA-only compare

执行前置：
- 由于 canonical E1 root 之前并没有完整保留新版 fusion 输出，先重新补齐：
  - `prepare_stage1_difix_dataset_s3po_internal.py --stage fusion`
- 同时为历史 root 构造了平铺的 DiFix symlink 根：
  - `_flat_difix_from_pseudo_cache_baseline/{left_fixed,right_fixed}`
- 然后在 canonical E1 root 上成功生成：
  - `.../20260414_signal_enhancement_e15_compare/signal_v2`

执行脚本：
- `scripts/run_p1_stageA_signal_compare.sh`

本轮 compare 固定：
- pseudo set = `E1 signal-aware-8`
- abs prior = `(3.0, 0.1)`
- `stageA_iters=80`

三臂：
1. `legacy`
2. `v2-rgb-only`
3. `v2-full`

关键结论：
- `v2-rgb-only` 不是明显坏方向；
- `full v2 depth` 当前明显过窄：`mean_target_depth_verified_ratio` 从 legacy 的约 `4.13%` 收到约 `1.96%`，`loss_depth_dense` 也几乎掉空；
- 因此当前若保留 v2，更值得保留的是 `RGB-only v2` 作为 signal probe，而不是直接让 `full v2 depth` 接管默认训练分支。

### 文档同步

本轮同时更新：
- `docs/STATUS.md`
- `docs/DESIGN.md`
- `docs/CHANGELOG.md`
- `docs/hermes.md`

并把当前下一步明确收敛到：
- `P2 local Gaussian gating`（优先 `StageA.5 + xyz-only + hard gating + pseudo-side only`）

### P2-G：signal-aware-8 固定下的 RGB-only v2 StageB real-branch short compare

本轮无代码改动，重点是验证：`RGB-only v2 + gated_rgb0192` 在 joint refine（带 real branch）里能不能站住。

运行位置：
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2g_stageB_v2rgbonly_realbranch_compare_e1`

新增运行：
- `stageB_from_stageA5_v2rgbonly_ungated_20`
- `stageB_from_stageA5_v2rgbonly_gated_rgb0192_20`

协议要点：
- fixed signal-aware-8 pseudo set，不扩 pseudo 数量
- `signal_pipeline=brpo_v2`
- `stageA_rgb_mask_mode=brpo_v2_raw`
- `stage_mode=stageB`
- `stageB_iters=20`
- `num_real_views=2`
- `lambda_real=lambda_pseudo=1.0`
- 两臂分别从各自的 `StageA.5` 自然分支输出进入 StageB

关键结果：
- `v2 StageB ungated` replay：`PSNR 23.994534 / SSIM 0.872871 / LPIPS 0.080385`
- `v2 StageB gated_rgb0192` replay：`PSNR 24.010606 / SSIM 0.873018 / LPIPS 0.080363`
- gated 相比 ungated：`+0.0161 PSNR / +0.000147 SSIM / -2.16e-05 LPIPS`
- gated 臂真实 rejection：`18/20` iter，`22/80` sample eval，被拒 id 主要是 `225 / 260`，原因全是 `rgb_mask_ratio`
- `loss_real_last` 与 ungated 几乎一致，说明 real branch 未被明显误伤
- 更重要的是：`RGB-only v2` 的 StageB 表现仍明显优于 legacy StageB

工程判断更新：
- `RGB-only v2 + gated_rgb0192` 已经从 A.5 候选升级成当前主候选 refine 分支；
- 下一步不该先去 densify raw RGB mask，而应该先做更长一点的 `StageB` / 完整 schedule verify；
- 若后续仍要扩张，优先考虑受几何约束的 support/depth expand。

新增报告：
- `docs/P2G_stageB_v2rgbonly_realbranch_compare_20260416.md`

### 文档同步

本轮同时更新：
- `docs/STATUS.md`
- `docs/DESIGN.md`
- `docs/CHANGELOG.md`
- `docs/hermes.md`

并把当前下一步明确收敛到：
- 以 `RGB-only v2 + gated_rgb0192` 为主候选做更长一点的验证；
- 暂不直接做 raw RGB densify。

### P2-F：RGB-only v2 StageA.5 ungated vs branch-specific gated compare

本轮无代码改动，重点是把 gating 主线从 legacy 转到 `RGB-only v2`，并按用户要求把新实验输出写到 `/data2`。

运行位置：
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1`

新增运行：
- `stageA5_v2rgbonly_xyz_ungated_80`
- `stageA5_v2rgbonly_xyz_gated_rgb0192_80`

协议要点：
- `signal_pipeline=brpo_v2`
- `stageA_rgb_mask_mode=brpo_v2_raw`
- `stageA_depth_mask_mode=train_mask`
- `stageA_target_depth_mode=target_depth_for_refine_v2`
- `StageA.5 + xyz-only`
- gated 臂使用 `min_rgb_mask_ratio=0.0192`，目的不是大扫，而是稳定拒掉最弱两档 RGB-only pseudo（`225 / 260`）

关键结果：
- `RGB-only v2 ungated` replay：`PSNR 23.923221 / SSIM 0.872280 / LPIPS 0.079949`
- `RGB-only v2 gated_rgb0192` replay：`PSNR 23.924801 / SSIM 0.872320 / LPIPS 0.079931`
- gated 相比 ungated：`+0.00158 PSNR / +3.92e-05 SSIM / -1.76e-05 LPIPS`
- gated 臂产生真实 rejection：`59/80` iter，`74/320` sample eval，被拒 id 为 `225 / 260`，原因全是 `rgb_mask_ratio`
- 更重要的是：`RGB-only v2` 整体在 `StageA.5` 上明显优于 legacy `StageA.5`

工程判断更新：
- gating 主线不必再停留在 legacy；`RGB-only v2` 已经是更值得继续推进的支路；
- raw RGB support 图虽然看起来点状，但当前证据还不支持“必须立刻做 RGB densify”；
- 若后续需要扩张，更合理的对象是受几何约束的 support/depth expand，而不是直接把 `raw_rgb_confidence_v2` 糊开。

新增报告：
- `docs/P2F_stageA5_v2rgbonly_gating_compare_20260416.md`

### 文档同步

本轮同时更新：
- `docs/STATUS.md`
- `docs/DESIGN.md`
- `docs/CHANGELOG.md`
- `docs/hermes.md`

并把当前下一步明确收敛到：
- 若继续推进 gating，围绕 `RGB-only v2` 做下一轮 branch-specific compare / StageB 验证；
- 暂不直接做 raw RGB densify。

### P2-E：legacy StageA.5 threshold calibration（`min_verified_ratio=0.02 / 0.03`）

本轮无代码改动，重点是把 `P2-D` 的阈值诊断真正落实成 replay-grounded short compare。

运行说明：
- 由于 canonical `output/` 实际落在 `/data`，而本轮执行时 `/data` 已满，新实验目录临时写到：
  - `/home/bzhang512/tmp_part3_brpo_outputs/20260416_p2e_stageA5_threshold_calibration_e1`
- 输入协议与旧 `P2-B` 保持一致，只改 `pseudo_local_gating_min_verified_ratio`。

新增运行：
- `stageA5_legacy_xyz_gatevr002_80`
- `stageA5_legacy_xyz_gatevr003_80`

关键结果：
- `vr=0.02`：开始稳定拒掉最弱 pseudo（`sample_id=260`），`37/80` iter 出现 rejection，`37/320` sample eval 被拒；
- `vr=0.03`：进一步稳定拒掉 `225 + 260`，`59/80` iter 出现 rejection，`74/320` sample eval 被拒；
- 所有 rejection 都只来自 `verified_ratio`，而不是 `rgb_mask_ratio` / `fallback_ratio`；
- 但 replay 仍几乎不动：
  - `vr=0.02`: `PSNR 23.852479 / SSIM 0.870652 / LPIPS 0.080929`
  - `vr=0.03`: `PSNR 23.852239 / SSIM 0.870651 / LPIPS 0.080930`
- 因而本轮真正得到的工程判断是：legacy `StageA.5` 上的 threshold softness 已不是主瓶颈；即便 gate 进入有效 reject 区，继续细磨 legacy threshold 的收益也很有限。

新增报告：
- `docs/P2E_stageA5_threshold_calibration_20260416.md`

### 文档同步

本轮同时更新：
- `docs/STATUS.md`
- `docs/DESIGN.md`
- `docs/CHANGELOG.md`

并把当前下一步明确收敛到：
- 若只需要 calibrated legacy 参考臂，优先保留 `vr=0.02`；
- 若继续推进 gating，更值得转到 `RGB-only v2` 做 branch-specific threshold calibration / short compare。

### P2-D：signal gate 阈值 / 统计诊断

本轮无代码改动，重点是把 `P2-B / P2-C` 里持续 `0 rejection` 的原因做成 code-grounded 诊断。

检查对象：
- `pseudo_branch/local_gating/gating_schema.py`
- `pseudo_branch/local_gating/signal_gate.py`
- `scripts/run_pseudo_refinement_v2.py`
- `20260415_p2b_stageA5_local_gating_compare_e1/.../stageA_history.json`
- `20260415_p2c_stageB_local_gating_compare_e1/.../stageB_history.json`
- `20260415_signal_compare_stageAonly_e1/{legacy,v2-rgb-only,v2-full}`

关键结论：
- current gate 默认阈值 `min_verified=0.01 / min_rgb=0.01 / max_fallback=0.995` 对 legacy sampled view 来说过松，因此 `StageA.5 / StageB` 上的 `0 rejection` 是结构必然，不是实现没生效；
- legacy sampled view 的实际范围约为：`verified 1.89%~5.17%`、`rgb 14.68%~22.05%`、`fallback 94.83%~98.11%`；
- 如果想让 legacy gate 真正开始筛 weak pseudo，第一轮更值得扫的是 `min_verified_ratio`，推荐先看 `0.02 / 0.03`；
- `RGB-only v2` 的 `rgb_confidence_nonzero_ratio` 只有约 `1.90%~2.01%`，与 legacy 的 `~18.9%` 不在同一量纲；因此在没做 branch-specific calibration 之前，不应直接把 current-threshold gating 挂到 `RGB-only v2` 上做可解释 compare。

新增报告：
- `docs/P2D_signal_gate_threshold_diagnostics_20260416.md`

### 文档同步

本轮同时更新：
- `docs/STATUS.md`
- `docs/DESIGN.md`
- `docs/CHANGELOG.md`

并把当前下一步明确收敛到：
- 先做 legacy 口径下的 signal gate threshold calibration；
- 再决定是否把 gating 挂到 `RGB-only v2` 上做 short compare。

### P2：local Gaussian gating 第一版实现 + smoke

代码变更：
- `scripts/run_pseudo_refinement_v2.py`
- `pseudo_branch/local_gating/__init__.py`
- `pseudo_branch/local_gating/gating_schema.py`
- `pseudo_branch/local_gating/signal_gate.py`
- `pseudo_branch/local_gating/visibility_union.py`
- `pseudo_branch/local_gating/grad_mask.py`
- `pseudo_branch/local_gating/gating_io.py`

本轮实现：
1. 在 `run_pseudo_refinement_v2.py` 中新增 `pseudo_local_gating_*` CLI；
2. 在 `StageA.5` 上接入：`pseudo-only backward -> local gate mask -> gaussian step`；
3. 在 `StageB` 上改成 split backward：先 `pseudo_backward(retain_graph=has_real_branch)`，只裁 pseudo-side Gaussian grad，再叠 `real_backward()`；
4. gating summary 现在会写进 `stageA_history.json / stageB_history.json`，包括 accepted/rejected pseudo ids、rejected reasons、visible union、grad keep ratio、pre/post grad norm；
5. 第一版模块已拆到 `pseudo_branch/local_gating/`，避免把 signal gate / visibility union / grad mask 全塞在主脚本里。

验证：
- 远端 `py_compile` 已通过：
  - `scripts/run_pseudo_refinement_v2.py`
  - `pseudo_branch/local_gating/*.py`
- 已完成 `StageA.5` smoke：
  - off：`.../20260415_local_gating_smoke/stageA5_legacy_off_2iter`
  - hard：`.../20260415_local_gating_smoke/stageA5_legacy_hard_2iter`
- 已完成严格阈值 reject harness：
  - `.../20260415_local_gating_smoke/stageA5_legacy_hard_reject_finalcheck`
  - 实测 `xyz_grad` 可从 `3.79e-01 -> 0.0`
- 已完成 `StageB` no-real smoke：
  - `.../20260415_local_gating_smoke/stageB_legacy_hard_noreal_1x1`
  - 实测 StageB split-backward + pseudo-side grad mask 路径可执行，`xyz_grad` 可从 `6.38e-02 -> 0.0`

结构判断更新：
- 第一版 local gating 已经从计划项变成真实代码路径；
- 在 `StageA.5 pseudo-only` 下，如果 sampled pseudo views 全部通过 signal gate，那么 `visibility_filter union` 往往不会额外收缩 xyz grad norm；
- 因而第一版 gating 的主要价值，体现在“拒绝 weak pseudo views 时阻断它们对 Gaussian 的更新”，而不是在全部 sampled views 都通过时再额外制造一层局部性。

新增报告：
- `docs/P2_local_gaussian_gating_first_impl_smoke_20260415.md`

### 文档同步（P2 后）

本轮在 P2 smoke 后再次同步：
- `docs/STATUS.md`
- `docs/DESIGN.md`
- `docs/CHANGELOG.md`

并把当前下一步更新为：
- 8-frame `StageA.5` gated vs ungated short compare
- 通过后再补 real-branch `StageB` smoke / short compare

### P2-B：8-frame StageA.5 gated vs ungated short compare

新增报告：
- `docs/P2B_stageA5_local_gating_compare_20260415.md`

运行根：
- `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_p2b_stageA5_local_gating_compare_e1`

固定口径：
- `signal_pipeline=legacy`
- `stage_mode=stageA5`
- `stageA_iters=80`
- `num_pseudo_views=4`
- `stageA5_trainable_params=xyz`
- abs prior 固定为 `lambda_abs_t=3.0`、`lambda_abs_r=0.1`
- init states 来自：`20260415_signal_compare_stageAonly_e1/stageA_legacy_80/pseudo_camera_states_stageA.json`

对照臂：
1. `stageA5_legacy_xyz_ungated_80`
2. `stageA5_legacy_xyz_gated_80`

关键结果：
- gated replay 相比 ungated 只有极轻微优势：
  - PSNR `+0.000439`
  - SSIM `+0.000001`
  - LPIPS `-0.000001`
- gated 没有把 depth loss 压没：`loss_depth_last` 约 `0.05947`，与 ungated 的 `0.05948` 基本一致；
- gated 的 `grad_keep_ratio_xyz` 最后一轮约 `0.8302`，80 iter 平均约 `0.7247`；
- 但这轮 compare 中 `iters_with_rejection = 0 / 80`，说明所有 sampled pseudo views 都通过了 signal gate。

结构判断更新：
- 第一版 gating 在当前 fixed threshold 下是“基本无害，但区分度很弱”；
- 当前 replay 改善极弱的核心原因，不是 wiring 没生效，而是 signal gate 没有真的拒掉 weak pseudo view；
- 因此下一步最该验证的已经变成：real branch 打开后，pseudo-side gating 会不会误伤 global correction。

### 文档同步（P2-B 后）

本轮在 StageA.5 compare 后再次同步：
- `docs/STATUS.md`
- `docs/DESIGN.md`
- `docs/CHANGELOG.md`

并把当前下一步更新为：
- 带 real branch 的 `StageB` gated vs ungated short compare

### P2-C：real-branch StageB gated vs ungated short compare

新增报告：
- `docs/P2C_stageB_local_gating_realbranch_compare_20260415.md`

运行根：
- `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_p2c_stageB_local_gating_compare_e1`

固定口径：
- 两臂都从同一个 `StageA.5 gated` 输出起跑：
  - `refined_gaussians.ply`
  - `pseudo_camera_states_final.json`
- `stage_mode=stageB`
- `stageA_iters=0`
- `stageB_iters=20`
- `num_pseudo_views=4`
- `num_real_views=2`
- `lambda_real=1.0`
- `lambda_pseudo=1.0`
- real branch:
  - `train_manifest=/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/split_manifest.json`
  - `train_rgb_dir=/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/rgb`

对照臂：
1. `stageB_from_stageA5gated_ungated_20`
2. `stageB_from_stageA5gated_gated_20`

关键结果：
- gated 不会明显误伤 real branch：`loss_real_last` 与 ungated 几乎完全一致（约 `0.14798`）
- gated replay 相比 ungated 甚至略差一点，但量级几乎可忽略：
  - PSNR `-0.000045`
  - SSIM `-0.00000056`
  - LPIPS `+0.00000057`
- 这轮 compare 中同样 `iters_with_rejection = 0 / 20`

结构判断更新：
- 当前最关键的问题已经不再是“real branch 会不会被误伤”；
- 更真实的问题是：current threshold 下 signal gate 在 StageA.5 / StageB 都持续 `0 rejection`，所以第一版 gating 大部分时间只是在做 visible union，而没有真正筛掉 weak pseudo view。

### 文档同步（P2-C 后）

本轮在 StageB compare 后再次同步：
- `docs/STATUS.md`
- `docs/DESIGN.md`
- `docs/CHANGELOG.md`

并把当前下一步更新为：
- 解释 current threshold 下为什么持续 `0 rejection`
- 再决定是先调 signal gate 阈值，还是先把 gating 挂到 `RGB-only v2`

## 2026-04-13

### A+B 实施：split+scaled abs prior + 梯度贡献诊断

代码变更：
- `pseudo_branch/pseudo_loss_v2.py`
- `pseudo_branch/pseudo_refine_scheduler.py`
- `pseudo_branch/pseudo_camera_state.py`
- `scripts/run_pseudo_refinement_v2.py`
- `scripts/diagnose_stageA_loss_contrib.py`（新增）

关键实现：
- Stage A abs prior 从 legacy 单标量扩展为 split+scaled 形式：
  - `--stageA_lambda_abs_t`
  - `--stageA_lambda_abs_r`
  - `--stageA_abs_pose_robust {charbonnier,huber,l2}`
  - `--stageA_abs_pose_scale_source {...}`
  - `--stageA_abs_pose_fixed_scale`
- history 新增：
  - `loss_abs_pose_trans/rot`
  - `abs_pose_rho_norm/theta_norm`
  - `scene_scale_used`
  - `grad_contrib`
- 新增在线梯度日志：
  - `--stageA_log_grad_contrib`
  - `--stageA_log_grad_interval`

### A 小网格：6组 × 80 iter

扫描网格：
- `lambda_t ∈ {0.3, 1.0, 3.0}`
- `lambda_r ∈ {0.03, 0.1}`

输出目录：
- `/home/bzhang512/CV_Project/output/part3_stage1_internal/re10k-1/full/2026-04-13_ab_small_grid/`

结果摘要：
- `loss_depth_last` 约在 `0.06837 ~ 0.06854`，组间差异小；
- `abs_pose_rho_norm_last` 从 `~7.93e-4`（弱约束）降到 `~1.83e-4`（强约束）；
- 结论：abs prior 对 drift 抑制明确有效，但对 depth 改善不显著。

### top3 深跑：3组 × 300 iter

输入：
- 基于小网格按“drift优先且兼顾depth”筛出的 top3。

输出目录：
- `/home/bzhang512/CV_Project/output/part3_stage1_internal/re10k-1/full/2026-04-13_ab_deep300_top3/`

结果摘要：
- `lt3.0_lr0.1`：drift 最小（`rho≈2.19e-4, theta≈1.17e-4`），depth 不优；
- `lt1.0_lr0.1`：depth 末值最低（`~0.06837`），但 drift 明显更大（`rho≈4.67e-4`）；
- 结论延续：目前没有出现“显著压 drift 且显著改善 depth”的统一最优组。


### A.5 midpoint 执行（xyz / xyz+opacity）

实现与数据：
- 新增 `pseudo_branch/gaussian_param_groups.py`（micro 参数组：`xyz` / `xyz+opacity`）；
- `pseudo_branch/pseudo_refine_scheduler.py` 增加 `StageA5Config` 与 `build_stageA5_optimizers()`；
- `scripts/run_pseudo_refinement_v2.py` 增加 `stageA5` 模式及 CLI：
  - `--stage_mode {stageA,stageA5}`
  - `--stageA5_trainable_params {xyz,xyz_opacity}`
  - `--stageA5_lr_xyz --stageA5_lr_opacity`
- midpoint pseudo cache：`re10k1__internal_afteropt__midpoint_proto_v1`，样本帧 `17/51/86/121/156/190/225/260`。

实验（80 iter，num_pseudo_views=8，fused legacy mask）：
- 输出目录：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_stageA5_midpoint/`
- 汇总：`summary_stageA5_midpoint_80iter.json`
- 结果（last10 mean）：
  - StageA baseline：`loss_total=0.005028, loss_depth=0.007047`
  - StageA5(xyz)：`0.005191, 0.007429`（劣于 baseline）
  - StageA5(xyz+opacity)：`0.004416, 0.006119`（优于 baseline）
  - StageA5(xyz)+abs(1.0/0.1)：`0.006291, 0.007335`（劣于 baseline）

过程问题：
- `prepare_stage1_difix_dataset_s3po_internal.py --stage verify` 在当前环境触发 MASt3R 依赖报错：`NameError: load_model_as_safetensor`；
- 因此本轮 A.5 使用 fused legacy mask 路径先完成机制对照，BRPO verify 依赖修复单列后续处理。

### 与 2026-04-12（mask 刚修好、depth 问题暴露时）的设计差异

1. 当时主问题：确认 depth 信号是否接通、闭环是否成立；
2. 现在主问题：在闭环已成立条件下，做 drift-depth 可解释权衡；
3. 当时重点在 upstream coverage 与链路真伪；现在重点在 Stage A 参数化与可解释诊断；
4. 不进 Stage B 的理由从“链路未充分可信”转为“tradeoff 未固化”。


### depth-heavy top3：3组 × 300 iter（Step-1）

实验设置：
- 复用 top3 参数组：`(3.0,0.1) / (3.0,0.03) / (1.0,0.1)`
- `stageA_beta_rgb=0.3`（depth-heavy）
- 其余同 A+B 主口径，启用 grad contrib 记录。

输出目录：
- `/home/bzhang512/CV_Project/output/part3_stage1_internal/re10k-1/full/2026-04-13_ab_deep300_top3_depthheavy/`

结果：
- best-drift：`lt3.0_lr0.1`（`rho≈5.19e-4, theta≈2.40e-4`）
- best-depth：`lt1.0_lr0.1`（`loss_depth≈0.06814`）
- 结论：tradeoff 在 depth-heavy 下未消失，仍是“drift组”和“depth组”分离。

### Step-2 梯度贡献汇总

汇总文件：
- `.../2026-04-13_ab_deep300_top3_depthheavy/grad_contrib_summary.json`
- `.../2026-04-13_ab_deep300_top3_depthheavy/grad_contrib_summary_cross.json`

核心结论：
- `rot` 与 `trans` 的主导更新项主要来自 `rgb` 与 `depth_total`（二者交替主导）；
- `abs_pose_trans/rot` 在均值上显著小于数据项，当前更像稳定边界，不是主驱动项。

### Step-3 口径固化

固化两套 Stage A 临时默认：
1. drift-prioritized：`beta_rgb=0.7, lambda_abs_t=3.0, lambda_abs_r=0.1`
2. depth-prioritized：`beta_rgb=0.3, lambda_abs_t=1.0, lambda_abs_r=0.1`

说明：这两套用于可解释对照，不代表已得到统一最优。


### 下游价值验证（StageA 三口径）

实验设置：
- 组A：`noabs_beta0.7_300`
- 组B：`drift_prio_beta0.7`（`lt3.0_lr0.1`）
- 组C：`depth_prio_beta0.3`（`lt1.0_lr0.1`）
- 在 internal cache `after_opt` 的 270 帧 non-KF replay 上比较。

输出目录：
- `/home/bzhang512/CV_Project/output/part3_stage1_internal/re10k-1/full/2026-04-13_value_eval/`
- 汇总：`value_eval_report.json`

结果：
- noabs: `PSNR 23.948825 / SSIM 0.8734876 / LPIPS 0.0787778`
- drift-prio: `23.948666 / 0.8734792 / 0.0787801`
- depth-prio: `23.948893 / 0.8734833 / 0.0787813`
- 相对 baseline (`23.948909 / 0.8734854 / 0.0787798`) 的差异在 `1e-4~1e-6` 量级。

结论：
- 当前 Stage A 口径虽然改变了 drift 行为，但未转化为可见的下游 replay 提升；
- 主要原因是当前仅更新 3 个 pseudo frame pose，对 270 帧总体指标影响过小。

---

## 2026-04-12

### Mask problem route：M1 fused-first verification + compatibility layer

根据 `CURRENT_MASK_PROBLEM.md` 与 `SOLVE_MASK_PROBLEM.md` 的判断，开始把主线从“继续扩 Stage A”转到“先解决 upstream mask problem”。

这一轮的重点不是 refine，而是 verification / pack / consumer 之间的 upstream 结构修正。

代码修改：
- `pseudo_branch/brpo_confidence_mask.py`
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
- `docs/SOLVE_MASK_PROBLEM.md`

关键实现：
- `verification_mode=branch_first|fused_first` 双模式落地；
- `fused_first` 模式可直接消费 `fusion/samples/<id>/target_rgb_fused.png`；
- verification 输出新增 `seed_support_left/right/both/single`；
- 同时保留 `support_*` 与 `confidence_mask_brpo_*` 兼容层，不污染旧实验与现有 consumer；
- `pack` 已能在 `verification_mode=fused_first` 下把 `seed_support_*` 带入 sample。

真实检查：
- `fused_first` 已在当前 3-frame prototype 上跑通；
- 但与旧 `branch_first` 相比，coverage 改善并不明显；
- 这验证了：**单纯改 verification 顺序还不够，M2 的 propagation 仍然必要。**

### Mask problem route：M2 train mask propagation 第一轮实现

在 M1 之后，开始做 `seed_support → train_confidence_mask` 的第一轮工程化传播。

新增文件：
- `pseudo_branch/brpo_train_mask.py`

关键实现：
- verify 阶段新增 `train_mask_mode=propagate`；
- 新增：
  - `train_confidence_mask_brpo_{left,right,fused}`
  - `train_support_{left,right,both,single}`
- 当前训练真正消费的 mask 通过 alias 保持兼容：
  - `confidence_mask_brpo_*` → 指向 train mask
  - `support_*` → 指向 seed support
- propagation 仍保留在 verify/pack upstream，不塞回 refine。

中途问题与修复：
- 第一版 M2 因重复落盘（seed/train/alias 全部物理保存）触发 `No space left on device`；
- 随后把 alias 改成 symlink，保留兼容的同时避免重复写大文件；
- 清掉失败的中间 verify 目录后重跑，验证通过。

真实结果：
- 机制成立，coverage 从 `~1%` 量级显著抬升；
- 但默认参数明显过宽：
  - frame 10: `~75.1%`
  - frame 50: `~68.0%`
  - frame 120: `~68.9%`
- 说明 propagation 机制是对的，但默认超参不能直接作为最终 training mask 定义。

### Mask problem route：M2.5 propagation 合理区间小范围研究

为了判断当前 propagation 的“合理区间”，没有继续大规模重跑 verify，而是基于现有 `fused_first + seed_support` 结果做了离线小范围 sweep。

目的：
- 不是寻找唯一最优超参；
- 而是先判断：哪些组合明显太稀、哪些组合明显太稠、哪些组合更像训练可用区间。

代表性结果：
- `radius=1, tau_rel_depth=0.01, tau_rgb_l1=0.03` → `avg_fused_nonzero ≈ 8.2%`
- `radius=1, tau_rel_depth=0.02, tau_rgb_l1=0.03` → `≈ 9.7%`
- `radius=2, tau_rel_depth=0.01, tau_rgb_l1=0.05` → `≈ 19.4%`
- `radius=2, tau_rel_depth=0.02, tau_rgb_l1=0.05` → `≈ 25.1%`
- `radius=3, tau_rel_depth=0.03, tau_rgb_l1=0.08` → `≈ 52.3%`
- `radius=5, tau_rel_depth=0.03, tau_rgb_l1=0.08` → `≈ 70.7%`

阶段判断：
- 当前 propagation 的合理研究区间明显更接近 `10% ~ 25%` coverage；
- 因此后续不应继续使用当前过宽默认值，而应优先围绕：
  - `radius ∈ {1,2}`
  - `tau_rel_depth ∈ [0.01, 0.02]`
  - `tau_rgb_l1 ∈ [0.03, 0.05]`
  做进一步收紧。

### Mask problem route：M3 blended `target_depth_for_refine` 落地

在 M2.5 之后，正式进入 M3：把 upstream depth target 从“render-depth fallback 占位符”升级成可审计的 blended target。

新增文件：
- `pseudo_branch/brpo_depth_target.py`

代码修改：
- `pseudo_branch/brpo_reprojection_verify.py`
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`

关键实现：
- verification 侧新增 pseudo-view sparse verified depth 输出：
  - `projected_depth_left.npy`
  - `projected_depth_right.npy`
  - `projected_depth_valid_left.npy`
  - `projected_depth_valid_right.npy`
- pack 阶段正式构造：
  - `target_depth_for_refine.npy`
  - `target_depth_for_refine_source_map.npy`
  - `target_depth_for_refine_meta.json`
  - `verified_depth_mask.npy`
- `target_depth_for_refine` 当前定义为：
  - `both` → 双边 verified depth average
  - `left/right` → 单边 verified depth
  - unsupported → `render_depth` fallback

真实 smoke：
- 新开 prototype 目录：
  - `.../internal_prepare/re10k1__internal_afteropt__brpo_proto_v4_stage3/`
- 在 3-frame（10/50/120）上重跑 `verify → pack` 成功；
- sample 层已真实出现：
  - `projected_depth_*`
  - `target_depth_for_refine.npy`
  - `target_depth_for_refine_source_map.npy`
  - `target_depth_for_refine_meta.json`

代表性结果：
- frame 10 verified depth ratio：`~1.63%`
- frame 50 verified depth ratio：`~1.51%`
- frame 120 verified depth ratio：`~1.55%`
- 其余区域为 `render_depth` fallback

阶段判断：
- M3 已完成 upstream depth target 的真实落地；
- 当前 `target_depth_for_refine.npy` 已不再只是 schema 预留位；
- 但当前 verified depth 覆盖偏保守，更像 sparse correction source，而不是大覆盖 supervision 主体。

提交：
- `c7965b4` — `part3: implement Stage M3 blended target_depth_for_refine pipeline`

### Mask problem route：M4 Stage A consumer 显式接入新 target

在 M3 之后，继续做 M4：让 `run_pseudo_refinement_v2.py` 明确消费新的 train mask 与 blended depth target，而不是继续依赖含糊 fallback。

代码修改：
- `scripts/run_pseudo_refinement_v2.py`

关键实现：
- Stage A 现在显式支持：
  - `stageA_mask_mode=train_mask|seed_support_only|legacy|auto`
  - `stageA_target_depth_mode=blended_depth|render_depth_only|target_depth_for_refine|target_depth|render_depth|auto`
- history 现在会显式记录：
  - 实际 confidence source / path
  - 实际 depth source / path
  - effective mask/depth mode
  - mask coverage
  - verified depth ratio / render fallback ratio
- 通过 sample 内显式文件读取 `train_confidence_mask_brpo_*` 与 `target_depth_for_refine.npy`，不再让 consumer 靠隐式默认值漂移。

真实 smoke：
- 目录：
  - `.../2026-04-12_m4_stageA_smoke/blended_depth/`
  - `.../2026-04-12_m4_stageA_smoke/render_depth_only/`
- `train_mask + blended_depth`：
  - mean mask coverage `~19.4%`
  - mean verified depth ratio `~1.56%`
  - `loss_depth ≈ 0.00478`（非零）
- `train_mask + render_depth_only`：
  - `loss_depth ≈ 0`
- 额外 debug quick smoke：
  - `seed_support_only + blended_depth` 2 iter 可跑，mean mask coverage `~1.56%`

阶段判断：
- M4 consumer 已真实接通；
- `blended_depth` 与 `render_depth_only` 在 Stage A 中已能被区分；
- 这说明 M3→M4 链路不是“名义接通”，而是 depth signal 已实际进入 loss。

提交：
- `63abe22` — `part3: wire Stage A to explicit M4 mask and depth modes`

### Mask problem route：M4.5 Stage A 长一点对照（blended vs render-only）

在 M4 smoke 之后，继续做一轮更长一点的 Stage A eval，目的不是追最终指标，而是回答：
- depth signal 是否真的被消费；
- 当前 depth signal 在更长一点的 Stage A 中是否会动起来；
- `blended_depth` 与 `render_depth_only` 的差异是否只停留在 smoke 级别。

实验目录：
- `.../2026-04-12_m45_stageA_eval/blended_depth_long/`
- `.../2026-04-12_m45_stageA_eval/render_depth_only_long/`
- `.../2026-04-12_m45_stageA_eval/analysis/summary.json`
- `.../2026-04-12_m45_stageA_eval/analysis/compare.txt`

实验口径：
- `target_side=fused`
- `confidence_mask_source=brpo`
- `stageA_mask_mode=train_mask`
- `num_pseudo_views=3`
- `stageA_iters=300`
- 唯一变量：
  - `stageA_target_depth_mode=blended_depth`
  - vs `stageA_target_depth_mode=render_depth_only`

结果：
- 两组 `loss_rgb` 基本一致：
  - `0.00915 -> 0.00734`
- `blended_depth_long`：
  - `loss_depth ≈ 0.004783`（非零）
  - `loss_total first -> last: 0.00784 -> 0.02269`
- `render_depth_only_long`：
  - `loss_depth ≈ 4.4e-07 ≈ 0`
  - `loss_total first -> last: 0.00641 -> 0.02207`
- 当前 `pose_reg` 会随着迭代持续增长，两组量级接近；
- 当前 `blended_depth` 虽然已非零，但在 300 iter 中基本不下降。

阶段判断：
- `blended_depth` 与 `render_depth_only` 的差异已经真实存在；
- 但当前差异主要体现在“depth loss 是否非零”，而不是“depth 是否明显驱动了更好的优化轨迹”；
- 更准确的结论是：

**depth signal 已接入，但当前 Stage A 对它的利用仍偏弱。**

因此下一步应进入 **M4.6 / depth-flatness diagnosis**，而不是直接跳到 Stage B。


### Mask problem route：M5-0 depth signal diagnosis

在 M4.5 之后，没有直接继续推进 Stage B，而是先做一轮更结构化的 diagnosis，确认当前 depth signal 到底卡在哪里。

新增文件：
- `scripts/analyze_m5_depth_signal.py`

实验目录：
- `.../2026-04-12_m50_m51_eval/analysis/m50_depth_signal.json`

关键结果：
- `train_mask coverage ≈ 19.41%`
- `verified depth coverage ≈ 1.56%`
- `verified_within_train_mask_ratio ≈ 8.05%`
- `render_fallback_within_train_mask_ratio ≈ 91.95%`

解释：
- 当前 depth loss 不是没接上；
- 但在训练真正消费的 mask 区域里，绝大多数像素仍然只是 fallback depth；
- 这说明旧的 M3 target 在 train-mask 内部的“新几何信息占比”确实太低。

阶段判断：
- M5-0 证明旧问题真实存在；
- 下一步值得先尝试 densify correction field，而不是继续堆 M4.5 式对照。

### Mask problem route：M5-1 densify depth correction field

根据 M5 规划，开始新增一层受控的 local depth correction densify，而不是去放宽 verify 阈值。

新增文件：
- `pseudo_branch/brpo_depth_densify.py`
- `scripts/materialize_m5_depth_targets.py`

代码修改：
- `pseudo_branch/brpo_depth_target.py`

关键实现：
- 在 verified seed 上构造 `log-depth correction`，而不是传播 absolute depth；
- 先实现首版 `patch-wise constant correction`；
- 输出：
  - `target_depth_for_refine_v2.npy`
  - `target_depth_dense_source_map.npy`
  - `depth_correction_seed.npy`
  - `depth_correction_dense.npy`
  - `depth_seed_valid_mask.npy`
  - `depth_dense_valid_mask.npy`
  - `depth_densify_meta.json`

默认参数第一次运行结果：
- 只把 densified 区域提升到 `~2.08%`
- 总 non-fallback 只有 `~3.64%`
- 结论：太保守，不足以支撑下游对照

随后做了小范围 sweep，目标不是追求极大 coverage，而是找一个“明显增强但仍受控”的配置。

当前选中的一组参数：
- `patch_size = 11`
- `stride = 5`
- `min_seed_count = 4`
- `max_seed_delta_std = 0.08`

在这组参数下：
- `seed_ratio ≈ 1.56%`
- `densified_ratio ≈ 14.21%`
- `total_nonfallback_ratio ≈ 15.77%`
- `render_fallback_ratio ≈ 84.23%`

更关键的是，在 train-mask 内部：
- `nonfallback_within_train_mask_ratio ≈ 81.22%`
- `seed_within_train_mask_ratio ≈ 8.05%`
- `dense_within_train_mask_ratio ≈ 73.17%`

阶段判断：
- M5-1 upstream 是可行的；
- 这一步已经把真正有新几何信息的区域从 seed 级抬到了 train-mask 的大部分区域；
- 因此后续值得测试 source-aware depth loss，而不是继续纠结 upstream coverage 是否还太低。

提交：
- `8a9c224` — `part3: add M5 depth signal analysis and densify targets`

### Mask problem route：M5-2 source-aware depth loss 接入 Stage A

在 M5-1 之后，继续把 Stage A 接到新的 densified target，并把 depth loss 按 source 拆开。

代码修改：
- `pseudo_branch/pseudo_loss_v2.py`
- `scripts/run_pseudo_refinement_v2.py`

关键实现：
- `run_pseudo_refinement_v2.py` 新增：
  - `stageA_target_depth_mode=blended_depth_m5`
  - `stageA_depth_loss_mode=legacy|source_aware`
- `pseudo_loss_v2.py` 新增：
  - `loss_depth_seed`
  - `loss_depth_dense`
  - `loss_depth_fallback`
- 当前 source-aware 默认权重：
  - `lambda_depth_seed = 1.0`
  - `lambda_depth_dense = 0.35`
  - `lambda_depth_fallback = 0.0`

实验目录：
- `.../2026-04-12_m52_stageA_loss_eval/`

对照结果：
1. `M5 + legacy depth loss`
   - `loss_depth ≈ 0.02770`
   - 300 iter 中基本完全不动
2. `M5 + source-aware depth loss`
   - `loss_depth ≈ 0.06867`
   - 其中：
     - `loss_depth_seed ≈ 0.05796`
     - `loss_depth_dense ≈ 0.03062`
     - `loss_depth_fallback ≈ 0`
   - 300 iter 中同样基本完全不动

阶段判断：
- source-aware depth loss 已经真实接通；
- fallback 也已基本从 depth loss 主体中剔除；
- 但 depth 仍然不动，说明问题已经不再只是 coverage 或 fallback 稀释。

提交：
- `d700194` — `part3: add M5 source-aware stageA depth loss`

### Stage A diagnosis：确认 `theta/rho` 在 renderer 中 forward 丢弃、backward 仍回梯度

在 M5-2 之后，继续追查为什么 loss 仍完全不动，并审核 refine 过程本身是否存在更底层的问题。

新增文件：
- `scripts/diagnose_stageA_gradients.py`

实验目录：
- `.../2026-04-12_m53_stageA_diagnosis/m53_gradients.json`

代码审计先确认：
- `cam_rot_delta / cam_trans_delta` 确实传进了 `gaussian_renderer`
- optimizer 也确实在更新它们

但真正关键的 diagnosis 结果是：

1. **forward sensitivity probe 极其异常**
   - 人工把 `cam_rot_delta / cam_trans_delta` 设到 `0.01 / 0.01`、`0.05 / 0.05`、`0.1 / 0.1`、`0.2 / 0.2`
   - 重新 render RGB / depth
   - 与 base render 比较，结果始终：
     - `rgb_mean_abs_change = 0.0`
     - `depth_mean_abs_change = 0.0`

2. **backward 却返回了非零 pose 梯度**
   - `mean_grad_rgb_rot ≈ 0.265`
   - `mean_grad_rgb_trans ≈ 0.0786`
   - `mean_grad_depth_legacy_rot ≈ 0.504`
   - `mean_grad_depth_legacy_trans ≈ 0.197`
   - `mean_grad_depth_src_rot ≈ 1.732`
   - `mean_grad_depth_src_trans ≈ 0.550`

3. **pose regularization 在初始化为 0 时梯度也是 0**
   - 因此最开始没有任何“拉回 pose”的约束力

阶段判断：
- 当前最可疑的问题已经被代码链路审计确认：

**Stage A 的 `theta/rho` 在 Python 层被传入，但在 C++/CUDA forward 中被完全丢弃；与此同时 backward 仍然单独构造并返回 `dL_dtau`。**

这也解释了为什么：
- loss 基本不下降；
- pose delta 却会持续增大；
- 当前 Stage A 的 pose refine 结果不应被直接信任。

进一步的代码证据是：
- `diff_gaussian_rasterization/__init__.py::_RasterizeGaussians.forward()` 的 `args` 不包含 `theta/rho`；
- `rasterize_points.h/.cu::RasterizeGaussiansCUDA` 的 forward 函数签名也没有 `theta/rho`；
- 但 `_RasterizeGaussians.backward()` 会从 `_C.rasterize_gaussians_backward(...)` 收到 `grad_tau`，再拆成 `grad_theta / grad_rho`。

因此当前明确不建议直接进入 Stage B。

补充修正判断：
- 经过进一步代码链路审计，这里更准确的结论不是“CUDA 单纯写坏了一个本该 forward 直接吃 `theta/rho` 的接口”；
- S3PO 原始代码里，`cam_rot_delta / cam_trans_delta` 本来就是一步一清的增量 residual，`slam_frontend.py` / `slam_backend.py` 会在每次 `optimizer.step()` 后立刻调用 `update_pose(viewpoint)`，把 `tau` 折回 `R/T` 并清零；
- 当前 Part3 Stage A 的真正问题是：**我们沿用了 residual 参数和 backward 梯度，但没有沿用 `update_pose()` 这一步，因此把 S3PO 的增量式 pose 优化闭环用断了。**


### Stage A 修复：补回 S3PO residual pose 闭环 + 最小验证

在继续追根因之后，进一步梳理了 S3PO 原始代码对 `cam_rot_delta / cam_trans_delta` 的使用方式。

关键发现：
- 原始 S3PO 不是把 `theta/rho` 当作持续直接作用于 forward render 的状态量；
- 它把它们当作**一步一清的 pose residual**；
- `slam_frontend.py` / `slam_backend.py` 在每次 `optimizer.step()` 后都会立刻调用 `update_pose(viewpoint)`；
- `pose_utils.py::update_pose()` 会把 `tau=[rho, theta]` 折回 `R/T` 并清零。

所以当前 Part3 Stage A 的真正问题不是“CUDA 单纯坏了”，而是：

**我们沿用了 residual 参数和 backward 梯度，但没有沿用 `update_pose()` 这一步，因此把 S3PO 的增量式 pose 优化闭环用断了。**

代码修改：
- `pseudo_branch/pseudo_camera_state.py`
- `scripts/run_pseudo_refinement_v2.py`

关键实现：
- 新增 `apply_pose_residual_()`：
  - 用当前 `tau=[cam_trans_delta, cam_rot_delta]` 计算新的 `w2c`；
  - 折回 `vp.R / vp.T`；
  - 刷新 `world_view_transform / full_proj_transform / camera_center`；
  - 清零 residual。
- `run_pseudo_refinement_v2.py` 现在在每次 `optimizer.step()` 后，默认会对本轮参与优化的 pseudo views 调 `apply_pose_residual_()`；
- 同时新增开关：
  - `--stageA_apply_pose_update`（默认开启）
  - `--stageA_no_apply_pose_update`（用于保留旧坏链路做对照）

最小验证 1：手动 residual -> fold 到 R/T -> render
- 修复后，render 会立刻随 pose 变化：
  - `(0.01, 0.01)`：`rgb_mean ≈ 0.0679`, `depth_mean ≈ 0.1648`
  - `(0.05, 0.05)`：`rgb_mean ≈ 0.1604`, `depth_mean ≈ 0.4364`
  - `(0.10, 0.10)`：`rgb_mean ≈ 0.2412`, `depth_mean ≈ 0.8877`

最小验证 2：M5 source-aware 80 iter smoke 对照
- `no_apply_pose_update`（旧坏链路）：
  - `loss_depth: 0.06867 -> 0.06867`（完全平）
  - `pose_updates_total = 0`
  - 最终 residual 堆积在 `rot_norm ~ 0.27~0.41`、`trans_norm ~ 0.13`
- `apply_pose_update`（修复后闭环）：
  - `loss_total: 0.02701 -> 0.02570`
  - `loss_depth: 0.06867 -> 0.06826`
  - `loss_depth_seed: 0.05796 -> 0.05760`
  - `loss_depth_dense: 0.03062 -> 0.03046`
  - `pose_updates_total = 240`
  - 最终 residual 全部回到 `0.0`

阶段判断：
- 这说明当前 Stage A 已经不再是之前那种“forward 不动、backward 乱推”的假优化；
- 但也要老实说：修复闭环之后，depth 只是开始轻微下降，还没有出现非常强的下降趋势；
- 所以下一步不应直接进入 Stage B，而应先做修复后的中等规模验证，再决定是否需要继续加强 loss / 训练长度 / 更深层 renderer 改动。


### Stage A 修复后 300-iter 参数/规模验证

在补回 `apply_pose_residual_()` 之后，没有直接进入下一阶段，而是先做一轮修复后的 300-iter 参数/规模验证，确认当前问题更像训练规模、权重还是结构。

实验目录：
- `.../2026-04-12_m55_pose_fix_scale_eval/default_300`
- `.../2026-04-12_m55_pose_fix_scale_eval/depth_heavy_300`
- `.../2026-04-12_m55_pose_fix_scale_eval/no_exposure_300`
- 汇总：`.../analysis/m55_summary.json`

实验口径：
1. `default_300`
   - `beta_rgb=0.7`
   - `lambda_depth_seed=1.0`
   - `lambda_depth_dense=0.35`
   - `lr_exp=0.01`
2. `depth_heavy_300`
   - `beta_rgb=0.3`
   - `lambda_depth_seed=1.0`
   - `lambda_depth_dense=1.0`
   - `lr_exp=0.01`
3. `no_exposure_300`
   - `beta_rgb=0.7`
   - `lambda_depth_seed=1.0`
   - `lambda_depth_dense=0.35`
   - `lr_exp=0.0`

结果：
- `default_300`
  - `loss_total: 0.02701 -> 0.02577`
  - `loss_depth: 0.06867 -> 0.06816`
  - `loss_depth_seed: 0.05796 -> 0.05750`
  - `loss_depth_dense: 0.03062 -> 0.03047`
- `depth_heavy_300`
  - `loss_total: 0.06475 -> 0.06364`
  - `loss_depth: 0.08857 -> 0.08747`
  - `loss_depth_seed: 0.05796 -> 0.05731`
  - `loss_depth_dense: 0.03062 -> 0.03016`
- `no_exposure_300`
  - `loss_total: 0.02701 -> 0.02696`
  - `loss_rgb: 0.00915 -> 0.00928`
  - `loss_depth: 0.06867 -> 0.06822`

附加分析：真实 pose 累计变化（init -> final）
- `default_300`
  - mean `trans_delta_l2 ≈ 0.00138`
  - mean `rot_delta_rad ≈ 0.00498`
- `depth_heavy_300`
  - mean `trans_delta_l2 ≈ 0.00390`
  - mean `rot_delta_rad ≈ 0.00453`
- `no_exposure_300`
  - mean `trans_delta_l2 ≈ 0.00141`
  - mean `rot_delta_rad ≈ 0.00463`

阶段判断：
- 修复后，depth 已经不是完全平线；
- 但三组设置都只表现出“弱下降”，没有出现质变；
- `depth_heavy_300` 的确更强一点，但提升仍有限；
- 冻结 exposure 会让 RGB 更差，但不会显著改善 depth；
- 这说明问题已经从“闭环断了”进一步收敛成：

**当前更像是权重/尺度与结构设计问题，而不再是链路完全断开。**

新的结构发现：
- 因为现在每步都会 `apply_pose_residual_()` 并清零 residual，当前 `pose_reg_loss = ||cam_rot_delta|| + ||cam_trans_delta||` 会在训练中几乎始终为 0；
- 这意味着 `stageA_lambda_pose` 当前已经无法约束“累计位姿偏移”；
- 因此下一步需要重点判断：是否要补一个基于 `R/T` 相对初值的 **absolute pose prior**，而不是继续只对 residual 本身做正则。

### 文档恢复后的同步补写

由于误删并恢复了 `docs/`，本轮重新按写作规范同步补回：
- `docs/STATUS.md`
- `docs/DESIGN.md`
- `docs/CHANGELOG.md`

补回原则：
- `STATUS.md` 只写当前状态，不追加过程细节；
- `DESIGN.md` 只写当前设计判断与默认实验方案；
- `CHANGELOG.md` 记录 M1 / M2 / M2.5 / M3 / M4 / M4.5 的真实过程与结果；
- 避免把“现状”和“历史过程”混写。


### Mask problem route：M5-4 absolute pose prior 接入与 smoke 对照

目标：把 `absolute_pose_prior.md` 从设计文档落地为可执行代码，并验证它在当前 Stage A 闭环下是否有效。

代码修改：
- `third_party/S3PO-GS/utils/pose_utils.py`
- `pseudo_branch/pseudo_camera_state.py`
- `pseudo_branch/pseudo_loss_v2.py`
- `pseudo_branch/pseudo_refine_scheduler.py`
- `scripts/run_pseudo_refinement_v2.py`

关键实现：
- 新增 `SO3_log / SE3_log`；
- `make_viewpoint_trainable()` 存 `R0/T0`；
- 新增 `absolute_pose_prior_loss`（基于当前 `w2c` 相对初始 `w2c0` 的 `SE3_log`）；
- Stage A 增加 `--stageA_lambda_abs_pose`，并写入 history 与导出状态；
- `export_view_state` 新增 `pose_w2c_initial / abs_pose_rho / abs_pose_theta / abs_pose_norm`。

最小验证：
- `py_compile` 全通过；
- CLI `--help` 可见 `--stageA_lambda_abs_pose`；
- `SE3_exp -> SE3_log` roundtrip 误差约 `3.95e-09`。

60-iter smoke（同一 pseudo_cache，seed=0）：
- `default_noabs` vs `default_abs(0.1)`：几乎重合；
- `depth_heavy_noabs` vs `depth_heavy_abs(0.1)`：几乎重合；
- `default_abs(100)`：
  - `abs_pose_norm mean 0.001535 -> 0.000338`（drift 明显收紧）
  - 但 `loss_depth` 从 `0.068364` 变为 `0.068543`（略变差）。

阶段判断：
- absolute prior“接入成功”≠“参数可用”；
- 当前结果更像权重尺度问题：`0.1` 太弱，`100` 太强；
- 下一步应做 300-iter 权重扫描与尺度化 prior，而不是直接拿 `100` 当默认值。

### 与 2026-04-11（mask problem 初始阶段）的设计差异

相较昨天，当前设计变化是实质性的：

1. 昨天主矛盾是 **upstream coverage 不足**（先做 verify/seed/train-mask/M3/M5）；
2. 现在 upstream 已可用，主矛盾转为 **Stage A 结构标定**（闭环已修，开始做 abs prior）；
3. 昨天主要在“把 depth 信号接进来”，现在是“如何在 drift 约束与 depth 对齐之间找平衡”；
4. 因此当前不进入 Stage B 的理由也变化了：
   - 不是“链路没接通”；
   - 而是“已接通但参数区间未定”。

## 2026-04-11

### Internal cache / replay：Phase 1 与 Phase 2 跑通

在 Re10k-1 的 80-frame smoke rerun 上，完成了 internal cache 导出与 replay 机制验证。

新增/修改：
- `third_party/S3PO-GS/utils/internal_eval_utils.py`
- `third_party/S3PO-GS/slam.py`（opt-in internal cache 导出）
- `part3_BRPO/scripts/replay_internal_eval.py`

导出结果结构：
- `internal_eval_cache/manifest.json`
- `internal_eval_cache/camera_states.json`
- `internal_eval_cache/before_opt/`
- `internal_eval_cache/after_opt/`

关键确认：
- `color_refinement()` 只优化 gaussians，不更新 pose；
- 因此当前 cache 设计改为：**camera states 共享一份，before/after 各自保存自己的 PLY 与 render cache**。

same-ply consistency check：
- after_opt：internal `31.4016 / 0.95257 / 0.04713` vs replay `31.3581 / 0.95206 / 0.04761`
- before_opt：internal `23.7945 / 0.88950 / 0.12889` vs replay `23.7256 / 0.88842 / 0.13007`

结论：replay 与官方 internal eval 已足够接近，可以用于后续 baseline/refined 公平比较。

### Internal cache / replay：Phase 3 smoke 协议验证

使用同一份 Re10k-1 smoke internal cache，对比：
- baseline sparse PLY：`/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_sparse/Re10k-1_part2_s3po/2026-04-04-00-43-29/point_cloud/final/point_cloud.ply`
- current E refined PLY：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_stage1/re10k-1/sparse/2026-04-08_joint_refine_tertile_freezegeom_lambda0p5/E_joint_realdensify_freezegeom_lambda0p5_tertile/refined.ply`

after_opt replay：
- baseline sparse：`21.3486 / 0.79084 / 0.10690`
- E refined：`19.0380 / 0.69438 / 0.23764`

delta（E - baseline）：
- `PSNR -2.3106`
- `SSIM -0.09646`
- `LPIPS +0.13073`

结论：在 smoke internal protocol 下，current E refined PLY 明显劣于 baseline sparse PLY。
但这仍是 **80-frame smoke**，还不是 full rerun 的最终正式结论。

### BRPO confidence mask：Phase B / Phase C 原型

基于 full internal cache：
`/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache/`

新增文件：
- `part3_BRPO/pseudo_branch/brpo_reprojection_verify.py`
- `part3_BRPO/pseudo_branch/brpo_confidence_mask.py`
- `part3_BRPO/scripts/brpo_verify_single_branch.py`
- `part3_BRPO/scripts/brpo_build_mask_from_internal_cache.py`

工程决策：
- BRPO 新线与旧 EDP 线隔离，不复用 `epipolar_depth.py` / `build_pseudo_cache.py`；
- verification 采用 **方案 B**：按需用 `ref pose + stage PLY` 现渲染 KF depth；
- 当前阶段不改 refine loss，只先跑 verified support / confidence mask。

Phase B（left 单分支）在 `after_opt` 的 3 个样本帧上跑通：
- 平均 `support_ratio_vs_matches = 0.8687`
- 平均 `mean_reproj_error = 1.1048 px`
- 平均 `mean_rel_depth_error = 0.0196`

Phase C（left + right + mask fusion）在同 3 个样本帧上跑通：
- 平均 `support_ratio_left = 0.01352`
- 平均 `support_ratio_right = 0.01003`
- 平均 `support_ratio_both = 0.00803`
- 平均 `support_ratio_single = 0.00749`
- 平均 `mean_reproj_error_left = 1.1048 px`
- 平均 `mean_reproj_error_right = 1.1354 px`
- 平均 `mean_rel_depth_error_left = 0.0196`
- 平均 `mean_rel_depth_error_right = 0.0241`

说明：
- verified support 没有出现“几乎全空”的坏现象；
- 双边交集 `support_both` 能稳定落到可见结构区域；
- 这说明 `internal cache + on-demand ref depth render + bidirectional reprojection verification` 这条新链已具备进入 mask-only replacement ablation 的条件。

### Internal prepare：Phase 3 正式并入 `select → difix → verify → pack`

在 full internal cache 上，把 internal prepare 正式扩成：

```text
select → difix → verify → pack
```

关键实现：
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py` 增加 `verify` stage；
- `scripts/brpo_build_mask_from_internal_cache.py` 支持 `--pseudo-left-root / --pseudo-right-root`；
- verification 可以直接消费 `difix/left_fixed` 与 `difix/right_fixed`；
- `pack` 阶段可自动把 verification 产物接入 `pseudo_cache/samples/<id>/`。

关键 prototype run：
- `re10k1__internal_afteropt__difix_proto_v2`
- `re10k1__internal_afteropt__brpo_proto_v3`

当前 3-frame prototype 使用显式 frame ids：
- `10`
- `50`
- `120`

说明：
- 这轮不是自动 `midpoint / tertile` 选点，而是显式 `frame_ids`；
- `Difix left/right`、`BRPO verification` 与 `pseudo_cache` 已在同一条 internal prepare 流程内打通。

### Phase 4：mask-only ablation（legacy vs brpo）

在同一份 after_opt PLY、同一份 3-frame internal pseudo_cache、同一份 v1 fixed-pose RGB-only refine 配置下，完成：
- `confidence_mask_source=legacy`
- `confidence_mask_source=brpo`

两组 refine 输出：
- legacy：`.../2026-04-11_mask_only_ablation_proto/legacy_left/refined.ply`
- brpo：`.../2026-04-11_mask_only_ablation_proto/brpo_left/refined.ply`

训练内现象：
- `brpo` 的 pseudo loss 下降更快、最终更低；
- 但 replay 结果并未更好。

replay eval（same internal camera states, after_opt）结果：
- baseline internal after_opt：`23.9489 / 0.87349 / 0.07878`
- legacy refine：`22.3862 / 0.84365 / 0.10834`
- brpo refine：`21.9324 / 0.82049 / 0.13134`

结论：
- 当前 BRPO verification 链条**能跑通**；
- 但当前 BRPO mask 在 `v1 fixed-pose RGB-only refine` 下**不优于 legacy mask**；
- 更可能的问题是：当前 mask 语义与当前 refine 消费方式不匹配，而不是 verification 完全失效。

### Phase 5：auditability / provenance 修补

根据 Phase 4 结果，继续检查了 `verify → pack → refine` 的 provenance 是否完整可审计。

发现的问题：
- verification 之前虽然已经支持消费 Difix left/right repaired 图，但 `verification_meta.json` 的 provenance 记录不够完整；
- `prepare_stage1_difix_dataset_s3po_internal.py` 的 sample manifest 对 `.json` 类型 BRPO 产物路径命名不够规范；
- `run_pseudo_refinement.py` 的 history 之前没有写出每个 sample 实际吃到的 target / confidence 来源。

对应补丁：
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
- `scripts/run_pseudo_refinement.py`

补丁后能力：
- verification 会记录 branch 实际输入、override 状态、policy 与 matcher/source 信息；
- pack 会记录更完整的 `source_meta` 与 `verification_*_path`；
- refine history 会记录每个 pseudo sample 的 `target_rgb_path / confidence_path / confidence_source_kind / coverage`。

说明：
- 这一步当前只完成了**代码与编译验证**；
- 若要让产物也带上新 provenance，需要在后续 Phase 6/7 中重跑 verify / pack / refine。

### Phase 6：schema solidification 第一轮实现与真实 smoke

本轮开始正式进入 Phase 6，不再停留在文档规划。

代码修改：
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
- `pseudo_branch/brpo_confidence_mask.py`
- `scripts/run_pseudo_refinement.py`

关键实现：
- `prepare_stage1_difix_dataset_s3po_internal.py` 新增 `fusion` stage，`all` 流程改为 `select → difix → fusion → verify → pack`；
- `fusion` 直接复用 `pseudo_fusion.py`，先实现 **rgb-only fused**，产出 `target_rgb_fused.png`、`confidence_mask_fused.npy/png`、`fusion_meta.json`；
- `pack` 阶段新增 `target_depth_for_refine.npy` 预留位，并将 fused 产物接入 `pseudo_cache/samples/<id>/`；
- `brpo_confidence_mask.py` 从单一 `confidence_mask_brpo.npy` 扩展为：
  - `confidence_mask_brpo_left.npy/png`
  - `confidence_mask_brpo_right.npy/png`
  - `confidence_mask_brpo_fused.npy/png`
  同时保留旧的 `confidence_mask_brpo.npy/png` 作为兼容 alias；
- `run_pseudo_refinement.py` 的 BRPO consumer 改为按 `target_side` 解析 confidence。

真实验证：
- 在现有 prototype `re10k1__internal_afteropt__brpo_proto_v3` 上，实际重跑了：
  `fusion → verify → pack`
- `verify` 目录中已真实产出新的 `left/right/fused` BRPO masks；
- `pack` 初次验证时发现一个真实漏链：verify 已产出新 masks，但 `maybe_link_brpo_artifacts()` 仍按旧白名单链接，导致 sample 里没有带入新字段；
- 随后补上 pack linkfix，并只重跑 `pack`，确认 `pseudo_cache/samples/10/` 中已真实出现：
  - `target_rgb_fused.png`
  - `target_depth_for_refine.npy`
  - `confidence_mask_brpo_left/right/fused.npy`

consumer smoke：
- 直接调用 `load_pseudo_viewpoints(...)`，确认以下三种模式都能正确解析本地 sample 内的新文件：
  - `left + brpo`
  - `right + brpo`
  - `fused + brpo`
- 进一步跑通了一次真实 refine smoke：
  - `target_side=fused`
  - `confidence_mask_source=brpo`
  - `num_iterations=2`
  - 成功进入训练循环并正常落盘。


### A.5 300iter + replay 对照（baseline vs xyz+opacity）

运行目录：
- `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_stageA5_midpoint_300iter_withply/`

关键结果：
- StageA baseline replay(270): `PSNR 23.948911 / SSIM 0.873485 / LPIPS 0.0787798`
- StageA5(xyz+opacity) replay(270): `PSNR 23.970262 / SSIM 0.873834 / LPIPS 0.0787174`
- delta(xyz+opacity - baseline): `PSNR +0.021351 / SSIM +0.000348 / LPIPS -6.23e-05`

结论：
- 与此前“3帧改动后 replay 几乎不变”不同，midpoint 8 帧 + A.5(xyz+opacity) 已出现可复现的下游正向信号；
- `xyz-only` 仍非优先路径。


### StageB conservative Phase0/1/2 执行（2026-04-13）

- 新增计划文档：`docs/STAGEB_CONSERVATIVE_ENTRY_PLAN.md`
- 新增执行报告：`docs/STAGEB_PHASE0_2_EXEC_REPORT_20260413.md`
- `run_pseudo_refinement_v2.py` 新增 `stageB` 模式（conservative joint）：
  - real anchor 稀疏视角分支（`--train_manifest`, `--num_real_views`, `--lambda_real`）
  - pseudo RGBD masked 分支（`--lambda_pseudo`）
  - joint 优化 pseudo camera 参数 + micro gaussian 参数（`xyz/xyz+opacity`）
- 本次 gate run（120iter）相对 A.5 baseline replay：
  - `PSNR +0.29383`
  - `SSIM +0.004369`
  - `LPIPS -0.0002153`
- gate 判定：`PASS`


### StageB Phase3 long run（300iter）

- run: `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_stageB_phase3_longrun/stageB300_conservative_xyz_opacity`
- replay: `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_stageB_phase3_longrun/stageB300_conservative_xyz_opacity/replay/stageB300_conservative_xyz_opacity/replay_eval.json`
- delta(vs A.5): PSNR -0.179275, SSIM -0.004948, LPIPS +0.002430
- gate: FAIL


### Strict midpoint8 + M5 + source-aware full pipeline execution（2026-04-13 夜）

新建完整执行与分析报告：
- `docs/MIDPOINT8_M5_STAGEA_A5_B_EXEC_REPORT_20260413.md`

本轮严格协议：
- midpoint 8 帧：`17/51/86/121/156/190/225/260`
- 先对 midpoint pseudo cache 补 `target_depth_for_refine_v2.npy`（M5）
- 然后运行：
  - `StageA(source_aware)`
  - `StageA.5(source_aware + xyz_opacity)`
  - `StageB120(conservative source-aware)`
- 每阶段都做 270-frame replay

结果（vs part2 after_opt baseline）：
- StageA：几乎完全不变（噪声级）
- StageA.5：`PSNR -0.11066 / SSIM -0.00794 / LPIPS +0.00878`
- StageB120：`PSNR -0.78787 / SSIM -0.02304 / LPIPS +0.01903`

关键分析结论：
1. `final_delta_summary` 会误导地显示 pose residual 为零；但真实 `pose_w2c` 仍有小幅变化；
2. StageA replay 不变的真正原因不是“pose 绝对没动”，而是 **StageA 不更新 Gaussian，导出的 PLY 与输入 hash 完全一致**；
3. 当前 `StageA -> A.5 -> B` 不是严格顺序相机 handoff，A.5 / B 都不会读取前一阶段的 `pseudo_camera_states_final.json`；
4. 当前 midpoint8 M5 的有效 non-fallback 几何监督偏弱（train-mask 内约 `18.65%`），不足以支撑强 pose 提升；
5. 现在更需要修 pipeline 结构，而不是继续把问题当成纯参数扫描。


### P0 repair: stage handoff + StageA reporting fix + repaired sequential rerun（2026-04-13 夜）

代码修复：
- `pseudo_branch/pseudo_camera_state.py`
  - 新增：
    - `load_exported_view_states(...)`
    - `apply_loaded_view_state_(...)`
    - `summarize_true_pose_deltas(...)`
- `scripts/run_pseudo_refinement_v2.py`
  - 新增 CLI：
    - `--init_pseudo_camera_states_json`
    - `--init_pseudo_reference_mode {keep,reset_to_loaded}`
  - 新增 handoff summary 记录
  - 新增 true pose delta summary / aggregate
  - 原 `final_delta_summary` 改为更准确的 `residual_delta_summary_post_foldback`

验证：
- `repair_seq_check/` short smoke 已确认：
  - `StageA final -> StageA.5 init` 完全对齐
  - `StageA.5 final -> StageB init` 完全对齐
- `repair_seq_rerun/` 已完成：
  - `StageA80`
  - `StageA.5(80, from StageA)`
  - `StageB120(from StageA.5)`

结果变化：
- handoff 修复后，A.5 / StageB 相对修复前 strict run 有小幅改善；
- 但整体仍然未超过 baseline，说明 handoff bug 只是其中一个结构问题，不是唯一瓶颈。

过程复盘沉淀：
- 以后 staged pipeline 设计必须先验证：
  1. handoff continuity
  2. evaluation-artifact continuity
  3. true-state summary correctness
  4. short smoke before long run

### P1 bottleneck review: signal / method mismatch for midpoint8 M5（2026-04-13 夜）

代码与数据链路复查：
- 复查 `brpo_train_mask.py` / `brpo_depth_densify.py` / `brpo_depth_target.py` / `pseudo_loss_v2.py` / `pseudo_refine_scheduler.py` / `gaussian_param_groups.py` / `run_pseudo_refinement_v2.py`
- 复查 midpoint8 M5 pseudo cache 与 `repair_seq_rerun/` 实验产物

新增确认的关键事实：
- pseudo RGB 与 pseudo depth 都只在 `train_confidence_mask_brpo_fused.npy` 非零区域生效；
- 在 `lambda_depth_fallback=0` 下，fallback depth 区域不提供 depth 梯度；
- midpoint8 M5 的平均有效几何支持仅约 `3.49%`，平均 fallback 约 `96.51%`；
- support 区域内 `target_depth_for_refine_v2` 相对 `render_depth` 的平均修正仅约 `1.53%`；
- A.5 / StageB 却直接打开全局 Gaussian `xyz/opacity`，导致 supervision scope 与 optimization scope 明显不匹配。

新增量化结论：
- StageA80 的平均 pose 平移改变量仅约 scene scale 的 `0.118%`；
- A.5 相对 baseline PLY：`74.1%` Gaussians 的 xyz 改动 `>1e-3`；
- StageB120 相对 baseline PLY：`86.2%` Gaussians 的 xyz 改动 `>1e-3`。

工程判断更新：
- 当前主瓶颈更接近 `signal weak + underconstrained global refinement`；
- 后续优先事项应从“继续扫同一组 refine 超参”转向：
  1. pseudo 选点 / signal 质量复查；
  2. A.5 / B 的可训练 Gaussian 作用域收缩；
  3. 将 support/correction 量化检查加入长跑前固定 gate。

### 2026-04-14：新增下一阶段执行计划（signal semantics + stable refinement）

新增文档：
- `docs/SIGNAL_SEMANTICS_AND_STABLE_REFINEMENT_PLAN_20260414.md`
- 同步更新：`docs/hermes.md`

本轮不是代码实现，而是基于更新后的 `DEBUG_CONVERSATION.md`、BRPO 差异分析和真实代码链路，形成下一阶段的工程方案。核心收敛：
1. 先做 `continuous confidence + agreement-aware support`；
2. 把 RGB/raw confidence 与 depth/train-mask 语义分离；
3. densify 改成 confidence-aware；
4. StageB 改成两段式 curriculum，并在 pseudo 分支补 local Gaussian gating；
5. full SPGM 暂不作为第一优先级，而是后续 stabilizer 候选。
### 2026-04-14：S1.1 完成（continuous confidence + agreement-aware support）

代码更新：
- `pseudo_branch/brpo_confidence_mask.py`
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`

本轮落地内容：
1. 在离散 `1.0 / 0.5 / 0.0` 之外，新增 raw continuous confidence 输出；
2. both-support 新增基于双侧 projected-depth 的 agreement-aware 连续权重；
3. verification 产物新增：
   - `raw_confidence_mask_brpo_{left,right,fused}.*`
   - `raw_confidence_mask_brpo_cont_{left,right,fused}.*`
   - `confidence_mask_brpo_agreement.*`
4. sample pack / link 阶段已可带上这些新 artifact。
### 2026-04-14：S1.2 完成（RGB/depth mask semantics split）

代码更新：
- `scripts/run_pseudo_refinement_v2.py`
- `pseudo_branch/pseudo_loss_v2.py`

本轮落地内容：
1. refine CLI 新增：`--stageA_rgb_mask_mode`、`--stageA_depth_mask_mode`、`--stageA_confidence_variant`；
2. `load_stageA_pseudo_views(...)` 现已分别解析 `conf_rgb` 与 `conf_depth`；
3. `build_stageA_loss*` 现已支持 RGB / depth 分别使用不同 mask；
4. history / per-view stats 已新增 RGB / depth mask 与 coverage 记录。
### 2026-04-14：S1.3 完成 + 2-frame smoke/ablation

代码更新：
- `pseudo_branch/brpo_depth_densify.py`
- `pseudo_branch/brpo_depth_target.py`
- `scripts/materialize_m5_depth_targets.py`

本轮落地内容：
1. densify 新增 confidence-aware patch acceptance；
2. 支持 both-rich patch 放宽 seed-count、single-dominant patch 收紧 delta-std；
3. M5 materialize 新增 `--use-continuous-confidence`、`--min-patch-confidence`、`--both-seed-count-relax`、`--single-std-tighten`。

2-frame smoke：
- 目录：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_semantics_smoke`
- 已确认 verify/pack 可真实产出 `raw_confidence_mask_brpo_cont_*` 与 `confidence_mask_brpo_agreement.*`，并成功进入 pseudo_cache sample。

2-frame compare：
- 目录：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_semantics_compare`
- baseline densify：`mean_dense_valid_ratio≈0.02505`
- conf-aware densify：`mean_dense_valid_ratio≈0.00129`
- 说明首轮 confidence-aware 阈值过于保守，densify coverage 被压得过小。

2-frame StageA-10iter ablation：
- baseline(shared train-mask/discrete) 与 conf-aware(split raw/train + continuous) 均已跑通；
- conf-aware 组的 RGB/depth 作用域显著收缩，pose drift 更小，但 loss 更高；
- 当前结论是：语义分离与 confidence-aware wiring 已打通，但 S1.3 需要先回调阈值，再进入更正式的 8-frame 短跑对照。
### 2026-04-14 晚：8-frame retuned compare

执行目录：
- `.../20260414_signal_semantics_mid8`
- `.../20260414_signal_semantics_mid8_compare`

本轮完成：
1. 8-frame verify/pack 联调；
2. retuned confidence-aware densify（`min_patch_confidence=0.12, both_relax=2, single_std_tighten=0.9`）；
3. 8-frame StageA-20iter baseline vs conf-aware 小对照。

关键量化：
- baseline densify：`mean_dense_valid_ratio≈0.02269`
- retuned conf-aware：`mean_dense_valid_ratio≈0.01634`
- baseline StageA20：`loss_total≈0.03673`, `pose_mean_trans≈0.00116`
- conf-aware StageA20：`loss_total≈0.04392`, `pose_mean_trans≈0.00128`

结论：
- 相比首轮过严版本，retuned conf-aware densify 已回到可用 coverage；
- 但当前 8-frame 短跑下，conf-aware 组还没有稳定优于 baseline；
- 因此下一步仍应先做 E1 小网格回调，而不是直接进入 StageB 两段式 curriculum。


### 2026-04-14 夜：E1 support-aware pseudo selection 第一轮完成

代码新增：
- `scripts/select_signal_aware_pseudos.py`

运行产物：
- `.../20260414_signal_enhancement_e1/reports/signal_aware_selection_report.json`
- `.../20260414_signal_enhancement_e1/manifests/signal_aware_selection_manifest.json`
- `docs/SIGNAL_AWARE_SELECTION_E1_REPORT_20260414.md`

本轮完成内容：
1. 对 8 个 KF gap 枚举 `{1/3, 1/2, 2/3}` 候选；
2. 基于 internal cache 的 render_rgb/render_depth 做 lightweight verify；
3. 用 `both_ratio + verified_ratio + correction_magnitude + balance` 做候选打分；
4. 生成一组 `signal-aware-8` 选中的 pseudo ids。

关键结果：
- midpoint8：`[17, 51, 86, 121, 156, 190, 225, 260]`
- signal-aware-8：`[23, 57, 92, 127, 162, 196, 225, 260]`
- `6/8` 个 gap 发生变化，前 6 个 gap 全部偏向 `2/3`
- aggregate 上，`verified_ratio / continuous confidence / balance / score` 均高于 midpoint8；`support_ratio_both` 略低。

结论更新：
- `midpoint` 不是当前 case 的稳定最优 pseudo 位置；
- E1 方向成立，但当前仍属于 lightweight render-based verify；
- 在进入 E2 之前，应先用 `signal-aware-8` 补一轮 apples-to-apples verify/pack + 8-frame StageA short compare。

### 2026-04-14 深夜：E1.5 完成（support-aware pseudo selection 正式短对照）

新增产物：
- `.../20260414_signal_enhancement_e15_compare/`
- `.../20260414_signal_enhancement_e15_compare/e15_compare_summary.json`
- `docs/SIGNAL_AWARE_SELECTION_E15_COMPARE_20260414.md`

本轮完成内容：
1. 用 `signal-aware-8 = [23, 57, 92, 127, 162, 196, 225, 260]` 重跑正式 `fusion -> verify -> pack`；
2. 复用 midpoint8_compare 的同一组 baseline / conf-aware M5 densify 参数；
3. 复用同一组 StageA-20iter baseline / conf-aware 参数，做 apples-to-apples 短对照。

关键结论：
- raw signal：`verified_ratio` 与 `continuous confidence` 有小幅提升，`support_ratio_both` 略降；
- baseline densify：`mean_dense_valid_ratio 0.02269 -> 0.02961`
- conf-aware densify：`mean_dense_valid_ratio 0.01634 -> 0.01831`
- baseline StageA20：`loss_total 0.03673 -> 0.03659`
- conf-aware StageA20：`loss_total 0.04392 -> 0.04340`

结论更新：
- support-aware pseudo selection 已通过正式短对照，不再只是 lightweight proposal；
- 它的收益主要先体现在 `verified signal -> dense coverage` 的传导；
- E1 可以收口，下一步可进入 E2（dual-pseudo allocation）。


### 2026-04-14 深夜：E2 完成（dual-pseudo allocation 正式对照）

代码更新：
- `scripts/select_signal_aware_pseudos.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`

本轮落地内容：
1. `select_signal_aware_pseudos.py` 新增 `--topk-per-gap` 与 `--allocation-policy`，支持每 gap 输出多 pseudo；
2. selection manifest 现会写入 `gap_index / allocation_rank / allocation_policy / candidate_fraction / selection_score / selection_label`；
3. `prepare_stage1_difix_dataset_s3po_internal.py` 新增 `--selection-manifest`，可直接消费 selection manifest；
4. pseudo-cache schema 升级到 `pseudo-cache-internal-v1.5`，sample/meta/manifest 均保留 allocation 元信息；
5. 已完成 E2 正式链路：`select -> fusion -> verify -> pack -> baseline/conf-aware densify -> StageA-20`。

运行产物：
- `.../20260414_signal_enhancement_e2_compare/`
- `.../20260414_signal_enhancement_e2_compare/e2_compare_summary.json`
- `docs/SIGNAL_DUAL_PSEUDO_E2_COMPARE_20260414.md`

关键结论：
- E2 的 top2 实际稳定落成 `1/2 + 2/3`，而不是固定 `1/3 + 2/3`；
- raw signal 相比 midpoint8 仍有轻微正向，但幅度有限；
- densify 层相对 midpoint8 为正，但弱于 E1 `signal-aware-8`；
- short StageA-20 baseline / conf-aware 两条线都明显差于 E1；
- 当前 default winner 仍应保持 E1 `signal-aware-8`；若进入 E3，应优先基于 E1 winner 做 multi-anchor verify，而不是直接沿用 E2 16-pseudo set。


### 2026-04-14 更深夜：补充 pipeline judgement + 归档部分已完成计划文档

新增文档：
- `docs/MASK_DEPTH_CONFIDENCE_PIPELINE_JUDGEMENT_20260414.md`

本轮新增结论：
1. `mask / depth / confidence` pipeline 不是伪方向，但其作为主增益来源的边际收益已接近见顶；
2. E1 已吃到这条线上最有价值的上游收益；
3. E2 说明“多 pseudo”在当前 consumer 机制下会稀释 winner，不应再默认沿这一路线继续扩张；
4. 若继续推进，这条线更适合作为 E3 final probe，而不是继续高强度主线投入。

文档归档：
- `docs/M5_STAGEA-A.5-B_RUNBOOK.md -> docs/archived/2026-04-superseded-plans/`
- `docs/part3_BRPO_M5_depth_densify_and_loss_plan.md -> docs/archived/2026-04-superseded-plans/`
- `docs/part3_stageA_pre_stageB_engineering_plan.md -> docs/archived/2026-04-superseded-plans/`

原因：以上文档对应的执行阶段/计划已完成或已被当前 `SIGNAL_ENHANCEMENT.md` 与最新 STATUS/DESIGN 判断取代。

### P2-K：SPGM Phase 0 plumbing 完成

新增产物:
- pseudo_branch/spgm/__init__.py, stats.py, score.py, policy.py 骨架模块
- pseudo_branch/local_gating/gating_schema.py 扩展 SPGM 字段和方法
- pseudo_branch/local_gating/gating_io.py 扩展 SPGM summary 字段
- scripts/run_pseudo_refinement_v2.py 新增 SPGM CLI, 三路分支逻辑, history 字段

备份文件:
- gating_schema.py.bak_20260416_spgm_v1
- gating_io.py.bak_20260416_spgm_v1
- run_pseudo_refinement_v2.py.bak_20260416_spgm_v1
- local_gating/__init__.py.bak_20260416_spgm_v1

新增 CLI 参数:
- pseudo_local_gating choices 扩展为 off/hard_visible_union_signal/soft_visible_union_signal/spgm_keep/spgm_soft
- pseudo_local_gating_spgm_num_clusters, alpha_depth, beta_entropy, gamma_entropy, support_eta, weight_floor, entropy_bins, density_mode
- pseudo_local_gating_spgm_cluster_keep_near/mid/far

新增 SPGM history 字段:
- spgm_active_ratio, accepted_view_count, support_mean/p50/max, depth_entropy, density_entropy
- spgm_importance_mean/p50, weight_mean/p10/p50/p90, cluster_count_near/mid/far

当前 wiring 状态:
- spgm import smoke 已通过
- run_pseudo_refinement_v2.py CLI 已正确显示 SPGM 参数
- 三路分支逻辑(visibility_union/spgm/off)已接入
- Phase 0 placeholder 实现返回零值/占位数据

下一步(Phase 1): 实现 stats.py 的真实统计量提取

### P2-K Phase 1: stats.py 真实实现完成

stats.py 实现:
- collect_spgm_stats 真实逻辑 (非placeholder)
- support_count: 加权聚合visibility_filter
- depth_value: 向量化weighted median (方案A: 排序+cumsum+阈值查找)
- density_proxy: normalized(opacity) * normalized(support)
- active_mask: support_count > 0

关键优化:
- 原版逐Gaussian Python循环卡死 (33452个元素, >5min/iter)
- 向量化版本: torch.argsort + torch.cumsum + argmax阈值查找
- smoke测试: 5iter @ 1.3s/iter (vs 原版>5min/iter)

修复:
- get_opacity是property而非函数: gaussians.get_opacity.detach() (不是callable)

smoke验证结果 (stageA5_spgm_keep_xyz_5iter):
- spgm_active_ratio: [0.587, 0.830, 0.525, 0.825, 0.667] 非全0
- spgm_accepted_view_count: [4,4,4,4,4]
- spgm_support_mean: 有真实值 (2.68~3.40)
- spgm_weight_mean: [0,0,0,0,0] (policy.py仍placeholder, Phase 3待实现)

下一步(Phase 2): 实现 score.py (depth partition / density entropy / importance score)

### P2-K Phase 2: score.py 真实实现完成

score.py 实现:
- build_depth_partition: K=3 quantile split (near/mid/far)
- depth_score: 1 - (z - z_min) / (z_max - z_min)
- compute_density_entropy: histogram entropy normalized to [0,1]
- density_score: rho_norm * (1 - beta*Hbar) + gamma*Hbar
- importance_score: alpha * depth_score + (1-alpha) * density_score
- support modifier: importance * support_norm^eta

smoke验证结果 (stageA5_spgm_keep_xyz_1iter_score):
- spgm_importance_mean: [0.557] 非零且在[0,1]
- spgm_importance_p50: [0.563]
- spgm_density_entropy: [0.095] 熵值正常
- spgm_cluster_count: near=6546, mid=6546, far=6545 合理分布

下一步(Phase 3): 实现 policy.py (deterministic soft keep, cluster-aware weighting)

### P2-K Phase 3: policy.py 真实实现完成

policy.py 实现:
- deterministic soft keep: w_keep = weight_floor + (1-weight_floor) * importance
- cluster-aware weighting: w = w_keep * cluster_keep[cluster_id]
  - cluster_keep: near=1.0, mid=0.8, far=0.6
- inactive -> 0
- summary: weight_mean/p10/p50/p90, active_ratio

完整 SPGM v1 pipeline 验证 (stageA5_spgm_keep_xyz_5iter_full):
- spgm_active_ratio: [0.587, 0.830, 0.525, 0.822, 0.667] 非全0
- spgm_weight_mean: [0.458, 0.368, 0.425, 0.386, 0.438] 权重分布合理
- spgm_importance_mean: [0.557, 0.440, 0.515, 0.465, 0.530] importance正常
- grad_keep_ratio_xyz: [0.587, 0.830, 0.525, 0.822, 0.667] 与active_ratio一致
- grad_norm_xyz: pre->post 有真实降低 (例如 0.069->0.046)

关键突破:
- Phase 0 placeholder 时 grad_keep_ratio_xyz=0.0, grad被全压成0
- Phase 3 后 grad_keep_ratio_xyz≈0.59, grad_norm真实被调制
- SPGM 不再是 no-op, 而是 real working per-Gaussian grad weighting

下一步(Phase 4): StageA.5 正式 compare (20 iter) vs baseline (hard_visible_union_signal)

### P2-K Phase 4: StageA.5 Compare 完成

实验设置:
- 两臂: baseline (hard_visible_union_signal) vs exp (spgm_keep)
- stageA_iters: 20
- trainable_params: xyz
- 其他参数与 P2-F winner 保持一致

关键对比结果:
1. Grad modulation:
   - Baseline: grad未调制 (pre=post=6.87e-02)
   - Exp: grad真实调制 (pre=6.87e-02 -> post=4.58e-02, 保留约67%)
   
2. Active ratio一致:
   - Baseline grad_keep_ratio_xyz: [0.587, 0.830] (等于visible_union)
   - Exp spgm_active_ratio: [0.587, 0.830] (完全一致)
   
3. SPGM stats正常:
   - spgm_weight_mean: [0.46, 0.37, 0.43, 0.39, 0.44]
   - spgm_importance_mean: [0.56, 0.44, 0.52, 0.46, 0.53]
   - spgm_density_entropy: [0.095, 0.089, 0.092, 0.080, 0.090]
   - cluster分布: near=6546, mid=6546, far=6545
   
4. Loss trajectory一致:
   - Baseline: 0.027 -> 0.038
   - Exp: 0.027 -> 0.038 (几乎完全一致)
   
5. Pose delta一致:
   - Baseline mean_trans_norm: 0.003064
   - Exp mean_trans_norm: 0.003063 (微秒级差异)
   
结论:
- SPGM v1 不是 no-op, grad真实被调制
- 两臂的loss和pose trajectory几乎完全一致
- 说明SPGM的grad modulation对优化动力学的影响很小
- 下一阶段需要更长时间的验证 (如80 iter) 或 StageB 接入

输出路径:
- /data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_spgm_v1_stageA5_compare_e1/

### P2-K Phase 5: StageB 接入完成

实验设置:
- 两臂: baseline (hard_visible_union_signal) vs exp (spgm_keep)
- stageB_iters: 40 (StageA 300 iter 前置)
- trainable_params: xyz
- num_real_views: 2, num_pseudo_views: 4
- lambda_real/lambda_pseudo: 1.0

关键对比结果:
1. Grad modulation 生效:
   - Baseline: grad未调制 (pre=post)
   - Exp: grad真实调制 (pre=8.13e-02 -> post=5.93e-02, 保留约73%)
   
2. Active ratio一致:
   - Baseline keep_xyz: [0.669, 0.824, 0.675]
   - Exp spgm_active_ratio: [0.669, 0.824, 0.676] (几乎一致)
   
3. SPGM stats正常:
   - spgm_weight_mean: [0.43, 0.40, 0.39, 0.46, 0.41]
   - spgm_importance_mean: [0.51, 0.49, 0.47, 0.56, 0.49]
   
4. Loss对比 (iter40):
   - Baseline total: 0.124
   - Exp total: 0.123 (几乎一致)
   - Baseline real: 0.091
   - Exp real: 0.090 (差异<1%)
   - Baseline pseudo: 0.032
   - Exp pseudo: 0.033 (差异<1%)
   
关键验证:
- SPGM只压pseudo backward后的Gaussian grad
- Real branch未受SPGM影响 (real_loss几乎一致)
- StageB的SPGM接入位置正确 (pseudo backward后、real backward前)

输出路径:
- /data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_spgm_v1_stageB_compare_e1/

下一步(Phase 6): 更长时间验证 (如120 iter) 或 正式replay评估

### P2-K Phase 6: StageB 120 iter Compare 完成

实验设置:
- 两臂: baseline (hard_visible_union_signal) vs exp (spgm_keep)
- stageB_iters: 120
- 其他参数与 P2-J winner 一致

关键对比结果:
1. Grad modulation 生效:
   - Baseline: iter1/iter120 grad pre=post (无调制)
   - Exp: iter1 0.075->0.050, iter120 0.069->0.043 (真实调制)
   
2. Loss 对比 (iter120):
   - Baseline total: 0.0576
   - Exp total: 0.0612 (略高)
   - Baseline real: 0.0335
   - Exp real: 0.0319 (**更低**)
   - Baseline pseudo: 0.0241
   - Exp pseudo: 0.0293 (更高)
   
3. 关键洞察:
   - SPGM 使 real loss 更低 (好事)
   - SPGM 使 pseudo loss 更高 (pseudo-side 被 grad weighting 压制)
   - 这说明 SPGM 可能是在帮助 real-side 优化，代价是 pseudo-side
   
4. SPGM stats 正常:
   - spgm_weight_mean: [0.43, 0.40, 0.40, 0.46, 0.41]
   - spgm_importance_mean: [0.51, 0.49, 0.47, 0.56, 0.49]
   - spgm_active_ratio: [0.67, 0.82, 0.68, 0.67, 0.67]

结论:
- SPGM v1 在 StageB 120 iter 下稳定工作
- Grad modulation 真实生效
- Real-side 受益（real loss 下降），pseudo-side 被压制（pseudo loss 上升）
- 下一步建议: 正式 replay 评估 (PSNR/SSIM/LPIPS) 确认视觉质量

输出路径:
- /data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_spgm_v1_stageB_compare_120iter/

### P2-K Phase 6 Replay 评估完成

Replay 评估结果:
- Baseline (hard_visible_union_signal_120iter):
  - PSNR: 24.136
  - SSIM: 0.8738
  - LPIPS: 0.0819
  
- Exp (spgm_keep_120iter):
  - PSNR: 23.594
  - SSIM: 0.8593
  - LPIPS: 0.0891

关键对比:
- PSNR: baseline 24.136 vs exp 23.594 → **baseline更好** (差距 -0.542)
- SSIM: baseline 0.8738 vs exp 0.8593 → **baseline更好** (差距 -0.0145)
- LPIPS: baseline 0.0819 vs exp 0.0891 → **baseline更好** (差距 +0.0072, LPIPS越低越好)

**出乎意料的发现**: SPGM 在 replay 评估中反而略差于 baseline！

可能原因分析:
1. SPGM 的 grad weighting 对 pseudo-side 压制过强, 导致 Gaussian 位置更新不够充分
2. 120 iter 还不够长, 需要更长时间让 SPGM 的效果显现
3. 当前 SPGM 超参 (alpha_depth/beta_entropy/gamma_entropy/support_eta/weight_floor/cluster_keep) 可能需要调优
4. SPGM v1 是 deterministic keep, 可能需要 stochastic drop 才能达到更好的效果

结论:
- SPGM v1 实现完整且稳定工作, 但当前超参下 replay 效果略差于 baseline
- 这是真实的实验结果, 需如实记录, 不应回避负面发现
- 后续需要: 超参调优、更长 iter、stochastic drop、xyz_opacity 组合

输出路径:
- Baseline replay: /data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_spgm_v1_stageB_compare_120iter/replay_baseline/replay_eval.json
- Exp replay: /data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_spgm_v1_stageB_compare_120iter/replay_exp/replay_eval.json

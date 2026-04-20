# CHANGELOG.md - Part3 Stage1 过程记录

> **书写规范**：
> 1. 增量更新，倒序排列（最新在最上）
> 2. 每条提炼为 3-5 行：做了什么 → 发现什么 → 结论是什么
> 3. 不重复设计判断（DESIGN.md 负责），不重复状态细节（STATUS.md 负责）
> 4. 实验细节引用归档文档：`[参见 archived/2026-04-experiments/P2X_*.md]`

---

## 2026-04-20

### A1-BRPO-D1：direct BRPO A1 builder + consumer + fixed new T1 formal compare（best new A1 branch, still below old A1）
- 不再停留在 `brpo_style_v2` 的小修上，而是直接在 `pseudo_branch/brpo_v2_signal/pseudo_observation_brpo_style.py` 新增 `build_brpo_direct_observation(...)`，并在 `build_brpo_v2_signal_from_internal_cache.py` 中并行产出 `pseudo_confidence_brpo_direct_v1 / pseudo_depth_target_brpo_direct_v1 / pseudo_source_map_brpo_direct_v1 / pseudo_valid_mask_brpo_direct_v1`；`run_pseudo_refinement_v2.py` 同步新增 `pseudo_observation_mode=brpo_direct_v1`，另新增 `scripts/run_a1_brpo_direct_compare.py` 做固定 `new T1 + summary_only` 四臂 compare。
- 先做 smoke：1) signal build 成功写出 `brpo_direct_observation_meta_v1.json` 与对应 direct bundle；2) consumer smoke 成功在 `stageA_history.json` 中真实记录 `requested/effective sample mode = brpo_direct_v1`、`target_kind = joint_depth_target_brpo_direct_v1`、`confidence_source_kind = joint_observation::pseudo_confidence_brpo_direct_v1.npy`，说明这不是 CLI 假接线，而是端到端真实消费。
- 随后在 `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_a1_brpo_direct_compare_e1/compare_summary.json` 完成四臂 formal compare：`oldA1=24.187551 / 0.875593 / 0.080362`，`newA1=24.149272 / 0.875576 / 0.079964`，`brpo_style_v2=24.167642 / 0.875317 / 0.080340`，`brpo_direct_v1=24.175408 / 0.875331 / 0.080346`。
- 结论：`brpo_direct_v1` 相对 current new A1 回升约 `+0.02614 PSNR`，相对 `brpo_style_v2` 再回升约 `+0.00777 PSNR`，说明 direct BRPO 语义方向比 current new A1 / style_v2 都更对；但它相对 old A1 仍约 `-0.01214 PSNR`，所以当前还不能接管 observation 主线。下一步不该再做单侧 toggle，而应直接定位 `old A1` vs `brpo_direct_v1` 的 residual gap（target builder / valid-set / fallback contract）。

### B3-O1-C1：delayed deterministic opacity participation formal compare（reuse old controls, still weak-negative）
- 按 `B3_opacity_participation_population_manager_engineering_plan.md` 先完成 O0/O1：在 `pseudo_branch/spgm/score.py` 显式拆出 `participation_score`，在 `pseudo_branch/spgm/manager.py` 新增 `deterministic_opacity_participation`，并把 `run_pseudo_refinement_v2.py` / `local_gating` / `gaussian_renderer` 接到 next-step `participation_opacity_scale`。先用 3-iter smoke 确认 opacity attenuation 真正在 pseudo render 生效。
- 随后做 C1 formal compare，但按用户要求**不重跑全部对照组**：复用 `20260419_b3_det_participation_compare_e2/compare_summary.json` 里的 `summary_only` 与旧 boolean conservative arm（`far=0.90, mid=1.00`），只新跑 `deterministic_opacity_participation (floor_near=1.0, floor_mid=1.0, floor_far=0.9)` 一臂。输出目录：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_b3_opacity_participation_compare_e1/`。
- 结果：复用 control `summary_only=24.187304 / 0.875587 / 0.080364`，旧 boolean arm `24.186847 / 0.875591 / 0.080387`，新 opacity arm `24.186731 / 0.875586 / 0.080398`（PSNR/SSIM/LPIPS）。新 opacity 相对 summary 为 `-0.000573 PSNR / -1.18e-06 SSIM / +3.41e-05 LPIPS`，相对旧 boolean 也略差。
- 结论：O0/O1 的**动作变量改造是成立的**，但 current delayed opacity path 仍未减轻 weak-negative；因此当前不应继续推进 O2a/b。若继续做 B3，应先回到 C0 / candidate-law / action-law 诊断，必要时再考虑 timing/current-step integration，而不是直接切 population universe。

### A1-V2-H1：fixed new T1 下 conf-only / depth-only hybrid compare（定位剩余差距归因）
- 按既定 A1 v2 路线补做了两条混合臂：`conf-only`（沿用 `pseudo_confidence_brpo_style_v2`，depth 回切 `joint_depth_target_v2`）与 `depth-only`（沿用 `pseudo_depth_target_brpo_style_v2`，confidence 回切 `joint_confidence_v2`），并与 `old A1`、`full brpo_style_v2` 在同一 `new T1 + summary_only` 协议下对齐 replay。
- 输出目录：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_a1_brpo_style_v2_hybrid_compare_e1/`；汇总文件：`compare_summary.json`。
- 结果：`old=24.187591/0.875591/0.080361`，`full_v2=24.187739/0.875594/0.080361`，`conf-only=24.187772/0.875593/0.080365`，`depth-only=24.187525/0.875592/0.080361`（PSNR/SSIM/LPIPS）。
- 结论：三条 v2 相关臂差异约在 1e-4 量级，当前 gap 不能归因成“只差 confidence”或“只差 depth target”单侧问题；下一步应转向 verifier/backend 与 target/stable-blend 的联合改动，再做同协议 compare。

### A1-BRPO-R1：BRPO-style v1 builder + consumer + fixed new T1 formal compare（near parity, not landed）
- 在 `verify proxy v1` 结论为 negative 后，A1 主线直接切到 **BRPO-style v1**：覆盖更新 A1 落地文档，不再把“只换 confidence 来源”的 proxy 当主方向；代码上新增 `pseudo_branch/brpo_v2_signal/pseudo_observation_brpo_style.py`，并在 `build_brpo_v2_signal_from_internal_cache.py` 中产出 `pseudo_confidence_brpo_style_v1 / pseudo_depth_target_brpo_style_v1 / pseudo_source_map_brpo_style_v1`；`run_pseudo_refinement_v2.py` 新增 `pseudo_observation_mode=brpo_style_v1`。
- 先做 smoke：1) 8-frame signal build 成功写出 `brpo_style_observation_summary`；2) 1-iter consumer smoke 成功跑通 `brpo_style_v1`。随后在固定 **new T1 + summary_only** 下完成三臂 compare：`oldA1_newT1_summary_only=24.187772 / 0.875594 / 0.080362`，`newA1_newT1_summary_only=24.149355 / 0.875578 / 0.079962`，`brpoStyleA1_newT1_summary_only=24.175377 / 0.875338 / 0.080344`。
- 发现：`brpo_style_v1` 相对 current new A1 已经明显回升（约 `+0.0260 PSNR`），说明把 A1 改成“shared `C_m` + verified projected depth target”这条方向是对的；但它相对 old A1 仍有轻微差距（约 `-0.0124 PSNR / -0.000256 SSIM / -0.000019 LPIPS`），说明当前第一版 BRPO-style builder 还没有完全超过 old A1 的工程稳定性。
- 结论：A1 的真正研究线应继续沿 **BRPO-style semantics** 推进，不应再回头打磨 `verify proxy v1`。当前 `brpo_style_v1` 的定位是：**方向正确、明显优于 current new A1、接近 parity，但还不是 landing 版本。**

### A1-V1：verifier-decoupled proxy 接线 + fixed new T1 formal compare（negative）
- 按新的 A1 规划新增 `pseudo_branch/brpo_v2_signal/pseudo_observation_verifier.py`，并在 `build_brpo_v2_signal_from_internal_cache.py` 中并行产出 `pseudo_confidence_verify_v1.npy / pseudo_valid_mask_verify_v1.npy / pseudo_verify_meta_v1.json`；`run_pseudo_refinement_v2.py` 同步新增 `pseudo_observation_mode=brpo_verify_v1`，consumer 在该模式下复用 `pseudo_depth_target_joint_v1`，但把 RGB/depth confidence 切到 verifier 输出。
- 先做 smoke：1) 8-frame signal build 成功写出 `verify_observation_summary`；2) 1-iter StageA smoke 成功跑通 `brpo_verify_v1` consumer，说明这次不是文档层设计，而是真实 wiring。随后在固定 **new T1 + summary_only** 下完成三臂 formal compare：`oldA1_newT1_summary_only=24.187430 / 0.875588 / 0.080364`，`newA1_newT1_summary_only=24.149516 / 0.875576 / 0.079962`，`verifyA1_newT1_summary_only=24.067703 / 0.873977 / 0.080955`。
- 发现：current new A1 在 fixed new T1 下仍没超过 old A1，而 verify A1 proxy 更差。这说明“仅把 confidence 从 candidate score 解耦出来”在当前保守 proxy 实现下并不足以带来正向收益。
- 结论：A1 的 proxy verifier decoupling 已完成了它该完成的实验职责——**它证明当前保守版 decoupling 不够强，不能 landing；但这不等于 full BRPO-style A1 被否定。**

## 2026-04-19

### B3-R1：deterministic participation controller 接通 + first compare（weak-negative）
- 按 B3 rewrite 方向，把 SPGM 的第一版动作从旧 `xyz_lr_scale` 的 post-backward grad probe，推进成 **pre-render deterministic participation control**：`manager_mode=deterministic_participation` 会在本轮统计 low-score candidate subset，并把 `participation_render_mask` 交给下一轮 pseudo render 消费。
- 执行过程中暴露并修了两处真 wiring bug：1) `gaussian_renderer.render(mask=...)` 的 masked 分支返回值签名和 unmasked 分支不一致；2) masked 分支返回的 `visibility_filter` 还是子集长度，导致 SPGM stats 读 full-length mask 时尺寸不一致。
- 先做 smoke，确认 `part_far≈0.875`、`drop_far` / `cand_far` history 已真实记录；随后在固定 `old A1 + new T1` 主线下完成 formal compare：`summary_only=24.185744 / 0.875419 / 0.080386`，`b3_det_participation=24.182511 / 0.875360 / 0.080516`。
- 结论：B3 的**方法对象已经切对**，不再只是旧 grad scaler；但当前 keep 配置（near=1.0, mid=0.9, far=0.75）首轮 compare 仍是 weak-negative，因此当前判断是 **no-go for landing**。

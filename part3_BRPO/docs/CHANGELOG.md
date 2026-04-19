# CHANGELOG.md - Part3 Stage1 过程记录

> **书写规范**：
> 1. 增量更新，倒序排列（最新在最上）
> 2. 每条提炼为 3-5 行：做了什么 → 发现什么 → 结论是什么
> 3. 不重复设计判断（DESIGN.md 负责），不重复状态细节（STATUS.md 负责）
> 4. 实验细节引用归档文档：`[参见 archived/2026-04-experiments/P2X_*.md]`

---

## 2026-04-19

### B3-R1：deterministic participation controller 接通 + first compare（weak-negative）
- 按 B3 rewrite 方向，把 SPGM 的第一版动作从旧 `xyz_lr_scale` 的 post-backward grad probe，推进成 **pre-render deterministic participation control**：`manager_mode=deterministic_participation` 会在本轮统计 low-score candidate subset，并把 `participation_render_mask` 交给下一轮 pseudo render 消费。
- 执行过程中暴露并修了两处真 wiring bug：1) `gaussian_renderer.render(mask=...)` 的 masked 分支返回值签名和 unmasked 分支不一致；2) masked 分支返回的 `visibility_filter` 还是子集长度，导致 SPGM stats 读 full-length mask 时尺寸不一致。
- 先做 smoke，确认 `part_far≈0.875`、`drop_far` / `cand_far` history 已真实记录；随后在固定 `old A1 + new T1` 主线下完成 formal compare：`summary_only=24.185744 / 0.875419 / 0.080386`，`b3_det_participation=24.182511 / 0.875360 / 0.080516`。
- 结论：B3 的**方法对象已经切对**，不再只是旧 grad scaler；但当前 keep 配置（near=1.0, mid=0.9, far=0.75）首轮 compare 仍是 weak-negative（约 `-0.00323 PSNR / -0.000059 SSIM / +0.000130 LPIPS`），因此当前判断是 **no-go for landing，下一轮应改成更保守的 participation keep**。

## 2026-04-18

### T1-R3：补齐 `old A1 + new T1`，完成 2×2 因子实验
- 单独补跑 `old_a1_new_topology`，把 observation × topology 的 2×2 因子补齐：`old A1 + old T1`、`old A1 + new T1`、`new A1 + old T1`、`new A1 + new T1`。目的不再是验证单条臂，而是判断主效应究竟来自 A1 还是 T1。
- 结果：`old A1 + new T1 = 24.185846 / 0.875423 / 0.080379`，不仅显著优于 `old A1 + old T1`（约 `+0.04843 PSNR / +0.00140 SSIM / -0.00076 LPIPS`），也优于 `new A1 + new T1`（约 `+0.05033 PSNR / +0.00009 SSIM / -0.00031 LPIPS`）。
- 发现：这说明 T1 的 topology 收益是跨 observation 稳定存在的，而 new A1 目前并不是主收益来源；至少在当前实现下，old A1 仍然是更稳的 observation 选择。
- 结论：当前最强候选主线从“new A1 + new T1”修正为“old A1 + new T1”。因此 B3 继续冻结——不是因为 B3 不重要，而是因为 observation/topology 主线刚刚收敛，现在再开新 B3 规划会让主因再次混淆。

### T1-R2：new A1 + new T1 confirmation / landing check
- 在独立运行根中补做三臂 confirmation：`old_a1_old_topology_ref`、`new_a1_old_topology`、`new_a1_new_topology`。目的不是再验证 T1 有没有效果，而是确认 `new A1 + new T1` 相对 old A1 强参考臂的 landing 是否已经足够稳定。
- 结果：`old_a1_old_topology_ref=24.137415 / 0.874025 / 0.081137`，`new_a1_old_topology=24.098963 / 0.874209 / 0.080708`，`new_a1_new_topology=24.135512 / 0.875333 / 0.080065`。因此，new topology 相对 new A1 old topology 仍有稳定正增益（约 `+0.03655 PSNR / +0.00112 SSIM / -0.00064 LPIPS`），但相对 old A1 强参考臂则是 **PSNR 近乎打平但略低、SSIM/LPIPS 更优**。
- 发现：这说明 T1 的 topology 收益已经被重复确认，但 `new A1 + new T1` 对 old A1 的接管结论还没有完全锁死；它更像“接近 landing，但仍处于 mixed confirmation”，而不是已经可以无条件宣布正式接管。
- 结论：当前继续做 `new A1 + new T1` 的 confirmation / landing，而**不启动新的 B3 规划**。B3 下一次应建立在更稳定的新主 topology 之上，否则会重新把因果关系搅混。

### T1-R1：old topology vs new topology apples-to-apples compare + orchestration landing
- 新增 `scripts/run_t1_topology_compare.py`，在同一 new A1 observation、同一 bounded StageB protocol、同一 repair A dense_keep 下重跑两臂：old topology（`joint_topology_mode=off`）与 new topology（`joint_topology_mode=brpo_joint_v1`）；runner 同时把 StageA.5 的角色显式降级成 `legacy_mainline` vs `optional_warmup_or_control`。
- 结果：old topology=`24.116956 / 0.874490 / 0.080566`，new topology=`24.149837 / 0.875580 / 0.079964`。new topology 相对 old topology 为 `+0.032881 PSNR / +0.001090 SSIM / -0.000601 LPIPS`；同时相对先前 old A1 winner 也有正增益（约 `+0.00736 PSNR / +0.00132 SSIM / -0.00112 LPIPS`）。
- 发现：这轮 compare 说明在 new A1 observation 下，真正的收益点不只是 object semantics，还包括 optimizer topology 本身；`joint_topology_mode=brpo_joint_v1` 不是命名切换，而是有效的结构改动。与此同时，StageA.5 已经可以被工程化地描述成可选 warmup / 对照，而不是默认主线必经阶段。
- 结论：当前 Re10k 最强候选从“old A1 support filter”推进到“new A1 + new T1”。下一步应做这条组合主线的 confirmation / landing，而不是回头继续补 old topology。

### A1-R2：new A1 rewrite StageB formal compare（control / old A1 / new A1）
- 按同一 canonical StageB protocol 重跑三臂：control=`RGB-first + depth-sidecar`，old A1=`joint_confidence_v2 + joint_depth_v2`，new A1=`pseudo_observation_mode=brpo_joint_v1`；三臂都接同一 StageA.5 handoff、同一 repair A dense_keep、同一 bounded schedule，并全部完成 replay eval。
- 结果：control=`24.079436 / 0.872832 / 0.082186`，old A1=`24.142473 / 0.874256 / 0.081089`，new A1=`24.116298 / 0.874483 / 0.080569`。new A1 相对 control 为 `+0.036862 PSNR / +0.001651 SSIM / -0.001617 LPIPS`；相对 old A1 为 `-0.026175 PSNR / +0.000227 SSIM / -0.000519 LPIPS`。
- 发现：new A1 的 consumer lock 是真实生效的，`pseudo_observation_mode_effective=brpo_joint_v1`，有效输入固定落到 `pseudo_confidence_joint_v1 / pseudo_depth_target_joint_v1 / pseudo_source_map_joint_v1`；因此这次不是旧 A1 换壳。与此同时，new A1 仍停留在约 2% 量级的 sparse supervision，没有单靠 observation rewrite 就压过 old A1 的 PSNR。
- 结论：A1 rewrite 已经完成并给出 replay 级正信号——它明确优于 control，但在当前旧 topology 下还没有全面取代 old A1。下一步主线应切到 T1 joint topology rewrite，而不是继续在 A1 上补小规则。

### A1-R：BRPO-style joint observation rewrite（builder + consumer lock）
- 新增 `pseudo_branch/brpo_v2_signal/joint_observation.py`，在 `build_brpo_v2_signal_from_internal_cache.py` 中并行产出 `pseudo_*_joint_v1` observation bundle，并保留 old A1 artifacts 作为对照臂。
- 在 `run_pseudo_refinement_v2.py` 增加 `--pseudo_observation_mode brpo_joint_v1`，并完成 consumer 锁定：该模式下统一读取 `pseudo_confidence_joint_v1` / `pseudo_depth_target_joint_v1` / `pseudo_source_map_joint_v1`，不再由旧 mask/target 旋钮主导。
- 发现：第一版 candidate scoring 过宽会把 valid 区域误扩到全图，已回收为 support-seeded scoring；8-frame smoke 下 valid 覆盖回到 sparse support 区间，且 `old_min_rule_diff_ratio` 与 `new_depth_vs_old_like_diff_ratio` 均非零。
- 结论：A1 已从“support filter 修补”切换到“observation object 重写”语义；下一步是同 protocol 的 StageB formal compare（control / old A1 / new A1）。

### B3-2：repair A 40iter short compare + replay
- 按 B3 文档建议，以 repair A 为锚点完成两臂 short compare：control=`summary_only`，study=`xyz_lr_scale`；随后按既有 Re10k internal replay 口径对两臂都做 replay eval。
- 发现：机制上 `xyz_lr_scale` 确实真实生效（study 的 `_xyz` grad 更低），但 40iter compare 没形成训练侧正收益，而且 replay 相对 control 小幅退化（约 `-0.0129 PSNR / -0.00031 SSIM / +0.00018 LPIPS`）。
- 结论：B3-1 当前参数设定判为 weak-negative / no-go；先不顺推 opacity decay，也不把当前 deterministic state action 升级为新默认方向。

### B3-1：xyz deterministic lr scaling plumbing + smoke
- 在 `manager.py` 中新增 `manager_mode=xyz_lr_scale`，把 B2 的 cluster-level diagnostics 变成真实 state action：在 `optimizer.step()` 前对 `_xyz.grad` 再乘一层温和的 per-Gaussian state scale；同时补了 config / history / logging。
- 发现：repair A 1-iter smoke 下 action 已真实生效，`spgm_state_action_applied=true`，且 `state_grad_norm_xyz` 从 `0.1446` 下降到 `0.1336`；三层 cluster 的 effective scale 也已经拉开（near≈0.884 / mid≈0.865 / far≈0.826）。
- 结论：B3 第一段 wiring 成立，SPGM 已经第一次真实影响 Gaussian state behavior；下一步应先做 repair A 锚点 short compare，再决定是否加 very mild opacity decay。

### B2-2：cluster-level manager diagnostics + compatibility smoke
- 在 `manager.py` 中把 B2 从“全局均值 diagnostics”补到 cluster-level：新增 near/mid/far 的 `state_score mean/p50` 与 summary-only `lr_scale candidate`，并让 candidate count 不再依赖 selector keep 差值。
- 发现：repair A dense_keep 下 manager diagnostics 终于有了非平凡信号（例如 far cluster `state_score_p50≈0.452`、`lr_scale_far≈0.587`、`candidate_count_far=4591`）；同时 selector-first 1-iter smoke 仍保持 `selected_ratio < active_ratio`，说明旧 selector 路径没有被 B2 改坏。
- 结论：B2 已从“分数接线”推进到“manager 可消费的 scene-aware diagnostics 完整版”；下一步可以进入 B3，把这些 diagnostics 变成 deterministic state action。

### B2：scene-aware density proxy + state_score 第一版接线
- 在 `stats.py` 中把统计从 accepted pseudo subset 扩到 current train window summary，引入 `population_support_count` / `struct_density_proxy`；在 `score.py` 中显式输出 `weight_score / ranking_score / state_score`。
- 发现：repair A smoke 下 `state_score_p50` 与 `ranking_score_p50` 已经分离，不再只是 selector ranking 的别名；`struct_density` mode 的 1-iter smoke 也能真实跑通。
- 结论：B2 第一版 diagnostics 已接通，后续可以继续往 B3 的 deterministic state action 推进，但当前还不启用真实 state edit。

### B1：SPGM manager shell 落地
- 新增 `spgm/manager.py`，并把 live call chain 从单层 `stats -> score -> grad weight -> step` 拆成 `stats -> score -> update policy -> state management -> step`。
- 发现：StageB smoke 已经真实打印 `manager=summary_only`，history 中也能看到 `spgm_manager_mode_effective / spgm_state_*` 字段，而且 `state_action_applied=false`。
- 结论：B1 结构层已经成立，默认行为保持 summary-only 兼容态，B2/B3 不必再继续把 manager 逻辑挤回 `run_pseudo_refinement_v2.py` 或 `policy.py`。

### P2-T：selector-first confirmation / precision sweep
- 固定 canonical StageB protocol 与 repair A control，只扫 far-only `0.88 / 0.90-repeat / 0.92 / 0.95` 微窗口。
- 发现：selector-first 真实停留在 practical-parity 区，但没有稳定越过 repair A；值得保留的只剩 `0.90~0.92` 极窄窗口。
- 结论：selector-first 保留为 near-parity 参考臂，不升级为新 anchor；主线从这一步正式转向 A1 [参见 `docs/P2T_selector_confirmation_precision_compare_20260418.md`]。

### A1：unified RGB-D joint confidence StageB formal compare
- 新增 `joint_confidence_v2 / joint_depth_v2` builder+consumer 路径，并补 generic depth path 对 `depth_confidence_mask` 的真实消费；基于现有 Re10k `signal_v2` sidecar 生成 A1 joint root。
- 在 canonical StageB + repair A dense_keep 下，A1 joint arm 为 `24.031512 / 0.872048 / 0.081744`，control sidecar arm 为 `24.001931 / 0.871322 / 0.082440`。
- 发现：A1 的 mask coverage 基本不变（约 `1.957%`），但 verified depth support 从约 `4.13%` 收缩到约 `1.96%`、render fallback 几乎归零；收益来自 unified trusted support，而不是更大 coverage。
- 结论：A1 是 Re10k 上第一条明确优于 sidecar control、并在 PSNR / LPIPS 上超过 canonical baseline 的 observation 正信号；当前主线进入 A1 confirmation / landing，A2 退居后续 widening 步骤 [参见 `docs/A1_joint_confidence_stageb_compare_20260418.md`]。

### A1+A2：geometry-constrained expand 三臂 compare
- 新增 support_expand.py 模块，实现双侧 geometry-supported expansion；builder 支持 --use-a2-expand 开关。
- 三臂 compare：control 24.106/0.8734/0.08145，A1 24.120/0.8736/0.08146，A1+A2 23.834/0.8683/0.08295。
- 发现：coverage 从 1.96% 扩到 6.05%，但 A1+A2 相比 A1 下降 -0.286 PSNR。
- 结论：A2 widening 策略失败，主线推进 B1/B2/B3 [参见 docs/A1_joint_confidence_stageb_compare_20260418.md section 8]。

---

## 2026-04-17

### DL3DV：Phase G baseline + SPGM bring-up
- 基于 canonical DL3DV chain（internal cache / signal-aware selection / canonical prepare root / signal_v2）完成首轮 StageA → StageA.5 → bounded StageB baseline → SPGM repair A，并统一 replay。
- 发现：当前 DL3DV pseudo_cache 不含 `target_depth_for_refine_v2`，所以首轮 refine 采用 `signal_pipeline=brpo_v2 + brpo_v2_raw`，depth target 回退到 canonical `target_depth_for_refine`；修正后链路可稳定跑通。
- 结论：DL3DV case bring-up 已完成，`canonical_baseline_post40_lr03_120` 首轮略优于 `spgm_repair_a_keep111_eta0_wf025`，selector-first 先不启动 [参见 `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_dl3dv_phaseg_baseline_spgm_e1/summary.json`]。

### P2-S：support_blend far-keep follow-up compare
- 固定 ranking=`support_blend`，只扫 far keep ratio (`0.90/0.85/0.80`)。
- 发现：`far=0.90` 已逼近 repair A parity（`+0.000069 PSNR`）。
- 结论：下一步围绕 `0.90` 做窄范围确认，不回强 selector [参见 archived/2026-04-experiments/P2S_*.md]。

### P2-R：score/ranking repair compare
- 拆分 `ranking_score` 与 `weighting_score`，新增 `support_blend` ranking。
- 发现：support-aware ranking 有小幅回升，但仍低于 repair A。
- 结论：ranking 方向有效，但不完全解决问题 [参见 archived/2026-04-experiments/P2R_*.md]。

### P2-O：selector-first formal compare
- 以 repair A 为 control，比较 selector S1/S2。
- 发现：selector 越强 replay 越差（误删有用更新）。
- 结论：当前 ranking 下 selector 会伤 replay，需先修 ranking [参见 archived/2026-04-experiments/P2O_*.md]。

### P2-N：selector-first plumbing smoke
- 实现 `selector_quantile` policy，selector 真实生效。
- 发现：`selected_ratio < active_ratio`，Gaussian 子集被缩小。
- 结论：plumbing 完成，下一步正式 compare [参见 archived/2026-04-experiments/P2N_*.md]。

### P2-M：conservative deterministic repair compare
- 更温和 suppress（A/B 两臂），grad weight回升。
- 发现：repair A 优于原始 SPGM，但仍低于 canonical baseline。
- 结论：suppress 过强是问题，但还需 selector-first [参见 archived/2026-04-experiments/P2M_*.md]。

### P2-L：canonical StageB formal compare
- 对齐 protocol 后，SPGM 仍低于 baseline。
- 发现：两臂 rejection 一致，grad weight 下降是主变化。
- 结论：protocol drift 不是主解释，需进入 repair [参见 archived/2026-04-experiments/P2L_*.md]。

### P2-K：forensic audit + code hygiene
- 审查 SPGM v1 实现，发现 entropy 归一化问题。
- 修复：entropy 按 `log(B)` 归一化，density_mode 真实接通。
- 结论：wiring 正确，但原始 compare protocol 与 baseline 漂离 [参见 archived/2026-04-experiments/P2K_*.md]。

---

## 2026-04-16

### P2-J：bounded StageB schedule compare
- 比较 `post40_lr03_120` / `post80_lr03_120` / `post80_lr03_real05_120`。
- 发现：`post40_lr03_120` 最佳，能把 120iter cliff 拉回。
- 结论：StageB 已有 bounded baseline，不再深调 schedule [参见 archived/2026-04-experiments/P2J_*.md]。

### P2-I：StageB 回落窗口定位
- 固定 gated winner，扫 `20/40/80/120` iter。
- 发现：PSNR 最佳在 40iter，80→120 出现 cliff。
- 结论：StageB 不是完全没价值，但需 bounded schedule [参见 archived/2026-04-experiments/P2I_*.md]。

### P2-H：RGB-only v2 StageB-120iter verify
- gated 仍真实工作（96/120 rejection），但两臂都低于 baseline。
- 发现：gating 不能阻止 120iter regression。
- 结论：需先修 StageB 后段 schedule [参见 archived/2026-04-experiments/P2H_*.md]。

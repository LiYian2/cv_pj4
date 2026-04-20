# A1 Exact BRPO Target Compare E1

## Protocol
- Fixed topology: `joint_topology_mode=brpo_joint_v1`
- Fixed StageB recipe: `summary_only`
- Signal root: `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_a1_full_brpo_target_signal_full`
- Compare root: `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_a1_exact_brpo_target_compare_e1`
- New exact arm: `exact_brpo_full_target_v1`
- Important semantic constraint: this arm is forced to run with `--stageA_depth_loss_mode legacy`, so RGB / depth both consume the same shared `C_m` instead of source-aware seed/dense/fallback tiers.

## Arms
1. `oldA1_newT1_summary_only`: old support-filter control.
2. `newA1_newT1_summary_only`: current candidate-built joint observation.
3. `hybridBrpoCmGeo_v1_newT1_summary_only`: geometry-gated hybrid branch.
4. `exactBrpoCm_oldTarget_v1_newT1_summary_only`: strict `C_m` + old target contract.
5. `exactBrpoFullTarget_v1_newT1_summary_only`: strict `C_m` + exact BRPO-style target-side proxy + shared-`C_m` depth loss.
6. `exactBrpoCm_hybridTarget_v1_newT1_summary_only`: strict `C_m` + hybrid direct target under current source-aware consumer.
7. `exactBrpoCm_stableTarget_v1_newT1_summary_only`: strict `C_m` + stable-target blend contract.

## Metrics
- `oldA1`: `24.1877374861 / 0.8755949248 / 0.0803590433`
- `exactBrpoCm_oldTarget_v1`: `24.1874953941 / 0.8755869766 / 0.0803696718`
- `exactBrpoFullTarget_v1`: `24.1744880747 / 0.8752853330 / 0.0803574205`
- `exactBrpoCm_hybridTarget_v1`: `24.1743486546 / 0.8752873319 / 0.0803571455`
- `exactBrpoCm_stableTarget_v1`: `24.1752181159 / 0.8753429188 / 0.0803407980`

## Key deltas
- `exact full target - old`: `-0.0132494 PSNR`
- `exact full target - exact old target`: `-0.0130073 PSNR`
- `exact full target - hybrid geometry-gated branch`: `-0.0008181 PSNR`
- `exact full target - exact hybrid target`: `+0.0001394 PSNR`
- `exact stable target - exact full target`: `+0.0007300 PSNR`

## Builder-side observations
From `/20260420_a1_full_brpo_target_signal_full/summary.json`:
- `exact_brpo_full_target_v1` keeps the same exact `C_m` coverage as `exact_brpo_cm_old_target_v1` (`cm_nonzero_ratio ≈ 0.019582`).
- But its target-side stats are almost identical to `exact_brpo_cm_hybrid_target_v1`:
  - `depth_target_nonzero_ratio ≈ 0.01921`
  - `source_left/right/both ≈ 0.006281 / 0.003873 / 0.009057`
- Frame-level arrays are also effectively the same as the hybrid-target arm: `pseudo_depth_target_exact_brpo_full_target_v1.npy` and `pseudo_depth_target_exact_brpo_cm_hybrid_target_v1.npy` differ only at floating-point noise level (max abs diff about `1.4e-06` on the checked frame), and source maps are identical.

## Grounded interpretation
1. This run answers the user’s exact question directly: under the current proxy `I_t^{fix}` / projected-depth backend, switching all the way to the BRPO-style target-side semantics does **not** recover the old-A1 gap.
2. The new exact arm lands at `24.174488`, almost the same as the hybrid-target arm (`24.174349`) and still about `-0.01325 PSNR` below old A1.
3. Therefore the remaining gap is not caused mainly by the source-aware consumer tiers. Removing those tiers and forcing shared-`C_m` depth loss barely changes the result.
4. The current target-side proxy itself is the limit: under today’s fused pseudo-frame / projected-depth construction, the exact-BRPO target contract is simply too close to the existing hybrid direct target to change replay materially.
5. In other words: Layer B exactness has now been tested more directly, and the bottleneck moves one level upstream — the current proxy pseudo-frame / verifier / projected-depth backend is not rich enough for exact BRPO target-side semantics to win by itself.

## Next-step implication
Do not spend another round only polishing fallback weights or consumer-side source tiers. The next meaningful step should be upstream of this exact target contract:
1. either improve the proxy `I_t^{fix}` / correspondence-verifier backend so exact BRPO target-side semantics see a genuinely different supervision field;
2. or explicitly accept that the current project objective is replay-first engineering rather than exact BRPO, and continue with the best-performing hybrid/stable target contract under that label.

# MIDPOINT8_M5_STAGEA_A5_B_EXEC_REPORT_20260413

## 0. Scope

This document records the strict midpoint-8 + M5 + source-aware execution that was requested after clarifying that the target protocol is:
1. first materialize midpoint pseudo-cache to `target_depth_for_refine_v2.npy` (M5);
2. then run `StageA (source_aware)`;
3. then run `StageA.5 (source_aware + xyz_opacity)`;
4. then run `StageB (source_aware conservative)`;
5. replay each stage on the 270-frame `after_opt` internal-eval split;
6. compare all stages against part2 `after_opt` baseline.

This report is intentionally separate from earlier midpoint A.5 notes because earlier positive midpoint records were not the same protocol: they mixed different upstream / mask / handoff assumptions and therefore cannot be treated as the same experiment.

## 1. Fixed experiment inputs

- repo root: `/home/bzhang512/CV_Project/part3_BRPO`
- git commit: `16788cd9830c463bb379e3b2f84d993e45ff64ac`
- python env: `/home/bzhang512/miniconda3/envs/s3po-gs/bin/python`
- canonical run root: `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_midpoint8_m5_fullpipeline`
- part2 baseline root: `/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58`
- baseline replay summary: `/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/psnr/after_opt/final_result.json`
- baseline PLY: `/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache/after_opt/point_cloud/point_cloud.ply`
- internal replay cache: `/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache`
- midpoint pseudo cache: `/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_prepare/re10k1__internal_afteropt__midpoint_proto_v1/pseudo_cache`
- midpoint frame ids: `[17, 51, 86, 121, 156, 190, 225, 260]`
- sparse real-anchor manifest: `/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/split_manifest.json`
- sparse real-anchor rgb dir: `/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/rgb`

## 2. Executed protocol

### 2.1 Upstream (Phase 0 + 1)

Materialized M5 depth targets in-place for all 8 midpoint samples using:
- `scripts/materialize_m5_depth_targets.py`
- output summary: `upstream_m5/materialize_m5_summary.json`

Generated per-sample M5 artifacts:
- `target_depth_for_refine_v2.npy`
- `target_depth_dense_source_map.npy`
- `depth_correction_seed.npy`
- `depth_correction_dense.npy`
- `depth_seed_valid_mask.npy`
- `depth_dense_valid_mask.npy`
- `depth_densify_meta.json`

### 2.2 StageA run

Command semantics:
- `stage_mode=stageA`
- `stageA_iters=300`
- `confidence_mask_source=brpo`
- `stageA_mask_mode=train_mask`
- `stageA_target_depth_mode=target_depth_for_refine_v2`
- `stageA_depth_loss_mode=source_aware`
- `num_pseudo_views=8`

Output:
- `stageA/`
- replay: `replay/stageA/`

### 2.3 StageA.5 run

Command semantics:
- `stage_mode=stageA5`
- `stageA_iters=300`
- same source-aware / M5 input protocol as StageA
- `stageA5_trainable_params=xyz_opacity`
- `num_pseudo_views=8`

Output:
- `stageA5/`
- replay: `replay/stageA5/`

### 2.4 StageB run

Command semantics:
- warm-start PLY = `stageA5/refined_gaussians.ply`
- `stage_mode=stageB`
- `stageA_iters=0`
- `stageB_iters=120`
- same source-aware / M5 input protocol as StageA
- `stageA5_trainable_params=xyz_opacity`
- `lambda_real=1.0`
- `lambda_pseudo=1.0`
- `num_real_views=2`
- `num_pseudo_views=4`
- real branch from sparse train manifest/rgb

Output:
- `stageB/`
- replay: `replay/stageB/`

## 3. Replay comparison

Baseline (`part2 after_opt`):
- PSNR = 23.94890858685529
- SSIM = 0.8734854221343994
- LPIPS = 0.0787798319839769

### 3.1 Unified table

| Run | PSNR | SSIM | LPIPS | ΔPSNR vs base | ΔSSIM vs base | ΔLPIPS vs base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline after_opt | 23.9489085869 | 0.8734854221 | 0.0787798320 | 0.000000 | 0.000000 | 0.000000 |
| StageA (M5 source-aware) | 23.9489114761 | 0.8734854696 | 0.0787797966 | +0.0000029 | +0.0000000 | -0.0000000 |
| StageA.5 (xyz_opacity) | 23.8382467835 | 0.8655473773 | 0.0875594761 | -0.1106618 | -0.0079380 | +0.0087796 |
| StageB120 conservative | 23.1610396279 | 0.8504478700 | 0.0978147150 | -0.7878690 | -0.0230376 | +0.0190349 |

### 3.2 Immediate interpretation

1. StageA replay is numerically identical to baseline.
2. StageA.5 lowers pseudo-side training loss a lot, but replay clearly regresses.
3. StageB120 regresses further and does not recover StageA.5 damage.
4. Therefore this strict midpoint-8 + M5 + source-aware pipeline is currently negative in downstream replay terms.

## 4. True camera-state movement vs misleading residual summary

A crucial debugging observation is that `stageA_history.json -> final_delta_summary` reports the residual parameter norms after `apply_pose_residual_()` zeroes them. That field can look like “camera did not move” even when `vp.R / vp.T` actually changed.

True pose movement measured from `pseudo_camera_states_init.json -> pseudo_camera_states_final.json`:

| Stage | mean translation Δ | max translation Δ | mean abs-pose norm | max abs-pose norm |
| --- | ---: | ---: | ---: | ---: |
| StageA | 0.00476 | 0.03390 | 0.00493 | 0.03452 |
| StageA.5 | 0.00670 | 0.02988 | 0.00676 | 0.03028 |
| StageB120 | 0.01302 | 0.02815 | 0.01306 | 0.02836 |

So the correct conclusion is not “camera state exactly never moved”. The correct conclusion is:
- camera pose did move, but only by a small amount relative to scene scale (`mean_stageA_scene_scale ≈ 3.1961`);
- the built-in residual summary under-reports this because it only logs post-fold-back residual tensors.

## 5. Why StageA replay does not change even though StageA loss / pose move slightly

This is the most important structural finding in this run.

### 5.1 StageA does not modify Gaussians

In `run_pseudo_refinement_v2.py`:
- `stage_mode=stageA` builds only the pseudo-view optimizer;
- Gaussian optimizer is `None`;
- the output PLY is just the original Gaussian model saved back out.

This is confirmed by file hash:
- input after_opt PLY SHA256 = `28b68975291b082f383e5cbb9685895bc4e478620211765b304a6b8464946580`
- StageA exported PLY SHA256 = `28b68975291b082f383e5cbb9685895bc4e478620211765b304a6b8464946580`

They are byte-identical.

### 5.2 Replay only consumes the PLY, not StageA pseudo camera states

`replay_internal_eval.py` replays using:
- fixed internal-eval camera states from part2 internal cache;
- the supplied PLY.

Therefore, if StageA only updates pseudo pose/exposure but leaves Gaussians unchanged, replay on the 270 internal frames must remain unchanged.

### 5.3 Consequence

Using 270-frame replay as a StageA gate is structurally invalid under the current implementation. StageA can only matter if:
1. its refined pseudo camera states are fed into a later stage that updates Gaussians; or
2. evaluation explicitly uses the refined pseudo camera states themselves.

Under the current code, neither happens.

## 6. Why StageA signal is weak in this strict midpoint-M5 run

### 6.1 Upstream geometric support is much weaker than expected on this midpoint set

From `materialize_m5_summary.json`:
- mean train-mask candidate region = 0.1870
- mean seed ratio = 0.0150
- mean densified ratio = 0.0199
- mean render-fallback ratio = 0.9651

More importantly, effective non-fallback geometric support inside the train-mask is only about:
- `(seed + dense) / train_mask ≈ 0.1865`

So only about 18.7% of the train-mask has non-fallback M5 depth on this midpoint-8 set.
This is very different from the previously assumed “train-mask 内大部分已有非-fallback depth” picture.

### 6.2 What this means for StageA

StageA optimizes only camera deltas + exposure. But in this run:
- RGB supervision exists on ~18.7% of the image;
- source-aware depth supervision is only active on ~3.5% of the full image, and only ~18.7% of the train-mask;
- fallback depth exists almost everywhere else but `lambda_depth_fallback=0`, so it is ignored.

So the actual pose-driving geometric signal is sparse and local.
This makes small pseudo-side loss improvements possible, but it makes large, globally-correct pose updates unlikely.

### 6.3 Gradient exists, but leverage is limited

Single-step gradient diagnosis on this exact M5 midpoint setup shows:
- RGB -> rot grad ≈ `1.368e-01`, trans grad ≈ `5.489e-02`
- depth_total -> rot grad ≈ `8.715e-01`, trans grad ≈ `3.727e-01`
- depth_seed dominates depth gradient; depth_dense is present but smaller.

So the pose path is connected. The issue is not “no gradient reaches pose”.
The issue is that the available geometry signal is still too sparse / too local to create large useful camera updates.

## 7. Why A.5 and StageB reduce loss but worsen replay

### 7.1 StageA.5 is optimizing a local pseudo objective, not guaranteed downstream geometry

StageA.5 lowered pseudo-side losses strongly:
- `loss_depth`: `0.06791 -> 0.03833`
- `loss_depth_seed`: `0.05693 -> 0.03353`
- `loss_depth_dense`: `0.03137 -> 0.01370`

But the replay regressed:
- PSNR `-0.11066`
- SSIM `-0.00794`
- LPIPS `+0.00878`

This means the unlocked `xyz+opacity` parameters found a way to fit the pseudo supervision better without improving the 270-frame downstream rendering. In practice this is consistent with pseudo-view overfitting or local geometry/visibility compensation.

### 7.2 Real anchor in StageB is too weak / mismatched to recover the damage

Current StageB real branch uses:
- only 2 sparse real views per iteration;
- RGB loss only on real views (`get_loss_mapping_rgb`);
- no real depth anchor in this branch;
- no explicit full-scene geometric consistency term;
- only `xyz+opacity` unlocked on the Gaussian side.

So StageB is not equivalent to “full BRPO joint refinement with strong geometric anchoring”.
Instead it is a limited RGB real-anchor correction on top of an already pseudo-overfit geometry. That is not enough to pull the model back; in this run it made replay worse.

### 7.3 Stage handoff is structurally incomplete

Another important implementation issue:
- StageA.5 does not load StageA refined pseudo camera states;
- StageB does not load StageA.5 refined pseudo camera states;
- each run reloads pseudo viewpoints from pseudo cache and re-initializes camera deltas from zero.

So the current A -> A.5 -> B chain is not a true sequential pipeline in camera-state space.
It is only a PLY handoff for the Gaussian model (and even that only happens from A.5 -> B, not StageA -> A.5).

This means StageA currently does not actually initialize later stages in the intended way.
That is a structural issue, not just a hyperparameter issue.

## 8. Internal pseudo cache chain: what is engineering replacement vs what is materially different from the paper

### 8.1 What the paper says at high level

From BRPO paper Section 3:
1. bidirectional pseudo-frame restoration;
2. confidence mask inference via geometric correspondence verification;
3. scene perception Gaussian management (depth-density score, cluster-aware stochastic dropping);
4. two-stage optimization:
   - first pose deltas + exposure stabilization;
   - then joint optimization of Gaussians and poses with weighted RGB-D reconstruction loss;
5. a final appearance refinement weighted by the confidence map.

The paper explicitly describes joint updates of Gaussian parameters and camera poses, and explicitly includes a Gaussian scale regularizer term.
It also states `beta` is typically `0.95`, i.e. the RGB term is dominant in the joint reconstruction objective.

### 8.2 What our current implementation actually does

Current part3 pipeline is:
- internal pseudo cache generation / pack from part2 outputs;
- BRPO-style confidence mask consumption from pseudo cache;
- custom M3 / M5 depth target generation (`target_depth_for_refine`, `target_depth_for_refine_v2`) with seed/densify/fallback source maps;
- StageA: pseudo pose + exposure only;
- StageA.5: pseudo pose + exposure + limited Gaussian subset (`xyz` or `xyz_opacity` only);
- StageB: limited joint refine with sparse real RGB anchor + pseudo RGBD loss.

### 8.3 Engineering replacements that are reasonable but still not the same thing

These are not necessarily wrong, but they are real deviations:
1. We use part2 internal cache and packed pseudo cache as the supervision substrate, rather than the paper’s end-to-end pseudo-view restoration presentation.
2. We introduced M5 densified depth target generation (`target_depth_for_refine_v2`) with source-aware weighting. This is an engineering extension, not a paper-identical module.
3. We split depth loss into seed / dense / fallback source-aware terms. Again useful engineering, but not paper-identical.

### 8.4 Material differences that likely matter for optimization outcome

These are not superficial substitutions; they plausibly change behavior materially:

1. StageA replay gate mismatch.
- Paper stage logic assumes pose stabilization feeds the later joint stage.
- Our current evaluation uses replay on a PLY that StageA never changes.
- So StageA’s measured downstream effect is structurally suppressed.

2. Stage handoff is incomplete.
- Paper logic is sequential: stabilized pose informs later joint optimization.
- Our code reinitializes pseudo viewpoints at each stage and does not import prior stage camera states.
- This breaks the intended effect of StageA as a true initializer.

3. Joint optimization is much narrower than the paper.
- Paper describes joint optimization of Gaussian parameters and poses with RGB-D loss plus scale regularization.
- Our A.5 / StageB only unlock `xyz` or `xyz+opacity`; no scaling, no rotation, no SH/color refinement in the joint stage itself.
- This changes the degrees of freedom drastically and encourages different local minima.

4. Missing paper-side scene perception Gaussian management in the refine loop.
- The paper has explicit depth-density scoring and cluster-aware drop probability.
- Current part3 refine loop does not implement this dynamic Gaussian management in StageA.5 / StageB.
- So one of the paper’s main anti-floating / anti-instability mechanisms is absent.

5. Real-anchor StageB is not the same as the paper’s joint pseudo confidence-weighted RGB-D optimization.
- Our real branch is sparse RGB only.
- No real-depth anchor and no full BRPO-style scene-perception management is present.
- This makes StageB less geometrically grounded than the paper’s description suggests.

6. Final appearance refinement stage from the paper is not the same as current A.5/B.
- The paper has a dedicated confidence-weighted appearance refinement with SSIM term.
- Current A.5/B does not reproduce this exact final stage.

## 9. Parameter problem or structure problem?

Current evidence says: both exist, but structure dominates.

### 9.1 Structural problems (primary)

1. StageA replay is structurally non-informative.
2. Stage handoff is incomplete: camera states are not carried from StageA -> A.5 -> B.
3. StageB real anchor is too weak and too appearance-only to repair pseudo-overfit geometry.
4. The current refine loop omits paper-important scene perception Gaussian management.
5. The current joint stage unlocks too narrow a Gaussian subset compared with the paper’s intended full joint optimization.

### 9.2 Signal-strength problems (secondary but real)

1. On this midpoint-8 set, M5 non-fallback geometric support is much weaker than expected.
2. Because fallback depth is ignored, most of the train-mask contributes RGB but not geometric supervision.
3. That makes pose updates small and fragile even though gradients are connected.

### 9.3 Parameter issues (present, but not the first-order explanation)

Parameters can still matter, for example:
- `beta_rgb=0.7` vs the paper’s described `beta≈0.95`;
- current LR choices for xyz / opacity / camera;
- `lambda_real : lambda_pseudo` ratio;
- number of real views per StageB step.

But parameter tuning alone will not fix:
- StageA replay insensitivity;
- missing stage-to-stage camera-state handoff;
- missing scene-perception Gaussian management;
- the large mismatch between current A.5/B micro-joint scope and the paper’s joint refinement design.

## 10. Per-subsystem diagnosis

### 10.1 Camera / pose optimization

What works:
- gradients do reach pose;
- true camera states do move a little.

What is wrong:
- movement is small relative to scene scale and likely insufficient to alter downstream geometry;
- StageA camera improvements are not propagated into later stages;
- built-in summary makes motion look zero because it reports post-fold-back residual tensors.

### 10.2 Geometry optimization

What works:
- `xyz+opacity` clearly changes the Gaussian model and lowers pseudo-side loss.

What is wrong:
- the unlocked parameter subset is narrow and may solve pseudo supervision via local geometry/visibility hacks;
- no paper-style scene-perception Gaussian management is present during refine;
- StageB real anchor is not strong enough to reject those local minima.

### 10.3 RGB / appearance optimization

What works:
- RGB and exposure losses are active and easy to optimize.

What is wrong:
- many masked pixels have RGB supervision without equally strong geometric support;
- this allows appearance-friendly but geometry-harmful updates;
- in StageB the real branch is RGB-only, which may further bias optimization toward appearance matching on a few sparse views.

## 11. Practical next-step directions suggested by this analysis

Priority should be structure-first, not another blind parameter scan.

1. Fix stage handoff.
- Add ability to load prior-stage pseudo camera states into A.5 / B.
- Otherwise StageA remains disconnected from the rest of the pipeline.

2. Stop using replay as a StageA success metric.
- For StageA, evaluate camera-state change / pseudo-side reprojection / downstream warm-start effect instead.

3. Re-check midpoint M5 support quality before more refine scans.
- This run shows only ~18.7% non-fallback support within train-mask on midpoint-8.
- If this remains true, stronger pose gains are unlikely without better geometry support.

4. Strengthen StageB’s geometric anchor if keeping the current pipeline style.
- real RGB only is too weak;
- either add stronger geometric anchoring or more faithful paper-side joint mechanisms.

5. Distinguish paper-faithful upgrades from pure parameter retuning.
- scene-perception Gaussian management during refine;
- true sequential pose-to-joint handoff;
- broader joint Gaussian parameter set and/or the paper’s appearance-refine stage.

## 12. Bottom line

For the strict requested protocol, the result is negative.

The most important reason is not simply “bad hyperparameters”. The deeper issue is that the current part3 pipeline has a structural disconnect:
- StageA is evaluated by a metric it cannot change;
- StageA pose results are not fed into later stages;
- A.5/B can lower pseudo-side losses but are not sufficiently constrained to convert that into better downstream replay.

So the current evidence points to:
- some signal weakness in the midpoint M5 pseudo cache;
- but more importantly, a structural mismatch between the current implementation and the paper’s intended sequential / joint optimization design.


## 13. P0 repair validation + sequential rerun follow-up（2026-04-13 夜）

After fixing stage handoff and StageA reporting, a repaired sequential rerun was executed under:
- `repair_seq_rerun/stageA80`
- `repair_seq_rerun/stageA5_80_from_stageA`
- `repair_seq_rerun/stageB120_from_stageA5`

### 13.1 What was verified structurally

1. `StageA final -> StageA.5 init` matched exactly:
- max pose abs diff = `0.0`
- max exposure diff = `0.0`

2. `StageA.5 final -> StageB init` matched exactly:
- max pose abs diff = `0.0`
- max exposure diff = `0.0`

3. Therefore the old defect is fixed:
- the pipeline is no longer “camera-side reinit every stage”;
- it is now a real sequential `StageA -> A.5 -> B` chain in pseudo-camera-state space.

### 13.2 Sequential rerun metrics

Against part2 baseline:
- StageA80 replay still remains structurally non-informative (StageA still does not modify PLY)
- StageA.5(80, from StageA) replay:
  - PSNR = `23.8929647658`
  - SSIM = `0.8700711683`
  - LPIPS = `0.0817763670`
  - delta vs baseline = `-0.05594 / -0.00341 / +0.002997`
- StageB120(from StageA.5) replay:
  - PSNR = `23.1763532285`
  - SSIM = `0.8527056645`
  - LPIPS = `0.0942044391`
  - delta vs baseline = `-0.77256 / -0.02078 / +0.01542`

Relative to the old non-sequential strict run:
- A.5 improved modestly after handoff repair:
  - PSNR `23.83825 -> 23.89296`
  - SSIM `0.86555 -> 0.87007`
  - LPIPS `0.08756 -> 0.08178`
- StageB120 also improved modestly after handoff repair:
  - PSNR `23.16104 -> 23.17635`
  - SSIM `0.85045 -> 0.85271`
  - LPIPS `0.09781 -> 0.09420`

But both stages still remain worse than baseline.

### 13.3 Interpretation after repair

The repaired sequential rerun changes the diagnosis in an important but limited way:
1. The previous “stages are not really connected” critique was valid and is now fixed.
2. That fix produces measurable downstream improvement for A.5 and StageB relative to the broken-handoff run.
3. However, the improvement is not large enough to reverse the negative result.

So the updated diagnosis is:
- handoff breakage was a real structural bug and did matter;
- but it was not the only bottleneck;
- after fixing handoff, the pipeline is still limited by weak geometric signal and by the current narrow/underconstrained joint refinement design.

## 14. Retrospective: how to avoid missing this kind of bug in future experiments

This debugging cycle exposed a process problem, not just a code problem.

### 14.1 What went wrong in process

1. We initially talked about `A -> A.5 -> B` as if it were a true sequential pipeline before verifying the actual handoff code path.
2. We relied too much on stage naming and high-level intent, and not enough on verifying what each stage actually loads and saves.
3. We also used a StageA metric (270 replay) that could not reflect StageA behavior under the current implementation.

### 14.2 What should be verified earlier next time

Before trusting any staged experiment design, verify these four items explicitly:
1. Artifact continuity:
   - does stage N+1 actually load stage N outputs?
2. State continuity:
   - are latent/camera/optimizer states inherited or reinitialized?
3. Evaluation continuity:
   - does the chosen metric consume the artifact that the stage modifies?
4. Signal coverage:
   - how much of the supervised region contains real non-fallback geometry?

### 14.3 Practical verification checklist for future staged experiments

For future designs, especially multi-stage ML/reconstruction pipelines, do this before long runs:
1. Run a 0-iter or very short smoke for each stage.
2. Save init/final states.
3. Compare `stageN final` vs `stageN+1 init` exactly.
4. Hash the evaluation artifact to confirm whether that stage can change the metric.
5. Only then start medium/long experiments.

This should become the default validation discipline for future staged pipeline work.

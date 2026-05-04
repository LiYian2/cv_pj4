# Part3 BRPO refine forensic master log — 2026-04-25

## Purpose
This file is the running master log for the current refine forensic pass. It serves three roles:
1. answer the user’s current grounded questions about real/pseudo sampling, depth supervision, and mask semantics;
2. define the ordered diagnostic plan (Step 1–6);
3. record results incrementally so follow-up work can resume from this document directly.

## Scope and protocol identity
Current focus: understand why stronger pseudo supervision keeps hurting replay under the current Part3 BRPO refine pipeline.
Relevant compared routes already on disk:
- legacy old-best StageB120 replay (`20260413_stageB_conservative_gate`) — all-ones legacy fused mask consumer
- exact sparse masked route (`20260424_m3d_consumer_compare_stageB120_replay/sparse_current`)
- GT-RGB-only legacy route (`20260425_gt_rgb_only_stageB120_replay`)
- GT RGB + GT-image-derived depth legacy all-ones route (`20260425_gt_rgb_mast3rdepth_stageB120_replay`)
- GT RGB + GT-image-derived depth + dense3d exact masked route (`20260425_gt_rgb_gtdepth_dense3d_q070_exactmask_stageB120_replay`)

## Grounded answers to the user’s current three questions
### Q1. Why did Hermes say `2/4` if there are `9` real and `8` pseudo?
Grounded answer: there are indeed 9 real views available in the sparse-train manifest and 8 pseudo views loaded from the pseudo cache, but StageB does not use all of them each iteration. It samples a subset per iteration.
- Loaded pools:
  - real pool size effectively 9 (`loaded real sparse-train views: 9` in StageB logs)
  - pseudo pool size 8 (`Loaded 8 pseudo viewpoints` in StageA/StageB logs)
- Per-iteration sampling is controlled by saved args:
  - `num_real_views = 2`
  - `num_pseudo_views = 4`
- This is not guesswork; it is saved in multiple run histories including:
  - old best `20260413_stageB_conservative_gate/phase1_stageB120/stageA_history.json`
  - GT RGB-only `20260425_gt_rgb_only_stageB120_replay/stageA_history.json`
  - exact GT+dense3d `20260425_gt_rgb_gtdepth_dense3d_q070_exactmask_stageB120_replay/stageB120_replay/stageA_history.json`
- The live code path in `scripts/run_pseudo_refinement_v2.py` samples:
  - real indices at line ~1860 using `args.num_real_views`
  - pseudo indices at line ~1872 using `cfg.num_pseudo_views`
So: 9/8 are pool sizes; 2/4 are per-iteration sampled counts.

### Q2. If we ignore GT and only use dual projected depth, can the current direct depth loss still optimize usefully?
Grounded answer: yes, in principle it can help, but only in some regimes; current evidence says it is not categorically broken, but it becomes harmful when the pseudo branch gets too strong or too noisy.
Evidence already on disk:
- In the exact sparse route, turning off pseudo depth made replay worse:
  - sparse current ≈ 24.0035 PSNR
  - sparse RGB-only ≈ 23.9243 PSNR
  - delta ≈ -0.0792
  This means projected-depth supervision is not inherently unusable.
- In the dense q070 exact route, turning off pseudo depth barely changed replay:
  - q070 current ≈ 23.66595
  - q070 RGB-only ≈ 23.66449
  So depth was not rescuing the dense route either.
- In the old-best legacy route, pseudo-side depth loss is tiny at convergence:
  - old best StageB last10 mean depth loss ≈ 0.00593
  This suggests pseudo depth there behaves more like a weak regularizer than a dominant driver.
- In strong GT-like pseudo routes, pseudo depth becomes much larger and replay collapses:
  - GT RGB + depth + all-ones: last10 mean depth loss ≈ 0.509, replay much worse
  - GT RGB + depth + dense3d exact mask: last10 mean depth loss ≈ 0.162, replay even worse than the all-ones version
Interpretation: the direct depth loss can optimize when the target is weak/close enough and the pseudo branch does not dominate. The current failure is more likely a contract/strength issue than a proof that projected depth can never work.

### Q3. Would replacing BRPO’s discrete `1 / 0.5 / 0` mask with continuous weights be better?
Grounded answer: current evidence does not support making that the first fix.
What is already known:
- We already tested a `cm-only` consumer ablation in the exact route.
- In the current exports, `C_m > 0` and `valid_mask > 0` are effectively the same support set, so that ablation mainly removed `target_confidence`.
- Result: both sparse and dense got slightly worse, not better.
  - sparse cm-only delta PSNR ≈ -0.00272
  - q070 cm-only delta PSNR ≈ -0.00923
What this means:
- Continuous weighting via `target_confidence` is not the main cause of the regression; if anything, it is slightly protective.
- Replacing the discrete BRPO levels entirely with a fully continuous mask would also erase the only explicit distinction between `both-supported` and `single-supported` pixels, which is one of the few semantically meaningful parts left in the current exact route.
Current judgement:
- Do not make `continuous mask instead of discrete C_m` the first repair direction.
- First localize whether the main problem is branch dominance, target residual scale, pose-vs-gaussian responsibility, or weak real-anchor balance.

## Current working hypothesis
The most likely issue is not simply “mask bad” or “depth bad”. The likely failure mode is:
- the pseudo branch becomes too strong relative to the real branch,
- depth is injected with a very direct L1 objective,
- exact masked supervision is sparse but still strong enough to drag global Gaussians,
- and the current consumer has no sufficiently strong real-side geometric anchor to absorb that safely.
In short: once pseudo supervision becomes stronger/cleaner, current refine does not exploit it robustly; it gets pushed around by it.

## Ordered diagnostic plan
### Step 1 — branch-wise gradient / residual audit
Goal: measure how much `real_rgb`, `pseudo_rgb`, and `pseudo_depth` actually contribute to Gaussian and pose gradients under representative runs.
Deliverable: branch-wise grad norms and initial residual magnitudes.
Status: pending.

### Step 2 — replay vs iteration curve
Goal: determine whether replay starts degrading immediately or only after longer optimization.
Planned checkpoints: iter `0 / 20 / 40 / 80 / 120` where feasible.
Deliverable: replay metrics across checkpoints for representative bad/good routes.
Status: pending.

### Step 3 — target residual audit before optimization
Goal: compare `render -> target` discrepancy for RGB and depth before StageB optimization, and split by support type (`both / single / invalid`).
Deliverable: per-frame and aggregated residual stats.
Status: pending.

### Step 4 — responsibility split
Goal: identify whether the damage mainly comes from pseudo updates on camera pose, Gaussian parameters, RGB branch, or depth branch.
Planned minimal axes:
- pose/exposure-only vs gaussian-only where feasible,
- RGB-only vs RGB+depth where relevant,
- additional short targeted splits if the first audit already localizes the problem.
Status: pending.

### Step 5 — real-anchor weakness check
Goal: test whether the real branch is too weak to counteract pseudo-induced geometry drift.
Deliverable: real-view drift / real-loss / replay-side evidence rather than just training loss.
Status: pending.

### Step 6 — supervision quality stratification
Goal: separate `both-supported` vs `single-supported` supervision quality and ask whether the regression is dominated by single-branch pseudo depth.
Deliverable: quality summaries by support class plus replay correlation.
Status: pending.

## Running log
### Step 1
Completed.
Method:
- Used a standalone audit script (`/data/bzhang512/tmp/stageb_branch_grad_audit_20260425.py`) without modifying live training code.
- For each representative route, loaded the live StageB protocol from saved `stageA_history.json` args.
- Important: this audit used the full loaded pools (`9` real, `8` pseudo) for deterministic branch comparison; it did not simulate the per-iteration random `2/4` sampling. The purpose here was branch magnitude attribution, not replay reproduction.
- Measured four weighted branches separately:
  - `real_rgb`
  - `pseudo_rgb`
  - `pseudo_depth`
  - `pseudo_aux` (pose/exposure/abs-pose regularization; effectively zero at init in these routes)
- Collected weighted loss means plus gradient L2 norms on Gaussian xyz / opacity and pseudo pose deltas.

Key grounded results:
1. The real branch is extremely stable across routes.
   - `real_rgb weighted_loss_mean` stays around `0.0961`
   - `real xyz_grad_l2` stays around `0.0384`
   This means the real anchor is not getting stronger when pseudo supervision becomes stronger.
2. Old best is strong partly because the pseudo branch is tiny, not because it is strongly helping.
   - `legacy_old_best_all1`
     - `pseudo_rgb weighted_loss_mean ≈ 0.00147`
     - `pseudo_depth weighted_loss_mean ≈ 0.00049`
     - both are tiny relative to `real_rgb ≈ 0.0961`
   Interpretation: old best pseudo supervision is weak enough not to dominate the real branch.
3. Current exact sparse already has pseudo-depth dominating real xyz gradients.
   - `exact_sparse_current`
     - `real xyz_grad_l2 ≈ 0.0385`
     - `pseudo_rgb xyz_grad_l2 ≈ 0.0456`
     - `pseudo_depth xyz_grad_l2 ≈ 0.2074`
     - `pseudo_depth weighted_loss_mean ≈ 0.0369` (already ~38% of real branch loss scale)
   Interpretation: even the current sparse exact route is already pseudo-depth-heavy on xyz updates.
4. GT RGB alone already pushes as hard as the real branch.
   - `legacy_gt_rgb_only`
     - `pseudo_rgb weighted_loss_mean ≈ 0.02725`
     - `pseudo_rgb xyz_grad_l2 ≈ 0.0370`
     - `real xyz_grad_l2 ≈ 0.0384`
   Interpretation: once pseudo RGB becomes strong, it already competes with the real branch rather than acting like a small nudging term.
5. The worst exact GT+dense route is dominated by pseudo supervision.
   - `exact_gt_rgb_gtdepth_dense3d_q070`
     - `real_rgb weighted_loss_mean ≈ 0.0961`, `real xyz_grad_l2 ≈ 0.0384`
     - `pseudo_rgb weighted_loss_mean ≈ 0.0302`, `pseudo_rgb xyz_grad_l2 ≈ 0.0825`
     - `pseudo_depth weighted_loss_mean ≈ 0.2030`, `pseudo_depth xyz_grad_l2 ≈ 0.1469`
   Interpretation: pseudo-depth alone is >2x the real branch in weighted loss scale, and both pseudo RGB/depth exceed real branch xyz-gradient scale. This is consistent with replay collapse: the pseudo branch is not a mild correction anymore; it becomes the main driver.
6. `pseudo_aux` is basically zero at initialization in these audited routes.
   - So the current issue is not hidden pose/exposure regularization overpowering optimization at step start.

Step-1 conclusion:
- The main suspicious pattern is now grounded: as pseudo supervision becomes stronger, the pseudo branch overtakes the real branch in effective optimization pressure, especially through pseudo depth on Gaussian xyz.
- Therefore the current failure is more likely a branch-balance / target-strength problem than a simple “mask semantics alone” problem.

### Step 2
Completed for the representative bad route `exact_gt_rgb_gtdepth_dense3d_q070`.
Method:
- Reused the exact same init PLY, pseudo cache, signal root, and StageB protocol as the bad route.
- Re-ran StageB with `iters = 20 / 40 / 80`.
- Used the existing finished `iter120` result from `20260425_gt_rgb_gtdepth_dense3d_q070_exactmask_stageB120_replay`.
- Used replay on the init PLY as `iter000` baseline.
- Saved the curve summary to:
  - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260425_step2_curve_exact_gt_rgb_gtdepth_dense3d_q070/curve_summary.json`

Replay curve:
- `iter000_init`: PSNR `23.9703`, SSIM `0.87383`, LPIPS `0.07872`
- `iter020`: PSNR `24.0432`, SSIM `0.87541`, LPIPS `0.07910`
- `iter040`: PSNR `23.8114`, SSIM `0.87124`, LPIPS `0.08213`
- `iter080`: PSNR `23.0587`, SSIM `0.85397`, LPIPS `0.09247`
- `iter120`: PSNR `22.2329`, SSIM `0.83018`, LPIPS `0.10619`

Key conclusion:
1. The bad route does **not** fail immediately.
   - It is actually slightly positive at `iter020` relative to the init PLY (`+0.0729 PSNR`).
2. The failure starts between `20 -> 40` iterations.
   - By `iter040`, replay has already dropped below init.
3. After that, the route degrades monotonically and severely.
   - `iter080` is already much worse than init.
   - `iter120` is catastrophic relative to init and even farther from old best.
4. This supports a `late over-optimization / unstable accumulation` reading rather than an `instant wrong-direction` reading.
   - In other words, the pseudo supervision can initially help a little, but the current StageB contract cannot stop before it starts harming the reconstruction.

Step-2 conclusion:
- The main problem is not “the branch is bad from the first step”.
- The stronger interpretation is: the branch starts in a potentially useful regime, then keeps optimizing past the safe point because the current real/pseudo balance and loss contract do not stabilize it.

### Step 3
Completed for the two representative exact routes:
- `exact_sparse_current`
- `exact_gt_rgb_gtdepth_dense3d_q070`

Method:
- Loaded the init PLY and the live consumer-facing pseudo views for each route.
- Rendered the init PLY before StageB optimization.
- Measured target residuals against the loaded StageB targets, split by exact `C_m` support class:
  - `both`  (`C_m = 1.0`)
  - `single` (`C_m = 0.5`)
  - `invalid` (`C_m = 0.0`)
- Metrics recorded per class:
  - RGB mean L1 per pixel
  - depth absolute error
  - depth relative error

Aggregated results:
1. `exact_sparse_current` starts very close to its pseudo targets on the valid union.
   - support composition:
     - `both_ratio_mean ≈ 0.00738`
     - `single_ratio_mean ≈ 0.00805`
     - union ≈ `0.01543`
   - residuals on valid union:
     - `rgb_l1_union_mean ≈ 0.00591`
     - `depth_abs_union_mean ≈ 0.0493`
     - `depth_rel_union_mean ≈ 0.0161`
2. `exact_gt_rgb_gtdepth_dense3d_q070` starts much farther from its targets even inside the valid union.
   - support composition:
     - `both_ratio_mean ≈ 0.04605`
     - `single_ratio_mean ≈ 0.08880`
     - union ≈ `0.13485`
   - residuals on valid union:
     - `rgb_l1_union_mean ≈ 0.04463`
     - `depth_abs_union_mean ≈ 0.2290`
     - `depth_rel_union_mean ≈ 0.0801`
3. So the strengthened GT+dense route is not just “more coverage”; it is also a much larger residual-matching problem.
   - Relative to exact sparse on the valid union:
     - RGB residual is ~`7.5x` larger
     - depth absolute residual is ~`4.6x` larger
     - depth relative residual is ~`5x` larger
4. In both routes, `single` support is worse than `both` support.
   - `exact_sparse_current`:
     - `depth_rel_both ≈ 0.00926`
     - `depth_rel_single ≈ 0.0222`
   - `exact_gt_rgb_gtdepth_dense3d_q070`:
     - `depth_rel_both ≈ 0.0675`
     - `depth_rel_single ≈ 0.0853`
   This reinforces the earlier suspicion that single-branch supervision is systematically noisier / harder to fit.

Step-3 conclusion:
- The bad GT+dense route does not merely fail because the optimizer is unstable in the abstract.
- It starts from a much larger target mismatch on the very pixels that are allowed to supervise the model.
- Combined with Step 1, the current picture is: the pseudo branch is both stronger and farther away, especially in depth, so StageB is asked to absorb a large target jump through a branch that already dominates real-anchor gradients.

### Step 4
Completed on the representative bad route `exact_gt_rgb_gtdepth_dense3d_q070` at the critical `40-iter` budget.
Goal: separate whether the degradation mainly requires pseudo-driven pose updates, pseudo-driven Gaussian updates, or their interaction.

Method:
- Reused the same route as Step 2.
- Compared three `40-iter` variants:
  - `full40_current` (reference from Step 2)
  - `pose_only`: froze Gaussian learning by setting `stageA5_lr_xyz = 0`, `stageA5_lr_opacity = 0`
  - `gaussians_only`: froze pseudo pose/exposure learning by setting `stageA_lr_rot = 0`, `stageA_lr_trans = 0`, `stageA_lr_exp = 0`
- Saved outputs under:
  - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260425_step4_pose_vs_gaussians_exact_gt_dense3d_q070/`

Replay results at 40 iters:
- `full40_current`: PSNR `23.8114`, SSIM `0.87124`, LPIPS `0.08213`
- `pose_only`: PSNR `23.9703`, SSIM `0.87383`, LPIPS `0.07872`
- `gaussians_only`: PSNR `23.9865`, SSIM `0.87465`, LPIPS `0.08166`

Interpretation:
1. The 40-iter degradation does **not** appear when only one side is allowed to move.
   - `pose_only` essentially returns to the init-Ply replay level.
   - `gaussians_only` is also near-init and slightly above it in PSNR/SSIM.
2. The clear drop only appears in the full joint case.
   - `full40_current` is substantially below both split variants.
3. Therefore the failure is not “pose updates alone are bad” or “Gaussian updates alone are bad”.
   - The bad behavior requires their joint coupling under the current strong pseudo contract.
4. This is consistent with a feedback-loop failure mode:
   - pseudo supervision moves pose a bit,
   - pseudo supervision also moves Gaussians,
   - the two updates reinforce each other under a target that is already far away,
   - and replay degrades after the initially helpful regime.

Step-4 conclusion:
- The main structural problem now looks like `joint pose + Gaussian adaptation under strong pseudo supervision`, not a single isolated branch.
- This means later fixes should focus on coupling / balance / stopping criteria, not only on “turn depth off” or “change mask values” in isolation.

### Step 5
Completed as a direct `real-anchor weakness` audit.
Goal: test whether the real branch can reliably distinguish the bad full-joint solution from the safer split variants.

Method:
- Evaluated the full set of `9` real sparse-train views (not the sampled `2`-view minibatch) on four PLYs:
  - `init_ply`
  - `full40_current`
  - `pose_only`
  - `gaussians_only`
- For each PLY, computed the full-9 mean real RGB mapping loss using the same `get_loss_mapping_rgb(...)` logic as the StageB real branch.
- Compared that against replay quality.

Results:
- `init_ply`
  - real-train RGB loss mean ≈ `0.09613`
  - replay PSNR ≈ `23.9703`
- `full40_current`
  - real-train RGB loss mean ≈ `0.09084`
  - replay PSNR ≈ `23.8114`
- `pose_only`
  - real-train RGB loss mean ≈ `0.09613`
  - replay PSNR ≈ `23.9703`
- `gaussians_only`
  - real-train RGB loss mean ≈ `0.09052`
  - replay PSNR ≈ `23.9865`

Interpretation:
1. The bad full-joint solution actually improves the full real-train RGB loss relative to init.
   - `0.09613 -> 0.09084`
2. But that improvement does **not** translate into better replay.
   - replay still gets worse (`23.9703 -> 23.8114`)
3. Even more revealing: `full40_current` and `gaussians_only` achieve almost the same real-train RGB loss,
   - but their replay outcomes differ materially.
   - `full40_current`: `23.8114`
   - `gaussians_only`: `23.9865`
4. So the real RGB anchor is not strong enough to reject / penalize the harmful joint pseudo adaptation.
   - It can be locally satisfied while global replay quality still drops.

Step-5 conclusion:
- The current real branch is a weak stabilizer for this problem.
- Satisfying the 9 real sparse-train RGB views is not enough to prevent bad joint pseudo updates.
- This directly supports the claim that the current failure is not caught by the real anchor, even when the real anchor itself improves numerically.

### Step 6
Completed as a `both vs single` supervision-quality audit on the two representative exact routes.
Goal: determine whether the bad route is dominated by a larger and noisier single-branch supervision component.

Method:
- For each exact route, used the live loaded pseudo views and exact-upstream bundle.
- Split valid supervision into:
  - `both` (`C_m = 1.0`)
  - `single` (`C_m = 0.5`)
- For each class, measured:
  - average `target_confidence`
  - average effective mass `C_m * target_confidence * valid_mask`
  - share of total effective mass carried by `both` vs `single`
  - RGB residual and depth-relative residual on the init PLY

Key results:
1. `exact_sparse_current` is effectively both-dominated.
   - area ratios:
     - `both_ratio_mean ≈ 0.00738`
     - `single_ratio_mean ≈ 0.00805`
   - confidence:
     - `target_conf_both_mean ≈ 0.680`
     - `target_conf_single_mean ≈ 0.592`
   - effective mass share:
     - `both ≈ 67.6%`
     - `single ≈ 32.4%`
   - residual quality:
     - `rgb_l1_both ≈ 0.00553` vs `single ≈ 0.00626`
     - `depth_rel_both ≈ 0.00926` vs `single ≈ 0.0222`
2. `exact_gt_rgb_gtdepth_dense3d_q070` becomes nearly balanced in effective mass between `both` and `single`, while both classes are much noisier.
   - area ratios:
     - `both_ratio_mean ≈ 0.04605`
     - `single_ratio_mean ≈ 0.08880`
   - confidence:
     - `target_conf_both_mean ≈ 0.457`
     - `target_conf_single_mean ≈ 0.414`
   - effective mass share:
     - `both ≈ 51.7%`
     - `single ≈ 48.3%`
   - residual quality:
     - `rgb_l1_both ≈ 0.0405` vs `single ≈ 0.0465`
     - `depth_rel_both ≈ 0.0675` vs `single ≈ 0.0853`
3. So the bad route does two harmful things simultaneously:
   - it increases total supervision mass a lot,
   - and it allocates much more of that mass to the weaker `single` regime.
4. Even the `both` region in the bad route is far worse than the sparse control.
   - so the problem is not only “too many single pixels”;
   - it is also that the overall target jump is much larger.

Step-6 conclusion:
- The regression is strongly consistent with `single-heavy strong supervision + larger target mismatch + weak real anchor + joint pose/Gaussian feedback`.
- This is now a much tighter diagnosis than the earlier vague statement “dense mask seems worse”.


## 2026-04-27 follow-up — dense q070 `4real+2pseudo` batch-ratio test
User requested a direct follow-up on whether lowering pseudo exposure per StageB minibatch helps under the dense3d q0.70 route, while keeping the loaded pools unchanged (`9` real, `8` pseudo).

Three arms were run under a shared dense q0.70 anchor and replayed:
- `current_4r2p_exact`: exact mask + exact target + depth on
- `rgb_only_4r2p_exact`: exact mask + exact target + depth off
- `allones_rgb_only_4r2p`: all-ones confidence mask + RGB-only (`exact_shared_cm_cm_only_v1` with all-ones copied confidence arrays)

Replay results:
- `current_4r2p_exact`: `23.5531 / 0.86312 / 0.08751`
- `rgb_only_4r2p_exact`: `23.5196 / 0.86317 / 0.08709`
- `allones_rgb_only_4r2p`: `24.1126 / 0.87379 / 0.08194`

Important comparisons:
- vs previous `2real+4pseudo` q070 controls:
  - `current_4r2p_exact - q070_current_2r4p = -0.1129 PSNR`
  - `rgb_only_4r2p_exact - q070_rgb_only_2r4p = -0.1449 PSNR`
- within the new `4real+2pseudo` group:
  - `rgb_only_4r2p_exact - current_4r2p_exact = -0.0335 PSNR`
  - `allones_rgb_only_4r2p - current_4r2p_exact = +0.5595 PSNR`
  - `allones_rgb_only_4r2p - rgb_only_4r2p_exact = +0.5930 PSNR`
- broader references:
  - `allones_rgb_only_4r2p - sparse_current_2r4p = +0.1081 PSNR`
  - `allones_rgb_only_4r2p - old_best_legacy = -0.1515 PSNR`

Follow-up conclusion:
- Lowering pseudo exposure from `2+4` to `4+2` is **not** by itself a fix for the dense exact route.
- Keeping the exact mask/target contract and just changing the batch ratio made both dense exact arms worse.
- The only strong positive result came from the weakest pseudo contract: all-ones mask + RGB-only.
- This sharpens the diagnosis: batch composition matters, but the dominant failure is still the current exact pseudo supervision contract, especially exact mask/depth participation under joint refine.


## 2026-04-27 follow-up — dense q070 continuous-confidence RGB-only
User then asked to revisit the older M-side idea of using continuous confidence instead of the discrete three-level `C_m`, under dense3d q0.70 with pseudo depth disabled.

Historical record check first:
- old M-side taxonomy already had continuous-ish families (`M1 Legacy Joint` semi-continuous, `M3 Hybrid Geometry-gated` continuous same-source).
- the older exact compare also matched the user's memory that this family was weaker:
  - `oldA1 = 24.18774`
  - `exactBrpoCm_oldTarget_v1 = 24.18750`
  - `exactBrpoCm_hybridTarget_v1 = 24.17435`
  - `exactBrpoCm_stableTarget_v1 = 24.17522`
  So the hybrid/continuous-like line was roughly `-0.012 ~ -0.013 PSNR` behind the old / exact-discrete control.

New dense3d no-depth follow-up:
- Reused the `4real+2pseudo` q0.70 compare anchor.
- Copied `dense3d_q070_signal` and replaced `pseudo_confidence_exact_brpo_upstream_target_v1.npy` with `pseudo_target_confidence_exact_brpo_upstream_target_v1.npy`.
- Ran `stageA_depth_loss_mode=exact_shared_cm_cm_only_v1` with `--stageA_disable_depth`, so the effective RGB supervision mask became the continuous confidence itself.

Result:
- `contconf_rgb_only_4r2p = 23.54058 / 0.86354 / 0.08686`

Comparisons:
- vs `rgb_only_4r2p_exact`: `+0.0210 PSNR`
- vs `current_4r2p_exact`: `-0.0125 PSNR`
- vs `allones_rgb_only_4r2p`: `-0.5720 PSNR`

Follow-up conclusion:
- Continuous confidence is only a tiny local improvement over the discrete exact RGB-only arm; it is nowhere near a real rescue.
- The best result in this dense q0.70 / `4real+2pseudo` family remains `allones_rgb_only_4r2p`, not the continuous-confidence arm.
- So this new dense3d no-depth test agrees with the older historical memory rather than overturning it: continuous confidence by itself is not the main fix.

## Archive note
Completed dense-matching execution notes were moved out of `docs/` root into:
- `docs/archived/2026-04-m3d-experiments/`
This keeps `docs/current/*` and this master file as the live entry points for the current forensic pass.

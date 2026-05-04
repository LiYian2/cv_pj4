# Q070 continuous-confidence RGB-only follow-up — 2026-04-27

## Purpose
After the `4real+2pseudo` dense q0.70 compare, user asked to revisit an older M-side idea: replacing the discrete three-level `C_m ∈ {1.0, 0.5, 0.0}` with continuous confidence, and test it under dense3d matching with pseudo depth disabled.

## Historical context checked first
Two older M-side families on record are relevant:
- `M1 Legacy Joint`: semi-continuous / half-continuous engineering mask, documented as "半连续值，工程最稳".
- `M3 Hybrid Geometry-gated` / historical `brpo_direct_v1`: continuous same-source confidence, but not strict BRPO.

The archived exact compare confirms the user’s memory that the continuous/hybrid line was not the winner:
- `oldA1`: `24.1877374861`
- `exactBrpoCm_oldTarget_v1`: `24.1874953941`
- `exactBrpoCm_hybridTarget_v1`: `24.1743486546`
- `exactBrpoCm_stableTarget_v1`: `24.1752181159`
So the continuous/hybrid family was roughly `-0.012 ~ -0.013 PSNR` behind the old / exact-discrete control on that earlier compare.

## Old-best contract check
The old best is **not** `all-ones mask + RGB-only`.
Grounded contract from `20260413_stageB_conservative_gate/phase1_stageB120/stageA_history.json`:
- `stageA_mask_mode = legacy`
- `stageA_target_depth_mode = target_depth_for_refine`
- `stageA_depth_loss_mode = legacy`
- `stageA_disable_depth = False`
It therefore did use depth. The depth term was just small at convergence:
- `loss_depth_last ≈ 0.005742`
- `loss_depth_last10_mean ≈ 0.005926`

## New experiment setup
Reused the existing dense q0.70 `4real+2pseudo` compare root:
- base root: `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260427_q070_4real2pseudo_compare_stageB120_replay`
- common anchor, PLY, init pseudo states, pseudo cache, replay cache, and StageB schedule unchanged
- kept:
  - `num_real_views=4`
  - `num_pseudo_views=2`
  - `stageB_iters=120`
  - `joint_topology_mode=brpo_joint_v1`
  - pseudo depth disabled

### What changed
Copied `dense3d_q070_signal` and replaced:
- `pseudo_confidence_exact_brpo_upstream_target_v1.npy`
with
- `pseudo_target_confidence_exact_brpo_upstream_target_v1.npy`

Then ran:
- `pseudo_observation_mode=exact_brpo_upstream_target_v1`
- `stageA_depth_loss_mode=exact_shared_cm_cm_only_v1`
- `--stageA_disable_depth`

This makes the effective RGB supervision mask be the continuous confidence itself, instead of discrete `C_m`.

## Smoke verification
The 1-iter smoke confirmed:
- `stageA_depth_loss_mode = exact_shared_cm_cm_only_v1`
- `stageA_disable_depth = true`
- `loss_depth_last = 0.0`
- `mean_confidence_nonzero_ratio = 0.1907806396484375`
- `mean_confidence_mean_positive = 0.587513342499733`
So this arm is neither all-ones nor discrete exact `C_m`; it is the intended continuous-confidence RGB-only contract.

## Replay result
New arm:
- `contconf_rgb_only_4r2p`: `23.5405803045 / 0.8635392026 / 0.0868628558`

## Direct comparisons inside the same 4real+2pseudo family
- vs `rgb_only_4r2p_exact`:
  - `PSNR +0.0210`
  - `SSIM +0.00037`
  - `LPIPS -0.00023`
- vs `current_4r2p_exact`:
  - `PSNR -0.0125`
  - `SSIM +0.00042`
  - `LPIPS -0.00064`
- vs `allones_rgb_only_4r2p`:
  - `PSNR -0.5720`
  - `SSIM -0.01025`
  - `LPIPS +0.00492`

## Interpretation
This result is weakly positive relative to the discrete exact RGB-only arm, but only by about `+0.021 PSNR`; that is far too small to count as a real fix. It remains clearly below the all-ones RGB-only arm, which is still the best result in this `4real+2pseudo` dense q0.70 family.

So the new dense3d no-depth test agrees with the older historical memory rather than overturning it:
- switching from discrete `C_m` to continuous confidence does not meaningfully rescue the route;
- it is at best a tiny local improvement over exact discrete RGB-only;
- it is still much weaker than the permissive all-ones RGB-only contract.

## Practical conclusion
For the current dense3d route, `continuous confidence instead of discrete C_m` is still not the main answer. The evidence now lines up across both old history and the new dense q0.70 follow-up:
- exact discrete control is not rescued by simply going continuous;
- the big gain still comes from weakening the pseudo contract much more aggressively (`all-ones + RGB-only`), not from a mild continuous-confidence substitution.

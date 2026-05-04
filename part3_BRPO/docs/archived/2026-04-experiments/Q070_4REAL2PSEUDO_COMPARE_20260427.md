# Q070 4real+2pseudo compare — 2026-04-27

## Purpose
User requested a direct test of whether lowering pseudo exposure in StageB helps under the dense3d q0.70 route, while keeping the pool sizes unchanged (`9` real loaded, `8` pseudo loaded). The requested experiment family was:
1. normal exact mask + dual-projected depth target
2. all-ones mask + RGB-only
3. normal exact mask + RGB-only
with per-iteration sampling changed from `2 real + 4 pseudo` to `4 real + 2 pseudo`.

## Common anchor and protocol
- anchor PLY: `/data/bzhang512/CV_Project/output/part3_BRPO/experiments/20260415_p2b_stageA5_local_gating_compare_e1/stageA5_legacy_xyz_gated_80/refined_gaussians.ply`
- init pseudo states: `/data/bzhang512/CV_Project/output/part3_BRPO/experiments/20260415_p2b_stageA5_local_gating_compare_e1/stageA5_legacy_xyz_gated_80/pseudo_camera_states_final.json`
- pseudo cache: `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/pseudo_cache_baseline`
- train manifest: `/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/split_manifest.json`
- replay cache: `/data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache`
- fixed StageB protocol retained from the earlier dense compare:
  - `stageB_iters=120`
  - `stageB_post_switch_iter=40`
  - `stageB_post_lr_scale_xyz=0.3`
  - `stageB_post_lr_scale_opacity=1.0`
  - `lambda_real=lambda_pseudo=1.0`
  - `joint_topology_mode=brpo_joint_v1`
  - `pseudo_local_gating_spgm_manager_mode=summary_only`
- changed only:
  - `num_real_views: 2 -> 4`
  - `num_pseudo_views: 4 -> 2`

## Arms
### 1. `current_4r2p_exact`
- signal root: `dense3d_q070_signal`
- `pseudo_observation_mode=exact_brpo_upstream_target_v1`
- `stageA_depth_loss_mode=exact_shared_cm_v1`
- depth enabled
- this is the direct `4real+2pseudo` version of the previous dense q070 exact route

### 2. `rgb_only_4r2p_exact`
- signal root: `dense3d_q070_signal`
- same exact mask contract as arm 1
- `--stageA_disable_depth`
- this isolates whether lowering pseudo exposure plus removing pseudo depth is enough while preserving exact mask semantics

### 3. `allones_rgb_only_4r2p`
- signal root copied from `dense3d_q070_signal`
- replaced `pseudo_confidence_exact_brpo_upstream_target_v1.npy` with all-ones arrays for all 8 frames
- `stageA_depth_loss_mode=exact_shared_cm_cm_only_v1`
- `--stageA_disable_depth`
- smoke verification confirmed:
  - `min_confidence_nonzero_ratio = 1.0`
  - `min_confidence_mean_positive = 1.0`
  - `loss_depth_last = 0.0`
- this is the requested true `all-ones mask + RGB-only` arm under the same dense q070 pseudo-target family

## Outputs
- output root: `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260427_q070_4real2pseudo_compare_stageB120_replay`
- compare summary: `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260427_q070_4real2pseudo_compare_stageB120_replay/compare_summary.json`
- run wrapper: `/data/bzhang512/tmp/run_q070_4real2pseudo_compare_stageb120_replay.sh`

## Replay results
- anchor replay: `23.8522 / 0.87064 / 0.08093`
- `current_4r2p_exact`: `23.5531 / 0.86312 / 0.08751`
- `rgb_only_4r2p_exact`: `23.5196 / 0.86317 / 0.08709`
- `allones_rgb_only_4r2p`: `24.1126 / 0.87379 / 0.08194`

## Key comparisons
### Against the earlier `2real+4pseudo` dense q070 controls
- `current_4r2p_exact - q070_current_2r4p`
  - `PSNR -0.1129`
  - `SSIM -0.00273`
  - `LPIPS +0.00173`
- `rgb_only_4r2p_exact - q070_rgb_only_2r4p`
  - `PSNR -0.1449`
  - `SSIM -0.00349`
  - `LPIPS +0.00209`

### Within the new `4real+2pseudo` group
- `rgb_only_4r2p_exact - current_4r2p_exact`
  - `PSNR -0.0335`
  - essentially no rescue from just turning depth off
- `allones_rgb_only_4r2p - current_4r2p_exact`
  - `PSNR +0.5595`
  - `SSIM +0.01068`
  - `LPIPS -0.00557`
- `allones_rgb_only_4r2p - rgb_only_4r2p_exact`
  - `PSNR +0.5930`
  - `SSIM +0.01062`
  - `LPIPS -0.00515`

### Against broader references
- `allones_rgb_only_4r2p - sparse_current_2r4p`
  - `PSNR +0.1081`
- `allones_rgb_only_4r2p - old_best_legacy`
  - `PSNR -0.1515`

## Train-side last-step stats
- `current_4r2p_exact`
  - `loss_total_last 0.15770`
  - `loss_real_last 0.10808`
  - `loss_pseudo_last 0.04962`
  - `loss_rgb_last 0.01247`
  - `loss_depth_last 0.04090`
- `rgb_only_4r2p_exact`
  - `loss_total_last 0.11233`
  - `loss_real_last 0.10696`
  - `loss_pseudo_last 0.00537`
  - `loss_rgb_last 0.00766`
  - `loss_depth_last 0.0`
- `allones_rgb_only_4r2p`
  - `loss_total_last 0.11525`
  - `loss_real_last 0.10753`
  - `loss_pseudo_last 0.00773`
  - `loss_rgb_last 0.01103`
  - `loss_depth_last 0.0`

## Interpretation
The main result is not “more real, fewer pseudo always helps.” Simply changing the minibatch composition from `2+4` to `4+2` did not rescue the dense exact route. In fact, both exact-mask arms became worse than their earlier `2+4` controls. So the current dense exact failure cannot be explained mainly by “pseudo count is too high per minibatch.”

The stronger signal is that `allones_rgb_only_4r2p` improved sharply and became the best arm in this group. That means lowering pseudo exposure becomes useful only when combined with a much weaker / more permissive pseudo contract: no pseudo depth loss and no exact sparse/dense BRPO mask gating. In other words, the problem appears to sit more in the exact pseudo supervision contract than in the raw `2+4` ratio itself.

This result is consistent with the previous forensic diagnosis:
- lowering pseudo exposure alone does not stabilize the current exact masked route;
- removing depth alone while keeping exact mask semantics also does not stabilize it;
- a legacy-like weak pseudo RGB-only contract with all-ones supervision becomes much safer and replay-positive under `4real+2pseudo`.

## Practical conclusion
For the current pipeline, `4real+2pseudo` is not a general fix. If the contract remains `exact mask + exact target + current consumer`, replay still degrades. The only strong positive result in this batch is the weakest pseudo contract (`all-ones + RGB-only`).

So the new evidence points to this narrower reading:
- minibatch ratio matters, but only secondarily;
- the primary bottleneck remains the current exact pseudo supervision contract, especially mask/depth-side constraints under joint refine;
- if we continue this line, the next repair axis should be contract-specific (for example both-vs-single treatment, single-side depth suppression, or weaker pseudo-depth participation), not just more real / fewer pseudo by itself.

# M5_STAGEA-A.5-B_RUNBOOK

Last updated: 2026-04-13 21:08:19 CST
Owner: Hermes / Boyi
Status: active execution runbook for midpoint-8 M5 full pipeline

## 1. Goal
Run the full midpoint-8 experiment under the strict M5 depth-target protocol:
1. First materialize M5 densified pseudo depth for the midpoint pseudo set, producing `target_depth_for_refine_v2.npy` and companion artifacts.
2. Then run `StageA (source_aware)`.
3. Then run `StageA.5 (source_aware + xyz_opacity)`.
4. Then run `StageB (source_aware conservative)`.
5. Save artifacts and perform 270-frame replay after every stage.
6. Compare each stage against the part2 `after_opt` baseline on PSNR / SSIM / LPIPS.

## 2. Fixed experiment inputs
### 2.1 Repo / code
- repo root: `/home/bzhang512/CV_Project/part3_BRPO`
- current git commit: `16788cd9830c463bb379e3b2f84d993e45ff64ac`
- python env: `/home/bzhang512/miniconda3/envs/s3po-gs/bin/python`

### 2.2 Part2 baseline / replay assets
- part2 experiment root: `/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58`
- baseline replay summary: `/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/psnr/after_opt/final_result.json`
- baseline metrics:
  - PSNR = 23.94890858685529
  - SSIM = 0.8734854221343994
  - LPIPS = 0.0787798319839769
- internal eval cache root: `/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache`
- initial/base PLY for StageA entry: `/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache/after_opt/point_cloud/point_cloud.ply`

### 2.3 Midpoint pseudo set
- source config: `/home/bzhang512/CV_Project/part2_s3po/configs/s3po_re10k1_full_full.yaml`
- forced keyframes: `[0, 34, 69, 104, 139, 173, 208, 243, 278]`
- midpoint pseudo frames: `[17, 51, 86, 121, 156, 190, 225, 260]`
- midpoint proto root: `/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_prepare/re10k1__internal_afteropt__midpoint_proto_v1`
- midpoint pseudo cache: `/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_prepare/re10k1__internal_afteropt__midpoint_proto_v1/pseudo_cache`

### 2.4 StageB real-anchor inputs
- sparse train manifest: `/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/split_manifest.json`
- sparse train rgb dir: `/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/rgb`

## 3. Canonical experiment root
`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_midpoint8_m5_fullpipeline`

Expected layout:
- `run_manifest.json`
- `upstream_m5/`
- `stageA/`
- `stageA5/`
- `stageB/`
- `replay/stageA/`
- `replay/stageA5/`
- `replay/stageB/`
- `reports/`

## 4. Canonical protocol
### 4.1 Upstream depth protocol
Strict M5. Do not directly consume the existing midpoint cache as final training input until these artifacts exist for all 8 samples:
- `target_depth_for_refine_v2.npy`
- `target_depth_dense_source_map.npy`
- `depth_correction_seed.npy`
- `depth_correction_dense.npy`
- `depth_seed_valid_mask.npy`
- `depth_dense_valid_mask.npy`
- `depth_densify_meta.json`

### 4.2 Training protocol
Use non-legacy BRPO/source-aware consumption:
- `confidence_mask_source=brpo`
- `stageA_mask_mode=train_mask`
- `stageA_target_depth_mode=target_depth_for_refine_v2` (or alias `blended_depth_m5`)
- `stageA_depth_loss_mode=source_aware`

### 4.3 Stage definitions
1. StageA
- optimize pseudo pose residual + exposure only
- no real branch
- save refined ply, camera states, history

2. StageA.5
- `stage_mode=stageA5`
- `stageA5_trainable_params=xyz_opacity`
- densify off, prune off
- same M5/source-aware input protocol as StageA

3. StageB
- conservative entry only
- same M5/source-aware input protocol
- use sparse real-anchor branch
- keep low LR and conservative weights / views

### 4.4 Replay protocol
After every stage, run `scripts/replay_internal_eval.py` on the 270 non-keyframe replay set and save:
- `final_result.json`
- full per-frame metrics / summary artifacts

## 5. Phase plan
### Phase 0. Preflight + freeze
Must complete before any new refine run:
1. confirm midpoint cache has all M5 materialization prerequisites;
2. create canonical experiment root and run manifest;
3. record code commit / baseline metrics / input paths;
4. verify replay assets exist;
5. verify StageB sparse manifest and rgb dir paths.

### Phase 1. Midpoint M5 materialization
Run `scripts/materialize_m5_depth_targets.py` on the midpoint pseudo cache.
Acceptance gates:
1. all 8 samples emit `target_depth_for_refine_v2.npy` and companion files;
2. aggregate summary exists;
3. no obvious NaN / inf / empty-output sample;
4. source-map composition is not degenerate (not all fallback everywhere).

### Phase 2. StageA + replay
Run StageA using the M5 target and source-aware depth loss, then replay on 270 frames.

### Phase 3. StageA.5 + replay
Run StageA.5 with `xyz_opacity`, then replay on 270 frames.

### Phase 4. StageB + replay
Run conservative StageB, then replay on 270 frames.

### Phase 5. Final comparison
Produce a 4-row comparison:
1. part2 after_opt baseline
2. midpoint8 M5 StageA replay
3. midpoint8 M5 StageA.5 replay
4. midpoint8 M5 StageB replay

## 6. Default hyperparameter stance
These are defaults unless later checkpoint evidence justifies a change:
- pseudo set: midpoint 8
- depth target: M5 `target_depth_for_refine_v2`
- mask source: BRPO / train_mask
- depth loss: source_aware
- A.5 mode: `xyz_opacity`
- StageB mode: conservative entry, not aggressive expansion

## 7. Risks to watch
1. M5 densify may still produce weak or fallback-heavy depth support on midpoint views.
2. StageB may again show short-run gain but long-run regression.
3. Tiny replay deltas must be interpreted as possible noise unless all three metrics trend coherently.

## 8. Execution cadence
Use checkpoint-based execution:
1. finish Phase 0 + 1 and report upstream status;
2. then StageA + replay checkpoint;
3. then StageA.5 + replay checkpoint;
4. then StageB + replay and final summary.

This runbook is the authoritative plan for the current midpoint-8 M5 experiment.

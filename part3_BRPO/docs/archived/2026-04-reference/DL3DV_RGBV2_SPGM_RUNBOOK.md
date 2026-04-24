# DL3DV_RGBV2_SPGM_RUNBOOK.md

> Goal: bring DL3DV onto the current Part3 main research path with Re10k-consistent structure:
> part2 full rerun with internal cache export -> replay parity -> internal prepare -> `signal_v2` RGB-only upstream -> bounded refine -> SPGM downstream.
>
> Status at write time: an older DL3DV full run exists and finished successfully, but it predates the current internal-cache-based Part3 protocol and does not provide a usable `internal_eval_cache/` for the current chain.

## 0. Starting point and design choice

### 0.1 What already exists
- Existing successful DL3DV full run record:
  - config family: `part2_s3po/configs/s3po_dl3dv2_full_full.yaml`
  - old result record: `part2_s3po/checks/s3po_dl3dv2_full_full_result.json`
  - old run dir: `/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/dl3dv-2/s3po_dl3dv-2_full/DL3DV-2_part2_s3po/2026-04-04-02-11-11`
- That old run has final PLY + PSNR files, but its run root does not contain the current `internal_eval_cache/` tree.

### 0.2 Therefore the DL3DV bring-up must restart from Part2
We should not try to patch the old DL3DV run into the current Part3 chain.
The correct entrypoint is:
1. rerun Part2 with internal cache export enabled,
2. validate same-ply replay parity,
3. then build the current Part3 upstream/downstream stack on top of that run root.

### 0.3 Current branch choices to carry over from Re10k
- Preferred upstream signal branch: `signal_v2` with `RGB-only v2`
- Current bounded StageB anchor idea: `post40_lr03_120`
- Current best SPGM control: repair A (`dense_keep`, not selector-first)
- Current selector-first candidate is only a follow-up, not the first DL3DV bring-up target

## 1. Path and schema contract

### 1.1 Dataset split layer
Use the existing DL3DV split contract under:
- `dataset/DL3DV-2/part2_s3po/full`
- `dataset/DL3DV-2/part2_s3po/sparse`

This must stay consistent with Re10k:
- `rgb/`
- `cameras.json`
- `intrinsics_norm.json`
- `intrinsics_px.json`
- `split_manifest.json`

### 1.2 Preferred Part2 output schema
Keep the Re10k directory shape exactly the same, only swap dataset tag:

```text
<PART2_OUTPUT_ROOT>/dl3dv-2/s3po_dl3dv-2_full_internal_cache/
  DL3DV-2_part2_s3po/<timestamp>/
    config.yml
    point_cloud/
    psnr/
    plot/
    internal_eval_cache/
      manifest.json
      camera_states.json
      before_opt/
      after_opt/
      brpo_phaseB/
      brpo_phaseC/
```

Preferred root pattern:
- first choice: `/data2/bzhang512/CV_Project/output/part2_s3po/dl3dv-2/s3po_dl3dv-2_full_internal_cache`
- fallback: `/data/bzhang512/CV_Project/output/part2_s3po/dl3dv-2/s3po_dl3dv-2_full_internal_cache`
- last fallback: `/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/dl3dv-2/s3po_dl3dv-2_full_internal_cache`

Important: even if the storage mount changes, the run-root schema above must stay unchanged.

### 1.3 Preferred internal prepare schema
Mirror the current Re10k layout:

```text
<run_root>/internal_prepare/<prepare_key>/
  manifests/
  inputs/
  difix/
  fusion/
  verification/
  pseudo_cache/
  signal_v2/   # after explicit build step
```

Preferred run-key names:
- coarse candidate fusion root: `dl3dv2__internal_afteropt__candidate_fusion_v1`
- canonical selected root: `dl3dv2__internal_afteropt__brpo_proto_v4_stage3`

### 1.4 Preferred Part3 experiment schema
Keep using:
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/<date>_dl3dv_*`

## 2. Phase A — Part2 full rerun with internal cache export

### Objective
Produce a DL3DV full run that is structurally equivalent to the current Re10k internal-cache source-of-truth.

### Inputs
- base config family: `part2_s3po/configs/s3po_dl3dv2_full_full.yaml`
- dataset path: `dataset/DL3DV-2/part2_s3po/full`
- current old force-KF pattern from the successful DL3DV full run:
  - `[0, 33, 67, 101, 135, 169, 203, 237, 271, 305]`

### Execution contract
- run in `s3po-gs`
- export internal cache explicitly via env or config:
  - `export S3PO_EXPORT_INTERNAL_CACHE=1`
- do not rely on an old run root that lacks `internal_eval_cache/`

### Command skeleton
```bash
source /home/bzhang512/miniconda3/etc/profile.d/conda.sh
conda activate s3po-gs
export TMPDIR=/data/bzhang512/tmp
mkdir -p "$TMPDIR"
export S3PO_EXPORT_INTERNAL_CACHE=1

cd /home/bzhang512/CV_Project/third_party/S3PO-GS
python slam.py --config <tmp_dl3dv2_full_internal_cache.yaml>
```

### Success gate
A valid new DL3DV part2 run must have all of:
- `config.yml`
- `point_cloud/final/point_cloud.ply`
- `psnr/before_opt/final_result.json`
- `psnr/after_opt/final_result.json`
- `internal_eval_cache/manifest.json`
- `internal_eval_cache/camera_states.json`
- `internal_eval_cache/before_opt/`
- `internal_eval_cache/after_opt/`

If `internal_eval_cache/` is missing, stop here and do not continue to Part3.

## 3. Phase B — Same-ply replay parity

### Objective
Verify that replaying the exported `after_opt` PLY under saved internal camera states reproduces the official internal eval almost exactly.

### Command skeleton
```bash
source /home/bzhang512/miniconda3/etc/profile.d/conda.sh
conda activate s3po-gs
export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO:$PYTHONPATH

cd /home/bzhang512/CV_Project/part3_BRPO
python scripts/replay_internal_eval.py \
  --internal-cache-root <run_root>/internal_eval_cache \
  --stage-tag after_opt \
  --ply-path <run_root>/internal_eval_cache/after_opt/point_cloud/point_cloud.ply \
  --config <run_root>/config.yml \
  --save-dir <run_root>/internal_eval_cache/replay_eval/after_opt_sameply
```

### Expected frame-count shape
For DL3DV full with 306 total frames and 10 forced keyframes, expect:
- `num_frames = 306`
- `num_kf = 10`
- `num_non_kf = 296`

### Acceptance gate
Treat parity as passed only if all hold:
- replay frame count matches `stage_meta.num_rendered_non_kf_frames`
- `|ΔPSNR| <= 0.02`
- `|ΔSSIM| <= 5e-4`
- `|ΔLPIPS| <= 5e-4`

If parity fails, stop. Do not continue to signal / refine until the Part2 source run is trustworthy.

## 4. Phase C — Coarse candidate fusion root for signal-aware selection

### Objective
Build a coarse candidate root that covers three pseudo candidates per KF gap so that DL3DV can reuse the same signal-aware selection logic used on Re10k.

Environment note:
- keep using `reggs` for `prepare_stage1_difix_dataset_s3po_internal.py` stages, matching the historically safe Re10k internal-prepare path.

### Why this step exists
`select_signal_aware_pseudos.py` can score candidates directly from internal cache, but it works better if it can read fused pseudo RGBs. Therefore we first materialize a coarse fusion root, then run signal-aware selection on top of it.

### Preferred coarse root
- run-key: `dl3dv2__internal_afteropt__candidate_fusion_v1`
- placement: `both` (covers midpoint + tertile-style candidates per gap)

### Command skeleton
```bash
source /home/bzhang512/miniconda3/etc/profile.d/conda.sh
conda activate reggs
export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO:$PYTHONPATH

cd /home/bzhang512/CV_Project/part3_BRPO
python scripts/prepare_stage1_difix_dataset_s3po_internal.py \
  --stage all \
  --internal-cache-root <run_root>/internal_eval_cache \
  --run-key dl3dv2__internal_afteropt__candidate_fusion_v1 \
  --scene-name DL3DV-2 \
  --stage-tag after_opt \
  --placement both \
  --target-rgb-source difix
```

### Success gate
The coarse candidate root must contain:
- `fusion/samples/<frame_id>/target_rgb_fused.png`
- `manifests/selection_summary.json`
- `pseudo_cache/manifest.json`

## 5. Phase D — Signal-aware pseudo selection

### Objective
Pick the canonical DL3DV pseudo set using the same gap-aware, score-based selection style as Re10k.

### Command skeleton
```bash
source /home/bzhang512/miniconda3/etc/profile.d/conda.sh
conda activate s3po-gs
export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO:$PYTHONPATH

cd /home/bzhang512/CV_Project/part3_BRPO
python scripts/select_signal_aware_pseudos.py \
  --internal-cache-root <run_root>/internal_eval_cache \
  --stage-tag after_opt \
  --output-root <run_root>/internal_prepare/dl3dv2__internal_afteropt__signal_select_e1 \
  --pseudo-fused-root <run_root>/internal_prepare/dl3dv2__internal_afteropt__candidate_fusion_v1/fusion/samples \
  --topk-per-gap 1
```

### Expected outputs
- `reports/signal_aware_selection_report.json`
- `manifests/signal_aware_selection_manifest.json`
- `manifests/selection_summary.json`

### Decision rule
This selection manifest becomes the single source of truth for the canonical DL3DV pseudo set.
Do not manually rewrite frame ids after this step unless a later report explicitly justifies it.

## 6. Phase E — Canonical internal prepare root

### Objective
Materialize the canonical DL3DV prepare root using the selection manifest from Phase D.

### Preferred canonical root
- run-key: `dl3dv2__internal_afteropt__brpo_proto_v4_stage3`
- scene name: `DL3DV-2`
- stage tag: `after_opt`

### Command skeleton
```bash
source /home/bzhang512/miniconda3/etc/profile.d/conda.sh
conda activate reggs
export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO:$PYTHONPATH

cd /home/bzhang512/CV_Project/part3_BRPO
python scripts/prepare_stage1_difix_dataset_s3po_internal.py \
  --stage all \
  --internal-cache-root <run_root>/internal_eval_cache \
  --run-key dl3dv2__internal_afteropt__brpo_proto_v4_stage3 \
  --scene-name DL3DV-2 \
  --stage-tag after_opt \
  --selection-manifest <run_root>/internal_prepare/dl3dv2__internal_afteropt__signal_select_e1/manifests/signal_aware_selection_manifest.json \
  --target-rgb-source difix
```

### Required schema checks
The canonical root must expose the same schema family as current Re10k roots:
- `manifests/`
- `difix/`
- `fusion/`
- `verification/`
- `pseudo_cache/manifest.json`
- `pseudo_cache/samples/<frame_id>/...`

## 7. Phase F — Build `signal_v2` for the canonical root

### Objective
Generate the current preferred upstream signal branch for DL3DV: isolated `signal_v2`, later consumed as `RGB-only v2`.

### Command skeleton
```bash
source /home/bzhang512/miniconda3/etc/profile.d/conda.sh
conda activate s3po-gs
export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO:$PYTHONPATH

cd /home/bzhang512/CV_Project/part3_BRPO
python scripts/build_brpo_v2_signal_from_internal_cache.py \
  --internal-cache-root <run_root>/internal_eval_cache \
  --prepare-root <run_root>/internal_prepare/dl3dv2__internal_afteropt__brpo_proto_v4_stage3 \
  --stage-tag after_opt
```

### Success gate
The canonical root must now contain:
- `signal_v2/frame_<id>/raw_rgb_confidence_v2.npy`
- `signal_v2/frame_<id>/target_depth_for_refine_v2_brpo.npy`
- `signal_v2/frame_<id>/depth_supervision_mask_v2_brpo.npy`

## 8. Phase G — Refine ladder

## G0. Core rule
On DL3DV, bring the chain up in the same order as Re10k:
1. verify signal loading,
2. establish a bounded non-SPGM anchor,
3. then run SPGM repair A,
4. only after that consider selector-first follow-up.

### G1. StageA smoke (signal-loading only)
Purpose:
- confirm `signal_v2` is readable,
- confirm `RGB-only v2` mask path is wired,
- not used for replay ranking.

Key settings to keep aligned with Re10k:
- `target_side=fused`
- `confidence_mask_source=brpo`
- `signal_pipeline=brpo_v2`
- `stage_mode=stageA`
- `stageA_rgb_mask_mode=brpo_v2_raw`
- `stageA_depth_mask_mode=train_mask`
- `stageA_depth_loss_mode=source_aware`
- `stageA_lambda_abs_t=3.0`
- `stageA_lambda_abs_r=0.1`

### G2. StageA.5 short anchor
Purpose:
- first real Gaussian-update check on DL3DV,
- verify that `RGB-only v2` behaves sanely before long StageB compare.

Key settings:
- `stage_mode=stageA5`
- `stageA5_trainable_params=xyz`
- disable densify/prune for the first bring-up
- keep local-gating-style anchor available for comparison

### G3. Canonical bounded StageB baseline (non-SPGM anchor)
Purpose:
- reproduce the current Re10k-style bounded StageB protocol on DL3DV before attributing anything to SPGM.

Key settings to mirror:
- `stage_mode=stageB`
- `stageB_iters=120`
- `stageB_post_switch_iter=40`
- `stageB_post_lr_scale_xyz=0.3`
- `target_side=fused`
- `signal_pipeline=brpo_v2`
- real branch uses DL3DV sparse split:
  - `train_manifest = dataset/DL3DV-2/part2_s3po/sparse/split_manifest.json`
  - `train_rgb_dir = dataset/DL3DV-2/part2_s3po/sparse/rgb`

### G4. SPGM repair A (primary DL3DV downstream target)
This is the first downstream SPGM run to prioritize on DL3DV.

Reason:
- it is the current best SPGM anchor on Re10k,
- it is safer than jumping directly to selector-first,
- it gives a stable control for any later selector study.

Key SPGM settings:
- `pseudo_local_gating=spgm_keep`
- `pseudo_local_gating_spgm_policy_mode=dense_keep`
- `pseudo_local_gating_spgm_ranking_mode=v1`
- `pseudo_local_gating_spgm_support_eta=0.0`
- `pseudo_local_gating_spgm_weight_floor=0.25`
- `pseudo_local_gating_spgm_cluster_keep_near=1.0`
- `pseudo_local_gating_spgm_cluster_keep_mid=1.0`
- `pseudo_local_gating_spgm_cluster_keep_far=1.0`

### G5. Optional selector-first follow-up (not first priority)
Only after G4 is stable:
- try `selector_quantile`
- `ranking_mode=support_blend`
- `lambda_support_rank=0.5`
- start directly from conservative far-only keep near `0.90`

Do not start DL3DV bring-up from stronger selector settings.

## 9. What counts as “DL3DV branch is ready”

Treat the DL3DV migration as structurally ready only when all of the following are true:
1. a new DL3DV full rerun exists with `internal_eval_cache/`,
2. same-ply replay parity passes,
3. canonical `internal_prepare/<prepare_key>/pseudo_cache/` exists,
4. canonical `signal_v2/` exists,
5. StageA smoke confirms signal loading,
6. StageA.5 / StageB bounded anchor runs complete,
7. SPGM repair A runs complete on the same DL3DV protocol.

## 10. What not to do on the first DL3DV pass

Do not start with:
- old DL3DV run roots that lack `internal_eval_cache/`
- selector-first as the first SPGM run
- stochastic SPGM
- `xyz+opacity` as the first A.5 bring-up
- full-v2 depth as the primary upstream choice
- skipping replay parity and going straight to refine

## 11. Minimal deliverables for the later execution phase

When we later execute this runbook, the minimum deliverables should be:
1. new DL3DV part2 run root
2. replay parity report
3. signal-aware selection report + manifest
4. canonical prepare root
5. canonical `signal_v2` root
6. StageA smoke output
7. bounded StageB anchor output
8. SPGM repair A output
9. short summary doc comparing DL3DV bring-up status against the current Re10k chain

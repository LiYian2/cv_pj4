# PIPELINE.md

> Purpose: a compact source-of-truth for drawing the current Part3 BRPO system.
> Scope: current design as of 2026-04-17, including the main data path, signal branches, and refine-time gradient-management branches.

## 1. One-sentence view

The current system is:
part2 S3PO full rerun -> internal_eval_cache -> internal prepare (select / Difix / fusion / verify / pack) -> signal branch (legacy or signal_v2) -> refine (StageA / StageA.5 / StageB) -> replay eval, where refine-time Gaussian updates can be controlled by either local gating or SPGM.

## 2. Main artifact layers

1. Dataset split layer
- `dataset/<scene>/part2_s3po/{full,sparse,...}`
- Contains `rgb/`, `cameras.json`, `intrinsics_*.json`, `split_manifest.json`

2. Part2 run layer
- `<part2_output>/<dataset-tag>/<run-group>/<scene_tag>/<timestamp>/`
- Main outputs: `config.yml`, `point_cloud/`, `psnr/`, `plot/`, optional `internal_eval_cache/`

3. Internal cache layer
- `<run_root>/internal_eval_cache/`
- Canonical schema:
  - `camera_states.json`
  - `manifest.json`
  - `before_opt/`
  - `after_opt/`
  - optional analysis folders such as `brpo_phaseB/`, `brpo_phaseC/`

4. Internal prepare layer
- `<run_root>/internal_prepare/<prepare_key>/`
- Main subtrees:
  - `manifests/`
  - `inputs/`
  - `difix/`
  - `fusion/`
  - `verification/`
  - `pseudo_cache/`
  - optional `signal_v2/`

5. Part3 experiment layer
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/<exp_name>/`
- Main outputs: `stageA_history.json`, `stageB_history.json`, `pseudo_camera_states_final.json`, `refined_gaussians.ply`, `replay_eval/`, `summary.json`

## 3. End-to-end flow

```text
Dataset split
  -> part2 S3PO full rerun
  -> internal_eval_cache
  -> same-ply replay consistency
  -> internal prepare
       -> select pseudo frames + left/right refs
       -> Difix left/right
       -> fusion -> target_rgb_fused
       -> verification -> support / confidence / projected depth
       -> pack -> pseudo_cache
       -> optional build_brpo_v2_signal_from_internal_cache -> signal_v2
  -> refine consumer
       -> StageA   (pose + exposure only)
       -> StageA.5 (first Gaussian update stage)
       -> StageB   (pseudo + real joint refine)
  -> replay_internal_eval.py
  -> compare / analysis
```

## 4. Upstream signal branches

### A. Legacy branch
- RGB supervision mainly comes from propagated train-mask semantics.
- Depth target comes from `target_depth_for_refine{,_v2}`.
- This is the old anchor branch, still kept for compatibility and comparison.

### B. `signal_v2` branch
- RGB mask is decoupled from old train-mask semantics.
- Main RGB variants:
  - `brpo_v2_raw`
  - `brpo_v2_cont`
- Main depth artifact:
  - `target_depth_for_refine_v2_brpo.npy`
  - `depth_supervision_mask_v2_brpo.npy`
- Current preferred upstream branch is `RGB-only v2`, i.e. use `signal_v2` raw RGB confidence as the main RGB gate, without forcing the narrow full-v2 depth branch to dominate training.

## 5. Refine stages

### StageA
- Updates pseudo pose + exposure
- Does NOT update Gaussians / PLY
- Use it for signal / drift diagnosis, not replay-on-PLY ranking

### StageA.5
- First stage that updates Gaussians
- Usually `xyz`-only for the current mainline
- Used for smoke tests and short branch comparisons

### StageB
- Joint refine with pseudo branch + real branch
- Current bounded schedule is the important stable regime
- Canonical bounded baseline on Re10k: `post40_lr03_120`

## 6. Refine-time gradient-management branches

### A. No extra Gaussian management
- Plain refine baseline
- Useful as a lower-level control

### B. Local gating
- View-conditioned gate
- Uses pseudo-view signal quality to decide which sampled pseudo views should contribute
- Main modes already wired in `run_pseudo_refinement_v2.py`:
  - `off`
  - `hard_visible_union_signal`
  - `soft_visible_union_signal`
- Interpretation: this branch answers "which pseudo views should be allowed in"

### C. SPGM
- Gaussian-state-aware gradient management
- Sits after pseudo-view gate, before optimizer step
- Interpretation: this branch answers "which Gaussians should be trusted / down-weighted / selected"

SPGM has two important subfamilies:
1. Dense soft repair (`dense_keep`)
- Current best SPGM anchor is repair A
- Key idea: keep the active set, weaken over-suppression via soft weighting

2. Selector-first (`selector_quantile`)
- First changes the selected Gaussian subset, then applies soft weighting
- Current best selector candidate on Re10k is:
  - `ranking_mode=support_blend`
  - `lambda_support_rank=0.5`
  - far-only selector around `keep_far=0.90`
- It has reached practical parity with repair A, but has not yet replaced repair A as the default SPGM anchor

## 7. Current recommended mainline

For the current Re10k branch, the most trusted mainline is:

```text
internal_eval_cache/after_opt
  -> signal-aware pseudo selection
  -> internal prepare canonical root
  -> signal_v2 (RGB-only v2)
  -> StageA.5 / StageB refine
  -> bounded StageB schedule (post40_lr03_120)
  -> SPGM repair A as current default SPGM control
```

More concretely:
- Upstream preferred signal: `RGB-only v2`
- StageA abs prior background: `lambda_abs_t=3.0`, `lambda_abs_r=0.1`
- Current canonical non-SPGM StageB anchor: `RGB-only v2 + gated_rgb0192 + post40_lr03_120`
- Current best SPGM anchor: repair A (`dense_keep`)
- Current selector-first follow-up zone: `support_blend + far≈0.90`

## 8. Minimal drawing guide

If this file is used to ask an AI to draw the system, the simplest useful figure is a 3-lane diagram:

1. Upstream lane
- dataset split
- part2 full rerun
- internal_eval_cache
- internal prepare
- pseudo_cache / signal_v2

2. Refine lane
- StageA
- StageA.5
- StageB
- real branch + pseudo branch merge inside StageB

3. Gradient-management lane
- local gating (view-conditioned)
- SPGM (Gaussian-conditioned)
- show SPGM split into `dense_keep` and `selector_quantile`

A second optional figure can show branch alternatives:
- legacy signal vs `signal_v2`
- no gating vs local gating vs SPGM
- repair A vs selector-first

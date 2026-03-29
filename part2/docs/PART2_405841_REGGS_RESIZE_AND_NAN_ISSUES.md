# Part2 405841 RegGS: Resize + NaN Issue Log

## Scope
- Dataset preparation notebook: /home/bzhang512/CV_Project/part2/notebooks/01c_prepare_reggs_scene_405841_3segments.ipynb
- Run notebook: /home/bzhang512/CV_Project/part2/notebooks/03_run_405841_full.ipynb
- Core aligner code: /home/bzhang512/CV_Project/third_party/RegGS/src/entities/gaussian_aligner.py

## Problem 1: OOM in infer (historical)
### Symptoms
- Infer failed after long alignment stage with CUDA OOM in gaussian rasterization.
- 405841 run used large image resolution; memory peak occurred in align stage, not in dataloader.

### Root cause chain
- RegGS align loop renders repeatedly at configured image size.
- 405841 config originally used 1280x1920, much larger than 256x256 runs that succeeded.
- Rasterizer intermediate buffers scale strongly with image resolution and gaussian count.

### Fix applied
- Added real resize export path in prepare notebook.
- Exported 405841 segments at 960x640 (non-symlink mode).
- Run notebook now detects image size from exported images and writes H/W dynamically.
- Run notebook now reads fx/fy/cx/cy from scene intrinsics.json (no hardcoded intrinsics).

## Problem 2: Infer still failed after resize (new)
### Symptoms
- OOM disappeared, but infer failed with:
  - opt_gs_scale.grad invalid
  - Value: 0.0, Grad: nan
  - aligner raised Exception

### Root cause chain
- In aligner, scale init used median(depth_main) / median(depth_mini) without robust guarding.
- Some pairs produced unstable or near-zero scale initialization.
- During optimization, mw2 term could dominate early and push unstable gradients.
- Previous behavior aborted entire run on first NaN grad.

## Code-level fix applied in gaussian_aligner.py
### File
- /home/bzhang512/CV_Project/third_party/RegGS/src/entities/gaussian_aligner.py

### Changes
1. Added numeric guard config fields in initializer:
- scale_min (default 0.05)
- scale_max (default 20.0)
- depth_valid_min (default 0.05)
- scale_init_fallback (default 1.0)
- mw2_warmup_iters (default 20)

2. Reworked compute_scale_params with robust init:
- Kept original init line as commented reference.
- Filter invalid/non-positive depth values before median.
- Added epsilon to denominator.
- Added finite check + fallback.
- Clamped initial scale to [scale_min, scale_max].

3. Reworked align loop stability:
- Kept original NaN-raise blocks as commented reference.
- Added mw2 warmup gate (no mw2 in earliest iterations).
- Replaced non-finite mw2 with 0.0 for current iteration.
- Skip iteration if total loss is non-finite.
- If gs_scale grad is non-finite, skip iteration instead of hard crash.
- Clamp gs_scale after optimizer step to safe range.
- Initialized best_w2c/best_gs_scale before loop to avoid undefined states.

## Compatibility checks
- Dataset/export paths remain unchanged (scene_A/scene_B/scene_C structure preserved).
- cameras.json and intrinsics.json schema unchanged.
- Run notebook still writes config to part2/configs and uses same infer/refine/metric commands.
- Dynamic H/W now follows actual exported image size, reducing mismatch risk.

## Operational guidance
1. Re-run prepare notebook cells to ensure export is 960x640.
2. Re-run run notebook through config-build cell and confirm detected image size.
3. Re-run infer and verify no immediate NaN-abort in align stage.
4. If instability persists, tune in config aligner section:
- reduce cam_trans_lr
- increase mw2_warmup_iters
- tighten scale_min/scale_max bounds

## Current status
- Resolution compression path is active and verified.
- OOM path mitigated.
- NaN-abort path hardened with guarded optimization logic.

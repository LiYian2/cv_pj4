# S3PO Part2 Pipeline Study and Notebook Plan (2026-03-30)

## Scope
- Repository reviewed: /home/bzhang512/CV_Project/third_party/S3PO-GS
- Workspace target for Part2 S3PO: /home/bzhang512/CV_Project/part2_s3po
- Existing prep notebooks reviewed:
  - part2_s3po/notebooks/01a_prepare_s3po_scene_re10k-1.ipynb
  - part2_s3po/notebooks/01b_prepare_s3po_scene_dl3dv2.ipynb
  - part2_s3po/notebooks/01c_prepare_s3po_scene_405841_3segments.ipynb

## Main finding
S3PO-GS is a single-entry SLAM pipeline. Unlike RegGS (infer/refine/metric as separate scripts), S3PO runs tracking, mapping, and evaluation inside slam.py.

## Runtime pipeline (code-level)
1) CLI entry and config merge
- slam.py reads --config and optional overrides.
- Inherited config is merged recursively via utils/config_utils.py.

2) Core components initialization
- SLAM class initializes Gaussian model, dataset, frontend, backend, and queues.
- Frontend and backend are connected by multiprocessing queues.

3) Frontend loop per frame
- Initialization on first frame:
  - use GT pose from cameras.json
  - estimate mono depth via MASt3R
  - send init message to backend
- Tracking on subsequent frames:
  - relative pose from MASt3R matches + rendered depth PnP
  - refine pose by photometric optimization
- Keyframe decision:
  - translation and overlap rules
  - maintain a fixed-size keyframe window

4) Backend loop
- On init:
  - create first keyframe gaussians and run initialization mapping iterations
- On keyframe:
  - run local map optimization over window
  - densify and prune gaussians
  - optimize selected keyframe poses and exposures
- Optional color refinement runs as post-SLAM optimization.

5) Evaluation and outputs
- ATE and rendering metrics are computed in-process.
- Output artifacts include trajectory json/plots, viz renders, PSNR/SSIM/LPIPS json, and final point cloud ply.

## Data format expected by S3PO parser
For dl3dv and KITTI modes, parser expects at least:
- dataset_path/rgb/*.png
- dataset_path/cameras.json
- calibration in config (fx, fy, cx, cy, distortion, width, height, depth_scale)

For monocular RGB-only usage in this repo:
- depth and mono_depth paths are often intentionally pointed to rgb files as placeholders.
- true mono depth used in optimization is generated from MASt3R inside runtime.

## Existing prep notebooks compatibility
Current 01 notebooks already generate compatible scene layouts and config templates:
- Symlink images -> rgb
- Symlink cameras.json
- Convert normalized intrinsics to pixel intrinsics
- Write scene-specific yaml inheriting S3PO base config

## Risks / blockers found in upstream code
1) slam.py references undefined CLI args
- args.ns and args.sh are used but never declared in ArgumentParser.
- This likely raises AttributeError at runtime unless patched.

2) eval_rendering iteration condition bug
- end_idx assignment uses: iteration == "final" or "before_opt"
- This expression is always truthy due Python boolean rules.
- Effect: iteration boundary may not behave as intended.

3) single_thread source mismatch
- frontend reads Training.single_thread
- backend reads Dataset.single_thread fallback
- Potential behavior inconsistency if only Training.single_thread is set.

## Proposed notebook structure for Part2 S3PO
Recommended to mirror prior RegGS notebook decomposition while adapting to S3PO single-entry runtime:

A) 02_s3po_smoke_test.ipynb
- select dataset/scene
- build or patch a smoke config (small frame range)
- run one slam.py command
- validate output tree and key metrics json exists

B) 03_s3po_run_full.ipynb
- formal config generation (full frame range)
- run with controlled env (CUDA device, python path, cwd)
- capture stdout/stderr to file
- summarize final metrics and output paths

C) 04_s3po_read_results.ipynb
- parse plot/stats_final.json and psnr/*/final_result.json
- aggregate multi-scene (especially 405841 A/B/C)
- export csv/json summaries to part2_s3po/reports

## Suggested run directory layout
- part2_s3po/configs
- part2_s3po/checks
- part2_s3po/reports
- output/part2_s3po
- results/part2_s3po

## Implementation notes for next step
- Prefer running slam.py from S3PO root so inherit_from relative paths resolve.
- Patch blocking upstream args bug before first executable run notebook.
- Add preflight checks in notebook:
  - dataset path exists
  - rgb count > 0
  - cameras length matches rgb count
  - intrinsics_px numeric sanity
  - mast3r model availability/network policy

# Part2 S3PO Data Preparation Record

Date: 2026-03-30

## 1. Scope

This record summarizes the derived data preparation for S3PO under dataset-level paths:

- `dataset/Re10k-1/part2_s3po`
- `dataset/DL3DV-2/part2_s3po`
- `dataset/405841/part2_s3po/scene_A`
- `dataset/405841/part2_s3po/scene_B`
- `dataset/405841/part2_s3po/scene_C`

The pipeline uses symlinks for source assets (no image copying), adapts directory naming to `rgb`, converts normalized intrinsics to pixel intrinsics, generates S3PO configs, and writes consistency check reports.

## 2. Derived Scene Contract

Each prepared scene follows this contract:

- `rgb` (symlink to source `images`)
- `cameras.json` (symlink)
- `intrinsics_norm.json` (symlink)
- `intrinsics_px.json` (new generated file)
- `scene_manifest.json` (new generated file)

## 3. Generated Configs

Configs are generated in:

- `part2_s3po/configs/s3po_re10k1.yaml`
- `part2_s3po/configs/s3po_dl3dv2.yaml`
- `part2_s3po/configs/s3po_405841_scene_a.yaml`
- `part2_s3po/configs/s3po_405841_scene_b.yaml`
- `part2_s3po/configs/s3po_405841_scene_c.yaml`

Key config checks:

- `dataset_path` points to each `part2_s3po` scene directory.
- `begin=0`, `end` matches frame count.
- `fx/fy/cx/cy` are pixel intrinsics converted from normalized intrinsics.
- `width/height` match prepared image resolution.

## 4. Check Reports and Meaning

Check outputs are stored in `part2_s3po/checks`:

- `re10k1_part2_s3po_check.json` and `.csv`
- `dl3dv2_part2_s3po_check.json` and `.csv`
- `405841_part2_s3po_check_detail.json`
- `405841_part2_s3po_check_summary.json`
- `405841_part2_s3po_check_summary.csv`

These files record:

- symlink validity (`rgb`, `cameras.json`, `intrinsics_norm.json`)
- target existence of linked files
- image/camera count consistency
- image name set consistency (`rgb` filenames vs `cameras.json.image_name`)
- quaternion norm sanity (close to 1)
- translation finite-value check
- pixel intrinsics sanity (positive focal, principal point in bounds)
- 405841 expected resized resolution check (`960x640`)
- final boolean `overall_pass`

## 5. Validation Summary

- Re10k-1: `overall_pass = true`, `n_images = 279`, `n_cameras = 279`
- DL3DV-2: `overall_pass = true`, `n_images = 306`, `n_cameras = 306`
- 405841 scene_A: `overall_pass = true`, `25/25`, `960x640`
- 405841 scene_B: `overall_pass = true`, `45/45`, `960x640`
- 405841 scene_C: `overall_pass = true`, `129/129`, `960x640`

## 6. Readiness Verdict

Current data preparation status is OK for S3PO dataset loading with the generated configs.

No structural/data-consistency blocker was found in this check stage.

## 7. Residual Risk Note

This record validates data structure and metadata consistency. It does not replace runtime validation inside full S3PO execution (environment/GPU/dependency/runtime behavior).

Recommended next step: run one short smoke test with one generated config to confirm end-to-end loader/runtime behavior.

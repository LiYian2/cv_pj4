# Part 2 Source Format Analysis

## Canonical prepared-data location
Prepared RegGS-ready scenes should live under:

```text
dataset/<scene>/part2/
```

Examples:
- `dataset/Re10k-1/part2/`
- `dataset/DL3DV-2/part2/`
- `dataset/405841/part2/`

This is more consistent with the existing Part 1 organization than a single shared `dataset/part2/` directory.

---

## Scene-by-scene source analysis

### 1. Re10k-1
Observed source assets:
- `dataset/Re10k-1/images/*.png`
- `dataset/Re10k-1/cameras.json`

What this means:
- image sequence already exists as a flat ordered folder;
- pose file already exists in a RegGS-like list-of-cameras form;
- each camera entry already contains `cam_quat`, `cam_trans`, and normalized intrinsics fields (`fx`, `fy`, `cx`, `cy`);
- image naming is already sortable (`00000.png`, ...).

Expected extraction logic:
- source images: `dataset/Re10k-1/images/*.png`
- source pose/intrinsics: `dataset/Re10k-1/cameras.json`
- likely easiest adapter target;
- if target resolution changes, we can symlink-or-resize images and write a fresh `intrinsics.json`.

Main note:
- Re10k-1 is the cleanest first adapter because the source format is already close to what RegGS expects.

### 2. DL3DV-2
Observed source assets:
- `dataset/DL3DV-2/rgb/frame_*.png`
- `dataset/DL3DV-2/intrinsics.json`
- `dataset/DL3DV-2/cameras.json`

What this means:
- image sequence exists as a flat ordered RGB folder;
- intrinsics already exist separately as normalized values;
- cameras are already stored as a list with `cam_quat`, `cam_trans`, and image name;
- image names use `frame_00001.png` style, so export should renumber to stable zero-padded numeric names if we want the output naming uniform.

Expected extraction logic:
- source images: `dataset/DL3DV-2/rgb/*.png`
- source intrinsics: `dataset/DL3DV-2/intrinsics.json`
- source poses: `dataset/DL3DV-2/cameras.json`
- export can mostly be metadata conversion + image linking/resizing.

Main note:
- also relatively straightforward; mostly a naming-normalization issue.

### 3. 405841
Observed source assets:
- `dataset/405841/FRONT/rgb/*.png`
- `dataset/405841/FRONT/calib/*.txt`
- `dataset/405841/FRONT/gt/*.txt`
- `dataset/405841/FRONT/gt_old/*.txt`

What this means:
- source data is more raw and not yet in RegGS-like json form;
- intrinsics appear inside per-frame calibration txt files;
- pose appears in 4x4 txt matrices under `gt/`;
- this adapter will need actual parsing and conversion logic rather than just reformatting json.

Expected extraction logic:
- source images: `dataset/405841/FRONT/rgb/*.png`
- source intrinsics: parse one calib txt format (`fx fy cx cy`, plus distortion if needed)
- source poses: parse 4x4 matrices from `gt/*.txt`
- convert poses to the quaternion/translation structure expected by RegGS export;
- decide whether `gt/` or `gt_old/` is the canonical trajectory source (current expectation: use `gt/`, because it looks already origin-shifted / normalized compared with `gt_old/`).

Main note:
- 405841 is the least plug-and-play adapter and should be handled after the Re10k-1 prototype is stable.

---

## Recommended adapter implementation order
1. Re10k-1
2. DL3DV-2
3. 405841

Reason:
- Re10k-1 and DL3DV-2 already have json-based pose metadata and are easier to validate quickly;
- 405841 requires text parsing plus pose-convention checking.

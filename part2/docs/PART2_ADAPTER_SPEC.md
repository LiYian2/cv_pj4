# Part 2 Adapter Spec

This document defines how upstream scenes should be adapted into the RegGS input format for Part 2.

## Main rule

Part 2 scene preparation should preserve the **full frame sequence** inside each exported scene directory.
Do **not** manually sparsify the directory before running RegGS if RegGS will still use `sample_rate` and `n_views` internally.

## Derived-data rule

For Part 2, derived data should prefer **symlink over copy** whenever possible.

Use symlinks for:
- source image files that do not need pixel modification;
- references to existing organized scene assets;
- any large intermediate files that are reused across stages.

Use real copied files only when necessary, for example:
- resized image outputs;
- reformatted metadata files (`intrinsics.json`, `cameras.json`);
- generated config files;
- metrics / result csv files.

In short:
- unchanged large assets -> symlink
- transformed assets -> write new files

## Target RegGS scene format

Each exported scene should look like:

```text
<dataset_root>/<scene_name>/
├── images/
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
├── intrinsics.json
└── cameras.json
```

## Shared adapter contract

Each dataset adapter should provide enough information to export the scene above.

### Required outputs
- ordered frame list;
- image paths or image arrays;
- camera intrinsics for the exported resolution;
- per-frame camera pose;
- stable scene name.

### Required fields to resolve
For each upstream dataset, the adapter must define:
- where images come from;
- how frames are ordered;
- where GT intrinsics come from;
- where GT poses come from;
- whether images can be symlinked directly or must be resized/written;
- how normalized intrinsics are computed after export;
- how the output scene name is formed.

## Normalized intrinsics rule

`intrinsics.json` should store normalized values:
- `fx / W`
- `fy / H`
- `cx / W`
- `cy / H`

If image resolution changes, intrinsics must be updated consistently.

## Pose export rule

`cameras.json` should store per-frame:
- `cam_quat`
- `cam_trans`

The exported pose convention must match what RegGS actually reads in `src/entities/datasets.py`.

## Naming rule

Image filenames must be sortable and gap-safe, e.g.:

```text
0000.png
0001.png
0002.png
```

Avoid mixed naming like `frame_1.png`, `frame_10.png`, `frame_2.png`.

## Part 2 directory responsibilities

```text
part2/
├── docs/
├── notebooks/
├── scripts/
├── configs/
└── reports/

dataset/Re10k-1/part2/
dataset/DL3DV-2/part2/
dataset/405841/part2/   # canonical location for scene-local RegGS-ready inputs
output/part2/           # raw RegGS outputs and logs
results/part2/          # collected metrics / csv summaries
plots/part2/            # figures derived from results
```

## Notebook-first workflow

For Part 2, preprocessing should be notebook-first.

That means:
- scene export / inspection happens primarily in `part2/notebooks/*.ipynb`;
- notebooks can call helper functions later if needed;
- but the main preparation workflow should remain easy to inspect interactively.

## First notebook scope

The first notebook should:
- choose a source scene;
- inspect frame count, image paths, and pose/intrinsic availability;
- export one RegGS-ready scene to `dataset/<scene>/part2/`;
- prefer symlinks when no resize is needed;
- write fresh files when resize / metadata conversion is needed;
- print a compact verification summary of the exported scene.

# Part1 Notebook Pipeline Plan

## Goal

Use notebooks as the main workflow entry for Part 1 data processing and reconstruction.

Keep the structure:

```text
dataset/<SCENE>/part1/shared/...
dataset/<SCENE>/part1/planA/...
dataset/<SCENE>/part1/planB/...
```

Do not create many extra scripts unless clearly necessary.

---

## Notebook Split

### 1. `part1/notebooks/01_data_workspace.ipynb`

Purpose: normalize different raw datasets into the same Part 1 workspace.

Main sections:

#### 1.1 Check dataset format

For each scene, inspect:

- scene root exists or not
- image source directory location
- metadata files location
- image count
- image size
- existing Part 1 workspace status

Expected raw formats:

- `Re10k-1`: `images/`, `cameras.json`, `intrinsics.json`
- `DL3DV-2`: `rgb/`, `cameras.json`, `intrinsics.json`
- `405841`: `FRONT/rgb/`, `FRONT/calib/` (and optional `depth/`, `gt/`)

#### 1.2 Initialize per-scene Part 1 workspace

Create:

```text
dataset/<SCENE>/part1/
├── shared/
│   ├── raw_images/
│   ├── images_512/
│   ├── meta/
│   └── subsets/
├── planA/
└── planB/
```

#### 1.3 Extract raw images and metadata into shared workspace

Per dataset, convert raw storage into a common shared structure:

- copy or link source RGB frames into `shared/raw_images/`
- write ordered image list into `shared/meta/image_list.txt`
- write dataset summary into `shared/meta/dataset_info.json`
- copy or convert metadata needed for later steps

Notes:

- `Re10k-1`: extract from `images/`
- `DL3DV-2`: extract from `rgb/`
- `405841`: extract from `FRONT/rgb/`, and summarize calibration from `FRONT/calib/`

#### 1.4 Resize images to 512

Use `shared/raw_images/` as input.

Output:

```text
dataset/<SCENE>/part1/shared/images_512/
```

This is the common image source for both Plan A and Plan B.

#### 1.5 Generate subsets

Generate uniform subsets from `images_512/`.

Output:

```text
dataset/<SCENE>/part1/shared/subsets/
├── subset_32.txt
├── subset_64.txt
├── subset_96.txt
└── subset_128.txt
```

These subset files are shared by:

- Plan A matched-subset experiments
- Plan B VGGT input preparation

---

### 2. `part1/notebooks/02_planA_3dgs.ipynb`

Purpose: run the COLMAP-based reconstruction pipeline and prepare Plan A inputs for 3DGS.

Main sections:

#### 2.1 Dependency and environment check

Check:

- `colmap` available
- correct conda environment active
- required Python packages available

#### 2.2 Select scene and mode

Support at least:

- full-view Plan A
- subset Plan A (such as 96-view)

Inputs come from:

- `dataset/<SCENE>/part1/shared/images_512/`
- optional subset file from `shared/subsets/`

#### 2.3 Prepare COLMAP workspace

Create workspace under:

```text
dataset/<SCENE>/part1/planA/<variant>/
```

Examples:

- `planA/colmap_279`
- `planA/colmap_96`

Expected internal structure after preparation:

```text
<variant>/
├── database.db
├── images/
├── sparse/
├── undistorted/
└── gs_scene/
```

#### 2.4 Feature extraction and matching

Record and run the final successful COLMAP settings:

- `SIMPLE_PINHOLE`
- single shared camera
- `SiftExtraction.max_num_features = 4096`
- sequential matcher
- `SequentialMatching.overlap = 10`
- `SequentialMatching.quadratic_overlap = 1`

These commands can be kept directly in notebook cells.

#### 2.5 Sparse reconstruction and model inspection

Run:

- mapper
- model analyzer
- optional model conversion when needed

Record key summary:

- registered images
- points
- observations
- reprojection error

#### 2.6 Undistort and organize 3DGS scene

Convert COLMAP output into the standard scene structure expected by 3DGS.

Important: keep the scene-internal structure compatible with model input requirements.

---

### 3. `part1/notebooks/03_planB_vggt.ipynb`

Purpose: run the VGGT-based initialization pipeline and prepare Plan B inputs for 3DGS.

Main sections:

#### 3.1 Dependency and environment check

Check imports before execution.

Important note:

- some modules should be imported only after `import torch`
- notebook should explicitly do torch import first, then import dependent modules

Check availability of:

- `torch`
- `lightglue`
- `hydra`
- `ninja`
- `pycolmap`
- `trimesh`

Notebook should check and report status, not repeatedly reinstall dependencies.

#### 3.2 Select scene and subset

Input comes from:

- `dataset/<SCENE>/part1/shared/images_512/`
- `dataset/<SCENE>/part1/shared/subsets/subset_<N>.txt`

Initial main target:

- 96-image subset

#### 3.3 Prepare VGGT input workspace

Create workspace under:

```text
dataset/<SCENE>/part1/planB/vggt_<N>/
```

Expected internal structure:

```text
vggt_<N>/
├── inputs/
├── raw_vggt/
├── converted_colmap/
├── eval/
└── gs_scene/
```

#### 3.4 Run VGGT reconstruction

Record and execute the final successful configuration:

- `shared_camera = True`
- `camera_type = SIMPLE_PINHOLE`
- `use_ba = True`
- `fine_tracking = False`
- `query_frame_num = 4`
- `max_query_pts = 1024`

Use notebook cells to run the actual command sequence.

#### 3.5 Check sparse output

Record:

- registered images
- points
- observations
- track length
- reprojection error

#### 3.6 Organize standard 3DGS scene

Convert VGGT output into the standard input structure needed by downstream models.

---

## Current Scope

Current notebook plan covers:

1. data inspection
2. per-scene workspace initialization
3. raw image and metadata extraction
4. resize to 512
5. subset generation
6. Plan A notebook
7. Plan B notebook

Result collection notebook is postponed for now.

---

## Design Principles

- Use notebooks as the main execution interface
- Keep per-scene raw format handling only in the data workspace notebook
- After workspace initialization, use the same logic for all scenes
- Keep commands visible in notebook cells for clarity
- Avoid excessive script splitting unless a step becomes repetitive or hard to maintain
- Preserve scene-internal directory layouts required by COLMAP, 3DGS, and VGGT

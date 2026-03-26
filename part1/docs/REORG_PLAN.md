# Part1 Reorganization Plan

## Rules

- Keep dataset names unchanged: `405841`, `DL3DV-2`, `Re10k-1`
- `part1/` keeps only code, notebooks, scripts, docs, configs, light reports
- Processed data moves under `dataset/<DATASET_NAME>/part1/`
- Training outputs, eval outputs, logs move under `output/part1/<DATASET_NAME>/`
- Scene structures used as model inputs must stay internally unchanged after moving
- Clone libraries are not touched in this step

## Target Tree

```text
CV_Project/
├── dataset/
│   ├── 405841/
│   │   ├── ...
│   │   └── part1/
│   ├── DL3DV-2/
│   │   ├── ...
│   │   └── part1/
│   └── Re10k-1/
│       ├── cameras.json
│       ├── intrinsics.json
│       ├── images/
│       └── part1/
│           ├── shared/
│           │   ├── gt_colmap_text/
│           │   └── meta/
│           ├── planA/
│           │   ├── colmap_279/
│           │   │   ├── database.db
│           │   │   ├── images/
│           │   │   ├── sparse/
│           │   │   ├── undistorted/
│           │   │   └── gs_scene/
│           │   └── colmap_96/
│           │       └── gs_scene/
│           └── planB/
│               └── vggt_96/
│                   ├── inputs/
│                   ├── raw_vggt/
│                   ├── converted_colmap/
│                   ├── eval/
│                   └── gs_scene/
├── output/
│   └── part1/
│       ├── 405841/
│       ├── DL3DV-2/
│       └── Re10k-1/
│           ├── planA/
│           │   ├── colmap_279/
│           │   │   ├── 3dgs/40k_2k/
│           │   │   ├── scaffold/40k_2k/
│           │   │   └── logs/
│           │   └── colmap_96/
│           │       ├── 3dgs/40k_2k/
│           │       ├── scaffold/40k_2k/
│           │       └── logs/
│           ├── planB/
│           │   └── vggt_96/
│           │       ├── 3dgs/40k_2k/
│           │       ├── scaffold/40k_2k/
│           │       └── logs/
│           └── verification/
│               ├── scaffold_40k/
│               └── scaffold_grid8_20k/
└── part1/
    ├── README.md
    ├── REORG_PLAN.md
    ├── docs/
    │   ├── SETUP.md
    │   ├── PLANA.md
    │   ├── PLANB.md
    │   ├── OPTSETUP.md
    │   └── COMPARISON.md
    ├── notebooks/
    ├── scripts/
    │   ├── prepare/
    │   ├── train/
    │   ├── eval/
    │   └── utils/
    ├── configs/
    └── reports/
        └── Part1_Metrics.csv
```

## Planned Moves from Current `part1/`

### Keep in `part1/`

- `SETUP.md` -> `docs/SETUP.md`
- `PLANA.md` -> `docs/PLANA.md`
- `PLANB.md` -> `docs/PLANB.md`
- `OPTSETUP.md` -> `docs/OPTSETUP.md`
- `COMPARISON.md` -> `docs/COMPARISON.md`
- `Part1_Metrics.csv` -> `reports/Part1_Metrics.csv`
- `part1_data/prepare_re10k1.py` -> `scripts/prepare/prepare_re10k1.py`
- `part1_data/re10k_1/create_planA96_scene.py` -> `scripts/prepare/create_planA96_scene.py`
- `part1_data/re10k_1/run_part1_40k.sh` -> `scripts/train/run_part1_40k.sh`
- `part1_data/re10k_1/run_planB96_40k.sh` -> `scripts/train/run_planB96_40k.sh`
- `part1_data/re10k_1/run_scaffold_40k_verification.sh` -> `scripts/train/run_scaffold_40k_verification.sh`
- `part1_data/re10k_1/run_scaffold_grid8_20k.sh` -> `scripts/train/run_scaffold_grid8_20k.sh`

### Move to `dataset/Re10k-1/part1/`

- `part1_data/re10k_1/gt_colmap_text` -> `shared/gt_colmap_text`
- `part1_data/re10k_1/meta` -> `shared/meta`
- `part1_data/re10k_1/planA_colmap` -> `planA/colmap_279`
- `part1_data/re10k_1/planA_colmap_96` -> `planA/colmap_96`
- `part1_data/re10k_1/planB_vggt` -> `planB/vggt_96`

### Move to `output/part1/Re10k-1/`

- `part1_data/re10k_1/logs_40k` -> `planA/colmap_279/logs` and/or `planA/colmap_96/logs` and/or `planB/vggt_96/logs` after file-level split
- `part1_data/re10k_1/scaffold_40k_verification` -> `verification/scaffold_40k`
- `part1_data/re10k_1/scaffold_test_planA279` -> `verification/scaffold_grid8_20k`
- `part1_data/re10k_1/scaffold_40k_verification_launcher.log` -> `verification/scaffold_40k/launcher.log`
- `part1_data/re10k_1/scaffold_grid8_launcher.log` -> `verification/scaffold_grid8_20k/launcher.log`

## Important Notes for the Move Step

- Keep internal scene folder layouts unchanged, especially directories like `gs_scene/`, `sparse/`, `images/`, `undistorted/`, `converted_colmap/`, `inputs/`
- Only change parent paths, not scene-internal organization
- Output directories will be moved out of scene folders, but their internal result files stay unchanged
- We will first reorganize files and folders, then update code/script paths in a later step

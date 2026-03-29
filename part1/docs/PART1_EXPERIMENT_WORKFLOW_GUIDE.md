# Part 1 End-to-End Experiment Workflow Guide

## 1. Purpose and Positioning
This document consolidates the planning logic in [part1/docs/REORG_PLAN.md](part1/docs/REORG_PLAN.md), [part1/docs/PIPELINE_NOTEBOOK_PLAN.md](part1/docs/PIPELINE_NOTEBOOK_PLAN.md), and [part1/docs/EXPERIMENT_NOTEBOOK_PLAN.md](part1/docs/EXPERIMENT_NOTEBOOK_PLAN.md) into one operational guide for the current Part 1 codebase. Its role is procedural rather than analytical: it explains how data are prepared, how experiments are launched, how outputs are organized, and how results are collected and visualized under the directory state that is actually present in this workspace.

The central design objective is reproducible execution across three scenes with aligned naming, stable paths, and minimal ambiguity between notebook flow and script flow. The practical outcome is a single reference that a new collaborator can follow to rerun Part 1 without reverse engineering historical path changes.

## 2. Current Repository Reality
The active Part 1 root is [part1](part1), with workflow assets in [part1/notebooks](part1/notebooks), [part1/scripts/train](part1/scripts/train), [part1/scripts/prepare](part1/scripts/prepare), and documentation in [part1/docs](part1/docs). Processed scene data are stored under [dataset](dataset), and training outputs are stored under [output/part1](output/part1). Parsed metrics are exported under [results/part1](results/part1), while publication-ready figures and summary tables are under [plots/part1](plots/part1).

A critical point is naming: current scene variants are colmap_full, colmap_108, and vggt_108. This supersedes older planning names such as colmap_279, colmap_96, and vggt_96 in the planning drafts. The workflow in this guide always follows the active naming used in output logs and metrics CSV files.

## 3. Data and Scene Layout
The three scene roots are [dataset/Re10k-1](dataset/Re10k-1), [dataset/DL3DV-2](dataset/DL3DV-2), and [dataset/405841](dataset/405841). For each scene, Part 1 assets follow the same internal structure:

scene_root/part1/shared
scene_root/part1/planA
scene_root/part1/planB

In shared, the active subfolders are raw_images, images_512, meta, and subsets. In the current state, subsets contains subset_108.txt for all three scenes, which reflects the standardized 108-view controlled setting used by the present 12/18-run workflows.

## 4. Workflow Architecture
Part 1 is implemented as a staged pipeline with a notebook-first interface and script-backed large-scale training:

1. Data workspace normalization and subset generation.
2. Plan A reconstruction and scene packaging.
3. Plan B reconstruction and scene packaging.
4. Multi-run training orchestration for 3DGS and Scaffold-GS.
5. Result parsing into stable CSV artifacts.
6. Figure generation for reporting.

The canonical notebook sequence is:

- [part1/notebooks/01_data_workspace.ipynb](part1/notebooks/01_data_workspace.ipynb)
- [part1/notebooks/02_planA_3dgs.ipynb](part1/notebooks/02_planA_3dgs.ipynb)
- [part1/notebooks/03_planB_vggt.ipynb](part1/notebooks/03_planB_vggt.ipynb)
- [part1/notebooks/04_read_results.ipynb](part1/notebooks/04_read_results.ipynb)
- [part1/notebooks/05_plot_results_revised.ipynb](part1/notebooks/05_plot_results_revised.ipynb)

This means the originally planned launch/read split in EXPERIMENT_NOTEBOOK_PLAN has been operationally implemented as direct run scripts plus a read notebook that scans output trees, rather than a manifest-first notebook pair.

## 5. Stage A: Workspace Preparation
The data normalization stage is performed in [part1/notebooks/01_data_workspace.ipynb](part1/notebooks/01_data_workspace.ipynb). The notebook is configured to iterate over Re10k-1, DL3DV-2, and 405841, while adapting to scene-specific raw formats and converging them into one shared structure per scene.

The key outputs of this stage are:

- shared/raw_images populated from scene-specific sources
- shared/images_512 resized image set
- shared/meta with image list and dataset summary
- shared/subsets/subset_108.txt produced by uniform sampling

At this stage, the pipeline intentionally standardizes only one subset file, subset_108.txt, because the active comparative protocol is anchored on the 108-view condition.

## 6. Stage B: Plan A (COLMAP) Initialization
Plan A reconstruction is executed in [part1/notebooks/02_planA_3dgs.ipynb](part1/notebooks/02_planA_3dgs.ipynb). The notebook supports two modes: colmap_108 and colmap_full, sourced respectively from subset_108 and full images_512.

The reconstruction chain is explicit and reproducible: feature extraction, sequential matching, mapper, model inspection, undistortion, then gs_scene packaging. The main COLMAP settings in the notebook are SIMPLE_PINHOLE, single shared camera, SIFT max features 4096, sequential overlap 10, and quadratic overlap enabled.

A practical detail with high impact is sparse model selection. COLMAP may emit more than one submodel in the sparse directory; the correct model is not guaranteed to be sparse/0. Selection should be based on reconstruction statistics, especially registered images and point support, before undistortion into gs_scene.

## 7. Stage C: Plan B (VGGT) Initialization
Plan B reconstruction is executed in [part1/notebooks/03_planB_vggt.ipynb](part1/notebooks/03_planB_vggt.ipynb), with variant naming vggt_108 and source images from subset_108.

In the current notebook state, the reconstruction path is feedforward without BA. The parameter block reflected in notebook outputs is use_ba=False, fine_tracking=False, query_frame_num=4, and max_query_pts=1024. This setup was chosen for hardware feasibility and memory stability, then converted into a COLMAP-like sparse structure and finally into gs_scene for downstream training.

Because this stage is sensitive to memory and exporter behavior, the notebook includes explicit sparse output checks and structure normalization steps before training handoff.

## 8. Stage D: Multi-Run Training Orchestration
The primary active launcher for standardized runs is [part1/scripts/train/run_part1_12_experiments.sh](part1/scripts/train/run_part1_12_experiments.sh). It executes 12 configurations across 3 scenes, 2 initialization families (colmap_108 and vggt_108), and 2 trainers (3dgs and scaffold), under 40k iterations with test evaluation every 2k iterations.

The script writes logs under each run tree and applies explicit naming tags that are later parsed by the result notebook. For 3DGS, the script also runs render.py and metrics.py after training to ensure final results.json availability.

The extended script [part1/scripts/train/run_part1_all_18_experiments.sh](part1/scripts/train/run_part1_all_18_experiments.sh) exists as a full matrix launcher, but it contains legacy variant tokens in parts of its logic and should be synchronized with current colmap_108 and vggt_108 naming before being treated as the authoritative launcher in future reruns.

## 9. Stage E: Result Parsing and Structured Exports
Result ingestion is handled in [part1/notebooks/04_read_results.ipynb](part1/notebooks/04_read_results.ipynb). The notebook scans [output/part1](output/part1), parses final metrics from results.json, parses convergence records from evaluation lines in logs, and exports clean tables to [results/part1](results/part1).

The central exported files are:

- [results/part1/final/final_metrics_18.csv](results/part1/final/final_metrics_18.csv)
- [results/part1/convergence/all_convergence_metrics.csv](results/part1/convergence/all_convergence_metrics.csv)

These two files are the canonical bridge between raw run directories and downstream plotting/reporting.

## 10. Stage F: Plotting and Reporting Assets
Visualization is produced in [part1/notebooks/05_plot_results_revised.ipynb](part1/notebooks/05_plot_results_revised.ipynb), which reads the CSV exports and writes figure/table assets to [plots/part1](plots/part1).

Main outputs include:

- [plots/part1/main/part1_final_test_metrics_paired_compact.png](plots/part1/main/part1_final_test_metrics_paired_compact.png)
- [plots/part1/main/part1_test_psnr_convergence_scene_by_model.png](plots/part1/main/part1_test_psnr_convergence_scene_by_model.png)
- [plots/part1/appendix/part1_test_l1_convergence_scene_by_model.png](plots/part1/appendix/part1_test_l1_convergence_scene_by_model.png)
- [plots/part1/appendix/part1_train_test_overlay_re10k1.png](plots/part1/appendix/part1_train_test_overlay_re10k1.png)
- [plots/part1/appendix/part1_aggregate_mean_metrics_heatmaps.png](plots/part1/appendix/part1_aggregate_mean_metrics_heatmaps.png)

Tabular exports include:

- [plots/part1/tables/part1_final_metrics_summary.csv](plots/part1/tables/part1_final_metrics_summary.csv)
- [plots/part1/tables/part1_final_metrics_wide_table.csv](plots/part1/tables/part1_final_metrics_wide_table.csv)
- [plots/part1/tables/part1_aggregate_mean_metrics.csv](plots/part1/tables/part1_aggregate_mean_metrics.csv)

## 11. Path and Naming Conventions for Reproducibility
The active run identity convention is:

scene__plan__variant__model__run_name

Examples are visible in final_metrics_18.csv and convergence exports, and they should be treated as immutable keys for joins between result_path and log_path. This strict naming convention prevents mismatch when combining final and convergence metrics across scenes and trainers.

Model output folders are consistently nested as:

output/part1/scene/plan/variant/model/run_name

Log folders are colocated under model/logs, enabling direct traceability from CSV rows to original logs.

## 12. Environment and Execution Responsibility
Part 1 uses multiple environments by design because dependency stacks differ across components. The setup reference is [part1/docs/PART1_SETUP.md](part1/docs/PART1_SETUP.md). In practical orchestration, launcher scripts call conda run explicitly per toolchain, reducing ambiguity caused by interactive shell state.

This environment separation is not a convenience detail; it is required for stable operation of COLMAP, 3DGS CUDA extensions, Scaffold-GS, and VGGT dependencies in the same repository.

## 13. Operational Risks and Guardrails
Three guardrails are necessary for stable reruns.

First, always verify sparse submodel quality before selecting the undistortion input in Plan A. Multiple sparse submodels may coexist, and index-based assumptions can silently select a fragment.

Second, keep variant names synchronized across dataset, output, scripts, and notebooks. Legacy tokens in older planning or scripts can cause path misses or partial run coverage.

Third, maintain the parse contract in the read-results notebook: only final metrics from results.json and evaluation lines from logs should enter the merged tables. This protects convergence analysis from noise in progress logs.

## 14. Reconciliation with Earlier Planning Documents
The three planning documents remain valuable as architectural intent, but current implementation has evolved in three concrete ways.

The first evolution is naming: full and 108 variants are now standard, replacing 279 and 96 naming in earlier drafts. The second evolution is execution orchestration: script-based launching became the dominant path, while manifest-first launch notebooks were deferred. The third evolution is result pipeline maturity: read and plotting notebooks are now fully integrated with an 18-run output structure and stable CSV exports.

This guide therefore serves as the authoritative operational layer: it preserves planning principles while documenting the state that actually runs today.

## 15. Practical Rerun Sequence
A clean rerun should follow the order below:

1. Prepare shared workspace and subset_108 for each scene in [part1/notebooks/01_data_workspace.ipynb](part1/notebooks/01_data_workspace.ipynb).
2. Build Plan A gs_scene for target scene and mode in [part1/notebooks/02_planA_3dgs.ipynb](part1/notebooks/02_planA_3dgs.ipynb).
3. Build Plan B gs_scene for target scene in [part1/notebooks/03_planB_vggt.ipynb](part1/notebooks/03_planB_vggt.ipynb).
4. Launch standardized training runs through [part1/scripts/train/run_part1_12_experiments.sh](part1/scripts/train/run_part1_12_experiments.sh) or a synchronized full-matrix launcher.
5. Parse and export metrics with [part1/notebooks/04_read_results.ipynb](part1/notebooks/04_read_results.ipynb).
6. Generate figures and publication tables with [part1/notebooks/05_plot_results_revised.ipynb](part1/notebooks/05_plot_results_revised.ipynb).

Following this order ensures that every downstream artifact has a traceable upstream source and that all path conventions remain aligned.

## 16. Internal Coverage Check
This consolidated workflow document covers all major requirements requested for a process-oriented Part 1 reference: repository organization, scene-level data preparation, Plan A and Plan B setup logic, launch strategy, output conventions, parsing contracts, plotting outputs, environment constraints, risk controls, and rerun order. It is intentionally distinct from result interpretation documents and is designed as an execution manual for reproducible experimentation.

# Experiment Notebook Plan

## Goal

Split experiment execution and result reading into two notebooks:

1. one notebook for writing and launching shell scripts
2. one notebook for reading final metrics and convergence metrics

Use a lightweight manifest so result-reading does not need to reverse-engineer shell scripts every time.

---

## Notebook 1: Launch Experiments

Suggested file:

```text
part1/notebooks/05_launch_scaffold_experiments.ipynb
```

### Responsibilities

- define experiment configurations
- generate shell scripts
- write output and log paths into the script
- `chmod +x` the generated script
- launch it in the background
- write a manifest file for downstream result reading

### Why this notebook exists

The shell script is the actual execution layer, but later result reading needs a stable mapping from:

- experiment name
- output directory
- log file
- command / script used

This mapping should be saved at launch time, not reconstructed later from memory.

### Suggested notebook inputs

- scene name
- experiment group name
- output base directory
- log directory
- iteration settings
- evaluation interval
- GPU assignment
- experiment list / parameter list

### Suggested outputs

- generated `.sh` file
- background process started
- manifest file written

---

## Manifest Design

Suggested location example:

```text
output/part1/Re10k-1/verification/scaffold_40k/manifest.json
```

### Purpose

The manifest should record the mapping between each experiment and its result/log locations.

### Suggested fields per experiment

```json
{
  "experiment_name": "colmap_vs0_uif4_app0_40k",
  "scene": "Re10k-1",
  "group": "scaffold_40k",
  "output_dir": "~/CV_Project/output/part1/Re10k-1/verification/scaffold_40k/data/colmap_vs0_uif4_app0_40k",
  "log_file": "~/CV_Project/output/part1/Re10k-1/verification/scaffold_40k/data/logs/colmap_vs0_uif4_app0_40k.log",
  "script_file": "~/CV_Project/part1/scripts/train/run_scaffold_40k_verification.sh",
  "iterations": 40000,
  "eval_interval": 2000
}
```

A single manifest can contain a list of experiments.

### Design principle

Result reading should depend on the manifest first, not on ad hoc shell parsing.

---

## Notebook 2: Read Results

Suggested file:

```text
part1/notebooks/06_read_scaffold_results.ipynb
```

### Responsibilities

Read two classes of results:

1. final results from `results.json`
2. convergence results from selected lines in log files

### Input priority

1. manifest
2. output base directory
3. shell script only as fallback

---

## Final Metrics Reading

### Source

Each experiment directory may contain a final result file such as:

```text
.../results.json
```

### Purpose

Extract final test metrics such as:

- final test L1
- final test PSNR
- any additional stable summary fields needed later

### Output form

A dataframe / table with one row per experiment.

Example columns:

- experiment_name
- scene
- group
- final_test_l1
- final_test_psnr
- output_dir

---

## Convergence Metrics Reading

### Source

Read the corresponding log file for each experiment.

Example location:

```text
.../data/logs/<experiment>.log
```

### Important constraint

Do **not** read and print full logs.

The log files are large and contain training progress lines that are not needed for convergence comparison.

### Only keep evaluation lines

The parser should only extract lines like:

```text
[ITER 2000] Evaluating test: L1 ... PSNR ...
[ITER 2000] Evaluating train: L1 ... PSNR ...
```

Ignore progress-bar lines such as:

```text
Training progress: ...
```

### Parsing strategy

- read log files line by line
- keep only lines containing:
  - `Evaluating test:`
  - `Evaluating train:`
- parse:
  - iteration
  - split (`test` / `train`)
  - L1
  - PSNR

### Output form

A dataframe / table with one row per `(experiment, iteration, split)`.

Example columns:

- experiment_name
- iter
- split
- l1
- psnr
- log_file

This table can later be used for convergence plots and controlled comparisons.

---

## Why Manifest Is Useful

Without a manifest, the result notebook must infer mappings by:

- scanning directories
- guessing experiment names from file names
- or parsing shell scripts repeatedly

That is fragile.

With a manifest:

- experiment to output mapping is explicit
- experiment to log mapping is explicit
- result reading becomes reproducible and much lighter

---

## Scope for the First Implementation

The first implementation should focus on:

1. launch notebook for Scaffold verification experiments
2. manifest writing at launch time
3. result notebook for:
   - `results.json`
   - selected evaluation lines in logs

No need yet to handle every experiment type or every historical shell script.

---

## Later Extensions

Possible later extensions:

- support 3DGS experiment launch in the same notebook style
- support reading historical runs without a manifest
- support plotting inside the result notebook
- support exporting merged CSV summaries

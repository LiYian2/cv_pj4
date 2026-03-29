# Part 2 Planning Notes

## What I verified from RegGS code

I checked the actual RegGS code in `third_party/RegGS` instead of relying only on the README.

### 1. Scene input format is full-scene based
`src/entities/datasets.py` defines `Re10KDataset`, and it reads data from:

```text
<input_path>/<scene_name>/
├── images/*.png
├── intrinsics.json
└── cameras.json
```

Important details:
- `images/*.png` are loaded by sorted filename stem.
- `intrinsics.json` stores normalized `fx, fy, cx, cy`; the loader multiplies them by configured `W, H`.
- `cameras.json` stores per-frame `cam_quat` and `cam_trans`, which are converted into poses.
- The loader reads the whole scene directory; it does not expect pre-split `train/` and `test/` folders.

### 2. Train/test split is done inside RegGS
Both of the following files implement the same logic:
- `src/entities/reggs.py`
- `run_refine.py`

They compute frame ids as:

```python
sample_rate = config["sample_rate"]
n_views = config["n_views"]
n_frames = len(self.dataset)
frame_ids = np.arange(n_frames)
test_frame_ids = frame_ids[int(sample_rate/2)::sample_rate]
remain_frame_ids = np.array([i for i in frame_ids if i not in test_frame_ids])
train_frame_ids = remain_frame_ids[np.linspace(0, remain_frame_ids.shape[0] - 1, n_views).astype(int)]
```

So the correct interpretation is:
- the scene directory should keep the full frame sequence;
- RegGS itself derives test frames from `sample_rate`;
- RegGS then picks `n_views` training frames from the remaining frames.

### 3. Practical consequence
The statement below is correct:

> Scene directories should keep the full sequence. We should **not** sparsify the folder first and then also keep RegGS sampling enabled. The actual train/test split should be controlled by RegGS through `sample_rate` and `n_views`.

If we manually pre-prune the directory and also keep RegGS sampling, we would effectively sample twice and change the intended protocol.

---

## Planning principles for Part 2

Part 2 should follow the same file-management logic as Part 1, but adapted to RegGS and the no-pose setting.

The main goals are:
- keep the project tree readable and predictable;
- separate raw outputs from summarized results and plots;
- make preprocessing reproducible;
- avoid mixing dataset conversion, config generation, training, and analysis in one place.

I will treat Part 2 as a small pipeline with four layers:
1. data preparation;
2. config generation;
3. RegGS execution;
4. metric collection and plotting.

---

## Proposed Part 2 directory logic

```text
part2/
├── docs/          # plans, protocol notes, adapter spec, setup notes
├── notebooks/     # preprocessing and analysis notebooks
├── scripts/       # lightweight runners / helpers / batch launchers
├── configs/       # generated or curated RegGS yaml configs
└── reports/       # optional markdown summaries for this part

dataset/<scene>/part2/   # scene-local RegGS-ready inputs, aligned with Part 1 organization
output/part2/            # raw RegGS outputs: infer/refine/metric artifacts, logs, checkpoints
results/part2/           # cleaned csv summaries and extracted metrics
plots/part2/             # figures for report / appendix
```

### Responsibilities
- `dataset/<scene>/part2/`: only RegGS-ready scene inputs for that scene, not model outputs.
- `output/part2/`: raw run artifacts from RegGS.
- `results/part2/`: collected metrics in csv form.
- `plots/part2/`: figures generated from `results/part2/`.

This mirrors the Part 1 logic and the existing per-scene layout already used in `dataset/<scene>/part1`:
- `dataset` = inputs / prepared scene data
- `output` = raw experiment outputs
- `results` = summarized tables
- `plots` = visualization outputs

---

## Data preparation strategy

You asked to keep preprocessing notebook-first, similar to Part 1. I agree.

So for Part 2, I plan to make the **primary preparation entry point an ipynb notebook**, not a standalone `.py` script.

### Notebook-first plan
A notebook in `part2/notebooks/` will:
- read the source scene from your current organized data;
- export a RegGS-ready scene directory under `dataset/<scene>/part2/`;
- resize images consistently;
- generate `intrinsics.json` with normalized parameters;
- generate `cameras.json` with `cam_quat` and `cam_trans`;
- keep frame naming sortable, e.g. `0000.png`, `0001.png`, ...

I may still factor some helper logic into importable utilities later if the notebook gets too long, but the preparation workflow itself will be notebook-driven.

### Why this fits Part 2
Part 2 preprocessing is not just file copying. It is where we must carefully control:
- image resolution;
- normalized intrinsics;
- pose export format;
- frame ordering;
- dataset-specific conversion differences.

Using a notebook first makes this easier to inspect and debug scene by scene.

---

## Dataset adapter idea

Before writing lots of code, I want a small adapter spec in `docs/`.

Reason: Re10k, DL3DV, and Waymo-405841 do not necessarily expose source data in the exact same format. If we do not define a common adapter contract first, the notebook will quickly become full of dataset-specific branching.

### Proposed adapter contract
For each upstream dataset, the adapter should define:
- where images come from;
- where GT poses come from;
- where intrinsics come from;
- how frames should be ordered;
- whether resize is required;
- how normalized intrinsics should be written after resize;
- how the exported scene name should be formed.

This will keep the notebook clean: the notebook can call a dataset-specific adapter layer, but always write the same RegGS scene format.

---

## Config generation strategy

A separate config step is still useful, but lighter than the preprocessing notebook.

The configs should control:
- `dataset_name`
- `n_views`
- `sample_rate`
- `new_submap_every`
- `nopo_checkpoint`
- `data.input_path`
- `data.output_path`
- `data.scene_name`
- camera size fields in `cam`

Based on the prompt, the intended sparse protocol is:
- Waymo-405841: `sample_rate = 10`
- DL3DV-2 and Re10k-1: `sample_rate = 30`

One thing to keep explicit in the docs: `cam.fx/fy/cx/cy` inside config are not the authoritative final intrinsics if `intrinsics.json` is loaded and overrides them. The true consistency must come from the exported `intrinsics.json`.

---

## Run pipeline plan

For execution, I plan to keep the official three-stage order visible:

```text
python run_infer.py <config>
python run_refine.py --checkpoint_path <output_scene_dir>
python run_metric.py --checkpoint_path <output_scene_dir>
```

I do not want to bury these stages too deeply. A thin launcher is enough.

The launcher should support:
- single scene;
- batch over multiple scenes;
- choosing GPU;
- skip completed scenes;
- logging failures clearly.

---

## Results and analysis plan

As with Part 1, I do not want metric reading to depend on manually browsing logs.

So Part 2 should eventually have:
- a results collector that extracts render metrics and ATE-related outputs into csv;
- a plotting notebook that reads only from `results/part2/`, not from raw `output/part2/`.

That keeps the workflow clean:
1. run experiments;
2. collect to csv;
3. plot from csv.

---

## Immediate next steps I propose

I do **not** plan to jump straight into all scripts at once.

I would do the following in order:

1. create the Part 2 directory skeleton so it mirrors Part 1 cleanly;
2. write a short adapter/spec document in `docs/`;
3. create the first preprocessing notebook for exporting RegGS-ready scenes;
4. then add config generation and batch-running helpers.

This keeps the project organized from the beginning instead of cleaning it up afterward.

---

## Summary

My current understanding is:
- the RegGS full-sequence interpretation is correct;
- we should not pre-sparsify scene folders if we still rely on `sample_rate` and `n_views`;
- Part 2 should reuse the Part 1 management philosophy;
- preprocessing should be notebook-first;
- `dataset / output / results / plots` separation should be preserved for Part 2 as well.

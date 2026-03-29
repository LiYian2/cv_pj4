import json
from pathlib import Path

cv = Path('/home/bzhang512/CV_Project')

# Ensure canonical per-scene part2 dirs exist
for scene in ['Re10k-1', 'DL3DV-2', '405841']:
    (cv / 'dataset' / scene / 'part2').mkdir(parents=True, exist_ok=True)

# Update PART2_PLAN.md
plan = cv / 'part2' / 'docs' / 'PART2_PLAN.md'
text = plan.read_text(encoding='utf-8')
text = text.replace(
    'dataset/<scene>/part2/   # scene-local RegGS-ready inputs, aligned with Part 1 organization\noutput/part2/            # raw RegGS outputs: infer/refine/metric artifacts, logs, checkpoints\nresults/part2/           # cleaned csv summaries and extracted metrics\nplots/part2/             # figures for report / appendix',
    'dataset/<scene>/part2/   # scene-local RegGS-ready inputs, aligned with Part 1 organization\noutput/part2/            # raw RegGS outputs: infer/refine/metric artifacts, logs, checkpoints\nresults/part2/           # cleaned csv summaries and extracted metrics\nplots/part2/             # figures for report / appendix'
)
text = text.replace(
    'dataset/part2/     # RegGS-ready scene directories\noutput/part2/      # raw RegGS outputs: infer/refine/metric artifacts, logs, checkpoints\nresults/part2/     # cleaned csv summaries and extracted metrics\nplots/part2/       # figures for report / appendix',
    'dataset/<scene>/part2/   # scene-local RegGS-ready inputs, aligned with Part 1 organization\noutput/part2/            # raw RegGS outputs: infer/refine/metric artifacts, logs, checkpoints\nresults/part2/           # cleaned csv summaries and extracted metrics\nplots/part2/             # figures for report / appendix'
)
text = text.replace(
    '- `dataset/part2/`: only RegGS-ready scene inputs, not model outputs.',
    '- `dataset/<scene>/part2/`: only RegGS-ready scene inputs for that scene, not model outputs.'
)
text = text.replace(
    'This mirrors the Part 1 logic:\n- `dataset` = inputs / prepared scene data',
    'This mirrors the Part 1 logic and the existing per-scene layout already used in `dataset/<scene>/part1`:\n- `dataset` = inputs / prepared scene data'
)
text = text.replace(
    '- export a RegGS-ready scene directory under `dataset/part2/`;',
    '- export a RegGS-ready scene directory under `dataset/<scene>/part2/`;'
)
plan.write_text(text, encoding='utf-8')

# Update PART2_ADAPTER_SPEC.md
spec = cv / 'part2' / 'docs' / 'PART2_ADAPTER_SPEC.md'
text = spec.read_text(encoding='utf-8')
text = text.replace(
    'dataset/part2/   # RegGS-ready input scenes\noutput/part2/    # raw RegGS outputs and logs\nresults/part2/   # collected metrics / csv summaries\nplots/part2/     # figures derived from results',
    'dataset/Re10k-1/part2/\ndataset/DL3DV-2/part2/\ndataset/405841/part2/   # canonical location for scene-local RegGS-ready inputs\noutput/part2/           # raw RegGS outputs and logs\nresults/part2/          # collected metrics / csv summaries\nplots/part2/            # figures derived from results'
)
text = text.replace(
    '- export one RegGS-ready scene to `dataset/part2/`;',
    '- export one RegGS-ready scene to `dataset/<scene>/part2/`;'
)
spec.write_text(text, encoding='utf-8')

# Update PART2_BUILD_ORDER.md
build = cv / 'part2' / 'docs' / 'PART2_BUILD_ORDER.md'
text = build.read_text(encoding='utf-8')
if 'Canonical prepared-scene path' not in text:
    text += '\n## Path convention\n\nCanonical prepared-scene path: `dataset/<scene>/part2/`\n\nThis matches the existing Part 1 organization (`dataset/<scene>/part1/`) and is preferred over a single shared `dataset/part2/` bucket.\n'
build.write_text(text, encoding='utf-8')

# Write source format analysis doc
analysis = cv / 'part2' / 'docs' / 'PART2_SOURCE_FORMAT_ANALYSIS.md'
analysis.write_text(
    '''# Part 2 Source Format Analysis

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
''',
    encoding='utf-8'
)

# Update notebook
nb = cv / 'part2' / 'notebooks' / '01_prepare_reggs_scene.ipynb'
obj = json.loads(nb.read_text(encoding='utf-8'))
obj['cells'][1]['source'] = [
    "from pathlib import Path\n",
    "import json\n",
    "import math\n",
    "import shutil\n",
    "import os\n",
    "\n",
    "CV_ROOT = Path('~/CV_Project').expanduser()\n",
    "DATASET_ROOT = CV_ROOT / 'dataset'\n",
    "\n",
    "print('Canonical prepared-scene location: dataset/<scene>/part2')\n",
]
obj['cells'][3]['source'] = [
    "# Example placeholders. Replace after adapter logic is finalized.\n",
    "SOURCE_SCENE = {\n",
    "    'dataset': 'Re10k-1',\n",
    "    'part': 'part1',\n",
    "    'plan': 'planA',\n",
    "    'variant': 'colmap_full',\n",
    "}\n",
    "\n",
    "PART2_DATASET_ROOT = DATASET_ROOT / SOURCE_SCENE['dataset'] / 'part2'\n",
    "PART2_DATASET_ROOT.mkdir(parents=True, exist_ok=True)\n",
    "EXPORT_SCENE_NAME = 're10k1_reggs_scene'\n",
    "EXPORT_ROOT = PART2_DATASET_ROOT / EXPORT_SCENE_NAME\n",
    "IMAGES_DIR = EXPORT_ROOT / 'images'\n",
    "\n",
    "print('source =', SOURCE_SCENE)\n",
    "print('part2 root =', PART2_DATASET_ROOT)\n",
    "print('export root =', EXPORT_ROOT)\n",
]
nb.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

print('updated docs and notebook successfully')

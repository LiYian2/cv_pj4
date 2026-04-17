#!/usr/bin/env bash
set -eo pipefail

source /home/bzhang512/miniconda3/etc/profile.d/conda.sh
conda activate s3po-gs

export CUDA_VISIBLE_DEVICES=1
export TMPDIR=/data/bzhang512/tmp
mkdir -p "$TMPDIR"
export S3PO_EXPORT_INTERNAL_CACHE=1
export PYTHONUNBUFFERED=1

RUN_CFG=/home/bzhang512/CV_Project/part2_s3po/configs/s3po_dl3dv2_full_full_internal_cache.yaml
SAVE_BASE=/data2/bzhang512/CV_Project/output/part2_s3po/dl3dv-2/s3po_dl3dv-2_full_internal_cache
GROUP_DIR=$SAVE_BASE/DL3DV-2_part2_s3po
REPLAY_LABEL=after_opt_sameply
mkdir -p "$GROUP_DIR"

cd /home/bzhang512/CV_Project/third_party/S3PO-GS
/home/bzhang512/miniconda3/envs/s3po-gs/bin/python slam.py --config "$RUN_CFG"

RUN_ROOT=$(
/home/bzhang512/miniconda3/envs/s3po-gs/bin/python - <<'PY2'
from pathlib import Path
root = Path('/data2/bzhang512/CV_Project/output/part2_s3po/dl3dv-2/s3po_dl3dv-2_full_internal_cache/DL3DV-2_part2_s3po')
runs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
if not runs:
    raise SystemExit('No run directories found under output group dir')
print(runs[-1])
PY2
)
export RUN_ROOT

/home/bzhang512/miniconda3/envs/s3po-gs/bin/python - <<'PY2'
import json, os
from pathlib import Path
run_root = Path(os.environ['RUN_ROOT'])
required = [
    run_root / 'config.yml',
    run_root / 'point_cloud/final/point_cloud.ply',
    run_root / 'psnr/before_opt/final_result.json',
    run_root / 'psnr/after_opt/final_result.json',
    run_root / 'internal_eval_cache/manifest.json',
    run_root / 'internal_eval_cache/camera_states.json',
    run_root / 'internal_eval_cache/before_opt',
    run_root / 'internal_eval_cache/after_opt',
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit('Missing required artifacts: ' + json.dumps(missing, indent=2))
print('PART2_DONE', run_root)
PY2

cd /home/bzhang512/CV_Project/part3_BRPO
export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO:${PYTHONPATH:-}
/home/bzhang512/miniconda3/envs/s3po-gs/bin/python scripts/replay_internal_eval.py \
  --internal-cache-root "$RUN_ROOT/internal_eval_cache" \
  --stage-tag after_opt \
  --ply-path "$RUN_ROOT/internal_eval_cache/after_opt/point_cloud/point_cloud.ply" \
  --config "$RUN_ROOT/config.yml" \
  --save-dir "$RUN_ROOT/internal_eval_cache/replay_eval/$REPLAY_LABEL"

/home/bzhang512/miniconda3/envs/s3po-gs/bin/python - <<'PY2'
import json, os
from pathlib import Path
run_root = Path(os.environ['RUN_ROOT'])
replay_dir = run_root / 'internal_eval_cache/replay_eval/after_opt_sameply'
meta = json.loads((replay_dir / 'replay_eval_meta.json').read_text())
compare = meta['compare_to_internal']
if compare is None:
    raise SystemExit('Replay compare_to_internal missing')
delta = compare['delta']
stage_meta = json.loads((run_root / 'internal_eval_cache/after_opt/stage_meta.json').read_text())
passed = (
    delta['num_frames'] == 0 and
    abs(delta['psnr']) <= 0.02 and
    abs(delta['ssim']) <= 5e-4 and
    abs(delta['lpips']) <= 5e-4
)
summary = {
    'run_root': str(run_root),
    'replay_dir': str(replay_dir),
    'passed': bool(passed),
    'frame_count_expected': int(stage_meta['num_rendered_non_kf_frames']),
    'frame_count_replay': int(compare['replay_eval']['num_frames']),
    'internal_eval': compare['internal_eval'],
    'replay_eval': compare['replay_eval'],
    'delta': delta,
}
out_path = replay_dir / 'parity_summary.json'
out_path.write_text(json.dumps(summary, indent=2))
print('REPLAY_DONE', out_path)
print(json.dumps(summary, indent=2))
if not passed:
    raise SystemExit('Replay parity gate failed')
print('ALL_DONE', run_root)
PY2

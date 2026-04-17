#!/usr/bin/env bash
set -eo pipefail

RUN_ROOT=/data2/bzhang512/CV_Project/output/part2_s3po/dl3dv-2/s3po_dl3dv-2_full_internal_cache/DL3DV-2_part2_s3po/2026-04-17-21-21-00
INTERNAL_CACHE_ROOT=$RUN_ROOT/internal_eval_cache
SELECT_ROOT=$RUN_ROOT/internal_prepare/dl3dv2__internal_afteropt__signal_select_e1
CANONICAL_ROOT=$RUN_ROOT/internal_prepare/dl3dv2__internal_afteropt__brpo_proto_v4_stage3
SELECTION_MANIFEST=$SELECT_ROOT/manifests/signal_aware_selection_manifest.json
SIGNAL_ROOT=$CANONICAL_ROOT/signal_v2

source /home/bzhang512/miniconda3/etc/profile.d/conda.sh
export TMPDIR=/data/bzhang512/tmp
mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1

canonical_ok=0
if [ -f "$CANONICAL_ROOT/manifests/selection_summary.json" ] && [ -f "$CANONICAL_ROOT/pseudo_cache/manifest.json" ]; then
  if /home/bzhang512/miniconda3/envs/s3po-gs/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('/data2/bzhang512/CV_Project/output/part2_s3po/dl3dv-2/s3po_dl3dv-2_full_internal_cache/DL3DV-2_part2_s3po/2026-04-17-21-21-00/internal_prepare/dl3dv2__internal_afteropt__brpo_proto_v4_stage3')
summary = json.loads((root / 'manifests' / 'selection_summary.json').read_text())
manifest = json.loads((root / 'pseudo_cache' / 'manifest.json').read_text())
selected = summary.get('selected_frame_ids', [])
if not selected:
    raise SystemExit(1)
first_id = int(selected[0])
required = [
    root / 'manifests',
    root / 'difix',
    root / 'fusion',
    root / 'verification',
    root / 'pseudo_cache' / 'manifest.json',
    root / 'pseudo_cache' / 'samples' / str(first_id) / 'camera.json',
    root / 'pseudo_cache' / 'samples' / str(first_id) / 'target_rgb_fused.png',
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(1)
print('ok')
PY
  then
    canonical_ok=1
  fi
fi

if [ "$canonical_ok" -ne 1 ]; then
  conda activate reggs
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO:${PYTHONPATH:-}
  cd /home/bzhang512/CV_Project/part3_BRPO
  /home/bzhang512/miniconda3/envs/reggs/bin/python scripts/prepare_stage1_difix_dataset_s3po_internal.py \
    --stage all \
    --internal-cache-root "$INTERNAL_CACHE_ROOT" \
    --run-key dl3dv2__internal_afteropt__brpo_proto_v4_stage3 \
    --scene-name DL3DV-2 \
    --stage-tag after_opt \
    --selection-manifest "$SELECTION_MANIFEST" \
    --target-rgb-source difix
fi

/home/bzhang512/miniconda3/envs/s3po-gs/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('/data2/bzhang512/CV_Project/output/part2_s3po/dl3dv-2/s3po_dl3dv-2_full_internal_cache/DL3DV-2_part2_s3po/2026-04-17-21-21-00/internal_prepare/dl3dv2__internal_afteropt__brpo_proto_v4_stage3')
summary = json.loads((root / 'manifests' / 'selection_summary.json').read_text())
manifest = json.loads((root / 'pseudo_cache' / 'manifest.json').read_text())
selected = summary['selected_frame_ids']
first_id = int(selected[0])
required = [
    root / 'manifests',
    root / 'difix',
    root / 'fusion',
    root / 'verification',
    root / 'pseudo_cache' / 'manifest.json',
    root / 'pseudo_cache' / 'samples' / str(first_id) / 'camera.json',
    root / 'pseudo_cache' / 'samples' / str(first_id) / 'target_rgb_fused.png',
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit('Canonical prepare root missing required artifacts: ' + json.dumps(missing, indent=2))
out = {
    'canonical_root': str(root),
    'num_selected': int(summary['num_selected']),
    'num_unique_gaps': int(summary['num_unique_gaps']),
    'selected_frame_ids': selected,
    'pseudo_cache_num_samples': int(manifest['num_samples']),
    'first_sample_dir': str(root / 'pseudo_cache' / 'samples' / str(first_id)),
}
print('PHASEE_DONE', json.dumps(out, ensure_ascii=False))
PY

signal_ok=0
if [ -f "$SIGNAL_ROOT/summary.json" ] && [ -f "$SIGNAL_ROOT/summary_meta.json" ]; then
  if /home/bzhang512/miniconda3/envs/s3po-gs/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('/data2/bzhang512/CV_Project/output/part2_s3po/dl3dv-2/s3po_dl3dv-2_full_internal_cache/DL3DV-2_part2_s3po/2026-04-17-21-21-00/internal_prepare/dl3dv2__internal_afteropt__brpo_proto_v4_stage3/signal_v2')
meta = json.loads((root / 'summary_meta.json').read_text())
frame_ids = meta.get('frame_ids', [])
if not frame_ids:
    raise SystemExit(1)
first_id = int(frame_ids[0])
required = [
    root / f'frame_{first_id:04d}' / 'raw_rgb_confidence_v2.npy',
    root / f'frame_{first_id:04d}' / 'target_depth_for_refine_v2_brpo.npy',
    root / f'frame_{first_id:04d}' / 'depth_supervision_mask_v2_brpo.npy',
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(1)
print('ok')
PY
  then
    signal_ok=1
  fi
fi

if [ "$signal_ok" -ne 1 ]; then
  conda deactivate || true
  conda activate s3po-gs
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO:${PYTHONPATH:-}
  cd /home/bzhang512/CV_Project/part3_BRPO
  /home/bzhang512/miniconda3/envs/s3po-gs/bin/python scripts/build_brpo_v2_signal_from_internal_cache.py \
    --internal-cache-root "$INTERNAL_CACHE_ROOT" \
    --prepare-root "$CANONICAL_ROOT" \
    --stage-tag after_opt
fi

/home/bzhang512/miniconda3/envs/s3po-gs/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('/data2/bzhang512/CV_Project/output/part2_s3po/dl3dv-2/s3po_dl3dv-2_full_internal_cache/DL3DV-2_part2_s3po/2026-04-17-21-21-00/internal_prepare/dl3dv2__internal_afteropt__brpo_proto_v4_stage3/signal_v2')
summary = json.loads((root / 'summary.json').read_text())
meta = json.loads((root / 'summary_meta.json').read_text())
frame_ids = [int(x) for x in meta['frame_ids']]
first_id = frame_ids[0]
required = [
    root / f'frame_{first_id:04d}' / 'raw_rgb_confidence_v2.npy',
    root / f'frame_{first_id:04d}' / 'target_depth_for_refine_v2_brpo.npy',
    root / f'frame_{first_id:04d}' / 'depth_supervision_mask_v2_brpo.npy',
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit('signal_v2 missing required artifacts: ' + json.dumps(missing, indent=2))
out = {
    'signal_root': str(root),
    'num_frames': int(meta['num_frames']),
    'frame_ids': frame_ids,
    'matcher_size': int(meta['matcher_size']),
    'min_rgb_conf_for_depth': float(meta['min_rgb_conf_for_depth']),
    'first_frame_dir': str(root / f'frame_{first_id:04d}'),
    'first_frame_summary': summary[0] if summary else None,
}
print('PHASEF_DONE', json.dumps(out, ensure_ascii=False))
print('ALL_DONE', str(root))
PY

#!/usr/bin/env bash
set -eo pipefail

RUN_ROOT=/data2/bzhang512/CV_Project/output/part2_s3po/dl3dv-2/s3po_dl3dv-2_full_internal_cache/DL3DV-2_part2_s3po/2026-04-17-21-21-00
INTERNAL_CACHE_ROOT=$RUN_ROOT/internal_eval_cache
COARSE_ROOT=$RUN_ROOT/internal_prepare/dl3dv2__internal_afteropt__candidate_fusion_v1
SELECT_ROOT=$RUN_ROOT/internal_prepare/dl3dv2__internal_afteropt__signal_select_e1

source /home/bzhang512/miniconda3/etc/profile.d/conda.sh
export TMPDIR=/data/bzhang512/tmp
mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1

/home/bzhang512/miniconda3/envs/s3po-gs/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('/data2/bzhang512/CV_Project/output/part2_s3po/dl3dv-2/s3po_dl3dv-2_full_internal_cache/DL3DV-2_part2_s3po/2026-04-17-21-21-00/internal_prepare/dl3dv2__internal_afteropt__candidate_fusion_v1')
summary = json.loads((root / 'manifests' / 'selection_summary.json').read_text())
manifest = json.loads((root / 'pseudo_cache' / 'manifest.json').read_text())
fusion_manifest = json.loads((root / 'manifests' / 'fusion_manifest.json').read_text())
samples = fusion_manifest.get('samples', [])
if not samples:
    raise SystemExit('Coarse root fusion_manifest has no samples')
first_fused = Path(samples[0]['target_rgb_fused_path'])
if not first_fused.exists():
    raise SystemExit(f'Missing fused artifact: {first_fused}')
out = {
    'coarse_root': str(root),
    'num_selected': int(summary.get('num_selected', 0)),
    'num_unique_gaps': int(summary.get('num_unique_gaps', 0)),
    'selected_frame_ids': summary.get('selected_frame_ids', []),
    'pseudo_cache_num_samples': len(manifest.get('samples', [])),
    'first_fused': str(first_fused),
}
print('PHASEC_DONE', json.dumps(out, ensure_ascii=False))
PY

if [ ! -f "$SELECT_ROOT/reports/signal_aware_selection_report.json" ] || [ ! -f "$SELECT_ROOT/manifests/signal_aware_selection_manifest.json" ] || [ ! -f "$SELECT_ROOT/manifests/selection_summary.json" ]; then
  conda activate s3po-gs
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO:${PYTHONPATH:-}
  cd /home/bzhang512/CV_Project/part3_BRPO
  /home/bzhang512/miniconda3/envs/s3po-gs/bin/python scripts/select_signal_aware_pseudos.py \
    --internal-cache-root "$INTERNAL_CACHE_ROOT" \
    --stage-tag after_opt \
    --output-root "$SELECT_ROOT" \
    --pseudo-fused-root "$COARSE_ROOT/fusion/samples" \
    --topk-per-gap 1
fi

/home/bzhang512/miniconda3/envs/s3po-gs/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('/data2/bzhang512/CV_Project/output/part2_s3po/dl3dv-2/s3po_dl3dv-2_full_internal_cache/DL3DV-2_part2_s3po/2026-04-17-21-21-00/internal_prepare/dl3dv2__internal_afteropt__signal_select_e1')
report = json.loads((root / 'reports' / 'signal_aware_selection_report.json').read_text())
manifest = json.loads((root / 'manifests' / 'signal_aware_selection_manifest.json').read_text())
summary = json.loads((root / 'manifests' / 'selection_summary.json').read_text())
selected_ids = [int(x['frame_id']) for x in manifest]
out = {
    'select_root': str(root),
    'num_selected': len(manifest),
    'selected_frame_ids': selected_ids,
    'changed_gaps': report.get('changed_gaps'),
    'midpoint_frame_ids': report.get('midpoint_frame_ids'),
    'selection_summary': summary,
}
print('PHASED_DONE', json.dumps(out, ensure_ascii=False))
print('ALL_DONE', str(root))
PY

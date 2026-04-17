#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT=/home/bzhang512/CV_Project/part3_BRPO
PY=/home/bzhang512/miniconda3/envs/s3po-gs/bin/python
RUN_ROOT=/data2/bzhang512/CV_Project/output/part2_s3po/dl3dv-2/s3po_dl3dv-2_full_internal_cache/DL3DV-2_part2_s3po/2026-04-17-21-21-00
INTERNAL_CACHE_ROOT=$RUN_ROOT/internal_eval_cache
PREPARE_ROOT=$RUN_ROOT/internal_prepare/dl3dv2__internal_afteropt__brpo_proto_v4_stage3
BASE_PLY=$INTERNAL_CACHE_ROOT/after_opt/point_cloud/point_cloud.ply
PSEUDO_CACHE=$PREPARE_ROOT/pseudo_cache
SIGNAL_V2_ROOT=$PREPARE_ROOT/signal_v2
TRAIN_MANIFEST=/home/bzhang512/CV_Project/dataset/DL3DV-2/part2_s3po/sparse/split_manifest.json
TRAIN_RGB_DIR=/home/bzhang512/CV_Project/dataset/DL3DV-2/part2_s3po/sparse/rgb
EXP_ROOT=/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_dl3dv_phaseg_baseline_spgm_e1
STAGEA_DIR=$EXP_ROOT/stageA_v2rgbonly_80
STAGEA5_DIR=$EXP_ROOT/stageA5_v2rgbonly_xyz_gated_rgb0192_80
BASELINE_DIR=$EXP_ROOT/canonical_baseline_post40_lr03_120
SPGM_DIR=$EXP_ROOT/spgm_repair_a_keep111_eta0_wf025

source /home/bzhang512/miniconda3/etc/profile.d/conda.sh
conda activate s3po-gs
export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO:${PYTHONPATH:-}
export TMPDIR=/data/bzhang512/tmp
mkdir -p "$TMPDIR" "$EXP_ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

cat > "$EXP_ROOT/RUN_PLAN.txt" <<'EOF'
DL3DV Phase G baseline + SPGM bring-up (2026-04-17)
Goal:
- Use canonical DL3DV signal_v2 root and selected pseudo cache.
- Run the first full refine ladder needed for DL3DV case tracking:
  1. StageA v2 RGB-only handoff
  2. StageA.5 gated_rgb0192 anchor
  3. Canonical bounded StageB baseline post40_lr03_120
  4. SPGM repair A on the same StageA.5 handoff / StageB protocol
- Replay all outputs at the end with the same internal evaluator.

Protocol choices carried from the current Re10k mainline:
- signal pipeline: brpo_v2
- RGB mask: brpo_v2_raw
- depth mask: train_mask
- depth target: target_depth_for_refine_v2
- depth loss: source_aware
- abs prior: t=3.0, r=0.1, charbonnier, render_depth_trainmask_median
- StageA.5 params: xyz-only, lr_xyz=1e-4, densify/prune disabled
- StageB baseline: post40_lr03_120 on top of gated_rgb0192 A.5 handoff
- SPGM first target: repair A dense_keep keep111 eta0 wf025 (v1 ranking)
EOF

run_replay() {
  local ply_path="$1"
  local label="$2"
  local save_dir="$3"
  if [[ -f "$save_dir/replay_eval.json" ]]; then
    echo "[skip] replay exists for $label"
    return
  fi
  mkdir -p "$save_dir"
  "$PY" "$PROJECT_ROOT/scripts/replay_internal_eval.py" \
    --internal-cache-root "$INTERNAL_CACHE_ROOT" \
    --stage-tag after_opt \
    --ply-path "$ply_path" \
    --label "$label" \
    --save-dir "$save_dir"
}

COMMON_STAGE_ARGS=(
  --pseudo_cache "$PSEUDO_CACHE"
  --target_side fused
  --confidence_mask_source brpo
  --signal_pipeline brpo_v2
  --signal_v2_root "$SIGNAL_V2_ROOT"
  --stageA_iters 80
  --num_pseudo_views 4
  --stageA_rgb_mask_mode brpo_v2_raw
  --stageA_depth_mask_mode train_mask
  --stageA_target_depth_mode target_depth_for_refine
  --stageA_depth_loss_mode source_aware
  --stageA_lambda_depth_seed 1.0
  --stageA_lambda_depth_dense 0.35
  --stageA_lambda_depth_fallback 0.0
  --stageA_lambda_abs_t 3.0
  --stageA_lambda_abs_r 0.1
  --stageA_abs_pose_robust charbonnier
  --stageA_abs_pose_scale_source render_depth_trainmask_median
  --seed 0
)

if [[ -f "$STAGEA_DIR/stageA_history.json" ]]; then
  echo "[skip] StageA exists"
else
  mkdir -p "$STAGEA_DIR"
  "$PY" "$PROJECT_ROOT/scripts/run_pseudo_refinement_v2.py" \
    --ply_path "$BASE_PLY" \
    --output_dir "$STAGEA_DIR" \
    --stage_mode stageA \
    "${COMMON_STAGE_ARGS[@]}"
fi

echo "STAGEA_DONE $STAGEA_DIR"

if [[ -f "$STAGEA5_DIR/stageA_history.json" ]]; then
  echo "[skip] StageA.5 exists"
else
  mkdir -p "$STAGEA5_DIR"
  "$PY" "$PROJECT_ROOT/scripts/run_pseudo_refinement_v2.py" \
    --ply_path "$BASE_PLY" \
    --output_dir "$STAGEA5_DIR" \
    --stage_mode stageA5 \
    --init_pseudo_camera_states_json "$STAGEA_DIR/pseudo_camera_states_stageA.json" \
    --init_pseudo_reference_mode keep \
    --stageA5_trainable_params xyz \
    --stageA5_lr_xyz 1e-4 \
    --pseudo_local_gating hard_visible_union_signal \
    --pseudo_local_gating_params xyz \
    --pseudo_local_gating_min_verified_ratio 0.01 \
    --pseudo_local_gating_min_rgb_mask_ratio 0.0192 \
    --pseudo_local_gating_max_fallback_ratio 0.995 \
    "${COMMON_STAGE_ARGS[@]}"
fi

echo "STAGEA5_DONE $STAGEA5_DIR"

COMMON_STAGEB_ARGS=(
  --ply_path "$STAGEA5_DIR/refined_gaussians.ply"
  --output_dir DUMMY
  --stage_mode stageB
  --stageA_iters 0
  --stageB_iters 120
  --num_pseudo_views 4
  --num_real_views 2
  --lambda_real 1.0
  --lambda_pseudo 1.0
  --stageA_rgb_mask_mode brpo_v2_raw
  --stageA_depth_mask_mode train_mask
  --stageA_target_depth_mode target_depth_for_refine
  --stageA_depth_loss_mode source_aware
  --stageA_lambda_depth_seed 1.0
  --stageA_lambda_depth_dense 0.35
  --stageA_lambda_depth_fallback 0.0
  --stageA_lambda_abs_t 3.0
  --stageA_lambda_abs_r 0.1
  --stageA_abs_pose_robust charbonnier
  --stageA_abs_pose_scale_source render_depth_trainmask_median
  --stageA5_trainable_params xyz
  --stageA5_lr_xyz 1e-4
  --init_pseudo_camera_states_json "$STAGEA5_DIR/pseudo_camera_states_final.json"
  --init_pseudo_reference_mode keep
  --train_manifest "$TRAIN_MANIFEST"
  --train_rgb_dir "$TRAIN_RGB_DIR"
  --seed 0
  --pseudo_cache "$PSEUDO_CACHE"
  --target_side fused
  --confidence_mask_source brpo
  --signal_pipeline brpo_v2
  --signal_v2_root "$SIGNAL_V2_ROOT"
  --stageB_post_switch_iter 40
  --stageB_post_lr_scale_xyz 0.3
)

if [[ -f "$BASELINE_DIR/stageB_history.json" ]]; then
  echo "[skip] StageB baseline exists"
else
  mkdir -p "$BASELINE_DIR"
  "$PY" "$PROJECT_ROOT/scripts/run_pseudo_refinement_v2.py" \
    --ply_path "$STAGEA5_DIR/refined_gaussians.ply" \
    --output_dir "$BASELINE_DIR" \
    --stage_mode stageB \
    --stageA_iters 0 \
    --stageB_iters 120 \
    --num_pseudo_views 4 \
    --num_real_views 2 \
    --lambda_real 1.0 \
    --lambda_pseudo 1.0 \
    --stageA_rgb_mask_mode brpo_v2_raw \
    --stageA_depth_mask_mode train_mask \
    --stageA_target_depth_mode target_depth_for_refine \
    --stageA_depth_loss_mode source_aware \
    --stageA_lambda_depth_seed 1.0 \
    --stageA_lambda_depth_dense 0.35 \
    --stageA_lambda_depth_fallback 0.0 \
    --stageA_lambda_abs_t 3.0 \
    --stageA_lambda_abs_r 0.1 \
    --stageA_abs_pose_robust charbonnier \
    --stageA_abs_pose_scale_source render_depth_trainmask_median \
    --stageA5_trainable_params xyz \
    --stageA5_lr_xyz 1e-4 \
    --init_pseudo_camera_states_json "$STAGEA5_DIR/pseudo_camera_states_final.json" \
    --init_pseudo_reference_mode keep \
    --train_manifest "$TRAIN_MANIFEST" \
    --train_rgb_dir "$TRAIN_RGB_DIR" \
    --seed 0 \
    --pseudo_cache "$PSEUDO_CACHE" \
    --target_side fused \
    --confidence_mask_source brpo \
    --signal_pipeline brpo_v2 \
    --signal_v2_root "$SIGNAL_V2_ROOT" \
    --pseudo_local_gating hard_visible_union_signal \
    --pseudo_local_gating_params xyz \
    --pseudo_local_gating_min_verified_ratio 0.01 \
    --pseudo_local_gating_min_rgb_mask_ratio 0.0192 \
    --pseudo_local_gating_max_fallback_ratio 0.995 \
    --stageB_post_switch_iter 40 \
    --stageB_post_lr_scale_xyz 0.3
fi

echo "STAGEB_BASELINE_DONE $BASELINE_DIR"

if [[ -f "$SPGM_DIR/stageB_history.json" ]]; then
  echo "[skip] StageB SPGM exists"
else
  mkdir -p "$SPGM_DIR"
  "$PY" "$PROJECT_ROOT/scripts/run_pseudo_refinement_v2.py" \
    --ply_path "$STAGEA5_DIR/refined_gaussians.ply" \
    --output_dir "$SPGM_DIR" \
    --stage_mode stageB \
    --stageA_iters 0 \
    --stageB_iters 120 \
    --num_pseudo_views 4 \
    --num_real_views 2 \
    --lambda_real 1.0 \
    --lambda_pseudo 1.0 \
    --stageA_rgb_mask_mode brpo_v2_raw \
    --stageA_depth_mask_mode train_mask \
    --stageA_target_depth_mode target_depth_for_refine \
    --stageA_depth_loss_mode source_aware \
    --stageA_lambda_depth_seed 1.0 \
    --stageA_lambda_depth_dense 0.35 \
    --stageA_lambda_depth_fallback 0.0 \
    --stageA_lambda_abs_t 3.0 \
    --stageA_lambda_abs_r 0.1 \
    --stageA_abs_pose_robust charbonnier \
    --stageA_abs_pose_scale_source render_depth_trainmask_median \
    --stageA5_trainable_params xyz \
    --stageA5_lr_xyz 1e-4 \
    --init_pseudo_camera_states_json "$STAGEA5_DIR/pseudo_camera_states_final.json" \
    --init_pseudo_reference_mode keep \
    --train_manifest "$TRAIN_MANIFEST" \
    --train_rgb_dir "$TRAIN_RGB_DIR" \
    --seed 0 \
    --pseudo_cache "$PSEUDO_CACHE" \
    --target_side fused \
    --confidence_mask_source brpo \
    --signal_pipeline brpo_v2 \
    --signal_v2_root "$SIGNAL_V2_ROOT" \
    --pseudo_local_gating spgm_keep \
    --pseudo_local_gating_params xyz \
    --pseudo_local_gating_min_verified_ratio 0.01 \
    --pseudo_local_gating_min_rgb_mask_ratio 0.0192 \
    --pseudo_local_gating_max_fallback_ratio 0.995 \
    --pseudo_local_gating_spgm_policy_mode dense_keep \
    --pseudo_local_gating_spgm_ranking_mode v1 \
    --pseudo_local_gating_spgm_support_eta 0.0 \
    --pseudo_local_gating_spgm_weight_floor 0.25 \
    --pseudo_local_gating_spgm_cluster_keep_near 1.0 \
    --pseudo_local_gating_spgm_cluster_keep_mid 1.0 \
    --pseudo_local_gating_spgm_cluster_keep_far 1.0 \
    --stageB_post_switch_iter 40 \
    --stageB_post_lr_scale_xyz 0.3
fi

echo "STAGEB_SPGM_DONE $SPGM_DIR"

run_replay "$BASE_PLY" after_opt_baseline "$EXP_ROOT/replay_after_opt_baseline"
run_replay "$STAGEA_DIR/refined_gaussians.ply" stageA_v2rgbonly_80 "$STAGEA_DIR/replay_eval"
run_replay "$STAGEA5_DIR/refined_gaussians.ply" stageA5_v2rgbonly_xyz_gated_rgb0192_80 "$STAGEA5_DIR/replay_eval"
run_replay "$BASELINE_DIR/refined_gaussians.ply" canonical_baseline_post40_lr03_120 "$BASELINE_DIR/replay_eval"
run_replay "$SPGM_DIR/refined_gaussians.ply" spgm_repair_a_keep111_eta0_wf025 "$SPGM_DIR/replay_eval"

echo "REPLAY_DONE $EXP_ROOT"

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path('/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_dl3dv_phaseg_baseline_spgm_e1')
base = json.loads((root/'replay_after_opt_baseline'/'replay_eval.json').read_text())
stageA = json.loads((root/'stageA_v2rgbonly_80'/'replay_eval'/'replay_eval.json').read_text())
stageA5 = json.loads((root/'stageA5_v2rgbonly_xyz_gated_rgb0192_80'/'replay_eval'/'replay_eval.json').read_text())
baseline = json.loads((root/'canonical_baseline_post40_lr03_120'/'replay_eval'/'replay_eval.json').read_text())
spgm = json.loads((root/'spgm_repair_a_keep111_eta0_wf025'/'replay_eval'/'replay_eval.json').read_text())
stageA_hist = json.loads((root/'stageA_v2rgbonly_80'/'stageA_history.json').read_text())
stageA5_hist = json.loads((root/'stageA5_v2rgbonly_xyz_gated_rgb0192_80'/'stageA_history.json').read_text())
baseline_hist = json.loads((root/'canonical_baseline_post40_lr03_120'/'stageB_history.json').read_text())
spgm_hist = json.loads((root/'spgm_repair_a_keep111_eta0_wf025'/'stageB_history.json').read_text())

def replay_metrics(d):
    return {'avg_psnr': d['avg_psnr'], 'avg_ssim': d['avg_ssim'], 'avg_lpips': d['avg_lpips']}

def delta(a,b):
    return {
        'psnr': a['avg_psnr'] - b['avg_psnr'],
        'ssim': a['avg_ssim'] - b['avg_ssim'],
        'lpips': a['avg_lpips'] - b['avg_lpips'],
    }

def rejection_stats(hist):
    ids_lists = hist.get('rejected_pseudo_sample_ids', [])
    return {
        'iters_with_rejection': sum(1 for ids in ids_lists if ids),
        'total_rejected_sample_evals': sum(len(ids) for ids in ids_lists),
        'unique_rejected_ids': sorted({int(x) for ids in ids_lists for x in ids}),
        'grad_keep_ratio_xyz_mean': (sum(hist.get('grad_keep_ratio_xyz', [])) / len(hist.get('grad_keep_ratio_xyz', []))) if hist.get('grad_keep_ratio_xyz') else None,
        'grad_keep_ratio_xyz_last': hist.get('grad_keep_ratio_xyz', [None])[-1],
    }

summary = {
    'protocol': 'DL3DV Phase G baseline+SPGM bring-up on canonical signal_v2 root',
    'paths': {
        'run_root': '/data2/bzhang512/CV_Project/output/part2_s3po/dl3dv-2/s3po_dl3dv-2_full_internal_cache/DL3DV-2_part2_s3po/2026-04-17-21-21-00',
        'prepare_root': '/data2/bzhang512/CV_Project/output/part2_s3po/dl3dv-2/s3po_dl3dv-2_full_internal_cache/DL3DV-2_part2_s3po/2026-04-17-21-21-00/internal_prepare/dl3dv2__internal_afteropt__brpo_proto_v4_stage3',
        'signal_v2_root': '/data2/bzhang512/CV_Project/output/part2_s3po/dl3dv-2/s3po_dl3dv-2_full_internal_cache/DL3DV-2_part2_s3po/2026-04-17-21-21-00/internal_prepare/dl3dv2__internal_afteropt__brpo_proto_v4_stage3/signal_v2',
        'experiment_root': str(root),
    },
    'after_opt_baseline': replay_metrics(base),
    'stageA_v2rgbonly_80': {
        **replay_metrics(stageA),
        'delta_vs_after_opt': delta(replay_metrics(stageA), replay_metrics(base)),
        'effective_source_summary': stageA_hist['effective_source_summary'],
        'final_true_pose_delta_aggregate': stageA_hist['final_true_pose_delta_aggregate'],
    },
    'stageA5_v2rgbonly_xyz_gated_rgb0192_80': {
        **replay_metrics(stageA5),
        'delta_vs_after_opt': delta(replay_metrics(stageA5), replay_metrics(base)),
        'delta_vs_stageA': delta(replay_metrics(stageA5), replay_metrics(stageA)),
        'effective_source_summary': stageA5_hist['effective_source_summary'],
        'gating': rejection_stats(stageA5_hist['history']),
        'final_true_pose_delta_aggregate': stageA5_hist['final_true_pose_delta_aggregate'],
    },
    'canonical_baseline_post40_lr03_120': {
        **replay_metrics(baseline),
        'delta_vs_after_opt': delta(replay_metrics(baseline), replay_metrics(base)),
        'delta_vs_stageA5': delta(replay_metrics(baseline), replay_metrics(stageA5)),
        'post_switch': {
            'applied': baseline_hist.get('post_switch_applied'),
            'applied_at_iter': baseline_hist.get('post_switch_applied_at_iter'),
            'gaussian_lr_xyz_last': baseline_hist.get('gaussian_lr_xyz_effective', [None])[-1],
        },
        'gating': rejection_stats(baseline_hist),
        'final_true_pose_delta_aggregate': baseline_hist['final_true_pose_delta_aggregate'],
    },
    'spgm_repair_a_keep111_eta0_wf025': {
        **replay_metrics(spgm),
        'delta_vs_after_opt': delta(replay_metrics(spgm), replay_metrics(base)),
        'delta_vs_stageA5': delta(replay_metrics(spgm), replay_metrics(stageA5)),
        'delta_vs_stageB_baseline': delta(replay_metrics(spgm), replay_metrics(baseline)),
        'post_switch': {
            'applied': spgm_hist.get('post_switch_applied'),
            'applied_at_iter': spgm_hist.get('post_switch_applied_at_iter'),
            'gaussian_lr_xyz_last': spgm_hist.get('gaussian_lr_xyz_effective', [None])[-1],
        },
        'gating': rejection_stats(spgm_hist),
        'spgm_policy_mode': spgm_hist['pseudo_local_gating'].get('spgm_policy_mode'),
        'spgm_ranking_mode': spgm_hist['pseudo_local_gating'].get('spgm_ranking_mode'),
        'final_true_pose_delta_aggregate': spgm_hist['final_true_pose_delta_aggregate'],
    },
}
(root/'summary.json').write_text(json.dumps(summary, indent=2))
lines = []
lines.append('# DL3DV Phase G baseline + SPGM summary')
lines.append('')
lines.append('| arm | PSNR | SSIM | LPIPS | ΔPSNR vs after_opt | ΔPSNR vs prev anchor |')
lines.append('| --- | ---: | ---: | ---: | ---: | ---: |')
lines.append(f"| stageA_v2rgbonly_80 | {summary['stageA_v2rgbonly_80']['avg_psnr']} | {summary['stageA_v2rgbonly_80']['avg_ssim']} | {summary['stageA_v2rgbonly_80']['avg_lpips']} | {summary['stageA_v2rgbonly_80']['delta_vs_after_opt']['psnr']} | n/a |")
lines.append(f"| stageA5_v2rgbonly_xyz_gated_rgb0192_80 | {summary['stageA5_v2rgbonly_xyz_gated_rgb0192_80']['avg_psnr']} | {summary['stageA5_v2rgbonly_xyz_gated_rgb0192_80']['avg_ssim']} | {summary['stageA5_v2rgbonly_xyz_gated_rgb0192_80']['avg_lpips']} | {summary['stageA5_v2rgbonly_xyz_gated_rgb0192_80']['delta_vs_after_opt']['psnr']} | {summary['stageA5_v2rgbonly_xyz_gated_rgb0192_80']['delta_vs_stageA']['psnr']} |")
lines.append(f"| canonical_baseline_post40_lr03_120 | {summary['canonical_baseline_post40_lr03_120']['avg_psnr']} | {summary['canonical_baseline_post40_lr03_120']['avg_ssim']} | {summary['canonical_baseline_post40_lr03_120']['avg_lpips']} | {summary['canonical_baseline_post40_lr03_120']['delta_vs_after_opt']['psnr']} | {summary['canonical_baseline_post40_lr03_120']['delta_vs_stageA5']['psnr']} |")
lines.append(f"| spgm_repair_a_keep111_eta0_wf025 | {summary['spgm_repair_a_keep111_eta0_wf025']['avg_psnr']} | {summary['spgm_repair_a_keep111_eta0_wf025']['avg_ssim']} | {summary['spgm_repair_a_keep111_eta0_wf025']['avg_lpips']} | {summary['spgm_repair_a_keep111_eta0_wf025']['delta_vs_after_opt']['psnr']} | {summary['spgm_repair_a_keep111_eta0_wf025']['delta_vs_stageB_baseline']['psnr']} |")
lines.append('')
lines.append('## gating')
lines.append(f"- StageA.5 rejection_iters = {summary['stageA5_v2rgbonly_xyz_gated_rgb0192_80']['gating']['iters_with_rejection']}")
lines.append(f"- StageB baseline rejection_iters = {summary['canonical_baseline_post40_lr03_120']['gating']['iters_with_rejection']}")
lines.append(f"- StageB SPGM rejection_iters = {summary['spgm_repair_a_keep111_eta0_wf025']['gating']['iters_with_rejection']}")
(root/'summary.md').write_text('\n'.join(lines) + '\n')
print(json.dumps(summary, indent=2))
PY

echo "SUMMARY_DONE $EXP_ROOT/summary.json"
echo "ALL_DONE $EXP_ROOT"

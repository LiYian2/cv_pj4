#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/bzhang512/CV_Project/part3_BRPO
PY=/home/bzhang512/miniconda3/envs/s3po-gs/bin/python
export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO:${PYTHONPATH:-}

INTERNAL_CACHE_ROOT=/data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache
PSEUDO_CACHE=/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/pseudo_cache_baseline
SIGNAL_V2_ROOT=/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/signal_v2
TRAIN_MANIFEST=/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/split_manifest.json
TRAIN_RGB_DIR=/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/rgb
STAGEA5_ROOT=/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_gated_rgb0192_80
P2I_ROOT=/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2i_stageB_window_localization_e1
P2J_ROOT=/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2j_stageB_bounded_schedule_compare_e1
EXP_ROOT=/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_p2k_canonical_stageB_compare_e1
mkdir -p "$EXP_ROOT"

cat > "$EXP_ROOT/RUN_PLAN.txt" <<'EOF'
P2-K canonical StageB compare (2026-04-17)
Goal:
- Re-run the SPGM StageB 120iter compare under the canonical bounded StageB protocol.
- Keep the canonical P2-J baseline unchanged: RGB-only v2 + gated_rgb0192 + post40_lr03_120 + sequential StageA.5 handoff.
- Change only the Gaussian-side pseudo grad manager: baseline hard_visible_union_signal vs SPGM keep.

Design:
- Shared StageA.5 handoff: 20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_gated_rgb0192_80
- Shared StageB schedule: 120iter with post-switch at iter 40, xyz lr *= 0.3
- Shared sequential handoff: init_pseudo_camera_states_json = pseudo_camera_states_final.json, reference_mode=keep
- Shared view-gate upstream: signal_pipeline=brpo_v2, same pseudo cache, same signal_v2, same real branch, same seed
- Only study variable: pseudo_local_gating mode
  1. canonical_baseline_post40_lr03_120 -> hard_visible_union_signal
  2. canonical_spgm_keep_post40_lr03_120 -> spgm_keep (current v1 deterministic defaults)
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

run_arm() {
  local label="$1"
  shift
  local out_dir="$EXP_ROOT/$label"
  if [[ -f "$out_dir/stageB_history.json" ]]; then
    echo "[skip] training exists for $label"
  else
    mkdir -p "$out_dir"
    "$PY" "$PROJECT_ROOT/scripts/run_pseudo_refinement_v2.py" \
      --ply_path "$STAGEA5_ROOT/refined_gaussians.ply" \
      --pseudo_cache "$PSEUDO_CACHE" \
      --output_dir "$out_dir" \
      --target_side fused \
      --confidence_mask_source brpo \
      --signal_pipeline brpo_v2 \
      --signal_v2_root "$SIGNAL_V2_ROOT" \
      --stage_mode stageB \
      --stageA_iters 0 \
      --stageB_iters 120 \
      --stageB_post_switch_iter 40 \
      --stageB_post_lr_scale_xyz 0.3 \
      --num_pseudo_views 4 \
      --num_real_views 2 \
      --lambda_real 1.0 \
      --lambda_pseudo 1.0 \
      --stageA_rgb_mask_mode brpo_v2_raw \
      --stageA_depth_mask_mode train_mask \
      --stageA_target_depth_mode target_depth_for_refine_v2 \
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
      --init_pseudo_camera_states_json "$STAGEA5_ROOT/pseudo_camera_states_final.json" \
      --init_pseudo_reference_mode keep \
      --train_manifest "$TRAIN_MANIFEST" \
      --train_rgb_dir "$TRAIN_RGB_DIR" \
      --seed 0 \
      --pseudo_local_gating_params xyz \
      --pseudo_local_gating_min_verified_ratio 0.01 \
      --pseudo_local_gating_min_rgb_mask_ratio 0.0192 \
      --pseudo_local_gating_max_fallback_ratio 0.995 \
      "$@"
  fi
  run_replay "$out_dir/refined_gaussians.ply" "$label" "$out_dir/replay_eval"
}

run_arm canonical_baseline_post40_lr03_120 \
  --pseudo_local_gating hard_visible_union_signal

run_arm canonical_spgm_keep_post40_lr03_120 \
  --pseudo_local_gating spgm_keep \
  --pseudo_local_gating_spgm_num_clusters 3 \
  --pseudo_local_gating_spgm_alpha_depth 0.5 \
  --pseudo_local_gating_spgm_beta_entropy 0.5 \
  --pseudo_local_gating_spgm_gamma_entropy 0.5 \
  --pseudo_local_gating_spgm_support_eta 0.5 \
  --pseudo_local_gating_spgm_weight_floor 0.05 \
  --pseudo_local_gating_spgm_entropy_bins 32 \
  --pseudo_local_gating_spgm_density_mode opacity_support \
  --pseudo_local_gating_spgm_cluster_keep_near 1.0 \
  --pseudo_local_gating_spgm_cluster_keep_mid 0.8 \
  --pseudo_local_gating_spgm_cluster_keep_far 0.6

"$PY" - <<'PY'
import json
import subprocess
from pathlib import Path


def mean_num(xs):
    ys = [x for x in xs if isinstance(x, (int, float))]
    return None if not ys else sum(ys) / len(ys)


root = Path('/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_p2k_canonical_stageB_compare_e1')
p2i_root = Path('/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2i_stageB_window_localization_e1')
p2j_root = Path('/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2j_stageB_bounded_schedule_compare_e1')
stagea5_root = Path('/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_gated_rgb0192_80')

ref = json.loads((p2i_root/'summary.json').read_text())
after_opt = ref['baseline_after_opt']
stagea5 = ref['stageA5_start']
p2j_summary = json.loads((p2j_root/'summary.json').read_text())
canonical_ref = p2j_summary['arms']['stageB_post40_lr03_120']

summary = {
    'protocol': 'P2-K canonical StageB compare on bounded RGB-only v2 baseline',
    'git_commit': subprocess.check_output(['git', '-C', '/home/bzhang512/CV_Project/part3_BRPO', 'rev-parse', '--short', 'HEAD'], text=True).strip(),
    'anchors': {
        'after_opt': after_opt,
        'stageA5_start': stagea5,
        'p2j_post40_lr03_120': canonical_ref,
    },
    'arms': {}
}

for name in ['canonical_baseline_post40_lr03_120', 'canonical_spgm_keep_post40_lr03_120']:
    replay = json.loads((root/name/'replay_eval'/'replay_eval.json').read_text())
    hist = json.loads((root/name/'stageB_history.json').read_text())
    arm = {
        'avg_psnr': replay['avg_psnr'],
        'avg_ssim': replay['avg_ssim'],
        'avg_lpips': replay['avg_lpips'],
        'delta_vs_after_opt': {
            'psnr': replay['avg_psnr'] - after_opt['avg_psnr'],
            'ssim': replay['avg_ssim'] - after_opt['avg_ssim'],
            'lpips': replay['avg_lpips'] - after_opt['avg_lpips'],
        },
        'delta_vs_stageA5_start': {
            'psnr': replay['avg_psnr'] - stagea5['avg_psnr'],
            'ssim': replay['avg_ssim'] - stagea5['avg_ssim'],
            'lpips': replay['avg_lpips'] - stagea5['avg_lpips'],
        },
        'delta_vs_p2j_post40': {
            'psnr': replay['avg_psnr'] - canonical_ref['avg_psnr'],
            'ssim': replay['avg_ssim'] - canonical_ref['avg_ssim'],
            'lpips': replay['avg_lpips'] - canonical_ref['avg_lpips'],
        },
        'loss_real_last': hist['loss_real'][-1],
        'loss_pseudo_last': hist['loss_pseudo'][-1],
        'loss_total_last': hist['loss_total'][-1],
        'iters_with_rejection': sum(1 for ids in hist['rejected_pseudo_sample_ids'] if ids),
        'total_rejected_sample_evals': sum(len(ids) for ids in hist['rejected_pseudo_sample_ids']),
        'unique_rejected_ids': sorted({sid for ids in hist['rejected_pseudo_sample_ids'] for sid in ids}),
        'grad_keep_ratio_xyz_mean': mean_num(hist.get('grad_keep_ratio_xyz', [])),
        'grad_weight_mean_xyz_mean': mean_num(hist.get('grad_weight_mean_xyz', [])),
        'grad_norm_xyz_mean': mean_num(hist.get('grad_norm_xyz', [])),
        'post_switch_applied': hist.get('post_switch_applied'),
        'post_switch_applied_at_iter': hist.get('post_switch_applied_at_iter'),
        'lambda_real_last': hist['lambda_real_effective'][-1],
        'gaussian_lr_xyz_last': hist['gaussian_lr_xyz_effective'][-1],
        'spgm_active_ratio_mean': mean_num(hist.get('spgm_active_ratio', [])),
        'spgm_support_ratio_mean': mean_num(hist.get('spgm_support_ratio', [])),
        'spgm_density_mode_effective_last': hist.get('spgm_density_mode_effective', [None])[-1] if hist.get('spgm_density_mode_effective') else None,
    }
    summary['arms'][name] = arm

base = summary['arms']['canonical_baseline_post40_lr03_120']
exp = summary['arms']['canonical_spgm_keep_post40_lr03_120']
summary['delta_spgm_minus_baseline'] = {
    'psnr': exp['avg_psnr'] - base['avg_psnr'],
    'ssim': exp['avg_ssim'] - base['avg_ssim'],
    'lpips': exp['avg_lpips'] - base['avg_lpips'],
    'loss_real_last': exp['loss_real_last'] - base['loss_real_last'],
    'loss_pseudo_last': exp['loss_pseudo_last'] - base['loss_pseudo_last'],
    'grad_keep_ratio_xyz_mean': None if exp['grad_keep_ratio_xyz_mean'] is None or base['grad_keep_ratio_xyz_mean'] is None else exp['grad_keep_ratio_xyz_mean'] - base['grad_keep_ratio_xyz_mean'],
    'grad_weight_mean_xyz_mean': None if exp['grad_weight_mean_xyz_mean'] is None or base['grad_weight_mean_xyz_mean'] is None else exp['grad_weight_mean_xyz_mean'] - base['grad_weight_mean_xyz_mean'],
    'spgm_active_ratio_mean': exp['spgm_active_ratio_mean'],
}

(root/'summary.json').write_text(json.dumps(summary, indent=2))

lines = []
lines.append('# P2-K canonical StageB compare summary')
lines.append('')
lines.append('| arm | PSNR | SSIM | LPIPS | ΔPSNR vs baseline | ΔPSNR vs P2-J post40 | loss_real_last | loss_pseudo_last | grad_keep_mean | grad_weight_mean |')
lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
for name in ['canonical_baseline_post40_lr03_120', 'canonical_spgm_keep_post40_lr03_120']:
    arm = summary['arms'][name]
    delta_vs_base = arm['avg_psnr'] - base['avg_psnr']
    lines.append(f"| {name} | {arm['avg_psnr']} | {arm['avg_ssim']} | {arm['avg_lpips']} | {delta_vs_base} | {arm['delta_vs_p2j_post40']['psnr']} | {arm['loss_real_last']} | {arm['loss_pseudo_last']} | {arm['grad_keep_ratio_xyz_mean']} | {arm['grad_weight_mean_xyz_mean']} |")
lines.append('')
lines.append('SPGM minus baseline: {}'.format(json.dumps(summary['delta_spgm_minus_baseline'], ensure_ascii=False)))
(root/'summary.md').write_text('\n'.join(lines) + '\n')
print(json.dumps(summary, indent=2))
PY

echo "DONE P2-K canonical StageB compare"

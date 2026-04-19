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
P2K_REF_ROOT=/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_p2k_canonical_stageB_compare_e1
P2S_REF_ROOT=/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_p2s_supportblend_farkeep_followup_compare_e1
EXP_ROOT=/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260418_p2t_selector_confirmation_precision_compare_e1
mkdir -p "$EXP_ROOT"

cat > "$EXP_ROOT/RUN_PLAN.txt" <<'EOF'
P2-T selector-first confirmation / precision sweep (2026-04-18)
Goal:
- Keep the canonical StageB protocol fixed.
- Keep repair A as the in-run control.
- Confirm whether the support_blend far-only selector really has anchor-upgrade potential near far≈0.90.
- Do a narrow precision sweep only: repeat 0.90, then 0.92 / 0.95, and include 0.88 as the slightly stronger-side boundary.

Fixed protocol:
- Same StageA.5 handoff as P2-M / P2-R / P2-S
- Same sequential pseudo-state handoff
- Same StageB post40_lr03_120 schedule
- Same upstream gated_rgb0192 view-gate settings
- Same replay evaluator
- Same repair-A-style weighting: support_eta=0.0 / weight_floor=0.25 / cluster_keep=(1,1,1)
- Same selector ranking: ranking_mode=support_blend, lambda_support_rank=0.5

Arms:
1. control_repair_a_dense_keep
2. selector_far_only_supportblend_l05_keep100_100_088
3. selector_far_only_supportblend_l05_keep100_100_090_repeat
4. selector_far_only_supportblend_l05_keep100_100_092
5. selector_far_only_supportblend_l05_keep100_100_095
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
    echo "[run] $label"
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
      --pseudo_local_gating spgm_keep \
      --pseudo_local_gating_params xyz \
      --pseudo_local_gating_min_verified_ratio 0.01 \
      --pseudo_local_gating_min_rgb_mask_ratio 0.0192 \
      --pseudo_local_gating_max_fallback_ratio 0.995 \
      --pseudo_local_gating_spgm_num_clusters 3 \
      --pseudo_local_gating_spgm_alpha_depth 0.5 \
      --pseudo_local_gating_spgm_beta_entropy 0.5 \
      --pseudo_local_gating_spgm_gamma_entropy 0.5 \
      --pseudo_local_gating_spgm_support_eta 0.0 \
      --pseudo_local_gating_spgm_weight_floor 0.25 \
      --pseudo_local_gating_spgm_entropy_bins 32 \
      --pseudo_local_gating_spgm_density_mode opacity_support \
      --pseudo_local_gating_spgm_cluster_keep_near 1.0 \
      --pseudo_local_gating_spgm_cluster_keep_mid 1.0 \
      --pseudo_local_gating_spgm_cluster_keep_far 1.0 \
      "$@"
  fi
  run_replay "$out_dir/refined_gaussians.ply" "$label" "$out_dir/replay_eval"
}

run_arm control_repair_a_dense_keep \
  --pseudo_local_gating_spgm_policy_mode dense_keep \
  --pseudo_local_gating_spgm_ranking_mode v1 \
  --pseudo_local_gating_spgm_lambda_support_rank 0.0 \
  --pseudo_local_gating_spgm_selector_keep_ratio_near 1.0 \
  --pseudo_local_gating_spgm_selector_keep_ratio_mid 1.0 \
  --pseudo_local_gating_spgm_selector_keep_ratio_far 1.0

run_arm selector_far_only_supportblend_l05_keep100_100_088 \
  --pseudo_local_gating_spgm_policy_mode selector_quantile \
  --pseudo_local_gating_spgm_ranking_mode support_blend \
  --pseudo_local_gating_spgm_lambda_support_rank 0.5 \
  --pseudo_local_gating_spgm_selector_keep_ratio_near 1.0 \
  --pseudo_local_gating_spgm_selector_keep_ratio_mid 1.0 \
  --pseudo_local_gating_spgm_selector_keep_ratio_far 0.88

run_arm selector_far_only_supportblend_l05_keep100_100_090_repeat \
  --pseudo_local_gating_spgm_policy_mode selector_quantile \
  --pseudo_local_gating_spgm_ranking_mode support_blend \
  --pseudo_local_gating_spgm_lambda_support_rank 0.5 \
  --pseudo_local_gating_spgm_selector_keep_ratio_near 1.0 \
  --pseudo_local_gating_spgm_selector_keep_ratio_mid 1.0 \
  --pseudo_local_gating_spgm_selector_keep_ratio_far 0.90

run_arm selector_far_only_supportblend_l05_keep100_100_092 \
  --pseudo_local_gating_spgm_policy_mode selector_quantile \
  --pseudo_local_gating_spgm_ranking_mode support_blend \
  --pseudo_local_gating_spgm_lambda_support_rank 0.5 \
  --pseudo_local_gating_spgm_selector_keep_ratio_near 1.0 \
  --pseudo_local_gating_spgm_selector_keep_ratio_mid 1.0 \
  --pseudo_local_gating_spgm_selector_keep_ratio_far 0.92

run_arm selector_far_only_supportblend_l05_keep100_100_095 \
  --pseudo_local_gating_spgm_policy_mode selector_quantile \
  --pseudo_local_gating_spgm_ranking_mode support_blend \
  --pseudo_local_gating_spgm_lambda_support_rank 0.5 \
  --pseudo_local_gating_spgm_selector_keep_ratio_near 1.0 \
  --pseudo_local_gating_spgm_selector_keep_ratio_mid 1.0 \
  --pseudo_local_gating_spgm_selector_keep_ratio_far 0.95

"$PY" - <<'PY'
import json
import subprocess
from pathlib import Path


def mean_num(xs):
    ys = [x for x in xs if isinstance(x, (int, float))]
    return None if not ys else sum(ys) / len(ys)

p2k_ref_root = Path('/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_p2k_canonical_stageB_compare_e1')
p2s_ref_root = Path('/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_p2s_supportblend_farkeep_followup_compare_e1')
root = Path('/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260418_p2t_selector_confirmation_precision_compare_e1')
p2k = json.loads((p2k_ref_root / 'summary.json').read_text())
p2s = json.loads((p2s_ref_root / 'summary.json').read_text())
base = p2k['arms']['canonical_baseline_post40_lr03_120']
old_control = p2s['arms']['control_repair_a_dense_keep']
old_090 = p2s['arms']['selector_far_only_supportblend_l05_keep100_100_090']
summary = {
    'protocol': 'P2-T selector-first confirmation / precision sweep on canonical StageB protocol',
    'git_commit': subprocess.check_output(['git', '-C', '/home/bzhang512/CV_Project/part3_BRPO', 'rev-parse', '--short', 'HEAD'], text=True).strip(),
    'git_status_short': subprocess.check_output(['git', '-C', '/home/bzhang512/CV_Project/part3_BRPO', 'status', '--short'], text=True).strip().splitlines(),
    'reference': {
        'canonical_baseline_post40_lr03_120': base,
        'p2s_control_repair_a_dense_keep': old_control,
        'p2s_selector_far_only_supportblend_l05_keep100_100_090': old_090,
    },
    'study_order': [
        'control_repair_a_dense_keep',
        'selector_far_only_supportblend_l05_keep100_100_088',
        'selector_far_only_supportblend_l05_keep100_100_090_repeat',
        'selector_far_only_supportblend_l05_keep100_100_092',
        'selector_far_only_supportblend_l05_keep100_100_095',
    ],
    'arms': {},
}
for name in summary['study_order']:
    replay = json.loads((root / name / 'replay_eval' / 'replay_eval.json').read_text())
    hist = json.loads((root / name / 'stageB_history.json').read_text())
    arm = {
        'avg_psnr': replay['avg_psnr'],
        'avg_ssim': replay['avg_ssim'],
        'avg_lpips': replay['avg_lpips'],
        'delta_vs_baseline': {
            'psnr': replay['avg_psnr'] - base['avg_psnr'],
            'ssim': replay['avg_ssim'] - base['avg_ssim'],
            'lpips': replay['avg_lpips'] - base['avg_lpips'],
        },
        'delta_vs_old_control': {
            'psnr': replay['avg_psnr'] - old_control['avg_psnr'],
            'ssim': replay['avg_ssim'] - old_control['avg_ssim'],
            'lpips': replay['avg_lpips'] - old_control['avg_lpips'],
        },
        'delta_vs_old_090': {
            'psnr': replay['avg_psnr'] - old_090['avg_psnr'],
            'ssim': replay['avg_ssim'] - old_090['avg_ssim'],
            'lpips': replay['avg_lpips'] - old_090['avg_lpips'],
        },
        'loss_real_last': hist['loss_real'][-1],
        'loss_pseudo_last': hist['loss_pseudo'][-1],
        'loss_total_last': hist['loss_total'][-1],
        'iters_with_rejection': sum(1 for ids in hist['rejected_pseudo_sample_ids'] if ids),
        'total_rejected_sample_evals': sum(len(ids) for ids in hist['rejected_pseudo_sample_ids']),
        'unique_rejected_ids': sorted({sid for ids in hist['rejected_pseudo_sample_ids'] for sid in ids}),
        'grad_keep_ratio_xyz_mean': mean_num(hist.get('grad_keep_ratio_xyz', [])),
        'grad_weight_mean_xyz_mean': mean_num(hist.get('grad_weight_mean_xyz', [])),
        'spgm_active_ratio_mean': mean_num(hist.get('spgm_active_ratio', [])),
        'spgm_selected_ratio_mean': mean_num(hist.get('spgm_selected_ratio', [])),
        'spgm_ranking_score_mean': mean_num(hist.get('spgm_ranking_score_mean', [])),
        'spgm_support_norm_mean': mean_num(hist.get('spgm_support_norm_mean', [])),
        'spgm_weight_mean_mean': mean_num(hist.get('spgm_weight_mean', [])),
        'spgm_policy_mode_effective_last': hist.get('spgm_policy_mode_effective', [None])[-1] if hist.get('spgm_policy_mode_effective') else None,
        'spgm_ranking_mode_effective_last': hist.get('spgm_ranking_mode_effective', [None])[-1] if hist.get('spgm_ranking_mode_effective') else None,
        'arm_root': str((root / name).resolve()),
    }
    summary['arms'][name] = arm

control = summary['arms']['control_repair_a_dense_keep']
for name, arm in summary['arms'].items():
    arm['delta_vs_control'] = {
        'psnr': arm['avg_psnr'] - control['avg_psnr'],
        'ssim': arm['avg_ssim'] - control['avg_ssim'],
        'lpips': arm['avg_lpips'] - control['avg_lpips'],
        'spgm_selected_ratio_mean': (arm['spgm_selected_ratio_mean'] - control['spgm_selected_ratio_mean']) if arm['spgm_selected_ratio_mean'] is not None and control['spgm_selected_ratio_mean'] is not None else None,
    }

selector_names = [n for n in summary['study_order'] if n != 'control_repair_a_dense_keep']
best_psnr = max(selector_names, key=lambda n: summary['arms'][n]['avg_psnr'])
best_ssim = max(selector_names, key=lambda n: summary['arms'][n]['avg_ssim'])
best_lpips = min(selector_names, key=lambda n: summary['arms'][n]['avg_lpips'])
summary['best_selector_arm'] = {
    'by_psnr': best_psnr,
    'by_ssim': best_ssim,
    'by_lpips': best_lpips,
}
(root / 'summary.json').write_text(json.dumps(summary, indent=2))
lines = []
lines.append('# P2-T selector-first confirmation / precision sweep summary')
lines.append('')
lines.append('| arm | PSNR | SSIM | LPIPS | ΔPSNR vs control | ΔPSNR vs old 0.90 | selected_ratio_mean | grad_weight_mean |')
lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
for name in summary['study_order']:
    arm = summary['arms'][name]
    lines.append(
        f"| {name} | {arm['avg_psnr']} | {arm['avg_ssim']} | {arm['avg_lpips']} | {arm['delta_vs_control']['psnr']} | {arm['delta_vs_old_090']['psnr']} | {arm['spgm_selected_ratio_mean']} | {arm['grad_weight_mean_xyz_mean']} |"
    )
lines.append('')
lines.append(f"Best selector arm by PSNR: {summary['best_selector_arm']['by_psnr']}")
lines.append(f"Best selector arm by SSIM: {summary['best_selector_arm']['by_ssim']}")
lines.append(f"Best selector arm by LPIPS: {summary['best_selector_arm']['by_lpips']}")
(root / 'summary.md').write_text('\n'.join(lines) + '\n')
print(json.dumps(summary, indent=2))
PY

echo "DONE P2-T selector-first confirmation / precision sweep"

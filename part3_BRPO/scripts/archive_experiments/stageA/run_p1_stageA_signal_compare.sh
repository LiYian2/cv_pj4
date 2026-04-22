#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/bzhang512/CV_Project/part3_BRPO
PY=/home/bzhang512/miniconda3/envs/s3po-gs/bin/python
export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO

BASE_PLY=/data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache/after_opt/point_cloud/point_cloud.ply
PREPARE_ROOT=/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare
PSEUDO_CACHE=$PREPARE_ROOT/pseudo_cache_baseline
SIGNAL_V2_ROOT=$PREPARE_ROOT/signal_v2
OUT_ROOT=/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_signal_compare_stageAonly_e1
SUMMARY_DIR=$OUT_ROOT/summary

mkdir -p "$OUT_ROOT" "$SUMMARY_DIR"

echo "P1A_START root=$OUT_ROOT"

labels=(
  stageA_legacy_80
  stageA_v2_rgbonly_80
  stageA_v2_full_80
)
signal_pipeline=(legacy brpo_v2 brpo_v2)
rgb_mode=(train_mask brpo_v2_raw brpo_v2_raw)
depth_mask_mode=(train_mask train_mask brpo_v2_depth)
depth_target_mode=(target_depth_for_refine_v2 target_depth_for_refine_v2 brpo_v2)

run_dirs=()
for idx in "${!labels[@]}"; do
  label=${labels[$idx]}
  run_dir="$OUT_ROOT/$label"
  run_dirs+=("$run_dir")
  if [[ ! -f "$run_dir/stageA_history.json" ]]; then
    echo "[P1A] run start label=$label"
    cmd=(
      "$PY" "$PROJECT_ROOT/scripts/run_pseudo_refinement_v2.py"
      --ply_path "$BASE_PLY"
      --pseudo_cache "$PSEUDO_CACHE"
      --output_dir "$run_dir"
      --target_side fused
      --confidence_mask_source brpo
      --signal_pipeline "${signal_pipeline[$idx]}"
      --stage_mode stageA
      --stageA_iters 80
      --stageA_rgb_mask_mode "${rgb_mode[$idx]}"
      --stageA_depth_mask_mode "${depth_mask_mode[$idx]}"
      --stageA_target_depth_mode "${depth_target_mode[$idx]}"
      --stageA_depth_loss_mode source_aware
      --stageA_lambda_depth_seed 1.0
      --stageA_lambda_depth_dense 0.35
      --stageA_lambda_depth_fallback 0.0
      --stageA_abs_pose_robust charbonnier
      --stageA_abs_pose_scale_source render_depth_trainmask_median
      --stageA_lambda_abs_t 3.0
      --stageA_lambda_abs_r 0.1
      --num_pseudo_views 4
      --seed 0
    )
    if [[ "${signal_pipeline[$idx]}" == "brpo_v2" ]]; then
      cmd+=(--signal_v2_root "$SIGNAL_V2_ROOT")
    fi
    "${cmd[@]}"
  else
    echo "[P1A] run already exists label=$label"
  fi

  "$PY" "$PROJECT_ROOT/scripts/diagnostics/summarize_stageA_compare.py" \
    --run-dirs "${run_dirs[@]}" \
    --output-root "$SUMMARY_DIR"

  echo "P1A_RUN_DONE label=$label run_dir=$run_dir"
done

echo "P1A_SUMMARY_DONE $SUMMARY_DIR"

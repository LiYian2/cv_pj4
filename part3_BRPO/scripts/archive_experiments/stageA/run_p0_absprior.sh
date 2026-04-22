#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/bzhang512/CV_Project/part3_BRPO
PY=/home/bzhang512/miniconda3/envs/s3po-gs/bin/python
export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO

INTERNAL_CACHE_ROOT=/data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache
BASE_PLY=$INTERNAL_CACHE_ROOT/after_opt/point_cloud/point_cloud.ply
PREPARE_ROOT=/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare
PSEUDO_CACHE=$PREPARE_ROOT/pseudo_cache_baseline
OUT_ROOT=/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_absprior_calibration_legacy_e1
BASELINE_DIR=$OUT_ROOT/replay_before
SUMMARY_DIR=$OUT_ROOT/summary
REF_JSON=$BASELINE_DIR/replay_eval.json

mkdir -p "$OUT_ROOT" "$SUMMARY_DIR"

echo "P0_START root=$OUT_ROOT"

if [[ ! -f "$REF_JSON" ]]; then
  echo "[P0] baseline replay start"
  "$PY" "$PROJECT_ROOT/scripts/replay_internal_eval.py" \
    --internal-cache-root "$INTERNAL_CACHE_ROOT" \
    --stage-tag after_opt \
    --ply-path "$BASE_PLY" \
    --label before_refine_baseline \
    --save-dir "$BASELINE_DIR"
  echo "P0_BASELINE_DONE $REF_JSON"
else
  echo "[P0] baseline replay already exists: $REF_JSON"
  echo "P0_BASELINE_DONE $REF_JSON"
fi

labels=(
  stageA_noabs_80
  stageA_abs_t1p5_r0p1_80
  stageA_abs_t3_r0p1_80
  stageA_abs_t6_r0p1_80
  stageA_abs_t3_r0p05_80
  stageA_abs_t3_r0p2_80
)
abs_t=(0 1.5 3.0 6.0 3.0 3.0)
abs_r=(0 0.1 0.1 0.1 0.05 0.2)

run_dirs=()
for idx in "${!labels[@]}"; do
  label=${labels[$idx]}
  lt=${abs_t[$idx]}
  lr=${abs_r[$idx]}
  run_dir="$OUT_ROOT/$label"
  run_dirs+=("$run_dir")

  if [[ ! -f "$run_dir/stageA_history.json" ]]; then
    echo "[P0] run start label=$label abs_t=$lt abs_r=$lr"
    "$PY" "$PROJECT_ROOT/scripts/run_pseudo_refinement_v2.py" \
      --ply_path "$BASE_PLY" \
      --pseudo_cache "$PSEUDO_CACHE" \
      --output_dir "$run_dir" \
      --target_side fused \
      --confidence_mask_source brpo \
      --signal_pipeline legacy \
      --stage_mode stageA \
      --stageA_iters 80 \
      --stageA_rgb_mask_mode train_mask \
      --stageA_depth_mask_mode train_mask \
      --stageA_target_depth_mode target_depth_for_refine_v2 \
      --stageA_depth_loss_mode source_aware \
      --stageA_lambda_depth_seed 1.0 \
      --stageA_lambda_depth_dense 0.35 \
      --stageA_lambda_depth_fallback 0.0 \
      --stageA_abs_pose_robust charbonnier \
      --stageA_abs_pose_scale_source render_depth_trainmask_median \
      --stageA_lambda_abs_t "$lt" \
      --stageA_lambda_abs_r "$lr" \
      --stageA_log_grad_contrib \
      --stageA_log_grad_interval 10 \
      --num_pseudo_views 4 \
      --seed 0
  else
    echo "[P0] run already exists label=$label"
  fi

  if [[ ! -f "$run_dir/replay_eval/replay_eval.json" ]]; then
    echo "[P0] replay start label=$label"
    "$PY" "$PROJECT_ROOT/scripts/replay_internal_eval.py" \
      --internal-cache-root "$INTERNAL_CACHE_ROOT" \
      --stage-tag after_opt \
      --ply-path "$run_dir/refined_gaussians.ply" \
      --label "$label" \
      --save-dir "$run_dir/replay_eval"
  else
    echo "[P0] replay already exists label=$label"
  fi

  "$PY" "$PROJECT_ROOT/scripts/diagnostics/summarize_stageA_compare.py" \
    --run-dirs "${run_dirs[@]}" \
    --output-root "$SUMMARY_DIR" \
    --reference-replay-json "$REF_JSON"

  echo "P0_RUN_DONE label=$label run_dir=$run_dir"
done

echo "[P0] final summary refresh"
"$PY" "$PROJECT_ROOT/scripts/diagnostics/summarize_stageA_compare.py" \
  --run-dirs "${run_dirs[@]}" \
  --output-root "$SUMMARY_DIR" \
  --reference-replay-json "$REF_JSON" \
  --strict-replay

echo "P0_SUMMARY_DONE $SUMMARY_DIR"

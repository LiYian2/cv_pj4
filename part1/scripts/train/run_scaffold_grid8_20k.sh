#!/usr/bin/env bash
set -euo pipefail

CONDA=/home/bzhang512/miniconda3/bin/conda
ENV_NAME=scaffold_gs_cu124
GS_ROOT=/home/bzhang512/CV_Project/Scaffold-GS
BASE=/home/bzhang512/CV_Project/part1/part1_data/re10k_1
SRC=$BASE/planA_colmap/gs_scene

LOGDIR=$BASE/logs_scaffold_20k_grid8
mkdir -p "$LOGDIR"

ITERS=$(seq 2000 2000 20000 | xargs)

run_train () {
  local gpu="$1"; shift
  local port="$1"; shift
  local tag="$1"; shift
  local source="$1"; shift
  local model="$1"; shift
  local extra_args="$1"; shift

  local log="$LOGDIR/${tag}.log"

  rm -rf "$model"
  echo "[$(date '+%F %T')] START $tag gpu=$gpu port=$port" | tee "$log"
  echo "source=$source" | tee -a "$log"
  echo "model=$model" | tee -a "$log"
  echo "extra_args=$extra_args" | tee -a "$log"

  cd "$GS_ROOT"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run -n "$ENV_NAME" python train.py \
    -s "$source" \
    -m "$model" \
    --eval \
    --port "$port" \
    --iterations 20000 \
    --test_iterations $ITERS \
    --save_iterations $ITERS \
    $extra_args 2>&1 | tee -a "$log"

  echo "[$(date '+%F %T')] END $tag gpu=$gpu port=$port" | tee -a "$log"
}

worker_gpu0 () {
  run_train 0 6010 \
    current_default_20k \
    "$SRC" \
    "$BASE/scaffold_test_planA279/output_scaffold_current_default_20k" \
    ""

  run_train 0 6010 \
    vs0_uif16_app0_20k \
    "$SRC" \
    "$BASE/scaffold_test_planA279/output_scaffold_vs0_uif16_app0_20k" \
    "--voxel_size 0 --update_init_factor 16 --appearance_dim 0"

  run_train 0 6010 \
    vs0_uif32_app0_20k \
    "$SRC" \
    "$BASE/scaffold_test_planA279/output_scaffold_vs0_uif32_app0_20k" \
    "--voxel_size 0 --update_init_factor 32 --appearance_dim 0"

  run_train 0 6010 \
    vs0_uif16_app16_u8k_d5e4_20k \
    "$SRC" \
    "$BASE/scaffold_test_planA279/output_scaffold_vs0_uif16_app16_u8k_d5e4_20k" \
    "--voxel_size 0 --update_init_factor 16 --appearance_dim 16 --update_until 8000 --densify_grad_threshold 0.0005"
}

worker_gpu1 () {
  run_train 1 6011 \
    vs0_uif16_app0_u8k_d5e4_20k \
    "$SRC" \
    "$BASE/scaffold_test_planA279/output_scaffold_vs0_uif16_app0_u8k_d5e4_20k" \
    "--voxel_size 0 --update_init_factor 16 --appearance_dim 0 --update_until 8000 --densify_grad_threshold 0.0005"

  run_train 1 6011 \
    vs0_uif8_app0_20k \
    "$SRC" \
    "$BASE/scaffold_test_planA279/output_scaffold_vs0_uif8_app0_20k" \
    "--voxel_size 0 --update_init_factor 8 --appearance_dim 0"

  run_train 1 6011 \
    vs0_uif16_app16_20k \
    "$SRC" \
    "$BASE/scaffold_test_planA279/output_scaffold_vs0_uif16_app16_20k" \
    "--voxel_size 0 --update_init_factor 16 --appearance_dim 16"

  run_train 1 6011 \
    vs0_uif16_app0_ud2_20k \
    "$SRC" \
    "$BASE/scaffold_test_planA279/output_scaffold_vs0_uif16_app0_ud2_20k" \
    "--voxel_size 0 --update_init_factor 16 --appearance_dim 0 --update_depth 2"
}

worker_gpu0 &
PID0=$!
worker_gpu1 &
PID1=$!

wait "$PID0"
wait "$PID1"

echo "[$(date '+%F %T')] ALL DONE"
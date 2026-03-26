#!/usr/bin/env bash
set -euo pipefail

CONDA=/home/bzhang512/miniconda3/bin/conda
ENV_NAME=scaffold_gs_cu124
GS_ROOT=/home/bzhang512/CV_Project/Scaffold-GS

BASE=/home/bzhang512/CV_Project/part1/part1_data/re10k_1
OUTROOT=$BASE/scaffold_40k_verification
LOGDIR=$OUTROOT/logs

SRC_COLMAP=$BASE/planA_colmap/gs_scene
SRC_COLMAP96=$BASE/planA_colmap_96/gs_scene
SRC_VGGT96=$BASE/planB_vggt/gs_scene

mkdir -p "$LOGDIR"

ITERS=$(seq 2000 2000 40000 | xargs)

run_train () {
  local gpu="$1"; shift
  local port="$1"; shift
  local dataset_tag="$1"; shift
  local variant_tag="$1"; shift
  local source="$1"; shift
  local extra_args="$1"; shift

  local tag="${dataset_tag}_${variant_tag}"
  local model="$OUTROOT/${tag}"
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
    --iterations 40000 \
    --test_iterations $ITERS \
    --save_iterations $ITERS \
    $extra_args 2>&1 | tee -a "$log"

  echo "[$(date '+%F %T')] END $tag gpu=$gpu port=$port" | tee -a "$log"
}

worker_gpu0 () {
  run_train 0 6010 colmap    vs0_uif8_app0_40k      "$SRC_COLMAP"   "--voxel_size 0 --update_init_factor 8  --appearance_dim 0"
  run_train 0 6010 colmap    vs0_uif4_app0_40k      "$SRC_COLMAP"   "--voxel_size 0 --update_init_factor 4  --appearance_dim 0"
  run_train 0 6010 colmap96  vs0_uif8_app0_40k      "$SRC_COLMAP96" "--voxel_size 0 --update_init_factor 8  --appearance_dim 0"
  run_train 0 6010 colmap96  vs0_uif4_app0_40k      "$SRC_COLMAP96" "--voxel_size 0 --update_init_factor 4  --appearance_dim 0"
  run_train 0 6010 vggt96    vs0_uif8_app0_40k      "$SRC_VGGT96"   "--voxel_size 0 --update_init_factor 8  --appearance_dim 0"
  run_train 0 6010 vggt96    vs0_uif4_app0_40k      "$SRC_VGGT96"   "--voxel_size 0 --update_init_factor 4  --appearance_dim 0"
}

worker_gpu1 () {
  run_train 1 6011 colmap    vs0_uif16_app0_40k     "$SRC_COLMAP"   "--voxel_size 0 --update_init_factor 16 --appearance_dim 0"
  run_train 1 6011 colmap    vs0_uif8_app0_ud2_40k  "$SRC_COLMAP"   "--voxel_size 0 --update_init_factor 8  --appearance_dim 0 --update_depth 2"
  run_train 1 6011 colmap96  vs0_uif16_app0_40k     "$SRC_COLMAP96" "--voxel_size 0 --update_init_factor 16 --appearance_dim 0"
  run_train 1 6011 colmap96  vs0_uif8_app0_ud2_40k  "$SRC_COLMAP96" "--voxel_size 0 --update_init_factor 8  --appearance_dim 0 --update_depth 2"
  run_train 1 6011 vggt96    vs0_uif16_app0_40k     "$SRC_VGGT96"   "--voxel_size 0 --update_init_factor 16 --appearance_dim 0"
  run_train 1 6011 vggt96    vs0_uif8_app0_ud2_40k  "$SRC_VGGT96"   "--voxel_size 0 --update_init_factor 8  --appearance_dim 0 --update_depth 2"
}

worker_gpu0 &
PID0=$!
worker_gpu1 &
PID1=$!

wait "$PID0"
wait "$PID1"

echo "[$(date '+%F %T')] ALL DONE"
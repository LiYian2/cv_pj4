#!/usr/bin/env bash
set -euo pipefail
CONDA=/home/bzhang512/miniconda3/bin/conda
GS_ROOT=/home/bzhang512/CV_Project/gaussian-splatting
BASE=/home/bzhang512/CV_Project/part1/part1_data/re10k_1
LOGDIR=$BASE/logs_40k
mkdir -p "$LOGDIR"
ITERS=$(seq 2000 2000 40000 | xargs)
run_train () {
  local gpu="$1"; shift
  local tag="$1"; shift
  local source="$1"; shift
  local model="$1"; shift
  local log="$LOGDIR/${tag}.log"
  rm -rf "$model"
  echo "[$(date +%F %T)] START $tag gpu=$gpu" | tee "$log"
  cd "$GS_ROOT"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run -n 3dgs python train.py \
    -s "$source" \
    -m "$model" \
    --eval \
    --iterations 40000 \
    --test_iterations $ITERS \
    --save_iterations $ITERS 2>&1 | tee -a "$log"
  echo "[$(date +%F %T)] END $tag" | tee -a "$log"
}

run_train 0 planA279_40k_2k "$BASE/planA_colmap/gs_scene" "$BASE/planA_colmap/output_3dgs_40k_2k" &
PID_A=$!
run_train 1 planB96_40k_2k "$BASE/planB_vggt/gs_scene" "$BASE/planB_vggt/output_3dgs_40k_2k" &
PID_B=$!
wait "$PID_A"
run_train 0 planA96_40k_2k "$BASE/planA_colmap_96/gs_scene" "$BASE/planA_colmap_96/output_3dgs_40k_2k"
wait "$PID_B"

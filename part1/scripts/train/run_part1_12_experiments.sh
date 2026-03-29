#!/usr/bin/env bash
set -euo pipefail

# Part 1 Experiments: 12 configurations
# 3 scenes x 2 init methods (colmap_108, vggt_108) x 2 trainers (3dgs, scaffold)
# NO colmap_full

CONDA=/home/bzhang512/miniconda3/bin/conda
GS3D_ROOT=/home/bzhang512/CV_Project/third_party/gaussian-splatting
SCAFFOLD_ROOT=/home/bzhang512/CV_Project/third_party/Scaffold-GS
OUTROOT=/home/bzhang512/CV_Project/output/part1
DATA_ROOT=/home/bzhang512/CV_Project/dataset

ITERS="$(seq 2000 2000 40000 | xargs)"

run_3dgs_post () {
  local gpu="$1"; shift
  local model_dir="$1"; shift
  local log_file="$1"; shift

  cd "$GS3D_ROOT"
  echo "[$(date '+%F %T')] POST 3dgs render.py model=$model_dir" | tee -a "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run -n 3dgs python render.py     -m "$model_dir"     --skip_train     2>&1 | tee -a "$log_file"

  echo "[$(date '+%F %T')] POST 3dgs metrics.py model=$model_dir" | tee -a "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run -n 3dgs python metrics.py     -m "$model_dir"     2>&1 | tee -a "$log_file"
}

run_3dgs () {
  local gpu="$1"; shift
  local port="$1"; shift
  local scene="$1"; shift
  local plan="$1"; shift
  local variant="$1"; shift
  local source="$1"; shift

  local tag="${scene}__${plan}__${variant}__3dgs_default_40k_eval2k"
  local model_dir="$OUTROOT/${scene}/${plan}/${variant}/3dgs/default_40k_eval2k"
  local log_dir="$OUTROOT/${scene}/${plan}/${variant}/3dgs/logs"
  local log_file="$log_dir/${tag}.log"

  mkdir -p "$log_dir"
  rm -rf "$model_dir"

  echo "[$(date '+%F %T')] START $tag gpu=$gpu port=$port" | tee "$log_file"
  echo "source=$source" | tee -a "$log_file"
  echo "model_dir=$model_dir" | tee -a "$log_file"

  cd "$GS3D_ROOT"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run -n 3dgs python train.py \
    -s "$source" \
    -m "$model_dir" \
    --eval \
    --port "$port" \
    --iterations 40000 \
    --test_iterations $ITERS \
    --save_iterations 40000 \
    2>&1 | tee -a "$log_file"

  run_3dgs_post "$gpu" "$model_dir" "$log_file"

  echo "[$(date '+%F %T')] END $tag" | tee -a "$log_file"
}

run_scaffold () {
  local gpu="$1"; shift
  local port="$1"; shift
  local scene="$1"; shift
  local plan="$1"; shift
  local variant="$1"; shift
  local source="$1"; shift
  local extra_args="$1"; shift

  local tag="${scene}__${plan}__${variant}__scaffold_final_40k_eval2k"
  local model_dir="$OUTROOT/${scene}/${plan}/${variant}/scaffold/final_40k_eval2k"
  local log_dir="$OUTROOT/${scene}/${plan}/${variant}/scaffold/logs"
  local log_file="$log_dir/${tag}.log"

  mkdir -p "$log_dir"
  rm -rf "$model_dir"

  echo "[$(date '+%F %T')] START $tag gpu=$gpu port=$port" | tee "$log_file"
  echo "source=$source" | tee -a "$log_file"
  echo "model_dir=$model_dir" | tee -a "$log_file"
  echo "extra_args=$extra_args" | tee -a "$log_file"

  cd "$SCAFFOLD_ROOT"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run -n scaffold_gs_cu124 python train.py \
    -s "$source" \
    -m "$model_dir" \
    --eval \
    --port "$port" \
    --iterations 40000 \
    --test_iterations $ITERS \
    --save_iterations 40000 \
    $extra_args \
    2>&1 | tee -a "$log_file"

  echo "[$(date '+%F %T')] END $tag" | tee -a "$log_file"
}

source_path () {
  local scene="$1"; shift
  local plan="$1"; shift
  local variant="$1"; shift
  echo "$DATA_ROOT/${scene}/part1/${plan}/${variant}/gs_scene"
}

scaffold_args () {
  local variant="$1"; shift
  case "$variant" in
    colmap_108) echo "--voxel_size 0 --update_init_factor 4 --appearance_dim 0" ;;
    vggt_108)   echo "--voxel_size 0 --update_init_factor 16 --appearance_dim 0" ;;
    *) echo "" ;;
  esac
}

# Worker for GPU 0: Re10k-1 and DL3DV-2
worker_gpu0 () {
  # Re10k-1 planA colmap_108
  run_3dgs     0 6210 Re10k-1 planA colmap_108 "$(source_path Re10k-1 planA colmap_108)"
  run_scaffold 0 6210 Re10k-1 planA colmap_108 "$(source_path Re10k-1 planA colmap_108)" "$(scaffold_args colmap_108)"

  # Re10k-1 planB vggt_108
  run_3dgs     0 6210 Re10k-1 planB vggt_108 "$(source_path Re10k-1 planB vggt_108)"
  run_scaffold 0 6210 Re10k-1 planB vggt_108 "$(source_path Re10k-1 planB vggt_108)" "$(scaffold_args vggt_108)"

  # DL3DV-2 planA colmap_108
  run_3dgs     0 6210 DL3DV-2 planA colmap_108 "$(source_path DL3DV-2 planA colmap_108)"
  run_scaffold 0 6210 DL3DV-2 planA colmap_108 "$(source_path DL3DV-2 planA colmap_108)" "$(scaffold_args colmap_108)"
}

# Worker for GPU 1: DL3DV-2 (continued) and 405841
worker_gpu1 () {
  # DL3DV-2 planB vggt_108
  run_3dgs     1 6211 DL3DV-2 planB vggt_108 "$(source_path DL3DV-2 planB vggt_108)"
  run_scaffold 1 6211 DL3DV-2 planB vggt_108 "$(source_path DL3DV-2 planB vggt_108)" "$(scaffold_args vggt_108)"

  # 405841 planA colmap_108
  run_3dgs     1 6211 405841 planA colmap_108 "$(source_path 405841 planA colmap_108)"
  run_scaffold 1 6211 405841 planA colmap_108 "$(source_path 405841 planA colmap_108)" "$(scaffold_args colmap_108)"

  # 405841 planB vggt_108
  run_3dgs     1 6211 405841 planB vggt_108 "$(source_path 405841 planB vggt_108)"
  run_scaffold 1 6211 405841 planB vggt_108 "$(source_path 405841 planB vggt_108)" "$(scaffold_args vggt_108)"
}

worker_gpu0 &
PID0=$!

worker_gpu1 &
PID1=$!

wait "$PID0"
wait "$PID1"

echo "[$(date '+%F %T')] ALL DONE - 12 experiments completed"

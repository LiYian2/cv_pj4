#!/usr/bin/env bash
set -euo pipefail

CONDA=/home/bzhang512/miniconda3/bin/conda
GS3D_ROOT=/home/bzhang512/CV_Project/third_party/gaussian-splatting
SCAFFOLD_ROOT=/home/bzhang512/CV_Project/third_party/Scaffold-GS
OUTROOT=/home/bzhang512/CV_Project/output/part1
DATA_ROOT=/home/bzhang512/CV_Project/dataset

ITERS="$(seq 2000 2000 40000 | xargs)"

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
    colmap_full) echo "--voxel_size 0 --update_init_factor 4 --appearance_dim 0" ;;
    colmap_96)   echo "--voxel_size 0 --update_init_factor 4 --appearance_dim 0" ;;
    vggt_96)     echo "--voxel_size 0 --update_init_factor 16 --appearance_dim 0" ;;
    *) echo "" ;;
  esac
}

worker_gpu0 () {
  run_3dgs     0 6210 Re10k-1 planA colmap_full "$(source_path Re10k-1 planA colmap_full)"
  run_scaffold 0 6210 Re10k-1 planA colmap_full "$(source_path Re10k-1 planA colmap_full)" "$(scaffold_args colmap_full)"

  run_3dgs     0 6210 Re10k-1 planB vggt_96 "$(source_path Re10k-1 planB vggt_96)"
  run_scaffold 0 6210 Re10k-1 planB vggt_96 "$(source_path Re10k-1 planB vggt_96)" "$(scaffold_args vggt_96)"

  run_3dgs     0 6210 DL3DV-2 planA colmap_96 "$(source_path DL3DV-2 planA colmap_96)"
  run_scaffold 0 6210 DL3DV-2 planA colmap_96 "$(source_path DL3DV-2 planA colmap_96)" "$(scaffold_args colmap_96)"

  run_3dgs     0 6210 405841 planA colmap_full "$(source_path 405841 planA colmap_full)"
  run_scaffold 0 6210 405841 planA colmap_full "$(source_path 405841 planA colmap_full)" "$(scaffold_args colmap_full)"

  run_3dgs     0 6210 405841 planB vggt_96 "$(source_path 405841 planB vggt_96)"
}

worker_gpu1 () {
  run_3dgs     1 6211 Re10k-1 planA colmap_96 "$(source_path Re10k-1 planA colmap_96)"
  run_scaffold 1 6211 Re10k-1 planA colmap_96 "$(source_path Re10k-1 planA colmap_96)" "$(scaffold_args colmap_96)"

  run_3dgs     1 6211 DL3DV-2 planA colmap_full "$(source_path DL3DV-2 planA colmap_full)"
  run_scaffold 1 6211 DL3DV-2 planA colmap_full "$(source_path DL3DV-2 planA colmap_full)" "$(scaffold_args colmap_full)"

  run_3dgs     1 6211 DL3DV-2 planB vggt_96 "$(source_path DL3DV-2 planB vggt_96)"
  run_scaffold 1 6211 DL3DV-2 planB vggt_96 "$(source_path DL3DV-2 planB vggt_96)" "$(scaffold_args vggt_96)"

  run_3dgs     1 6211 405841 planA colmap_96 "$(source_path 405841 planA colmap_96)"
  run_scaffold 1 6211 405841 planA colmap_96 "$(source_path 405841 planA colmap_96)" "$(scaffold_args colmap_96)"

  run_scaffold 1 6211 405841 planB vggt_96 "$(source_path 405841 planB vggt_96)" "$(scaffold_args vggt_96)"
}

worker_gpu0 &
PID0=$!

worker_gpu1 &
PID1=$!

wait "$PID0"
wait "$PID1"

echo "[$(date '+%F %T')] ALL DONE"
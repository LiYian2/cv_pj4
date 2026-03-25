#!/usr/bin/env bash
set -euo pipefail
cd /home/bzhang512/CV_Project/gaussian-splatting
LOG=/home/bzhang512/CV_Project/part1/part1_data/re10k_1/logs_40k/planB96_40k_2k.log
rm -rf /home/bzhang512/CV_Project/part1/part1_data/re10k_1/planB_vggt/output_3dgs_40k_2k
echo "[START $(date "+%F %T")] planB96_40k_2k gpu=1 port=6010" > "$LOG"
CUDA_VISIBLE_DEVICES=1 /home/bzhang512/miniconda3/bin/conda run -n 3dgs python train.py \
  -s /home/bzhang512/CV_Project/part1/part1_data/re10k_1/planB_vggt/gs_scene \
  -m /home/bzhang512/CV_Project/part1/part1_data/re10k_1/planB_vggt/output_3dgs_40k_2k \
  --eval --port 6010 --iterations 40000 \
  --test_iterations 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000 32000 34000 36000 38000 40000 \
  --save_iterations 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000 32000 34000 36000 38000 40000 \
  >> "$LOG" 2>&1
echo "[END $(date "+%F %T")] planB96_40k_2k" >> "$LOG"

#!/bin/bash
# Pipeline verification run after code cleanup
# Single run only to verify import chain works

set -euo pipefail

CONFIG="/home/bzhang512/CV_Project/part3_BRPO/configs/e9_verify_pipeline.yaml"
GPU_ID=1
OUTPUT_DIR="/data3/bzhang512/part3_online_mapping_experiments/E9_verify_pipeline"
LOG_FILE="${OUTPUT_DIR}/run_log.txt"

mkdir -p "${OUTPUT_DIR}"

ulimit -n 65536
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO:
export HF_HOME=/data3/bzhang512/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data3/bzhang512/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/data3/bzhang512/.cache/huggingface/transformers
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

cd /home/bzhang512/CV_Project/third_party/S3PO-GS

echo "=== Pipeline verification run started: $(date) ===" | tee "${LOG_FILE}"

/home/bzhang512/miniconda3/envs/s3po-gs/bin/python slam.py --config "${CONFIG}" 2>&1 | tee -a "${LOG_FILE}"

echo "=== Pipeline verification run completed: $(date) ===" | tee -a "${LOG_FILE}"
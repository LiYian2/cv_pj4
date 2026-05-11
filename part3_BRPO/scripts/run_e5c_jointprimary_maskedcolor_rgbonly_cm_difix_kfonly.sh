#!/bin/bash
set -euo pipefail

ulimit -n 65536
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO:
export HF_HOME=/data3/bzhang512/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data3/bzhang512/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/data3/bzhang512/.cache/huggingface/transformers
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
cd /home/bzhang512/CV_Project/third_party/S3PO-GS
exec /home/bzhang512/miniconda3/envs/s3po-gs/bin/python slam.py --config /home/bzhang512/CV_Project/part3_BRPO/configs/e5c_jointprimary_maskedcolor_rgbonly_cm_difix_kfonly.yaml

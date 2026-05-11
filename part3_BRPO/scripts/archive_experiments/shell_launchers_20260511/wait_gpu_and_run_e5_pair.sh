#!/bin/bash
set -euo pipefail

THRESHOLD_MB=20000
CHECK_INTERVAL=30
SCRIPT_DIR=/home/bzhang512/CV_Project/part3_BRPO

echo "Waiting for GPU memory < ${THRESHOLD_MB}MiB on any GPU (checking every ${CHECK_INTERVAL}s)..."

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Do not overlap E5 with an active E6 pair. E6 is the direct run;
    # E5 is a queued wait job and should start only after E6 fully exits.
    if ps -fu bzhang512 | grep -E "run_e6|configs/e6|e6a_jointprimary|e6b_jointprimary" | grep -v grep >/dev/null; then
        echo "[$TIMESTAMP] E6 still active; waiting before starting E5 pair."
        sleep "$CHECK_INTERVAL"
        continue
    fi
    
    # Read per-GPU memory usage
    mapfile -t GPU_MEM < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    
    for GPU_ID in "${!GPU_MEM[@]}"; do
        USED_MB=${GPU_MEM[$GPU_ID]%%[[:space:]]}
        
        echo "[$TIMESTAMP] GPU $GPU_ID: ${USED_MB}MiB used"
        
        if [ "$USED_MB" -lt "$THRESHOLD_MB" ]; then
            echo "[$TIMESTAMP] GPU $GPU_ID available (< ${THRESHOLD_MB}MiB)! Starting E5 pair on GPU $GPU_ID..."
            export CUDA_VISIBLE_DEVICES=$GPU_ID
            cd "$SCRIPT_DIR"
            exec "$SCRIPT_DIR/scripts/run_e5_pair.sh"
        fi
    done
    
    sleep "$CHECK_INTERVAL"
done

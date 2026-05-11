#!/bin/bash
set -euo pipefail
cd /home/bzhang512/CV_Project/part3_BRPO
./scripts/run_d6_midpoint_exact_difix.sh
./scripts/run_d7_midpoint_exact_difix_gn.sh

#!/bin/bash
set -euo pipefail
cd /home/bzhang512/CV_Project/part3_BRPO
./scripts/run_d8_midpoint_paper_difix.sh
./scripts/run_d9_midpoint_paper_difix_gn.sh
./scripts/run_d10_midpoint_paper_depthconf_difix_gn.sh
./scripts/run_d11_uniform2_exact_difix.sh
./scripts/run_d12_midpoint_exact_lambda2_difix.sh
./scripts/run_d13_midpoint_exact_tau_strict_difix.sh

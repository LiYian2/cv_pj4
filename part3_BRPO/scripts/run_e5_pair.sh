#!/bin/bash
set -euo pipefail
cd /home/bzhang512/CV_Project/part3_BRPO
./scripts/run_e5a_jointprimary_maskedcolor_rgbonly_cm_difix.sh
./scripts/run_e5b_jointprimary_maskedcolor_rgbonly_cm_difix_realpose.sh

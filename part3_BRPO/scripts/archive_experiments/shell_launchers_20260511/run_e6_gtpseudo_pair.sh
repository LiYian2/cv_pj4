#!/bin/bash
set -euo pipefail
cd /home/bzhang512/CV_Project/part3_BRPO
./scripts/run_e6a_jointprimary_maskedcolor_rgbonly_cm_gtpseudo_nodifix.sh
./scripts/run_e6b_jointprimary_maskedcolor_rgbonly_cm_gtpseudo_nodifix_realpose.sh

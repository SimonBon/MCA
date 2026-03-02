#!/bin/bash
# Run LARS-setup experiments (CIM + EarlyFusion32) on IMC_NB and MIBI_TNBC.
#
# Usage:
#   bash run_lars.sh [gpu]
#
# Examples:
#   bash run_lars.sh
#   bash run_lars.sh 1

set -e

GPU=${1:-0}

TRAIN=/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/mmselfsup/tools/train.py
CONFIGS=/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/MCA/configs/_experiments_

run() {
    local cfg=$1
    if [ ! -f "$cfg" ]; then
        echo "  SKIP (not found): $cfg"
        return
    fi
    echo ""
    echo "============================================================"
    echo "  START: $(basename $cfg .py)  |  GPU $GPU  |  $(date '+%H:%M:%S')"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=$GPU python $TRAIN "$cfg"
    echo "  DONE:  $(basename $cfg .py)  |  $(date '+%H:%M:%S')"
}

echo "============================================================"
echo "  LARS experiments: IMC_NB + MIBI_TNBC"
echo "  GPU     : $GPU"
echo "  Started : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ── IMC_NB ────────────────────────────────────────────────────────────────────
run $CONFIGS/IMC_NB/CIM_VICReg_LARS.py
run $CONFIGS/IMC_NB/EarlyFusion32_VICReg_LARS.py

# ── MIBI_TNBC ─────────────────────────────────────────────────────────────────
run $CONFIGS/MIBI_TNBC/CIM_VICReg_LARS.py
run $CONFIGS/MIBI_TNBC/EarlyFusion32_VICReg_LARS.py

echo ""
echo "============================================================"
echo "  ALL DONE  |  $(date '+%H:%M:%S')"
echo "============================================================"

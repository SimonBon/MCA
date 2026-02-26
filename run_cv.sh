#!/bin/bash
# Run all main models across all MIBI_TNBC CV splits on a single GPU.
#
# Usage:
#   bash run_cv.sh [gpu] [dataset_prefix]
#
# Examples:
#   bash run_cv.sh 0
#   bash run_cv.sh 2 MIBI_TNBC_CV
#
# This runs all splits for one model before moving to the next:
# CIM → CIM_Norm → CIM_ProgFusion → EarlyFusion32 → ResNet, each across splits 0-4.

set -e

GPU=${1:-0}
PREFIX=${2:-MIBI_TNBC_CV}
N_SPLITS=5

TRAIN=/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/mmselfsup/tools/train.py
CONFIG_BASE=/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/MCA/configs/_experiments_

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
echo "  CV prefix : $PREFIX  (splits 0-$((N_SPLITS-1)))"
echo "  GPU       : $GPU"
echo "  Started   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

for model in CIM_VICReg CIM_Norm_VICReg CIM_ProgFusion_VICReg EarlyFusion32_VICReg ResNet_VICReg; do
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  Model: $model"
    echo "────────────────────────────────────────────────────────────"

    for k in $(seq 0 $((N_SPLITS-1))); do
        run $CONFIG_BASE/${PREFIX}${k}/${model}.py
    done
done

echo ""
echo "============================================================"
echo "  ALL CV SPLITS DONE  |  $(date '+%H:%M:%S')"
echo "============================================================"

#!/bin/bash
# Run all experiments for a given dataset consecutively.
#
# Usage:
#   bash run_experiments.sh <dataset> [gpu]
#
# Examples:
#   bash run_experiments.sh CODEX_cHL
#   bash run_experiments.sh CODEX_DLBCL 1
#   bash run_experiments.sh IMC_NB 2
#   bash run_experiments.sh MIBI_TNBC 3
#
# Available datasets: CODEX_cHL  CODEX_DLBCL  IMC_NB  MIBI_TNBC

set -e

DATASET=${1:-}
GPU=${2:-0}

if [ -z "$DATASET" ]; then
    echo "ERROR: no dataset specified."
    echo "Usage: bash run_experiments.sh <dataset> [gpu]"
    echo "Available datasets: CODEX_cHL  CODEX_DLBCL  IMC_NB  MIBI_TNBC"
    exit 1
fi

TRAIN=/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/mmselfsup/tools/train.py
CONFIGS=/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/MCA/configs/_experiments_/$DATASET

if [ ! -d "$CONFIGS" ]; then
    echo "ERROR: config folder not found: $CONFIGS"
    exit 1
fi

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
echo "  Dataset : $DATASET"
echo "  GPU     : $GPU"
echo "  Started : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ── CIM variants ──────────────────────────────────────────────────────────────
run $CONFIGS/CIM_VICReg.py
run $CONFIGS/CIM_Norm_VICReg.py         # CIM with input L2-norm
run $CONFIGS/CIM_ProgFusion_VICReg.py   # CIM with progressive cross-marker fusion

# ── ResNet baseline ───────────────────────────────────────────────────────────
run $CONFIGS/ResNet_VICReg.py

# ── EarlyFusion baseline ──────────────────────────────────────────────────────
run $CONFIGS/EarlyFusion32_VICReg.py    # stem_width=32 (same width as CIM)

echo ""
echo "============================================================"
echo "  ALL DONE  |  $DATASET  |  $(date '+%H:%M:%S')"
echo "============================================================"

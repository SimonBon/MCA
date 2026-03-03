#!/bin/bash
# Run all paper experiments — 4 models × 4 datasets, LARS 4k iters.
#
# Results land in z_RUNS/paper/<dataset>_<model>/metrics.json
# Already-completed runs (metrics.json exists) are skipped automatically.
#
# Usage:
#   bash run_paper.sh [gpu]          # run all sequentially on one GPU
#   bash run_paper.sh 0 &            # run in background on GPU 0
#
# Models : CIM | CIM_ProgFusion | EarlyFusion32 | ResNet
# Datasets: CODEX_cHL_KRONOS18 | CODEX_DLBCL | IMC_NB_TumorSub | MIBI_TNBC

set -e

GPU=${1:-0}

TRAIN=/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/mmselfsup/tools/train.py
CONFIGS=/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/MCA/configs/_experiments_/paper
RUNS=/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/MCA/z_RUNS/paper

run() {
    local cfg=$1
    local name=$(basename $(dirname $cfg))/$(basename $cfg .py)
    local work_dir=$(python3 -c "
import re
txt = open('$cfg').read()
m = re.search(r\"work_dir\s*=\s*'([^']+)'\", txt)
print(m.group(1) if m else '')
")
    local metrics="$work_dir/metrics.json"

    if [ -f "$metrics" ]; then
        echo "  SKIP (done): $name"
        return
    fi

    if [ ! -f "$cfg" ]; then
        echo "  SKIP (not found): $cfg"
        return
    fi

    echo ""
    echo "============================================================"
    echo "  START: $name"
    echo "  GPU $GPU  |  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=$GPU python $TRAIN "$cfg"
    echo "  DONE:  $name  |  $(date '+%H:%M:%S')"
}

echo "============================================================"
echo "  Paper experiments — LARS 4k"
echo "  GPU     : $GPU"
echo "  Started : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ── CODEX_cHL KRONOS18 ────────────────────────────────────────────────────────
run $CONFIGS/CODEX_cHL_KRONOS18/CIM_VICReg_LARS.py
run $CONFIGS/CODEX_cHL_KRONOS18/CIM_ProgFusion_VICReg_LARS.py
run $CONFIGS/CODEX_cHL_KRONOS18/EarlyFusion32_VICReg_LARS.py
run $CONFIGS/CODEX_cHL_KRONOS18/ResNet_VICReg_LARS.py

# ── CODEX_DLBCL ───────────────────────────────────────────────────────────────
run $CONFIGS/CODEX_DLBCL/CIM_VICReg_LARS.py
run $CONFIGS/CODEX_DLBCL/CIM_ProgFusion_VICReg_LARS.py
run $CONFIGS/CODEX_DLBCL/EarlyFusion32_VICReg_LARS.py
run $CONFIGS/CODEX_DLBCL/ResNet_VICReg_LARS.py

# ── IMC_NB_TumorSub ───────────────────────────────────────────────────────────
run $CONFIGS/IMC_NB_TumorSub/CIM_VICReg_LARS.py
run $CONFIGS/IMC_NB_TumorSub/CIM_ProgFusion_VICReg_LARS.py
run $CONFIGS/IMC_NB_TumorSub/EarlyFusion32_VICReg_LARS.py
run $CONFIGS/IMC_NB_TumorSub/ResNet_VICReg_LARS.py

# ── MIBI_TNBC ─────────────────────────────────────────────────────────────────
run $CONFIGS/MIBI_TNBC/CIM_VICReg_LARS.py
run $CONFIGS/MIBI_TNBC/CIM_ProgFusion_VICReg_LARS.py
run $CONFIGS/MIBI_TNBC/EarlyFusion32_VICReg_LARS.py
run $CONFIGS/MIBI_TNBC/ResNet_VICReg_LARS.py

echo ""
echo "============================================================"
echo "  ALL DONE  |  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

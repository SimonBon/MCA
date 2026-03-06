#!/bin/bash
# Run region analysis for all datasets — 4 models each, GPU-accelerated embedding.
#
# Embeddings are cached to z_RUNS/region_analysis_paper/<DATASET>/embeddings/
# so re-running after the first time is fast (only recomputes plots).
#
# Usage:
#   bash run_region_analysis.sh [gpu]       # all datasets sequentially
#   bash run_region_analysis.sh 0           # GPU 0

GPU=${1:-0}
SCRIPT=/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/MCA/tools/region_analysis.py

echo "============================================================"
echo "  Region Analysis — all models × all datasets"
echo "  GPU: $GPU"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

cd /home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/MCA

# cHL KRONOS18 (18 markers)
echo ""
echo "--- CODEX_cHL_KRONOS18 ---"
CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT --dataset CODEX_cHL_KRONOS18 --gpu 0

# cHL full panel (41 markers)
echo ""
echo "--- CODEX_cHL ---"
CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT --dataset CODEX_cHL --gpu 0

# MIBI TNBC (40 patients, cross-patient reproducibility)
echo ""
echo "--- MIBI_TNBC ---"
CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT --dataset MIBI_TNBC --gpu 0

echo ""
echo "============================================================"
echo "  ALL DONE  |  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

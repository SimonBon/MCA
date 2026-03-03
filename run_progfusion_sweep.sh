#!/bin/bash
# Sweep over WideModelProgressiveFusion architecture configs (500 iters).
# Dataset: IMC_NB_TumorSub (31 markers, 19 tumour-subtype classes)
# Eval:    LP (epochs=500, max_samples=15k), kNN, clustering, silhouette — no label efficiency
#
# Usage:
#   bash run_progfusion_sweep.sh [gpu]
#
# Examples:
#   bash run_progfusion_sweep.sh
#   bash run_progfusion_sweep.sh 1
#
# Variants (500 iters each):
#   sw32_bw2_l11   stem=32 block=2 layers=[1,1]     — baseline
#   sw32_bw2_l22   stem=32 block=2 layers=[2,2]     — deeper stages
#   sw32_bw2_l33   stem=32 block=2 layers=[3,3]     — very deep stages
#   sw32_bw2_l111  stem=32 block=2 layers=[1,1,1]   — 3 fusion stages
#   sw32_bw2_l222  stem=32 block=2 layers=[2,2,2]   — 3 deep fusion stages
#   sw32_bw4_l11   stem=32 block=4 layers=[1,1]     — wider FFN
#   sw16_bw2_l11   stem=16 block=2 layers=[1,1]     — small
#   sw16_bw2_l22   stem=16 block=2 layers=[2,2]     — small + deeper
#   sw16_bw2_l111  stem=16 block=2 layers=[1,1,1]   — small + 3 stages
#   sw8_bw2_l11    stem=8  block=2 layers=[1,1]     — very small
#   sw8_bw2_l22    stem=8  block=2 layers=[2,2]     — very small + deeper
#   sw64_bw2_l11   stem=64 block=2 layers=[1,1]     — large

set -e

GPU=${1:-0}

TRAIN=/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/mmselfsup/tools/train.py
CONFIGS=/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/MCA/configs/_experiments_/ProgFusion_sweep

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
echo "  ProgFusion architecture sweep (500 iters)"
echo "  Dataset : IMC_NB_TumorSub"
echo "  GPU     : $GPU"
echo "  Started : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

run $CONFIGS/sw32_bw2_l11.py
run $CONFIGS/sw32_bw2_l22.py
run $CONFIGS/sw32_bw2_l33.py
run $CONFIGS/sw32_bw2_l111.py
run $CONFIGS/sw32_bw2_l222.py
run $CONFIGS/sw32_bw4_l11.py
run $CONFIGS/sw16_bw2_l11.py
run $CONFIGS/sw16_bw2_l22.py
run $CONFIGS/sw16_bw2_l111.py
run $CONFIGS/sw8_bw2_l11.py
run $CONFIGS/sw8_bw2_l22.py
run $CONFIGS/sw64_bw2_l11.py

echo ""
echo "============================================================"
echo "  ALL DONE  |  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

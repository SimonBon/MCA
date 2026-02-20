#!/bin/bash
# Run all CODEX_cHL ablation experiments consecutively.
# Shortest runs first, longest last.
#
# Usage:
#   screen -S ablations bash run_ablations.sh        # GPU 0 (default)
#   screen -S ablations bash run_ablations.sh 1      # GPU 1

set -e

GPU=${1:-0}
TRAIN=/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/mmselfsup/tools/train.py
CONFIGS=/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/MCA/configs/_experiments_/CODEX_cHL

run() {
    echo ""
    echo "============================================================"
    echo "  START: $(basename $1 .py)  |  GPU $GPU  |  $(date '+%H:%M:%S')"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=$GPU python $TRAIN "$1"
    echo "  DONE:  $(basename $1 .py)  |  $(date '+%H:%M:%S')"
}

# ── Augmentation ablation (1000 iters each) ───────────────────────────────────
run $CONFIGS/ablations/aug_none.py
run $CONFIGS/ablations/aug_spatial.py
run $CONFIGS/ablations/aug_low.py
run $CONFIGS/CIM_VICReg.py           # baseline: aug=high, iters=1000, stem=32

# ── Model capacity ablation (1000 iters each) ─────────────────────────────────
run $CONFIGS/ablations/capacity_stem8.py
run $CONFIGS/ablations/capacity_stem16.py
# stem=32 is already covered by CIM_VICReg.py above
run $CONFIGS/ablations/capacity_stem64.py

# ── Training length ablation (high aug, stem=32) ──────────────────────────────
run $CONFIGS/ablations/iters_500.py
run $CONFIGS/ablations/iters_2000.py
run $CONFIGS/ablations/iters_5000.py
run $CONFIGS/ablations/iters_10000.py  # longest — runs last

echo ""
echo "============================================================"
echo "  ALL DONE  |  $(date '+%H:%M:%S')"
echo "============================================================"

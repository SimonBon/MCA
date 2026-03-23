#!/bin/bash
#SBATCH --job-name=kronos_MIBI
#SBATCH --output=/home/sgutwein/logs/kronos_MIBI_%A_%a.log
#SBATCH --error=/home/sgutwein/logs/kronos_MIBI_%A_%a.log
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l4_gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-4

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310
export PYTHONPATH=/home/sgutwein/src:$PYTHONPATH

BASE=/nobackup/lab_taschner-mandl/simongutwein
KRONOS_META=$BASE/model_assets/kronos/models--MahmoodLab--kronos/snapshots/8edc2719ad67b2e2b766073b35c6cf8e6f5da516/marker_metadata.csv
FOLD=${SLURM_ARRAY_TASK_ID}

# 26/37 markers have known KRONOS IDs after aliases:
#   IDO→IDO1, Beta catenin→B-CATENIN, HLA-DR→HLA_DR, PD-L1→PDL1
# Unknowns (11): CD209, CD63, H3K9ac, HLA_Class_1, Keratin17, Keratin6,
#   OX40, Pan-Keratin, SMA, dsDNA, phospho-S6 → fallback IDs + data stats

python -u /home/sgutwein/src/MCA/tools/baseline_kronos.py \
    --h5              $BASE/h5_files/MIBI_TNBC/MIBI_TNBC.h5 \
    --markers         $BASE/h5_files/MIBI_TNBC/used_markers.txt \
    --train           $BASE/h5_files/MIBI_TNBC/cv_splits_paper/split_${FOLD}/train.txt \
    --val             $BASE/h5_files/MIBI_TNBC/cv_splits_paper/split_${FOLD}/test.txt \
    --out             $BASE/z_RUNS/paper_clean/MIBI_TNBC/KRONOS/fold_${FOLD} \
    --checkpoint      $BASE/model_assets/kronos/models--MahmoodLab--kronos/snapshots/8edc2719ad67b2e2b766073b35c6cf8e6f5da516/kronos_vits16_model.pt \
    --kronos_src      /home/sgutwein/src/KRONOS \
    --marker_meta_csv $KRONOS_META \
    --marker_max_values 1.0 \
    --patch_size      64 \
    --batch_size      64 \
    --num_workers     8 \
    --lp_subsample    50000 \
    --ignore          "Unidentified"

echo "Done fold ${FOLD}: $(date)"

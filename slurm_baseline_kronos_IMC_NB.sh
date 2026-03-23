#!/bin/bash
#SBATCH --job-name=kronos_IMC
#SBATCH --output=/home/sgutwein/logs/kronos_IMC_%A_%a.log
#SBATCH --error=/home/sgutwein/logs/kronos_IMC_%A_%a.log
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l4_gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-4

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310
export PYTHONPATH=/home/sgutwein/src:$PYTHONPATH

BASE=/nobackup/lab_taschner-mandl/simongutwein
KRONOS_META=$BASE/model_assets/kronos/models--MahmoodLab--kronos/snapshots/8edc2719ad67b2e2b766073b35c6cf8e6f5da516/marker_metadata.csv
FOLD=${SLURM_ARRAY_TASK_ID}

# 20/31 markers have known KRONOS IDs (CD8a→CD8, Ki-67→KI67, CD274→PDL1, CD279→PD1,
# HLA-DR→HLA_DR, GZMB, Vimentin direct). The remaining 11 (CD24, CHGA, CXCR4, DNA2,
# ELAVL4, GD2, HLA-ABC, LUM, PRPH, S100B, SOX10) use sequential fallback IDs + data stats.

python -u /home/sgutwein/src/MCA/tools/baseline_kronos.py \
    --h5             $BASE/h5_files/IMC_NB_TumorSub/IMC_NB_TumorSub.h5 \
    --markers        $BASE/h5_files/IMC_NB_TumorSub/used_markers.txt \
    --train          $BASE/h5_files/IMC_NB_TumorSub/cv_splits/split_${FOLD}/train.txt \
    --val            $BASE/h5_files/IMC_NB_TumorSub/cv_splits/split_${FOLD}/test.txt \
    --out            $BASE/z_RUNS/paper_clean/IMC_NB_TumorSub/KRONOS/fold_${FOLD} \
    --checkpoint     $BASE/model_assets/kronos/models--MahmoodLab--kronos/snapshots/8edc2719ad67b2e2b766073b35c6cf8e6f5da516/kronos_vits16_model.pt \
    --kronos_src     /home/sgutwein/src/KRONOS \
    --marker_meta_csv $KRONOS_META \
    --marker_max_values 1.0 \
    --patch_size     64 \
    --batch_size     64 \
    --num_workers    8 \
    --lp_subsample   50000 \
    --ignore         "Other" "Seg Artifact"

echo "Done fold ${FOLD}: $(date)"

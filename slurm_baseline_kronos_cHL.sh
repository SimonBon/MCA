#!/bin/bash
#SBATCH --job-name=kronos_cHL
#SBATCH --output=/home/sgutwein/logs/kronos_cHL_%j.log
#SBATCH --error=/home/sgutwein/logs/kronos_cHL_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l4_gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310
export PYTHONPATH=/home/sgutwein/src:$PYTHONPATH

BASE=/nobackup/lab_taschner-mandl/simongutwein
KRONOS_META=$BASE/model_assets/kronos/models--MahmoodLab--kronos/snapshots/8edc2719ad67b2e2b766073b35c6cf8e6f5da516/marker_metadata.csv

python -u /home/sgutwein/src/MCA/tools/baseline_kronos.py \
    --h5             $BASE/h5_files/CODEX_cHL/CODEX_cHL.h5 \
    --markers        $BASE/h5_files/CODEX_cHL/used_markers.txt \
    --train          $BASE/h5_files/CODEX_cHL/train.txt \
    --val            $BASE/h5_files/CODEX_cHL/test.txt \
    --out            $BASE/z_RUNS/paper_clean/CODEX_cHL/KRONOS \
    --checkpoint     $BASE/model_assets/kronos/models--MahmoodLab--kronos/snapshots/8edc2719ad67b2e2b766073b35c6cf8e6f5da516/kronos_vits16_model.pt \
    --kronos_src     /home/sgutwein/src/KRONOS \
    --marker_meta_csv $KRONOS_META \
    --marker_max_values 1.0 \
    --patch_size     64 \
    --batch_size     64 \
    --num_workers    8 \
    --annotation_map "Cytotoxic CD8:CD8,TReg:Treg" \
    --ignore         "Seg Artifact"

echo "Done: $(date)"

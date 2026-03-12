#!/bin/bash
#SBATCH --job-name=region_cHL_CIM
#SBATCH --output=/home/sgutwein/logs/region_cHL_CIM_%j.log
#SBATCH --error=/home/sgutwein/logs/region_cHL_CIM_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l4_gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310

BASE=/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/paper
H5=/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/CODEX_cHL.h5
MARKERS=/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/used_markers.txt
OUT=/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/region_analysis/CODEX_cHL_CIM

mkdir -p "$OUT"

python /home/sgutwein/src/MCA/tools/region_analysis.py \
    --model_dir  "$BASE/CODEX_cHL/CIM" \
    --h5         "$H5" \
    --markers    "$MARKERS" \
    --out        "$OUT" \
    --patch_size 64 \
    --k          6 \
    --n_jobs     8 \
    --display_markers Cytokeritin CD3 CD68 Vimentin DAPI-01 CD20

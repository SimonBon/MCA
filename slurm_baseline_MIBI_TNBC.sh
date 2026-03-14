#!/bin/bash
#SBATCH --job-name=ExprBaseline
#SBATCH --output=/home/sgutwein/logs/ExprBaseline_%j.log
#SBATCH --error=/home/sgutwein/logs/ExprBaseline_%j.log
#SBATCH --partition=shortq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310

export PYTHONPATH=/home/sgutwein/src:$PYTHONPATH

_DATA=/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC

echo "Starting: Expression Baseline MIBI_TNBC (mean)"
echo "Date: $(date)"

python /home/sgutwein/src/MCA/tools/baseline_expression.py \
    --h5      $_DATA/MIBI_TNBC.h5 \
    --markers $_DATA/used_markers.txt \
    --train   $_DATA/train.txt \
    --val     $_DATA/val.txt \
    --out     /nobackup/lab_taschner-mandl/simongutwein/z_RUNS/MIBI_TNBC_ExprBaseline_mean \
    --feat    mean \
    --patch_size 32 \
    --n_jobs  16

echo "---"
echo "Starting: Expression Baseline MIBI_TNBC (mean+std)"

python /home/sgutwein/src/MCA/tools/baseline_expression.py \
    --h5      $_DATA/MIBI_TNBC.h5 \
    --markers $_DATA/used_markers.txt \
    --train   $_DATA/train.txt \
    --val     $_DATA/val.txt \
    --out     /nobackup/lab_taschner-mandl/simongutwein/z_RUNS/MIBI_TNBC_ExprBaseline_mean_std \
    --feat    mean+std \
    --patch_size 32 \
    --n_jobs  16

echo "Done: $(date)"

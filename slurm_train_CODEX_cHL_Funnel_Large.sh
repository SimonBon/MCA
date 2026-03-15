#!/bin/bash
#SBATCH --job-name=FunnelLarge_cHL
#SBATCH --output=/home/sgutwein/logs/FunnelLarge_cHL_%j.log
#SBATCH --error=/home/sgutwein/logs/FunnelLarge_cHL_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l4_gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310

export PYTHONPATH=/home/sgutwein/src:$PYTHONPATH
export PYTHONUNBUFFERED=1

TRAIN=/home/sgutwein/src/mmselfsup/tools/train.py
CFGBASE=/home/sgutwein/src/MCA/configs/_experiments_/CODEX_cHL

echo "=== Funnel_Large CODEX_cHL (batch=128) ==="; echo "Date: $(date)"
CUDA_VISIBLE_DEVICES=0 python $TRAIN $CFGBASE/CIM_VICReg_Funnel_Large.py
echo "Done: $(date)"

echo "=== Funnel_Large_Norm CODEX_cHL (batch=128) ==="; echo "Date: $(date)"
CUDA_VISIBLE_DEVICES=0 python $TRAIN $CFGBASE/CIM_VICReg_Funnel_Large_Norm.py
echo "Done: $(date)"

#!/bin/bash
#SBATCH --job-name=CIM_VeryStrongAug
#SBATCH --output=/home/sgutwein/logs/CIM_VeryStrongAug_%j.log
#SBATCH --error=/home/sgutwein/logs/CIM_VeryStrongAug_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l4_gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310

export PYTHONPATH=/home/sgutwein/src:$PYTHONPATH

TRAIN=/home/sgutwein/src/mmselfsup/tools/train.py
CFG=/home/sgutwein/src/MCA/configs/_experiments_/CODEX_cHL/CIM_VICReg_VeryStrongAug.py

echo "Starting: CIM VeryStrongAug on CODEX_cHL"
echo "Config: $CFG"
echo "Date: $(date)"

CUDA_VISIBLE_DEVICES=0 python $TRAIN $CFG

echo "Done: $(date)"

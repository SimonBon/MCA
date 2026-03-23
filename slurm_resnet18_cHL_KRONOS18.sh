#!/bin/bash
#SBATCH --job-name=ResNet18_K18
#SBATCH --output=/home/sgutwein/logs/ResNet18_K18_%j.log
#SBATCH --error=/home/sgutwein/logs/ResNet18_K18_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l4_gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310
export PYTHONPATH=/home/sgutwein/src:$PYTHONPATH

python /home/sgutwein/src/mmselfsup/tools/train.py \
    /home/sgutwein/src/MCA/configs/_experiments_/paper/CODEX_cHL_KRONOS18/ResNet18_VICReg.py

echo "Done: $(date)"

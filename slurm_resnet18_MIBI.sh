#!/bin/bash
#SBATCH --job-name=ResNet18_MIBI
#SBATCH --output=/home/sgutwein/logs/ResNet18_MIBI_%A_%a.log
#SBATCH --error=/home/sgutwein/logs/ResNet18_MIBI_%A_%a.log
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l4_gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-4

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310
export PYTHONPATH=/home/sgutwein/src:$PYTHONPATH

python /home/sgutwein/src/mmselfsup/tools/train.py \
    /home/sgutwein/src/MCA/configs/_experiments_/paper/MIBI_TNBC/ResNet18_VICReg_fold${SLURM_ARRAY_TASK_ID}.py

echo "Done fold ${SLURM_ARRAY_TASK_ID}: $(date)"

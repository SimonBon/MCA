#!/bin/bash
#SBATCH --job-name=Funnel_Large
#SBATCH --output=/home/sgutwein/logs/Funnel_Large_%j.log
#SBATCH --error=/home/sgutwein/logs/Funnel_Large_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l4_gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310

export PYTHONPATH=/home/sgutwein/src:$PYTHONPATH
export PYTHONUNBUFFERED=1

TRAIN=/home/sgutwein/src/mmselfsup/tools/train.py
CFG=/home/sgutwein/src/MCA/configs/_experiments_/MIBI_TNBC/CIM_VICReg_Funnel_Large.py

echo "Starting: CIM_Funnel_Large 16k iters MIBI_TNBC"
echo "Config: $CFG"
echo "Date: $(date)"

CUDA_VISIBLE_DEVICES=0 python $TRAIN $CFG

echo "Done: $(date)"

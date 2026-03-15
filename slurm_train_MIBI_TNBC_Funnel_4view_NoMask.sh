#!/bin/bash
#SBATCH --job-name=Funnel_4v_NoMask
#SBATCH --output=/home/sgutwein/logs/Funnel_4v_NoMask_%j.log
#SBATCH --error=/home/sgutwein/logs/Funnel_4v_NoMask_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l4_gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310

export PYTHONPATH=/home/sgutwein/src:$PYTHONPATH

CFG=/home/sgutwein/src/MCA/configs/_experiments_/MIBI_TNBC/CIM_VICReg_Funnel_4view_NoMask.py

echo "Starting: Funnel_4view NoMask (context included) MIBI_TNBC"
echo "Config: $CFG"
echo "Date: $(date)"

CUDA_VISIBLE_DEVICES=0 python /home/sgutwein/src/mmselfsup/tools/train.py $CFG

echo "Done: $(date)"

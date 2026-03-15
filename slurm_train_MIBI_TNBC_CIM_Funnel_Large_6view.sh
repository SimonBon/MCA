#!/bin/bash
#SBATCH --job-name=FunnelLarge_6v
#SBATCH --output=/home/sgutwein/logs/FunnelLarge_6v_%j.log
#SBATCH --error=/home/sgutwein/logs/FunnelLarge_6v_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100pcie:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310

export PYTHONPATH=/home/sgutwein/src:$PYTHONPATH
export PYTHONUNBUFFERED=1

CFG=/home/sgutwein/src/MCA/configs/_experiments_/MIBI_TNBC/CIM_VICReg_Funnel_Large_6view.py

echo "Starting: CIM_Funnel_Large 6-view 16k iters MIBI_TNBC (H100 PCIe)"
echo "Config: $CFG"
echo "Date: $(date)"

CUDA_VISIBLE_DEVICES=0 python /home/sgutwein/src/mmselfsup/tools/train.py $CFG

echo "Done: $(date)"

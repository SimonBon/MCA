#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l4_gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --job-name=mca-check-env
#SBATCH --output=/nobackup/lab_taschner-mandl/simongutwein/logs/mca-check-env-%j.log

mkdir -p /nobackup/lab_taschner-mandl/simongutwein/logs

echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
conda activate mca

python ~/check_env.py

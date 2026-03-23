#!/bin/bash
#SBATCH --job-name=ResNet18_smoke
#SBATCH --output=/home/sgutwein/logs/ResNet18_smoke_%j.log
#SBATCH --error=/home/sgutwein/logs/ResNet18_smoke_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l4_gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310
export PYTHONPATH=/home/sgutwein/src:$PYTHONPATH

python /home/sgutwein/src/MCA/tools/test_resnet18_cluster.py
EXIT_CODE=$?

echo "Smoke test exit code: ${EXIT_CODE}"
echo "Done: $(date)"
exit ${EXIT_CODE}

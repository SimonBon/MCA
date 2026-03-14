#!/bin/bash
# Interactive SLURM session with GPU for testing the MCA environment.
#
# Usage:
#   bash slurm_interactive_gpu.sh [n_gpus] [hours]
#
# Examples:
#   bash slurm_interactive_gpu.sh           # 1 GPU, 2 hours
#   bash slurm_interactive_gpu.sh 2 4       # 2 GPUs, 4 hours

N_GPUS=${1:-1}
HOURS=${2:-2}

CONDA_ENV="mca"

echo "Requesting ${N_GPUS} l4_gpu(s) for ${HOURS}h..."
echo "Once the shell opens, activate your env and run: python ~/check_env.py"
echo ""

srun \
    --partition=gpu \
    --qos=gpu \
    --gres=gpu:l4_gpu:${N_GPUS} \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=${HOURS}:00:00 \
    --job-name=mca-env-test \
    --pty bash -l

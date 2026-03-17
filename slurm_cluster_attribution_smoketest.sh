#!/bin/bash
#SBATCH --job-name=cluster_attr_smoke
#SBATCH --output=/home/sgutwein/logs/cluster_attr_smoke_%j.log
#SBATCH --error=/home/sgutwein/logs/cluster_attr_smoke_%j.log
#SBATCH --partition=shortq
#SBATCH --qos=shortq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310

ATTR=/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/marker_attribution/CODEX_cHL_CIM_Funnel/attribution.npz
OUT=/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/marker_attribution/CODEX_cHL_CIM_Funnel/clusters_smoketest

python /home/sgutwein/src/MCA/tools/cluster_attribution.py \
    --attribution "$ATTR" \
    --out         "$OUT" \
    --resolution  0.5 \
    --n_neighbors 10

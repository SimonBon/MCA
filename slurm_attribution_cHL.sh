#!/bin/bash
#SBATCH --job-name=attribution_cHL
#SBATCH --output=/home/sgutwein/logs/attribution_cHL_%j.log
#SBATCH --error=/home/sgutwein/logs/attribution_cHL_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l4_gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310
export PYTHONPATH=/home/sgutwein/src:$PYTHONPATH

python /home/sgutwein/src/MCA/tools/marker_attribution.py \
    --model_dir      /nobackup/lab_taschner-mandl/simongutwein/z_RUNS/paper_clean/CODEX_cHL/CIM_Funnel_Large \
    --h5             /nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/CODEX_cHL.h5 \
    --markers        /nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/used_markers.txt \
    --val            /nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/test.txt \
    --out            /nobackup/lab_taschner-mandl/simongutwein/z_RUNS/marker_attribution/CODEX_cHL_CIM_Funnel \
    --annotation_map "Cytotoxic CD8:CD8,TReg:Treg" \
    --ignore         "Seg Artifact,Unidentified,Other" \
    --n_steps        20 \
    --batch_size     64

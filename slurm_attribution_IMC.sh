#!/bin/bash
#SBATCH --job-name=attribution_IMC
#SBATCH --output=/home/sgutwein/logs/attribution_IMC_%j.log
#SBATCH --error=/home/sgutwein/logs/attribution_IMC_%j.log
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

BASE=/nobackup/lab_taschner-mandl/simongutwein

python /home/sgutwein/src/MCA/tools/marker_attribution.py \
    --model_dir  $BASE/z_RUNS/paper_clean/IMC_NB_TumorSub/CIM_Funnel_Large_fulldata \
    --h5         $BASE/h5_files/IMC_NB_TumorSub/IMC_NB_TumorSub.h5 \
    --markers    $BASE/h5_files/IMC_NB_TumorSub/used_markers.txt \
    --val        $BASE/h5_files/IMC_NB_TumorSub/all_cells.txt \
    --out        $BASE/z_RUNS/marker_attribution/IMC_NB_CIM_Funnel_fulldata \
    --ignore     "Other" \
    --patch_size 24 \
    --n_steps    20 \
    --batch_size 64 \
    --n_workers  16

echo "Done: $(date)"

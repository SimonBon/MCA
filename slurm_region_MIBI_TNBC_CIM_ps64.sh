#!/bin/bash
#SBATCH --job-name=region_TNBC_CIM
#SBATCH --output=/home/sgutwein/logs/region_TNBC_CIM_%j.log
#SBATCH --error=/home/sgutwein/logs/region_TNBC_CIM_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l4_gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310

BASE=/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/paper
H5=/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/MIBI_TNBC.h5
MARKERS=/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/used_markers.txt
PATIENT_CLASSES=/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/patient_class.csv
OUT=/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/region_analysis/MIBI_TNBC_CIM_ps64
EMB=/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/region_analysis/embeddings/MIBI_TNBC_CIM_ps64

mkdir -p "$OUT"

python /home/sgutwein/src/MCA/tools/region_analysis.py \
    --model_dir      "$BASE/MIBI_TNBC/CIM" \
    --h5             "$H5" \
    --markers        "$MARKERS" \
    --out            "$OUT" \
    --emb_dir        "$EMB" \
    --patch_size     64 \
    --k              500 \
    --batch_size     64 \
    --n_jobs         8 \
    --n_show         4 \
    --patient_classes "$PATIENT_CLASSES" \
    --display_markers Pan-Keratin CD3 CD68 Vimentin dsDNA PD-L1

#!/bin/bash
#SBATCH --job-name=cluster_attr_embUMAP
#SBATCH --output=/home/sgutwein/logs/cluster_attr_embUMAP_%j.log
#SBATCH --error=/home/sgutwein/logs/cluster_attr_embUMAP_%j.log
#SBATCH --partition=shortq
#SBATCH --qos=shortq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310

ATTR=/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/marker_attribution/CODEX_cHL_CIM_Funnel/attribution.npz
UMAP_EMB=/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/paper_clean/CODEX_cHL/CIM_Funnel_Large/umap_embeddings.npz
OUT=/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/marker_attribution/CODEX_cHL_CIM_Funnel/clusters_embUMAP

python /home/sgutwein/src/MCA/tools/cluster_attribution.py \
    --attribution    "$ATTR" \
    --umap_embeddings "$UMAP_EMB" \
    --ignore         "Seg Artifact,Unidentified,Other" \
    --annotation_map "Cytotoxic CD8:CD8,TReg:Treg" \
    --out            "$OUT" \
    --resolution     0.5

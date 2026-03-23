#!/bin/bash
#SBATCH --job-name=attr_modscore_K18
#SBATCH --output=/home/sgutwein/logs/attr_modscore_K18_%j.log
#SBATCH --error=/home/sgutwein/logs/attr_modscore_K18_%j.log
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

BASE=/nobackup/lab_taschner-mandl/simongutwein

ATTR_OUT=$BASE/z_RUNS/paper_clean/CODEX_cHL_KRONOS18/CIM_Funnel_Large/attribution
MOD_OUT=$BASE/z_RUNS/paper_clean/CODEX_cHL_KRONOS18/CIM_Funnel_Large/module_scoring
UMAP_EMB=$BASE/z_RUNS/paper_clean/CODEX_cHL_KRONOS18/CIM_Funnel_Large/umap_embeddings.npz

echo "=== Attribution + Module scoring: CODEX_cHL KRONOS18 (18 markers) ==="
echo "Attribution out: $ATTR_OUT"
echo "Module score out: $MOD_OUT"
echo "Date: $(date)"

# Step 1: Integrated Gradients attribution
python /home/sgutwein/src/MCA/tools/marker_attribution.py \
    --model_dir      $BASE/z_RUNS/paper_clean/CODEX_cHL_KRONOS18/CIM_Funnel_Large \
    --h5             $BASE/h5_files/CODEX_cHL/CODEX_cHL.h5 \
    --markers        $BASE/h5_files/CODEX_cHL/used_markers_KRONOS18.txt \
    --val            $BASE/h5_files/CODEX_cHL/test.txt \
    --out            "$ATTR_OUT" \
    --annotation_map "Cytotoxic CD8:CD8,TReg:Treg" \
    --ignore         "Other,Seg Artifact" \
    --n_steps        20 \
    --batch_size     64

echo "Attribution done: $(date)"

# Step 2: Module scoring with KRONOS18 panel-adapted modules
# Pass umap_emb as fallback since cluster's marker_attribution.py does not embed UMAP
python /home/sgutwein/src/MCA/tools/module_score_attribution.py \
    --attribution "$ATTR_OUT/attribution.npz" \
    --umap_emb    "$UMAP_EMB" \
    --out         "$MOD_OUT" \
    --kronos18

echo "Module scoring done: $(date)"

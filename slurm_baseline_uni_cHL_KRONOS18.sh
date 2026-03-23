#!/bin/bash
#SBATCH --job-name=uni_K18
#SBATCH --output=/home/sgutwein/logs/uni_K18_%j.log
#SBATCH --error=/home/sgutwein/logs/uni_K18_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l4_gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310
export PYTHONPATH=/home/sgutwein/src:$PYTHONPATH

BASE=/nobackup/lab_taschner-mandl/simongutwein
UNI_DIR=$BASE/model_assets/UNI

# ── Download UNI weights if not already present ──────────────────────────────
if [ ! -f "$UNI_DIR/pytorch_model.bin" ]; then
    echo "Downloading UNI weights from MahmoodLab/UNI ..."
    mkdir -p $UNI_DIR
    python3 -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='MahmoodLab/UNI', filename='pytorch_model.bin',
                    local_dir='$UNI_DIR')
print('Downloaded to', p)
"
fi
echo "UNI weights: $UNI_DIR/pytorch_model.bin"

# ── Feature extraction + evaluation ──────────────────────────────────────────
# img_size=64: 64×64 patches, multiple of 16 (ViT-L/16 patch stride)
# → 4×4=16 spatial tokens per marker, comparable to KRONOS pretraining resolution
# 18 markers → output dim 18×1024=18432
# batch_size=32: 32 cells × 18 markers = 576 ViT-L forward passes per batch

python -u /home/sgutwein/src/MCA/tools/extract_external_features.py \
    --model          uni \
    --uni_ckpt       $UNI_DIR/pytorch_model.bin \
    --h5             $BASE/h5_files/CODEX_cHL/CODEX_cHL.h5 \
    --markers        $BASE/h5_files/CODEX_cHL/used_markers_KRONOS18.txt \
    --train_idx      $BASE/h5_files/CODEX_cHL/train.txt \
    --val_idx        $BASE/h5_files/CODEX_cHL/test.txt \
    --patch_size     64 \
    --img_size       64 \
    --out            $BASE/z_RUNS/paper_clean/CODEX_cHL_KRONOS18/UNI \
    --annotation_map "Cytotoxic CD8:CD8,TReg:Treg" \
    --ignore         "Seg Artifact" \
    --batch_size     32 \
    --num_workers    2 \
    --n_jobs         8 \
    --skip_label_efficiency

echo "Done: $(date)"

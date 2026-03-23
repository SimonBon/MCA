#!/bin/bash
#SBATCH --job-name=uni_cHL
#SBATCH --output=/home/sgutwein/logs/uni_cHL_%j.log
#SBATCH --error=/home/sgutwein/logs/uni_cHL_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l4_gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310
export PYTHONPATH=/home/sgutwein/src:$PYTHONPATH

BASE=/nobackup/lab_taschner-mandl/simongutwein
UNI_DIR=$BASE/model_assets/UNI

# Download UNI weights if not present
if [ ! -f "$UNI_DIR/pytorch_model.bin" ]; then
    echo "Downloading UNI weights..."
    mkdir -p $UNI_DIR
    python3 -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='MahmoodLab/UNI', filename='pytorch_model.bin',
                    local_dir='$UNI_DIR')
print('Downloaded to', p)
"
fi
echo "UNI weights: $UNI_DIR/pytorch_model.bin"

# CODEX_cHL full panel: 41 markers × 1024 = 41984 dims per cell
# annotation_map: "Cytotoxic CD8" → "CD8", "TReg" → "Treg"  (matches CIM eval)
# ignore: "Seg Artifact" only (Other is kept → 16 classes)
# lp_subsample 50000: LP fitting capped to avoid hours of lbfgs on 41984 dims
# batch_size 16: 16 cells × 41 markers = 656 ViT-L forward passes per step

python -u /home/sgutwein/src/MCA/tools/extract_external_features.py \
    --model          uni \
    --uni_ckpt       $UNI_DIR/pytorch_model.bin \
    --h5             $BASE/h5_files/CODEX_cHL/CODEX_cHL.h5 \
    --markers        $BASE/h5_files/CODEX_cHL/used_markers.txt \
    --train_idx      $BASE/h5_files/CODEX_cHL/train.txt \
    --val_idx        $BASE/h5_files/CODEX_cHL/test.txt \
    --patch_size     64 \
    --img_size       64 \
    --out            $BASE/z_RUNS/paper_clean/CODEX_cHL/UNI \
    --annotation_map "Cytotoxic CD8:CD8,TReg:Treg" \
    --ignore         "Seg Artifact" \
    --batch_size     16 \
    --num_workers    2 \
    --n_jobs         16 \
    --lp_subsample   50000 \
    --skip_label_efficiency

echo "Done: $(date)"

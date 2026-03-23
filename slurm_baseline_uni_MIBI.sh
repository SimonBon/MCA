#!/bin/bash
#SBATCH --job-name=uni_MIBI
#SBATCH --output=/home/sgutwein/logs/uni_MIBI_%A_%a.log
#SBATCH --error=/home/sgutwein/logs/uni_MIBI_%A_%a.log
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l4_gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=16:00:00
#SBATCH --array=0-4

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310
export PYTHONPATH=/home/sgutwein/src:$PYTHONPATH

BASE=/nobackup/lab_taschner-mandl/simongutwein
UNI_DIR=$BASE/model_assets/UNI
FOLD=${SLURM_ARRAY_TASK_ID}

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

# MIBI_TNBC: 37 markers × 1024 = 37888 dims per cell
# CV splits: cv_splits_paper (patient-level splits, same as CIM)
# ignore: "Unidentified" only (Other immune is a real class → 16 classes)
# no annotation_map needed: raw class names match CIM exactly
# lp_subsample 50000: cap LP training set (full fold ~157k train cells)
# batch_size 16: 16 × 37 = 592 ViT-L forward passes per step
# 192G: ~157k×37888×4 ≈ 24GB train feats + clustering overhead

python -u /home/sgutwein/src/MCA/tools/extract_external_features.py \
    --model          uni \
    --uni_ckpt       $UNI_DIR/pytorch_model.bin \
    --h5             $BASE/h5_files/MIBI_TNBC/MIBI_TNBC.h5 \
    --markers        $BASE/h5_files/MIBI_TNBC/used_markers.txt \
    --train_idx      $BASE/h5_files/MIBI_TNBC/cv_splits_paper/split_${FOLD}/train.txt \
    --val_idx        $BASE/h5_files/MIBI_TNBC/cv_splits_paper/split_${FOLD}/test.txt \
    --patch_size     64 \
    --img_size       64 \
    --out            $BASE/z_RUNS/paper_clean/MIBI_TNBC/UNI/fold_${FOLD} \
    --ignore         "Unidentified" \
    --batch_size     16 \
    --num_workers    2 \
    --n_jobs         16 \
    --lp_subsample   50000 \
    --skip_label_efficiency

echo "Done fold ${FOLD}: $(date)"

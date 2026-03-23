#!/bin/bash
#SBATCH --job-name=uni_IMC
#SBATCH --output=/home/sgutwein/logs/uni_IMC_%A_%a.log
#SBATCH --error=/home/sgutwein/logs/uni_IMC_%A_%a.log
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

# IMC_NB_TumorSub: 31 markers × 1024 = 31744 dims per cell
# CV splits: cv_splits (matches CIM configs)
# ignore: "Other" only — no "Seg Artifact" class exists in IMC annotations → 19 classes
# no annotation_map needed: raw class names match CIM exactly
# lp_subsample 50000: cap LP training set (full fold ~192k train cells)
# batch_size 16: 16 × 31 = 496 ViT-L forward passes per step
# 192G: ~192k×31744×4 ≈ 24GB train feats + clustering overhead

python -u /home/sgutwein/src/MCA/tools/extract_external_features.py \
    --model          uni \
    --uni_ckpt       $UNI_DIR/pytorch_model.bin \
    --h5             $BASE/h5_files/IMC_NB_TumorSub/IMC_NB_TumorSub.h5 \
    --markers        $BASE/h5_files/IMC_NB_TumorSub/used_markers.txt \
    --train_idx      $BASE/h5_files/IMC_NB_TumorSub/cv_splits/split_${FOLD}/train.txt \
    --val_idx        $BASE/h5_files/IMC_NB_TumorSub/cv_splits/split_${FOLD}/test.txt \
    --patch_size     64 \
    --img_size       64 \
    --out            $BASE/z_RUNS/paper_clean/IMC_NB_TumorSub/UNI/fold_${FOLD} \
    --ignore         "Other" \
    --batch_size     16 \
    --num_workers    2 \
    --n_jobs         16 \
    --lp_subsample   50000 \
    --skip_label_efficiency

echo "Done fold ${FOLD}: $(date)"

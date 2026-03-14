#!/bin/bash
#SBATCH --job-name=FunnelLarge_all
#SBATCH --output=/home/sgutwein/logs/FunnelLarge_all_%j.log
#SBATCH --error=/home/sgutwein/logs/FunnelLarge_all_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l4_gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=20:00:00

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310

export PYTHONPATH=/home/sgutwein/src:$PYTHONPATH

TRAIN=/home/sgutwein/src/mmselfsup/tools/train.py
CFGBASE=/home/sgutwein/src/MCA/configs/_experiments_

run() {
    local cfg=$1
    echo ""; echo "=== $cfg ==="; echo "Date: $(date)"
    CUDA_VISIBLE_DEVICES=0 python $TRAIN $cfg
    echo "Done: $(date)"
}

run $CFGBASE/CODEX_cHL/CIM_VICReg_Funnel_Large.py
run $CFGBASE/CODEX_cHL/CIM_VICReg_Funnel_Large_Norm.py
run $CFGBASE/CODEX_DLBCL/CIM_VICReg_Funnel_Large.py
run $CFGBASE/CODEX_DLBCL/CIM_VICReg_Funnel_Large_Norm.py
run $CFGBASE/IMC_NB/CIM_VICReg_Funnel_Large.py
run $CFGBASE/IMC_NB/CIM_VICReg_Funnel_Large_Norm.py
run $CFGBASE/IMC_NB_FineCT/CIM_VICReg_Funnel_Large.py
run $CFGBASE/IMC_NB_FineCT/CIM_VICReg_Funnel_Large_Norm.py

echo ""; echo "All done: $(date)"

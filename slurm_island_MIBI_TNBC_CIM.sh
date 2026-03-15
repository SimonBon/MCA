#!/bin/bash
#SBATCH --job-name=island_TNBC
#SBATCH --output=/home/sgutwein/logs/island_TNBC_%j.log
#SBATCH --error=/home/sgutwein/logs/island_TNBC_%j.log
#SBATCH --partition=shortq
#SBATCH --qos=shortq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310

export PYTHONPATH=/home/sgutwein/src:$PYTHONPATH

MODEL=MIBI_TNBC_CIM_VICReg_Funnel_Large

echo "=== island_analysis ==="
python /home/sgutwein/src/MCA/tools/island_analysis.py --model $MODEL

echo "=== island_figure ==="
python /home/sgutwein/src/MCA/tools/island_figure.py --model $MODEL

echo "=== sample_batch_effect (sample 41) ==="
python /home/sgutwein/src/MCA/tools/sample_batch_effect.py --model $MODEL --sample 41

echo "Done: $(date)"

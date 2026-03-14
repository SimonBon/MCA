#!/bin/bash
#SBATCH --job-name=ExprBaseline_all
#SBATCH --output=/home/sgutwein/logs/ExprBaseline_all_%j.log
#SBATCH --error=/home/sgutwein/logs/ExprBaseline_all_%j.log
#SBATCH --partition=shortq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00

source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh
conda activate mca310

export PYTHONPATH=/home/sgutwein/src:$PYTHONPATH

BASE=/nobackup/lab_taschner-mandl/simongutwein
SCRIPT=/home/sgutwein/src/MCA/tools/baseline_expression.py
OUT=$BASE/z_RUNS

run() {
    local name=$1; local data=$2; local patch=$3
    echo ""; echo "=== $name (mean) ==="; echo "Date: $(date)"
    python $SCRIPT --h5 $data/${name}.h5 --markers $data/used_markers.txt \
        --train $data/train.txt --val $data/val.txt \
        --out $OUT/${name}_ExprBaseline_mean --feat mean --patch_size $patch --n_jobs 16
    echo ""; echo "=== $name (mean+std) ==="; echo "Date: $(date)"
    python $SCRIPT --h5 $data/${name}.h5 --markers $data/used_markers.txt \
        --train $data/train.txt --val $data/val.txt \
        --out $OUT/${name}_ExprBaseline_mean_std --feat mean+std --patch_size $patch --n_jobs 16
}

run CODEX_cHL                   $BASE/h5_files/CODEX_cHL                    32
run CODEX_DLBCL2                $BASE/h5_files/CODEX_DLBCL2                 24
run IMC_NeuroblastomaMetaCluster $BASE/h5_files/IMC_NeuroblastomaMetaCluster 24
run IMC_NB_FineCT               $BASE/h5_files/IMC_NB_FineCT                24

echo ""; echo "All done: $(date)"

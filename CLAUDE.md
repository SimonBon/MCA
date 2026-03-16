# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Self-supervised cell representation learning for multiplexed imaging (CODEX, IMC, MIBI-TOF). Cells are represented as multi-channel image patches (one channel per protein marker). Models learn embeddings via VICReg that are useful for cell-type classification, clustering, and spatial analysis.

## Cluster Access

Always connect with:
```bash
ssh cemm
```
Never use `ssh login.int.cemm.at` — it requires a password and will be denied.

### Key Paths on the Cluster

| Purpose | Path |
|---|---|
| Repo | `/home/sgutwein/src/MCA` |
| mmselfsup train script | `/home/sgutwein/src/mmselfsup/tools/train.py` |
| Conda env | `/nobackup/lab_taschner-mandl/simongutwein/miniconda3/envs/mca310` |
| H5 data files | `/nobackup/lab_taschner-mandl/simongutwein/h5_files/<DATASET>/` |
| Paper results | `/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/paper_clean/` |
| Ablation results | `/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/ablations/CIM_Funnel/` |
| SLURM logs | `/home/sgutwein/logs/` |

### SLURM Job Queues

**Every partition requires `--qos=<partition-name>`** — the QOS name always matches the partition name exactly.

**GPU jobs:**
```bash
sbatch --partition=gpu --qos=gpu --gres=gpu:l4_gpu:1 --ntasks=1 \
  --cpus-per-task=16 --mem=64G --time=02:00:00 \
  --output=/home/sgutwein/logs/<name>_%j.log ...
```

**CPU jobs** (no `--gres`):
```bash
sbatch --partition=shortq --qos=shortq --ntasks=1 --cpus-per-task=16 --mem=32G --time=02:00:00 \
  --output=/home/sgutwein/logs/<name>_%j.log ...
```

Queue reference:

| Queue | Time limit | Group limit | Nodes | Use for |
|---|---|---|---|---|
| `tinyq` | 2h | 400 | 21 | quick CPU jobs |
| `shortq` | 12h | 200 | 20 | standard CPU jobs (baseline, analysis) |
| `mediumq` | 2d | 50 | 13 | longer CPU jobs |
| `longq` | 30d | 30 | 8 | very long CPU jobs |
| `gpu` | 3d | — | 18 | GPU training (add `--gres=gpu:l4_gpu:1`) |

For inline `--wrap` commands, avoid shell quoting issues by writing a temp script file and submitting that instead of `--wrap`.

## Training

Training uses the external **mmselfsup** train script — it is not inside this repo:

**On the CeMM cluster (SLURM):**
```bash
# Single config
sbatch --wrap="source /nobackup/lab_taschner-mandl/simongutwein/miniconda3/etc/profile.d/conda.sh && \
  conda activate mca310 && \
  export PYTHONPATH=/home/sgutwein/src:$PYTHONPATH && \
  python /home/sgutwein/src/mmselfsup/tools/train.py configs/_experiments_/paper/CODEX_cHL/CIM_VICReg.py" \
  --partition=gpu --qos=gpu --gres=gpu:l4_gpu:1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=02:00:00 \
  --output=/home/sgutwein/logs/<name>_%j.log

# Resume from checkpoint
python /home/sgutwein/src/mmselfsup/tools/train.py <config.py> --resume auto
```

**Locally (for debugging, DEBUG=1 runs 5 iters):**
```bash
DEBUG=1 CUDA_VISIBLE_DEVICES=0 python /path/to/mmselfsup/tools/train.py configs/...py
```

## Config System

Configs use MMEngine's `_base_` inheritance. Each experiment config composes:

```python
_base_ = [
    '../../../_base_/default.py',       # mmengine hooks, logging, imports
    '../../../_base_/train_cfg.py',     # AdamW 1k iters (200 warmup + 800 cosine); overridden for Funnel
    '../../../_base_/val_cfg.py',       # EvaluateModelRich hook defaults
    '../../../_datasets_/CODEX_cHL.py', # n_markers, patch_size, cutter_size, dataset class
    '../../../_backbones_/CIM.py',      # backbone dict
    '../../../_algorithms_/VICReg.py',  # MVVICReg model + NonLinearNeck
]
```

After importing, the experiment config wires everything together: sets absolute paths, builds `train_dataset`, configures the eval hook, and sets `work_dir`.

**Key variables set by dataset configs:**
- `n_markers` — number of protein channels
- `patch_size` — spatial input size (e.g., 32)
- `cutter_size` — eval crop size (smaller than patch_size)
- `features_per_marker` / `mix_channels` — neck `in_channels`

**Standard models use AdamW** (1000 iters, default `train_cfg.py`). **CIM_Funnel_Large uses LARS** (16 000 iters, inline scheduler in config).

**To generate all paper experiment configs:**
```bash
python tools/generate_paper_configs.py
```

**To generate ablation configs:**
```bash
python tools/generate_ablation_configs.py
```

## Result Collection

```bash
# On cluster — collect all metrics.json into results.csv
python tools/collect_paper_results.py

# Generate Excel with per-dataset sheets + per-class AP
python tools/make_paper_excel.py
```

Results land in `/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/paper_clean/<DATASET>/<MODEL>/metrics.json`.
Ablations land in `/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/ablations/CIM_Funnel/<variant>/metrics.json`.

## Architecture

### Model Hierarchy

All backbones take `[B, C, H, W]` (C = n_markers) and return `([B, feature_dim, 1, 1],)`.

**CIM (`WideModel`)** — channel-independent throughout:
- Stem: grouped conv `[B, C, H, W] → [B, C×stem_width, H, W]` (groups=C)
- Stages: `ConvBlock` (depthwise, ConvNeXt-style) + `AvgPool2d` between stages
- Output dim: `n_markers × stem_width` (e.g., 41 × 32 = 1312)
- Config: `stem_width=32, block_width=2, sep_layer_config=[1,1]`

**CIM_Funnel_Large (`CIM_Funnel`)** — two-phase mixing:
- Phase 1: depthwise CIM stages (channel-independent)
- Phase 2: `MixBlock` ConvNeXt blocks with `groups=1` (full cross-channel mixing)
- Output dim: `mix_channels` (default 512)
- Uses LARS + 16k iters; eval hook fires `after_train`

**EarlyFusionModel** — standard grouped=1 convs from stem; mixes all channels immediately.

**WideModelProgressiveFusion (CIM_ProgFusion)** — dual branch: CIM + fusion stream with injectors.

**ResNetBaseline** — standard ResNet, output dim=256.

**WideModelAttentionGated (CIMATT_Gate)** — CIM + marker gating via relative expression + self-attention.

### VICReg Loss (`src/VICReg.py`)

`MVVICReg` takes two views, computes sim (invariance) + std (variance) + cov (covariance) terms. Coefficients: `sim=25, std=25, cov=1`. No negatives needed. Optional `SampleCentroidBank` for cross-sample alignment.

### Evaluation Hook (`src/val_hook_rich.py`)

`EvaluateModelRich` runs `after_train` and computes:
- Linear probe (balanced acc, F1, mAP, per-class AP)
- k-NN (k=15, balanced acc)
- k-Means clustering (NMI, ARI)
- cLISI / iLISI (sample mixing)
- UMAP → `umap.pdf`, `umap_sample.pdf`

The `epochs` parameter controls LP solver `max_iter`, **not** the evaluation interval — the hook always fires once after training ends.

### Dataset (`src/dataset.py`)

`MCIDataset` reads from HDF5. HDF5 structure: `coords/DIM1, DIM2, sample_id`, `annotation`, `marker_names`, per-sample `image [H, W, n_markers]`. Worker-local HDF5 handles (lazy init) avoid multiprocessing pickling issues.

## Paper Experiments

| Dataset | Folds | Models |
|---|---|---|
| CODEX_cHL | single split | CIM, CIM_LateFusion, CIM_Funnel_Large, ResNet |
| CODEX_cHL_KRONOS18 | single split | same (18-marker panel for KRONOS comparison) |
| MIBI_TNBC | 5-fold CV | same |
| IMC_NB_TumorSub | 5-fold CV | same |

Configs: `configs/_experiments_/paper/<DATASET>/`
Results: `/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/paper_clean/<DATASET>/<MODEL>/`
Excel: `paper_results.xlsx` (synced locally; 8 sheets including `_AP` per-class sheets with KRONOS reference column)

## External / Baseline Models

### ExprBaseline
Mean marker intensity per cell (scalar per marker → `[n_markers]` feature vector). CPU-only, produced by `tools/baseline_expression.py`. Results stored in `paper_clean/<DATASET>/ExprBaseline/` (single split) or `paper_clean/<DATASET>/ExprBaseline/fold_<k>/` (CV). Loaded into Excel via `EXTERNAL_MODELS` dict in `make_paper_excel.py`.

### DINOv2 / UNI / CA-MAE (KRONOS paper description)
The KRONOS paper (arXiv:2506.03373) uses these external models as follows — **note: our implementation differs** (see below):

**DINOv2 (ViT-L/14)** and **UNI (ViT-L/16)**: Each marker channel is individually replicated to 3×RGB and passed through the model. The CLS token is extracted per marker and concatenated → feature vector of size `1024 × M`. Images are center-cropped to multiples of the patch token size (14px). Marker channels normalised with marker-specific mean/std from SPM-47M dataset.

**CA-MAE**: Channel-agnostic masked autoencoder pretrained on fluorescence cell profiling (RxRx360 + JUMP-CP). Accepts arbitrary channel counts natively. Marker-wise embeddings of size `384 × M`.

**Our implementation** (`src/models_external.py`) differs: we use a single-channel `patch_embed.proj` (Conv2d 1→D), process all markers as a joint token sequence with shared positional embeddings tiled C times, and return a **single CLS token** rather than per-marker CLS + concat. This allows cross-marker attention in the ViT but produces lower-dimensional features (`D` vs `1024×M`). Results were poor; models removed from paper Excel pending reassessment of implementation.

## Ablation Experiments

Configs: `configs/_experiments_/ablations/CIM_Funnel/`
Results: `/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/ablations/CIM_Funnel/<variant>/`

Groups:
- `aug_*` — channel augmentation ablations (no_channel, no_drop, no_shift, no_noise, drop_prob sweep)
- `iters_*` — training length (2k, 4k; 8k = `aug_full_8k`; 16k = paper baseline)
- `cap_*` — model capacity (mix_n_blocks 4/8/12; mix_channels 256/512/768)

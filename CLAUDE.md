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

### KRONOS (foundation model — completed)

KRONOS (arXiv:2506.03373, `hf_hub:MahmoodLab/kronos`) is a ViT-S/16 pretrained on SPM-47M multiplexed imaging data. It is **natively multi-channel** and the correct external baseline to compare against.

**Repo:** `/home/sgutwein/src/KRONOS` (also locally at `/Users/simon.gutwein/src/KRONOS`)
**Script:** `tools/baseline_kronos.py`
**Smoke test:** `tools/test_kronos_smoke.py`
**Marker metadata:** downloaded from HuggingFace → `model_assets/kronos/.../marker_metadata.csv` (177 markers with pretrained IDs, mean, std)

**Architecture:**
- Input: `[B, C, H, W]` — any number of markers, H = W must be divisible by 16
- PatchEmbed: processes each marker channel independently (Conv2d with `in_chans=1`), concatenates → `[B, C×N_patches, D]`
- Sinusoidal marker embeddings added by `marker_id` — **use real IDs from marker_metadata.csv, not sequential**
- Transformer self-attention operates over ALL tokens across ALL markers → full cross-marker interaction
- Positional embeddings bicubic-interpolated → works for any input size; **use 64×64 patches** (4×4=16 tokens/marker, close to pretraining)
- Outputs: `patch_features [B,384]` (CLS), `marker_features [B,C,384]` (per-marker), `token_features [B,C,h,w,384]`
- **Use `marker_features.flatten()` → [B, C×384] as the cell feature** — matches KRONOS tutorial for cell phenotyping
- Feature dim scales with markers: 18×384=6912 (KRONOS18), 41×384=15744 (full panel)

**Critical preprocessing (must match KRONOS tutorial exactly):**
1. Divide raw intensities by `marker_max_values` (65535 for uint16; **1.0 if data already in [0,1]**)
2. Per-marker `(x - mean) / std` — use official stats from `marker_metadata.csv` for known markers, compute from training cells for unknowns
3. Multiply by binary cell mask to zero out non-cell pixels: `patch = patch * cell_mask`
4. Pass `marker_ids` as `[B, C]` int64 tensor

**Marker ID resolution (`load_marker_meta` in baseline_kronos.py):**
- 177 markers in SPM-47M metadata; our markers are matched by name (uppercase)
- Name aliases handle mismatches: `DAPI-01→DAPI`, `Cytokeritin→CYTOKERATIN`, `HLA-DR→HLA_DR`, `PD-1→PD1`, `GranzymeB→GZMB`, etc.
- Markers not in metadata get sequential fallback IDs (4, 5, ...) and data-computed stats
- For a new dataset: check coverage with `marker_metadata.csv`, add aliases to `ALIASES` dict in `load_marker_meta` for any name mismatches

**Running on a new dataset:**
1. Check which markers are in `marker_metadata.csv`: `df[df['marker_name'].str.upper().isin([m.upper() for m in your_markers])]`
2. Add name aliases to `ALIASES` dict in `load_marker_meta` for any mismatches
3. Set `--marker_max_values 1.0` if data is already float [0,1], else `65535`
4. Set `--ignore` and `--annotation_map` to match the other models for that dataset
5. Copy `slurm_baseline_kronos_cHL.sh`, update paths/dataset, submit

**Results location:** `paper_clean/<DATASET>/KRONOS/`
**Completed runs:** CODEX_cHL (41 markers, LP=0.751) and CODEX_cHL_KRONOS18 (18 markers, LP=0.749)

### DINOv2 / UNI / CA-MAE (KRONOS paper description)
The KRONOS paper uses these external models as follows — **note: our implementation differs** (see below):

**DINOv2 (ViT-L/14)** and **UNI (ViT-L/16)**: Each marker channel is individually replicated to 3×RGB and passed through the model. The CLS token is extracted per marker and concatenated → feature vector of size `1024 × M`. Images are center-cropped to multiples of the patch token size (14px). Marker channels normalised with marker-specific mean/std from SPM-47M dataset.

**CA-MAE**: Channel-agnostic masked autoencoder pretrained on fluorescence cell profiling (RxRx360 + JUMP-CP). Accepts arbitrary channel counts natively. Marker-wise embeddings of size `384 × M`.

**Our implementation** (`src/models_external.py`) now matches the paper: original 3-channel `patch_embed.proj` is kept unchanged; each marker channel is replicated to 3×RGB and passed independently through the frozen ViT; per-marker CLS tokens are concatenated → `[B, D*C, 1, 1]`. No cross-marker interaction. Feature dim scales with C: 1024×M for DINOv2-L/UNI. Use `tools/extract_external_features.py` to run evaluation.

## Marker Attribution & Module Scoring

Integrated Gradients (IG) attribution + UCell-style module scoring for biological validation.
Completed for CODEX_cHL (41 markers) and CODEX_cHL_KRONOS18 (18 markers), both using CIM_Funnel_Large.

**Scripts:**
- `tools/marker_attribution.py` — computes IG attribution per cell, saves `attribution.npz`
- `tools/module_score_attribution.py` — scores cell type modules, assigns labels, plots UMAPs

**Attribution outputs:**
- Full panel: `z_RUNS/marker_attribution/CODEX_cHL_CIM_Funnel/attribution.npz`
- KRONOS18:   `z_RUNS/marker_attribution/CODEX_cHL_KRONOS18_CIM_Funnel_Large/attribution.npz`

**Clean module scoring outputs (paper-ready):**
- `paper_clean/CODEX_cHL/CIM_Funnel_Large/module_scoring/`
- `paper_clean/CODEX_cHL_KRONOS18/CIM_Funnel_Large/module_scoring/`

Each folder: `attribution.npz`, `module_scores.npz`, `modules.csv`, `umap_marker_attribution.png`, `umap_module_scores.png`, `umap_gt_vs_assigned.png`, `umap_assigned_celltypes.png`

**Key flags for module_score_attribution.py:**
- `--data_driven` → DATA_DRIVEN_MODULES (41-marker panel, biologically validated)
- `--kronos18`    → KRONOS18_MODULES (18-marker panel)
- `--umap_emb`    → fallback UMAP coords if attribution.npz lacks `umap_coords`

**Important:** `find_latest_timestamp_folder` in `src/utils.py` picks the most recently modified subdir of `model_dir`. Never put attribution/module_scoring output dirs inside the model dir — they will break `load_checkpoint`.

## Ablation Experiments

Configs: `configs/_experiments_/ablations/CIM_Funnel/`
Results: `/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/ablations/CIM_Funnel/<variant>/`

Groups:
- `aug_*` — channel augmentation ablations (no_channel, no_drop, no_shift, no_noise, drop_prob sweep)
- `iters_*` — training length (2k, 4k; 8k = `aug_full_8k`; 16k = paper baseline)
- `cap_*` — model capacity (mix_n_blocks 4/8/12; mix_channels 256/512/768)

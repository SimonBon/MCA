# Paper Experiment Runs

**Date started:** 2026-03-16
**Cluster:** CeMM SLURM (login.int.cemm.at), L4 GPU nodes
**Code repo:** `/home/sgutwein/src/MCA` (GitHub: SimonBon/MCA)
**Output root:** `/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/paper/`

---

## Goal

Fresh, coherent training runs for the paper. Clean ablation of **when cross-channel mixing is introduced**:

| Model | First cross-channel op |
|---|---|
| CIM | Never — fully channel-independent |
| CIM_LateFusion | After global average pooling (MLP mixer) |
| CIM_Funnel_Large | Phase 2 MixBlocks (after sep_stages + transition) |
| ResNet | From the very first convolution (baseline) |

---

## Datasets

### CODEX_cHL
- **H5:** `/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/CODEX_cHL.h5`
- **Markers file:** `.../CODEX_cHL/used_markers.txt`
- **n_markers:** 41, **patch_size:** 32, **cutter_size:** 24
- **Split:** single train/test (existing)
  - train: `.../CODEX_cHL/train.txt`
  - test:  `.../CODEX_cHL/test.txt`
- **ignore_annotation:** `['Seg Artifact']`
- **annotation_map:** `{'Cytotoxic CD8': 'CD8', 'TReg': 'Treg'}` → 16 eval classes (matching KRONOS paper)
- **Config:** `configs/_datasets_/CODEX_cHL.py`

### MIBI_TNBC
- **H5:** `/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/MIBI_TNBC.h5`
- **Markers file:** `.../MIBI_TNBC/used_markers.txt`
- **n_markers:** 37, **patch_size:** 32, **cutter_size:** 20
- **Patients:** 40 total → 32 train / 8 test per fold
- **CV splits:** 5-fold patient-level, generated with `tools/generate_cv_splits.py --seed 42`
  - Location: `.../MIBI_TNBC/cv_splits_paper/split_{0-4}/train.txt` + `test.txt`
  - Approximate cells per fold: ~157–160k train / ~37–43k test
- **ignore_annotation:** `['Unidentified']` → 16 eval classes
- **Config:** `configs/_datasets_/MIBI_TNBC.py`

### IMC_NB_TumorSub
- **H5:** `/nobackup/lab_taschner-mandl/simongutwein/h5_files/IMC_NB_TumorSub/IMC_NB_TumorSub.h5`
- **Markers file:** `.../IMC_NB_TumorSub/used_markers.txt`
- **n_markers:** 31, **patch_size:** 24, **cutter_size:** 12
- **Patients:** 25 total → 20 train / 5 test per fold (patient IDs extracted via `sample_id.split("-TU-")[0]`)
- **CV splits:** 5-fold patient-level, generated with `tools/generate_cv_splits.py --seed 42 --patient_sep="-TU-"`
  - Location: `.../IMC_NB_TumorSub/cv_splits/split_{0-4}/train.txt` + `test.txt`
  - Approximate cells per fold: ~222–252k train / ~48–78k test
- **ignore_annotation:** `['Other']` → 17 eval classes (NK_DC retained)
- **Config:** `configs/_datasets_/IMC_NB_TumorSub.py`

---

## Models

### CIM (`WideModel`, `late_fusion=False`)
- **Backbone config:** `configs/_backbones_/CIM.py`
- **Architecture:** depthwise grouped stem → 2×[1 depthwise block] → global avgpool
  - `stem_width=32`, `block_width=2`, `layer_config=[1,1]`, `drop_prob=0.05`
  - Zero cross-channel mixing at any point
- **neck.in_channels:** `n_markers × 32`  (e.g. 37×32=1184 for MIBI_TNBC)
- **Training:** AdamW, `lr=3e-4`, `wd=0.05`, 200 linear warmup + 800 cosine = **1000 iters**, `batch_size=256`
- **GPU memory:** ~11 GB on L4

### CIM_LateFusion (`WideModelLateFusion`)
- **Backbone config:** `configs/_backbones_/CIM_LateFusion_MLP.py`
- **Architecture:** identical sep_stages to CIM → global avgpool → **2-layer MLP mixer**
  - MLP: `Linear(C·D → 4·C·D) → BN → ReLU → Linear(4·C·D → C·D) → BN`
  - `mlp_ratio=4`, `out_channels=None` (defaults to `n_markers × stem_width`)
  - First and only cross-channel operation happens after pooling
- **neck.in_channels:** `n_markers × 32` (same as CIM)
- **Training:** same as CIM — AdamW 1000 iters, `batch_size=256`
- **GPU memory:** ~11 GB on L4

### CIM_Funnel_Large (`CIM_Funnel`)
- **Backbone config:** `configs/_backbones_/CIM_Funnel_Large.py`
- **Architecture:** sep_stages (phase 1) → 1×1 transition conv → 8 MixBlocks at 512ch (phase 2)
  - `stem_width=16`, `block_width=4`, `sep_layer_config=[2,2]`
  - `mix_n_blocks=8`, `mix_channels=512`
  - MixBlock = depthwise 3×3 spatial + pointwise FFN (groups=1)
- **neck.in_channels:** 512 (fixed, from `mix_channels`)
- **Training:** LARS, `lr=0.3`, `momentum=0.9`, `wd=1e-5`, 800 linear warmup + 15200 cosine = **16000 iters**, `batch_size=128`
- **Augmentations:** inlined per dataset (cutter_size strong / cutter_size−2 weak+val)
- **GPU memory:** TBD (first Funnel job running to verify)

### ResNet (`ResNetBaseline`)
- **Backbone config:** `configs/_backbones_/ResNet.py`
- **Architecture:** `base_width=64`, output dim=256 (fixed, independent of n_markers)
- **neck.in_channels:** 256
- **Training:** same as CIM — AdamW 1000 iters, `batch_size=256`
- **GPU memory:** ~175 MB on L4 (very lightweight)

---

## Training Setup

### Algorithm: VICReg
- **Config:** `configs/_algorithms_/VICReg.py`
- **Loss weights:** `sim=25`, `std=25`, `cov=1`
- **Two views:** strong + weak augmentation (flip, affine, channel shift/scale, noise, channel drop, center crop)

### Standard AdamW schedule (`configs/_base_/train_cfg.py`)
```
optimizer:     AdamW, lr=3e-4, weight_decay=0.05
warmup:        LinearLR 200 iters (start_factor=1e-4)
decay:         CosineAnnealingLR 800 iters
total:         1000 iters
```

### LARS schedule (CIM_Funnel_Large only, inlined in config)
```
optimizer:     LARS, lr=0.3, momentum=0.9, weight_decay=1e-5
warmup:        LinearLR 800 iters (start_factor=1e-4)
decay:         CosineAnnealingLR 15200 iters (eta_min=0.03)
total:         16000 iters
```

---

## Evaluation (`EvaluateModelRich` hook, `src/val_hook_rich.py`)

Evaluation runs at end of training on **test split** (LP train on train split, LP eval on test split).

### Metrics
| Metric | Details |
|---|---|
| **LP** (linear probe) | `LogisticRegression(max_iter=5000, verbose=1)`, balanced accuracy |
| **kNN** | k=15, cosine distance |
| **NMI / ARI** | KMeans on embeddings |
| **cLISI** | cell-type LISI, k=90 (Harmony standard), lower normalised = more compact |
| **iLISI** | sample integration LISI, k=90, higher normalised = better mixed |

### Hook parameters
```python
EvaluateModelRich(
    train_indicies = <split_k>/train.txt,   # fit LP
    val_indicies   = <split_k>/test.txt,    # eval LP + all metrics
    lisi_k         = 90,
    knn_k          = 15,
    n_jobs         = 8,
    epochs         = 2000,                  # LP max_iter (5000 for Funnel)
    annotation_map = {...}                  # CODEX_cHL only
)
```

### Output files (per run)
```
z_RUNS/paper/<DATASET>/<MODEL>[/fold_k]/
  metrics.json       — all eval metrics
  umap.pdf           — UMAP coloured by cell type
  umap_sample.pdf    — UMAP coloured by patient/sample
  loss_curve.png     — training loss
  confusion_matrix.pdf
  iter_XXXXXX/       — checkpoint(s)
```

---

## Config Files

### Generator
```
tools/generate_paper_configs.py   — generates all 44 experiment configs
```

### Experiment configs
```
configs/_experiments_/paper/
  CODEX_cHL/
    CIM_VICReg.py
    CIM_LateFusion_VICReg.py
    CIM_Funnel_Large_VICReg.py
    ResNet_VICReg.py
  MIBI_TNBC/
    {CIM,CIM_LateFusion,CIM_Funnel_Large,ResNet}_VICReg_fold{0-4}.py   (20 files)
  IMC_NB_TumorSub/
    {CIM,CIM_LateFusion,CIM_Funnel_Large,ResNet}_VICReg_fold{0-4}.py   (20 files)
```

### Key design decisions in configs
- All paths point to `/nobackup/lab_taschner-mandl/simongutwein/h5_files/`
- `val_indicies` in hook = `test_indicies` (no val split — train/test only)
- CODEX_cHL uses `dataset_kwargs` pattern; MIBI_TNBC + IMC use direct `dataset` dict
- Funnel_Large configs do NOT include `_augmentations_/high.py` or `train_cfg.py` (both inlined)

---

## Job Summary

| Dataset | Model | Folds | iters | Time limit | Jobs |
|---|---|---|---|---|---|
| CODEX_cHL | CIM | 1 | 1000 | 2h | 1 |
| CODEX_cHL | CIM_LateFusion | 1 | 1000 | 2h | 1 |
| CODEX_cHL | CIM_Funnel_Large | 1 | 16000 | 12h | 1 |
| CODEX_cHL | ResNet | 1 | 1000 | 2h | 1 |
| MIBI_TNBC | CIM | 5 | 1000 | 2h | 5 |
| MIBI_TNBC | CIM_LateFusion | 5 | 1000 | 2h | 5 |
| MIBI_TNBC | CIM_Funnel_Large | 5 | 16000 | 12h | 5 |
| MIBI_TNBC | ResNet | 5 | 1000 | 2h | 5 |
| IMC_NB_TumorSub | CIM | 5 | 1000 | 2h | 5 |
| IMC_NB_TumorSub | CIM_LateFusion | 5 | 1000 | 2h | 5 |
| IMC_NB_TumorSub | CIM_Funnel_Large | 5 | 16000 | 12h | 5 |
| IMC_NB_TumorSub | ResNet | 5 | 1000 | 2h | 5 |
| **Total** | | | | | **44** |

---

## CV Split Details

Generated with `tools/generate_cv_splits.py --seed 42`:

### MIBI_TNBC — 40 patients, 197678 cells
| Fold | Train cells | Test cells |
|---|---|---|
| 0 | 157,493 (32 patients) | 40,185 (8 patients) |
| 1 | 159,583 (32 patients) | 38,095 (8 patients) |
| 2 | 160,252 (32 patients) | 37,426 (8 patients) |
| 3 | 154,960 (32 patients) | 42,718 (8 patients) |
| 4 | 158,424 (32 patients) | 39,254 (8 patients) |

### IMC_NB_TumorSub — 25 patients, 300250 cells
| Fold | Train cells | Test cells |
|---|---|---|
| 0 | 241,725 (20 patients) | 58,525 (5 patients) |
| 1 | 222,001 (20 patients) | 78,249 (5 patients) |
| 2 | 240,893 (20 patients) | 59,357 (5 patients) |
| 3 | 244,261 (20 patients) | 55,989 (5 patients) |
| 4 | 252,120 (20 patients) | 48,130 (5 patients) |

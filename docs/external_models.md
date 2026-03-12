# External Model Baselines — Feature Extraction Pipeline

This document describes the three external pretrained models used as baselines
for comparison against the MCA (CIM) models, and explains in detail how features
are obtained from each.

---

## Overview

All three models are evaluated on the same cell-level multiplexed imaging data
(e.g. IMC_NB_TumorSub: 31 markers, ~240k cells) using a unified pipeline:

1. Extract a fixed-size image patch centred on each cell from the HDF5 file.
2. Pass the patch through the frozen pretrained model to obtain a feature vector.
3. Evaluate with linear probe (LP), kNN, clustering (NMI/ARI), silhouette, and neighbourhood purity.

None of these models was trained on multiplexed imaging data. They serve as
domain-transfer baselines.

---

## 1. Data Loading and Patch Extraction

**File:** `tools/extract_external_features.py`, class `MCIDatasetH5`

### HDF5 structure
The dataset is stored in an HDF5 file with the following layout:
```
h5f['coords']['DIM1']      — row (y) cell centroid coordinates  [N cells]
h5f['coords']['DIM2']      — col (x) cell centroid coordinates  [N cells]
h5f['coords']['sample_id'] — patient/slide ID string            [N cells]
h5f['annotation']          — cell-type label string             [N cells]
h5f['marker_names']        — all marker names                   [M]
h5f['data'][sample_id]['image']  — full-slide image             [H, W, M]
h5f['data'][sample_id]['masks']  — segmentation mask            [H, W]
```

### Patch extraction (per cell)
For each cell centroid `(r, c)` and a patch half-size `half = patch_size // 2`:

1. A `patch_size × patch_size` window is cropped from the full-slide image
   centred on `(r, c)`, using only the selected marker channels.
2. **Segmentation masking**: pixels outside the target cell are zeroed using the
   segmentation mask. The centre pixel's mask value identifies the cell ID;
   pixels with a different mask value are set to zero. This ensures the model
   sees only the target cell and not neighbouring cells.
3. **Boundary padding**: cells near image borders are zero-padded to maintain
   the fixed patch size.
4. **Normalisation**: the patch is divided by its maximum value (per-patch
   max-normalisation to `[0, 1]`).

Output per cell: float32 tensor `[C, patch_size, patch_size]`.

### Sorted DataLoader & in-memory cache
Cells are sorted by `sample_id` before batching so consecutive batches come
from the same patient slide. Each worker keeps the current slide's full image
array in memory (`_img_cache`), avoiding repeated HDF5 decompression.
The first batch for a new sample incurs a ~1.75 s decompression overhead;
all subsequent batches for the same sample are instantaneous.

---

## 2. Model Descriptions and Feature Extraction

### 2.1 DINOv2

**Architecture:** ViT-B/14 (Vision Transformer Base, patch size 14)
- 12 transformer blocks, 12 attention heads, embedding dimension D = 768
- 86M parameters
- Includes 4 register tokens in addition to the CLS token
- Pretrained by Meta with self-supervised DINO v2 on natural images (LVD-142M, ~142M images)

**Adaptation for multiplexed imaging (channel-agnostic patch embedding):**

The pretrained patch projection `Conv2d(3, 768, 14, 14)` is replaced with a
`Conv2d(1, 768, 14, 14)` initialised by averaging the 3 RGB filter weights:

```python
new_proj.weight = proj.weight.mean(dim=1, keepdim=True)  # [768, 1, 14, 14]
```

All transformer weights (attention, FFN, layer norms) are kept frozen.

**Forward pass:**

1. **Resize**: each patch is bilinearly resized to `img_size × img_size`
   (default: 56 px, a multiple of 14 → 4×4 = 16 spatial patches per channel).
2. **Per-channel instance normalisation**: each channel is independently
   zero-mean / unit-std normalised across the spatial dimensions:
   ```
   x_norm[b, c] = (x[b, c] - mean(x[b, c])) / std(x[b, c])
   ```
   This removes absolute intensity variation across patients.
3. **Batched single-channel patch embedding**:
   ```python
   x_flat = x.reshape(B*C, 1, H, W)   # treat each channel as a separate image
   p_flat = proj(x_flat)               # [B*C, D, h_p, w_p]  (single conv call)
   patch_tokens = p_flat.flatten(2).transpose(1,2).reshape(B, C*N, D)
   # result: [B, C*N_patches, D] = [B, 31*16, 768] = [B, 496, 768]
   ```
4. **Positional embeddings**: the pretrained 518×518 spatial pos_embed
   (1369 patches) is bicubically interpolated to the actual `h_p × w_p` grid
   (4×4 = 16 patches). This interpolated embedding is tiled C times:
   ```python
   spatial_pos = interp(model.pos_embed[:, 1:])  # [1, 16, D]
   spatial_pos = spatial_pos.repeat(1, C, 1)      # [1, C*16, D]
   ```
5. **CLS token + register tokens** are prepended. Total sequence length:
   `1 + 4_registers + C*16` = 501 tokens.
6. **Transformer**: all 12 blocks process the full sequence jointly, enabling
   cross-channel attention.
7. **Output**: the CLS token `[B, 768]` is taken as the cell representation.
   Reshaped to `[B, 768, 1, 1]` for compatibility with the MCA interface.

**Runtime:** ~56 px input, ~501 tokens, batch_size=128.

---

### 2.2 OpenPhenom

**Architecture:** ViT-S/16 (Vision Transformer Small, patch size 16)
- 12 transformer blocks, 6 attention heads, embedding dimension D = 384
- 22M parameters
- Natively channel-agnostic: the model was designed for multi-channel microscopy
  and uses a single `Conv2d(1, 384, 16, 16)` patch projection from the start
- Pretrained by Recursion Pharmaceuticals on Cell Painting assay images
  (6-channel fluorescence microscopy; ~60k compound treatments)
  using a Masked Autoencoder (MAE) self-supervised objective

**No adaptation needed**: the patch projection already accepts single-channel input.

**Forward pass:**

1. **Resize**: each patch is bilinearly resized to `img_size × img_size`
   (default: 64 px → 4×4 = 16 spatial patches per channel).
2. **Per-channel forward**: because OpenPhenom is designed for a fixed channel
   set (Cell Painting: 6 specific stains), it cannot directly accept arbitrary
   channel counts. Instead, **each marker is processed independently**:
   ```python
   per_channel = [model.predict(x[:, c:c+1]) for c in range(C)]  # C × [B, 384]
   feat = torch.stack(per_channel, dim=0).mean(dim=0)             # [B, 384]
   ```
   Each single-channel patch is independently embedded, and the C embeddings
   are averaged into a single cell-level feature vector.
3. **Preprocessing**: the model expects uint8 images in `[0, 255]`. The
   `[0, 1]` normalised patches are scaled by 255 before being passed in.
4. **Output**: `[B, 384, 1, 1]`.

**Note:** Unlike DINOv2 and UNI, OpenPhenom does **not** do cross-channel
attention — each channel is embedded in isolation and the results are averaged.
This is a symmetric, grouping-artifact-free treatment but loses any cross-marker
interaction within the model.

**Runtime:** 31 forward passes per batch (one per channel), each at 64 px, 16 patches.

---

### 2.3 UNI

**Architecture:** ViT-L/16 (Vision Transformer Large, patch size 16)
- 24 transformer blocks, 16 attention heads, embedding dimension D = 1024
- 307M parameters
- Pretrained by the Mahmood Lab (Harvard) on >200M H&E and IHC pathology images
  using a DINOv2 self-supervised objective
- Weights are gated on HuggingFace (`MahmoodLab/UNI`); access requires registration

**Adaptation for multiplexed imaging:**

Same strategy as DINOv2: the pretrained `Conv2d(3, 1024, 16, 16)` is replaced
with `Conv2d(1, 1024, 16, 16)` initialised from the average of the 3 RGB weights.
All transformer weights are frozen.

The model is always created at native `img_size=224` (the checkpoint's pos_embed
has shape `[1, 197, 1024]`: 1 CLS + 14×14 = 196 spatial patches). Positional
embeddings are interpolated at forward time to the actual input resolution.

**Forward pass:**

1. **Resize**: each patch is bilinearly resized to `img_size × img_size`
   (default: 32 px → 2×2 = 4 spatial patches per channel).
   Using 32 px instead of 224 px is critical: at 224 px each channel produces
   196 patches → 31 × 196 = 6,076 tokens per example, requiring ~9.6 GiB for
   a single QKV projection. At 32 px: 4 patches/channel × 31 channels = 124
   tokens, fitting easily in memory.
2. **Per-channel instance normalisation**: same as DINOv2.
3. **Batched single-channel patch embedding** (identical to DINOv2):
   ```python
   x_flat = x.reshape(B*C, 1, H, W)
   p_flat = proj(x_flat)               # [B*C, D, 2, 2]
   patch_tokens = p_flat.flatten(2).transpose(1,2).reshape(B, C*4, D)
   # result: [B, 31*4, 1024] = [B, 124, 1024]
   ```
4. **Positional embeddings**: the native 196-patch pos_embed is bicubically
   interpolated down to the 2×2 = 4-patch grid, then tiled 31 times:
   ```python
   spatial_pos = interp(model.pos_embed[:, 1:], target=(2,2))  # [1, 4, D]
   spatial_pos = spatial_pos.repeat(1, 31, 1)                   # [1, 124, D]
   ```
5. **CLS token** is prepended. Sequence length = 1 + 124 = 125 tokens.
6. **Transformer**: all 24 ViT-L blocks process the sequence.
7. **Output**: CLS token `[B, 1024]`, reshaped to `[B, 1024, 1, 1]`.

**Runtime:** 32 px input, 125 tokens, batch_size=64, ~1.73 batches/sec on L4 GPU.
Full extraction of 210k train + 30k val cells takes ~40 minutes.

---

## 3. Evaluation Protocol

After extracting features for all train and val cells, the following metrics
are computed using frozen, linear-only methods (no fine-tuning):

| Metric | Method |
|--------|--------|
| **LP** (linear probe balanced accuracy) | `LogisticRegression` fit on train features, evaluated on val |
| **kNN** (kNN balanced accuracy) | `KNeighborsClassifier(k=15, cosine)` fit on train, evaluated on val |
| **NMI / ARI** | `MiniBatchKMeans(k=n_classes)` on train+val concatenated, averaged over 10 random seeds |
| **Silhouette** | Cosine silhouette on up to 10k val cells |
| **Neighbourhood purity** | For each val cell, fraction of its 15 nearest training neighbours sharing its label |
| **Label efficiency** | LP accuracy at {1%, 10%} of training labels and {10, 50, 100, 200, 1000} labels per class |

Features are **not** L2-normalised before evaluation (raw model output is used).

---

## 4. Key Design Choices and Limitations

| | DINOv2 | OpenPhenom | UNI |
|---|---|---|---|
| Training domain | Natural images | Cell Painting (6ch fluorescence) | H&E / IHC pathology |
| Input channels | 3 (RGB) → adapted to 1 | 1 (native) | 3 (RGB) → adapted to 1 |
| Cross-channel attention | Yes (all C channels attend jointly) | No (per-channel, then averaged) | Yes (all C channels attend jointly) |
| img_size used | 56 px (4×4 patches/ch) | 64 px (4×4 patches/ch) | 32 px (2×2 patches/ch) |
| Output dim | 768 | 384 | 1024 |
| Params | 86M | 22M | 307M |
| Pretrain objective | DINO v2 | MAE | DINO v2 |

**Limitation of the small img_size for UNI**: at 32 px, each cell is represented
by only 4 spatial tokens per channel. The spatial resolution is very coarse —
most of the patch content is captured via the patch-mean intensity rather than
local texture. A larger img_size would give richer spatial features but requires
significantly more GPU memory and compute.

**Segmentation masking** is applied to all three models equally, ensuring that
only the pixels belonging to the target cell (as determined by the segmentation
mask) contribute to the feature. Background and neighbouring cells are zeroed out.

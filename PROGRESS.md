# Progress Log

## 2026-02-18

- Refactored augmentation configs: replaced `augmentations.py` and `augmentations_virtualpatients.py` with three named strength files (`augmentations_high.py`, `augmentations_low.py`, `augmentations_none.py`), all using consistent variable names `train_aug_strong` / `train_aug_weak`. Updated all 15 experiment files accordingly.



- Created `configs/_experiments_/CODEX_cHL/CIMATT_VICReg_VP.py` — experiment config combining the CIMATT backbone (WideModel + Attention) with the VICReg algorithm on CODEX_cHL with virtual-patient augmentations.

---

## 2026-02-19

### Experiments run on CODEX_cHL — results summary

All runs evaluated via linear probe (logistic regression, sklearn) on frozen features. Metric: top-1 balanced accuracy on validation set.

| Run | Val Bal Acc | Notes |
|-----|-------------|-------|
| CIMATT_BYOL_SPATIAL_VP | **0.7333** | Best overall |
| CIM_VICReg | 0.7302 | Clean baseline, high aug |
| CIM_MCM_VICReg | 0.7242 | MCM auxiliary loss, slight regression |
| CIM_MASK_VP (baseline) | 0.7240 | |
| CIM_MASK_VP_LONG (10k iters) | 0.7229 | No gain from longer training |
| CIMATT_MASK_VP | 0.7253 | |
| CIMATT_VICReg_MASK_VP | 0.7149 | |
| CIM_VICReg_LOW (low aug) | 0.7213 | Low aug ≈ no improvement |
| CIM_DEEP_VICReg_MASK_VP | 0.6987 | Deeper backbone hurt |
| ResNet_VICReg | 0.6215 | Early fusion baseline — **−10.9pp vs CIM** |
| ATT_MASK_VP | 0.5327 | Attention-only poor |

### Key findings

**1. Performance ceiling at ~0.73 is real and panel-limited.**
All SSL algorithms (SimCLR, VICReg, BYOL), augmentation strategies, model sizes, and training lengths converge to ~0.72–0.73 balanced accuracy. This matches what KRONOS (ViT-Large, 47M patches) achieves on cHL (0.7358), confirming this is the Bayes error rate for this panel — not a model limitation.

**2. Root cause: missing key markers.**
Inspection of `used_markers.txt` revealed that FoxP3 and CD56 are excluded from the 41-marker panel. These are the canonical markers that define the two most confused cell type pairs:
- **CD4 vs TReg** (11% confusion): TReg = CD4+FoxP3+, but FoxP3 is absent. CD25 is a weak proxy.
- **Monocyte vs NK** (16% confusion): NK cells are CD56+, but CD56 is absent. CD16 is shared between both.
FoxP3 and CD56 were excluded because they have notoriously poor staining quality in CODEX (FoxP3 is a nuclear transcription factor with low SNR; CD56 has high background in tissue sections). The confusion is irreducible from the available signal.

**3. Channel augmentations were harmful for T-cell discrimination.**
Original augmentation configs included channel drop/mixup/copy, which taught the model to be invariant to marker expression — precisely the signal needed to distinguish CD4 from TReg. Switching to spatial-only augmentations (`augmentations_spatial_only.py`) improved CD4 recall from ~0.05 to 0.50 and was the single most impactful change (+1.1% balanced accuracy, BYOL+spatial = best run).

**4. LARS optimizer wrong for batch size 256.**
LARS is designed for very large batches (>4096). Replaced with AdamW (lr=3e-4, weight_decay=0.05) and lowered cosine schedule eta_min from 0.1×lr to 1e-6.

**5. Model size and training length do not help.**
CIM_MASK_VP_LONG (10k iters) vs CIM_MASK_VP (1k iters): +0.0009 balanced accuracy. CIM_DEEP_VICReg was worse than the standard CIM. The bottleneck is the SSL objective and data quality, not capacity.

**6. Cell masking is already implemented.**
`MCIDataset` applies `patch * mask` when `mask_patch=True`, zeroing pixels outside the cell boundary. All CODEX_cHL configs use `mask_patch=True`. This was already handled correctly.

**7. Marker identity is implicit via fixed channel positions.**
The panel is fixed for each dataset, so depthwise conv weights per channel implicitly encode marker identity. No explicit marker embeddings are needed (unlike KRONOS which must handle panel-agnostic inputs).

**9. CIM vs ResNet: channel separability gives +10.9pp balanced accuracy.**
Direct comparison of parameter-matched CIM VICReg (0.730) vs ResNet VICReg (0.621) on CODEX_cHL. Early fusion hurts most on cell types requiring fine-grained single-marker discrimination:
- Epithelial: +0.50 (ResNet collapses to CD4/Endothelial/Other)
- Cytotoxic CD8: +0.29 (ResNet confuses with CD8 at 23.7%)
- CD8: +0.20, Monocyte: +0.18, Lymphatic: +0.15
- Neutrophil: 0.00 (strong unique marker — architecture doesn't matter)
- B cells: ResNet marginally better (−0.02, dominant CD20 signal easy for any architecture)

**10. MCM auxiliary loss does not improve over plain VICReg (−0.006).**
CIM + MCM_VICReg (mask_ratio=0.3, mcm_coeff=0.1) scores 0.724 vs CIM + VICReg at 0.730. Losses on Cytotoxic CD8 (−0.026) and TReg (−0.024) outweigh gains on M1 (+0.013) and NK (+0.005). Likely causes:
- `mcm_coeff=0.1` may be pulling backbone gradients away from the VICReg objective
- Mean intensity reconstruction target may bias toward global intensity rather than discriminative patterns
- With 70% of channels visible, MCM target may be trivially predictable, contributing noise rather than signal
- Three backbone forward passes per step (VICReg ×2 + MCM ×1) may destabilize BatchNorm statistics
This is a useful negative result — VICReg's variance/covariance regularization already captures channel relationships effectively. MCM in its current form is not complementary.

**8. Annotation noise as a contributor to the ceiling.**
Some model "errors" are likely annotation errors rather than model failures. The key evidence: CD4-labeled cells are predicted as B cells at 8% — but CD4 T cells and B cells have completely non-overlapping marker profiles (CD4/TCRb vs CD20), both present in the panel. A model seeing both channels should not confuse them, suggesting those cells are actually mislabeled B cells. Likely mechanisms in cHL CODEX: segmentation spillover in crowded tumor microenvironments, rule-based gating errors on borderline cells, and HRS cell rosette proximity causing mask-based misassignment. The model may be *more correct* than the annotation on these cells, meaning the true ceiling is slightly above 0.73.

---

### Plan: Testing for annotation noise

**Hypothesis:** A fraction of model errors are annotation errors — the model's prediction matches the true biology better than the assigned label.

**Test 1 — High-confidence disagreement analysis (most important)**
- Extract embeddings for all val cells using the best model (CIMATT_BYOL_SPATIAL_VP)
- Run the linear probe and collect per-cell prediction + confidence (softmax probability)
- Select cells where: (a) model prediction ≠ annotation label, and (b) model confidence > threshold (e.g. 0.90)
- For each such cell, plot the full marker expression profile (all 41 channels) and visualize the raw patch
- Ask: does the marker profile match the annotation label or the model prediction?
- Specific pairs to check: CD4-labeled → predicted B (CD20 high?), CD4-labeled → predicted TReg (CD25 high?), Monocyte-labeled → predicted NK (CD16 high, CD11b low?)

**Test 2 — Marker expression sanity check**
- For each class, compute the mean expression of its key defining markers across all cells assigned to that class
- Flag cells that are >2 std from the class mean on their defining marker (e.g. a "CD4" cell with very low CD4 expression)
- These are candidate mislabeled cells
- Check whether the model's prediction for these cells is more consistent with their actual marker expression

**Test 3 — Cross-patient consistency**
- If the same cell type from different patients is being confused, that's more likely a model issue
- If confusion is concentrated in specific patients/tissue sections, that's more likely annotation inconsistency (different annotators, different gating strategies per sample)
- Split confusion matrix by patient ID and check if some patients drive most of the errors

**Test 4 — Noise-robust training (if annotation noise confirmed)**
- Re-train with label smoothing (e.g. smoothing=0.1) to reduce overconfidence on noisy labels
- Or use a noise-robust loss function (e.g. symmetric cross-entropy, GCE loss)
- Compare val balanced accuracy — if it improves, label noise was real and was hurting training

**What to implement:**
A notebook `notebooks/annotation_noise_analysis.ipynb` that loads the val set, runs the best model, and produces:
1. Per-cell confidence + prediction vs label table (CSV)
2. Marker expression heatmap for high-confidence disagreements, grouped by (annotation, prediction) pair
3. Patient-stratified confusion matrices

---

### Plan: Testing whether the linear probe is the bottleneck

**Hypothesis:** The representations learned by the model contain more discriminative information than a logistic regression (linear probe) can exploit. The 0.73 ceiling may partly reflect the evaluation method, not just the data.

**Background:** A linear probe can only separate classes with a hyperplane in the embedding space. If the model encodes cell types in a curved or entangled manifold (which is common in self-supervised models not explicitly trained for this task), a non-linear classifier will outperform it.

**Test 1 — kNN probe**
- Compute embeddings for all train and val cells
- Run k-nearest-neighbor classification (k = 5, 15, 50) directly in embedding space
- kNN is assumption-free about the decision boundary shape
- If kNN >> linear probe: the features are good but not linearly separable → nonlinear head needed
- If kNN ≈ linear probe: the features themselves are the bottleneck

**Test 2 — MLP probe**
- Train a small MLP (2 hidden layers, e.g. 256→128→n_classes, ReLU, dropout 0.3) on frozen embeddings
- Use the same train/val split as the linear probe
- If MLP >> linear probe: confirms nonlinearity in the feature space
- Try with and without batch normalization between layers

**Test 3 — UMAP visualization**
- Compute UMAP on val embeddings, color by cell type
- Visually assess: are classes clustered and separated, or mixed?
- If clusters exist but are non-linearly arranged → nonlinear probe would help
- If classes are fully intermixed → the features genuinely lack discriminative signal

**Test 4 — t-SNE / silhouette score**
- Compute silhouette score per class in embedding space (cosine distance)
- Classes with low silhouette score (< 0.1) have poor separation regardless of classifier
- This gives a per-class diagnosis: which classes are genuinely not separated in feature space vs just not linearly separable

**What to implement:**
Extend `src/val_hook.py` (or a separate notebook) to run kNN and MLP evaluation alongside logistic regression at each validation step, reporting all three metrics. This would immediately show during training whether the linear probe is underreporting true feature quality.

---

### Implemented during this session

- `src/BYOL.py` — MVBYOL class (EMA target network, no negatives)
- `configs/_algorithms_/BYOL.py`
- `configs/_base_/augmentations_spatial_only.py` — spatial augs only, minimal channel shift/scale
- `configs/_backbones_/CIM_deep.py` — deeper WideModel (layer_config=[2,2], late_fusion)
- `configs/_base_/train_cfg.py` / `train_cfg_long.py` — AdamW, lower eta_min
- `src/val_hook.py` — added JSON confusion matrix output alongside PNG
- Multiple experiment configs in `configs/_experiments_/CODEX_cHL/`

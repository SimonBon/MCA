# Publication Plan: Channel Separability as an Inductive Bias for Multiplex Spatial Proteomics

## Core Claim

> Depthwise (channel-separable) convolutions are a more appropriate inductive bias than standard (early fusion) convolutions for self-supervised representation learning on multiplex spatial proteomics data. This holds consistently across datasets, imaging modalities, and SSL algorithms.

The preprint covers CODEX_cHL on one dataset with one architecture. The paper strengthens this into a **generalizable principle** by ablating the fusion point, testing across four datasets/modalities, and providing a mechanistic explanation for why early fusion fails.

---

## Narrative Arc

1. **Problem**: Multiplex spatial proteomics images differ fundamentally from RGB photos — each channel encodes a distinct protein marker with independent biological meaning. Standard vision models treat all channels identically.

2. **Hypothesis**: Channel-separable architectures, by preventing channels from mixing until a late stage (or not at all), preserve marker-specific signal and learn better cell representations under SSL.

3. **Evidence**: Ablation across a spectrum from fully-separable to fully-fused, replicated across 4 datasets (CODEX, IMC, MIBI) and 3 SSL algorithms.

4. **Mechanism**: Early fusion conflates marker signals early, forcing the model to disentangle them again. Channel-separable models never mix unrelated channels — augmentation invariances are also more biologically meaningful.

5. **Contribution**: (a) The ablation study itself, (b) MCM — a domain-specific pretext task that exploits the channel structure to learn marker co-expression, (c) practical recommendations for practitioners building SSL models for spatial proteomics.

---

## Datasets

| Dataset | Modality | Markers | Cell types | Patients | Status |
|---------|----------|---------|------------|----------|--------|
| CODEX_cHL | CODEX | 41 | ~14 | multi | ✓ preprint |
| CODEX_DLCBL | CODEX | ~38 | ~10 | 2 | ✓ preliminary |
| IMC_NB | IMC | ~30 | TBD | TBD | not run |
| MIBI_TNBC | MIBI | ~36 | TBD | TBD | not run |

Multi-modality (CODEX, IMC, MIBI) is important — it shows the finding is not instrument-specific.

---

## Experiments

### Experiment 1 — Backbone Ablation (Core Contribution)

**Question**: Where in the network does early fusion become harmful?

Run all five architectures with VICReg on all four datasets. The architectures form a spectrum:

| Architecture | Fusion point | Groups | Expected |
|---|---|---|---|
| WideModel (CIM) | Never | Depthwise throughout | Best |
| ProjectionFusionModel | Before spatial processing (1×1) | Depthwise after | ? |
| MidFusionModel | After stem | Standard after stem | ? |
| EarlyFusionModel | Stem | Standard throughout | Worse |
| ResNetBaseline | Stem | Standard throughout | Worst |

All parameter-matched (~1.1M params). Algorithm fixed: VICReg.

**Configs needed**: EarlyFusion, MidFusion, ProjectionFusion configs for all 4 datasets (currently only CODEX_cHL has ResNet + CIM).

---

### Experiment 2 — Algorithm Sensitivity (Robustness Check)

**Question**: Is the advantage of channel separability specific to one SSL algorithm?

Run the best channel-separable (CIM) and best early-fusion (ResNet) architectures with all three SSL algorithms on CODEX_cHL.

| | SimCLR | BYOL | VICReg |
|---|---|---|---|
| CIM | ✓ config ready | ✓ config ready | ✓ config ready |
| ResNet | add config | add config | ✓ config ready |

If CIM > ResNet consistently across all three algorithms, the finding is algorithm-agnostic.

---

### Experiment 3 — Masked Channel Modeling (Domain-Specific Contribution)

**Question**: Does explicitly learning marker co-expression improve upon standard SSL?

| Model | Val Bal Acc |
|---|---|
| CIM + VICReg | baseline |
| CIM + MCM_VICReg | ? |

Run on CODEX_cHL first, then on the best cross-dataset config.

Expected: MCM should improve most on cell types that require integrating multiple markers (e.g. TReg = CD4+CD25+, NK = CD56-CD16+).

---

### Experiment 4 — Cross-Dataset Generalization

**Question**: Does the channel-separable model generalize better across patients and datasets?

Two sub-experiments:

**4a — Cross-patient (CODEX_DLCBL)**
Already have preliminary results: same-patient 0.699 vs cross-patient 0.578.
Need to repeat with ResNet baseline to compare the drop.
If CIM's cross-patient drop < ResNet's, channel separability generalizes better.

**4b — Cross-modality**
Train on CODEX_cHL, evaluate (linear probe only) on IMC_NB or MIBI_TNBC after aligning shared markers.
This is ambitious but would be a strong result if feasible. Alternatively, just show that the ranking (CIM > ResNet) holds within each dataset independently.

---

### Experiment 5 — Augmentation Strategy Analysis

Already partially done. Formalize as a controlled experiment:

| Augmentation | CIM Val Acc | Notes |
|---|---|---|
| None | ? | |
| Spatial only | 0.7333 (BYOL, best) | T-cell discrimination improves |
| Low channel | ? | |
| High channel | ~0.72 | CD4 recall collapses |

**Finding**: Channel augmentations (drop, mixup, copy) are harmful because they teach invariance to marker expression — the primary signal. This is worth reporting as a practical recommendation.

---

### Experiment 6 — Mechanistic Analysis (Why Early Fusion Fails)

**Question**: What specifically does early fusion get wrong?

**6a — Per-class breakdown**
Report per-class recall for CIM vs ResNet on all datasets.
Hypothesis: early fusion hurts most on cell types defined by a single unique marker (e.g. CD20+ B cells), because the unique marker signal gets diluted by early mixing.

**6b — Embedding visualization**
UMAP of val embeddings for CIM vs ResNet, colored by cell type.
Qualitatively show that CIM produces more separated clusters.

**6c — Parameter count vs accuracy**
Plot accuracy vs #params for all architectures.
Show that CIM's advantage is not just from parameter efficiency.

---

### Experiment 7 — Probe Quality (Supplementary)

Already planned in PROGRESS.md. Run kNN and MLP probe alongside logistic regression.
Shows whether 0.73 ceiling is feature-limited or probe-limited.
Positions as a calibration for all reported numbers — if kNN ≈ linear, the linear probe is reliable.

---

## Figures Plan

| Figure | Content | Experiment |
|---|---|---|
| 1 | Architecture diagram: fusion spectrum from early to never | Conceptual |
| 2 | Main result table / bar chart: all architectures × all datasets | Exp 1 |
| 3 | Algorithm sensitivity: CIM vs ResNet × SimCLR/BYOL/VICReg | Exp 2 |
| 4 | MCM: training curves + per-class improvement over VICReg | Exp 3 |
| 5 | Cross-patient generalization: CIM vs ResNet drop | Exp 4a |
| 6 | Per-class recall: CIM vs ResNet, highlighting single-marker cell types | Exp 6a |
| 7 | UMAP: CIM vs ResNet embeddings on CODEX_cHL | Exp 6b |
| S1 | Augmentation ablation table | Exp 5 |
| S2 | Probe quality: linear vs kNN vs MLP | Exp 7 |
| S3 | Parameter matching analysis | Methods justification |

---

## What Is Already Done

- [x] WideModel (CIM) on CODEX_cHL — all algorithms (SimCLR, BYOL, VICReg)
- [x] Cross-patient DLCBL preliminary results
- [x] Augmentation ablation (spatial-only finding)
- [x] Configs for CIM, ResNet, EarlyFusion, MidFusion, ProjectionFusion on CODEX_cHL
- [x] MCM_VICReg implemented and sanity-checked
- [x] Key findings on performance ceiling (panel-limited, not model-limited)

## What Needs to Be Run

### High priority (core paper)
- [ ] ResNet + SimCLR, BYOL configs for CODEX_cHL (Exp 2)
- [ ] EarlyFusion, MidFusion, ProjectionFusion + VICReg on CODEX_cHL (Exp 1)
- [ ] CIM + MCM_VICReg on CODEX_cHL (Exp 3)
- [ ] All backbone configs on IMC_NB and MIBI_TNBC (Exp 1 generalization)
- [ ] Cross-patient DLCBL with ResNet baseline (Exp 4a)

### Medium priority (strengthens claims)
- [ ] Annotation noise analysis notebook (PROGRESS.md plan)
- [ ] kNN/MLP probe in val_hook (Exp 7)
- [ ] UMAP visualization notebook (Exp 6b)
- [ ] Per-class recall analysis across architectures (Exp 6a)

### Low priority (supplementary)
- [ ] Augmentation ablation formalized as table (currently implicit in existing runs)
- [ ] Cross-modality generalization (Exp 4b) — only if IMC/MIBI runs succeed

---

## Missing Configs to Create

```
configs/_experiments_/
├── CODEX_cHL/
│   ├── ResNet_SimCLR.py       (Exp 2)
│   ├── ResNet_BYOL.py         (Exp 2)
│   ├── EarlyFusion_VICReg.py  (Exp 1)
│   ├── MidFusion_VICReg.py    (Exp 1)
│   └── ProjectionFusion_VICReg.py  (Exp 1)
├── CODEX_DLCBL/
│   ├── CIM_VICReg.py          (Exp 4a)
│   └── ResNet_VICReg.py       (Exp 4a)
├── IMC_NB/
│   ├── CIM_VICReg.py          (Exp 1 generalization)
│   └── ResNet_VICReg.py       (Exp 1 generalization)
└── MIBI_TNBC/
    ├── CIM_VICReg.py          (Exp 1 generalization)
    └── ResNet_VICReg.py       (Exp 1 generalization)
```

---

## Key Narrative Points to Make in Paper

1. **Fair comparison**: All architectures parameter-matched to ~1.1M. Without this, any difference could be attributed to capacity.

2. **The O(C) vs O(C²) scaling argument**: Standard convs scale quadratically with channel count; depthwise scale linearly. As panels grow (100+ markers on the horizon), this becomes critical.

3. **Augmentation design is architecture-dependent**: Channel drop/mixup augmentations, commonly used in SSL, are *specifically harmful* for channel-separable data. This is a practical finding for the community.

4. **The ceiling is biological, not architectural**: 0.73 on cHL is the Bayes error rate given the available markers. This contextualizes all results and shows that model improvements have reached the information-theoretic limit of the current panels.

5. **MCM as a generalizable pretext task**: Masked channel modeling is analogous to BERT/MAE pretraining but for protein marker panels. The co-expression structure it learns is the same signal biologists use for manual cell type gating.

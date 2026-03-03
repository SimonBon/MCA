# Publication Journal — MCA

_Last updated: 2026-03-03_

---

## Working Title

**"Channel-Independent Self-Supervised Learning for Robust Cell Representation in Multiplexed Imaging"**

_(tentative — can shift depending on final story emphasis)_

---

## Direction

Focusing on **Direction A or C**, keeping options open:

- **Direction A (Architecture):** Clean, verifiable story — CIM's channel-independent design outperforms early-fusion and ResNet baselines across imaging technologies. Core claim: depthwise grouped conv avoids learning spurious cross-channel correlations that are panel/batch-specific.

- **Direction C (Full framework):** Broader contribution — a complete SSL pipeline for multiplexed imaging covering augmentation design, architecture, objective (VICReg), and evaluation protocol. Positions as a reusable tool comparable to KRONOS but open and more flexible.

Both directions share the same experimental core. Direction C requires more experiments (LARS on IMC_NB/MIBI_TNBC, label efficiency across all datasets) but targets a higher venue.

**Target venues (TBD):**
- Nature Methods (Direction C, strong tool + novelty angle)
- Bioinformatics / NAR (Direction A or B, solid methods benchmark)
- Cell Systems / iScience (if biological insight becomes prominent)

---

## Core Narrative

Multiplexed imaging (CODEX, IMC, MIBI-TOF) captures 20–40+ protein channels per single cell. Existing self-supervised approaches (e.g. KRONOS) use standard convolutions that mix channels from the input stem — making representations sensitive to panel composition, staining intensity variation, and batch effects across patients.

**CIM** (Channel-Independent Model) processes each protein channel through a shared depthwise grouped conv backbone before fusing representations. This inductive bias:
1. Prevents learning spurious cross-channel correlations tied to a specific panel
2. Makes representations more robust to inter-sample intensity variation
3. Enables better geometric cluster structure alongside competitive classification accuracy

**Key supporting findings:**
- CIM outperforms EarlyFusion32 and ResNet on LP accuracy, NMI, ARI, and silhouette across all tested datasets
- EarlyFusion fragmentation (negative silhouette) is structural, not a training artefact — consistent across MIBI_TNBC CV splits
- CIM_Norm (L2 input normalisation) is the fix for inter-sample intensity variation — only model with positive silhouette on MIBI_TNBC
- CIM matches/beats KRONOS on the cHL KRONOS18 panel (LP_bal 0.738 vs KRONOS ~0.736, mAP 0.740 vs 0.761)
- Label efficiency: at 100 cells/class (1600 total), CIM already reaches ~66% bal_acc vs 74% at 100% labels (~89% of full performance)

---

## Datasets

| Dataset | Technology | Markers | Classes | Cells | Role |
|---|---|---|---|---|---|
| CODEX_cHL | CODEX | 41 (18 KRONOS) | 16 | ~115k | KRONOS comparison, main benchmark |
| CODEX_DLBCL | CODEX | 40 | 18 | ~416k | Scale + generalization |
| IMC_NB | IMC | 31 | 7 coarse / 11 fine | ~240k | Cross-technology |
| MIBI_TNBC | MIBI-TOF | 37 | 16 | ~196k | 5-fold CV, robustness, label efficiency |

All four datasets span three imaging technologies and three tissue/disease contexts — strong cross-technology generalization story.

---

## Models in Paper

| Model | Description | Key role |
|---|---|---|
| CIM (WideModel) | Depthwise grouped conv, channel-independent | **Proposed method** |
| CIM_Norm | CIM + L2 input normalisation | Best cross-sample consistency |
| CIM_ProgFusion | Dual-branch with progressive fusion | Best cluster geometry |
| EarlyFusion32 | Standard conv, channels mixed from stem | Architecture baseline |
| ResNet | ResNet baseline | Standard DL baseline |
| KRONOS | External SOTA (published) | External comparison |

_CIMATT_Gate can be included as ablation if it adds to the story (fragmented UMAP despite high LP)._

---

## Key Metrics

- **LP balanced accuracy** — primary classification metric
- **Mean Average Precision (mAP)** — per-class AP, matches KRONOS reporting
- **kNN balanced accuracy** — zero-shot proxy
- **NMI / ARI** — cluster quality
- **Silhouette (cosine)** — embedding geometry / cross-sample mixing
- **Label efficiency curve** — practical metric: bal_acc vs n_labeled cells

---

## Confirmed Results (as of 2026-03-03)

### CODEX_cHL — KRONOS18 panel (16 classes)

| Model | LP_bal | mAP | kNN_bal | NMI | ARI | Sil |
|---|---|---|---|---|---|---|
| CIM_32 (4k) | **0.7384** | 0.7398 | 0.6500 | **0.3839** | 0.2166 | **0.0541** |
| CIM_32 (8k) | 0.7371 | **0.7405** | 0.6537 | 0.3807 | **0.2228** | 0.0505 |
| CIM_64 (4k) | 0.7356 | 0.7392 | 0.6506 | 0.3813 | 0.2160 | 0.0530 |
| EF32 (4k) | 0.7099 | 0.7222 | 0.6125 | 0.3403 | 0.1678 | 0.0096 |
| KRONOS (published) | ~0.736 | ~0.761 | — | — | — | — |

### MIBI_TNBC — 5-fold CV (mean ± std)

| Model | LP_bal | Sil | NMI |
|---|---|---|---|
| CIM | 0.810 ± 0.013 | −0.082 ± 0.030 | — |
| CIM_Norm | 0.809 ± 0.011 | **+0.003 ± 0.024** | — |
| CIM_ProgFusion | 0.799 ± 0.010 | — | **0.333 ± 0.008** |
| EarlyFusion32 | 0.807 ± 0.009 | −0.228 ± 0.034 | 0.215 ± 0.035 |

### Label Efficiency — CIM_32 KRONOS18 (C=10, needs re-run with C=1)

| n_labeled | bal_acc | mAP |
|---|---|---|
| 1001 (1%) | 0.595 | 0.592 |
| 1600 (100/cls) | 0.657 | 0.637 |
| 8000 (500/cls) | 0.718 | 0.716 |
| 100315 (100%) | 0.744 | 0.746 |

---

## Experiments Still Needed

### High priority (needed for any paper version)
- [ ] **LARS runs on IMC_NB** (CIM + EF32) — configs exist, not yet run
- [ ] **LARS runs on MIBI_TNBC** (CIM + EF32) — configs exist, not yet run
- [ ] **Label efficiency re-run with C=1** for CIM_32 KRONOS18 (previous run used C=10)
- [ ] **Label efficiency for all key models** on all 4 datasets
- [ ] **CIM_Norm full eval** on CODEX_cHL and CODEX_DLBCL (currently only MIBI_TNBC CV)
- [ ] **Full CODEX_DLBCL results** — confirm CIM vs EF32 gap holds at scale

### Medium priority (strengthens story)
- [ ] **Ablation: why does channel independence help?** — e.g. per-channel feature norm analysis, or CIM vs EF32 on same exact training budget/setup
- [ ] **Supervised baseline** — at least one supervised model to give LP numbers context (how close does SSL get?)
- [ ] **CIM_ProgFusion on cHL KRONOS18** — does it beat CIM on cluster metrics here too?

### Lower priority / nice to have
- [ ] CIMATT_Gate full analysis (fragmented UMAP despite high LP — interesting failure mode)
- [ ] Cross-dataset transfer (limited by differing panels, but partial marker overlap possible)
- [ ] Spatial downstream task using region analysis (already have notebook for MIBI_TNBC)

---

## Proposed Figure Structure (draft)

1. **Overview figure** — problem statement: multiplexed imaging, patch extraction, SSL pipeline, downstream tasks
2. **Architecture figure** — CIM vs EarlyFusion vs ResNet design; depthwise grouped conv explanation
3. **Main results table/figure** — LP + mAP + NMI/ARI + silhouette across all 4 datasets
4. **UMAP panel** — CIM vs EF32 on MIBI_TNBC (colour by cell type + by sample) — shows fragmentation
5. **Label efficiency curves** — bal_acc vs n_labeled for CIM (+ optionally CIM_Norm, EF32) on 2-3 datasets
6. **Cross-sample consistency** — silhouette by model across CV splits; CIM_Norm positive silhouette
7. **KRONOS comparison** — head-to-head on cHL KRONOS18 panel

---

## Open Questions / Decisions Pending

- Final venue decision (Nature Methods vs Bioinformatics/NAR vs other)
- Whether to include CIMATT_Gate or keep model list lean
- Whether to add a biological application section (spatial analysis, patient stratification)
- Whether to include CIM_ProgFusion as a variant or just mention in discussion
- Authorship and collaboration

---

## Notes / Session Log

### 2026-03-03
- Agreed to focus on Direction A or C, keeping options open
- Direction A: architecture-first, publishable with current experiments
- Direction C: full framework paper, needs LARS runs + label efficiency across all datasets
- All C=10 → C=1 change applied to val_hook_rich.py and tools/label_efficiency.py
- KRONOS uses Optuna with C_low=1e-10, C_high=1e5, 25 trials (inverted convention: passes C=1/C to sklearn)
- Label efficiency for CIM_32 KRONOS18 exists but was run with C=10 — needs re-run

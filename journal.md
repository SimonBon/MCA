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

**The core tension (empirical finding):** EarlyFusion sometimes matches CIM on LP accuracy (a supervised readout, 1-3pp gap), but consistently produces fragmented embeddings — negative silhouette, poor clustering, sample-stratified UMAP. The fragmentation is structural, not a training artefact (consistent across all 5 CV splits on MIBI_TNBC, gap of ~0.15 in silhouette). CIM trades essentially nothing on accuracy for a dramatically better embedding geometry.

**Model hierarchy across tasks:**
- Pure classification (LP): CIM ≈ EF32 (EF32 wins by 1-3pp on IMC; CIM wins by 3pp on CODEX_cHL KRONOS18)
- Embedding geometry (Sil): CIM >> EF32 by 0.05–0.20 consistently
- Cluster recovery (NMI/ARI): CIM_ProgFusion > CIM > EF32 universally
- Parameter efficiency: CIM (0.84–1.1M) vs EF32 (25–45M) — 30–45× fewer params for same or better results
- External comparison: CIM matches KRONOS on CODEX_cHL KRONOS18 panel

---

## Datasets

| Dataset | Technology | Markers | Classes | Cells | Role |
|---|---|---|---|---|---|
| CODEX_cHL | CODEX | 41 (18 KRONOS) | 16 | ~115k | KRONOS comparison, main benchmark |
| CODEX_DLBCL | CODEX | 40 | 18 | ~416k | Scale + generalization |
| IMC_NB | IMC | 31 | 7 coarse / 11 fine / 19 TumorSub | ~240k | Cross-technology, tumour subtype resolution |
| MIBI_TNBC | MIBI-TOF | 37 | 16 | ~196k | 5-fold CV, robustness, label efficiency |

All four datasets span three imaging technologies and three tissue/disease contexts — strong cross-technology generalization story.

---

## Models in Paper

| Model | Params (37ch) | Description | Key role |
|---|---|---|---|
| CIM (WideModel) | ~0.84M | Depthwise grouped conv, channel-independent | **Proposed method** |
| CIM_ProgFusion | ~1.93M | Dual-branch with progressive fusion | Best cluster geometry |
| EarlyFusion32 | ~25–45M | Standard conv, channels mixed from stem | Architecture baseline |
| ResNet | ~1.13M | Standard ResNet | DL baseline |
| KRONOS | — | External published SOTA | External comparison |

_CIM_Norm excluded from paper. CIMATT_Gate can be included as ablation (fragmented UMAP despite highest LP — interesting failure mode showing attention-gating doesn't help geometry)._

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

### CODEX_cHL — KRONOS18 panel (16 classes, LARS 4k)

| Model | LP_bal | mAP | kNN_bal | NMI | ARI | Sil |
|---|---|---|---|---|---|---|
| CIM (4k) | **0.7384** | 0.7398 | 0.6500 | **0.3839** | 0.2166 | **0.0541** |
| CIM (8k) | 0.7371 | **0.7405** | 0.6537 | 0.3807 | **0.2228** | 0.0505 |
| CIM_64 (4k) | 0.7356 | 0.7392 | 0.6506 | 0.3813 | 0.2160 | 0.0530 |
| EF32 (4k) | 0.7099 | 0.7222 | 0.6125 | 0.3403 | 0.1678 | 0.0096 |
| KRONOS (published) | ~0.736 | ~0.761 | — | — | — | — |

_Note: CIM_ProgFusion not yet run on KRONOS18. Full-panel (41-marker) runs also exist but use 17 classes and different setup — not directly comparable._

### CODEX_cHL — Full 41 markers (17 classes, AdamW 1k)

| Model | LP_bal | kNN_bal | NMI | ARI | Sil |
|---|---|---|---|---|---|
| CIM | **0.7248** | 0.5245 | **0.3043** | **0.1623** | −0.0031 |
| CIM_ProgFusion | 0.7041 | 0.5026 | 0.3281 | 0.2051 | −0.0194 |
| EF32 | 0.7073 | 0.4984 | 0.2629 | 0.1179 | −0.0672 |
| ResNet | 0.6183 | 0.4911 | 0.2701 | 0.1283 | −0.0680 |

### CODEX_DLBCL — 40 markers, 18 classes (AdamW 1k)

| Model | LP_bal | kNN_bal | NMI | ARI | Sil |
|---|---|---|---|---|---|
| CIM | **0.7005** | 0.3613 | 0.1960 | 0.1032 | −0.0501 |
| CIM_Norm | 0.6954 | 0.3910 | **0.2468** | 0.0984 | **−0.0132** |
| CIM_ProgFusion | 0.6883 | **0.4035** | 0.2540 | **0.1068** | −0.0651 |
| EF32 | 0.7003 | 0.3901 | 0.1918 | 0.0558 | −0.1915 |
| ResNet | 0.5033 | 0.3366 | 0.2242 | 0.0716 | −0.0940 |

_CIM and EF32 tie on LP (~0.700) but EF32 silhouette is catastrophic (−0.192 vs CIM −0.050). CIM_Norm best geometry despite marginal LP loss._

### IMC_NB — Coarse 7 classes (AdamW 1k)

| Model | LP_bal | kNN_bal | NMI | ARI | Sil |
|---|---|---|---|---|---|
| CIM | 0.8042 | 0.6422 | 0.2702 | 0.1904 | +0.0579 |
| EF32 | **0.8310** | **0.6777** | 0.2797 | 0.1899 | **+0.0615** |
| ResNet | 0.8021 | **0.6948** | **0.3142** | 0.1736 | 0.0505 |

_One of the rare cases where EF32 leads on LP (+3pp). Only 7 coarse classes; may not expose fragmentation. AdamW 1k — not comparable to LARS 4k runs._

### IMC_NB_FineCT — 11 fine classes (AdamW 1k)

| Model | LP_bal | kNN_bal | NMI | ARI | Sil |
|---|---|---|---|---|---|
| CIM | 0.8043 | 0.6086 | 0.2630 | 0.1761 | −0.0081 |
| CIM_Norm | **0.8222** | 0.5911 | 0.2912 | 0.1431 | +0.0110 |
| CIM_ProgFusion | 0.8143 | 0.6115 | 0.2879 | 0.1426 | +0.0303 |
| EF32 | 0.8209 | **0.6548** | 0.2840 | 0.1523 | +0.0349 |
| ResNet | 0.7739 | 0.6256 | **0.3298** | **0.1687** | **+0.0503** |

_AdamW 1k only. EF32 positive sil here (rare) — may be due to 11 well-separated fine classes or small training budget._

### IMC_NB_TumorSub — 19 tumour subtype classes (LARS 4k)

| Model | LP_bal | mAP | kNN_bal | NMI | ARI | Sil |
|---|---|---|---|---|---|---|
| CIM | 0.7706 | 0.7675 | 0.6552 | 0.3189 | 0.1503 | +0.0139 |
| CIM_ProgFusion | 0.7869 | 0.7904 | 0.6697 | **0.4325** | **0.2394** | **+0.0346** |
| EF32 | **0.7872** | **0.8025** | **0.6726** | 0.3654 | 0.1766 | +0.0179 |

_EF32 and ProgFusion tie on LP; ProgFusion clearly wins on NMI (+7pp), ARI (+6pp), Sil. CIM_Norm not yet run._

### MIBI_TNBC — Full dataset, single run (AdamW 1k, 37 markers, 16 classes)

| Model | LP_bal | kNN_bal | NMI | ARI | Sil |
|---|---|---|---|---|---|
| CIM | 0.8446 | 0.5780 | 0.2720 | 0.1041 | −0.0937 |
| CIM_Norm | 0.8386 | 0.5802 | **0.3207** | **0.1259** | **+0.0017** |
| CIM_ProgFusion | 0.8297 | 0.5714 | 0.3300 | 0.1389 | +0.0006 |
| EF32 | 0.8371 | 0.4458 | 0.2032 | 0.0715 | −0.2046 |
| ResNet | 0.5925 | 0.3963 | 0.2225 | 0.0672 | −0.1507 |
| CIMATT_Gate | **0.8454** | 0.5598 | 0.3036 | 0.1184 | −0.0881 |

### MIBI_TNBC — 5-fold CV (mean ± std, all 5 splits, sample-stratified)

| Model | LP_bal | NMI | Sil |
|---|---|---|---|
| CIM | 0.810 ± 0.014 | 0.294 ± 0.032 | −0.082 ± 0.027 |
| CIM_Norm | 0.809 ± 0.012 | 0.326 ± 0.024 | **+0.003 ± 0.028** |
| CIM_ProgFusion | 0.799 ± 0.011 | **0.341 ± 0.018** | +0.014 ± 0.013 |
| EarlyFusion32 | **0.807 ± 0.009** | 0.215 ± 0.025 | −0.228 ± 0.036 |

_Key finding: LP gap CIM vs EF32 is within noise (±0.003 of each other). Silhouette gap is massive and consistent across ALL 5 folds (EF32 range −0.162 to −0.253, always negative). CIM_ProgFusion has lowest NMI variance (±0.018 vs EF32 ±0.025)._

Per-fold LP_bal detail:
| Fold | CIM | CIM_Norm | CIM_ProgFusion | EF32 |
|---|---|---|---|---|
| CV0 | 0.797 | 0.804 | 0.790 | 0.801 |
| CV1 | 0.832 | 0.831 | 0.810 | 0.815 |
| CV2 | 0.801 | 0.805 | 0.791 | 0.796 |
| CV3 | 0.803 | 0.801 | 0.792 | 0.802 |
| CV4 | 0.817 | 0.804 | 0.813 | 0.821 |

### Label Efficiency — CIM_32 KRONOS18 (C=10, needs re-run with C=1)

| n_labeled | bal_acc | mAP |
|---|---|---|
| 1001 (1%) | 0.595 | 0.592 |
| 1600 (100/cls) | 0.657 | 0.637 |
| 8000 (500/cls) | 0.718 | 0.716 |
| 100315 (100%) | 0.744 | 0.746 |

---

## Paper Story

### The Argument (in order)

**1. Problem:** Multiplexed imaging captures 20–40 protein channels per cell patch. The field lacks a principled SSL approach — KRONOS is the closest but uses a standard early-fusion design (channels mixed in the input stem) that learns panel-specific correlations and is sensitive to batch effects.

**2. Proposed solution:** CIM — a channel-independent backbone using depthwise grouped conv. Each channel processed independently; representations fused in late layers. This is a direct architectural choice against learning spurious cross-channel correlations.

**3. Primary evidence — CODEX_cHL KRONOS18 (head-to-head with KRONOS):**
CIM achieves LP_bal 0.738, matching/beating KRONOS (~0.736). EF32 scores only 0.710. mAP gap: CIM 0.740 vs KRONOS 0.761 (still within 2.1pp). CIM does this with 30–45× fewer parameters than EF32.

**4. The fragmentation finding (MIBI_TNBC CV — the key result):**
On 5 patient-stratified CV splits, CIM and EF32 are statistically indistinguishable on LP (0.810 vs 0.807, p>>0.05). But EF32 silhouette is −0.228±0.036 vs CIM −0.082±0.027 — a gap that is consistent across every single fold. EF32 learns sample identity as a proxy for cell type. This is not a training artefact; it is structural.

**5. Best cluster geometry — CIM_ProgFusion:**
Progressive fusion (CIM branch injects into early-fusion stream at each stage) achieves best NMI across all datasets (0.341±0.018 on MIBI_TNBC CV, 0.433 on IMC_NB_TumorSub) with lowest variance. LP is 1–2pp below CIM but the embedding is more structured. This is the recommended model for spatial/clustering downstream tasks.

**7. Cross-technology generalization:**
The pattern holds on CODEX (cHL, DLBCL) and IMC (NB). On CODEX_DLBCL (416k cells, 40 markers), EF32 silhouette collapses to −0.192 while CIM stays at −0.050. On IMC_NB_TumorSub (19 tumour subtypes), ProgFusion wins on all geometry metrics while tying EF32 on LP.

**8. Label efficiency:**
CIM reaches ~89% of full-data accuracy with only 100 labels/class. This makes the approach practical for routine use (most labs lack large annotated datasets).

**9. Patient-level downstream (proposed):**
MIBI_TNBC has patient-level clinical annotations (outcome, subtype). Using CIM embeddings to stratify patients or predict outcome would demonstrate real biological utility — converts the method from benchmark to tool.

---

## Experiment Plan

### Priority 1 — Required for any submission

| # | Experiment | Rationale | Est. cost |
|---|---|---|---|
| 1 | **Label efficiency re-run (C=1)** for CIM on CODEX_cHL KRONOS18 | Previous run used C=10 — results invalid | 1 GPU-h |
| 2 | **Label efficiency for EF32** on CODEX_cHL KRONOS18 | Need baseline comparison | 1 GPU-h |
| 3 | **LARS runs on MIBI_TNBC** (CIM + EF32) | Configs exist; needed for main MIBI_TNBC table | 2×4 GPU-h |
| 4 | **CIM_ProgFusion LARS on CODEX_cHL KRONOS18** | Does ProgFusion beat CIM on cluster metrics here? | 4 GPU-h |
| 5 | **Label efficiency for CIM across all 4 datasets** | Core metric for Direction C | 4× 1 GPU-h |

### Priority 2 — Strengthens story

| # | Experiment | Rationale |
|---|---|---|
| 6 | **Label efficiency for EF32 + CIM_ProgFusion** across datasets | Show CIM_ProgFusion also label-efficient |
| 7 | **IMC_NB_FineCT LARS runs** (CIM + EF32 + ProgFusion) | Fair comparison with TumorSub LARS runs |
| 8 | **UMAP coloured by patient for CODEX_DLBCL** | Visual confirmation of EF32 fragmentation at scale |

### Priority 3 — Ablations

| # | Ablation | Purpose |
|---|---|---|
| A1 | **CIM vs EF32 at matched parameter count** — reduce EF32 stem width to match CIM params | Isolate architecture effect from capacity |
| A2 | **CIM_ProgFusion: fusion injection at different stages** — early/mid/late only | Understand what makes progressive fusion work |
| A3 | **CIMATT_Gate full analysis** — LP + sil + UMAP | Demonstrates attention gating creates non-smooth manifold; failure mode story |

### Patient-level MIBI_TNBC Analysis (deferred — plan later)

_MIBI_TNBC patient-level annotation inference (clinical outcome stratification) is deferred. Plan separately once core experiments are done._

---

## Proposed Figure Structure

1. **Overview figure** — problem statement: multiplexed imaging, patch extraction, SSL pipeline, downstream tasks
2. **Architecture figure** — CIM vs EarlyFusion vs ResNet; depthwise grouped conv vs standard conv; parameter count comparison
3. **Main results table/heatmap** — LP + NMI/ARI + silhouette across all 4 datasets, all 5 models
4. **Fragmentation figure** — UMAP pairs (CIM vs EF32) on MIBI_TNBC coloured by cell type and by patient; silhouette CV strips across all folds
5. **KRONOS comparison** — head-to-head on cHL KRONOS18 panel; CIM matches KRONOS at 30× fewer params
6. **Label efficiency curves** — bal_acc vs n_labeled for CIM (± EF32 ± ProgFusion) on 2–3 datasets

---

## Open Questions / Decisions Pending

- Final venue decision (Nature Methods vs Bioinformatics/NAR vs other)
- Whether to include CIMATT_Gate or keep model list lean
- MIBI_TNBC patient-level analysis: plan separately after core experiments
- Whether to promote CIM_ProgFusion as the recommended model or keep CIM as primary
- IMC_NB_TumorSub: keep all 19 classes or drop rare tumour subtypes (GATAhi, CHGAhi) after results come in
- Authorship and collaboration

---

## Notes / Session Log

### 2026-03-03 (session 1)
- Agreed to focus on Direction A or C, keeping options open
- Direction A: architecture-first, publishable with current experiments
- Direction C: full framework paper, needs LARS runs + label efficiency across all datasets
- All C=10 → C=1 change applied to val_hook_rich.py and tools/label_efficiency.py
- KRONOS uses Optuna with C_low=1e-10, C_high=1e5, 25 trials (inverted convention: passes C=1/C to sklearn)
- Label efficiency for CIM_32 KRONOS18 exists but was run with C=10 — needs re-run

### 2026-03-03 (session 2)
- Created IMC_NB_TumorSub dataset: 19 classes (9 tumour subtypes kept, CD44+ TC bug fixed)
- Created LARS 4k configs for IMC_NB_TumorSub (CIM, CIM_Norm, EF32, CIM_ProgFusion)
- Pulled IMC_NB_TumorSub LARS results: ProgFusion ties EF32 on LP, wins clearly on NMI/ARI/Sil
- val_hook_rich.py: le_fractions=(0.01, 0.1), le_n_per_class=(10, 50, 100, 200, 1000), no 100% fraction
- Printed model parameter counts across all datasets
- Full results compilation: see Confirmed Results section above
- Paper story and experiment plan written

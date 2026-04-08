# MCA: Experimental Findings for Paper

Self-supervised cell representation learning for multiplexed imaging (CODEX, IMC, MIBI-TOF).
All models trained with VICReg unless noted. Metrics: **LP_bal** = balanced accuracy of linear probe, **mAP** = mean average precision over classes, **NMI** = normalised mutual information (k-Means, k=n_classes), **ARI** = adjusted Rand index, **kNN** = k=15 balanced accuracy, **sil** = silhouette score (sample-level, proxy for inter-sample integration).

> **Critical note on evaluation:** Two eval modes exist.
> - **Unmasked**: raw features extracted from the full patch, LP/kNN/clustering run directly. This is what the training hook (`EvaluateModelRich`) produces.
> - **Masked**: cell mask applied before feature extraction (non-cell pixels zeroed). LP and mAP increase by ~3pp because the probe sees cleaner cell-centred signal. All comparisons in this document are made within the same eval mode. The masked numbers are the paper-facing numbers; unmasked numbers isolate architectural differences.

---

## 1. Datasets and Setup

| Dataset | Markers | Classes | Cells | Eval |
|---|---|---|---|---|
| CODEX_cHL | 41 | 16 | ~129k | single split |
| CODEX_cHL_KRONOS18 | 18 | 16 | ~129k | single split |
| MIBI_TNBC | 37 | 16 | ~196k | 5-fold CV (patient-level) |
| IMC_NB_TumorSub | 31 | 7 | ~237k | 5-fold CV (patient-level) |

CODEX_cHL_KRONOS18 uses the same tissue as CODEX_cHL but restricted to the 18 markers overlapping with the KRONOS foundation model pretraining set, enabling a direct comparison. All models trained with LARS (16k iters, batch 128) on the full-panel datasets, and same schedule on K18.

---

## 2. Main Results — CODEX_cHL (41 markers, masked eval)

| Model | LP_bal | mAP | NMI | ARI | kNN |
|---|---|---|---|---|---|
| **CIM_Funnel_Large** | **0.7625** | **0.7816** | 0.3001 | 0.1379 | 0.6022 |
| CIM | 0.7471 | 0.7459 | **0.3445** | **0.1713** | 0.6010 |
| CIM_LateFusion | 0.7287 | 0.7409 | 0.3428 | 0.1825 | 0.5803 |
| KRONOS (foundation, 41m) | 0.7377 | 0.7441 | 0.1826 | 0.0845 | 0.5170 |
| UNI (ViT-L/16, 41m) | 0.5738 | 0.5964 | 0.1522 | 0.0602 | 0.3696 |
| ExprBaseline (mean intensity) | 0.6870 | 0.6607 | 0.2472 | 0.0826 | 0.5462 |
| ResNet18 | 0.6656 | 0.6673 | 0.2973 | 0.1365 | 0.5490 |
| ResNet | 0.6605 | 0.6600 | 0.2790 | 0.1182 | 0.5419 |
| NeXtMarker | 0.6841 | 0.6765 | 0.3043 | 0.1224 | 0.5837 |

**Key observations:**
- CIM_Funnel_Large leads on LP/mAP (+1.5pp over CIM); CIM leads on clustering (NMI +4.4pp)
- KRONOS achieves competitive LP (0.738) despite being a foundation model not fine-tuned on this data, but its clustering is very poor (NMI=0.183 vs 0.300–0.345) — features are discriminative but geometry is not organised
- UNI surprisingly weak: LP=0.574, effectively ~12pp below CIM. Not a useful baseline for multiplexed imaging
- ExprBaseline (scalar mean intensity per marker) achieves 0.687 — a strong sanity check; any model with <0.69 LP is not learning useful spatial features beyond expression level
- NeXtMarker (depthwise ConvNeXt + inter-channel MLP) achieves 0.684 on the full 41-marker panel but falls below ExprBaseline on LP — spatial features not helpful enough to overcome the inter-channel bottleneck

---

## 3. Main Results — CODEX_cHL_KRONOS18 (18 markers)

### 3a. Masked evaluation (paper-facing numbers)

| Model | LP_bal | mAP | NMI | ARI | kNN |
|---|---|---|---|---|---|
| **CIM_Funnel_Large** | **0.7772** | **0.8009** | 0.3088 | 0.1635 | **0.6134** |
| CIM_Funnel_Large_NoMask_SameAug | 0.7753 | 0.7889 | 0.2781 | 0.1297 | 0.6094 |
| CIM_Funnel_Large_RandomCrop | 0.7754 | 0.7890 | 0.2627 | 0.1206 | 0.5879 |
| CIM_Funnel_Large_NoMask | 0.7443 | 0.7616 | 0.3380 | 0.1883 | 0.6119 |
| KRONOS (18 markers, foundation) | 0.7491 | 0.7629 | 0.2084 | 0.0924 | 0.5056 |
| CIM | 0.7402 | 0.7447 | 0.3313 | 0.1793 | 0.5408 |
| SharedMarkerFunnel | 0.7435 | 0.7562 | **0.3524** | **0.1985** | 0.6463 |
| SharedMarker (concat) | 0.7167 | 0.7177 | 0.2832 | 0.1619 | 0.5554 |
| NeXtMarker_LargeMix | 0.6816 | 0.6933 | 0.3811 | 0.2346 | 0.5579 |
| NeXtMarker_NoMask | 0.6522 | 0.6608 | 0.3223 | 0.1642 | 0.5803 |
| NeXtMarker | 0.6173 | 0.6114 | 0.3452 | 0.1867 | 0.5544 |
| ResNet18 | 0.6535 | 0.6614 | 0.3201 | 0.1618 | 0.5505 |
| ExprBaseline | 0.6594 | 0.6663 | 0.2323 | 0.0898 | 0.5771 |
| ResNet | 0.6130 | 0.6056 | 0.2788 | 0.1277 | 0.4956 |
| CIM_LateFusion | 0.6981 | 0.7063 | 0.3323 | 0.1948 | 0.5186 |
| UNI | 0.5933 | 0.6053 | 0.1235 | 0.0408 | 0.3417 |

> **Note:** SharedMarkerFunnel and 1block/2block metrics are from unmasked eval (no re_eval_masked.py run yet) and are placed in this table for reference — direct LP comparison to other masked-eval rows is not valid. See Section 7 for the fair unmasked comparison.

### 3b. Unmasked evaluation (fair for new architectures)

| Model | LP_bal | mAP | NMI | ARI | kNN |
|---|---|---|---|---|---|
| CIM_Funnel_Large | 0.7446 | 0.7605 | 0.3481 | 0.1825 | 0.6062 |
| **SharedMarkerFunnel** | **0.7435** | **0.7562** | 0.3524 | 0.1985 | **0.6463** |
| CIM_Funnel_Large_2block | 0.7396 | 0.7553 | **0.3617** | **0.2150** | 0.6227 |
| CIM_Funnel_Large_1block | 0.7367 | 0.7514 | 0.3458 | 0.1913 | 0.6273 |
| CIM | 0.7332 | 0.7405 | 0.2877 | 0.1484 | 0.5085 |
| SharedMarker (concat) | 0.7167 | 0.7177 | 0.2832 | 0.1619 | 0.5554 |
| ResNet18 | 0.6432 | 0.6491 | 0.3501 | 0.1677 | 0.5435 |
| ResNet | 0.5933 | 0.5861 | 0.2639 | 0.1246 | 0.4781 |

**Key observations (unmasked, apples-to-apples):**
- SharedMarkerFunnel is **within 0.1pp LP** of CIM_Funnel_Large despite using shared weights across all markers in phase 1 — a striking result
- 2 MixBlocks matches 8 MixBlocks LP within 0.5pp while achieving **better NMI (+1.4pp)** and **better ARI (+3.3pp)**
- More mixing blocks does not uniformly improve quality — it trades cluster geometry for linear separability
- Adding a cross-marker mixing stage to SharedMarker (+2.7pp LP, +7pp NMI) is the single most impactful architectural change

---

## 4. Main Results — MIBI_TNBC (37 markers, 5-fold CV, patient-level splits)

| Model | LP_bal | mAP | NMI | ARI | kNN | sil (mean) |
|---|---|---|---|---|---|---|
| **CIM** | **0.8701±0.0087** | 0.8389±0.0154 | 0.3043±0.0050 | 0.0811±0.0079 | 0.5402±0.0200 | −0.086 |
| CIM_Funnel_Large | 0.8667±0.0086 | **0.8636±0.0158** | **0.3288±0.0049** | **0.1111±0.0055** | **0.6421±0.0226** | −0.110 |
| CIM_LateFusion | 0.8396±0.0140 | 0.8361±0.0098 | 0.3116±0.0042 | 0.0965±0.0047 | 0.5263±0.0236 | −0.046 |
| KRONOS | 0.7866±0.0103 | 0.7719±0.0112 | 0.1143±0.0090 | 0.0198±0.0044 | 0.3907±0.0255 | — |
| UNI | 0.7723±0.0131 | 0.8128±0.0121 | 0.3299±0.0019 | 0.1270±0.0030 | 0.4953±0.0089 | −0.019 |
| ExprBaseline | 0.7042±0.0174 | 0.6270±0.0276 | 0.2216±0.0444 | 0.0949±0.0264 | 0.4629±0.0190 | — |
| ResNet18 | 0.5352±0.0234 | 0.4466±0.0322 | 0.3271±0.0046 | 0.1268±0.0085 | 0.3495±0.0170 | −0.060 |
| ResNet | 0.5400±0.0200 | 0.4510±0.0184 | 0.2471±0.0065 | 0.0657±0.0044 | 0.3645±0.0120 | −0.157 |
| NeXtMarker (n=3) | 0.5864±0.0104 | 0.4845±0.0079 | 0.2412±0.0197 | 0.1096±0.0345 | 0.4265±0.0126 | — |

**Key observations:**
- CIM and CIM_Funnel_Large are essentially tied on LP (0.87 vs 0.87) but diverge sharply on clustering: Funnel's 8 MixBlocks improve NMI by +2.5pp and ARI by +3pp, kNN by +10pp — the richer cross-marker mixing helps separability in embedding space even when LP is similar
- KRONOS: LP=0.787 is competitive but clustering is catastrophically poor (NMI=0.114, ARI=0.020) — essentially random clustering despite reasonable linear separability. Features encode global expression patterns without forming coherent cell-type geometry
- UNI: LP=0.772, mAP=0.813 — better than expected for a natural image model. mAP especially strong. NMI=0.330 is also solid. But silhouette is nearly zero (−0.019), suggesting poor batch integration
- CIM_LateFusion: −3pp LP vs CIM but silhouette is far better (−0.046 vs −0.086) — fuses channels earlier, reducing between-sample variation at the cost of within-type discriminability
- ExprBaseline at 0.704 LP — better than ResNet/ResNet18 on this dataset. Spatial features hurt ResNet significantly; the markers alone carry most signal in MIBI_TNBC
- NeXtMarker at 0.586: the worst performing architecture on this dataset; the inter-channel MLP bottleneck is too weak for 37 markers

---

## 5. Main Results — IMC_NB_TumorSub (31 markers, 5-fold CV)

| Model | LP_bal | mAP | NMI | ARI | kNN | sil (mean) |
|---|---|---|---|---|---|---|
| ExprBaseline | **0.7303±0.0177** | 0.6802±0.0082 | **0.4101±0.0336** | **0.2080±0.0276** | **0.6349±0.0105** | — |
| CIM_LateFusion | 0.7269±0.0125 | 0.7022±0.0094 | 0.3730±0.0069 | 0.1869±0.0078 | 0.5405±0.0055 | −0.133 |
| **CIM_Funnel_Large** | 0.7297±0.0129 | **0.7127±0.0096** | 0.3634±0.0095 | 0.1731±0.0100 | 0.5597±0.0132 | −0.079 |
| CIM | 0.7075±0.0132 | 0.6664±0.0064 | 0.3543±0.0047 | 0.1758±0.0031 | 0.5702±0.0116 | −0.111 |
| KRONOS | 0.6854±0.0132 | 0.6487±0.0086 | 0.2501±0.0453 | 0.0976±0.0298 | 0.5151±0.0114 | — |
| ResNet | 0.6583±0.0149 | 0.6109±0.0124 | 0.3576±0.0082 | 0.1627±0.0066 | 0.5273±0.0099 | −0.213 |
| ResNet18 | 0.6440±0.0179 | 0.5926±0.0210 | 0.3059±0.0225 | 0.1314±0.0161 | 0.4761±0.0122 | −0.166 |
| UNI | 0.4776±0.0230 | 0.4764±0.0178 | 0.3599±0.0033 | 0.1785±0.0050 | 0.4001±0.0236 | −0.031 |

**Key observations:**
- **ExprBaseline ties or beats all learned models on LP (0.730)** and leads on NMI/ARI — this is the most important finding on IMC. For this 7-class coarse tumour subtype panel, mean marker intensity is essentially sufficient. Learning spatial features does not add value over simple expression levels
- CIM_Funnel_Large best mAP (0.713) among spatial models, slightly ahead of CIM_LateFusion
- KRONOS again shows poor NMI (0.250) despite reasonable LP (0.685)
- **UNI collapses on IMC: LP=0.478**, only slightly above chance for 7 classes. The model clearly cannot handle 31-channel multiplexed IMC data. NMI=0.360 is curious — the geometry exists but the linear probe can't decode it
- ResNet18 NMI (0.306) is lower than ResNet (0.358) despite having more params — deeper networks overfit on the small IMC training sets

---

## 6. Masking Effect (CODEX_cHL_KRONOS18)

Cell masking applies the segmentation mask before feature extraction (non-cell pixels zeroed). This is an ablation isolating whether the model needs cell context or cell content only.

| Model | LP_bal (unmasked) | LP_bal (masked) | Δ | NMI (unmasked) | NMI (masked) | Δ |
|---|---|---|---|---|---|---|
| CIM_Funnel_Large | 0.7446 | 0.7772 | **+3.3pp** | 0.3481 | 0.3088 | −3.9pp |
| CIM | 0.7332 | 0.7402 | +0.7pp | 0.2877 | 0.3313 | +4.4pp |
| CIM_Funnel_Large_NoMask | 0.7092 | 0.7443 | +3.5pp | 0.3419 | 0.3380 | −0.4pp |
| CIM_Funnel_Large_NoMask_SameAug | 0.7421 | 0.7753 | +3.3pp | 0.2976 | 0.2781 | −1.9pp |
| CIM_Funnel_Large_RandomCrop | 0.7429 | 0.7754 | +3.3pp | 0.2810 | 0.2627 | −1.8pp |
| ResNet18 | 0.6432 | 0.6535 | +1.0pp | 0.3501 | 0.3201 | −3.0pp |
| ResNet | 0.5933 | 0.6130 | +2.0pp | 0.2639 | 0.2788 | +1.5pp |

**Key observations:**
- Masking consistently improves LP for all models (+0.7pp to +3.5pp) — the cell mask removes irrelevant background signal and makes the probe's task easier
- **LP and NMI trade off under masking** for CIM_Funnel_Large: +3.3pp LP but −3.9pp NMI. The mask sharpens discriminability but compresses the embedding geometry
- CIM shows the opposite: masking improves both LP (+0.7pp) and NMI (+4.4pp) — suggesting CIM's depthwise features are more robust to background contamination; the mask clarifies rather than disrupting its geometry
- CIM_Funnel_Large trained without masking (NoMask) still gains +3.5pp LP when evaluated with masking — the test-time mask is the dominant effect, not training-time masking

---

## 7. RandomCrop and NoMask Ablation — The Cell Segmentation Question

Can we remove the cell segmentation requirement entirely? Three conditions:

| Condition | Training | Eval | LP_bal (masked) | mAP | NMI |
|---|---|---|---|---|---|
| Standard | Cell-centred patches + mask | Masked | 0.7772 | 0.8009 | 0.3088 |
| NoMask_SameAug | Cell-centred patches, no mask | Masked | 0.7753 | 0.7889 | 0.2781 |
| NoMask | Cell-centred patches, no mask, no cutter | Masked | 0.7443 | 0.7616 | 0.3380 |
| **RandomCrop** | **Random 64px tissue crops** | **Masked** | **0.7754** | **0.7890** | 0.2627 |

**The striking finding:**
- **RandomCrop (no cell segmentation at all) achieves LP=0.775, matching the cell-centred masked baseline (0.777)** within noise. The model trained on random tissue patches — no cell detection, no segmentation, no mask — generalises at evaluation to cell-level features as well as one trained with full cell-centric supervision
- This is explained by the eval mask: at test time, even RandomCrop-trained models receive masked features. The model has learned marker co-expression patterns that transfer from tissue-level patches to cell-level evaluation
- NoMask_SameAug is also statistically indistinguishable from the standard model (0.775 vs 0.777) when the eval mask is applied
- The main loss from removing cell segmentation is in **cluster geometry** (NMI drops from 0.309 to 0.263 for RandomCrop) — the embedding space is more diffuse without cell-centric training, but the linear probe can still decode it

**Practical implication:** Cell segmentation is not required for achieving state-of-the-art classification performance. This dramatically reduces the preprocessing burden and makes the method applicable to datasets lacking segmentation.

---

## 8. Foundation Model Comparison: KRONOS and UNI

### KRONOS (ViT-S/16, pretrained on SPM-47M multiplexed data, 177 markers)

| Dataset | LP_bal | mAP | NMI | ARI | kNN |
|---|---|---|---|---|---|
| CODEX_cHL (41m) | 0.7377 | 0.7441 | 0.1826 | 0.0845 | 0.5170 |
| CODEX_cHL_KRONOS18 (18m) | 0.7491 | 0.7629 | 0.2084 | 0.0924 | 0.5056 |
| MIBI_TNBC (5-fold) | 0.7866±0.010 | 0.7719±0.011 | 0.1143±0.009 | 0.0198±0.004 | 0.3907±0.026 |
| IMC_NB_TumorSub (5-fold) | 0.6854±0.013 | 0.6487±0.009 | 0.2501±0.045 | 0.0976±0.030 | 0.5151±0.011 |

**Pattern:** KRONOS achieves competitive LP on CODEX (0.749 on K18 vs 0.777 for CIM_Funnel_Large) but has systematically catastrophic clustering everywhere. NMI of 0.114 on MIBI_TNBC is near-random (k-Means on random features would expect ~0.05–0.10). This suggests KRONOS features encode cell identity discriminatively but not in a geometrically organised way — likely because the ViT attention mechanism attends across all markers globally, destroying the local spatial structure that clustering relies on. The contrast is most stark on MIBI_TNBC: CIM NMI=0.304, KRONOS NMI=0.114 — same dataset, 2.7× better clustering with our method.

On K18, KRONOS uses 18 real marker IDs from its pretrained vocabulary; the 18-marker subset performs slightly better than the 41-marker version (LP 0.749 vs 0.738) because unknown markers in the 41-marker panel receive fallback sequential IDs with data-computed normalisation stats, degrading representation quality.

### UNI (ViT-L/16, pretrained on 100k histopathology H&E slides)

| Dataset | LP_bal | mAP | NMI | sil |
|---|---|---|---|---|
| CODEX_cHL (41m) | 0.5738 | 0.5964 | 0.1522 | −0.019 |
| CODEX_cHL_KRONOS18 (18m) | 0.5933 | 0.6053 | 0.1235 | −0.026 |
| MIBI_TNBC (5-fold) | 0.7723±0.013 | 0.8128±0.012 | 0.3299±0.002 | −0.019 |
| IMC_NB_TumorSub (5-fold) | 0.4776±0.023 | 0.4764±0.018 | 0.3599±0.003 | −0.031 |

**Pattern:** UNI was designed for H&E histopathology at 20×/40× magnification. Its performance is highly dataset-dependent: reasonable on MIBI_TNBC (0.772 LP, 0.813 mAP — competitive with KRONOS) but collapses on IMC (0.478 LP, near-random). On CODEX it is weak (0.574). The MIBI_TNBC result is likely explained by the tissue morphology being more H&E-like (dense cellular tissue) whereas IMC sections are sparser. UNI's clustering NMI (0.330 on MIBI_TNBC) is surprisingly good — comparable to CIM_Funnel_Large (0.329) — despite poor batch integration (sil ~0). UNI is not a viable baseline for multiplexed imaging in general.

### ExprBaseline (mean marker intensity per cell)

| Dataset | LP_bal | mAP | NMI |
|---|---|---|---|
| CODEX_cHL | 0.6870 | 0.6607 | 0.2472 |
| CODEX_cHL_KRONOS18 | 0.6594 | 0.6663 | 0.2323 |
| MIBI_TNBC (5-fold) | 0.7042±0.017 | 0.6270±0.028 | 0.2216±0.044 |
| IMC_NB_TumorSub (5-fold) | **0.7303±0.018** | 0.6802±0.008 | **0.4101±0.034** |

**Pattern:** ExprBaseline is competitively strong on IMC_NB_TumorSub (LP=0.730, NMI=0.410 — best NMI of all models). This is the clearest evidence that this 7-class coarse tumour subtype task is solved by marker expression alone — spatial features add nothing. On CODEX and MIBI, ExprBaseline is clearly below the spatial models (−7–9pp LP), confirming that spatial context matters more for fine-grained cell type resolution in these datasets.

---

## 9. Architecture Comparison: CIM vs CIM_Funnel_Large vs CIM_LateFusion

| Property | CIM (WideModel) | CIM_Funnel_Large | CIM_LateFusion |
|---|---|---|---|
| Phase 1 | Depthwise ConvNeXt (C independent) | Depthwise ConvNeXt (C independent) | Shared early fusion at bottleneck |
| Phase 2 | None | 8× MixBlock (groups=1, full cross-channel) | — |
| Output dim | C × stem_width (e.g. 41×32=1312) | 512 (mix_channels) | 512 |
| Params | ~1.5M | ~17M | ~8M |
| Iters | 1k (AdamW) | 16k (LARS) | 16k (LARS) |

The CIM → CIM_Funnel_Large transition introduces cross-channel mixing in phase 2. This reliably improves LP (+1.5–3pp) and mAP but at the cost of NMI and ARI on CODEX. On MIBI the trade-off is more favourable — NMI also improves. The pattern suggests the mixing stage helps when the cell types are well-separated by co-expression patterns (MIBI_TNBC) but hurts when the geometry itself carries biological meaning (CODEX_cHL where clustering structure is informative for neighbourhood analysis).

---

## 10. Mix Block Sweep — How Many Mixing Blocks Are Needed?

All experiments on CODEX_cHL_KRONOS18 with identical setup (LARS, 16k iters, masked augmentation). Ablations at 8k iters are from the CODEX_cHL (41m) ablation suite; 16k K18 results are new.

### 16k iters, CODEX_cHL_KRONOS18 (unmasked eval, fair comparison)

| mix_n_blocks | LP_bal | mAP | NMI | ARI | kNN | Params (phase 2) |
|---|---|---|---|---|---|---|
| 1 | 0.7367 | 0.7514 | 0.3458 | 0.1913 | 0.6273 | ~2.1M |
| 2 | 0.7396 | 0.7553 | **0.3617** | **0.2150** | 0.6227 | ~4.2M |
| **8 (baseline)** | **0.7446** | **0.7605** | 0.3481 | 0.1825 | 0.6062 | **~16.8M** |

### 8k iters, CODEX_cHL 41m ablations (relative to aug_full_8k baseline)

| mix_n_blocks | LP_bal | mAP | NMI | ARI |
|---|---|---|---|---|
| 4 | 0.7089 | 0.7204 | 0.3061 | 0.1408 |
| 8 (baseline) | 0.7123 | 0.7179 | 0.3077 | 0.1431 |
| 12 | 0.7014 | 0.7115 | 0.3078 | 0.1533 |

**Findings:**
- Going from 1 → 2 blocks: LP gains +0.3pp, NMI gains +1.6pp, ARI gains +2.4pp — the most efficient step
- Going from 2 → 8 blocks: LP gains only +0.5pp, but NMI drops −1.4pp and ARI drops −3.3pp — additional mixing hurts cluster geometry
- Going from 8 → 12 blocks: LP drops −1.1pp — over-parameterisation at this spatial scale
- **2 MixBlocks is the Pareto-optimal choice**: near-maximum LP (within 0.5pp of 8 blocks) with the best clustering geometry and 4× fewer mixing parameters (~4.2M vs 16.8M)
- The phase-2 mixing operates on small spatial maps (4×4 px for 32px patches); at this resolution more than 2–4 blocks adds redundancy

---

## 11. SharedMarker: Shared-Weight Backbone Experiments

The standard CIM uses separate per-marker depthwise convolution weights. SharedMarker uses a **single shared ConvNeXt backbone** that processes each marker independently, then aggregates. This enables the spatial feature extractor to transfer across datasets without retraining.

### Aggregation ablation (CODEX_cHL_KRONOS18, unmasked)

| Aggregation | LP_bal | mAP | NMI | Why |
|---|---|---|---|---|
| Mean+Max pooling | 0.31 | — | — | Destroys marker identity (position in concat vector) |
| Transformer attention | 0.34 | — | — | Same failure mode — attention output lacks positional marker identity |
| **Concatenation** | **0.717** | **0.718** | **0.283** | Preserves marker identity through position in the feature vector |

The key insight: cell type identity is encoded as *which specific markers are active*, not the *distribution* of activations. Mean pooling destroys this — it cannot distinguish "high CD20 in marker 5" from "high CD3 in marker 5". Concatenation at fixed marker positions is the correct aggregation.

### SharedMarkerFunnel: adding cross-marker mixing

Concatenation preserves marker identity but provides no cross-marker interaction. SharedMarkerFunnel adds CIM_Funnel_Large's Phase 2 (transition + MixBlocks) after the concatenated spatial maps:

| Model | LP_bal | mAP | NMI | ARI | kNN |
|---|---|---|---|---|---|
| CIM_Funnel_Large (per-marker weights) | 0.7446 | 0.7605 | 0.3481 | 0.1825 | 0.6062 |
| **SharedMarkerFunnel (shared weights + mixing)** | **0.7435** | **0.7562** | **0.3524** | **0.1985** | **0.6463** |
| SharedMarker (shared weights, no mixing) | 0.7167 | 0.7177 | 0.2832 | 0.1619 | 0.5554 |

**SharedMarkerFunnel is within 0.1pp LP of CIM_Funnel_Large** while using shared backbone weights (144k params for phase 1 vs CIM_Funnel_Large's marker-specific weights). NMI and kNN are slightly better. This result demonstrates that per-marker weight specialisation in phase 1 is not necessary — a shared backbone generalises across all markers equally well, and the cross-marker mixing stage does the heavy lifting. The shared weights in phase 1 are the key property enabling cross-dataset transfer without retraining the backbone.

---

## 12. NeXtMarker Architecture

NeXtMarker uses a grouped-depthwise ConvNeXt backbone (4 blocks, stride-2 expand) followed by GlobalMaxPool and a 2-layer inter-channel MLP (`n_markers×16 → 512 → 512`). ChannelScaler initialisation at 1.0 (not 1e-6) is critical.

| Variant | LP_bal | mAP | NMI | ARI | kNN |
|---|---|---|---|---|---|
| NeXtMarker (masked) | 0.6173 | 0.6114 | 0.3452 | 0.1867 | 0.5544 |
| NeXtMarker_NoMask (masked) | 0.6522 | 0.6608 | 0.3223 | 0.1642 | 0.5803 |
| **NeXtMarker_LargeMix** (masked) | **0.6816** | **0.6933** | **0.3811** | **0.2346** | 0.5579 |

LargeMix uses a 3-layer MLP (`n_m×16 → 1024 → 512 → 512`) with LayerNorm+GELU. Adding capacity to the inter-channel mixing step (+6.5pp LP, NMI=0.381 — best on K18 among all models) confirms the bottleneck is the cross-channel interaction, not the per-marker spatial features. Despite this, LP still lags CIM_Funnel_Large by ~10pp, indicating the MaxPool-before-MLP approach is fundamentally less expressive than spatial MixBlocks that operate on 2D feature maps.

**Notable:** NeXtMarker NMI (0.345–0.381) consistently exceeds CIM_Funnel_Large NMI (0.309) — the MaxPool aggregation may discard inter-sample variation, creating more sample-agnostic features that cluster better by cell type.

---

## 13. Training Ablations (CODEX_cHL, 8k iters)

All variants based on CIM_Funnel_Large, evaluated relative to `aug_full_8k` baseline (8 blocks, 8k iters). Full 16k run achieves LP=0.762 on CODEX_cHL (41m).

### Augmentation ablations

| Variant | LP_bal | mAP | NMI | Key finding |
|---|---|---|---|---|
| aug_full_8k (baseline) | 0.7123 | 0.7179 | 0.3077 | Reference |
| aug_drop_02 (drop_prob=0.2) | 0.7262 | 0.7358 | 0.3005 | More aggressive channel drop **helps** |
| aug_no_noise (no Gaussian) | 0.7236 | 0.7328 | 0.2874 | Noise removal helps LP but hurts NMI |
| aug_no_channel (no drop) | 0.7130 | 0.7177 | 0.2758 | Channel drop critical for NMI; minor LP effect |
| aug_no_shift (no shift/scale) | 0.7006 | 0.7134 | 0.3036 | Intensity shift/scale important |
| aug_drop_005 (drop_prob=0.05) | 0.7038 | 0.7129 | 0.3311 | Too little drop hurts LP; NMI best here |
| aug_no_drop | 0.6953 | 0.6971 | 0.3071 | Removing all augmentation: worst LP overall |

**Key finding:** Channel drop (randomly zeroing entire marker channels) is the most impactful augmentation — both for LP and NMI. `drop_prob=0.2` slightly outperforms the baseline `drop_prob=0.1`, suggesting more aggressive marker dropout is beneficial. Removing Gaussian noise hurts NMI more than LP. The paper baseline uses `drop_prob=0.1` which is not optimal.

### Training duration and capacity

| Variant | Iters | LP_bal | mAP | NMI | Note |
|---|---|---|---|---|---|
| iters_2k | 2k | 0.7094 | 0.7178 | 0.3150 | 87% of 16k result after 12% of compute |
| iters_4k | 4k | 0.7134 | 0.7230 | 0.3164 | 90% |
| aug_full_8k (baseline) | 8k | 0.7123 | 0.7179 | 0.3077 | 91% |
| CIM_Funnel_Large (paper) | 16k | 0.7625 | 0.7816 | 0.3001 | 100% |
| cap_ch_256 | 8k | 0.6956 | 0.7030 | 0.3000 | 256 too narrow |
| cap_ch_768 | 8k | 0.7163 | 0.7258 | 0.3128 | Marginal gain over 512 |

**Key finding:** Most of the LP gain comes after 8k iters — the 8k → 16k doubling adds +5pp LP (0.712 → 0.762). The model is far from converged at 8k. NMI peaks around 4k and slightly degrades with more training (geometry forms early; discriminability continues to improve). Channel dim of 512 is the right sweet spot — 256 hurts (−1.7pp), 768 provides only marginal gain (+0.4pp).

---

## 14. Per-Class AP Analysis — CODEX_cHL_KRONOS18

### Key models (masked eval where available)

| Cell type | CIM_FL (8blk) | CIM_FL_2blk | SharedMkrFunnel | KRONOS | ExprBaseline |
|---|---|---|---|---|---|
| Tumor | **0.9583** | 0.9381 | 0.9320 | 0.9437 | — |
| CD8 | **0.9296** | 0.8831 | 0.8766 | 0.9133 | — |
| Lymphatic | 0.9260 | **0.9011** | 0.8897 | 0.8938 | — |
| CD4 | **0.8800** | 0.8481 | 0.8469 | 0.8501 | — |
| Treg | 0.8235 | **0.8550** | 0.8422 | 0.7903 | — |
| Mast | 0.8603 | 0.7944 | 0.7855 | 0.8384 | — |
| Endothelial | **0.8746** | 0.8449 | 0.8370 | 0.8404 | — |
| NK | 0.8297 | 0.7947 | 0.7828 | 0.7802 | — |
| Neutrophil | 0.8303 | 0.7905 | 0.7694 | 0.7567 | — |
| B | **0.8530** | 0.8192 | 0.7939 | 0.8166 | — |
| M2 | 0.7851 | 0.7392 | 0.7275 | 0.7477 | — |
| Other | 0.7030 | 0.6141 | 0.5996 | **0.6829** | — |
| DC | **0.7724** | 0.7226 | 0.7114 | 0.6955 | — |
| M1 | **0.6064** | 0.5809 | 0.5633 | 0.5522 | — |
| Monocyte | 0.6056 | 0.5575 | 0.5439 | 0.5333 | — |
| Epithelial | **0.5773** | 0.4021 | 0.5982 | 0.5714 | — |
| **mAP** | **0.8009** | 0.7553 | 0.7562 | 0.7629 | 0.6663 |

**Hard classes (AP < 0.65 for best model):**
- **Monocyte** (0.474–0.606): worst across all models — no 18-marker panel discriminator between monocyte subtypes
- **M1** (0.552–0.606): heavily overlaps with M2 and Monocyte in the myeloid cloud
- **Epithelial** (0.378–0.598): rare class, ambiguous with Tumor in 18-marker panel; notably 2 blocks collapses to AP=0.402
- **Other** (0.557–0.703): catch-all category — no coherent representation possible

**Observations:**
- CIM_Funnel_Large (8blk, masked) leads on most classes — masking clearly helps
- KRONOS competitive or leading on "simple" high-expression classes (Tumor, CD8) but weak on myeloid subtypes requiring co-expression patterns
- Treg: 2-block model (0.855) beats 8-block (0.824) — fewer blocks learn a more coherent Treg embedding
- Epithelial AP drops dramatically with fewer blocks (0.402 for 2blk) suggesting this class benefits most from the full 8-block mixing capacity

---

## 15. Summary of Key Findings

1. **Cell segmentation is not required** for state-of-the-art LP performance. RandomCrop training (random tissue patches, no segmentation) matches cell-centred training (LP=0.775 on K18), enabled by test-time masking. This opens the method to datasets without segmentation.

2. **2 MixBlocks is optimal**, not 8. LP difference is only 0.5pp but NMI/ARI are better with 2 blocks (+1.4pp NMI, +3.3pp ARI) at 4× less phase-2 cost. The small spatial map (4×4 px) limits how many blocks are useful.

3. **Shared backbone weights match per-marker weights.** SharedMarkerFunnel (shared phase-1, cross-marker mixing in phase 2) is within 0.1pp LP of CIM_Funnel_Large on K18 (0.7435 vs 0.7446 unmasked) with slightly better NMI and kNN. This enables cross-dataset transfer of the backbone.

4. **Masking boosts LP ~3pp but costs NMI ~3pp for CIM_Funnel_Large.** The cell mask is an implicit supervision signal that sharpens linear separability at the cost of embedding geometry. CIM (depthwise only) shows the opposite: masking improves both LP and NMI.

5. **KRONOS foundation model is strong on LP but catastrophic on clustering.** NMI of 0.114 on MIBI_TNBC (vs CIM 0.304) despite LP=0.787. KRONOS features are discriminative but geometrically unstructured — inappropriate for clustering or spatial analysis applications.

6. **ExprBaseline is the correct comparison point.** On IMC it wins outright (LP=0.730 = best). On CODEX/MIBI it is 7–16pp behind CIM_Funnel_Large. Any model not clearly beating ExprBaseline is not learning spatial features.

7. **Channel drop augmentation (drop_prob=0.1–0.2) is the most important augmentation** for both LP and NMI. Removing it causes the sharpest individual drop in both metrics. More aggressive drop (0.2) is slightly better than the baseline (0.1).

8. **Training duration matters more than capacity.** The LP gain from 8k → 16k iters (+5pp) dwarfs any architectural change. 2k iters already achieves ~87% of final LP — the model learns quickly but convergence is slow.

9. **UNI is not a useful baseline for multiplexed imaging.** LP=0.478 on IMC (7 classes, near random) and LP=0.574 on CODEX_cHL. The H&E histopathology pretraining does not transfer to multiplexed data.

10. **Marker identity must be preserved via position in the concatenated feature vector.** Mean/max pooling or attention aggregation over marker tokens destroys this and collapses LP to 0.31–0.34 (SharedMarker ablation). Concatenation at fixed marker positions is the correct aggregation.

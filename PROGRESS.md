# Progress Log

## 2026-02-18

- Refactored augmentation configs: replaced `augmentations.py` and `augmentations_virtualpatients.py` with three named strength files (`augmentations_high.py`, `augmentations_low.py`, `augmentations_none.py`), all using consistent variable names `train_aug_strong` / `train_aug_weak`. Updated all 15 experiment files accordingly.



- Created `configs/_experiments_/CODEX_cHL/CIMATT_VICReg_VP.py` — experiment config combining the CIMATT backbone (WideModel + Attention) with the VICReg algorithm on CODEX_cHL with virtual-patient augmentations.

"""
Generate 5-fold patient-level CV splits (train / test) for a dataset.

For each fold:
  - test  : 1/5 of patients (held out, never seen during training)
  - train : 4/5 of patients (SSL training + LP fitting)

Output structure:
  <out_dir>/split_<k>/train.txt
  <out_dir>/split_<k>/test.txt

Each file contains cell indices (0-based integers matching h5 row order),
one per line — same format as existing MIBI_TNBC CV splits.

Usage:
  python tools/generate_cv_splits.py \
      --h5   /path/to/dataset.h5 \
      --out  /path/to/cv_splits \
      --n_folds 5 \
      --seed 42 \
      [--patient_key sample_id]   # key inside h5['coords']
      [--patient_sep "-TU-"]      # split sample_id to get patient ID (optional)
"""

import argparse
import h5py
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--h5',          required=True)
parser.add_argument('--out',         required=True)
parser.add_argument('--n_folds',     type=int, default=5)
parser.add_argument('--seed',        type=int, default=42)
parser.add_argument('--patient_key', default='sample_id')
parser.add_argument('--patient_sep', default=None,
                    help='If set, split sample_id on this string and use the '
                         'left part as the patient ID (e.g. "-TU-")')
args = parser.parse_args()

np.random.seed(args.seed)

# ── Load sample IDs ───────────────────────────────────────────────────────────
with h5py.File(args.h5, 'r') as f:
    raw_ids = f['coords'][args.patient_key][:].astype(str)

n_cells = len(raw_ids)

if args.patient_sep:
    patient_ids = np.array([s.split(args.patient_sep)[0] for s in raw_ids])
else:
    patient_ids = raw_ids

unique_patients = np.array(sorted(set(patient_ids)))
n_patients = len(unique_patients)
print(f'{n_patients} patients, {n_cells} cells')

# ── Shuffle patients ──────────────────────────────────────────────────────────
perm = np.random.permutation(n_patients)
shuffled = unique_patients[perm]

# ── Build folds ───────────────────────────────────────────────────────────────
folds = np.array_split(shuffled, args.n_folds)

out_root = Path(args.out)
out_root.mkdir(parents=True, exist_ok=True)

for k in range(args.n_folds):
    test_patients  = folds[k]
    train_patients = np.concatenate([folds[i] for i in range(args.n_folds) if i != k])

    test_idx  = np.where(np.isin(patient_ids, test_patients))[0]
    train_idx = np.where(np.isin(patient_ids, train_patients))[0]

    split_dir = out_root / f'split_{k}'
    split_dir.mkdir(exist_ok=True)

    for name, idx in [('train', train_idx), ('test', test_idx)]:
        path = split_dir / f'{name}.txt'
        np.savetxt(path, np.sort(idx), fmt='%d')
        print(f'  split_{k}/{name}.txt: {len(idx)} cells '
              f'({len(np.unique(patient_ids[idx]))} patients)')

    print()

print(f'Saved {args.n_folds} splits to {out_root}')

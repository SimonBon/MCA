"""
Create sample-level cross-validation splits for MIBI_TNBC.

For each split, 80% of samples are used for training and 20% for validation.
Splits are stratified at the sample level (all cells of a sample stay together).
Cell-level index files (positions in the H5 coordinate arrays) are written for
each split so they can be passed directly as `used_indicies` in experiment configs.

Usage (run on server):
    python scripts/create_mibi_tnbc_cv.py \
        --h5 /path/to/MIBI_TNBC.h5 \
        --output /path/to/MIBI_TNBC/cv_splits \
        --n_splits 5 \
        --val_frac 0.2 \
        --seed 42
"""

import argparse
import h5py
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Create MIBI_TNBC CV splits')
    parser.add_argument('--h5', required=True, help='Path to MIBI_TNBC.h5')
    parser.add_argument('--output', required=True, help='Directory to write split index files')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of random splits (default: 5)')
    parser.add_argument('--val_frac', type=float, default=0.2, help='Fraction of samples for validation (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()

    output = Path(args.output)

    print(f'Reading {args.h5} ...')
    with h5py.File(args.h5, 'r') as f:
        sample_ids = f['coords']['sample_id'][:].astype(str)

    n_cells = len(sample_ids)
    unique_samples = np.array(sorted(set(sample_ids)))
    n_samples = len(unique_samples)
    print(f'Found {n_cells} cells across {n_samples} samples:')
    for sid in unique_samples:
        n = (sample_ids == sid).sum()
        print(f'  {sid}: {n} cells')

    n_val = max(1, int(round(n_samples * args.val_frac)))
    n_train = n_samples - n_val
    print(f'\nCreating {args.n_splits} splits: {n_train} train / {n_val} val samples each')

    rng = np.random.default_rng(args.seed)

    for k in range(args.n_splits):
        split_dir = output / f'split_{k}'
        split_dir.mkdir(parents=True, exist_ok=True)

        shuffled = rng.permutation(unique_samples)
        val_samples = set(shuffled[:n_val])
        train_samples = set(shuffled[n_val:])

        train_idx = np.where(np.isin(sample_ids, list(train_samples)))[0]
        val_idx = np.where(np.isin(sample_ids, list(val_samples)))[0]

        np.savetxt(split_dir / 'train.txt', train_idx, fmt='%d')
        np.savetxt(split_dir / 'val.txt', val_idx, fmt='%d')

        print(f'\nSplit {k}:')
        print(f'  train: {len(train_samples)} samples, {len(train_idx)} cells')
        print(f'  val:   {len(val_samples)} samples ({sorted(val_samples)}), {len(val_idx)} cells')

    print(f'\nDone. Index files written to {output}')


if __name__ == '__main__':
    main()

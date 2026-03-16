#!/usr/bin/env python3
"""Patch cLISI/iLISI into existing ExprBaseline metrics.json files.

Loads val_results.npz (already saved by baseline_expression.py) and
computes cLISI / iLISI using the same method as EvaluateModelRich,
then writes the values back into metrics.json.

Usage:
    python tools/patch_lisi.py \
        /nobackup/.../paper_clean/CODEX_cHL_KRONOS18/ExprBaseline \
        /nobackup/.../paper_clean/MIBI_TNBC/ExprBaseline/fold_0 \
        ...
    # or glob:
    python tools/patch_lisi.py /nobackup/.../paper_clean/*/ExprBaseline \
        /nobackup/.../paper_clean/*/ExprBaseline/fold_*
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_lisi(features, labels, n_neighbors=90, metric='cosine'):
    """LISI score — same implementation as val_hook_rich.py._compute_lisi."""
    n = len(features)
    k = min(n_neighbors, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(features)
    _, indices = nbrs.kneighbors(features)

    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    label_idx = np.array([label_to_idx[l] for l in labels])

    lisi_scores = []
    for i in range(n):
        neigh_labels = label_idx[indices[i]]
        counts = np.bincount(neigh_labels, minlength=n_labels)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        # Simpson's diversity index
        lisi_scores.append(1.0 / np.sum(probs ** 2))

    return float(np.mean(lisi_scores))


def patch_dir(d):
    d = Path(d)
    npz_path     = d / 'val_results.npz'
    metrics_path = d / 'metrics.json'

    if not npz_path.exists():
        print(f'  SKIP (no val_results.npz): {d}')
        return
    if not metrics_path.exists():
        print(f'  SKIP (no metrics.json): {d}')
        return

    data       = np.load(npz_path, allow_pickle=True)
    features   = data['features']
    cell_labels = data['labels_str'].astype(str)
    sample_ids  = data['sample_ids'].astype(str)

    n_cell_types = len(np.unique(cell_labels))
    n_samples    = len(np.unique(sample_ids))

    if n_cell_types < 2 or n_samples < 2:
        print(f'  SKIP (too few labels/samples): {d}')
        return

    print(f'  Computing LISI for {d.name} ({len(features)} cells)...')
    clisi_raw = compute_lisi(features, cell_labels, n_neighbors=90)
    ilisi_raw = compute_lisi(features, sample_ids,  n_neighbors=90)

    # Normalise: (score - 1) / (n_unique - 1) → [0, 1]
    clisi_norm = (clisi_raw - 1) / max(n_cell_types - 1, 1)
    ilisi_norm = (ilisi_raw - 1) / max(n_samples    - 1, 1)

    metrics = json.load(open(metrics_path))
    metrics['clisi'] = {'raw': clisi_raw, 'normalised': clisi_norm,
                        'n_cell_types': n_cell_types}
    metrics['ilisi'] = {'raw': ilisi_raw, 'normalised': ilisi_norm,
                        'n_samples': n_samples}

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'    cLISI={clisi_norm:.4f}  iLISI={ilisi_norm:.4f}  -> patched {metrics_path.name}')


if __name__ == '__main__':
    dirs = sys.argv[1:]
    if not dirs:
        print('Usage: patch_lisi.py <dir> [<dir> ...]')
        sys.exit(1)
    for d in dirs:
        patch_dir(d)

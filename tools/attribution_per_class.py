#!/usr/bin/env python3
"""Mean IG attribution per ground-truth cell type.

For each cell type, computes mean row-normalised IG attribution across all
cells of that type, then plots a dot plot / heatmap showing which markers
are most important per class.

Usage:
    python tools/attribution_per_class.py \
        --attribution /nobackup/.../attribution.npz \
        --out         /nobackup/.../attribution_per_class.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--attribution', required=True)
    p.add_argument('--out',         required=True)
    p.add_argument('--top_n',       type=int, default=5,
                   help='Top N markers to highlight per class')
    return p.parse_args()


def main():
    args = parse_args()

    d            = np.load(args.attribution, allow_pickle=True)
    attr         = d['attribution'].astype(np.float32)
    marker_names = d['marker_names'].astype(str)
    gt_labels    = d['labels'].astype(str)

    # Drop DAPI
    dapi_mask    = np.array([m.upper() != 'DAPI-01' for m in marker_names])
    attr         = attr[:, dapi_mask]
    marker_names = marker_names[dapi_mask]

    # Row-normalise
    attr_norm = attr / (attr.sum(axis=1, keepdims=True) + 1e-8)

    classes = sorted(set(gt_labels))
    n_cls   = len(classes)
    n_mrk   = len(marker_names)

    # Mean attribution per class [n_cls, n_markers]
    mean_mat = np.zeros((n_cls, n_mrk))
    for ci, c in enumerate(classes):
        mask = gt_labels == c
        mean_mat[ci] = attr_norm[mask].mean(axis=0)

    # Z-score per marker across classes (highlights relative differences)
    mu      = mean_mat.mean(axis=0, keepdims=True)
    std     = mean_mat.std(axis=0,  keepdims=True) + 1e-8
    z_mat   = (mean_mat - mu) / std

    # Order markers by variance across classes (most discriminative first)
    marker_order = np.argsort(mean_mat.std(axis=0))[::-1]
    z_sorted     = z_mat[:, marker_order]
    names_sorted = marker_names[marker_order]

    # ── Figure ────────────────────────────────────────────────────────────
    fig_w = max(14, n_mrk * 0.45)
    fig_h = max(5,  n_cls * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(z_sorted, aspect='auto', cmap='RdBu_r',
                   vmin=-3, vmax=3, interpolation='nearest')

    # Mark top_n markers per class with a star
    for ci in range(n_cls):
        top_idx = np.argsort(mean_mat[ci])[::-1][:args.top_n]
        # Find their position in sorted order
        for idx in top_idx:
            pos = np.where(marker_order == idx)[0][0]
            ax.text(pos, ci, '★', ha='center', va='center',
                    fontsize=5, color='black', alpha=0.7)

    ax.set_xticks(range(n_mrk))
    ax.set_xticklabels(names_sorted, rotation=90, fontsize=7)
    ax.set_yticks(range(n_cls))
    ax.set_yticklabels(classes, fontsize=8)

    cb = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cb.set_label('Mean IG attribution\n(z-scored per marker)', fontsize=7)
    cb.ax.tick_params(labelsize=6)

    ax.set_title('Mean marker IG attribution per cell type\n'
                 '(★ = top 5 markers per class, markers ordered by discriminability)',
                 fontsize=9, pad=8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.4)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

    # Print top markers per class
    print('\n── Top markers per class ──')
    for ci, c in enumerate(classes):
        top_idx = np.argsort(mean_mat[ci])[::-1][:args.top_n]
        top     = ', '.join(f'{marker_names[i]} ({mean_mat[ci,i]:.3f})'
                            for i in top_idx)
        print(f'  {c:20s} {top}')


if __name__ == '__main__':
    main()

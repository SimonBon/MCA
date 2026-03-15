"""
UMAP coloured by sample ID with each sample's ID printed at its centroid.
Uses a maximally-distinct 40-colour palette so no two samples share a colour.

Usage:
    python tools/umap_samples_labelled.py [--model FOLDER]

Saves: <model_dir>/umap_samples_labelled.pdf
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

Z = Path('/nobackup/lab_taschner-mandl/simongutwein/z_RUNS')

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MIBI_TNBC_CIM_VICReg_Funnel_Large')
args = parser.parse_args()

OUT = Z / args.model / 'umap_samples_labelled.pdf'

umap_data  = np.load(Z / args.model / 'umap_embeddings.npz', allow_pickle=True)
val_data   = np.load(Z / args.model / 'val_results.npz',     allow_pickle=True)
xy         = umap_data['embedding']
sample_ids = val_data['sample_ids']

all_samples = sorted(set(sample_ids), key=lambda s: int(s) if s.isdigit() else s)
n = len(all_samples)

# Build a maximally-distinct palette by sampling HSV evenly
hues = np.linspace(0, 1, n, endpoint=False)
palette = [mcolors.hsv_to_rgb([h, 0.75, 0.85]) for h in hues]
colour_map = {s: palette[i] for i, s in enumerate(all_samples)}

fig, ax = plt.subplots(figsize=(12, 10))

for sid in all_samples:
    mask = sample_ids == sid
    ax.scatter(xy[mask, 0], xy[mask, 1],
               color=colour_map[sid], s=2, linewidths=0,
               rasterized=True, alpha=0.7)
    # label at centroid
    cx, cy = xy[mask, 0].mean(), xy[mask, 1].mean()
    ax.text(cx, cy, sid, fontsize=6, ha='center', va='center',
            fontweight='bold', color='black',
            bbox=dict(boxstyle='round,pad=0.1', fc=colour_map[sid],
                      ec='none', alpha=0.7))

x1, x99 = np.percentile(xy[:, 0], [1, 99]); pad_x = (x99 - x1) * 0.05
y1, y99 = np.percentile(xy[:, 1], [1, 99]); pad_y = (y99 - y1) * 0.05
ax.set_xlim(x1 - pad_x, x99 + pad_x)
ax.set_ylim(y1 - pad_y, y99 + pad_y)
ax.set_aspect('equal', adjustable='datalim')
ax.set_xlabel('UMAP 1', fontsize=10)
ax.set_ylabel('UMAP 2', fontsize=10)
ax.set_title(f'UMAP coloured by sample ID — {args.model}\n'
             f'({n} samples, IDs printed at centroid)', fontsize=11)
ax.tick_params(labelsize=8)

fig.savefig(OUT, bbox_inches='tight', dpi=200)
print(f'Saved: {OUT}')

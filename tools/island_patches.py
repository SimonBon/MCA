"""
Show example patches for cells in sample-specific UMAP islands and the
EGFR-bright T-cell area, fetched directly from H5 by (sample_id, cell_type).

Usage:
    python tools/island_patches.py [--model FOLDER]

Saves: <model_dir>/island_patches.pdf
"""

import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import DBSCAN

Z            = Path('/nobackup/lab_taschner-mandl/simongutwein/z_RUNS')
H5           = Path('/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/MIBI_TNBC.h5')
MARKERS_FILE = Path('/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/used_markers.txt')
VAL_IDX      = Path('/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/val.txt')

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MIBI_TNBC_CIM_VICReg_Funnel_Large')
parser.add_argument('--patch_size', type=int, default=32)
parser.add_argument('--n_patches',  type=int, default=8)
args = parser.parse_args()

OUT  = Z / args.model / 'island_patches.pdf'
half = args.patch_size // 2

# ── Load marker names and pick display channels ────────────────────────────────
marker_names = [l.strip() for l in open(MARKERS_FILE)]
DISPLAY      = ['EGFR', 'Pan-Keratin', 'CD8', 'CD3', 'CD4', 'CD68', 'dsDNA']
display_names = [m for m in DISPLAY if m in marker_names]
print(f'Display markers: {display_names}')

# ── Load H5 metadata ──────────────────────────────────────────────────────────
with h5py.File(H5, 'r') as f:
    all_dim1        = f['coords']['DIM1'][:]
    all_dim2        = f['coords']['DIM2'][:]
    all_sids        = f['coords']['sample_id'][:].astype(str)
    all_annots      = f['annotation'][()].astype(str)
    all_marker_names = f['marker_names'][:].astype(str)

# indices of used markers in the H5 marker array
h5_marker_idx = [int(np.where(all_marker_names == m)[0][0]) for m in marker_names]

# restrict to val cells only
val_indices = np.loadtxt(VAL_IDX, dtype=int)
dim1   = all_dim1[val_indices]
dim2   = all_dim2[val_indices]
sids   = all_sids[val_indices]
annots = all_annots[val_indices]

# ── Identify island groups from UMAP ──────────────────────────────────────────
umap_data  = np.load(Z / args.model / 'umap_embeddings.npz', allow_pickle=True)
val_data   = np.load(Z / args.model / 'val_results.npz',     allow_pickle=True)
xy         = umap_data['embedding']
labels_str = umap_data['labels_str']
sample_ids = val_data['sample_ids']

db = DBSCAN(eps=0.4, min_samples=5, n_jobs=-1).fit(xy)
cluster_ids = db.labels_

DOMINANCE_THRESH, MIN_CELLS, MAX_CELLS = 0.80, 5, 300
islands = []
for cid in range(cluster_ids.max() + 1):
    mask = cluster_ids == cid
    n    = mask.sum()
    if n < MIN_CELLS or n > MAX_CELLS:
        continue
    s = sample_ids[mask]
    unique, counts = np.unique(s, return_counts=True)
    top_sample = unique[counts.argmax()]
    dominance  = counts.max() / n
    if dominance >= DOMINANCE_THRESH:
        ctypes, ccounts = np.unique(labels_str[mask], return_counts=True)
        top_type = ctypes[ccounts.argmax()]
        islands.append((top_sample, top_type, dominance, n))
        print(f'Island: sample={top_sample}, type={top_type}, n={n}, dom={dominance:.0%}')

# ── Define groups: (label, sample_id filter, cell_type filter) ────────────────
immune_types = {'CD3 T', 'CD4 T', 'CD8 T', 'NK', 'Tregs', 'B'}

groups = []
for top_sample, top_type, dominance, n in islands:
    groups.append((
        f'Island — sample {top_sample} ({dominance:.0%})\n{top_type}',
        lambda i, s=top_sample, t=top_type: sids[i] == s and annots[i] == t
    ))
groups.append((
    'Sample 2 immune cells\n(EGFR-bright T-cell area)',
    lambda i: sids[i] == '2' and annots[i] in immune_types
))
groups.append((
    'Reference: other-sample T cells',
    lambda i: sids[i] not in {'2', '34'} and annots[i] in {'CD8 T', 'CD4 T'}
))

# ── Patch extraction ──────────────────────────────────────────────────────────
def get_patch(f, sid, d1, d2):
    grp = f['data'][sid]
    H_img, W_img = grp['image'].shape[:2]
    r0, r1 = int(d1) - half, int(d1) + half
    c0, c1 = int(d2) - half, int(d2) + half
    r0c, r1c = max(0, r0), min(H_img, r1)
    c0c, c1c = max(0, c0), min(W_img, c1)
    chunk = grp['image'][r0c:r1c, c0c:c1c, :][:, :, h5_marker_idx].astype(np.float32)
    patch = np.zeros((args.patch_size, args.patch_size, len(marker_names)), np.float32)
    patch[r0c - r0:r1c - r0, c0c - c0:c1c - c0] = chunk
    return patch

def normalize_channel(img):
    p99 = np.percentile(img, 99)
    return np.clip(img, 0, p99) / p99 if p99 > 0 else img

np.random.seed(42)

# ── Figure ────────────────────────────────────────────────────────────────────
n_groups  = len(groups)
n_markers = len(display_names)
n_p       = args.n_patches

fig, axes = plt.subplots(
    n_groups * (n_markers + 1),   # +1 row per group for separator
    n_p,
    figsize=(n_p * 1.4, n_groups * (n_markers + 0.6) * 1.3),
    gridspec_kw=dict(hspace=0.06, wspace=0.04),
)

with h5py.File(H5, 'r') as hf:
    for g_idx, (group_label, selector) in enumerate(groups):
        # find all val cells matching this group
        cell_idxs = [i for i in range(len(sids)) if selector(i)]
        chosen    = np.random.choice(cell_idxs, min(n_p, len(cell_idxs)), replace=False)
        print(f'Group "{group_label.split(chr(10))[0]}": {len(cell_idxs)} cells, showing {len(chosen)}')

        row_offset = g_idx * (n_markers + 1)

        # separator row (blank)
        for p in range(n_p):
            ax = axes[row_offset, p]
            ax.axis('off')
        axes[row_offset, 0].text(
            0, 0.5, group_label, transform=axes[row_offset, 0].transAxes,
            fontsize=8, fontweight='bold', va='center', ha='left',
        )

        for p_idx, cell_idx in enumerate(chosen):
            sid = sids[cell_idx]
            d1  = dim1[cell_idx]
            d2  = dim2[cell_idx]
            patch = get_patch(hf, sid, d1, d2)

            for m_idx, mname in enumerate(display_names):
                ax_row = row_offset + 1 + m_idx
                ax = axes[ax_row, p_idx]
                ch = normalize_channel(patch[:, :, marker_names.index(mname)])
                ax.imshow(ch, cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])

                if p_idx == 0:
                    ax.set_ylabel(mname, fontsize=7, rotation=0,
                                  ha='right', va='center', labelpad=38)
                if m_idx == 0:
                    ax.set_title(f's{sid} / {annots[cell_idx]}', fontsize=5.5)

        # fill any unused patch columns
        for p_idx in range(len(chosen), n_p):
            for m_idx in range(n_markers + 1):
                axes[row_offset + m_idx, p_idx].axis('off')

fig.suptitle(
    f'Cell patches by UMAP region — {args.model}\n'
    f'(each column = one cell, rows = marker channels, inferno scale)',
    fontsize=9, y=1.002,
)
fig.savefig(OUT, bbox_inches='tight', dpi=180)
print(f'\nSaved: {OUT}')

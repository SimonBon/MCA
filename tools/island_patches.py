"""
Show example patches for cells in sample-specific UMAP islands and the
suspicious EGFR-bright T-cell area.

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
parser.add_argument('--n_patches', type=int, default=8)
args = parser.parse_args()

OUT  = Z / args.model / 'island_patches.pdf'
half = args.patch_size // 2

# ── Load marker names and pick display channels ────────────────────────────────
marker_names = [l.strip() for l in open(MARKERS_FILE)]
DISPLAY = ['EGFR', 'Pan-Keratin', 'CD8', 'CD3', 'CD4', 'CD68', 'dsDNA']
display_idx = [marker_names.index(m) for m in DISPLAY if m in marker_names]
display_names = [marker_names[i] for i in display_idx]
print(f'Display markers: {display_names}')

# ── Rebuild val dataset metadata (same filtering as MCIDataset) ───────────────
IGNORE = {'Unidentified'}
val_indices = np.loadtxt(VAL_IDX, dtype=int)

with h5py.File(H5, 'r') as f:
    all_dim1   = f['coords']['DIM1'][:]
    all_dim2   = f['coords']['DIM2'][:]
    all_sids   = f['coords']['sample_id'][:].astype(str)
    all_annots = f['annotation'][()].astype(str)
    all_marker_names = f['marker_names'][:].astype(str)

# map used_markers → H5 marker indices
h5_marker_idx = [np.where(all_marker_names == m)[0][0]
                 for m in marker_names if m in all_marker_names]

dim1   = all_dim1[val_indices]
dim2   = all_dim2[val_indices]
sids   = all_sids[val_indices]
annots = all_annots[val_indices]

keep = np.array([a not in IGNORE for a in annots])
dim1   = dim1[keep]
dim2   = dim2[keep]
sids   = sids[keep]
annots = annots[keep]

# ── Verify alignment with val_results ─────────────────────────────────────────
val_data  = np.load(Z / args.model / 'val_results.npz',     allow_pickle=True)
umap_data = np.load(Z / args.model / 'umap_embeddings.npz', allow_pickle=True)
xy        = umap_data['embedding']
labels_str = umap_data['labels_str']
sample_ids = val_data['sample_ids']

sid_match    = (sids == sample_ids).mean()
annot_match  = (annots == labels_str).mean()
print(f'Alignment check — sample_id: {sid_match:.4f}, annotation: {annot_match:.4f}')
if sid_match < 0.99:
    print('WARNING: alignment < 99%, patches may be mismatched')

# ── Identify island cells via DBSCAN ──────────────────────────────────────────
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
        islands.append((cid, mask, top_sample, dominance))
        print(f'Island: cluster={cid}, sample={top_sample}, n={n}, '
              f'dominance={dominance:.0%}, types={np.unique(labels_str[mask])}')

# ── Identify yellow T-cell area: sample 2 immune cells ───────────────────────
immune_types = {'CD3 T', 'CD4 T', 'CD8 T', 'NK', 'Tregs', 'B'}
s2_immune_mask = np.array([
    sample_ids[i] == '2' and labels_str[i] in immune_types
    for i in range(len(sample_ids))
])
print(f'Sample 2 immune cells: {s2_immune_mask.sum()}')

# ── Patch extraction ──────────────────────────────────────────────────────────
def get_patch(f, sid, d1, d2):
    grp    = f['data'][sid]
    H_img, W_img = grp['image'].shape[:2]
    r0, r1 = d1 - half, d1 + half
    c0, c1 = d2 - half, d2 + half
    r0c, r1c = max(0, r0), min(H_img, r1)
    c0c, c1c = max(0, c0), min(W_img, c1)
    chunk = grp['image'][r0c:r1c, c0c:c1c, :][:, :, h5_marker_idx].astype(np.float32)
    patch = np.zeros((args.patch_size, args.patch_size, len(marker_names)), np.float32)
    patch[r0c - r0:r1c - r0, c0c - c0:c1c - c0, :] = chunk
    # mask overlay: segmentation for this cell (cell_id from masks = index in H5)
    mask_img = grp['masks'][r0c:r1c, c0c:c1c].astype(np.float32)
    mask_patch = np.zeros((args.patch_size, args.patch_size), np.float32)
    mask_patch[r0c - r0:r1c - r0, c0c - c0:c1c - c0] = mask_img
    return patch, mask_patch

def pick_indices(mask, n):
    idxs = np.where(mask)[0]
    np.random.seed(42)
    return idxs[np.random.choice(len(idxs), min(n, len(idxs)), replace=False)]

def normalize_channel(img):
    p99 = np.percentile(img, 99)
    if p99 > 0:
        img = np.clip(img, 0, p99) / p99
    return img

# ── Build groups to plot ──────────────────────────────────────────────────────
groups = []
for cid, island_mask, top_sample, dominance in islands:
    label = f'Island (sample {top_sample}, {dominance:.0%})\n{np.unique(labels_str[island_mask])[0]}'
    groups.append((label, island_mask))
groups.append(('Sample 2 immune cells\n(EGFR-bright T-cell area)', s2_immune_mask))

# also add a "normal" T cell group for comparison
normal_tcell_mask = np.array([
    sample_ids[i] != '2' and labels_str[i] in {'CD8 T', 'CD4 T'}
    for i in range(len(sample_ids))
])
groups.append(('Normal T cells\n(other samples, reference)', normal_tcell_mask))

# ── Figure ────────────────────────────────────────────────────────────────────
n_groups   = len(groups)
n_markers  = len(display_idx)
n_patches  = args.n_patches

fig, axes = plt.subplots(
    n_groups * n_markers, n_patches,
    figsize=(n_patches * 1.3, n_groups * n_markers * 1.3 + n_groups * 0.4),
    gridspec_kw=dict(hspace=0.05, wspace=0.05)
)

with h5py.File(H5, 'r') as hf:
    for g_idx, (group_label, group_mask) in enumerate(groups):
        cell_indices = pick_indices(group_mask, n_patches)

        for p_idx, cell_idx in enumerate(cell_indices):
            sid = sids[cell_idx]
            d1  = dim1[cell_idx]
            d2  = dim2[cell_idx]
            patch, seg = get_patch(hf, sid, d1, d2)

            for m_idx, h5m in enumerate(display_idx):
                ax = axes[g_idx * n_markers + m_idx, p_idx]
                channel = normalize_channel(patch[:, :, marker_names.index(display_names[m_idx])])
                ax.imshow(channel, cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])

                # left label: marker name on first patch
                if p_idx == 0:
                    ax.set_ylabel(display_names[m_idx], fontsize=7, rotation=0,
                                  ha='right', va='center', labelpad=35)

                # top label: cell type on first marker row
                if m_idx == 0:
                    ct = labels_str[cell_idx]
                    ax.set_title(f's{sid}\n{ct}', fontsize=5.5)

        # group label on far left of first marker row
        axes[g_idx * n_markers, 0].annotate(
            group_label, xy=(-0.7, 0.5),
            xycoords='axes fraction', fontsize=7.5, fontweight='bold',
            va='center', ha='right', rotation=0,
            annotation_clip=False,
        )

        # horizontal separator between groups
        if g_idx < n_groups - 1:
            last_row = g_idx * n_markers + n_markers - 1
            for p_idx in range(n_patches):
                axes[last_row, p_idx].spines['bottom'].set_linewidth(1.5)
                axes[last_row, p_idx].spines['bottom'].set_color('gray')

fig.suptitle(
    f'Cell patches by UMAP region — {args.model}\n'
    f'(rows = markers, columns = cells; inferno scale per channel)',
    fontsize=9, y=1.005
)
fig.savefig(OUT, bbox_inches='tight', dpi=180)
print(f'\nSaved: {OUT}')

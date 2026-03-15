"""
Combined figure: UMAP with annotated regions + example patches per region.

Usage:
    python tools/island_figure.py [--model FOLDER]

Saves: <model_dir>/island_figure.pdf
"""

import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from sklearn.cluster import DBSCAN

Z            = Path('/nobackup/lab_taschner-mandl/simongutwein/z_RUNS')
H5           = Path('/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/MIBI_TNBC.h5')
MARKERS_FILE = Path('/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/used_markers.txt')
VAL_IDX      = Path('/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/val.txt')

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MIBI_TNBC_CIM_VICReg_Funnel_Large')
parser.add_argument('--patch_size', type=int, default=32)
parser.add_argument('--n_patches',  type=int, default=6)
args = parser.parse_args()

OUT  = Z / args.model / 'island_figure.pdf'
half = args.patch_size // 2

# ── Markers ────────────────────────────────────────────────────────────────────
marker_names  = [l.strip() for l in open(MARKERS_FILE)]
DISPLAY       = ['EGFR', 'Pan-Keratin', 'CD8', 'CD3', 'dsDNA']
display_names = [m for m in DISPLAY if m in marker_names]

# ── Load H5 metadata ──────────────────────────────────────────────────────────
with h5py.File(H5, 'r') as f:
    all_dim1         = f['coords']['DIM1'][:]
    all_dim2         = f['coords']['DIM2'][:]
    all_sids         = f['coords']['sample_id'][:].astype(str)
    all_annots       = f['annotation'][()].astype(str)
    all_marker_names = f['marker_names'][:].astype(str)

h5_marker_idx = [int(np.where(all_marker_names == m)[0][0]) for m in marker_names]

val_indices = np.loadtxt(VAL_IDX, dtype=int)
dim1   = all_dim1[val_indices]
dim2   = all_dim2[val_indices]
sids   = all_sids[val_indices]
annots = all_annots[val_indices]

# ── UMAP + val_results ────────────────────────────────────────────────────────
umap_data  = np.load(Z / args.model / 'umap_embeddings.npz', allow_pickle=True)
val_data   = np.load(Z / args.model / 'val_results.npz',     allow_pickle=True)
xy         = umap_data['embedding']
labels_str = umap_data['labels_str']
sample_ids = val_data['sample_ids']

# ── DBSCAN islands ────────────────────────────────────────────────────────────
db = DBSCAN(eps=0.4, min_samples=5, n_jobs=-1).fit(xy)
cluster_ids = db.labels_
islands = []
for cid in range(cluster_ids.max() + 1):
    mask = cluster_ids == cid
    n    = mask.sum()
    if not (5 <= n <= 300):
        continue
    s = sample_ids[mask]
    unique, counts = np.unique(s, return_counts=True)
    top_sample = unique[counts.argmax()]
    dominance  = counts.max() / n
    if dominance >= 0.80:
        ctypes, ccounts = np.unique(labels_str[mask], return_counts=True)
        top_type = ctypes[ccounts.argmax()]
        islands.append(dict(mask=mask, sample=top_sample, dominance=dominance,
                            n=n, top_type=top_type))

immune_types = {'CD3 T', 'CD4 T', 'CD8 T', 'NK', 'Tregs', 'B'}
s2_immune_mask = np.array([
    sample_ids[i] == '2' and labels_str[i] in immune_types
    for i in range(len(sample_ids))
])

# ── Region definitions ────────────────────────────────────────────────────────
# colour, umap_mask (for UMAP highlighting), h5_selector (for patch fetching)
REGIONS = []
COLOURS = ['#e6194b', '#f58231', '#4363d8', '#808080']

for isl in islands:
    s, t = isl['sample'], isl['top_type']
    REGIONS.append(dict(
        label    = f'Island — sample {s}\n({isl["dominance"]:.0%} one patient)',
        colour   = COLOURS[len(REGIONS)],
        umap_mask= isl['mask'],
        h5_sel   = lambda i, s=s, t=t: sids[i] == s and annots[i] == t,
    ))

REGIONS.append(dict(
    label    = f'Sample 2 immune cells\n(EGFR-bright T-cell area)',
    colour   = COLOURS[len(REGIONS)],
    umap_mask= s2_immune_mask,
    h5_sel   = lambda i: sids[i] == '2' and annots[i] in immune_types,
))
REGIONS.append(dict(
    label    = 'Reference T cells\n(other samples)',
    colour   = COLOURS[len(REGIONS)],
    umap_mask= np.array([
        sample_ids[i] not in {'2', '34'} and labels_str[i] in {'CD8 T', 'CD4 T'}
        for i in range(len(sample_ids))
    ]),
    h5_sel   = lambda i: sids[i] not in {'2', '34'} and annots[i] in {'CD8 T', 'CD4 T'},
))

n_regions = len(REGIONS)
n_markers = len(display_names)
n_p       = args.n_patches

# ── Patch helper ──────────────────────────────────────────────────────────────
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

def norm(img):
    p = np.percentile(img, 99)
    return np.clip(img, 0, p) / p if p > 0 else img

np.random.seed(42)

# ── Pre-fetch patches ─────────────────────────────────────────────────────────
region_patches = []   # list of list of (patch, sid, annot)
with h5py.File(H5, 'r') as hf:
    for reg in REGIONS:
        cell_idxs = [i for i in range(len(sids)) if reg['h5_sel'](i)]
        chosen    = np.random.choice(cell_idxs, min(n_p, len(cell_idxs)), replace=False)
        patches   = [(get_patch(hf, sids[i], dim1[i], dim2[i]), sids[i], annots[i])
                     for i in chosen]
        region_patches.append(patches)
        print(f'  {reg["label"].split(chr(10))[0]}: {len(cell_idxs)} cells, showing {len(chosen)}')

# ── Figure layout ─────────────────────────────────────────────────────────────
# Top: UMAP (full width)
# Bottom: n_regions columns × n_markers rows of patch panels
fig = plt.figure(figsize=(n_p * n_regions * 1.1 + 0.5, 14))
gs_outer = GridSpec(2, 1, figure=fig, height_ratios=[1.6, 1], hspace=0.12)

# ── UMAP panel ────────────────────────────────────────────────────────────────
ax_umap = fig.add_subplot(gs_outer[0])

# all cells light gray background
ax_umap.scatter(xy[:, 0], xy[:, 1], c='#cccccc', s=1.2,
                linewidths=0, rasterized=True, alpha=0.4, zorder=1)

# highlight each region; reference last so it doesn't dominate
for reg in REGIONS[::-1]:
    m = reg['umap_mask']
    ax_umap.scatter(xy[m, 0], xy[m, 1], c=reg['colour'],
                    s=6 if 'Reference' not in reg['label'] else 3,
                    linewidths=0, rasterized=True, alpha=0.9, zorder=2,
                    label=reg['label'].replace('\n', ' '))

ax_umap.legend(fontsize=8, markerscale=2.5, loc='upper right',
               framealpha=0.85, edgecolor='#999999')
ax_umap.set_xlabel('UMAP 1', fontsize=9)
ax_umap.set_ylabel('UMAP 2', fontsize=9)
ax_umap.set_title('UMAP — highlighted regions correspond to patch panels below',
                  fontsize=10, pad=6)
ax_umap.set_aspect('equal', adjustable='datalim')
ax_umap.tick_params(labelsize=8)

# draw a coloured bracket under the UMAP connecting to each column
# (done via fig.text annotations after axes are placed)

# ── Patch grid ────────────────────────────────────────────────────────────────
gs_patches = gs_outer[1].subgridspec(
    n_markers, n_regions * n_p,
    hspace=0.04, wspace=0.03,
)

for r_idx, (reg, patches) in enumerate(zip(REGIONS, region_patches)):
    col_start = r_idx * n_p
    colour    = reg['colour']

    for p_idx, (patch, sid, ann) in enumerate(patches):
        for m_idx, mname in enumerate(display_names):
            ax = fig.add_subplot(gs_patches[m_idx, col_start + p_idx])
            ch = norm(patch[:, :, marker_names.index(mname)])
            ax.imshow(ch, cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])

            # coloured border matching UMAP region colour
            for spine in ax.spines.values():
                spine.set_edgecolor(colour)
                spine.set_linewidth(1.8)

            # marker label on leftmost column of each region
            if p_idx == 0:
                ax.set_ylabel(mname, fontsize=7, rotation=0,
                              ha='right', va='center', labelpad=38)

            # cell annotation on top row first patch
            if m_idx == 0 and p_idx == 0:
                ax.set_title(reg['label'], fontsize=7, fontweight='bold',
                             color=colour, loc='left', pad=3)

            # sample/type on top row
            if m_idx == 0:
                ax.set_title(
                    (ax.get_title() or '') + f'\ns{sid}/{ann[:6]}',
                    fontsize=5, color='#444444', loc='left', pad=3
                ) if p_idx == 0 else ax.set_title(
                    f's{sid}/{ann[:6]}', fontsize=5, color='#444444', pad=3
                )

fig.suptitle(
    f'UMAP regions and representative cell patches — {args.model}',
    fontsize=10, y=1.002,
)
fig.savefig(OUT, bbox_inches='tight', dpi=180)
print(f'\nSaved: {OUT}')

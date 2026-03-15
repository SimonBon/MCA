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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path
from sklearn.cluster import DBSCAN

Z            = Path('/nobackup/lab_taschner-mandl/simongutwein/z_RUNS')
H5           = Path('/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/MIBI_TNBC.h5')
MARKERS_FILE = Path('/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/used_markers.txt')
VAL_IDX      = Path('/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/val.txt')

parser = argparse.ArgumentParser()
parser.add_argument('--model',      default='MIBI_TNBC_CIM_VICReg_Funnel_Large')
parser.add_argument('--patch_size', type=int, default=32)
parser.add_argument('--n_patches',  type=int, default=4)
args = parser.parse_args()

OUT  = Z / args.model / 'island_figure.pdf'
half = args.patch_size // 2

# ── Markers ────────────────────────────────────────────────────────────────────
marker_names  = [l.strip() for l in open(MARKERS_FILE)]
DISPLAY       = ['EGFR', 'Pan-Keratin', 'CD8', 'CD3', 'dsDNA']
display_names = [m for m in DISPLAY if m in marker_names]

# ── H5 metadata ───────────────────────────────────────────────────────────────
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

# ── UMAP ──────────────────────────────────────────────────────────────────────
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
        islands.append(dict(mask=mask, sample=top_sample,
                            dominance=dominance, n=n, top_type=top_type))

immune_types = {'CD3 T', 'CD4 T', 'CD8 T', 'NK', 'Tregs', 'B'}
s2_immune = np.array([sample_ids[i] == '2' and labels_str[i] in immune_types
                      for i in range(len(sample_ids))])

# ── Region definitions ────────────────────────────────────────────────────────
COLOURS = ['#e6194b', '#ff8c00', '#00aaff', '#2ca02c']
REGIONS = []
for isl in islands:
    s, t = isl['sample'], isl['top_type']
    REGIONS.append(dict(
        label     = f'Island — sample {s}\n({isl["dominance"]:.0%} one patient)',
        short     = f'Island s{s}',
        colour    = COLOURS[len(REGIONS)],
        umap_mask = isl['mask'],
        h5_sel    = lambda i, s=s, t=t: sids[i] == s and annots[i] == t,
    ))
REGIONS.append(dict(
    label     = 'Sample 2 immune cells\n(EGFR-bright T-cell area)',
    short     = 'S2 immune',
    colour    = COLOURS[len(REGIONS)],
    umap_mask = s2_immune,
    h5_sel    = lambda i: sids[i] == '2' and annots[i] in immune_types,
))
REGIONS.append(dict(
    label     = 'Reference T cells\n(other samples)',
    short     = 'Ref T cells',
    colour    = COLOURS[len(REGIONS)],
    umap_mask = np.array([sample_ids[i] not in {'2','34'} and
                          labels_str[i] in {'CD8 T','CD4 T'}
                          for i in range(len(sample_ids))]),
    h5_sel    = lambda i: sids[i] not in {'2','34'} and
                          annots[i] in {'CD8 T','CD4 T'},
))

n_regions = len(REGIONS)
n_markers = len(display_names)
n_p       = args.n_patches

# ── Pre-fetch patches ─────────────────────────────────────────────────────────
def get_patch(f, sid, d1, d2):
    grp = f['data'][sid]
    H_img, W_img = grp['image'].shape[:2]
    r0, r1 = int(d1)-half, int(d1)+half
    c0, c1 = int(d2)-half, int(d2)+half
    r0c,r1c = max(0,r0), min(H_img,r1)
    c0c,c1c = max(0,c0), min(W_img,c1)
    chunk = grp['image'][r0c:r1c, c0c:c1c, :][:,:,h5_marker_idx].astype(np.float32)
    patch = np.zeros((args.patch_size, args.patch_size, len(marker_names)), np.float32)
    patch[r0c-r0:r1c-r0, c0c-c0:c1c-c0] = chunk
    return patch

def norm(img):
    p = np.percentile(img, 99)
    return np.clip(img, 0, p) / p if p > 0 else img

np.random.seed(42)
region_patches = []
with h5py.File(H5, 'r') as hf:
    for reg in REGIONS:
        cell_idxs = [i for i in range(len(sids)) if reg['h5_sel'](i)]
        chosen    = np.random.choice(cell_idxs, min(n_p, len(cell_idxs)), replace=False)
        region_patches.append(
            [(get_patch(hf, sids[i], dim1[i], dim2[i]), sids[i], annots[i]) for i in chosen]
        )
        print(f'  {reg["short"]}: {len(cell_idxs)} cells, showing {len(chosen)}')

# ── Figure ────────────────────────────────────────────────────────────────────
# Layout: 1 UMAP row, then per-region: 1 header row + n_markers patch rows
# All columns: n_regions × n_p patch columns + label column on left per region
fig_w = 4 + n_regions * (n_p * 1.5 + 0.5)
fig_h = 8 + n_markers * 1.2
fig = plt.figure(figsize=(fig_w, fig_h))

# Outer: UMAP on top, patches below
gs_outer = GridSpec(2, 1, figure=fig, height_ratios=[2.2, n_markers * 0.8],
                    hspace=0.25)

# ── UMAP ──────────────────────────────────────────────────────────────────────
ax_umap = fig.add_subplot(gs_outer[0])

highlighted = np.zeros(len(xy), dtype=bool)
for reg in REGIONS:
    highlighted |= reg['umap_mask']

ax_umap.scatter(xy[~highlighted,0], xy[~highlighted,1],
                c='#e0e0e0', s=1.2, linewidths=0, rasterized=True, alpha=0.6, zorder=1)

for reg in REGIONS[::-1]:
    m   = reg['umap_mask']
    ref = 'Reference' in reg['label']
    ax_umap.scatter(xy[m,0], xy[m,1], c=reg['colour'],
                    s=3 if ref else 20,
                    linewidths=0, rasterized=True,
                    alpha=0.5 if ref else 1.0, zorder=2 if ref else 3,
                    label=reg['label'].replace('\n', '  '))

ax_umap.legend(fontsize=8.5, markerscale=2.5, loc='upper right',
               framealpha=0.9, edgecolor='#aaaaaa')
ax_umap.set_xlabel('UMAP 1', fontsize=9)
ax_umap.set_ylabel('UMAP 2', fontsize=9)
ax_umap.set_title('UMAP — coloured regions correspond to patch panels below',
                  fontsize=10, pad=8)
# clip to 1st–99th percentile range to exclude outlier points
x1,x99 = np.percentile(xy[:,0],[1,99]); pad_x = (x99-x1)*0.08
y1,y99 = np.percentile(xy[:,1],[1,99]); pad_y = (y99-y1)*0.08
ax_umap.set_xlim(x1-pad_x, x99+pad_x)
ax_umap.set_ylim(y1-pad_y, y99+pad_y)
ax_umap.set_aspect('equal', adjustable='box')
ax_umap.tick_params(labelsize=8)

# ── Patch panels ──────────────────────────────────────────────────────────────
# Divide bottom strip into n_regions columns (with small gap between groups)
gs_bottom = GridSpecFromSubplotSpec(
    1, n_regions, subplot_spec=gs_outer[1],
    wspace=0.08,
)

for r_idx, (reg, patches) in enumerate(zip(REGIONS, region_patches)):
    colour = reg['colour']

    # Inner grid: header row + n_markers patch rows, n_p columns
    inner = GridSpecFromSubplotSpec(
        n_markers + 1, n_p,
        subplot_spec=gs_bottom[0, r_idx],
        hspace=0.05, wspace=0.04,
        height_ratios=[0.32] + [1] * n_markers,
    )

    # ── Coloured header spanning full group width ──────────────────────────
    # Draw n_p header cells, all same colour; put label text on the spanning row
    hdr_axes = []
    for p in range(n_p):
        ax_h = fig.add_subplot(inner[0, p])
        ax_h.set_facecolor(colour)
        ax_h.set_xticks([]); ax_h.set_yticks([])
        for sp in ax_h.spines.values():
            sp.set_visible(False)
        hdr_axes.append(ax_h)

    # Centre the group label across the header by annotating the middle cell
    mid = n_p // 2
    hdr_axes[mid].text(
        0.5, 0.5, reg['label'].replace('\n', '\n'),
        transform=hdr_axes[mid].transAxes,
        ha='center', va='center',
        fontsize=7, fontweight='bold', color='white',
        clip_on=False, linespacing=1.3,
    )

    # ── Patch cells ───────────────────────────────────────────────────────
    for p_idx, (patch, sid, ann) in enumerate(patches):
        for m_idx, mname in enumerate(display_names):
            ax = fig.add_subplot(inner[m_idx + 1, p_idx])
            ch = norm(patch[:, :, marker_names.index(mname)])
            ax.imshow(ch, cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])

            # Coloured border per group
            for sp in ax.spines.values():
                sp.set_edgecolor(colour)
                sp.set_linewidth(1.4)

            # Marker name on leftmost column of first group only
            if p_idx == 0 and r_idx == 0:
                ax.set_ylabel(mname, fontsize=8, rotation=0,
                              ha='right', va='center', labelpad=42)

            # Cell label on top patch row
            if m_idx == 0:
                ax.set_title(f's{sid} / {ann}', fontsize=5.5,
                             color='#333333', pad=2)

fig.suptitle(
    f'UMAP regions and representative cell patches — {args.model}',
    fontsize=10, y=1.002,
)
fig.savefig(OUT, bbox_inches='tight', dpi=180)
print(f'\nSaved: {OUT}')

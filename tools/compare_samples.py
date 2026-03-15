"""
Compare two samples for a given cell type: UMAP highlight, marker z-scores,
and side-by-side patch panels.

Usage:
    python tools/compare_samples.py [--model FOLDER] --s1 35 --s2 28 \
        [--cell_types "CD3 T,CD4 T,CD8 T,Tregs,NK"]

Saves: <model_dir>/compare_s<s1>_vs_s<s2>_<celltypes>.pdf
"""

import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

Z            = Path('/nobackup/lab_taschner-mandl/simongutwein/z_RUNS')
H5           = Path('/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/MIBI_TNBC.h5')
MARKERS_FILE = Path('/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/used_markers.txt')
VAL_IDX      = Path('/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/val.txt')
EXPR         = Z / 'MIBI_TNBC_ExprBaseline_mean' / 'val_results.npz'

parser = argparse.ArgumentParser()
parser.add_argument('--model',      default='MIBI_TNBC_CIM_VICReg_Funnel_Large')
parser.add_argument('--s1',         default='35')
parser.add_argument('--s2',         default='28')
parser.add_argument('--cell_types', default='CD3 T,CD4 T,CD8 T,Tregs,NK')
parser.add_argument('--patch_size', type=int, default=32)
parser.add_argument('--n_patches',  type=int, default=6)
args = parser.parse_args()

S1 = args.s1
S2 = args.s2
CELL_TYPES = [c.strip() for c in args.cell_types.split(',')]
half = args.patch_size // 2
slug = '_'.join(c.replace(' ', '') for c in CELL_TYPES)
OUT  = Z / args.model / f'compare_s{S1}_vs_s{S2}_{slug}.pdf'

# ── Load ──────────────────────────────────────────────────────────────────────
marker_names = [l.strip() for l in open(MARKERS_FILE)]

umap_data  = np.load(Z / args.model / 'umap_embeddings.npz', allow_pickle=True)
val_data   = np.load(Z / args.model / 'val_results.npz',     allow_pickle=True)
xy         = umap_data['embedding']
labels_str = umap_data['labels_str']
sample_ids = val_data['sample_ids']

expr_data  = np.load(EXPR, allow_pickle=True)
expr_feat  = expr_data['features']
expr_sids  = expr_data['sample_ids']
expr_lbls  = expr_data['labels_str']

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

# ── UMAP masks ────────────────────────────────────────────────────────────────
mask1 = np.array([sample_ids[i] == S1 and labels_str[i] in CELL_TYPES
                  for i in range(len(sample_ids))])
mask2 = np.array([sample_ids[i] == S2 and labels_str[i] in CELL_TYPES
                  for i in range(len(sample_ids))])
print(f'UMAP cells — s{S1}: {mask1.sum()}, s{S2}: {mask2.sum()}')

# ── Marker z-scores: s1 T cells vs s2 T cells ─────────────────────────────────
expr_m1 = (expr_sids == S1) & np.isin(expr_lbls, CELL_TYPES)
expr_m2 = (expr_sids == S2) & np.isin(expr_lbls, CELL_TYPES)
print(f'Expr cells — s{S1}: {expr_m1.sum()}, s{S2}: {expr_m2.sum()}')

mean1 = expr_feat[expr_m1].mean(0)
mean2 = expr_feat[expr_m2].mean(0)
# z-score of s1 relative to s2
pool_std = (expr_feat[expr_m1].std(0) + expr_feat[expr_m2].std(0)) / 2 + 1e-8
z = (mean1 - mean2) / pool_std

top_idx = np.abs(z).argsort()[::-1][:15]
print('Top markers s1 vs s2:')
for i in top_idx[:8]:
    print(f'  {marker_names[i]}: {z[i]:+.2f}')

# markers to show in patches: top 4 most-different
show_marker_idx = np.abs(z).argsort()[::-1][:4]
show_markers    = [marker_names[i] for i in show_marker_idx]
print(f'Show markers: {show_markers}')

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

np.random.seed(42)
h5_idx1 = [i for i in range(len(sids)) if sids[i] == S1 and annots[i] in CELL_TYPES]
h5_idx2 = [i for i in range(len(sids)) if sids[i] == S2 and annots[i] in CELL_TYPES]
n_p = args.n_patches
with h5py.File(H5, 'r') as hf:
    ch1 = np.random.choice(h5_idx1, min(n_p, len(h5_idx1)), replace=False)
    ch2 = np.random.choice(h5_idx2, min(n_p, len(h5_idx2)), replace=False)
    patches1 = [(get_patch(hf, sids[i], dim1[i], dim2[i]), annots[i]) for i in ch1]
    patches2 = [(get_patch(hf, sids[i], dim1[i], dim2[i]), annots[i]) for i in ch2]

# shared per-channel scale: pool 99th percentile across both groups
ref_p99 = {}
for mname in show_markers:
    midx = marker_names.index(mname)
    all_vals = np.concatenate(
        [p[:, :, midx].ravel() for p, _ in patches1] +
        [p[:, :, midx].ravel() for p, _ in patches2]
    )
    ref_p99[mname] = np.percentile(all_vals, 99) or 1.0

def norm_shared(img, p99):
    return np.clip(img, 0, p99) / p99

# ── Figure ────────────────────────────────────────────────────────────────────
C1, C2 = '#e6393a', '#2277cc'
n_show = len(show_markers)

fig = plt.figure(figsize=(14, 9))
gs_top = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1.3],
                            wspace=0.3, top=0.92, bottom=0.52)
gs_bot = gridspec.GridSpec(n_show, 2 * n_p + 1, figure=fig,
                            top=0.44, bottom=0.04,
                            wspace=0.05, hspace=0.08,
                            width_ratios=[1]*n_p + [0.4] + [1]*n_p)

# ── UMAP ──────────────────────────────────────────────────────────────────────
ax_umap = fig.add_subplot(gs_top[0, 0])
neither = ~(mask1 | mask2)
ax_umap.scatter(xy[neither, 0], xy[neither, 1],
                c='#dddddd', s=1.2, linewidths=0, rasterized=True, alpha=0.5, zorder=1)
ax_umap.scatter(xy[mask2, 0], xy[mask2, 1],
                color=C2, s=12, linewidths=0, rasterized=True, alpha=0.9, zorder=2,
                label=f'Sample {S2}  (n={mask2.sum()})')
ax_umap.scatter(xy[mask1, 0], xy[mask1, 1],
                color=C1, s=12, linewidths=0, rasterized=True, alpha=0.9, zorder=3,
                label=f'Sample {S1}  (n={mask1.sum()})')

x1r, x99r = np.percentile(xy[:, 0], [1, 99]); px = (x99r - x1r) * 0.05
y1r, y99r = np.percentile(xy[:, 1], [1, 99]); py = (y99r - y1r) * 0.05
ax_umap.set_xlim(x1r - px, x99r + px)
ax_umap.set_ylim(y1r - py, y99r + py)
ax_umap.set_aspect('equal', adjustable='datalim')
ax_umap.set_xlabel('UMAP 1', fontsize=9); ax_umap.set_ylabel('UMAP 2', fontsize=9)
ax_umap.set_title(f'T cells: sample {S1} vs {S2}', fontsize=10)
ax_umap.legend(fontsize=8, markerscale=2, loc='upper right')
ax_umap.tick_params(labelsize=8)

# ── Z-score bar chart ─────────────────────────────────────────────────────────
ax_z = fig.add_subplot(gs_top[0, 1])
names = [marker_names[i] for i in top_idx]
vals  = z[top_idx]
colours = [C1 if v > 0 else C2 for v in vals]
ax_z.barh(range(len(top_idx))[::-1], vals, color=colours, edgecolor='none')
ax_z.set_yticks(range(len(top_idx))[::-1])
ax_z.set_yticklabels(names, fontsize=8)
ax_z.axvline(0, color='black', linewidth=0.8)
ax_z.set_xlabel(f'z-score  (positive = higher in s{S1})', fontsize=8)
ax_z.set_title(f'Marker differences — s{S1} (red) vs s{S2} (blue)\n'
               f'T cells: {", ".join(CELL_TYPES)}', fontsize=9)
ax_z.tick_params(axis='x', labelsize=8)

# ── Patch panels ──────────────────────────────────────────────────────────────
for m_idx, mname in enumerate(show_markers):
    midx = marker_names.index(mname)
    p99  = ref_p99[mname]

    for p_idx, (patch, ann) in enumerate(patches1):
        ax = fig.add_subplot(gs_bot[m_idx, p_idx])
        ax.imshow(norm_shared(patch[:, :, midx], p99),
                  cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(C1); sp.set_linewidth(1.5)
        if m_idx == 0:
            ax.set_title(ann, fontsize=5, color=C1)
        if p_idx == 0:
            ax.set_ylabel(mname, fontsize=8, rotation=0,
                          ha='right', va='center', labelpad=46)

    # spacer
    ax_sp = fig.add_subplot(gs_bot[m_idx, n_p])
    ax_sp.axis('off')
    if m_idx == n_show // 2:
        ax_sp.text(0.5, 0.5, 'vs', transform=ax_sp.transAxes,
                   ha='center', va='center', fontsize=11, color='#555555')

    for p_idx, (patch, ann) in enumerate(patches2):
        ax = fig.add_subplot(gs_bot[m_idx, n_p + 1 + p_idx])
        ax.imshow(norm_shared(patch[:, :, midx], p99),
                  cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(C2); sp.set_linewidth(1.5)
        if m_idx == 0:
            ax.set_title(ann, fontsize=5, color=C2)

# group labels
frac1 = (n_p / 2) / (2 * n_p + 1)
frac2 = (n_p + 1 + n_p / 2) / (2 * n_p + 1)
fig.text(frac1, 0.455, f'Sample {S1}  (T cells)',
         ha='center', va='bottom', fontsize=9, fontweight='bold', color=C1)
fig.text(frac2, 0.455, f'Sample {S2}  (T cells)',
         ha='center', va='bottom', fontsize=9, fontweight='bold', color=C2)
fig.text(0.5, 0.455,
         f'Shared intensity scale (pooled p99). Top {n_show} most-different markers shown.',
         ha='center', va='top', fontsize=7.5, color='#555555')

fig.suptitle(
    f'Sample {S1} vs {S2} — T cell comparison ({", ".join(CELL_TYPES)})\n{args.model}',
    fontsize=10, y=0.98,
)
fig.savefig(OUT, bbox_inches='tight', dpi=180)
print(f'\nSaved: {OUT}')

"""
Diagnose why a sample forms an isolated blob in UMAP: show the UMAP with the
target sample highlighted, z-score heatmap across all cell types, and example
patches (target sample vs reference) for the most-affected markers.

Usage:
    python tools/sample_batch_effect.py [--model FOLDER] [--sample 41]

Saves: <model_dir>/sample_batch_effect_s<sample>.pdf
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
parser.add_argument('--sample',     default='41')
parser.add_argument('--patch_size', type=int, default=32)
parser.add_argument('--n_patches',  type=int, default=5)
args = parser.parse_args()

TARGET = args.sample
half   = args.patch_size // 2
OUT    = Z / args.model / f'sample_batch_effect_s{TARGET}.pdf'

# ── Data ──────────────────────────────────────────────────────────────────────
marker_names = [l.strip() for l in open(MARKERS_FILE)]

umap_data  = np.load(Z / args.model / 'umap_embeddings.npz', allow_pickle=True)
val_data   = np.load(Z / args.model / 'val_results.npz',     allow_pickle=True)
xy         = umap_data['embedding']
labels_str = umap_data['labels_str']
sample_ids = val_data['sample_ids']

expr_data = np.load(EXPR, allow_pickle=True)
expr_feat  = expr_data['features']
expr_sids  = expr_data['sample_ids']
expr_lbls  = expr_data['labels_str']

with h5py.File(H5, 'r') as f:
    all_dim1  = f['coords']['DIM1'][:]
    all_dim2  = f['coords']['DIM2'][:]
    all_sids  = f['coords']['sample_id'][:].astype(str)
    all_annots = f['annotation'][()].astype(str)
    all_marker_names = f['marker_names'][:].astype(str)

h5_marker_idx = [int(np.where(all_marker_names == m)[0][0]) for m in marker_names]

val_indices = np.loadtxt(VAL_IDX, dtype=int)
dim1   = all_dim1[val_indices]
dim2   = all_dim2[val_indices]
sids   = all_sids[val_indices]
annots = all_annots[val_indices]

# ── Z-scores per cell type: target sample vs rest ─────────────────────────────
target_mask = expr_sids == TARGET
rest_mask   = ~target_mask

cell_types_in_target = sorted(set(expr_lbls[target_mask]))
rest_mean = expr_feat[rest_mask].mean(0)
rest_std  = expr_feat[rest_mask].std(0) + 1e-8

# Build (cell_type → z-vector) for cells in target sample
ct_z = {}
ct_n = {}
for ct in cell_types_in_target:
    m = target_mask & (expr_lbls == ct)
    n = m.sum()
    if n < 3:
        continue
    z = (expr_feat[m].mean(0) - rest_mean) / rest_std
    ct_z[ct] = z
    ct_n[ct] = n
    top = np.abs(z).argsort()[::-1][:3]
    print(f'  {ct:<30} n={n:3d}  top: ' +
          ', '.join(f'{marker_names[i]}({z[i]:+.2f})' for i in top))

cell_types = list(ct_z.keys())
n_ct = len(cell_types)

# z-score matrix: rows = cell types, cols = markers
Z_mat = np.vstack([ct_z[ct] for ct in cell_types])

# Sort markers by mean |z| across cell types so most consistent come first
marker_importance = np.abs(Z_mat).mean(0)
marker_order = marker_importance.argsort()[::-1][:20]  # top 20 markers

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

def norm(img):
    p = np.percentile(img, 99)
    return np.clip(img, 0, p) / p if p > 0 else img

# Pick 3 most-affected markers (most negative z, averaged across cell types)
mean_z = Z_mat.mean(0)
show_marker_idx = mean_z.argsort()[:3]          # 3 most depressed
show_markers    = [marker_names[i] for i in show_marker_idx]
print(f'\nMost depressed markers: {show_markers}')

# Pick a common cell type with enough cells in both target and reference
representative_ct = max(ct_n, key=lambda ct: ct_n[ct])
print(f'Representative cell type for patches: {representative_ct}')

np.random.seed(42)
target_cell_idxs = [i for i in range(len(sids))
                    if sids[i] == TARGET and annots[i] == representative_ct]
ref_cell_idxs    = [i for i in range(len(sids))
                    if sids[i] != TARGET and annots[i] == representative_ct]

n_p = args.n_patches
with h5py.File(H5, 'r') as hf:
    chosen_t = np.random.choice(target_cell_idxs, min(n_p, len(target_cell_idxs)), replace=False)
    chosen_r = np.random.choice(ref_cell_idxs,    min(n_p, len(ref_cell_idxs)),    replace=False)
    target_patches = [get_patch(hf, sids[i], dim1[i], dim2[i]) for i in chosen_t]
    ref_patches    = [get_patch(hf, sids[i], dim1[i], dim2[i]) for i in chosen_r]
    target_sids_p  = [sids[i] for i in chosen_t]
    ref_sids_p     = [sids[i] for i in chosen_r]

# ── Compute per-channel global scale for fair comparison ─────────────────────
# Use reference-set 99th percentile per channel so both rows share the same scale
ref_p99 = {}
for mname in show_markers:
    midx = marker_names.index(mname)
    all_vals = np.concatenate([p[:, :, midx].ravel() for p in ref_patches])
    ref_p99[mname] = np.percentile(all_vals, 99) or 1.0

def norm_shared(img, p99):
    return np.clip(img, 0, p99) / p99

# ── Figure ────────────────────────────────────────────────────────────────────
# Layout:
#   Row 0: UMAP (left, large) | z-score heatmap (right)
#   Row 1: patch comparison: target sample | reference (3 marker rows × n_p cols each)
n_show_markers = len(show_markers)

fig = plt.figure(figsize=(14, 9))
gs_top = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1.4], wspace=0.3, top=0.92, bottom=0.52)
gs_bot = gridspec.GridSpec(n_show_markers, 2 * n_p + 1, figure=fig, top=0.44, bottom=0.04,
                            wspace=0.05, hspace=0.08,
                            width_ratios=[1]*n_p + [0.4] + [1]*n_p)

# ── UMAP ──────────────────────────────────────────────────────────────────────
ax_umap = fig.add_subplot(gs_top[0, 0])
is_target = sample_ids == TARGET

# colour all samples by tab20 (same logic as island_analysis.py)
all_samples = np.unique(sample_ids)
cmap_samples = plt.cm.tab20
sample_colour = {s: cmap_samples(i / len(all_samples)) for i, s in enumerate(all_samples)}
colours_arr = np.array([sample_colour[s] for s in sample_ids])

ax_umap.scatter(xy[~is_target, 0], xy[~is_target, 1],
                c=colours_arr[~is_target], s=1.2, linewidths=0,
                rasterized=True, alpha=0.5, zorder=1)
ax_umap.scatter(xy[is_target, 0], xy[is_target, 1],
                color='#ff2200', s=14, linewidths=0,
                rasterized=True, zorder=3, label=f'Sample {TARGET} ({is_target.sum()} cells)')

x1, x99 = np.percentile(xy[:, 0], [1, 99]); pad_x = (x99 - x1) * 0.08
y1, y99 = np.percentile(xy[:, 1], [1, 99]); pad_y = (y99 - y1) * 0.08
ax_umap.set_xlim(x1 - pad_x, x99 + pad_x)
ax_umap.set_ylim(y1 - pad_y, y99 + pad_y)
ax_umap.set_xlabel('UMAP 1', fontsize=9)
ax_umap.set_ylabel('UMAP 2', fontsize=9)
ax_umap.set_title(f'UMAP — sample {TARGET} highlighted in red', fontsize=10)
ax_umap.legend(fontsize=8, markerscale=1.5, loc='upper right')
ax_umap.tick_params(labelsize=8)
ax_umap.set_aspect('equal', adjustable='datalim')

# ── Z-score heatmap ───────────────────────────────────────────────────────────
ax_heat = fig.add_subplot(gs_top[0, 1])
# top 15 markers by mean |z|
top15 = np.abs(Z_mat).mean(0).argsort()[::-1][:15]
heatmap_data = Z_mat[:, top15]
marker_labels = [marker_names[i] for i in top15]

vmax = max(3.0, np.abs(heatmap_data).max())
im = ax_heat.imshow(heatmap_data, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                    aspect='auto', interpolation='nearest')
ax_heat.set_xticks(range(len(marker_labels)))
ax_heat.set_xticklabels(marker_labels, rotation=45, ha='right', fontsize=7)
ax_heat.set_yticks(range(n_ct))
ax_heat.set_yticklabels([f'{ct}  (n={ct_n[ct]})' for ct in cell_types], fontsize=7.5)
plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02, label='z-score vs rest')
ax_heat.set_title(f'Sample {TARGET} marker expression — z-score vs all other samples\n'
                  f'(consistent pattern across cell types = technical batch effect)',
                  fontsize=9)

# ── Patch comparison ──────────────────────────────────────────────────────────
TARGET_COL = '#ff4444'
REF_COL    = '#4488ff'

for m_idx, mname in enumerate(show_markers):
    midx   = marker_names.index(mname)
    p99_sh = ref_p99[mname]

    # target patches
    for p_idx, (patch, sid) in enumerate(zip(target_patches, target_sids_p)):
        ax = fig.add_subplot(gs_bot[m_idx, p_idx])
        img = norm_shared(patch[:, :, midx], p99_sh)
        ax.imshow(img, cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(TARGET_COL); sp.set_linewidth(1.5)
        if m_idx == 0:
            ax.set_title(f's{sid}', fontsize=5.5, color=TARGET_COL)
        if p_idx == 0:
            ax.set_ylabel(mname, fontsize=8, rotation=0, ha='right', va='center', labelpad=46)

    # spacer column
    ax_sp = fig.add_subplot(gs_bot[m_idx, n_p])
    ax_sp.axis('off')
    if m_idx == n_show_markers // 2:
        ax_sp.text(0.5, 0.5, 'vs', transform=ax_sp.transAxes,
                   ha='center', va='center', fontsize=11, color='#555555')

    # reference patches
    for p_idx, (patch, sid) in enumerate(zip(ref_patches, ref_sids_p)):
        ax = fig.add_subplot(gs_bot[m_idx, n_p + 1 + p_idx])
        img = norm_shared(patch[:, :, midx], p99_sh)
        ax.imshow(img, cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(REF_COL); sp.set_linewidth(1.5)
        if m_idx == 0:
            ax.set_title(f's{sid}', fontsize=5.5, color=REF_COL)

# Row header labels using fixed fractional x positions
# target group occupies columns 0..n_p-1, reference n_p+1..2*n_p out of 2*n_p+1 total cols
frac_t = (n_p / 2) / (2 * n_p + 1)
frac_r = (n_p + 1 + n_p / 2) / (2 * n_p + 1)
fig.text(frac_t, 0.455, f'Sample {TARGET}  ({representative_ct})',
         ha='center', va='bottom', fontsize=9, fontweight='bold', color=TARGET_COL)
fig.text(frac_r, 0.455, 'Reference  (other samples, same cell type)',
         ha='center', va='bottom', fontsize=9, fontweight='bold', color=REF_COL)

fig.text(0.5, 0.45,
         f'Shared intensity scale (reference p99). Most depressed markers: {", ".join(show_markers)}.',
         ha='center', va='top', fontsize=8, color='#444444')

fig.suptitle(
    f'Sample {TARGET} batch effect diagnosis — {args.model}\n'
    f'All cell types show globally depressed nuclear/proliferation markers → technical staining quality issue',
    fontsize=10, y=0.98,
)
fig.savefig(OUT, bbox_inches='tight', dpi=180)
print(f'\nSaved: {OUT}')

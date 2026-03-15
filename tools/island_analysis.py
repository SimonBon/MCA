"""
Identify sample-specific islands in UMAP and characterise them by marker expression.

Usage:
    python tools/island_analysis.py [--model FOLDER]

Default model: MIBI_TNBC_CIM_VICReg_Funnel_Large
Saves:  <model_dir>/island_analysis.pdf
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
from sklearn.cluster import DBSCAN

Z = Path('/nobackup/lab_taschner-mandl/simongutwein/z_RUNS')
EXPR = Z / 'MIBI_TNBC_ExprBaseline_mean' / 'val_results.npz'
MARKERS_FILE = '/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/used_markers.txt'

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MIBI_TNBC_CIM_VICReg_Funnel_Large')
args = parser.parse_args()

OUT = Z / args.model / 'island_analysis.pdf'

# ── Load data ─────────────────────────────────────────────────────────────────
umap_data = np.load(Z / args.model / 'umap_embeddings.npz', allow_pickle=True)
val_data  = np.load(Z / args.model / 'val_results.npz',     allow_pickle=True)

xy         = umap_data['embedding']
labels_str = umap_data['labels_str']
sample_ids = val_data['sample_ids']

marker_names = [l.strip() for l in open(MARKERS_FILE)]

expr_data = np.load(EXPR, allow_pickle=True)
expr_feat  = expr_data['features']   # (N, 37) mean expression per cell
expr_sids  = expr_data['sample_ids']
expr_lbls  = expr_data['labels_str']

# ── DBSCAN on 2D UMAP ─────────────────────────────────────────────────────────
db = DBSCAN(eps=0.4, min_samples=5, n_jobs=-1).fit(xy)
cluster_ids = db.labels_
n_clusters = cluster_ids.max() + 1
print(f'DBSCAN: {n_clusters} clusters')

DOMINANCE_THRESH = 0.80
MIN_CELLS, MAX_CELLS = 5, 300

islands = []
for cid in range(n_clusters):
    mask = cluster_ids == cid
    n    = mask.sum()
    if n < MIN_CELLS or n > MAX_CELLS:
        continue
    sids = sample_ids[mask]
    unique, counts = np.unique(sids, return_counts=True)
    top_sample = unique[counts.argmax()]
    dominance  = counts.max() / n
    if dominance >= DOMINANCE_THRESH:
        island_types = set(labels_str[mask])
        ctypes, ccounts = np.unique(labels_str[mask], return_counts=True)
        order = ccounts.argsort()[::-1]
        top_types = ', '.join(f'{ctypes[i]}({ccounts[i]})' for i in order[:2])
        print(f'  Island cluster {cid}: sample={top_sample}, n={n}, '
              f'dominance={dominance:.0%}, types={top_types}')
        islands.append((cid, mask, top_sample, dominance))

if not islands:
    print('No islands found.')
    raise SystemExit

# ── Compute z-scores for each island ──────────────────────────────────────────
island_results = []
for cid, island_mask, top_sample, dominance in islands:
    island_types = set(labels_str[island_mask])
    expr_island_mask = np.array([
        expr_sids[i] == top_sample and expr_lbls[i] in island_types
        for i in range(len(expr_sids))
    ])
    expr_rest_mask = ~expr_island_mask
    island_expr = expr_feat[expr_island_mask]
    rest_expr   = expr_feat[expr_rest_mask]
    rest_mean   = rest_expr.mean(0)
    rest_std    = rest_expr.std(0) + 1e-8
    z = (island_expr.mean(0) - rest_mean) / rest_std
    island_results.append(dict(
        cid=cid, mask=island_mask, sample=top_sample,
        dominance=dominance, z=z,
        island_mean=island_expr.mean(0), rest_mean=rest_mean,
        n_umap=island_mask.sum(), n_expr=expr_island_mask.sum(),
    ))

# ── EGFR expression for all val cells (from ExprBaseline, aligned by cell) ───
# Use ExprBaseline features directly indexed by alignment with val_results
# (same val_indicies → same N cells, but different order)
# Re-derive per-cell EGFR by matching (sample_id, label) probabilistically
# is unreliable; instead use the direct val_results alignment:
# ExprBaseline val_results uses the same dataset.val_indicies → same N cells,
# potentially in different order. We need to re-sort by (sample_id, label_num)
# to align. Instead, use a simpler direct approach: get EGFR from ExprBaseline
# for each (sample_id, label) and map back to UMAP cells via lookup table.
egfr_idx = marker_names.index('EGFR')

# Build lookup: (sample_id, label_str) -> list of EGFR values in ExprBaseline
from collections import defaultdict
egfr_lookup = defaultdict(list)
for i in range(len(expr_sids)):
    egfr_lookup[(expr_sids[i], expr_lbls[i])].append(expr_feat[i, egfr_idx])
egfr_mean_lookup = {k: np.mean(v) for k, v in egfr_lookup.items()}

egfr_per_cell = np.array([
    egfr_mean_lookup.get((sample_ids[i], labels_str[i]), np.nan)
    for i in range(len(sample_ids))
])

# ── Build island label array for UMAP highlight ───────────────────────────────
island_label = np.zeros(len(xy), dtype=int)  # 0 = not island
for idx, (cid, island_mask, top_sample, dominance) in enumerate(islands):
    island_label[island_mask] = idx + 1

# ── Figure layout ──────────────────────────────────────────────────────────────
n_islands = len(island_results)
# rows: UMAP-sample | UMAP-EGFR | UMAP-islands  +  one z-score bar per island
fig = plt.figure(figsize=(6 + 4 * n_islands, 13))
gs = gridspec.GridSpec(
    3, 1 + n_islands,
    figure=fig,
    hspace=0.45, wspace=0.35,
    height_ratios=[1, 1, 1],
)

POINT_KW = dict(s=1.5, linewidths=0, rasterized=True)

# colour palette for samples
all_samples = np.unique(sample_ids)
cmap_samples = plt.cm.tab20
sample_colour = {s: cmap_samples(i / len(all_samples)) for i, s in enumerate(all_samples)}
sample_colours_arr = np.array([sample_colour[s] for s in sample_ids])

# ── Row 0: UMAP coloured by sample ────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
island_highlight = np.zeros(len(xy), dtype=bool)
for _, island_mask, _, _ in islands:
    island_highlight |= island_mask
# background cells first, then island cells on top in bright colour
ax0.scatter(xy[~island_highlight, 0], xy[~island_highlight, 1],
            c=sample_colours_arr[~island_highlight], **POINT_KW, alpha=0.5)
island_colours = ['#ff4500', '#00bfff']
for idx, (cid, island_mask, top_sample, dominance) in enumerate(islands):
    ax0.scatter(xy[island_mask, 0], xy[island_mask, 1],
                s=10, color=island_colours[idx % len(island_colours)],
                label=f'Island {idx+1}: sample {top_sample}', zorder=5,
                linewidths=0)
ax0.set_title('UMAP coloured by patient ID  (islands highlighted)', fontsize=10)
ax0.legend(fontsize=8, markerscale=2)
ax0.set_xlabel('UMAP 1'); ax0.set_ylabel('UMAP 2')
ax0.set_aspect('equal', adjustable='datalim')

# ── Row 1: UMAP coloured by EGFR expression ───────────────────────────────────
ax1 = fig.add_subplot(gs[1, :])
# clip at 95th percentile so island cells (which are high) stand out visually
vmax = np.nanpercentile(egfr_per_cell, 95)
sc = ax1.scatter(xy[~island_highlight, 0], xy[~island_highlight, 1],
                 c=egfr_per_cell[~island_highlight],
                 cmap='viridis', vmin=0, vmax=vmax, **POINT_KW)
# draw island cells last so they sit on top
for idx, (cid, island_mask, top_sample, dominance) in enumerate(islands):
    ax1.scatter(xy[island_mask, 0], xy[island_mask, 1],
                c=egfr_per_cell[island_mask],
                cmap='viridis', vmin=0, vmax=vmax,
                s=10, linewidths=0.5, edgecolors=island_colours[idx % len(island_colours)],
                label=f'Island {idx+1}: sample {top_sample}', zorder=5)
plt.colorbar(sc, ax=ax1, fraction=0.02, pad=0.01, label='mean EGFR expression')
ax1.set_title('UMAP coloured by EGFR expression  (islands outlined)', fontsize=10)
ax1.legend(fontsize=8, markerscale=2)
ax1.set_xlabel('UMAP 1'); ax1.set_ylabel('UMAP 2')
ax1.set_aspect('equal', adjustable='datalim')

# ── Row 2: z-score bar charts, one per island ─────────────────────────────────
N_TOP = 12
for col, res in enumerate(island_results):
    ax = fig.add_subplot(gs[2, col])
    z = res['z']
    order = np.abs(z).argsort()[::-1][:N_TOP]
    names = [marker_names[i] for i in order]
    vals  = z[order]
    colours = ['#d73027' if v > 0 else '#4575b4' for v in vals]
    bars = ax.barh(range(N_TOP)[::-1], vals, color=colours, edgecolor='none')
    ax.set_yticks(range(N_TOP)[::-1])
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('z-score vs rest', fontsize=8)
    ax.set_title(
        f'Island {col+1} — sample {res["sample"]}\n'
        f'n={res["n_umap"]} cells, {res["dominance"]:.0%} from one patient',
        fontsize=9
    )
    ax.tick_params(axis='x', labelsize=8)

# fill remaining columns in row 2 if only 1 island
for col in range(len(island_results), 1 + n_islands):
    fig.add_subplot(gs[2, col]).set_visible(False)

fig.suptitle(f'Sample-specific UMAP islands — {args.model}', fontsize=11, y=1.01)
fig.savefig(OUT, bbox_inches='tight', dpi=150)
print(f'\nSaved: {OUT}')

"""
Identify sample-specific islands in UMAP and characterise them by marker expression.

Usage:
    python tools/island_analysis.py [--model FOLDER]

Default model: MIBI_TNBC_CIM_VICReg_Funnel_Large
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN

Z = Path('/nobackup/lab_taschner-mandl/simongutwein/z_RUNS')
EXPR = Z / 'MIBI_TNBC_ExprBaseline_mean' / 'val_results.npz'
MARKERS_FILE = '/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/used_markers.txt'

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MIBI_TNBC_CIM_VICReg_Funnel_Large')
args = parser.parse_args()

# ── Load UMAP + sample IDs (perfectly aligned) ────────────────────────────────
umap_data = np.load(Z / args.model / 'umap_embeddings.npz', allow_pickle=True)
val_data  = np.load(Z / args.model / 'val_results.npz',     allow_pickle=True)

xy         = umap_data['embedding']          # (N, 2)
labels_str = umap_data['labels_str']         # (N,) cell type
sample_ids = val_data['sample_ids']          # (N,) patient ID

# ── Load marker names ──────────────────────────────────────────────────────────
marker_names = [l.strip() for l in open(MARKERS_FILE)]

# ── Load ExprBaseline features (mean marker expression per cell) ───────────────
expr_data = np.load(EXPR, allow_pickle=True)
expr_feat  = expr_data['features']    # (N, 37) — same N, different ordering
expr_sids  = expr_data['sample_ids']
expr_lbls  = expr_data['labels_str']

# ── DBSCAN clustering on 2D UMAP space ────────────────────────────────────────
# eps tuned to detect small isolated islands; min_samples=5 to ignore noise
db = DBSCAN(eps=0.4, min_samples=5, n_jobs=-1).fit(xy)
cluster_ids = db.labels_   # -1 = noise

n_clusters = cluster_ids.max() + 1
print(f'\nDBSCAN found {n_clusters} clusters (eps=0.4, min_samples=5)\n')

# ── Find sample-dominated islands ─────────────────────────────────────────────
DOMINANCE_THRESH = 0.80   # cluster is an "island" if one sample contributes >80%
MIN_CELLS        = 5
MAX_CELLS        = 300    # ignore large main clusters

print(f'{"Cluster":>8}  {"n":>5}  {"top sample":>12}  {"dominance":>10}  {"cell types (top 3)"}')
print('─' * 80)

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
        # top cell types
        ctypes, ccounts = np.unique(labels_str[mask], return_counts=True)
        order = ccounts.argsort()[::-1]
        top_types = ', '.join(f'{ctypes[i]}({ccounts[i]})' for i in order[:3])
        print(f'{cid:>8}  {n:>5}  {top_sample:>12}  {dominance:>9.1%}  {top_types}')
        islands.append((cid, mask, top_sample, dominance))

if not islands:
    print('No sample-dominated islands found. Try increasing eps.')
    raise SystemExit

# ── For each island: compare marker expression to the rest ────────────────────
print(f'\n{"="*80}')
print('  Marker expression: island cells vs rest  (z-score, ExprBaseline features)')
print(f'{"="*80}')

for cid, island_mask, top_sample, dominance in islands:
    # Find these cells in ExprBaseline by matching (sample_id, cell_type)
    # Build boolean mask over ExprBaseline cells: same sample AND same cell type
    # as any island cell. This is the best proxy without cell indices.
    island_types = set(labels_str[island_mask])

    expr_island_mask = np.array([
        (expr_sids[i] == top_sample and expr_lbls[i] in island_types)
        for i in range(len(expr_sids))
    ])
    expr_rest_mask = ~expr_island_mask

    if expr_island_mask.sum() == 0:
        print(f'\nCluster {cid} (sample {top_sample}): no matching ExprBaseline cells found')
        continue

    island_expr = expr_feat[expr_island_mask]   # (k, 37)
    rest_expr   = expr_feat[expr_rest_mask]     # (m, 37)

    # z-score: how many stds above the rest mean is the island mean?
    rest_mean = rest_expr.mean(axis=0)
    rest_std  = rest_expr.std(axis=0) + 1e-8
    z = (island_expr.mean(axis=0) - rest_mean) / rest_std

    order = np.abs(z).argsort()[::-1]

    print(f'\nCluster {cid}  —  sample {top_sample}  ({dominance:.0%} dominance, '
          f'{island_mask.sum()} UMAP cells, {expr_island_mask.sum()} ExprBaseline cells)')
    print(f'  {"Marker":<30}  {"island_mean":>11}  {"rest_mean":>9}  {"z-score":>8}')
    print(f'  {"─"*30}  {"─"*11}  {"─"*9}  {"─"*8}')
    for i in order[:15]:
        print(f'  {marker_names[i]:<30}  {island_expr.mean(axis=0)[i]:>11.4f}  '
              f'{rest_mean[i]:>9.4f}  {z[i]:>+8.2f}')

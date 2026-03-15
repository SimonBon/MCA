"""
UMAP of a cell type subset coloured by (a) cluster and (b) sample,
with a z-score bar chart per cluster showing the driving markers.

Usage:
    python tools/subpop_umap.py [--model FOLDER] \
        --s1 35 --s2 28 --cell_type "CD4 T" [--k 5]

Saves: <model_dir>/subpop_umap_s<s1>_s<s2>_<celltype>.pdf
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from umap import UMAP
from pathlib import Path

Z        = Path('/nobackup/lab_taschner-mandl/simongutwein/z_RUNS')
MARKERS_FILE = Path('/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/used_markers.txt')
EXPR     = Z / 'MIBI_TNBC_ExprBaseline_mean' / 'val_results.npz'

parser = argparse.ArgumentParser()
parser.add_argument('--model',     default='MIBI_TNBC_CIM_VICReg_Funnel_Large')
parser.add_argument('--s1',        default='35')
parser.add_argument('--s2',        default='28')
parser.add_argument('--cell_type', default='CD4 T')
parser.add_argument('--k',         type=int, default=0,
                    help='Number of clusters. 0 = auto-select by silhouette.')
parser.add_argument('--k_max',     type=int, default=6)
args = parser.parse_args()

S1, S2 = args.s1, args.s2
CT  = args.cell_type
slug = CT.replace(' ', '')
OUT  = Z / args.model / f'subpop_umap_s{S1}_s{S2}_{slug}.pdf'

# ── Load ──────────────────────────────────────────────────────────────────────
marker_names = [l.strip() for l in open(MARKERS_FILE)]

expr_data = np.load(EXPR, allow_pickle=True)
expr_feat  = expr_data['features']
expr_sids  = expr_data['sample_ids']
expr_lbls  = expr_data['labels_str']

sel      = np.array([expr_sids[i] in {S1, S2} and expr_lbls[i] == CT
                     for i in range(len(expr_sids))])
X        = expr_feat[sel]
sids_sel = expr_sids[sel]
print(f'{CT}: s{S1}={( sids_sel==S1).sum()}, s{S2}={(sids_sel==S2).sum()}, total={len(X)}')

# ── PCA ───────────────────────────────────────────────────────────────────────
X_sc = StandardScaler().fit_transform(X)
n_pc = min(20, X_sc.shape[0] - 1, X_sc.shape[1])
X_pc = PCA(n_components=n_pc, random_state=42).fit_transform(X_sc)

# ── Pick k ────────────────────────────────────────────────────────────────────
if args.k > 0:
    best_k = args.k
else:
    sil = {}
    for k in range(2, min(args.k_max + 1, len(X))):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_pc)
        if len(np.unique(labels)) >= 2:
            sil[k] = silhouette_score(X_pc, labels)
            print(f'  k={k}  sil={sil[k]:.3f}')
    best_k = max(sil, key=sil.get)
    print(f'Best k={best_k}')

cluster_labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_pc)

# ── UMAP ──────────────────────────────────────────────────────────────────────
n_neighbors = min(15, len(X) - 1)
xy = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.3,
          random_state=42).fit_transform(X_pc)

# ── Driving markers per cluster ───────────────────────────────────────────────
N_TOP = 10
cluster_z   = {}
cluster_n   = {}
cluster_frac = {}
for c in range(best_k):
    m_c = cluster_labels == c
    m_r = ~m_c
    z   = (X[m_c].mean(0) - X[m_r].mean(0)) / (X[m_r].std(0) + 1e-8)
    cluster_z[c]    = z
    cluster_n[c]    = m_c.sum()
    cluster_frac[c] = (sids_sel[m_c] == S1).mean()

# ── Colours ───────────────────────────────────────────────────────────────────
CLUSTER_COLS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4']
S_COLS = {S1: '#e6393a', S2: '#2277cc'}

# ── Figure: 2 rows × (best_k cols for z-score + 2 UMAP cols) ─────────────────
# Top row: UMAP by cluster (left) | UMAP by sample (right)
# Bottom row: z-score bar per cluster

fig = plt.figure(figsize=(max(10, best_k * 2.5 + 4), 9))
gs  = gridspec.GridSpec(2, best_k, figure=fig,
                        height_ratios=[2.5, 2],
                        hspace=0.45, wspace=0.35,
                        top=0.92, bottom=0.05)

# Merge top-row cells into two wide axes
ax_cl = fig.add_subplot(gs[0, :best_k // 2])
ax_s  = fig.add_subplot(gs[0, best_k // 2:])

# UMAP coloured by cluster
for c in range(best_k):
    m = cluster_labels == c
    ax_cl.scatter(xy[m, 0], xy[m, 1],
                  color=CLUSTER_COLS[c % len(CLUSTER_COLS)],
                  s=22, linewidths=0, alpha=0.85,
                  label=f'C{c} (n={cluster_n[c]}, {cluster_frac[c]:.0%} s{S1})')
ax_cl.set_title(f'{CT} — coloured by cluster', fontsize=10)
ax_cl.legend(fontsize=7.5, markerscale=1.5, loc='best',
             framealpha=0.85, edgecolor='#cccccc')
ax_cl.set_xlabel('UMAP 1', fontsize=9); ax_cl.set_ylabel('UMAP 2', fontsize=9)
ax_cl.set_aspect('equal', adjustable='datalim')
ax_cl.tick_params(labelsize=8)

# UMAP coloured by sample
for sid, col in S_COLS.items():
    m = sids_sel == sid
    ax_s.scatter(xy[m, 0], xy[m, 1],
                 color=col, s=22, linewidths=0, alpha=0.85,
                 label=f'Sample {sid} (n={m.sum()})')
ax_s.set_title(f'{CT} — coloured by sample', fontsize=10)
ax_s.legend(fontsize=8.5, markerscale=1.5, loc='best',
            framealpha=0.85, edgecolor='#cccccc')
ax_s.set_xlabel('UMAP 1', fontsize=9); ax_s.set_ylabel('UMAP 2', fontsize=9)
ax_s.set_aspect('equal', adjustable='datalim')
ax_s.tick_params(labelsize=8)

# Bottom row: z-score bars per cluster
for c in range(best_k):
    ax = fig.add_subplot(gs[1, c])
    z  = cluster_z[c]
    order = np.abs(z).argsort()[::-1][:N_TOP]
    vals  = z[order]
    names = [marker_names[i] for i in order]
    bar_cols = [CLUSTER_COLS[c % len(CLUSTER_COLS)] if v > 0 else '#bbbbbb' for v in vals]
    ax.barh(range(N_TOP)[::-1], vals, color=bar_cols, edgecolor='none')
    ax.set_yticks(range(N_TOP)[::-1])
    ax.set_yticklabels(names, fontsize=7)
    ax.axvline(0, color='black', linewidth=0.7)
    ax.set_title(f'Cluster {c}\n'
                 f'n={cluster_n[c]}, {cluster_frac[c]:.0%} s{S1} / '
                 f'{1-cluster_frac[c]:.0%} s{S2}',
                 fontsize=8, color=CLUSTER_COLS[c % len(CLUSTER_COLS)])
    ax.tick_params(axis='x', labelsize=7)
    if c == 0:
        ax.set_xlabel('z-score vs rest', fontsize=7)

fig.suptitle(
    f'{CT} subpopulations — sample {S1} vs {S2} — {args.model}  '
    f'(k={best_k})',
    fontsize=10, y=0.97,
)
fig.savefig(OUT, bbox_inches='tight', dpi=180)
print(f'Saved: {OUT}')

"""
Find subpopulations within a cell type across two samples.
Clusters cells by marker expression, identifies driving markers per cluster,
and shows patches.

Usage:
    python tools/subpopulation_analysis.py [--model FOLDER] \
        --s1 35 --s2 28 --cell_type "CD4 T" [--k_max 6]

Saves: <model_dir>/subpop_s<s1>_s<s2>_<celltype>.pdf
"""

import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from umap import UMAP
from pathlib import Path

Z            = Path('/nobackup/lab_taschner-mandl/simongutwein/z_RUNS')
H5           = Path('/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/MIBI_TNBC.h5')
MARKERS_FILE = Path('/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/used_markers.txt')
VAL_IDX      = Path('/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/val.txt')
EXPR         = Z / 'MIBI_TNBC_ExprBaseline_mean' / 'val_results.npz'

parser = argparse.ArgumentParser()
parser.add_argument('--model',     default='MIBI_TNBC_CIM_VICReg_Funnel_Large')
parser.add_argument('--s1',        default='35')
parser.add_argument('--s2',        default='28')
parser.add_argument('--cell_type', default='CD4 T')
parser.add_argument('--k_max',     type=int, default=6)
parser.add_argument('--patch_size',type=int, default=32)
parser.add_argument('--n_patches', type=int, default=4)
args = parser.parse_args()

S1, S2 = args.s1, args.s2
CT  = args.cell_type
half = args.patch_size // 2
slug = CT.replace(' ', '')
OUT  = Z / args.model / f'subpop_s{S1}_s{S2}_{slug}.pdf'

# ── Load expression features ──────────────────────────────────────────────────
marker_names = [l.strip() for l in open(MARKERS_FILE)]

expr_data = np.load(EXPR, allow_pickle=True)
expr_feat  = expr_data['features']
expr_sids  = expr_data['sample_ids']
expr_lbls  = expr_data['labels_str']

sel = np.array([expr_sids[i] in {S1, S2} and expr_lbls[i] == CT
                for i in range(len(expr_sids))])
X    = expr_feat[sel]
sids_sel = expr_sids[sel]
print(f'{CT} cells: s{S1}={( sids_sel==S1).sum()}, s{S2}={(sids_sel==S2).sum()}, total={len(X)}')

if len(X) < 10:
    raise SystemExit(f'Too few cells ({len(X)}) to cluster.')

# ── Standardise + PCA ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)
n_pc   = min(20, X_sc.shape[0] - 1, X_sc.shape[1])
pca    = PCA(n_components=n_pc, random_state=42)
X_pc   = pca.fit_transform(X_sc)
print(f'PCA variance explained (first 5): {pca.explained_variance_ratio_[:5].round(3)}')

# ── Pick best k by silhouette ─────────────────────────────────────────────────
sil_scores = {}
for k in range(2, min(args.k_max + 1, len(X))):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pc)
    if len(np.unique(labels)) < 2:
        continue
    sil_scores[k] = silhouette_score(X_pc, labels)
    print(f'  k={k}  silhouette={sil_scores[k]:.3f}')

best_k = max(sil_scores, key=sil_scores.get)
print(f'Best k={best_k} (sil={sil_scores[best_k]:.3f})')

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = km_final.fit_predict(X_pc)

# ── UMAP of these cells ───────────────────────────────────────────────────────
n_neighbors = min(15, len(X) - 1)
reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.3, random_state=42)
xy_sub  = reducer.fit_transform(X_pc)

# ── Driving markers per cluster ───────────────────────────────────────────────
# z-score: cluster mean vs rest (all other CD4 T cells in selection)
N_TOP = 10
cluster_z = {}
for c in range(best_k):
    m_c   = cluster_labels == c
    m_r   = ~m_c
    mu_c  = X[m_c].mean(0)
    mu_r  = X[m_r].mean(0)
    std_r = X[m_r].std(0) + 1e-8
    z     = (mu_c - mu_r) / std_r
    cluster_z[c] = z
    top = np.abs(z).argsort()[::-1][:5]
    s_frac = (sids_sel[m_c] == S1).mean()
    print(f'  Cluster {c} (n={m_c.sum()}, {s_frac:.0%} from s{S1}): '
          + ', '.join(f'{marker_names[i]}({z[i]:+.2f})' for i in top))

# ── Patches ───────────────────────────────────────────────────────────────────
with h5py.File(H5, 'r') as f:
    all_dim1  = f['coords']['DIM1'][:]
    all_dim2  = f['coords']['DIM2'][:]
    all_sids  = f['coords']['sample_id'][:].astype(str)
    all_annots = f['annotation'][()].astype(str)
    all_marker_names = f['marker_names'][:].astype(str)

h5_marker_idx = [int(np.where(all_marker_names == m)[0][0]) for m in marker_names]
val_indices   = np.loadtxt(VAL_IDX, dtype=int)
dim1   = all_dim1[val_indices]
dim2   = all_dim2[val_indices]
h5_sids  = all_sids[val_indices]
h5_annots = all_annots[val_indices]

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

# top 3 markers by mean |z| across all clusters
mean_abs_z = np.mean(np.abs(np.vstack(list(cluster_z.values()))), axis=0)
top3_idx   = mean_abs_z.argsort()[::-1][:3]
top3       = [marker_names[i] for i in top3_idx]
print(f'Patch markers: {top3}')

np.random.seed(42)
cluster_patches = {}
with h5py.File(H5, 'r') as hf:
    for c in range(best_k):
        m_c = cluster_labels == c
        # ExprBaseline and val share the same val_indices ordering for the same
        # (sample_id, cell_type) — match by sample and type
        sample_for_c = sids_sel[m_c]
        s_options = list(set(sample_for_c) & {S1, S2})
        chosen_s  = s_options[0]  # pick whichever sample has more
        if (sample_for_c == S1).sum() >= (sample_for_c == S2).sum():
            chosen_s = S1
        else:
            chosen_s = S2
        idxs = [i for i in range(len(h5_sids))
                if h5_sids[i] == chosen_s and h5_annots[i] == CT]
        chosen = np.random.choice(idxs, min(args.n_patches, len(idxs)), replace=False)
        cluster_patches[c] = [get_patch(hf, h5_sids[i], dim1[i], dim2[i]) for i in chosen]

# shared scale per marker
ref_p99 = {}
for mname in top3:
    midx = marker_names.index(mname)
    all_vals = np.concatenate([p[:, :, midx].ravel()
                               for patches in cluster_patches.values()
                               for p in patches])
    ref_p99[mname] = np.percentile(all_vals, 99) or 1.0

def norm_shared(img, p99):
    return np.clip(img, 0, p99) / p99

# ── Colours ───────────────────────────────────────────────────────────────────
CLUSTER_COLOURS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
                   '#911eb4', '#42d4f4'][:best_k]
SAMPLE_COLOURS  = {'35': '#e6393a', '28': '#2277cc'}
sample_colour_arr = np.array([SAMPLE_COLOURS.get(s, '#888888') for s in sids_sel])

# ── Figure ────────────────────────────────────────────────────────────────────
n_p = args.n_patches
fig = plt.figure(figsize=(5 + best_k * 2.5, 11))

gs = gridspec.GridSpec(3, best_k, figure=fig,
                       height_ratios=[2.2, 2.2, 1.8],
                       hspace=0.45, wspace=0.3,
                       top=0.93, bottom=0.04)

# ── Row 0: UMAP by cluster (left) + UMAP by sample (right) ───────────────────
ax_cl = fig.add_subplot(gs[0, :best_k//2] if best_k > 2 else gs[0, :])
ax_s  = fig.add_subplot(gs[0, best_k//2:] if best_k > 2 else None) if best_k > 2 else None

for c in range(best_k):
    m = cluster_labels == c
    ax_cl.scatter(xy_sub[m, 0], xy_sub[m, 1],
                  color=CLUSTER_COLOURS[c], s=18, linewidths=0,
                  label=f'Cluster {c} (n={m.sum()})', alpha=0.8)
ax_cl.set_title(f'{CT} cells — clusters (k={best_k})', fontsize=9)
ax_cl.legend(fontsize=7, markerscale=1.5)
ax_cl.set_xlabel('UMAP 1', fontsize=8); ax_cl.set_ylabel('UMAP 2', fontsize=8)
ax_cl.tick_params(labelsize=7)
ax_cl.set_aspect('equal', adjustable='datalim')

if ax_s is not None:
    for sid, col in SAMPLE_COLOURS.items():
        m = sids_sel == sid
        ax_s.scatter(xy_sub[m, 0], xy_sub[m, 1],
                     color=col, s=18, linewidths=0,
                     label=f'Sample {sid} (n={m.sum()})', alpha=0.8)
    ax_s.set_title(f'{CT} cells — by sample', fontsize=9)
    ax_s.legend(fontsize=7, markerscale=1.5)
    ax_s.set_xlabel('UMAP 1', fontsize=8); ax_s.set_ylabel('UMAP 2', fontsize=8)
    ax_s.tick_params(labelsize=7)
    ax_s.set_aspect('equal', adjustable='datalim')

# ── Row 1: z-score bar charts per cluster ─────────────────────────────────────
for c in range(best_k):
    ax = fig.add_subplot(gs[1, c])
    z  = cluster_z[c]
    order = np.abs(z).argsort()[::-1][:N_TOP]
    vals  = z[order]
    names = [marker_names[i] for i in order]
    colours = [CLUSTER_COLOURS[c] if v > 0 else '#aaaaaa' for v in vals]
    ax.barh(range(N_TOP)[::-1], vals, color=colours, edgecolor='none')
    ax.set_yticks(range(N_TOP)[::-1])
    ax.set_yticklabels(names, fontsize=7)
    ax.axvline(0, color='black', linewidth=0.7)
    m_c   = cluster_labels == c
    s_frac = (sids_sel[m_c] == S1).mean()
    ax.set_title(f'Cluster {c}  (n={m_c.sum()})\n'
                 f'{s_frac:.0%} s{S1} / {1-s_frac:.0%} s{S2}',
                 fontsize=8, color=CLUSTER_COLOURS[c])
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlabel('z-score vs rest', fontsize=7)

# ── Row 2: patches per cluster (top 3 markers as rows, n_p cols) ──────────────
# Use a sub-GridSpec per cluster column
for c in range(best_k):
    inner = gridspec.GridSpecFromSubplotSpec(
        3, n_p, subplot_spec=gs[2, c], hspace=0.06, wspace=0.04)
    patches = cluster_patches[c]
    for m_idx, mname in enumerate(top3):
        midx = marker_names.index(mname)
        p99  = ref_p99[mname]
        for p_idx, patch in enumerate(patches[:n_p]):
            ax = fig.add_subplot(inner[m_idx, p_idx])
            ax.imshow(norm_shared(patch[:, :, midx], p99),
                      cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(CLUSTER_COLOURS[c]); sp.set_linewidth(1.3)
            if p_idx == 0:
                ax.set_ylabel(mname, fontsize=6.5, rotation=0,
                              ha='right', va='center', labelpad=40)
        if m_idx == 0:
            fig.add_subplot(inner[0, n_p//2]).set_title(
                f'Cluster {c}', fontsize=7.5, color=CLUSTER_COLOURS[c], pad=3)

fig.suptitle(
    f'{CT} subpopulations — s{S1} + s{S2} — {args.model}\n'
    f'Best k={best_k} (silhouette={sil_scores[best_k]:.3f})  |  '
    f'Patch markers: {", ".join(top3)}',
    fontsize=9, y=0.98,
)
fig.savefig(OUT, bbox_inches='tight', dpi=180)
print(f'\nSaved: {OUT}')

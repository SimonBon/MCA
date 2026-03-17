#!/usr/bin/env python3
"""Cluster per-cell marker attribution vectors to discover cell types.

Loads attribution.npz produced by marker_attribution.py, clusters the
[N_cells × n_markers] importance matrix with Leiden (via scanpy), and
produces interpretable outputs — all without using ground-truth labels.

Labels stored in attribution.npz are only used for validation plots.

Optionally accepts --val_results (path to val_results.npz from the training
run) to project cells onto the *original embedding UMAP* rather than the
attribution-space UMAP. Use --ignore / --annotation_map to match the same
filtering applied in marker_attribution.py so cell counts align.

Usage:
    # Attribution-space UMAP (default)
    python tools/cluster_attribution.py \\
        --attribution /nobackup/.../marker_attribution/CODEX_cHL_CIM_Funnel/attribution.npz \\
        --out         /nobackup/.../marker_attribution/CODEX_cHL_CIM_Funnel/clusters \\
        --resolution  0.5

    # Original embedding UMAP coloured by marker attribution
    python tools/cluster_attribution.py \\
        --attribution  /nobackup/.../marker_attribution/CODEX_cHL_CIM_Funnel/attribution.npz \\
        --val_results  /nobackup/.../paper_clean/CODEX_cHL/CIM_Funnel_Large/val_results.npz \\
        --ignore       "Seg Artifact,Unidentified,Other" \\
        --annotation_map "Cytotoxic CD8:CD8,TReg:Treg" \\
        --out          /nobackup/.../marker_attribution/CODEX_cHL_CIM_Funnel/clusters_embUMAP \\
        --resolution   0.5

Outputs:
    umap_clusters.png          — UMAP coloured by Leiden cluster
    umap_groundtruth.png       — UMAP coloured by ground-truth label (validation)
    heatmap.png                — mean attribution per cluster × marker
    umap_marker_influence.png  — grid: one panel per marker coloured by IG attribution
    clusters.npz               — cluster assignments + UMAP coords
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    import scanpy as sc
    import anndata as ad
except ImportError:
    sys.exit('scanpy not installed. Run: pip install scanpy')


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_attribution(path):
    d = np.load(path, allow_pickle=True)
    return (
        d['attribution'],               # [N, n_markers]
        d['labels'].astype(str),        # [N]
        d['sample_ids'].astype(str),    # [N]
        d['marker_names'].astype(str),  # [n_markers]
    )


def plot_umap(coords, color_vals, title, out_path, categorical=True, cmap='tab20'):
    fig, ax = plt.subplots(figsize=(8, 7))
    if categorical:
        categories = sorted(set(color_vals))
        palette    = cm.get_cmap(cmap, len(categories))
        cmap_dict  = {c: palette(i) for i, c in enumerate(categories)}
        colors     = [cmap_dict[v] for v in color_vals]
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=2, alpha=0.5, linewidths=0)
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=cmap_dict[c], markersize=6, label=c)
                   for c in categories]
        ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc='upper left',
                  fontsize=7, frameon=False)
    else:
        sc_plot = ax.scatter(coords[:, 0], coords[:, 1], c=color_vals,
                             s=2, alpha=0.5, linewidths=0, cmap=cmap)
        plt.colorbar(sc_plot, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved {out_path}')


def plot_heatmap(attr_matrix, marker_names, cluster_labels, out_path):
    """Mean attribution per cluster × marker, row-normalised."""
    clusters  = sorted(set(cluster_labels), key=lambda x: int(x))
    mat       = np.array([
        attr_matrix[cluster_labels == c].mean(axis=0) for c in clusters
    ])
    mat_norm  = mat / (mat.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(max(10, len(marker_names) * 0.4),
                                    max(4,  len(clusters) * 0.4)))
    im = ax.imshow(mat_norm, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(marker_names)))
    ax.set_xticklabels(marker_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f'Cluster {c}' for c in clusters], fontsize=8)
    ax.set_title('Mean marker attribution per cluster (row-normalised)')
    plt.colorbar(im, ax=ax, label='Relative IG attribution')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved {out_path}')


def plot_marker_panels(attr_matrix, marker_names, coords, out_path, percentile=99):
    """Grid of UMAPs: one panel per marker, coloured by attribution intensity.

    Colour scale is per-marker (0 → 99th percentile) so rare high-attribution
    cells stand out. Row-normalised attribution so total influence sums to 1
    per cell (removes cell-size bias).
    """
    # Row-normalise so each cell's attributions sum to 1
    attr_norm = attr_matrix / (attr_matrix.sum(axis=1, keepdims=True) + 1e-8)

    n = len(marker_names)
    ncols = 8
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.2, nrows * 2.0))
    axes = axes.flatten()

    for i, name in enumerate(marker_names):
        ax  = axes[i]
        val = attr_norm[:, i]
        vmax = np.percentile(val, percentile)

        sc_plot = ax.scatter(coords[:, 0], coords[:, 1],
                             c=val, s=1, alpha=0.6, linewidths=0,
                             cmap='YlOrRd', vmin=0, vmax=vmax)
        ax.set_title(name, fontsize=7, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused panels
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Marker attribution influence on UMAP\n(row-normalised IG, per-marker colour scale)',
                 fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out_path}')



def print_cluster_summary(attr_matrix, marker_names, cluster_labels, gt_labels):
    clusters = sorted(set(cluster_labels), key=lambda x: int(x))
    print('\n── Cluster summary ──────────────────────────────────────────────')
    for c in clusters:
        mask     = cluster_labels == c
        n        = mask.sum()
        # Top markers by mean attribution
        mean_attr = attr_matrix[mask].mean(axis=0)
        top3_idx  = np.argsort(mean_attr)[::-1][:3]
        top3      = ', '.join(f'{marker_names[i]}' for i in top3_idx)
        # Most common ground-truth label
        gt_in_cluster = gt_labels[mask]
        unique, counts = np.unique(gt_in_cluster, return_counts=True)
        dominant = unique[np.argmax(counts)]
        dom_pct  = counts.max() / n * 100
        print(f'  Cluster {c:>2} | n={n:>5} | top markers: {top3:40s} | '
              f'GT majority: {dominant} ({dom_pct:.0f}%)')


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--attribution',    required=True, help='Path to attribution.npz')
    p.add_argument('--out',            required=True)
    p.add_argument('--resolution',     type=float, default=0.5,
                   help='Leiden resolution — higher = more clusters')
    p.add_argument('--n_neighbors',    type=int,   default=15)
    p.add_argument('--n_pcs',          type=int,   default=20,
                   help='PCA dims before KNN (0 = skip PCA, use raw attribution)')
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load ───────────────────────────────────────────────────────────────
    print('Loading attribution matrix...')
    attr, labels, sample_ids, marker_names = load_attribution(args.attribution)
    print(f'  {attr.shape[0]} cells × {attr.shape[1]} markers')
    print(f'  GT classes: {sorted(set(labels))}')

    # ── Build AnnData ──────────────────────────────────────────────────────
    adata = ad.AnnData(X=attr.astype(np.float32))
    adata.var_names  = marker_names
    adata.obs['gt']  = labels
    adata.obs['sid'] = sample_ids

    # ── Normalise rows (relative attribution) ──────────────────────────────
    sc.pp.normalize_total(adata, target_sum=1.0)

    # ── PCA → KNN → Leiden ────────────────────────────────────────────────
    if args.n_pcs > 0:
        print(f'PCA ({args.n_pcs} components)...')
        sc.pp.pca(adata, n_comps=min(args.n_pcs, attr.shape[1] - 1))
        sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, use_rep='X_pca')
    else:
        print('Skipping PCA — using raw attribution for KNN...')
        sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, use_rep='X')

    print(f'Leiden clustering (resolution={args.resolution})...')
    sc.tl.leiden(adata, resolution=args.resolution, key_added='leiden')
    n_clusters = adata.obs['leiden'].nunique()
    print(f'  Found {n_clusters} clusters')

    # ── UMAP ──────────────────────────────────────────────────────────────
    d_attr = np.load(args.attribution, allow_pickle=True)
    if 'features' in d_attr:
        print('Computing UMAP from backbone features (embedding space)...')
        features = d_attr['features'].astype(np.float32)
        adata_emb = ad.AnnData(X=features)
        sc.pp.neighbors(adata_emb, n_neighbors=15, use_rep='X', metric='cosine')
        sc.tl.umap(adata_emb)
        coords = adata_emb.obsm['X_umap']
        umap_title_suffix = '(embedding space)'
    else:
        print('Computing UMAP from attribution space (no features in npz)...')
        sc.tl.umap(adata)
        coords = adata.obsm['X_umap']
        umap_title_suffix = '(attribution space)'

    cluster_labels = adata.obs['leiden'].values.astype(str)

    plot_umap(coords, cluster_labels,
              title=f'Leiden clusters (res={args.resolution}, n={n_clusters}) {umap_title_suffix}',
              out_path=out / 'umap_clusters.png')

    plot_umap(coords, labels,
              title=f'Ground-truth labels {umap_title_suffix}',
              out_path=out / 'umap_groundtruth.png')

    plot_umap(coords, sample_ids,
              title=f'Sample IDs {umap_title_suffix}',
              out_path=out / 'umap_samples.png')

    # ── Heatmap ───────────────────────────────────────────────────────────
    plot_heatmap(attr, marker_names, cluster_labels, out / 'heatmap.png')

    # ── Marker influence panels ────────────────────────────────────────────
    plot_marker_panels(attr, marker_names, coords, out / 'umap_marker_influence.png')

    # ── Summary ───────────────────────────────────────────────────────────
    print_cluster_summary(attr, marker_names, cluster_labels, labels)

    # ── Save ──────────────────────────────────────────────────────────────
    np.savez_compressed(
        out / 'clusters.npz',
        cluster_labels = cluster_labels,
        umap_coords    = coords,
        gt_labels      = labels,
        sample_ids     = sample_ids,
        marker_names   = marker_names,
    )
    print(f'\nDone. Outputs in {out}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Cluster embedding space with PhenoGraph/Leiden, run Wilcoxon DE on IG
attribution per cluster, and produce a dot plot.

Dot plot:
  - y-axis: clusters (PhenoGraph)
  - x-axis: markers (top N per cluster by Wilcoxon log-fold-change, FDR < 0.05)
  - dot size:  fraction of cells in cluster with attribution > dataset median
  - dot color: mean row-normalised IG attribution (z-scored across clusters)

Usage:
    python tools/dotplot_attribution.py \
        --attribution /nobackup/.../marker_attribution/CODEX_cHL_CIM_Funnel/attribution.npz \
        --out         /nobackup/.../marker_attribution/CODEX_cHL_CIM_Funnel/dotplot \
        --top_n       5
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

try:
    import scanpy as sc
    import anndata as ad
except ImportError:
    sys.exit('scanpy not installed. Run: pip install scanpy')

try:
    import phenograph
    HAS_PHENOGRAPH = True
except ImportError:
    HAS_PHENOGRAPH = False
    print('phenograph not installed — falling back to Leiden. '
          'Install with: pip install phenograph')


# ── Clustering ─────────────────────────────────────────────────────────────────

def cluster_features(features, k=30, resolution=1.0):
    """PhenoGraph if available, otherwise Leiden on a KNN graph."""
    if HAS_PHENOGRAPH:
        print(f'PhenoGraph clustering (k={k})...')
        communities, _, _ = phenograph.cluster(features, k=k, seed=42)
        labels = communities.astype(str)
        n = len(set(labels))
        print(f'  Found {n} clusters')
        return labels
    else:
        print(f'Leiden clustering (resolution={resolution})...')
        adata = ad.AnnData(X=features.astype(np.float32))
        sc.pp.neighbors(adata, n_neighbors=k, use_rep='X', metric='cosine')
        sc.tl.leiden(adata, resolution=resolution, key_added='leiden')
        labels = adata.obs['leiden'].values.astype(str)
        n = len(set(labels))
        print(f'  Found {n} clusters')
        return labels


# ── Wilcoxon DE ────────────────────────────────────────────────────────────────

def wilcoxon_de(attr_norm, marker_names, cluster_labels):
    """Run Wilcoxon rank-sum (one-vs-rest) per cluster via scanpy.

    Returns AnnData with ranked genes stored in uns['rank_genes_groups'].
    """
    adata = ad.AnnData(X=attr_norm.astype(np.float32))
    adata.var_names   = marker_names
    adata.obs['cluster'] = cluster_labels

    sc.tl.rank_genes_groups(
        adata, groupby='cluster',
        method='wilcoxon',
        key_added='rank_genes',
        use_raw=False,
    )
    return adata


def extract_top_markers(adata, top_n, fdr_cutoff=0.05):
    """Return dict: cluster → list of (marker, logFC, pval_adj) sorted by logFC."""
    rgg    = adata.uns['rank_genes']
    names  = rgg['names']
    logfcs = rgg['logfoldchanges']
    padjs  = rgg['pvals_adj']

    clusters = names.dtype.names
    result   = {}
    for c in clusters:
        markers = []
        for name, lfc, padj in zip(names[c], logfcs[c], padjs[c]):
            if padj < fdr_cutoff and lfc > 0:
                markers.append((name, lfc, padj))
        # already sorted by score descending
        result[c] = markers[:top_n]
    return result


# ── Dot plot ───────────────────────────────────────────────────────────────────

def make_dotplot(attr_norm, marker_names, cluster_labels, top_markers, out_path):
    """Publication-style dot plot.

    Dot size  = fraction of cells in cluster with attribution > dataset median.
    Dot color = mean attribution (z-scored per marker across clusters).
    """
    clusters  = sorted(set(cluster_labels), key=lambda x: int(x) if x.lstrip('-').isdigit() else x)
    marker2idx = {m: i for i, m in enumerate(marker_names)}

    # Collect unique markers (preserve per-cluster order, deduplicate)
    seen, ordered_markers = set(), []
    for c in clusters:
        for m, _, _ in top_markers.get(c, []):
            if m not in seen:
                seen.add(m)
                ordered_markers.append(m)

    if not ordered_markers:
        print('No significant markers found — try relaxing --fdr or --top_n')
        return

    n_clusters = len(clusters)
    n_markers  = len(ordered_markers)

    # Compute mean attribution and fraction-expressing per cell
    medians   = np.median(attr_norm, axis=0)   # per-marker dataset median
    mean_mat  = np.zeros((n_clusters, n_markers))
    frac_mat  = np.zeros((n_clusters, n_markers))

    for ci, c in enumerate(clusters):
        mask = cluster_labels == c
        for mi, m in enumerate(ordered_markers):
            idx = marker2idx[m]
            vals = attr_norm[mask, idx]
            mean_mat[ci, mi] = vals.mean()
            frac_mat[ci, mi] = (vals > medians[idx]).mean()

    # Z-score mean_mat per marker (column) for colour scale
    mu  = mean_mat.mean(axis=0, keepdims=True)
    std = mean_mat.std(axis=0, keepdims=True) + 1e-8
    z_mat = (mean_mat - mu) / std

    # ── Plot ──────────────────────────────────────────────────────────────
    fig_w = max(8, n_markers * 0.55 + 2)
    fig_h = max(4, n_clusters * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    max_size = 300
    cmap = plt.cm.RdBu_r
    vmax = max(abs(z_mat.min()), abs(z_mat.max()))

    for ci, c in enumerate(clusters):
        for mi in range(n_markers):
            size  = frac_mat[ci, mi] * max_size
            color = z_mat[ci, mi]
            ax.scatter(mi, ci, s=size, c=[color], cmap=cmap,
                       vmin=-vmax, vmax=vmax, linewidths=0.3,
                       edgecolors='grey', zorder=2)

    ax.set_xticks(range(n_markers))
    ax.set_xticklabels(ordered_markers, rotation=90, fontsize=8)
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f'Cluster {c}' for c in clusters], fontsize=8)
    ax.set_xlim(-0.5, n_markers - 0.5)
    ax.set_ylim(-0.5, n_clusters - 0.5)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label('Mean IG attribution\n(z-scored per marker)', fontsize=7)

    # Size legend
    for frac, label in [(0.25, '25%'), (0.5, '50%'), (0.75, '75%')]:
        ax.scatter([], [], s=frac * max_size, c='grey', label=label,
                   linewidths=0.3, edgecolors='grey')
    ax.legend(title='Fraction\nabove median', bbox_to_anchor=(1.18, 1),
              loc='upper left', fontsize=7, title_fontsize=7, frameon=False)

    ax.set_title('Marker IG attribution per cluster\n'
                 '(top DE markers, Wilcoxon one-vs-rest, FDR < 0.05)',
                 fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out_path}')


# ── UMAP coloured by cluster ───────────────────────────────────────────────────

def plot_umap_clusters(umap_coords, cluster_labels, gt_labels, out_dir):
    import matplotlib.cm as cm

    for color_vals, fname, title in [
        (cluster_labels, 'umap_phenograph_clusters.png', 'PhenoGraph clusters'),
        (gt_labels,      'umap_groundtruth.png',         'Ground-truth labels'),
    ]:
        categories = sorted(set(color_vals),
                            key=lambda x: int(x) if x.lstrip('-').isdigit() else x)
        palette   = cm.get_cmap('tab20', len(categories))
        cmap_dict = {c: palette(i) for i, c in enumerate(categories)}
        colors    = [cmap_dict[v] for v in color_vals]

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                   c=colors, s=2, alpha=0.5, linewidths=0)
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=cmap_dict[c], markersize=6, label=c)
                   for c in categories]
        ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc='upper left',
                  fontsize=7, frameon=False)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {out_dir / fname}')


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--attribution', required=True, help='Path to attribution.npz')
    p.add_argument('--out',         required=True)
    p.add_argument('--k',           type=int,   default=30,
                   help='KNN for PhenoGraph/Leiden')
    p.add_argument('--resolution',  type=float, default=1.0,
                   help='Leiden resolution (ignored if phenograph available)')
    p.add_argument('--top_n',       type=int,   default=5,
                   help='Top N DE markers per cluster to show')
    p.add_argument('--fdr',         type=float, default=0.05,
                   help='FDR cutoff for Wilcoxon test')
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load ───────────────────────────────────────────────────────────────
    print('Loading attribution.npz...')
    d = np.load(args.attribution, allow_pickle=True)
    attr         = d['attribution'].astype(np.float32)   # [N, M]
    features     = d['features'].astype(np.float32)      # [N, D]
    labels       = d['labels'].astype(str)
    marker_names = d['marker_names'].astype(str)
    umap_coords  = d['umap_coords']                      # [N, 2]
    print(f'  {attr.shape[0]} cells × {attr.shape[1]} markers, features dim={features.shape[1]}')

    # ── Row-normalise attribution ──────────────────────────────────────────
    attr_norm = attr / (attr.sum(axis=1, keepdims=True) + 1e-8)

    # ── Cluster on backbone features ──────────────────────────────────────
    cluster_labels = cluster_features(features, k=args.k, resolution=args.resolution)

    # ── Wilcoxon DE ───────────────────────────────────────────────────────
    print('Running Wilcoxon rank-sum test per cluster...')
    adata = wilcoxon_de(attr_norm, marker_names, cluster_labels)
    top_markers = extract_top_markers(adata, top_n=args.top_n, fdr_cutoff=args.fdr)

    n_sig = sum(len(v) for v in top_markers.values())
    print(f'  {n_sig} significant marker-cluster pairs (FDR < {args.fdr})')

    # ── Dot plot ──────────────────────────────────────────────────────────
    make_dotplot(attr_norm, marker_names, cluster_labels,
                 top_markers, out / 'dotplot_attribution.png')

    # ── UMAP panels ───────────────────────────────────────────────────────
    plot_umap_clusters(umap_coords, cluster_labels, labels, out)

    # ── Save cluster assignments ───────────────────────────────────────────
    np.savez_compressed(
        out / 'phenograph_clusters.npz',
        cluster_labels = cluster_labels,
        gt_labels      = labels,
        umap_coords    = umap_coords,
        marker_names   = marker_names,
    )
    print(f'\nDone. Outputs in {out}')


if __name__ == '__main__':
    main()

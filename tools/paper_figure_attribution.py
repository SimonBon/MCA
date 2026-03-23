#!/usr/bin/env python3
"""Publication-style attribution figure.

Panel A: Ground-truth UMAP + selected marker UMAPs (2×3 grid, shared colorbar).
Panel B: Dot plot — top DE markers per cluster (only clusters > min_cells).

Usage:
    python tools/paper_figure_attribution.py \
        --attribution /nobackup/.../attribution.npz \
        --clusters    /nobackup/.../dotplot/phenograph_clusters.npz \
        --out         /nobackup/.../paper_figure.png \
        --markers     CD30 CD20 CD31 CD68 CD11b
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import scanpy as sc
    import anndata as ad
except ImportError:
    sys.exit('scanpy not installed.')


# ── Colour palette ─────────────────────────────────────────────────────────────

GT_PALETTE = {
    'B':           '#4878CF', 'CD4':         '#A2C8EC',
    'CD8':         '#F18248', 'DC':          '#FCC48D',
    'Endothelial': '#2CA02C', 'Epithelial':  '#98DF8A',
    'Lymphatic':   '#17BECF', 'M1':          '#D62728',
    'M2':          '#F7B6D2', 'Mast':        '#9467BD',
    'Monocyte':    '#C5B0D5', 'NK':          '#8C564B',
    'Neutrophil':  '#C49C94', 'Treg':        '#E377C2',
    'Tumor':       '#BCBD22', 'Other':       '#AAAAAA',
}


# ── Data helpers ───────────────────────────────────────────────────────────────

def row_normalise(attr):
    return attr / (attr.sum(axis=1, keepdims=True) + 1e-8)


def wilcoxon_top_markers(attr_norm, marker_names, cluster_labels, top_n, fdr):
    adata = ad.AnnData(X=attr_norm.astype(np.float32))
    adata.var_names      = marker_names
    adata.obs['cluster'] = cluster_labels
    sc.tl.rank_genes_groups(adata, groupby='cluster', method='wilcoxon',
                             key_added='rank_genes', use_raw=False)
    rgg    = adata.uns['rank_genes']
    result = {}
    for c in rgg['names'].dtype.names:
        markers = []
        for name, lfc, padj in zip(rgg['names'][c],
                                    rgg['logfoldchanges'][c],
                                    rgg['pvals_adj'][c]):
            if padj < fdr and lfc > 0:
                markers.append((name, float(lfc), float(padj)))
        result[c] = markers[:top_n]
    return result


def dominant_gt(cluster_labels, gt_labels, c):
    mask = np.array(cluster_labels) == c
    gt   = np.array(gt_labels)[mask]
    vals, cnts = np.unique(gt, return_counts=True)
    dom  = vals[np.argmax(cnts)]
    pct  = cnts.max() / mask.sum() * 100
    return dom, pct


# ── Panel A ────────────────────────────────────────────────────────────────────

def draw_umap_gt(ax, coords, gt_labels):
    categories = sorted(set(gt_labels))
    for cat in categories:
        mask = np.array(gt_labels) == cat
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=GT_PALETTE.get(cat, '#aaaaaa'),
                   s=0.8, alpha=0.6, linewidths=0, rasterized=True)
    handles = [Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=GT_PALETTE.get(c, '#aaaaaa'),
                       markersize=4.5, label=c)
               for c in categories]
    ax.legend(handles=handles, fontsize=5, frameon=False,
              loc='lower left', ncol=2, handletextpad=0.2,
              columnspacing=0.4, borderpad=0.2)
    ax.set_title('Cell types', fontsize=8, fontweight='bold', pad=3)
    _clean_ax(ax)


def draw_umap_marker(ax, coords, vals, marker_name, vmax):
    sc_plot = ax.scatter(coords[:, 0], coords[:, 1],
                         c=vals, s=0.8, alpha=0.7, linewidths=0,
                         cmap='YlOrRd', vmin=0, vmax=vmax, rasterized=True)
    ax.set_title(marker_name, fontsize=8, fontweight='bold', pad=3)
    _clean_ax(ax)
    return sc_plot


def _clean_ax(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)


# ── Panel B ────────────────────────────────────────────────────────────────────

def draw_dotplot(ax, attr_norm, marker_names, cluster_labels, gt_labels,
                 top_markers, clusters):
    marker2idx = {m: i for i, m in enumerate(marker_names)}

    # Ordered markers: preserve per-cluster DE order, deduplicate
    seen, ordered = set(), []
    for c in clusters:
        for m, _, _ in top_markers.get(c, []):
            if m not in seen:
                seen.add(m); ordered.append(m)

    if not ordered:
        ax.text(0.5, 0.5, 'No significant markers (try relaxing --fdr)',
                ha='center', va='center', transform=ax.transAxes, fontsize=8)
        return

    n_c = len(clusters); n_m = len(ordered)
    medians  = np.median(attr_norm, axis=0)
    mean_mat = np.zeros((n_c, n_m))
    frac_mat = np.zeros((n_c, n_m))

    for ci, c in enumerate(clusters):
        mask = np.array(cluster_labels) == c
        for mi, m in enumerate(ordered):
            idx = marker2idx[m]
            vals = attr_norm[mask, idx]
            mean_mat[ci, mi] = vals.mean()
            frac_mat[ci, mi] = (vals > medians[idx]).mean()

    # Z-score per marker across clusters
    mu    = mean_mat.mean(axis=0, keepdims=True)
    std   = mean_mat.std(axis=0,  keepdims=True) + 1e-8
    z_mat = (mean_mat - mu) / std
    vmax  = np.percentile(np.abs(z_mat), 95)

    cmap     = plt.cm.RdBu_r
    max_size = 140

    for ci in range(n_c):
        for mi in range(n_m):
            s = max(frac_mat[ci, mi] * max_size, 1.0)
            ax.scatter(mi, ci, s=s, c=[z_mat[ci, mi]], cmap=cmap,
                       vmin=-vmax, vmax=vmax,
                       linewidths=0.25, edgecolors='#888888', zorder=2)

    # Y-axis: cluster label + dominant cell type
    ylabels = []
    for c in clusters:
        dom, pct = dominant_gt(cluster_labels, gt_labels, c)
        color    = GT_PALETTE.get(dom, '#aaaaaa')
        ylabels.append((f'C{c}  {dom} ({pct:.0f}%)', color))

    ax.set_yticks(range(n_c))
    ytick_labels = ax.set_yticklabels([y[0] for y in ylabels], fontsize=6.5)
    for tick, (_, col) in zip(ytick_labels, ylabels):
        tick.set_color(col)

    ax.set_xticks(range(n_m))
    ax.set_xticklabels(ordered, rotation=45, ha='right', fontsize=6.5)
    ax.set_xlim(-0.5, n_m - 0.5)
    ax.set_ylim(-0.5, n_c - 0.5)
    ax.grid(True, linewidth=0.25, alpha=0.5, color='#cccccc')
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.012, pad=0.01, aspect=25)
    cb.set_label('Mean IG\n(z-score)', fontsize=6)
    cb.ax.tick_params(labelsize=5)
    cb.outline.set_linewidth(0.4)

    # Size legend
    leg_handles = []
    for frac, lbl in [(0.25, '25%'), (0.5, '50%'), (0.75, '75%')]:
        leg_handles.append(
            Line2D([0], [0], marker='o', color='w', label=lbl,
                   markerfacecolor='#888888', markersize=np.sqrt(frac * max_size))
        )
    ax.legend(handles=leg_handles, title='Fraction\n> median',
              title_fontsize=5.5, fontsize=5.5,
              loc='upper left', bbox_to_anchor=(1.06, 1.0),
              frameon=False, handletextpad=0.2)

    ax.set_title('Marker IG attribution per cluster  '
                 '(Wilcoxon one-vs-rest, FDR < 0.05)',
                 fontsize=8, pad=5)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--attribution', required=True)
    p.add_argument('--clusters',    required=True)
    p.add_argument('--out',         required=True)
    p.add_argument('--markers', nargs='+',
                   default=['CD30', 'CD20', 'CD31', 'CD68', 'CD11b'])
    p.add_argument('--top_n',       type=int,   default=4)
    p.add_argument('--fdr',         type=float, default=0.05)
    p.add_argument('--min_cells',   type=int,   default=300,
                   help='Minimum cluster size to show in dot plot')
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load ───────────────────────────────────────────────────────────────
    d            = np.load(args.attribution, allow_pickle=True)
    attr         = d['attribution'].astype(np.float32)
    marker_names = d['marker_names'].astype(str)
    gt_labels    = d['labels'].astype(str)
    coords       = d['umap_coords']
    attr_norm    = row_normalise(attr)

    dc             = np.load(args.clusters, allow_pickle=True)
    cluster_labels = dc['cluster_labels'].astype(str)

    # Filter clusters by size
    unique, counts = np.unique(cluster_labels, return_counts=True)
    clusters = sorted(
        [c for c, n in zip(unique, counts) if n >= args.min_cells],
        key=lambda x: int(x) if x.lstrip('-').isdigit() else x
    )
    print(f'Showing {len(clusters)}/{len(unique)} clusters (≥{args.min_cells} cells)')

    # Validate markers
    marker2idx = {m: i for i, m in enumerate(marker_names)}
    selected   = [m for m in args.markers if m in marker2idx]
    missing    = [m for m in args.markers if m not in marker2idx]
    if missing:
        print(f'WARNING: markers not found: {missing}')

    # ── Wilcoxon DE (on filtered clusters only) ────────────────────────────
    print('Running Wilcoxon DE...')
    mask_keep      = np.isin(cluster_labels, clusters)
    attr_sub       = attr_norm[mask_keep]
    cluster_sub    = cluster_labels[mask_keep]
    gt_sub         = gt_labels[mask_keep]
    top_markers    = wilcoxon_top_markers(attr_sub, marker_names, cluster_sub,
                                          top_n=args.top_n, fdr=args.fdr)

    # ── Figure layout ──────────────────────────────────────────────────────
    # Panel A: 2 rows × 3 cols UMAP (GT + 5 markers)
    # Panel B: dot plot full width

    n_umap   = 1 + len(selected)          # GT + markers
    ncols_a  = 3
    nrows_a  = int(np.ceil(n_umap / ncols_a))

    umap_sz  = 2.6                        # inches per UMAP panel
    dot_h    = max(4.0, len(clusters) * 0.30 + 1.5)
    fig_w    = ncols_a * umap_sz + 0.4
    fig_h    = nrows_a * umap_sz + dot_h + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[nrows_a * umap_sz, dot_h],
        hspace=0.35,
    )

    # Panel A sub-grid
    gs_a = gridspec.GridSpecFromSubplotSpec(
        nrows_a, ncols_a, subplot_spec=gs[0],
        hspace=0.35, wspace=0.15,
    )

    umap_axes   = []
    umap_panels = ['GT'] + selected        # first is GT, rest are markers
    for i, label in enumerate(umap_panels):
        row, col = divmod(i, ncols_a)
        ax = fig.add_subplot(gs_a[row, col])
        umap_axes.append((ax, label))

    # Hide unused UMAP slots
    for i in range(len(umap_panels), nrows_a * ncols_a):
        row, col = divmod(i, ncols_a)
        fig.add_subplot(gs_a[row, col]).set_visible(False)

    # Draw GT
    draw_umap_gt(umap_axes[0][0], coords, gt_labels)

    # Shared vmax across marker panels
    vmaxes = {}
    for m in selected:
        vals = attr_norm[:, marker2idx[m]]
        vmaxes[m] = np.percentile(vals, 99)

    sc_plots = {}
    for ax, mname in umap_axes[1:]:
        vals = attr_norm[:, marker2idx[mname]]
        sc_plots[mname] = draw_umap_marker(ax, coords, vals, mname, vmaxes[mname])

    # Shared colorbar for marker UMAPs — attach to last marker axis
    last_sc = list(sc_plots.values())[-1]
    last_ax = umap_axes[-1][0]
    divider = make_axes_locatable(last_ax)
    cbar_ax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(last_sc, cax=cbar_ax)
    cb.set_label('IG attribution\n(row-norm.)', fontsize=5.5)
    cb.ax.tick_params(labelsize=5)
    cb.outline.set_linewidth(0.4)

    # Panel B
    ax_dot = fig.add_subplot(gs[1])
    draw_dotplot(ax_dot, attr_sub, marker_names, cluster_sub, gt_sub,
                 top_markers, clusters)

    # Panel labels
    fig.text(0.01, 0.995, 'A', fontsize=11, fontweight='bold', va='top')
    dot_top = dot_h / fig_h
    fig.text(0.01, dot_top + 0.01, 'B', fontsize=11, fontweight='bold', va='top')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()

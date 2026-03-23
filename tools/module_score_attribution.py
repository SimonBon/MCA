#!/usr/bin/env python3
"""Cell type module scoring on IG attribution using CellMarker 2.0.

Workflow:
  1. Download CellMarker 2.0 human markers (or load cached copy)
  2. Filter to immune/stromal cell types relevant to the panel
  3. Cross-reference with the marker panel → keep modules with ≥2 hits
  4. Compute per-cell module scores (scanpy score_genes on attribution matrix)
  5. Assign cell type label = highest-scoring module (with confidence)
  6. Plot UMAP coloured by: (a) module scores, (b) assigned cell type

Usage:
    python tools/module_score_attribution.py \
        --attribution /nobackup/.../attribution.npz \
        --out         /nobackup/.../module_scores \
        [--cellmarker /path/to/cellmarker2_human.txt]  # optional cached file
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
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# Shared palette — must cover all GT classes + module names (they are the same).
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

try:
    import pandas as pd
except ImportError:
    sys.exit('pandas not installed.')

try:
    import scanpy as sc
    import anndata as ad
except ImportError:
    sys.exit('scanpy not installed.')

try:
    import urllib.request
    import io
except ImportError:
    pass


# ── CellMarker download ────────────────────────────────────────────────────────

CELLMARKER_URL = (
    'http://bio-bigdata.hrbmu.edu.cn/CellMarker/download/Human_cell_markers.txt'
)

# Fallback: manually curated modules from CellMarker 2.0 + literature
# for cHL-relevant cell types, restricted to protein markers
# Data-driven modules: top markers per ground-truth class from IG attribution
# analysis (attribution_per_class.py), validated against known biology.
# Source: CIM_Funnel_Large IG attribution on CODEX_cHL val set.
# Modules named to match GT class labels exactly for consistent coloring.
DATA_DRIVEN_MODULES = {
    'B':           ['CD20', 'CD45RA'],          # CD45RA(1.08) > HLA-DR(0.75)
    'CD4':         ['CD4', 'CD45RO'],           # CCR6 near-zero; CD45RO(0.33) = memory CD4
    'CD8':         ['CD8', 'BCL-2'],            # GranzymeB≈0 for CD8 and HIGH in M1; BCL-2(0.39)
    'Treg':        ['CD25', 'LAG-3'],           # LAG-3(1.32) more Treg-specific than CD4
    'NK':          ['CD57', 'CD11b'],           # CD7 is negative(-0.44); CD11b what model uses
    'Monocyte':    ['CD11b', 'IDO-1'],          # IDO-1(0.79) > CD16(0.38)
    'M1':          ['CD68', 'GranzymeB'],       # GranzymeB(1.52) >> HLA-DR(0.30) for M1 in cHL
    'M2':          ['CD163', 'CD206'],          # drop CD68; CD163(2.64)+CD206(1.27) most specific
    'DC':          ['CD11c', 'HLA-DR'],         # both solid
    'Mast':        ['CD44', 'MCT'],             # perfect (4.60, 4.35)
    'Neutrophil':  ['CD15', 'CD11b'],           # CD11b(1.48) > CD16(0.90)
    'Endothelial': ['CD31', 'b-Catenin', 'Collagen4'],  # Collagen4(2.12) > Vimentin(1.24)
    'Lymphatic':   ['Podoplanin'],               # Podoplanin(3.33) is the only specific lymphatic marker in panel
    'Tumor':       ['CD30', 'CD5'],
    'Epithelial':  ['Cytokeritin', 'MUC-1', 'Vimentin'],  # canonical epithelial markers; MUC-1(EMA) co-expressed with CK in epithelium
}

FALLBACK_MODULES = DATA_DRIVEN_MODULES

# KRONOS18 panel (18 markers): CD11b, CD11c, CD15, CD163, CD20, CD206, CD30,
# CD31, CD4, CD56, CD68, CD7, CD8, Cytokeritin, FoxP3, MCT, Podoplanin
# Key gains vs full panel: FoxP3 (canonical Treg) and CD56 (canonical NK)
# Key losses: CD45RA/RO, GranzymeB, BCL-2, CD25, LAG-3, HLA-DR, b-Catenin, etc.
KRONOS18_MODULES = {
    'B':           ['CD20'],                   # only B-cell marker available
    'CD4':         ['CD4', 'CD7'],             # CD7 pan-T but best available
    'CD8':         ['CD8'],                    # CD8 alone (no BCL-2/GranzymeB)
    'Treg':        ['FoxP3', 'CD4'],           # FoxP3 canonical Treg — better than full panel
    'NK':          ['CD56', 'CD11b'],          # CD56 canonical NK — better than full panel
    'Monocyte':    ['CD11b'],                  # CD11b only (no IDO-1/CD16)
    'M1':          ['CD68'],                   # CD68 only (no GranzymeB/HLA-DR)
    'M2':          ['CD163', 'CD206'],         # both available — unchanged
    'DC':          ['CD11c', 'CD206'],         # CD11c primary, CD206 co-expressed on DCs
    'Mast':        ['MCT'],                    # MCT only (no CD44)
    'Neutrophil':  ['CD15', 'CD11b'],          # both available — unchanged
    'Endothelial': ['CD31'],                   # CD31 only (no b-Catenin/Collagen4)
    'Lymphatic':   ['Podoplanin', 'CD206'],    # CD206(1.16) available as second marker
    'Tumor':       ['CD30', 'CD15'],           # CD15(1.78) canonical cHL diagnostic marker alongside CD30
    'Epithelial':  ['Cytokeritin'],            # Cytokeritin only (no MUC-1/Vimentin)
}

# Canonical name mapping: CellMarker names → panel names
ALIAS_MAP = {
    'CD3E': 'CD3', 'CD3D': 'CD3', 'CD3G': 'CD3',
    'PTPRC': 'CD45', 'CD45RA': 'CD45RA', 'CD45RO': 'CD45RO',
    'PECAM1': 'CD31', 'CD34': 'CD34',
    'PDPN': 'Podoplanin',
    'TNFRSF8': 'CD30',
    'MS4A1': 'CD20',
    'ITGAM': 'CD11b', 'ITGAX': 'CD11c',
    'FCGR3A': 'CD16', 'FCGR3B': 'CD16',
    'FCGR2B': 'CD16',
    'IL2RA': 'CD25',
    'B3GAT1': 'CD57',
    'NCR1': 'NK',
    'GZMB': 'GranzymeB',
    'TRAC': 'TCRb', 'TRBC1': 'TCRb', 'TRBC2': 'TCRb',
    'MUC1': 'MUC-1',
    'COL4A1': 'Collagen4', 'COL4A2': 'Collagen4',
    'HLA-DRA': 'HLA-DR', 'HLA-DRB1': 'HLA-DR',
    'CD274': 'PD-L1',
    'PDCD1': 'PD-1',
    'HAVCR2': 'Tim-3',
    'LAG3': 'LAG-3',
    'VIM': 'Vimentin',
    'KRT': 'Cytokeritin',
    'CTNNB1': 'b-Catenin',
    'IDO1': 'IDO-1',
    'SLC16A1': 'MCT',
}

# Cell types to extract from CellMarker (substring match)
TOP_N_MARKERS = 3   # top N markers per module by CellMarker frequency

# Tissues to include (substring match on tissue_type column)
RELEVANT_TISSUES = [
    'blood', 'lymph node', 'lymph', 'spleen', 'bone marrow',
    'thymus', 'tonsil', 'lymphoid',
]

# Coarse cell type groups: output_name → list of cell_name substrings to match
# Order matters — first match wins for a given CellMarker cell_name
COARSE_GROUPS = {
    'B cell':          ['b cell', 'b-cell', 'b lymphocyte'],
    'CD4 T cell':      ['cd4', 'helper t', 'th1', 'th2', 'th17', 'tfh'],
    'CD8 T cell':      ['cd8', 'cytotoxic t', 'ctl'],
    'Treg':            ['regulatory t', 'treg', 'foxp3'],
    'NK cell':         ['natural killer', 'nk cell', 'nk-cell'],
    'Monocyte':        ['monocyte'],
    'Macrophage':      ['macrophage'],
    'Dendritic cell':  ['dendritic cell', 'dendritic-cell'],
    'Mast cell':       ['mast cell'],
    'Neutrophil':      ['neutrophil'],
    'Endothelial':     ['endothelial cell'],
    'Lymphatic endo.': ['lymphatic endothelial'],
    'Tumor (RS cell)': ['reed-sternberg', 'hodgkin', 'classical hodgkin'],
}


def download_cellmarker(url, cache_path):
    print(f'Downloading CellMarker 2.0 from {url}...')
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(data), sep='\t')
        df.to_csv(cache_path, sep='\t', index=False)
        print(f'  Cached to {cache_path}')
        return df
    except Exception as e:
        print(f'  Download failed: {e}')
        return None


def map_to_coarse(cell_name):
    """Map a CellMarker cell_name to a coarse group, or None if not of interest."""
    cl = cell_name.lower()
    for coarse, keywords in COARSE_GROUPS.items():
        if any(kw in cl for kw in keywords):
            return coarse
    return None


def parse_cellmarker(df, panel_markers):
    """Extract coarse cell type modules from CellMarker xlsx.

    - Filters to relevant tissues (blood, lymph node, etc.)
    - Consolidates subtypes into coarse groups
    - Cross-references symbols with panel via direct match + ALIAS_MAP
    - Keeps only markers seen in ≥2 independent CellMarker entries (frequency filter)
    """
    panel_set = set(panel_markers)

    # Normalise columns
    df.columns = [c.strip() for c in df.columns]

    # Filter to relevant tissues
    tissue_col = 'tissue_type' if 'tissue_type' in df.columns else 'tissueType'
    df[tissue_col] = df[tissue_col].fillna('').astype(str).str.lower()
    mask = df[tissue_col].apply(
        lambda t: any(rt in t for rt in RELEVANT_TISSUES)
    )
    df = df[mask].copy()
    print(f'  After tissue filter: {len(df)} rows')

    name_col = 'cell_name' if 'cell_name' in df.columns else 'cellName'
    sym_col  = 'Symbol'    if 'Symbol'    in df.columns else 'marker'

    # Count marker occurrences per coarse group (for frequency filtering)
    from collections import Counter
    coarse_marker_counts = {g: Counter() for g in COARSE_GROUPS}

    for _, row in df.iterrows():
        cell_name = str(row.get(name_col, '')).strip()
        symbol    = str(row.get(sym_col,  '')).strip()
        if not cell_name or cell_name == 'nan':
            continue

        coarse = map_to_coarse(cell_name)
        if coarse is None:
            continue

        # Map symbol to panel marker
        hit = None
        if symbol in panel_set:
            hit = symbol
        elif symbol in ALIAS_MAP and ALIAS_MAP[symbol] in panel_set:
            hit = ALIAS_MAP[symbol]

        if hit:
            coarse_marker_counts[coarse][hit] += 1

    # Build modules: top N markers by frequency per group
    modules = {}
    for coarse, counts in coarse_marker_counts.items():
        top = [m for m, _ in counts.most_common() if counts[m] >= 2][:TOP_N_MARKERS]
        if len(top) >= 2:
            modules[coarse] = sorted(top)

    # Manual additions for cell types not well covered in CellMarker blood/lymph entries
    MANUAL = {
        'Tumor (RS cell)':  ['CD30', 'MUC-1'],
        'Mast cell':        ['CD25', 'CD69'],
        'Lymphatic endo.':  ['Podoplanin', 'CD31'],
        'Neutrophil':       ['CD15', 'CD16'],
    }
    panel_set = set(panel_markers)
    for ct, markers in MANUAL.items():
        hits = [m for m in markers if m in panel_set]
        if len(hits) >= 2 and ct not in modules:
            modules[ct] = hits

    return modules


def build_modules(panel_markers, cellmarker_path=None, out_dir=None):
    """Try CellMarker download, fall back to curated modules."""
    panel_set = set(panel_markers)

    df = None
    if cellmarker_path and Path(cellmarker_path).exists():
        print(f'Loading CellMarker from {cellmarker_path}')
        p = Path(cellmarker_path)
        if p.suffix in ('.xlsx', '.xls'):
            df = pd.read_excel(p)
        else:
            df = pd.read_csv(p, sep='\t', on_bad_lines='skip')
    else:
        cache = (out_dir / 'cellmarker2_human.txt') if out_dir else None
        if cache and cache.exists():
            print(f'Loading cached CellMarker from {cache}')
            df = pd.read_csv(cache, sep='\t', on_bad_lines='skip')
        else:
            df = download_cellmarker(CELLMARKER_URL, cache)

    if df is not None:
        modules = parse_cellmarker(df, panel_markers)
        if modules:
            print(f'  Built {len(modules)} modules from CellMarker 2.0')
            return modules
        print('  No modules passed filter — using fallback.')

    # Fallback: manually curated
    print('Using curated fallback modules.')
    modules = {}
    for ct, markers in FALLBACK_MODULES.items():
        hits = [m for m in markers if m in panel_set]
        if len(hits) >= 2:
            modules[ct] = hits
    return modules


# ── Module scoring ─────────────────────────────────────────────────────────────

def score_modules(attr_norm, marker_names, modules):
    """Score modules as mean of per-marker [0,1]-normalised attribution.

    Input attr_norm is already per-marker normalised to [0,1] (99.9th pct cap),
    so all markers are on equal footing. Clipping negative values to 0 before
    averaging means a weak secondary marker contributes 0 instead of dragging
    the score down. Mean keeps scores size-unbiased across modules.
    Output scores are in [0, 1] and subsequently normalised for plotting.
    """
    marker2idx = {m: i for i, m in enumerate(marker_names)}
    scores = {}

    for ct, markers in modules.items():
        valid = [m for m in markers if m in marker2idx]
        if len(valid) < 1:
            continue
        mod_idx    = [marker2idx[m] for m in valid]
        vals       = np.clip(attr_norm[:, mod_idx], 0, None)
        scores[ct] = vals.mean(axis=1)

        print(f'  {ct:30s} markers={valid}  '
              f'score range [{scores[ct].min():.3f}, {scores[ct].max():.3f}]')

    return scores


def assign_labels(scores):
    """Assign each cell to the highest-scoring module."""
    cts   = list(scores.keys())
    mat   = np.stack([scores[ct] for ct in cts], axis=1)  # [N, n_ct]
    idx   = np.argmax(mat, axis=1)
    labels = np.array([cts[i] for i in idx])

    # Confidence = max score − second max score (margin)
    sorted_scores = np.sort(mat, axis=1)
    confidence = sorted_scores[:, -1] - sorted_scores[:, -2]
    return labels, confidence, mat, cts


# ── Plotting ───────────────────────────────────────────────────────────────────

def _scatter_with_legend(ax, coords, labels, title):
    categories = sorted(set(labels))
    for cat in categories:
        mask = np.array(labels) == cat
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[GT_PALETTE.get(cat, '#AAAAAA')], s=1.2, alpha=0.6,
                   linewidths=0, rasterized=True)
    handles = [Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=GT_PALETTE.get(c, '#AAAAAA'),
                       markersize=5, label=c)
               for c in categories]
    ax.legend(handles=handles, fontsize=6, frameon=False,
              bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def plot_assigned_labels(coords, labels, confidence, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    _scatter_with_legend(axes[0], coords, labels, 'Assigned cell type (highest module score)')

    sc_plot = axes[1].scatter(coords[:, 0], coords[:, 1],
                               c=normalise_01(confidence), s=1.2, alpha=0.7,
                               cmap='magma', vmin=0, vmax=1,
                               linewidths=0, rasterized=True)
    plt.colorbar(sc_plot, ax=axes[1], fraction=0.046, pad=0.04,
                 label='Confidence\n(normalised 0–1)').ax.tick_params(labelsize=6)
    axes[1].set_title('Assignment confidence', fontsize=9)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    plt.tight_layout()
    fig.savefig(out_dir / 'umap_assigned_celltypes.png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved umap_assigned_celltypes.png')


def normalise_01(vals, percentile=99.5):
    """Normalise to [0, 1] clipped at given percentile to avoid outlier dominance."""
    lo = vals.min()
    hi = np.percentile(vals, percentile)
    return np.clip((vals - lo) / (hi - lo + 1e-8), 0, 1)


def plot_module_scores(coords, scores, out_dir):
    """One UMAP panel per module coloured by min-max normalised score [0, 1]."""
    n     = len(scores)
    ncols = 5
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 2.5, nrows * 2.3))
    axes = axes.flatten()

    for i, (ct, vals) in enumerate(scores.items()):
        ax      = axes[i]
        vals_01 = normalise_01(vals)
        sc_p = ax.scatter(coords[:, 0], coords[:, 1],
                          c=vals_01, s=1.0, alpha=0.7, linewidths=0,
                          cmap='magma', vmin=0, vmax=1,
                          rasterized=True)
        plt.colorbar(sc_p, ax=ax, fraction=0.046,
                     pad=0.04).ax.tick_params(labelsize=5)
        ax.set_title(ct, fontsize=7, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Module scores (normalised 0–1, UCell rank-based)',
                 fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / 'umap_module_scores.png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved umap_module_scores.png')


def plot_vs_groundtruth(coords, assigned, gt_labels, out_dir):
    """Side-by-side: ground truth vs assigned — same color per class in both panels."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    _scatter_with_legend(axes[0], coords, gt_labels, 'Ground truth')
    _scatter_with_legend(axes[1], coords, assigned,  'Module score assignment')
    plt.tight_layout()
    fig.savefig(out_dir / 'umap_gt_vs_assigned.png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved umap_gt_vs_assigned.png')


def plot_marker_attribution(coords, attr_norm, marker_names, out_dir):
    """One UMAP panel per marker coloured by per-marker normalised IG attribution [0,1]."""
    # Drop DAPI
    keep = np.array([m.upper() != 'DAPI-01' for m in marker_names])
    attr_plot  = attr_norm[:, keep]
    names_plot = marker_names[keep]

    n     = len(names_plot)
    ncols = 6
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 2.4, nrows * 2.2))
    axes = axes.flatten()

    for i, mname in enumerate(names_plot):
        ax      = axes[i]
        vals_01 = normalise_01(attr_plot[:, i])
        sc_p = ax.scatter(coords[:, 0], coords[:, 1],
                          c=vals_01, s=1.0, alpha=0.7, linewidths=0,
                          cmap='magma', vmin=0, vmax=1,
                          rasterized=True)
        plt.colorbar(sc_p, ax=ax, fraction=0.046,
                     pad=0.04).ax.tick_params(labelsize=5)
        ax.set_title(mname, fontsize=7, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Per-marker IG attribution (row-norm, clipped 99.9th pct, rescaled 0–1)',
                 fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / 'umap_marker_attribution.png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved umap_marker_attribution.png')


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--attribution',   required=True)
    p.add_argument('--out',           required=True)
    p.add_argument('--umap_emb',      default=None,
                   help='Path to umap_embeddings.npz (fallback if umap_coords not in attribution.npz)')
    p.add_argument('--cellmarker',    default=None,
                   help='Path to CellMarker 2.0 xlsx (optional)')
    p.add_argument('--data_driven',   action='store_true',
                   help='Use data-driven modules from IG attribution analysis')
    p.add_argument('--kronos18',      action='store_true',
                   help='Use KRONOS18 panel-adapted modules (18-marker panel)')
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load ───────────────────────────────────────────────────────────────
    print('Loading attribution.npz...')
    d            = np.load(args.attribution, allow_pickle=True)
    attr         = d['attribution'].astype(np.float32)
    marker_names = d['marker_names'].astype(str)
    gt_labels    = d['labels'].astype(str)

    # Load UMAP coords — from attribution.npz or fallback --umap_emb
    if 'umap_coords' in d:
        coords = d['umap_coords']
    elif args.umap_emb:
        emb = np.load(args.umap_emb, allow_pickle=True)
        coords = emb['embedding']
        print(f'  Loaded UMAP coords from {args.umap_emb}: {coords.shape}')
        if coords.shape[0] != attr.shape[0]:
            raise ValueError(f'UMAP rows ({coords.shape[0]}) != attribution rows ({attr.shape[0]})')
    else:
        raise ValueError('No umap_coords in attribution.npz — pass --umap_emb')

    # Row-normalise (each cell sums to 1)
    attr_norm = attr / (attr.sum(axis=1, keepdims=True) + 1e-8)

    # Per-marker normalise to [0, 1] clipped at 99.9th percentile
    # Prevents outlier cells from dominating and puts all markers on equal footing
    p999      = np.percentile(attr_norm, 99.9, axis=0, keepdims=True)
    attr_norm = np.clip(attr_norm, 0, p999) / (p999 + 1e-8)

    print(f'  {attr.shape[0]} cells × {attr.shape[1]} markers')
    print(f'  Panel: {list(marker_names)}')

    # ── Build modules ──────────────────────────────────────────────────────
    print('\nBuilding cell type modules...')
    if args.kronos18:
        print('Using KRONOS18 panel-adapted modules.')
        panel_set = set(marker_names)
        modules = {ct: [m for m in markers if m in panel_set]
                   for ct, markers in KRONOS18_MODULES.items()}
        modules = {ct: m for ct, m in modules.items() if len(m) >= 1}
    elif args.data_driven:
        print('Using data-driven modules from IG attribution analysis.')
        panel_set = set(marker_names)
        modules = {ct: [m for m in markers if m in panel_set]
                   for ct, markers in DATA_DRIVEN_MODULES.items()}
        modules = {ct: m for ct, m in modules.items() if len(m) >= 1}
    else:
        modules = build_modules(marker_names, args.cellmarker, out)
    print(f'\nFinal modules ({len(modules)}):')
    for ct, markers in modules.items():
        print(f'  {ct:30s} → {markers}')

    # Save module definitions
    pd.DataFrame([
        {'cell_type': ct, 'markers': ', '.join(m)}
        for ct, m in modules.items()
    ]).to_csv(out / 'modules.csv', index=False)

    # ── Score ──────────────────────────────────────────────────────────────
    print('\nScoring modules...')
    scores = score_modules(attr_norm, marker_names, modules)

    # ── Assign labels ──────────────────────────────────────────────────────
    assigned, confidence, score_mat, cts = assign_labels(scores)
    print(f'\nCell type assignment:')
    for ct in sorted(set(assigned)):
        n   = (assigned == ct).sum()
        pct = n / len(assigned) * 100
        print(f'  {ct:30s} {n:>5} cells ({pct:.1f}%)')

    # ── Plots ──────────────────────────────────────────────────────────────
    print('\nPlotting...')
    plot_module_scores(coords, scores, out)
    plot_assigned_labels(coords, assigned, confidence, out)
    plot_vs_groundtruth(coords, assigned, gt_labels, out)
    plot_marker_attribution(coords, attr_norm, marker_names, out)

    # ── Save ───────────────────────────────────────────────────────────────
    import shutil
    src_attr = Path(args.attribution)
    dst_attr = out / 'attribution.npz'
    if src_attr.resolve() != dst_attr.resolve():
        shutil.copy2(src_attr, dst_attr)
        print(f'  Copied attribution.npz → {dst_attr}')

    np.savez_compressed(
        out / 'module_scores.npz',
        assigned_labels = assigned,
        confidence      = confidence,
        score_matrix    = score_mat,
        cell_types      = np.array(cts),
        gt_labels       = gt_labels,
        umap_coords     = coords,
    )
    print(f'\nDone. Outputs in {out}')


if __name__ == '__main__':
    main()

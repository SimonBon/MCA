#!/usr/bin/env python3
"""Assemble UMAP summary figures — one PDF per dataset.

Layout: rows = models, columns = folds (or single for CODEX_cHL).
Both cell-type UMAPs (umap.pdf) are shown.
"""

import fitz                          # pymupdf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT = Path('/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/paper_clean')
OUT  = ROOT  # save next to results.csv

MODELS  = ['CIM', 'CIM_LateFusion', 'CIM_Funnel_Large', 'ResNet']
MODEL_LABELS = {
    'CIM':              'CIM\n(channel-indep.)',
    'CIM_LateFusion':   'CIM Late Fusion\n(mix after pool)',
    'CIM_Funnel_Large': 'CIM Funnel\n(mix in phase 2)',
    'ResNet':           'ResNet\n(baseline)',
}

DATASETS = {
    'CODEX_cHL':      {'folds': [None],           'n_folds': 1},
    'MIBI_TNBC':      {'folds': list(range(5)),   'n_folds': 5},
    'IMC_NB_TumorSub':{'folds': list(range(5)),   'n_folds': 5},
}

DPI = 150   # rasterisation DPI for PDF pages


def pdf_to_array(pdf_path, dpi=DPI):
    doc  = fitz.open(str(pdf_path))
    page = doc[0]
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    arr  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    doc.close()
    return arr


def umap_path(ds_name, model, fold):
    if fold is None:
        return ROOT / ds_name / model / 'umap.pdf'
    return ROOT / ds_name / model / f'fold_{fold}' / 'umap.pdf'


def make_figure(ds_name, cfg):
    folds   = cfg['folds']
    n_cols  = len(folds)
    n_rows  = len(MODELS)

    fig_w = max(4 * n_cols + 1.5, 8)
    fig_h = 4 * n_rows + 1.0

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    fig.suptitle(ds_name, fontsize=16, fontweight='bold', y=1.002)

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        left=0.13, right=0.99,
        top=0.97,  bottom=0.01,
        hspace=0.08, wspace=0.04,
    )

    for ri, model in enumerate(MODELS):
        for ci, fold in enumerate(folds):
            ax = fig.add_subplot(gs[ri, ci])
            p  = umap_path(ds_name, model, fold)
            if p.exists():
                img = pdf_to_array(p)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'missing', ha='center', va='center',
                        transform=ax.transAxes, color='red', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

            # column header (fold label)
            if ri == 0:
                title = 'single' if fold is None else f'fold {fold}'
                ax.set_title(title, fontsize=9, pad=3)

            # row label (model)
            if ci == 0:
                ax.set_ylabel(MODEL_LABELS[model], fontsize=8,
                              rotation=0, ha='right', va='center',
                              labelpad=6)

    out_path = OUT / f'umap_summary_{ds_name}.pdf'
    fig.savefig(out_path, bbox_inches='tight', dpi=100)
    plt.close(fig)
    print(f'  saved {out_path}')


for ds_name, cfg in DATASETS.items():
    print(f'Building {ds_name} ...')
    make_figure(ds_name, cfg)

print('Done.')

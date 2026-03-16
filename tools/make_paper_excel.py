#!/usr/bin/env python3
"""Generate paper results Excel with one sheet per dataset.

Sheets:
  <DATASET>          — summary table: models × metrics (mean ± std for CV)
  <DATASET>_per_class — per-class AP for each model (single run or mean across folds)

CODEX_cHL / CODEX_cHL_KRONOS18: single run per model.
MIBI_TNBC / IMC_NB_TumorSub: 5-fold CV, summarised as mean ± std.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path('/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/paper_clean')
CSV  = ROOT / 'results.csv'
XLSX = ROOT / 'results.xlsx'

METRICS = ['lp_balanced', 'lp_macro_f1', 'lp_map', 'knn_balanced',
           'nmi', 'ari', 'clisi_norm', 'ilisi_norm']

METRIC_LABELS = {
    'lp_balanced':  'LP Balanced Acc',
    'lp_macro_f1':  'LP Macro F1',
    'lp_map':       'LP mAP',
    'knn_balanced': 'kNN Balanced Acc',
    'nmi':          'NMI',
    'ari':          'ARI',
    'clisi_norm':   'cLISI (norm)',
    'ilisi_norm':   'iLISI (norm)',
}

MODEL_ORDER = ['CIM', 'CIM_LateFusion', 'CIM_Funnel_Large', 'ResNet']
DATASETS    = ['CODEX_cHL', 'CODEX_cHL_KRONOS18', 'MIBI_TNBC', 'IMC_NB_TumorSub']

# ── Load CSV ──────────────────────────────────────────────────────────────────
rows = list(csv.DictReader(open(CSV)))

data = defaultdict(lambda: defaultdict(list))
for r in rows:
    vals = {m: float(r[m]) for m in METRICS if r[m] != ''}
    data[r['dataset']][r['model']].append(vals)


def fmt(mean, std=None):
    if std is None:
        return f'{mean:.4f}'
    return f'{mean:.4f} ± {std:.4f}'


def metrics_json_path(ds_name, model, fold):
    if fold is None:
        return ROOT / ds_name / model / 'metrics.json'
    return ROOT / ds_name / model / f'fold_{fold}' / 'metrics.json'


def load_per_class_ap(ds_name, model, folds):
    """Return dict: class → mean AP (averaged over folds if CV)."""
    all_ap = defaultdict(list)
    for fold in folds:
        p = metrics_json_path(ds_name, model, fold)
        if not p.exists():
            continue
        m = json.load(open(p))
        per_class = m.get('linear_probe', {}).get('val', {}).get('per_class_ap', {})
        for cls, ap in per_class.items():
            all_ap[cls].append(ap)
    if not all_ap:
        return {}
    return {cls: np.mean(vals) for cls, vals in all_ap.items()}


# ── Folds per dataset ─────────────────────────────────────────────────────────
DS_FOLDS = {
    'CODEX_cHL':       [None],
    'CODEX_cHL_KRONOS18': [None],
    'MIBI_TNBC':       list(range(5)),
    'IMC_NB_TumorSub': list(range(5)),
}


def build_summary_table(ds_name):
    ds_data = data[ds_name]
    rows_out = []
    for model in MODEL_ORDER:
        if model not in ds_data:
            continue
        fold_list = ds_data[model]
        row = {'Model': model,
               'n_classes': next(r for r in rows
                                 if r['dataset'] == ds_name and r['model'] == model)['n_classes']}
        for m in METRICS:
            vals = [f[m] for f in fold_list if m in f]
            if not vals:
                row[METRIC_LABELS[m]] = ''
            elif len(vals) == 1:
                row[METRIC_LABELS[m]] = fmt(vals[0])
            else:
                row[METRIC_LABELS[m]] = fmt(np.mean(vals), np.std(vals))
        rows_out.append(row)
    return pd.DataFrame(rows_out)


def build_per_class_table(ds_name):
    folds = DS_FOLDS[ds_name]
    rows_out = []
    all_classes = None
    model_aps = {}
    for model in MODEL_ORDER:
        ap = load_per_class_ap(ds_name, model, folds)
        if ap:
            model_aps[model] = ap
            if all_classes is None:
                all_classes = sorted(ap.keys())

    if not all_classes:
        return None

    for cls in all_classes:
        row = {'Class': cls}
        for model in MODEL_ORDER:
            if model in model_aps:
                row[model] = f"{model_aps[model].get(cls, ''):.4f}" if model_aps[model].get(cls, '') != '' else ''
        rows_out.append(row)

    # Append mean row
    mean_row = {'Class': 'MEAN (mAP)'}
    for model in MODEL_ORDER:
        if model in model_aps:
            mean_row[model] = f"{np.mean(list(model_aps[model].values())):.4f}"
    rows_out.append(mean_row)

    return pd.DataFrame(rows_out)


# ── Write Excel ───────────────────────────────────────────────────────────────
def autosize(ws):
    for col in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = max_len + 3


with pd.ExcelWriter(XLSX, engine='openpyxl') as writer:
    for ds in DATASETS:
        # Summary sheet
        df_sum = build_summary_table(ds)
        if df_sum.empty:
            print(f'  Skipping {ds} (no data in CSV yet)')
            continue
        sheet_name = ds[:31]  # Excel sheet name limit
        df_sum.to_excel(writer, sheet_name=sheet_name, index=False)
        autosize(writer.sheets[sheet_name])

        # Per-class AP sheet
        df_pc = build_per_class_table(ds)
        if df_pc is not None:
            pc_sheet = (ds + '_AP')[:31]
            df_pc.to_excel(writer, sheet_name=pc_sheet, index=False)
            autosize(writer.sheets[pc_sheet])

print(f'Written {XLSX}')

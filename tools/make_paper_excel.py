#!/usr/bin/env python3
"""Generate paper results Excel with one sheet per dataset.

CODEX_cHL: single run per model, shown as-is.
MIBI_TNBC / IMC_NB_TumorSub: 5-fold CV, summarised as mean ± std.
"""

import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

CSV  = Path('/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/paper_clean/results.csv')
XLSX = Path('/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/paper_clean/results.xlsx')

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
DATASETS    = ['CODEX_cHL', 'MIBI_TNBC', 'IMC_NB_TumorSub']

# ── Load CSV ──────────────────────────────────────────────────────────────────
rows = list(csv.DictReader(open(CSV)))

# Group by dataset → model → list of metric dicts
data = defaultdict(lambda: defaultdict(list))
for r in rows:
    vals = {m: float(r[m]) for m in METRICS if r[m] != ''}
    data[r['dataset']][r['model']].append(vals)


def fmt(mean, std=None):
    if std is None:
        return f'{mean:.4f}'
    return f'{mean:.4f} ± {std:.4f}'


def build_table(ds_name):
    """Return a DataFrame for one dataset."""
    ds_data = data[ds_name]
    rows_out = []
    for model in MODEL_ORDER:
        if model not in ds_data:
            continue
        fold_list = ds_data[model]
        row = {'Model': model, 'n_classes': int(list(fold_list[0].values()) and
               next(r for r in rows if r['dataset'] == ds_name and r['model'] == model)['n_classes'])}
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


# ── Write Excel ───────────────────────────────────────────────────────────────
with pd.ExcelWriter(XLSX, engine='openpyxl') as writer:
    for ds in DATASETS:
        df = build_table(ds)
        df.to_excel(writer, sheet_name=ds, index=False)

        # Auto-size columns
        ws = writer.sheets[ds]
        for col in ws.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = max_len + 3

print(f'Written {XLSX}')

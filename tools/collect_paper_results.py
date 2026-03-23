#!/usr/bin/env python3
"""Collect paper_clean results into a CSV file.

Output columns:
  dataset, model, fold, n_classes,
  lp_balanced, lp_macro,
  nmi, ari,
  clisi_norm, ilisi_norm,
  knn_balanced
"""

import json
import csv
import sys
from pathlib import Path

ROOT    = Path('/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/paper_clean')
OUT_CSV = Path('/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/paper_clean/results.csv')

DATASETS = ['CODEX_cHL', 'CODEX_cHL_KRONOS18', 'MIBI_TNBC', 'IMC_NB_TumorSub']
MODELS   = ['CIM', 'CIM_LateFusion', 'CIM_Funnel_Large', 'ResNet', 'ResNet18']


def load(path):
    with open(path) as f:
        return json.load(f)


def extract(m):
    lp_val  = m.get('linear_probe', {}).get('val',  {})
    knn_val = m.get('knn',          {}).get('val',  {})
    clus    = m.get('clustering',   {}).get('val',  {})
    cl      = m.get('clisi',        {})
    il      = m.get('ilisi',        {})
    return {
        'n_classes':    m.get('n_classes',                        ''),
        'lp_balanced':  lp_val.get('top1_balanced_accuracy',     ''),
        'lp_macro_f1':  lp_val.get('f1',                         ''),
        'lp_map':       lp_val.get('mean_average_precision',     ''),
        'knn_balanced': knn_val.get('top1_balanced_accuracy',    ''),
        'nmi':          clus.get('nmi',                          ''),
        'ari':          clus.get('ari',                          ''),
        'clisi_norm':   cl.get('normalised',                     ''),
        'ilisi_norm':   il.get('normalised',                     ''),
    }


rows = []
missing = []

for ds in DATASETS:
    for model in MODELS:
        mp = ROOT / ds / model
        if not mp.exists():
            missing.append(f'{ds}/{model}')
            continue

        fold_dirs = sorted(mp.glob('fold_*'))
        if fold_dirs:
            for fd in fold_dirs:
                mf = fd / 'metrics.json'
                if not mf.exists():
                    missing.append(str(fd))
                    continue
                row = {'dataset': ds, 'model': model, 'fold': fd.name}
                row.update(extract(load(mf)))
                rows.append(row)
        else:
            mf = mp / 'metrics.json'
            if not mf.exists():
                missing.append(str(mp))
                continue
            row = {'dataset': ds, 'model': model, 'fold': 'single'}
            row.update(extract(load(mf)))
            rows.append(row)

FIELDS = ['dataset', 'model', 'fold', 'n_classes',
          'lp_balanced', 'lp_macro_f1', 'lp_map', 'knn_balanced',
          'nmi', 'ari', 'clisi_norm', 'ilisi_norm']

with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

print(f'Written {len(rows)} rows to {OUT_CSV}')
if missing:
    print(f'Missing ({len(missing)}):')
    for m in missing:
        print(f'  {m}')

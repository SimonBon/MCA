#!/usr/bin/env python3
"""Generate paper results Excel with one sheet per dataset.

Sheets:
  <DATASET>          — summary table: models × metrics (mean ± std for CV)
  <DATASET>_per_class — per-class AP for each model (single run or mean across folds)

CODEX_cHL / CODEX_cHL_KRONOS18: single run per model.
MIBI_TNBC / IMC_NB_TumorSub: 5-fold CV, summarised as mean ± std.

External / baseline models are loaded from LOCAL_ZRUNS and normalised
to match our annotation_map before being merged into the same sheets.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent / 'z_RUNS' / 'paper_clean'
LOCAL_ZRUNS = Path(__file__).parent.parent / 'z_RUNS'
CSV         = ROOT / 'results.csv'
XLSX        = Path(__file__).parent.parent / 'paper_results.xlsx'

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

# ── Annotation maps (same as training configs) ────────────────────────────────
# Applied to normalise external/baseline model class names
ANNOTATION_MAPS = {
    'CODEX_cHL':          {'Cytotoxic CD8': 'CD8', 'TReg': 'Treg'},
    'CODEX_cHL_KRONOS18': {'Cytotoxic CD8': 'CD8', 'TReg': 'Treg'},
    'MIBI_TNBC':          {},
    'IMC_NB_TumorSub':    {},
}

# Classes to drop from external runs (noise / artefact labels)
IGNORE_CLASSES = {'Seg Artifact', 'Unidentified'}

# ── External model locations in LOCAL_ZRUNS ───────────────────────────────────
# Format: {ds_name: [(display_name, folder_name), ...]}
EXTERNAL_MODELS = {
    'CODEX_cHL': [
        ('ExprBaseline', 'CODEX_cHL_ExprBaseline_mean'),
        ('DINOv2',       'CODEX_cHL_DINOv2_vitb14'),
        ('OpenPhenom',   'CODEX_cHL_OpenPhenom'),
        ('UNI',          'CODEX_cHL_UNI'),
    ],
    'MIBI_TNBC': [
        ('ExprBaseline', 'MIBI_TNBC_ExprBaseline_mean'),
        ('DINOv2',       'MIBI_TNBC_DINOv2_vitb14'),
        ('OpenPhenom',   'MIBI_TNBC_OpenPhenom'),
        ('UNI',          'MIBI_TNBC_UNI'),
    ],
    'IMC_NB_TumorSub': [
        ('DINOv2',     'IMC_NB_TumorSub_DINOv2_vitb14'),
        ('OpenPhenom', 'IMC_NB_TumorSub_OpenPhenom'),
        ('UNI',        'IMC_NB_TumorSub_UNI'),
    ],
}

# ── KRONOS per-class AP (Table S7, arXiv:2506.03373, mean over 4 folds) ───────
KRONOS_AP = {
    'CODEX_cHL_KRONOS18': {
        'B':           0.7927,
        'CD4':         0.8369,
        'CD8':         0.9089,
        'DC':          0.6817,
        'Endothelial': 0.8473,
        'Epithelial':  0.6186,
        'Lymphatic':   0.9053,
        'M1':          0.5368,
        'M2':          0.7317,
        'Mast':        0.8357,
        'Monocyte':    0.5737,
        'NK':          0.7747,
        'Neutrophil':  0.7505,
        'Other':       0.6458,
        'Treg':        0.8150,
        'Tumor':       0.9267,
    },
}

# ── Load CSV (our models via paper_clean) ──────────────────────────────────────
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


# ── Folds per dataset ─────────────────────────────────────────────────────────
DS_FOLDS = {
    'CODEX_cHL':          [None],
    'CODEX_cHL_KRONOS18': [None],
    'MIBI_TNBC':          list(range(5)),
    'IMC_NB_TumorSub':    list(range(5)),
}


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


# ── External model loader ─────────────────────────────────────────────────────

def _normalise_class_names(per_class_ap, annotation_map, ignore):
    """Apply annotation_map merges and drop ignore classes.

    When two source names map to the same target (e.g. 'CD8' and
    'Cytotoxic CD8' both → 'CD8'), their APs are averaged.
    """
    merged = defaultdict(list)
    for cls, ap in per_class_ap.items():
        if cls in ignore:
            continue
        target = annotation_map.get(cls, cls)
        merged[target].append(ap)
    return {cls: float(np.mean(vals)) for cls, vals in merged.items()}


def load_external_metrics(ds_name, folder):
    """Load metrics.json from LOCAL_ZRUNS, handle both old and new formats.

    Returns dict with keys matching METRIC_LABELS plus 'per_class_ap'.
    """
    p = LOCAL_ZRUNS / folder / 'metrics.json'
    if not p.exists():
        return None
    d = json.load(open(p))
    amap   = ANNOTATION_MAPS.get(ds_name, {})
    ignore = IGNORE_CLASSES

    # Old flat format: top-level 'val' key
    if 'val' in d and isinstance(d['val'], dict) and 'lp_balanced_accuracy' in d['val']:
        val = d['val']
        pc_raw = val.get('per_class_ap', {})
        pc     = _normalise_class_names(pc_raw, amap, ignore)
        return {
            'lp_balanced':  val.get('lp_balanced_accuracy', ''),
            'lp_macro_f1':  '',
            'lp_map':       np.mean(list(pc.values())) if pc else val.get('mean_ap', ''),
            'knn_balanced': val.get('knn_balanced_accuracy', ''),
            'nmi':          val.get('nmi', ''),
            'ari':          val.get('ari', ''),
            'clisi_norm':   '',
            'ilisi_norm':   '',
            'per_class_ap': pc,
            'n_classes':    len(pc),
        }

    # New nested format: linear_probe.val (EvaluateModelRich)
    lp_val = d.get('linear_probe', {}).get('val', {})
    if lp_val:
        pc_raw = lp_val.get('per_class_ap', {})
        pc     = _normalise_class_names(pc_raw, amap, ignore)
        knn    = d.get('knn', {}).get('val', {}).get('top1_balanced_accuracy', '')
        clus   = d.get('clustering', {}).get('val', {})
        cl     = d.get('clisi', {})
        il     = d.get('ilisi', {})
        return {
            'lp_balanced':  lp_val.get('top1_balanced_accuracy', ''),
            'lp_macro_f1':  lp_val.get('f1', ''),
            'lp_map':       np.mean(list(pc.values())) if pc else lp_val.get('mean_average_precision', ''),
            'knn_balanced': knn,
            'nmi':          clus.get('nmi', ''),
            'ari':          clus.get('ari', ''),
            'clisi_norm':   cl.get('normalised', ''),
            'ilisi_norm':   il.get('normalised', ''),
            'per_class_ap': pc,
            'n_classes':    len(pc),
        }

    return None


def get_external_models(ds_name):
    """Return list of (display_name, metrics_dict) for external models on ds_name."""
    result = []
    for display_name, folder in EXTERNAL_MODELS.get(ds_name, []):
        m = load_external_metrics(ds_name, folder)
        if m is not None:
            result.append((display_name, m))
    return result


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary_table(ds_name):
    ds_data    = data[ds_name]
    ext_models = get_external_models(ds_name)
    rows_out   = []

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

    # Append external / baseline models
    for display_name, ext in ext_models:
        row = {'Model': display_name, 'n_classes': ext.get('n_classes', '')}
        for m in METRICS:
            v = ext.get(m, '')
            row[METRIC_LABELS[m]] = fmt(v) if isinstance(v, float) else ''
        rows_out.append(row)

    return pd.DataFrame(rows_out)


# ── Per-class AP table ────────────────────────────────────────────────────────

def build_per_class_table(ds_name):
    folds      = DS_FOLDS[ds_name]
    ext_models = get_external_models(ds_name)
    kronos_ap  = KRONOS_AP.get(ds_name, {})

    rows_out   = []
    all_classes = None
    model_aps   = {}

    for model in MODEL_ORDER:
        ap = load_per_class_ap(ds_name, model, folds)
        if ap:
            model_aps[model] = ap
            if all_classes is None:
                all_classes = sorted(ap.keys())

    if all_classes is None and not ext_models and not kronos_ap:
        return None

    # Fall back to classes from first external model if our models have none
    if all_classes is None:
        for _, ext in ext_models:
            if ext.get('per_class_ap'):
                all_classes = sorted(ext['per_class_ap'].keys())
                break

    if all_classes is None:
        return None

    # Collect external per-class APs, aligned to all_classes
    ext_aps = {name: ext['per_class_ap'] for name, ext in ext_models
               if ext.get('per_class_ap')}

    for cls in all_classes:
        row = {'Class': cls}
        for model in MODEL_ORDER:
            if model in model_aps:
                v = model_aps[model].get(cls, '')
                row[model] = f'{v:.4f}' if isinstance(v, float) else ''
        for name, ap_dict in ext_aps.items():
            v = ap_dict.get(cls, '')
            row[name] = f'{v:.4f}' if isinstance(v, float) else ''
        if kronos_ap:
            v = kronos_ap.get(cls, '')
            row['KRONOS'] = f'{v:.4f}' if isinstance(v, float) else ''
        rows_out.append(row)

    # Mean (mAP) row
    mean_row = {'Class': 'MEAN (mAP)'}
    for model in MODEL_ORDER:
        if model in model_aps:
            mean_row[model] = f"{np.mean(list(model_aps[model].values())):.4f}"
    for name, ap_dict in ext_aps.items():
        vals = [v for v in ap_dict.values() if isinstance(v, float)]
        mean_row[name] = f'{np.mean(vals):.4f}' if vals else ''
    if kronos_ap:
        mean_row['KRONOS'] = f"{np.mean(list(kronos_ap.values())):.4f}"
    rows_out.append(mean_row)

    return pd.DataFrame(rows_out)


# ── Write Excel ───────────────────────────────────────────────────────────────
def autosize(ws):
    for col in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = max_len + 3


with pd.ExcelWriter(XLSX, engine='openpyxl') as writer:
    for ds in DATASETS:
        df_sum = build_summary_table(ds)
        if df_sum.empty:
            print(f'  Skipping {ds} (no data)')
            continue
        sheet_name = ds[:31]
        df_sum.to_excel(writer, sheet_name=sheet_name, index=False)
        autosize(writer.sheets[sheet_name])

        df_pc = build_per_class_table(ds)
        if df_pc is not None:
            pc_sheet = (ds + '_AP')[:31]
            df_pc.to_excel(writer, sheet_name=pc_sheet, index=False)
            autosize(writer.sheets[pc_sheet])

print(f'Written {XLSX}')

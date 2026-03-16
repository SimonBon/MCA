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

# ── External model locations ───────────────────────────────────────────────────
# Single-split models loaded from LOCAL_ZRUNS (old-format metrics.json).
# CV models (MIBI_TNBC, IMC_NB_TumorSub) loaded from paper_clean via DS_FOLDS.
# Format: {ds_name: [(display_name, folder_or_None), ...]}
# folder_or_None=None means load from paper_clean (CV, fold-averaged).
EXTERNAL_MODELS = {
    'CODEX_cHL': [
        ('ExprBaseline', None),   # paper_clean/CODEX_cHL/ExprBaseline/metrics.json
    ],
    'CODEX_cHL_KRONOS18': [
        ('ExprBaseline', None),   # paper_clean/CODEX_cHL_KRONOS18/ExprBaseline/metrics.json
    ],
    'MIBI_TNBC': [
        ('ExprBaseline', None),   # paper_clean/MIBI_TNBC/ExprBaseline/fold_*/metrics.json
    ],
    'IMC_NB_TumorSub': [
        ('ExprBaseline', None),   # paper_clean/IMC_NB_TumorSub/ExprBaseline/fold_*/metrics.json
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


def load_external_metrics_paper_clean(ds_name, model_name):
    """Load + average metrics from paper_clean (handles single split and CV folds)."""
    folds = DS_FOLDS[ds_name]
    all_pc = defaultdict(list)
    scalar_keys = ['lp_balanced', 'lp_macro_f1', 'lp_map', 'knn_balanced',
                   'nmi', 'ari', 'clisi_norm', 'ilisi_norm']
    scalar_vals = defaultdict(list)

    for fold in folds:
        p = metrics_json_path(ds_name, model_name, fold)
        if not p.exists():
            continue
        d = json.load(open(p))
        amap   = ANNOTATION_MAPS.get(ds_name, {})
        ignore = IGNORE_CLASSES

        lp_val = d.get('linear_probe', {}).get('val', {})
        knn    = d.get('knn', {}).get('val', {}).get('top1_balanced_accuracy', '')
        clus   = d.get('clustering', {}).get('val', {})
        cl     = d.get('clisi', {})
        il     = d.get('ilisi', {})
        pc_raw = lp_val.get('per_class_ap', {})
        pc     = _normalise_class_names(pc_raw, amap, ignore)

        for cls, ap in pc.items():
            all_pc[cls].append(ap)

        def _add(key, val):
            if val != '':
                scalar_vals[key].append(val)

        _add('lp_balanced',  lp_val.get('top1_balanced_accuracy', ''))
        _add('lp_macro_f1',  lp_val.get('f1', ''))
        _add('lp_map',       np.mean(list(pc.values())) if pc else lp_val.get('mean_average_precision', ''))
        _add('knn_balanced', knn)
        _add('nmi',          clus.get('nmi', ''))
        _add('ari',          clus.get('ari', ''))
        _add('clisi_norm',   cl.get('normalised', ''))
        _add('ilisi_norm',   il.get('normalised', ''))

    if not scalar_vals:
        return None

    result = {k: float(np.mean(v)) for k, v in scalar_vals.items()}
    result['per_class_ap'] = {cls: float(np.mean(v)) for cls, v in all_pc.items()}
    result['n_classes']    = len(all_pc)
    return result


def get_external_models(ds_name):
    """Return list of (display_name, metrics_dict) for external models on ds_name."""
    result = []
    for display_name, folder in EXTERNAL_MODELS.get(ds_name, []):
        if folder is None:
            # Load from paper_clean (single split or CV-averaged)
            m = load_external_metrics_paper_clean(ds_name, display_name)
        else:
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


# ── Ablation table ────────────────────────────────────────────────────────────

ABLATION_DIR = LOCAL_ZRUNS / 'ablations' / 'CIM_Funnel'

ABLATION_GROUPS = {
    'Channel Aug':    ['aug_full_8k', 'aug_no_channel', 'aug_no_drop', 'aug_no_shift', 'aug_no_noise', 'aug_drop_005', 'aug_drop_02'],
    'Training Length':['iters_2k', 'iters_4k', 'aug_full_8k'],
    'Model Capacity': ['cap_blocks_4', 'aug_full_8k', 'cap_blocks_12', 'cap_ch_256', 'cap_ch_768'],
}

ABLATION_DISPLAY = {
    'aug_full_8k':    'Full aug (8k) ★',
    'aug_no_channel': 'No channel aug',
    'aug_no_drop':    'No channel drop',
    'aug_no_shift':   'No channel shift',
    'aug_no_noise':   'No noise',
    'aug_drop_005':   'Drop p=0.05',
    'aug_drop_02':    'Drop p=0.20',
    'iters_2k':       '2k iters',
    'iters_4k':       '4k iters',
    'cap_blocks_4':   '4 mix blocks',
    'cap_blocks_12':  '12 mix blocks',
    'cap_ch_256':     '256 mix channels',
    'cap_ch_768':     '768 mix channels',
}


def build_ablation_table():
    rows_out = []
    seen = set()
    for group, variants in ABLATION_GROUPS.items():
        for name in variants:
            key = (group, name)
            if key in seen:
                continue
            seen.add(key)
            p = ABLATION_DIR / name / 'metrics.json'
            if not p.exists():
                continue
            d = json.load(open(p))
            lp_val = d.get('linear_probe', {}).get('val', {})
            knn    = d.get('knn', {}).get('val', {}).get('top1_balanced_accuracy', '')
            clus   = d.get('clustering', {}).get('val', {})
            cl     = d.get('clisi', {})
            il     = d.get('ilisi', {})
            row = {
                'Group':   group,
                'Variant': ABLATION_DISPLAY.get(name, name),
                METRIC_LABELS['lp_balanced']:  fmt(lp_val['top1_balanced_accuracy']) if 'top1_balanced_accuracy' in lp_val else '',
                METRIC_LABELS['lp_macro_f1']:  fmt(lp_val['f1'])                    if 'f1' in lp_val else '',
                METRIC_LABELS['lp_map']:       fmt(lp_val['mean_average_precision']) if 'mean_average_precision' in lp_val else '',
                METRIC_LABELS['knn_balanced']: fmt(knn) if isinstance(knn, float) else '',
                METRIC_LABELS['nmi']:          fmt(clus['nmi']) if 'nmi' in clus else '',
                METRIC_LABELS['ari']:          fmt(clus['ari']) if 'ari' in clus else '',
                METRIC_LABELS['clisi_norm']:   fmt(cl['normalised']) if 'normalised' in cl else '',
                METRIC_LABELS['ilisi_norm']:   fmt(il['normalised']) if 'normalised' in il else '',
            }
            rows_out.append(row)
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

    df_abl = build_ablation_table()
    if not df_abl.empty:
        df_abl.to_excel(writer, sheet_name='Ablations', index=False)
        autosize(writer.sheets['Ablations'])
        print(f'  Ablations sheet: {len(df_abl)} rows')

print(f'Written {XLSX}')

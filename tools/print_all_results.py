"""
Full results comparison across all datasets.
Run from repo root: python tools/print_all_results.py
"""

import json
import math
from pathlib import Path

Z = Path(__file__).parent.parent / 'z_RUNS'


def load(folder):
    p = Z / folder / 'metrics.json'
    if not p.exists():
        return None
    return json.loads(p.read_text())


def metrics(folder):
    d = load(folder)
    if d is None:
        return None
    lp  = d.get('linear_probe', {}).get('val', {}).get('top1_balanced_accuracy')
    nmi = d.get('clustering',   {}).get('val', {}).get('nmi')
    ari = d.get('clustering',   {}).get('val', {}).get('ari')
    si  = d.get('sample_integration', {}).get('score')
    if isinstance(si, float) and math.isnan(si):
        si = None
    return lp, nmi, ari, si


def f(v, w=6):
    return f'{v:.3f}'.rjust(w) if v is not None else '  —  '.rjust(w)

def fsi(v):
    return f'{v:+.3f}'.rjust(7) if v is not None else '  —  '.rjust(7)


# ── table layout ──────────────────────────────────────────────────────────────
# Each dataset block: list of (display_label, folder_name or None for divider)
DATASETS = [
    ('CODEX_cHL  ·  41 markers  ·  17 classes', [
        ('CIM',                   'CODEX_cHL_CIM_VICReg'),
        ('CIM_ProgFusion',        'CODEX_cHL_CIM_ProgFusion_VICReg'),
        ('EarlyFusion32',         'CODEX_cHL_EarlyFusion32_VICReg'),
        ('ResNet',                'CODEX_cHL_ResNet_VICReg'),
        (None, None),
        ('ExprBaseline mean',     'CODEX_cHL_ExprBaseline_mean'),
        ('ExprBaseline mean+std', 'CODEX_cHL_ExprBaseline_mean_std'),
        (None, None),
        ('DINOv2',                'CODEX_cHL_DINOv2_vitb14'),
        ('OpenPhenom',            'CODEX_cHL_OpenPhenom'),
        ('UNI',                   'CODEX_cHL_UNI'),
        (None, None),
        ('Funnel_Large',          'CODEX_cHL_CIM_VICReg_Funnel_Large'),
        ('Funnel_Large_Norm',     'CODEX_cHL_CIM_VICReg_Funnel_Large_Norm'),
    ]),
    ('CODEX_DLBCL2  ·  40 markers  ·  18 classes', [
        ('CIM',                   'CODEX_DLBCL_CIM_VICReg'),
        ('CIM_Norm',              'CODEX_DLBCL_CIM_Norm_VICReg'),
        ('CIM_ProgFusion',        'CODEX_DLBCL_CIM_ProgFusion_VICReg'),
        ('EarlyFusion32',         'CODEX_DLBCL_EarlyFusion32_VICReg'),
        ('ResNet',                'CODEX_DLBCL_ResNet_VICReg'),
        (None, None),
        ('ExprBaseline mean',     'CODEX_DLBCL2_ExprBaseline_mean'),
        ('ExprBaseline mean+std', 'CODEX_DLBCL2_ExprBaseline_mean_std'),
        (None, None),
        ('Funnel_Large',          'CODEX_DLBCL2_CIM_VICReg_Funnel_Large'),
        ('Funnel_Large_Norm',     'CODEX_DLBCL2_CIM_VICReg_Funnel_Large_Norm'),
    ]),
    ('IMC_NeuroblastomaMetaCluster  ·  31 markers  ·  7 classes', [
        ('CIM',                   'IMC_NB_CIM_VICReg'),
        ('EarlyFusion32',         'IMC_NB_EarlyFusion32_VICReg'),
        ('ResNet',                'IMC_NB_ResNet_VICReg'),
        (None, None),
        ('ExprBaseline mean',     'IMC_NeuroblastomaMetaCluster_ExprBaseline_mean'),
        ('ExprBaseline mean+std', 'IMC_NeuroblastomaMetaCluster_ExprBaseline_mean_std'),
        (None, None),
        ('DINOv2',                'IMC_NeuroblastomaMetaCluster_DINOv2_vitb14'),
        ('OpenPhenom',            'IMC_NeuroblastomaMetaCluster_OpenPhenom'),
        ('UNI',                   'IMC_NeuroblastomaMetaCluster_UNI'),
        (None, None),
        ('Funnel_Large',          'IMC_NeuroblastomaMetaCluster_CIM_VICReg_Funnel_Large'),
        ('Funnel_Large_Norm',     'IMC_NeuroblastomaMetaCluster_CIM_VICReg_Funnel_Large_Norm'),
    ]),
    ('IMC_NB_FineCT  ·  31 markers  ·  11 classes', [
        ('CIM',                   'IMC_NB_FineCT_CIM_VICReg'),
        ('CIM_Norm',              'IMC_NB_FineCT_CIM_Norm_VICReg'),
        ('CIM_ProgFusion',        'IMC_NB_FineCT_CIM_ProgFusion_VICReg'),
        ('EarlyFusion32',         'IMC_NB_FineCT_EarlyFusion32_VICReg'),
        ('ResNet',                'IMC_NB_FineCT_ResNet_VICReg'),
        (None, None),
        ('ExprBaseline mean',     'IMC_NB_FineCT_ExprBaseline_mean'),
        ('ExprBaseline mean+std', 'IMC_NB_FineCT_ExprBaseline_mean_std'),
        (None, None),
        ('DINOv2',                'IMC_NB_FineCT_DINOv2_vitb14'),
        ('OpenPhenom',            'IMC_NB_FineCT_OpenPhenom'),
        ('UNI',                   'IMC_NB_FineCT_UNI'),
        (None, None),
        ('Funnel_Large',          'IMC_NB_FineCT_CIM_VICReg_Funnel_Large'),
        ('Funnel_Large_Norm',     'IMC_NB_FineCT_CIM_VICReg_Funnel_Large_Norm'),
    ]),
    ('MIBI_TNBC  ·  37 markers  ·  16 classes', [
        ('CIM',                   'MIBI_TNBC_CIM_VICReg'),
        ('CIM_Norm',              'MIBI_TNBC_CIM_Norm_VICReg'),
        ('CIM_ProgFusion',        'MIBI_TNBC_CIM_ProgFusion_VICReg'),
        ('EarlyFusion32',         'MIBI_TNBC_EarlyFusion32_VICReg'),
        ('ResNet',                'MIBI_TNBC_ResNet_VICReg'),
        (None, None),
        ('ExprBaseline mean',     'MIBI_TNBC_ExprBaseline_mean'),
        ('ExprBaseline mean+std', 'MIBI_TNBC_ExprBaseline_mean_std'),
        (None, None),
        ('DINOv2',                'MIBI_TNBC_DINOv2_vitb14'),
        ('OpenPhenom',            'MIBI_TNBC_OpenPhenom'),
        ('UNI',                   'MIBI_TNBC_UNI'),
        (None, None),
        ('Funnel_Large',          'MIBI_TNBC_CIM_VICReg_Funnel_Large'),
        ('Funnel_Large_Norm',     'MIBI_TNBC_CIM_VICReg_Funnel_Large_Norm'),
    ]),
]

W = 78
HDR = f'  {"Model":<30}  {"LP bal":>7}  {"NMI":>6}  {"ARI":>6}  {"SI":>7}'
DIV = f'  {"─"*30}  {"─"*7}  {"─"*6}  {"─"*6}  {"─"*7}'

for title, rows in DATASETS:
    print(f'\n╔{"═"*W}╗')
    print(f'║  {title:<{W-2}}║')
    print(f'╠{"═"*W}╣')
    print(f'║{HDR:<{W}}║')
    print(f'║{DIV:<{W}}║')
    for label, folder in rows:
        if label is None:
            print(f'║{"":>{W}}║')
            continue
        vals = metrics(folder)
        if vals is None:
            row = f'  {label:<30}  {"(no results)":>34}'
        else:
            lp, nmi, ari, si = vals
            row = f'  {label:<30}  {f(lp,7)}  {f(nmi)}  {f(ari)}  {fsi(si)}'
        print(f'║{row:<{W}}║')
    print(f'╚{"═"*W}╝')

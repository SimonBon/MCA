"""
Compare model results across datasets.
Reads metrics.json from local z_RUNS and (optionally) remote cemm paths.

Usage:
    python tools/compare_results.py
"""

import json
import subprocess
from pathlib import Path

LOCAL_RUNS = Path(__file__).parent.parent / 'z_RUNS'
CEMM_RUNS  = '/nobackup/lab_taschner-mandl/simongutwein/z_RUNS'


def read_local(folder):
    p = LOCAL_RUNS / folder / 'metrics.json'
    if p.exists():
        return json.loads(p.read_text())
    return None


def read_cemm(folder):
    path = f'{CEMM_RUNS}/{folder}/metrics.json'
    r = subprocess.run(
        ['ssh', 'cemm', f'cat {path}'],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        return json.loads(r.stdout)
    return None


def get_metrics(m):
    if m is None:
        return None
    lp  = m.get('linear_probe', {}).get('val', {}).get('top1_balanced_accuracy')
    nmi = m.get('clustering',    {}).get('val', {}).get('nmi')
    ari = m.get('clustering',    {}).get('val', {}).get('ari')
    si  = m.get('sample_integration', {}).get('score')
    return lp, nmi, ari, si


def fmt(v, decimals=3):
    return f'{v:.{decimals}f}' if v is not None else '  —  '


def print_table(dataset_label, rows):
    print(f'\n{"="*75}')
    print(f'  {dataset_label}')
    print(f'{"="*75}')
    print(f'  {"Model":<40}  {"LP":>6}  {"NMI":>6}  {"ARI":>6}  {"SI":>7}')
    print(f'  {"-"*40}  {"-"*6}  {"-"*6}  {"-"*6}  {"-"*7}')
    for label, metrics in rows:
        if metrics is None:
            print(f'  {label:<40}  (no metrics.json)')
            continue
        lp, nmi, ari, si = metrics
        si_str = f'{si:+.3f}' if si is not None else '  —  '
        print(f'  {label:<40}  {fmt(lp):>6}  {fmt(nmi):>6}  {fmt(ari):>6}  {si_str:>7}')
    print()


# ── CODEX_DLBCL2 ──────────────────────────────────────────────────────────────
rows_dlbcl = [
    ('CIM',                   get_metrics(read_local('CODEX_DLBCL_CIM_VICReg'))),
    ('CIM_Norm',              get_metrics(read_local('CODEX_DLBCL_CIM_Norm_VICReg'))),
    ('CIM_ProgFusion',        get_metrics(read_local('CODEX_DLBCL_CIM_ProgFusion_VICReg'))),
    ('EarlyFusion32',         get_metrics(read_local('CODEX_DLBCL_EarlyFusion32_VICReg'))),
    ('ResNet',                get_metrics(read_local('CODEX_DLBCL_ResNet_VICReg'))),
    ('ExprBaseline_mean',     get_metrics(read_local('CODEX_DLBCL2_ExprBaseline_mean'))),
    ('ExprBaseline_mean+std', get_metrics(read_local('CODEX_DLBCL2_ExprBaseline_mean_std'))),
    ('DINOv2',                get_metrics(read_local('CODEX_DLBCL2_DINOv2_vitb14'))),
    ('OpenPhenom',            get_metrics(read_local('CODEX_DLBCL2_OpenPhenom'))),
    ('UNI',                   get_metrics(read_local('CODEX_DLBCL2_UNI'))),
    ('--- new ---',           None),
    ('Funnel_Large',          get_metrics(read_cemm('CODEX_DLBCL2_CIM_VICReg_Funnel_Large'))),
    ('Funnel_Large_Norm',     get_metrics(read_cemm('CODEX_DLBCL2_CIM_VICReg_Funnel_Large_Norm'))),
]
print_table('CODEX_DLBCL2', rows_dlbcl)

# ── IMC_NeuroblastomaMetaCluster (7 coarse classes) ───────────────────────────
rows_imcnb = [
    ('CIM',                   get_metrics(read_local('IMC_NB_CIM_VICReg'))),
    ('EarlyFusion32',         get_metrics(read_local('IMC_NB_EarlyFusion32_VICReg'))),
    ('ResNet',                get_metrics(read_local('IMC_NB_ResNet_VICReg'))),
    ('ExprBaseline_mean',     get_metrics(read_cemm('IMC_NeuroblastomaMetaCluster_ExprBaseline_mean'))),
    ('ExprBaseline_mean+std', get_metrics(read_cemm('IMC_NeuroblastomaMetaCluster_ExprBaseline_mean_std'))),
    ('DINOv2',                get_metrics(read_local('IMC_NeuroblastomaMetaCluster_DINOv2_vitb14'))),
    ('OpenPhenom',            get_metrics(read_local('IMC_NeuroblastomaMetaCluster_OpenPhenom'))),
    ('UNI',                   get_metrics(read_local('IMC_NeuroblastomaMetaCluster_UNI'))),
    ('--- new ---',           None),
    ('Funnel_Large',          get_metrics(read_cemm('IMC_NeuroblastomaMetaCluster_CIM_VICReg_Funnel_Large'))),
    ('Funnel_Large_Norm',     get_metrics(read_cemm('IMC_NeuroblastomaMetaCluster_CIM_VICReg_Funnel_Large_Norm'))),
]
print_table('IMC_NeuroblastomaMetaCluster (7 coarse classes)', rows_imcnb)

# ── IMC_NB_FineCT (11 fine classes) ───────────────────────────────────────────
rows_finect = [
    ('CIM',                   get_metrics(read_local('IMC_NB_FineCT_CIM_VICReg'))),
    ('CIM_Norm',              get_metrics(read_local('IMC_NB_FineCT_CIM_Norm_VICReg'))),
    ('CIM_Norm_Large',        get_metrics(read_local('IMC_NB_FineCT_CIM_Norm_Large_VICReg'))),
    ('CIM_ProgFusion',        get_metrics(read_local('IMC_NB_FineCT_CIM_ProgFusion_VICReg'))),
    ('EarlyFusion32',         get_metrics(read_local('IMC_NB_FineCT_EarlyFusion32_VICReg'))),
    ('ResNet',                get_metrics(read_local('IMC_NB_FineCT_ResNet_VICReg'))),
    ('ExprBaseline_mean',     get_metrics(read_local('IMC_NB_FineCT_ExprBaseline_mean'))),
    ('ExprBaseline_mean+std', get_metrics(read_local('IMC_NB_FineCT_ExprBaseline_mean_std'))),
    ('DINOv2',                get_metrics(read_local('IMC_NB_FineCT_DINOv2_vitb14'))),
    ('OpenPhenom',            get_metrics(read_local('IMC_NB_FineCT_OpenPhenom'))),
    ('UNI',                   get_metrics(read_local('IMC_NB_FineCT_UNI'))),
    ('--- new ---',           None),
    ('Funnel_Large',          get_metrics(read_cemm('IMC_NB_FineCT_CIM_VICReg_Funnel_Large'))),
    ('Funnel_Large_Norm',     get_metrics(read_cemm('IMC_NB_FineCT_CIM_VICReg_Funnel_Large_Norm'))),
]
print_table('IMC_NB_FineCT (11 fine classes)', rows_finect)

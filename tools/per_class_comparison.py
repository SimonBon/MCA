"""
Compare per-class neighbourhood purity between CIM and mixing/fusion models.
Run: python tools/per_class_comparison.py
"""

import json
from pathlib import Path

Z = Path(__file__).parent.parent / 'z_RUNS'

MODELS = {
    'CIM':            'MIBI_TNBC_CIM_VICReg',
    'CIM_Norm':       'MIBI_TNBC_CIM_Norm_VICReg',
    'CIM_ProgFusion': 'MIBI_TNBC_CIM_ProgFusion_VICReg',
    'EarlyFusion32':  'MIBI_TNBC_EarlyFusion32_VICReg',
    'Funnel_Large':   'MIBI_TNBC_CIM_VICReg_Funnel_Large',
}

def load(folder):
    p = Z / folder / 'metrics.json'
    return json.loads(p.read_text()) if p.exists() else None

data = {name: load(folder) for name, folder in MODELS.items()}
model_names = list(data.keys())

# collect all cell types
all_classes = []
for d in data.values():
    if d:
        for c in d['neighbourhood_purity']['per_class']:
            if c not in all_classes:
                all_classes.append(c)
all_classes = sorted(all_classes)

# ── Neighbourhood purity ──────────────────────────────────────────────────────
W = 22
col = 9
print(f'\n{"="*85}')
print(f'  Neighbourhood Purity (k=15)  —  MIBI_TNBC')
print(f'{"="*85}')
hdr = f'  {"Cell Type":<{W}}' + ''.join(f'  {m:>{col}}' for m in model_names) + '   n_cells   CIM best?'
print(hdr)
print('  ' + '─'*W + ('  ' + '─'*col) * len(model_names))

flagged = []
for cls in all_classes:
    vals = {}
    n_cells = None
    for name, d in data.items():
        entry = d['neighbourhood_purity']['per_class'].get(cls) if d else None
        vals[name] = entry['purity'] if entry else None
        if name == 'CIM' and entry:
            n_cells = entry['n_cells']

    row = f'  {cls:<{W}}' + ''.join(
        f'  {v:>{col}.3f}' if v is not None else f'  {"—":>{col}}' for v in vals.values()
    )
    row += f'   {n_cells or "—":>7}'

    cim = vals.get('CIM')
    others = {k: v for k, v in vals.items() if k != 'CIM' and v is not None}
    if cim is not None and others:
        worst_model = min(others, key=lambda k: others[k])
        worst_val   = others[worst_model]
        drop = cim - worst_val
        if drop > 0.04:
            row += f'   ↓{drop:.2f} vs {worst_model}'
            flagged.append((cls, cim, worst_model, worst_val, drop, n_cells))
    print(row)

# ── Summary of problematic classes ───────────────────────────────────────────
print(f'\n{"="*85}')
print(f'  Classes where a mixing model drops >0.04 below CIM purity:')
print(f'{"="*85}')
flagged.sort(key=lambda x: -x[4])
for cls, cim_p, worst_m, worst_p, drop, n in flagged:
    print(f'  {cls:<22}  CIM={cim_p:.3f}  {worst_m}={worst_p:.3f}  drop={drop:.3f}  n={n}')

# ── kNN balanced accuracy ─────────────────────────────────────────────────────
print(f'\n{"="*60}')
print(f'  Overall metrics summary')
print(f'{"="*60}')
print(f'  {"Model":<18}  {"LP bal":>7}  {"kNN bal":>8}  {"NMI":>6}  {"ARI":>6}')
print(f'  {"─"*18}  {"─"*7}  {"─"*8}  {"─"*6}  {"─"*6}')
for name, d in data.items():
    if d is None:
        continue
    lp  = d['linear_probe']['val']['top1_balanced_accuracy']
    knn = d['knn']['val']['top1_balanced_accuracy']
    nmi = d['clustering']['val']['nmi']
    ari = d['clustering']['val']['ari']
    print(f'  {name:<18}  {lp:>7.3f}  {knn:>8.3f}  {nmi:>6.3f}  {ari:>6.3f}')

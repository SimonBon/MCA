#!/usr/bin/env python3
"""
Region Analysis — all models × dataset comparison.

For each dataset, embeds all spatial patches with every paper model,
then compares models on:
  1. k-sweep silhouette (best k per model)
  2. Compartment distinctiveness (between-cluster JS divergence)
  3. Side-by-side spatial maps at best shared k
  4. Cell-type composition per cluster
  5. Cross-patient reproducibility (MIBI_TNBC only)

Embeddings are cached to disk — re-running skips the GPU step.

Usage:
    python tools/region_analysis.py --dataset MIBI_TNBC
    python tools/region_analysis.py --dataset CODEX_cHL_KRONOS18
    python tools/region_analysis.py --dataset CODEX_cHL

All outputs (PDF + PNG) land in z_RUNS/region_analysis_paper/<DATASET>/.
"""

import argparse, os, sys, json, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '8'

import numpy as np
import h5py
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
import umap as umap_lib

from MCA.src.utils import load_checkpoint

# ── Dataset configs ────────────────────────────────────────────────────────────
BASE       = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon'
PAPER_RUNS = f'{BASE}/src/MCA/z_RUNS/paper'
H5_BASE    = f'{BASE}/data/MCI_data/h5_files'

DATASET_CONFIGS = {
    'CODEX_cHL_KRONOS18': dict(
        h5_path     = f'{H5_BASE}/CODEX_cHL/CODEX_cHL.h5',
        markers_path= f'{H5_BASE}/CODEX_cHL/used_markers_KRONOS18.txt',
        ignore_classes = set(),
        models = {
            'CIM':           f'{PAPER_RUNS}/CODEX_cHL_KRONOS18/CIM',
            'CIM_ProgFusion':f'{PAPER_RUNS}/CODEX_cHL_KRONOS18/CIM_ProgFusion',
            'EarlyFusion32': f'{PAPER_RUNS}/CODEX_cHL_KRONOS18/EarlyFusion32',
            'ResNet':        f'{PAPER_RUNS}/CODEX_cHL_KRONOS18/ResNet',
        },
        multi_patient = False,
        k_list        = [2, 4, 6, 8, 10, 12],
        k_repro       = 6,
    ),
    'CODEX_cHL': dict(
        h5_path     = f'{H5_BASE}/CODEX_cHL/CODEX_cHL.h5',
        markers_path= f'{H5_BASE}/CODEX_cHL/used_markers.txt',
        ignore_classes = set(),
        models = {
            'CIM':           f'{PAPER_RUNS}/CODEX_cHL/CIM',
            'CIM_ProgFusion':f'{PAPER_RUNS}/CODEX_cHL/CIM_ProgFusion',
            'EarlyFusion32': f'{PAPER_RUNS}/CODEX_cHL/EarlyFusion32',
            'ResNet':        f'{PAPER_RUNS}/CODEX_cHL/ResNet',
        },
        multi_patient = False,
        k_list        = [2, 4, 6, 8, 10, 12],
        k_repro       = 6,
    ),
    'MIBI_TNBC': dict(
        h5_path     = f'{H5_BASE}/MIBI_TNBC/MIBI_TNBC.h5',
        markers_path= f'{H5_BASE}/MIBI_TNBC/used_markers.txt',
        ignore_classes = {'Unidentified'},
        models = {
            'CIM':           f'{PAPER_RUNS}/MIBI_TNBC/CIM',
            'CIM_ProgFusion':f'{PAPER_RUNS}/MIBI_TNBC/CIM_ProgFusion',
            'EarlyFusion32': f'{PAPER_RUNS}/MIBI_TNBC/EarlyFusion32',
            'ResNet':        f'{PAPER_RUNS}/MIBI_TNBC/ResNet',
        },
        multi_patient = True,
        k_list        = [2, 4, 6, 8, 10],
        k_repro       = 6,
    ),
}

MODEL_COLORS = {
    'CIM':           '#2166AC',
    'CIM_ProgFusion':'#F4A736',
    'EarlyFusion32': '#1A9641',
    'ResNet':        '#D6604D',
}

# ── Global params ──────────────────────────────────────────────────────────────
REGION_PATCH_SIZE = 64
STRIDE            = REGION_PATCH_SIZE // 2
PCA_COMPONENTS    = 64
BATCH_SIZE        = 64
UMAP_MAX_SAMPLES  = 40_000
N_JOBS            = 8


def decode(arr):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])


def savefig(path, dpi=150):
    plt.savefig(str(path).replace('.png', '.pdf'), bbox_inches='tight')
    plt.savefig(str(path), dpi=dpi, bbox_inches='tight')
    plt.close()


# ── 1. Load HDF5 metadata ──────────────────────────────────────────────────────
def load_metadata(cfg):
    ps = REGION_PATCH_SIZE
    with h5py.File(cfg['h5_path'], 'r') as f:
        all_marker_names = decode(f['marker_names'][:])
        cell_sample_ids  = decode(f['coords']['sample_id'][:])
        cell_dim1        = f['coords']['DIM1'][:].astype(int)
        cell_dim2        = f['coords']['DIM2'][:].astype(int)
        cell_annotations = decode(f['annotation'][:])
        unique_samples   = decode(f['sample_ids'][:])

    with open(cfg['markers_path']) as fh:
        used_names = np.array([l.strip() for l in fh if l.strip()])

    marker2idx     = {m: i for i, m in enumerate(all_marker_names)}
    marker_indices = np.array([marker2idx[m] for m in used_names])

    ignore = cfg.get('ignore_classes', set())
    cell_mask = np.array([a not in ignore for a in cell_annotations])

    # Sliding window grid
    all_meta  = []
    sample_dims = {}
    with h5py.File(cfg['h5_path'], 'r') as f:
        for sid in unique_samples:
            H, W = f['data'][sid]['image'].shape[:2]
            sample_dims[sid] = (H, W)
            for y in range(0, H - ps + 1, STRIDE):
                for x in range(0, W - ps + 1, STRIDE):
                    all_meta.append((sid, y, x, H, W))

    patch_index = {(sid, y, x): i for i, (sid, y, x, H, W) in enumerate(all_meta)}
    sample_to_meta_indices = defaultdict(list)
    for i, (sid, *_) in enumerate(all_meta):
        sample_to_meta_indices[sid].append(i)

    print(f'Patches: {len(all_meta):,}  |  Markers: {len(marker_indices)}  |  Patients: {len(unique_samples)}')

    # Cell → patch mapping
    patch_to_cell_annotations = defaultdict(list)
    for cidx in tqdm(range(len(cell_sample_ids)), desc='Mapping cells→patches', unit='cell'):
        if not cell_mask[cidx]:
            continue
        sid = cell_sample_ids[cidx]
        if sid not in sample_dims:
            continue
        cy, cx = int(cell_dim1[cidx]), int(cell_dim2[cidx])
        H, W   = sample_dims[sid]
        ann    = cell_annotations[cidx]
        py_start = (max(0, cy - ps + 1) // STRIDE) * STRIDE
        px_start = (max(0, cx - ps + 1) // STRIDE) * STRIDE
        for py in range(py_start, min(cy + 1, H - ps + 1), STRIDE):
            for px in range(px_start, min(cx + 1, W - ps + 1), STRIDE):
                pidx = patch_index.get((sid, py, px))
                if pidx is not None:
                    patch_to_cell_annotations[pidx].append(ann)

    all_cell_types = sorted({a for anns in patch_to_cell_annotations.values() for a in anns})
    print(f'Cell types: {all_cell_types}')

    return dict(
        all_meta=all_meta,
        patch_index=patch_index,
        sample_to_meta_indices=sample_to_meta_indices,
        sample_dims=sample_dims,
        unique_samples=unique_samples,
        marker_indices=marker_indices,
        patch_to_cell_annotations=patch_to_cell_annotations,
        all_cell_types=all_cell_types,
    )


# ── 2. Embed all patches ───────────────────────────────────────────────────────
@torch.no_grad()
def _flush(batch_list, backbone):
    device = next(backbone.parameters()).device
    t = torch.from_numpy(np.array(batch_list, dtype=np.float32)).to(device)
    feats = backbone(t)
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    return F.normalize(feats.squeeze(-1).squeeze(-1), dim=1).cpu().numpy().astype(np.float16)


def embed_model(work_dir, meta, cfg, cache_dir, device):
    model_name = Path(work_dir).name
    cache_path = cache_dir / f'embeddings_{model_name}.npy'
    if cache_path.exists():
        print(f'  [{model_name}] Loading cached embeddings from {cache_path}')
        return np.load(str(cache_path))

    print(f'  [{model_name}] Embedding {len(meta["all_meta"]):,} patches...')
    result   = load_checkpoint(work_dir, device=device)
    backbone = result['model'].backbone.eval()

    # Determine feature dim
    ps = REGION_PATCH_SIZE
    n_markers = len(meta['marker_indices'])
    with torch.no_grad():
        dummy = torch.zeros(1, n_markers, ps, ps, device=device)
        feat = backbone(dummy)
        if isinstance(feat, (tuple, list)): feat = feat[0]
        feat_dim = feat.squeeze(-1).squeeze(-1).shape[1]
    print(f'  [{model_name}] Feature dim: {feat_dim}')

    embeddings = np.zeros((len(meta['all_meta']), feat_dim), dtype=np.float16)
    ps = REGION_PATCH_SIZE

    with h5py.File(cfg['h5_path'], 'r') as f:
        pbar = tqdm(total=len(meta['all_meta']), desc=f'  [{model_name}]', unit='patch')
        for sid in meta['unique_samples']:
            image   = f['data'][sid]['image'][:].astype(np.float32)[:, :, meta['marker_indices']]
            indices = meta['sample_to_meta_indices'][sid]
            patches, idx_buf = [], []
            for i in indices:
                _, y, x, H, W = meta['all_meta'][i]
                patches.append(image[y:y+ps, x:x+ps, :].transpose(2, 0, 1))
                idx_buf.append(i)
                if len(patches) == BATCH_SIZE:
                    embeddings[idx_buf] = _flush(patches, backbone)
                    pbar.update(len(patches))
                    patches, idx_buf = [], []
            if patches:
                embeddings[idx_buf] = _flush(patches, backbone)
                pbar.update(len(patches))
        pbar.close()

    np.save(str(cache_path), embeddings)
    print(f'  [{model_name}] Saved embeddings → {cache_path}')
    del backbone
    torch.cuda.empty_cache()
    return embeddings


# ── 3. PCA + k-sweep ──────────────────────────────────────────────────────────
def pca_and_sweep(embeddings, k_list, label='model'):
    pca   = PCA(n_components=PCA_COMPONENTS, random_state=42)
    emb_p = pca.fit_transform(embeddings.astype(np.float32))
    print(f'  [{label}] PCA explained: {pca.explained_variance_ratio_.sum():.1%}')

    idx_sub = np.random.RandomState(42).choice(len(emb_p), min(15_000, len(emb_p)), replace=False)
    X_sub   = emb_p[idx_sub]

    sil_scores, inertias = [], []
    for k in k_list:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5, batch_size=2048)
        km.fit(X_sub)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_sub, km.labels_, metric='euclidean', sample_size=5000))

    best_k = k_list[int(np.argmax(sil_scores))]
    print(f'  [{label}] Best k={best_k}  sil={max(sil_scores):.3f}')
    return emb_p, sil_scores, inertias, best_k


# ── 4. Cluster + composition ───────────────────────────────────────────────────
def cluster_and_compose(emb_pca, k, meta):
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10, batch_size=4096, max_iter=300)
    labels = km.fit_predict(emb_pca)
    ct     = meta['all_cell_types']
    p2a    = meta['patch_to_cell_annotations']

    cluster_counts = defaultdict(Counter)
    for pidx, ann_list in p2a.items():
        cluster_counts[int(labels[pidx])].update(ann_list)

    comp = np.zeros((k, len(ct)))
    for cl in range(k):
        total = sum(cluster_counts[cl].values())
        if total:
            for j, c in enumerate(ct):
                comp[cl, j] = cluster_counts[cl].get(c, 0) / total
    return labels, comp, cluster_counts


def between_cluster_js(comp):
    k, vals = len(comp), []
    for i in range(k):
        for j in range(i+1, k):
            p = comp[i] + 1e-9; p /= p.sum()
            q = comp[j] + 1e-9; q /= q.sum()
            vals.append(float(jensenshannon(p, q)))
    return float(np.mean(vals)), vals


# ── 5. UMAP ───────────────────────────────────────────────────────────────────
def compute_umap(emb_pca, labels):
    n = min(UMAP_MAX_SAMPLES, len(emb_pca))
    idx = np.random.RandomState(42).choice(len(emb_pca), n, replace=False)
    reducer = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                            metric='cosine', random_state=42, n_jobs=N_JOBS, verbose=False)
    xy = reducer.fit_transform(emb_pca[idx].astype(np.float32))
    return xy, idx


# ── 6. Cross-patient reproducibility (MIBI only) ─────────────────────────────
def per_patient_cluster(sid, emb_pca, k, meta):
    idx   = meta['sample_to_meta_indices'][sid]
    X     = emb_pca[idx]
    if len(X) < k:
        return None
    km    = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10, batch_size=min(2048, len(X)))
    local = km.fit_predict(X)
    ct    = meta['all_cell_types']
    p2a   = meta['patch_to_cell_annotations']
    counts = defaultdict(Counter)
    for li, gi in enumerate(idx):
        counts[int(local[li])].update(p2a.get(gi, []))
    comp = np.zeros((k, len(ct)))
    for cl in range(k):
        total = sum(counts[cl].values())
        if total:
            for j, c in enumerate(ct):
                comp[cl, j] = counts[cl].get(c, 0) / total
    return comp


def match_clusters(comp_a, comp_b):
    k    = len(comp_a)
    cost = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            p = comp_a[i] + 1e-9; p /= p.sum()
            q = comp_b[j] + 1e-9; q /= q.sum()
            cost[i, j] = jensenshannon(p, q)
    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind, cost[row_ind, col_ind]


# ── Plotting helpers ──────────────────────────────────────────────────────────
def spatial_canvas(labels, sid, meta, k, cmap):
    ps = REGION_PATCH_SIZE
    H, W = meta['sample_dims'][sid]
    canvas = np.full((H, W, 4), [0.12, 0.12, 0.12, 1.0])
    for i in meta['sample_to_meta_indices'][sid]:
        _, y, x, _, _ = meta['all_meta'][i]
        canvas[y:y+ps, x:x+ps] = np.array(cmap(int(labels[i])))
    return canvas


# ── Main analysis ─────────────────────────────────────────────────────────────
def run(dataset_name, gpu=0):
    cfg    = DATASET_CONFIGS[dataset_name]
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    print(f'\n{"="*70}')
    print(f'  Region Analysis: {dataset_name}')
    print(f'  Device: {device}')
    print(f'{"="*70}')

    # Output
    save_dir = Path(f'z_RUNS/region_analysis_paper/{dataset_name}')
    save_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = save_dir / 'embeddings'
    cache_dir.mkdir(exist_ok=True)

    # Load metadata (once, shared across all models)
    print('\n[1] Loading metadata...')
    meta = load_metadata(cfg)

    models = list(cfg['models'].keys())
    k_list = cfg['k_list']

    # ── Embed all models ──────────────────────────────────────────────────
    print('\n[2] Embedding patches for all models...')
    embeddings_all = {}
    for mname, work_dir in cfg['models'].items():
        embeddings_all[mname] = embed_model(work_dir, meta, cfg, cache_dir, device)

    # ── PCA + k-sweep ─────────────────────────────────────────────────────
    print('\n[3] PCA + k-sweep...')
    pca_all   = {}   # mname -> emb_pca
    sil_all   = {}   # mname -> [sil at each k]
    best_k_all= {}   # mname -> best k
    for mname, emb in embeddings_all.items():
        emb_pca, sil_scores, inertias, best_k = pca_and_sweep(emb, k_list, mname)
        pca_all[mname]    = emb_pca
        sil_all[mname]    = sil_scores
        best_k_all[mname] = best_k

    # Plot: k-sweep comparison (all models, one plot)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for mname in models:
        axes[0].plot(k_list, sil_all[mname], 'o-', color=MODEL_COLORS[mname], label=mname)
        axes[0].axvline(best_k_all[mname], color=MODEL_COLORS[mname], ls='--', alpha=0.4)
    axes[0].set(xlabel='k', ylabel='Silhouette', title='k-sweep Silhouette — all models')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Best silhouette per model bar chart
    best_sils = [max(sil_all[m]) for m in models]
    bars = axes[1].bar(models, best_sils, color=[MODEL_COLORS[m] for m in models], alpha=0.85)
    for bar, v in zip(bars, best_sils):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.002, f'{v:.3f}',
                     ha='center', va='bottom', fontsize=9)
    axes[1].set(ylabel='Best silhouette', title='Best silhouette per model')
    axes[1].grid(axis='y', alpha=0.3)
    plt.suptitle(f'k-sweep — {dataset_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    savefig(save_dir / 'k_sweep_comparison.png')
    print(f'  Saved: k_sweep_comparison')

    # ── Cluster at shared k (consensus: most common best_k) ──────────────
    shared_k = cfg['k_repro']
    print(f'\n[4] Clustering all models at shared k={shared_k}...')
    labels_all = {}
    comp_all   = {}
    js_all     = {}
    for mname in models:
        labels, comp, _ = cluster_and_compose(pca_all[mname], shared_k, meta)
        labels_all[mname] = labels
        comp_all[mname]   = comp
        js_mean, js_vals  = between_cluster_js(comp)
        js_all[mname]     = (js_mean, js_vals)
        print(f'  [{mname}] between-cluster JS = {js_mean:.3f}')

    # Save clustering summary
    summary = {
        'dataset': dataset_name,
        'shared_k': shared_k,
        'k_sweep': {m: {'k_list': k_list, 'sil': sil_all[m], 'best_k': best_k_all[m]} for m in models},
        'compartment_js': {m: {'mean': js_all[m][0]} for m in models},
    }
    with open(save_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot: compartment distinctiveness comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    js_means = [js_all[m][0] for m in models]
    js_vals_list = [js_all[m][1] for m in models]
    bars = axes[0].bar(models, js_means, color=[MODEL_COLORS[m] for m in models], alpha=0.85)
    for bar, v in zip(bars, js_means):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.003, f'{v:.3f}',
                     ha='center', va='bottom', fontsize=9)
    axes[0].set(ylabel='Mean between-cluster JS divergence',
                title=f'Compartment distinctiveness (k={shared_k})')
    axes[0].grid(axis='y', alpha=0.3)

    bp = axes[1].boxplot(js_vals_list, patch_artist=True,
                         labels=models, medianprops=dict(color='black', linewidth=2))
    for patch, mname in zip(bp['boxes'], models):
        patch.set_facecolor(MODEL_COLORS[mname])
        patch.set_alpha(0.7)
    axes[1].set(ylabel='Pairwise cluster JS', title='Distribution of pairwise JS')
    plt.suptitle(f'Compartment distinctiveness — {dataset_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    savefig(save_dir / 'compartment_distinctiveness.png')
    print(f'  Saved: compartment_distinctiveness')

    # ── UMAP per model ────────────────────────────────────────────────────
    print(f'\n[5] Computing UMAPs...')
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4))
    cmap_k = plt.cm.get_cmap('tab10', shared_k)
    for ax, mname in zip(axes, models):
        xy, idx_u = compute_umap(pca_all[mname], labels_all[mname])
        sc = ax.scatter(xy[:, 0], xy[:, 1],
                        c=labels_all[mname][idx_u], cmap=cmap_k,
                        vmin=-0.5, vmax=shared_k-0.5, s=2, alpha=0.5, rasterized=True)
        ax.set_title(f'{mname}\nJS={js_all[mname][0]:.3f}', fontsize=9, fontweight='bold')
        ax.axis('off')
    handles = [mpatches.Patch(color=cmap_k(c), label=f'C{c}') for c in range(shared_k)]
    fig.legend(handles=handles, loc='lower center', ncol=shared_k, fontsize=8, frameon=False)
    plt.suptitle(f'UMAP coloured by region cluster (k={shared_k}) — {dataset_name}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig(save_dir / f'umap_k{shared_k}_all_models.png')
    print(f'  Saved: umap_k{shared_k}_all_models')

    # ── Spatial maps: all models, representative patients ─────────────────
    print(f'\n[6] Spatial maps...')
    sample_patch_counts = Counter(m[0] for m in meta['all_meta'])
    show_patients = [sid for sid, _ in sample_patch_counts.most_common(3)]

    n_show = len(show_patients)
    fig, axes = plt.subplots(n_show, len(models),
                             figsize=(4.5*len(models), 4.5*n_show), squeeze=False)
    for col, mname in enumerate(models):
        for row, sid in enumerate(show_patients):
            canvas = spatial_canvas(labels_all[mname], sid, meta, shared_k, cmap_k)
            axes[row][col].imshow(canvas)
            axes[row][col].axis('off')
            if row == 0:
                axes[row][col].set_title(f'{mname}\nJS={js_all[mname][0]:.3f}',
                                         fontsize=9, fontweight='bold')
            if col == 0:
                axes[row][col].set_ylabel(f'Patient {sid}', fontsize=8)

    handles = [mpatches.Patch(color=cmap_k(c), label=f'C{c}') for c in range(shared_k)]
    fig.legend(handles=handles, loc='lower center', ncol=shared_k, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    plt.suptitle(f'Spatial region maps (k={shared_k}) — {dataset_name}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    savefig(save_dir / f'spatial_maps_k{shared_k}_all_models.png')
    print(f'  Saved: spatial_maps_k{shared_k}_all_models')

    # ── Composition bars: all models ──────────────────────────────────────
    print(f'\n[7] Composition plots...')
    ct = meta['all_cell_types']
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 5), sharey=True)
    for ax, mname in zip(axes, models):
        comp = comp_all[mname]
        df   = pd.DataFrame(comp, columns=ct)
        df.plot(kind='bar', stacked=True, ax=ax, colormap='tab20', legend=False, width=0.8)
        ax.set_title(f'{mname}\nJS={js_all[mname][0]:.3f}', fontsize=9, fontweight='bold')
        ax.set_xlabel('Cluster')
        ax.set_xticklabels([f'C{c}' for c in range(shared_k)], rotation=0, fontsize=8)
        ax.grid(axis='y', alpha=0.2)
    axes[0].set_ylabel('Cell-type fraction')
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=min(len(ct), 6),
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.08))
    plt.suptitle(f'Cell-type composition per cluster (k={shared_k}) — {dataset_name}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    savefig(save_dir / f'composition_k{shared_k}_all_models.png')
    print(f'  Saved: composition_k{shared_k}_all_models')

    # ── Cross-patient reproducibility (MIBI only) ─────────────────────────
    if cfg.get('multi_patient'):
        print(f'\n[8] Cross-patient reproducibility (k={shared_k})...')
        valid_patients = meta['unique_samples']
        repro_results = {}

        for mname in models:
            print(f'  [{mname}] Per-patient clustering...')
            per_patient_comps = {}
            for sid in tqdm(valid_patients, desc=f'  {mname}', leave=False):
                comp = per_patient_cluster(sid, pca_all[mname], shared_k, meta)
                if comp is not None:
                    per_patient_comps[sid] = comp

            valid = list(per_patient_comps.keys())
            ref_sid  = valid[0]
            ref_comp = per_patient_comps[ref_sid]
            all_js   = [[] for _ in range(shared_k)]
            for sid in valid[1:]:
                _, js_vals = match_clusters(ref_comp, per_patient_comps[sid])
                for cl, v in enumerate(js_vals):
                    all_js[cl].append(float(v))
            mean_js = [np.mean(v) if v else float('nan') for v in all_js]
            repro_results[mname] = {'mean_js_per_cluster': mean_js,
                                    'all_js': all_js,
                                    'overall_mean': float(np.nanmean(mean_js))}
            print(f'  [{mname}] Overall mean JS = {repro_results[mname]["overall_mean"]:.3f}')

        # Reproducibility comparison: overall mean JS per model
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Bar: overall mean JS
        r_means = [repro_results[m]['overall_mean'] for m in models]
        bars = axes[0].bar(models, r_means,
                           color=[MODEL_COLORS[m] for m in models], alpha=0.85)
        for bar, v in zip(bars, r_means):
            axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.003, f'{v:.3f}',
                         ha='center', va='bottom', fontsize=9)
        axes[0].set(ylabel='Mean JS divergence (lower = more reproducible)',
                    title='Cross-patient reproducibility\n(lower JS = same compartment across patients)')
        axes[0].grid(axis='y', alpha=0.3)

        # Per-cluster JS boxplots for each model
        x = np.arange(shared_k)
        width = 0.8 / len(models)
        for i, mname in enumerate(models):
            pos = x + (i - len(models)/2 + 0.5) * width
            vals = [repro_results[mname]['all_js'][cl] for cl in range(shared_k)]
            medians = [np.median(v) if v else 0 for v in vals]
            axes[1].bar(pos, medians, width, label=mname,
                        color=MODEL_COLORS[mname], alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'C{c}' for c in range(shared_k)])
        axes[1].set(ylabel='Median JS divergence', title='Per-cluster reproducibility')
        axes[1].legend(fontsize=8)
        axes[1].grid(axis='y', alpha=0.3)

        plt.suptitle(f'Cross-patient reproducibility — {dataset_name} (k={shared_k})',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        savefig(save_dir / 'reproducibility_comparison.png')
        print(f'  Saved: reproducibility_comparison')

        # Update summary
        summary['reproducibility'] = {m: {'overall_mean_js': repro_results[m]['overall_mean'],
                                           'per_cluster_mean_js': repro_results[m]['mean_js_per_cluster']}
                                       for m in models}
        with open(save_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    # ── Final summary plot ─────────────────────────────────────────────────
    print(f'\n[9] Final summary plot...')
    metrics = ['Best sil', 'Compartment JS']
    n_metrics = 2 + (1 if cfg.get('multi_patient') else 0)
    if cfg.get('multi_patient'):
        metrics.append('Repro JS (↓)')

    values = {
        'Best sil':      [max(sil_all[m]) for m in models],
        'Compartment JS':[js_all[m][0]    for m in models],
    }
    if cfg.get('multi_patient'):
        values['Repro JS (↓)'] = [repro_results[m]['overall_mean'] for m in models]

    fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 4))
    for ax, (metric, vals) in zip(axes, values.items()):
        bars = ax.bar(models, vals, color=[MODEL_COLORS[m] for m in models], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + max(vals)*0.01, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=9)
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.set_xticklabels(models, rotation=15, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle(f'Region Analysis Summary — {dataset_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    savefig(save_dir / 'summary_metrics.png')
    print(f'  Saved: summary_metrics')

    print(f'\n{"="*70}')
    print(f'  DONE: {dataset_name}')
    print(f'  Output: {save_dir.resolve()}')
    print(f'{"="*70}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Region analysis — all models')
    parser.add_argument('--dataset', required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    run(args.dataset, gpu=args.gpu)

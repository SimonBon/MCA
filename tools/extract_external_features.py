#!/usr/bin/env python3
"""Extract and evaluate features from external pretrained models.

Loads an MCA dataset, runs a frozen external model (OpenPhenom, DINOv2, UNI)
to extract cell embeddings, saves them in the same format as EvaluateModelRich,
then runs the full evaluation suite (LP, kNN, clustering, silhouette, UMAP,
label efficiency).

The output directory structure matches the paper runs so all existing
post-processing scripts work unchanged:
    z_RUNS/paper/<DATASET>/<MODEL>/
        metrics.json
        label_efficiency.json
        train_results.npz
        val_results.npz
        umap.pdf
        label_efficiency.pdf

Usage examples:
    # DINOv2 ViT-B/14 on CODEX cHL KRONOS18
    python tools/extract_external_features.py \\
        --model dinov2_vitb14 \\
        --dataset CODEX_cHL_KRONOS18 \\
        --out z_RUNS/paper/CODEX_cHL_KRONOS18/DINOv2_vitb14

    # OpenPhenom on MIBI_TNBC
    python tools/extract_external_features.py \\
        --model openphenom \\
        --dataset MIBI_TNBC \\
        --out z_RUNS/paper/MIBI_TNBC/OpenPhenom

    # UNI on CODEX DLBCL (requires --uni_ckpt)
    python tools/extract_external_features.py \\
        --model uni \\
        --uni_ckpt /path/to/pytorch_model.bin \\
        --dataset CODEX_DLBCL \\
        --out z_RUNS/paper/CODEX_DLBCL/UNI
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# Make sure MCA src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    balanced_accuracy_score, average_precision_score,
    normalized_mutual_info_score, adjusted_rand_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Dataset configs (mirrors configs/_datasets_/*.py)
# ─────────────────────────────────────────────────────────────────────────────

BASE = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files'

DATASET_CONFIGS = {
    'CODEX_cHL_KRONOS18': dict(
        h5_filepath      = f'{BASE}/CODEX_cHL/CODEX_cHL.h5',
        used_markers     = f'{BASE}/CODEX_cHL/used_markers_KRONOS18.txt',
        patch_size       = 32,
        train_indicies   = f'{BASE}/CODEX_cHL/train.txt',
        val_indicies     = f'{BASE}/CODEX_cHL/val.txt',
        ignore_annotation= ['Seg Artifact', 'Cytotoxic CD8'],
    ),
    'CODEX_cHL': dict(
        h5_filepath      = f'{BASE}/CODEX_cHL/CODEX_cHL.h5',
        used_markers     = f'{BASE}/CODEX_cHL/used_markers.txt',
        patch_size       = 32,
        train_indicies   = f'{BASE}/CODEX_cHL/train.txt',
        val_indicies     = f'{BASE}/CODEX_cHL/val.txt',
        ignore_annotation= ['Seg Artifact', 'Cytotoxic CD8'],
    ),
    'CODEX_DLBCL': dict(
        h5_filepath      = f'{BASE}/CODEX_DLBCL/CODEX_DLBCL.h5',
        used_markers     = f'{BASE}/CODEX_DLBCL/used_markers.txt',
        patch_size       = 24,
        train_indicies   = f'{BASE}/CODEX_DLBCL/train.txt',
        val_indicies     = f'{BASE}/CODEX_DLBCL/val.txt',
        ignore_annotation= ['Seg Artifact'],
    ),
    'IMC_NB_TumorSub': dict(
        h5_filepath      = f'{BASE}/IMC_NB/IMC_NB.h5',
        used_markers     = f'{BASE}/IMC_NB/used_markers.txt',
        patch_size       = 24,
        train_indicies   = f'{BASE}/IMC_NB/train.txt',
        val_indicies     = f'{BASE}/IMC_NB/val.txt',
        ignore_annotation= ['Seg Artifact'],
    ),
    'MIBI_TNBC': dict(
        h5_filepath      = f'{BASE}/MIBI_TNBC/MIBI_TNBC.h5',
        used_markers     = f'{BASE}/MIBI_TNBC/used_markers.txt',
        patch_size       = 32,
        train_indicies   = f'{BASE}/MIBI_TNBC/train.txt',
        val_indicies     = f'{BASE}/MIBI_TNBC/val.txt',
        ignore_annotation= ['Seg Artifact'],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloader(dataset_cfg: dict, split: str, batch_size: int,
                     num_workers: int):
    """Build an MCIDataset dataloader using the mmengine registry."""
    from mmengine.registry import DATASETS, build_from_cfg
    from mmengine.dataset import DefaultSampler
    from torch.utils.data import DataLoader

    indicies_key = 'train_indicies' if split == 'train' else 'val_indicies'
    ds_kwargs = {
        'type': 'MCIDataset',
        'h5_filepath':       dataset_cfg['h5_filepath'],
        'used_markers':      dataset_cfg['used_markers'],
        'patch_size':        dataset_cfg['patch_size'],
        'used_indicies':     dataset_cfg[indicies_key],
        'ignore_annotation': dataset_cfg.get('ignore_annotation', []),
        'pipeline':          [],   # no augmentation — raw patches
    }
    dataset = DATASETS.build(ds_kwargs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader


@torch.no_grad()
def extract_features(model, dataloader, device, desc='extracting'):
    """Run model on all batches, collect features and labels."""
    features, labels_str, sample_ids = [], [], []
    model.eval()
    for batch in tqdm(dataloader, desc=desc):
        x = batch['inputs'].to(device, dtype=torch.float32)
        feat = model(x)[0]                              # [B, D, 1, 1]
        feat = feat.squeeze(-1).squeeze(-1)             # [B, D]
        features.append(feat.cpu().numpy())
        labels_str.extend(list(batch['data_samples']['annotation'][0]))
        sample_ids.extend(list(batch['data_samples']['sample_id'][0]))
    return (
        np.concatenate(features, axis=0),
        np.array(labels_str),
        np.array(sample_ids),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (mirrors val_hook_rich.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(train_feats, train_labels_str, val_feats, val_labels_str,
                   val_sample_ids, work_dir, n_jobs=4, knn_k=15,
                   silhouette_max=10_000,
                   le_fractions=(0.01, 0.1),
                   le_n_per_class=(10, 50, 100, 200, 1000)):

    le = LabelEncoder().fit(np.concatenate([train_labels_str, val_labels_str]))
    classes    = list(le.classes_)
    n_classes  = len(classes)
    train_y    = le.transform(train_labels_str)
    val_y      = le.transform(val_labels_str)

    # ── Linear probe ─────────────────────────────────────────────────────────
    print('  Linear probe...')
    clf = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=2000,
                             class_weight='balanced', C=1, n_jobs=n_jobs, tol=1e-6)
    clf.fit(train_feats, train_y)
    val_pred  = clf.predict(val_feats)
    val_proba = clf.predict_proba(val_feats)

    lp_bal   = float(balanced_accuracy_score(val_y, val_pred))
    val_bin  = label_binarize(val_y, classes=list(range(n_classes)))
    if val_bin.shape[1] == 1:
        val_bin = np.hstack([1 - val_bin, val_bin])
    per_class_ap = average_precision_score(val_bin, val_proba, average=None)
    mean_ap  = float(np.mean(per_class_ap))

    # ── kNN ──────────────────────────────────────────────────────────────────
    print('  kNN...')
    knn = KNeighborsClassifier(n_neighbors=knn_k, metric='cosine',
                               algorithm='brute', n_jobs=n_jobs)
    knn.fit(train_feats, train_y)
    knn_pred = knn.predict(val_feats)
    knn_bal  = float(balanced_accuracy_score(val_y, knn_pred))

    # ── Clustering (k-means) ─────────────────────────────────────────────────
    print('  Clustering...')
    nmi_scores, ari_scores = [], []
    all_feats = np.concatenate([train_feats, val_feats], axis=0)
    all_y     = np.concatenate([train_y,     val_y],     axis=0)
    for seed in range(10):
        km = MiniBatchKMeans(n_clusters=n_classes, random_state=seed,
                             n_init=3, batch_size=4096)
        km.fit(all_feats)
        nmi_scores.append(normalized_mutual_info_score(all_y, km.labels_))
        ari_scores.append(adjusted_rand_score(all_y, km.labels_))
    nmi = float(np.mean(nmi_scores))
    ari = float(np.mean(ari_scores))

    # ── Silhouette ───────────────────────────────────────────────────────────
    print('  Silhouette...')
    rng   = np.random.default_rng(42)
    n_sil = min(silhouette_max, len(val_feats))
    idx   = rng.choice(len(val_feats), n_sil, replace=False)
    sil   = float(silhouette_score(val_feats[idx], val_y[idx], metric='cosine'))

    # ── Neighbourhood purity ─────────────────────────────────────────────────
    print('  Neighbourhood purity...')
    nbrs = knn.kneighbors(val_feats, n_neighbors=knn_k, return_distance=False)
    purity = float(np.mean(
        [np.mean(val_y[nbrs[i]] == val_y[i]) for i in range(len(val_y))]
    ))

    # ── UMAP ─────────────────────────────────────────────────────────────────
    try:
        import umap as umap_lib
        print('  UMAP...')
        n_umap = min(20_000, len(val_feats))
        idx_u  = rng.choice(len(val_feats), n_umap, replace=False)
        reducer = umap_lib.UMAP(n_components=2, metric='cosine',
                                random_state=42, n_jobs=n_jobs)
        emb = reducer.fit_transform(val_feats[idx_u])
        np.savez(os.path.join(work_dir, 'umap_embeddings.npz'),
                 embedding=emb, labels=val_labels_str[idx_u],
                 sample_ids=val_sample_ids[idx_u])

        fig, ax = plt.subplots(figsize=(10, 8))
        for i, cls in enumerate(classes):
            mask = val_labels_str[idx_u] == cls
            ax.scatter(emb[mask, 0], emb[mask, 1], s=1, alpha=0.5, label=cls)
        ax.legend(markerscale=5, fontsize=7, loc='best')
        ax.set_title(os.path.basename(work_dir))
        plt.savefig(os.path.join(work_dir, 'umap.pdf'), bbox_inches='tight')
        plt.close()
    except ImportError:
        print('  umap-learn not installed — skipping UMAP')

    # ── Metrics JSON ─────────────────────────────────────────────────────────
    per_class_ap_dict = {cls: float(ap)
                         for cls, ap in zip(classes, per_class_ap)}
    metrics = {
        'val': {
            'lp_balanced_accuracy': lp_bal,
            'mean_ap': mean_ap,
            'per_class_ap': per_class_ap_dict,
            'knn_balanced_accuracy': knn_bal,
            'nmi': nmi,
            'ari': ari,
            'silhouette': sil,
            'neighbourhood_purity': purity,
        },
        'n_classes':  n_classes,
        'classes':    classes,
    }
    with open(os.path.join(work_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\n  LP={lp_bal:.3f}  MAP={mean_ap:.3f}  kNN={knn_bal:.3f}'
          f'  NMI={nmi:.3f}  ARI={ari:.3f}  Sil={sil:+.3f}')

    # ── Label efficiency ─────────────────────────────────────────────────────
    print('  Label efficiency...')
    le_points = []
    rng2 = np.random.default_rng(0)

    def _lp(tr_f, tr_y):
        c = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=2000,
                               class_weight='balanced', C=1, n_jobs=n_jobs, tol=1e-6)
        c.fit(tr_f, tr_y)
        p  = c.predict(val_feats)
        pr = c.predict_proba(val_feats)
        ba = float(balanced_accuracy_score(val_y, p))
        ap = float(np.mean(average_precision_score(val_bin, pr, average=None)))
        return ba, ap

    for n in sorted(le_n_per_class):
        bas, aps, ns = [], [], []
        for rep in range(3):
            idx_le = []
            for c in range(n_classes):
                ci = np.where(train_y == c)[0]
                if len(ci) == 0: continue
                idx_le.append(rng2.choice(ci, min(n, len(ci)), replace=False))
            idx_le = np.concatenate(idx_le)
            ba, ap = _lp(train_feats[idx_le], train_y[idx_le])
            bas.append(ba); aps.append(ap); ns.append(len(idx_le))
        le_points.append(dict(label=f'{n}/cls', n_labeled=int(np.mean(ns)),
                              bal_acc_mean=float(np.mean(bas)),
                              bal_acc_std=float(np.std(bas)),
                              mean_ap_mean=float(np.mean(aps)),
                              mean_ap_std=float(np.std(aps))))

    for frac in sorted(le_fractions):
        bas, aps, ns = [], [], []
        reps = 1 if frac >= 1.0 else 3
        for rep in range(reps):
            idx_le = []
            for c in range(n_classes):
                ci = np.where(train_y == c)[0]
                if len(ci) == 0: continue
                n_take = max(1, int(round(len(ci) * frac)))
                idx_le.append(rng2.choice(ci, min(n_take, len(ci)), replace=False))
            idx_le = np.concatenate(idx_le)
            ba, ap = _lp(train_feats[idx_le], train_y[idx_le])
            bas.append(ba); aps.append(ap); ns.append(len(idx_le))
        le_points.append(dict(label=f'{int(frac*100)}%', n_labeled=int(np.mean(ns)),
                              bal_acc_mean=float(np.mean(bas)),
                              bal_acc_std=float(np.std(bas)),
                              mean_ap_mean=float(np.mean(aps)),
                              mean_ap_std=float(np.std(aps))))

    le_points.sort(key=lambda p: p['n_labeled'])
    with open(os.path.join(work_dir, 'label_efficiency.json'), 'w') as f:
        json.dump({'classes': classes, 'n_classes': n_classes,
                   'points': le_points}, f, indent=2)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(args):
    from models_external import OpenPhenomBackbone, DINOv2Backbone, UNIBackbone

    name = args.model.lower()
    if name == 'openphenom':
        print('Loading OpenPhenom...')
        model = OpenPhenomBackbone(
            hf_model_path=args.openphenom_dir or 'recursionpharma/OpenPhenom',
            img_size=256,
        )
    elif name.startswith('dinov2'):
        variant = args.model  # e.g. 'dinov2_vitb14'
        print(f'Loading DINOv2 ({variant})...')
        model = DINOv2Backbone(variant=variant, img_size=224)
    elif name == 'uni':
        assert args.uni_ckpt, '--uni_ckpt required for UNI model'
        print('Loading UNI...')
        model = UNIBackbone(ckpt_path=args.uni_ckpt, img_size=224)
    else:
        raise ValueError(f'Unknown model: {args.model}')

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract features from external pretrained models and evaluate.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--model', required=True,
                        choices=['openphenom', 'dinov2_vits14', 'dinov2_vitb14',
                                 'dinov2_vitl14', 'dinov2_vitg14', 'uni'],
                        help='Which external model to use')
    parser.add_argument('--dataset', required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset to evaluate on')
    parser.add_argument('--out', required=True,
                        help='Output directory (e.g. z_RUNS/paper/CODEX_cHL_KRONOS18/DINOv2_vitb14)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='CUDA device index (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--n_jobs', type=int, default=8,
                        help='Parallel jobs for sklearn (default: 8)')
    parser.add_argument('--skip_extract', action='store_true',
                        help='Skip extraction if train/val_results.npz already exist')
    # Model-specific
    parser.add_argument('--uni_ckpt', default=None,
                        help='Path to UNI pytorch_model.bin (required for --model uni)')
    parser.add_argument('--openphenom_dir', default=None,
                        help='Local path to downloaded OpenPhenom weights '
                             '(default: download from HuggingFace)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ds_cfg = DATASET_CONFIGS[args.dataset]
    train_npz = os.path.join(args.out, 'train_results.npz')
    val_npz   = os.path.join(args.out, 'val_results.npz')

    # ── Feature extraction (skip if cached) ──────────────────────────────────
    if args.skip_extract and os.path.exists(train_npz) and os.path.exists(val_npz):
        print('Loading cached features...')
        tr = np.load(train_npz, allow_pickle=True)
        vl = np.load(val_npz,   allow_pickle=True)
        train_feats, train_labels_str, train_sids = (
            tr['features'], tr['labels_str'], tr['sample_ids'])
        val_feats, val_labels_str, val_sids = (
            vl['features'], vl['labels_str'], vl['sample_ids'])
    else:
        model = build_model(args).to(device)
        print(f'Model output dim: {model.out_channels}')

        print('\nBuilding dataloaders...')
        train_loader = build_dataloader(ds_cfg, 'train', args.batch_size, args.num_workers)
        val_loader   = build_dataloader(ds_cfg, 'val',   args.batch_size, args.num_workers)

        print('Extracting train features...')
        train_feats, train_labels_str, train_sids = extract_features(
            model, train_loader, device, desc='train')

        print('Extracting val features...')
        val_feats, val_labels_str, val_sids = extract_features(
            model, val_loader, device, desc='val')

        # Save — same format as EvaluateModelRich
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(np.concatenate([train_labels_str, val_labels_str]))
        np.savez(train_npz, features=train_feats, labels_str=train_labels_str,
                 labels_num=le.transform(train_labels_str),
                 classes=le.classes_, sample_ids=train_sids)
        np.savez(val_npz,   features=val_feats,   labels_str=val_labels_str,
                 labels_num=le.transform(val_labels_str),
                 classes=le.classes_, sample_ids=val_sids)
        print(f'Saved features → {args.out}')

    # ── Evaluation ────────────────────────────────────────────────────────────
    print('\nRunning evaluation...')
    run_evaluation(
        train_feats, train_labels_str,
        val_feats,   val_labels_str, val_sids,
        work_dir=args.out,
        n_jobs=args.n_jobs,
    )
    print(f'\nDone. Results in {args.out}')


if __name__ == '__main__':
    main()

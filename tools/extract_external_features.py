#!/usr/bin/env python3
"""Extract and evaluate features from external pretrained models.

Reads MCA HDF5 files directly — no mmcv / mmengine dependency, so the
script runs both locally and on the server with the same environment.

Required packages (pip install):
    h5py torch torchvision timm transformers huggingface_hub
    scikit-learn umap-learn tqdm matplotlib numpy

Usage examples:
    # DINOv2 ViT-B/14 on CODEX cHL KRONOS18
    python tools/extract_external_features.py \\
        --model dinov2_vitb14 \\
        --h5  /path/to/CODEX_cHL.h5 \\
        --markers /path/to/used_markers_KRONOS18.txt \\
        --train_idx /path/to/train.txt \\
        --val_idx   /path/to/val.txt \\
        --ignore "Seg Artifact,Cytotoxic CD8" \\
        --patch_size 32 \\
        --out z_RUNS/paper/CODEX_cHL_KRONOS18/DINOv2_vitb14

    # OpenPhenom (no --ignore needed for clean datasets)
    python tools/extract_external_features.py \\
        --model openphenom \\
        --h5  /path/to/MIBI_TNBC.h5 \\
        --markers /path/to/used_markers.txt \\
        --train_idx /path/to/train.txt \\
        --val_idx   /path/to/val.txt \\
        --patch_size 32 \\
        --out z_RUNS/paper/MIBI_TNBC/OpenPhenom

    # UNI (requires --uni_ckpt path to downloaded pytorch_model.bin)
    python tools/extract_external_features.py \\
        --model uni \\
        --uni_ckpt /path/to/UNI/pytorch_model.bin \\
        --h5  /path/to/CODEX_DLBCL.h5 \\
        --markers /path/to/used_markers.txt \\
        --train_idx /path/to/train.txt \\
        --val_idx   /path/to/val.txt \\
        --patch_size 24 \\
        --out z_RUNS/paper/CODEX_DLBCL/UNI

    # Re-run evaluation only (skip slow GPU extraction)
    python tools/extract_external_features.py \\
        --model dinov2_vitb14 --h5 ... --out ... --skip_extract
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
# Standalone HDF5 dataset (no mmcv / mmengine)
# ─────────────────────────────────────────────────────────────────────────────

class MCIDatasetH5(Dataset):
    """Read cell patches directly from an MCA HDF5 file.

    HDF5 layout (same as produced by MCA preprocessing):
        h5f['coords']['DIM1']      — cell row coordinates   [N]
        h5f['coords']['DIM2']      — cell column coordinates [N]
        h5f['coords']['sample_id'] — patient/slide ID        [N]
        h5f['annotation']          — cell-type label string  [N]
        h5f['marker_names']        — all marker names        [M]
        h5f[sample_id]['image']    — whole-slide image       [H, W, M]
        h5f[sample_id]['masks']    — cell segmentation mask  [H, W]

    Returns float32 patches [C, patch_size, patch_size] in [0, 1].
    """

    def __init__(self, h5_filepath: str, patch_size: int,
                 used_markers_file: str = None,
                 used_indicies_file: str = None,
                 ignore_annotations: list = None):
        self.h5_filepath  = h5_filepath
        self.patch_size   = patch_size
        self.half         = patch_size // 2
        self._h5f         = None   # opened per-worker in __getitem__
        self._img_cache   = {}     # sample_id → (img_np, mask_np) in-memory cache

        with h5py.File(h5_filepath, 'r') as f:
            coords            = f['coords']
            dim1              = coords['DIM1'][:]
            dim2              = coords['DIM2'][:]
            sample_id         = coords['sample_id'][:].astype(str)
            annotation        = f['annotation'][()].astype(str)
            all_markers       = f['marker_names'][:].astype(str)

        # Marker selection
        if used_markers_file is not None:
            selected = np.loadtxt(used_markers_file, dtype=str, delimiter=',')
            marker2idx = {n: i for i, n in enumerate(all_markers)}
            self.marker_idx = np.array(sorted([marker2idx[m] for m in selected]))
        else:
            self.marker_idx = np.arange(len(all_markers))
        self.marker_names = all_markers[self.marker_idx]

        # Index filter
        if used_indicies_file is not None:
            mask = np.loadtxt(used_indicies_file, dtype=int)
            dim1       = dim1[mask]
            dim2       = dim2[mask]
            sample_id  = sample_id[mask]
            annotation = annotation[mask]

        # Annotation filter
        if ignore_annotations:
            keep = np.ones(len(annotation), dtype=bool)
            for label in ignore_annotations:
                keep &= (annotation != label.strip())
            dim1       = dim1[keep]
            dim2       = dim2[keep]
            sample_id  = sample_id[keep]
            annotation = annotation[keep]

        self.dim1       = dim1
        self.dim2       = dim2
        self.sample_id  = sample_id
        self.annotation = annotation

    def __len__(self):
        return len(self.dim1)

    def _open_h5(self):
        """Open HDF5 lazily (safe for DataLoader workers)."""
        if self._h5f is None:
            self._h5f = h5py.File(self.h5_filepath, 'r')

    def _get_patch(self, dim1, dim2, sample_id):
        self._open_h5()
        if sample_id not in self._img_cache:
            grp = self._h5f['data'][sample_id] if 'data' in self._h5f else self._h5f[sample_id]
            self._img_cache[sample_id] = (grp['image'][:], grp['masks'][:])
        img, mask = self._img_cache[sample_id]
        H, W = img.shape[0], img.shape[1]
        half = self.half

        # Clamp bounds and compute padding
        r0, r1 = dim1 - half, dim1 + half
        c0, c1 = dim2 - half, dim2 + half
        pr0 = max(0, -r0);  pr1 = max(0, r1 - H)
        pc0 = max(0, -c0);  pc1 = max(0, c1 - W)
        r0, r1 = max(0, r0), min(H, r1)
        c0, c1 = max(0, c0), min(W, c1)

        patch = img[r0:r1, c0:c1][:, :, self.marker_idx]   # [h, w, C]
        msk   = mask[r0:r1, c0:c1]

        if any([pr0, pr1, pc0, pc1]):
            patch = np.pad(patch, ((pr0, pr1), (pc0, pc1), (0, 0)))
            msk   = np.pad(msk,   ((pr0, pr1), (pc0, pc1)))

        # Zero out pixels that don't belong to the centre cell
        centre_id = msk[self.half, self.half]
        msk_bin   = (msk == centre_id).astype(np.float32)
        patch     = patch * msk_bin[:, :, None]

        return patch.astype(np.float32)  # [H, W, C]

    def __getitem__(self, idx):
        patch = self._get_patch(self.dim1[idx], self.dim2[idx], self.sample_id[idx])
        # [H, W, C] → [C, H, W], normalise to [0, 1]
        patch = np.transpose(patch, (2, 0, 1))
        patch = patch / (patch.max() + 1e-8)
        return {
            'image':      torch.from_numpy(patch),
            'annotation': self.annotation[idx],
            'sample_id':  self.sample_id[idx],
        }


def build_dataloader(h5_path, patch_size, markers_file, indicies_file,
                     ignore, batch_size, num_workers):
    ds = MCIDatasetH5(
        h5_filepath=h5_path,
        patch_size=patch_size,
        used_markers_file=markers_file,
        used_indicies_file=indicies_file,
        ignore_annotations=ignore,
    )
    print(f'  {len(ds)} cells | {len(ds.marker_names)} markers: {list(ds.marker_names)}')
    # Sort by sample_id so each worker keeps its sample image cached across consecutive cells
    order = np.argsort(ds.sample_id, kind='stable')
    loader = DataLoader(
        torch.utils.data.Subset(ds, order),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return loader


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, dataloader, device, desc='extracting'):
    features, labels_str, sample_ids = [], [], []
    model.eval()
    for batch in tqdm(dataloader, desc=desc):
        x    = batch['image'].to(device, dtype=torch.float32)
        feat = model(x)[0].squeeze(-1).squeeze(-1)   # [B, D]
        features.append(feat.cpu().numpy())
        labels_str.extend(batch['annotation'])
        sample_ids.extend(batch['sample_id'])
    return (
        np.concatenate(features, axis=0),
        np.array(labels_str),
        np.array(sample_ids),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (LP / kNN / clustering / silhouette / UMAP / label efficiency)
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(train_feats, train_labels_str, val_feats, val_labels_str,
                   val_sample_ids, work_dir, n_jobs=4, knn_k=15,
                   silhouette_max=10_000,
                   le_fractions=(0.01, 0.1),
                   le_n_per_class=(10, 50, 100, 200, 1000)):

    le = LabelEncoder().fit(np.concatenate([train_labels_str, val_labels_str]))
    classes   = list(le.classes_)
    n_classes = len(classes)
    train_y   = le.transform(train_labels_str)
    val_y     = le.transform(val_labels_str)

    val_bin = label_binarize(val_y, classes=list(range(n_classes)))
    if val_bin.shape[1] == 1:
        val_bin = np.hstack([1 - val_bin, val_bin])

    # ── Linear probe ────────────────────────────────────────────────────────
    print('  Linear probe...')
    clf = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=2000,
                             class_weight='balanced', C=1, n_jobs=n_jobs, tol=1e-6)
    clf.fit(train_feats, train_y)
    val_pred  = clf.predict(val_feats)
    val_proba = clf.predict_proba(val_feats)
    lp_bal  = float(balanced_accuracy_score(val_y, val_pred))
    per_cls = average_precision_score(val_bin, val_proba, average=None)
    mean_ap = float(np.mean(per_cls))

    # ── kNN ─────────────────────────────────────────────────────────────────
    print('  kNN...')
    knn = KNeighborsClassifier(n_neighbors=knn_k, metric='cosine',
                               algorithm='brute', n_jobs=n_jobs)
    knn.fit(train_feats, train_y)
    knn_bal = float(balanced_accuracy_score(val_y, knn.predict(val_feats)))

    # ── Clustering ──────────────────────────────────────────────────────────
    print('  Clustering...')
    all_feats = np.concatenate([train_feats, val_feats])
    all_y     = np.concatenate([train_y,     val_y])
    nmis, aris = [], []
    for seed in range(10):
        km = MiniBatchKMeans(n_clusters=n_classes, random_state=seed,
                             n_init=3, batch_size=4096).fit(all_feats)
        nmis.append(normalized_mutual_info_score(all_y, km.labels_))
        aris.append(adjusted_rand_score(all_y, km.labels_))
    nmi = float(np.mean(nmis))
    ari = float(np.mean(aris))

    # ── Silhouette ──────────────────────────────────────────────────────────
    print('  Silhouette...')
    rng   = np.random.default_rng(42)
    n_sil = min(silhouette_max, len(val_feats))
    idx   = rng.choice(len(val_feats), n_sil, replace=False)
    sil   = float(silhouette_score(val_feats[idx], val_y[idx], metric='cosine'))

    # ── Neighbourhood purity ────────────────────────────────────────────────
    nbrs   = knn.kneighbors(val_feats, n_neighbors=knn_k, return_distance=False)
    purity = float(np.mean([np.mean(val_y[nbrs[i]] == val_y[i])
                             for i in range(len(val_y))]))

    # ── UMAP ────────────────────────────────────────────────────────────────
    try:
        import umap as umap_lib
        print('  UMAP...')
        n_u = min(20_000, len(val_feats))
        idx_u = rng.choice(len(val_feats), n_u, replace=False)
        emb = umap_lib.UMAP(n_components=2, metric='cosine',
                             random_state=42, n_jobs=n_jobs).fit_transform(val_feats[idx_u])
        np.savez(os.path.join(work_dir, 'umap_embeddings.npz'),
                 embedding=emb, labels=val_labels_str[idx_u],
                 sample_ids=val_sample_ids[idx_u])
        fig, ax = plt.subplots(figsize=(10, 8))
        for cls in classes:
            m = val_labels_str[idx_u] == cls
            ax.scatter(emb[m, 0], emb[m, 1], s=1, alpha=0.5, label=cls)
        ax.legend(markerscale=5, fontsize=7)
        ax.set_title(os.path.basename(work_dir))
        plt.savefig(os.path.join(work_dir, 'umap.pdf'), bbox_inches='tight')
        plt.close()
    except ImportError:
        print('  umap-learn not installed — skipping UMAP')

    # ── Metrics JSON ────────────────────────────────────────────────────────
    metrics = {
        'val': {
            'lp_balanced_accuracy':   lp_bal,
            'mean_ap':                mean_ap,
            'per_class_ap':           {c: float(a) for c, a in zip(classes, per_cls)},
            'knn_balanced_accuracy':  knn_bal,
            'nmi':                    nmi,
            'ari':                    ari,
            'silhouette':             sil,
            'neighbourhood_purity':   purity,
        },
        'n_classes': n_classes,
        'classes':   classes,
    }
    with open(os.path.join(work_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\n  LP={lp_bal:.3f}  MAP={mean_ap:.3f}  kNN={knn_bal:.3f}'
          f'  NMI={nmi:.3f}  ARI={ari:.3f}  Sil={sil:+.3f}')

    # ── Label efficiency ────────────────────────────────────────────────────
    print('  Label efficiency...')
    rng2 = np.random.default_rng(0)

    def _lp(tr_f, tr_y):
        c = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=2000,
                               class_weight='balanced', C=1, n_jobs=n_jobs, tol=1e-6)
        c.fit(tr_f, tr_y)
        p  = c.predict(val_feats)
        pr = c.predict_proba(val_feats)
        return (float(balanced_accuracy_score(val_y, p)),
                float(np.mean(average_precision_score(val_bin, pr, average=None))))

    le_points = []
    for n in sorted(le_n_per_class):
        bas, aps, ns = [], [], []
        for _ in range(3):
            idx_le = np.concatenate([
                rng2.choice(ci := np.where(train_y == c)[0],
                            min(n, len(ci)), replace=False)
                for c in range(n_classes) if len(np.where(train_y == c)[0]) > 0
            ])
            ba, ap = _lp(train_feats[idx_le], train_y[idx_le])
            bas.append(ba); aps.append(ap); ns.append(len(idx_le))
        le_points.append(dict(label=f'{n}/cls', n_labeled=int(np.mean(ns)),
                              bal_acc_mean=float(np.mean(bas)), bal_acc_std=float(np.std(bas)),
                              mean_ap_mean=float(np.mean(aps)), mean_ap_std=float(np.std(aps))))

    for frac in sorted(le_fractions):
        bas, aps, ns = [], [], []
        for _ in range(1 if frac >= 1.0 else 3):
            idx_le = np.concatenate([
                rng2.choice(ci := np.where(train_y == c)[0],
                            max(1, min(int(round(len(ci) * frac)), len(ci))), replace=False)
                for c in range(n_classes) if len(np.where(train_y == c)[0]) > 0
            ])
            ba, ap = _lp(train_feats[idx_le], train_y[idx_le])
            bas.append(ba); aps.append(ap); ns.append(len(idx_le))
        le_points.append(dict(label=f'{int(frac*100)}%', n_labeled=int(np.mean(ns)),
                              bal_acc_mean=float(np.mean(bas)), bal_acc_std=float(np.std(bas)),
                              mean_ap_mean=float(np.mean(aps)), mean_ap_std=float(np.std(aps))))

    le_points.sort(key=lambda p: p['n_labeled'])
    with open(os.path.join(work_dir, 'label_efficiency.json'), 'w') as f:
        json.dump({'classes': classes, 'n_classes': n_classes, 'points': le_points}, f, indent=2)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(args):
    from models_external import OpenPhenomBackbone, DINOv2Backbone, UNIBackbone

    name = args.model.lower()
    if name == 'openphenom':
        print('Loading OpenPhenom...')
        return OpenPhenomBackbone(
            hf_model_path=args.openphenom_dir or 'recursionpharma/OpenPhenom',
            img_size=256,
        )
    elif name.startswith('dinov2'):
        print(f'Loading DINOv2 ({args.model})...')
        return DINOv2Backbone(variant=args.model, img_size=224)
    elif name == 'uni':
        assert args.uni_ckpt, '--uni_ckpt is required for UNI'
        print('Loading UNI...')
        return UNIBackbone(ckpt_path=args.uni_ckpt, img_size=224)
    else:
        raise ValueError(f'Unknown model: {args.model}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Extract features from a frozen external model and evaluate.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Model
    p.add_argument('--model', required=True,
                   choices=['openphenom', 'dinov2_vits14', 'dinov2_vitb14',
                            'dinov2_vitl14', 'dinov2_vitg14', 'uni'])
    p.add_argument('--uni_ckpt',       default=None, help='Path to UNI pytorch_model.bin')
    p.add_argument('--openphenom_dir', default=None, help='Local path to OpenPhenom weights')

    # Data
    p.add_argument('--h5',         required=True, help='Path to .h5 file')
    p.add_argument('--markers',    default=None,  help='Path to used_markers.txt')
    p.add_argument('--train_idx',  required=True, help='Path to train.txt indices')
    p.add_argument('--val_idx',    required=True, help='Path to val.txt indices')
    p.add_argument('--patch_size', type=int, default=32)
    p.add_argument('--ignore',     default='',
                   help='Comma-separated annotation labels to ignore, e.g. "Seg Artifact,Cytotoxic CD8"')

    # Output / hardware
    p.add_argument('--out',          required=True)
    p.add_argument('--gpu',          type=int, default=0)
    p.add_argument('--batch_size',   type=int, default=128)
    p.add_argument('--num_workers',  type=int, default=4)
    p.add_argument('--n_jobs',       type=int, default=4)
    p.add_argument('--skip_extract', action='store_true',
                   help='Skip GPU extraction if train/val_results.npz already exist')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ignore = [s.strip() for s in args.ignore.split(',') if s.strip()]
    train_npz = os.path.join(args.out, 'train_results.npz')
    val_npz   = os.path.join(args.out, 'val_results.npz')

    # ── Feature extraction (skip if cached) ───────────────────────────────
    if args.skip_extract and os.path.exists(train_npz) and os.path.exists(val_npz):
        print('Loading cached features...')
        tr = np.load(train_npz, allow_pickle=True)
        vl = np.load(val_npz,   allow_pickle=True)
        train_feats, train_labels_str, train_sids = tr['features'], tr['labels_str'], tr['sample_ids']
        val_feats,   val_labels_str,   val_sids   = vl['features'], vl['labels_str'], vl['sample_ids']
    else:
        model = build_model(args).to(device)
        print(f'Model output dim: {model.out_channels}')

        print('\nBuilding dataloaders...')
        print('Train:')
        train_loader = build_dataloader(args.h5, args.patch_size, args.markers,
                                        args.train_idx, ignore,
                                        args.batch_size, args.num_workers)
        print('Val:')
        val_loader   = build_dataloader(args.h5, args.patch_size, args.markers,
                                        args.val_idx,   ignore,
                                        args.batch_size, args.num_workers)

        print('\nExtracting train features...')
        train_feats, train_labels_str, train_sids = extract_features(
            model, train_loader, device, 'train')

        print('Extracting val features...')
        val_feats, val_labels_str, val_sids = extract_features(
            model, val_loader, device, 'val')

        le_enc = LabelEncoder().fit(np.concatenate([train_labels_str, val_labels_str]))
        np.savez(train_npz, features=train_feats, labels_str=train_labels_str,
                 labels_num=le_enc.transform(train_labels_str),
                 classes=le_enc.classes_, sample_ids=train_sids)
        np.savez(val_npz,   features=val_feats,   labels_str=val_labels_str,
                 labels_num=le_enc.transform(val_labels_str),
                 classes=le_enc.classes_, sample_ids=val_sids)
        print(f'\nFeatures saved → {args.out}')

    # ── Evaluation ────────────────────────────────────────────────────────
    print('\nRunning evaluation...')
    run_evaluation(train_feats, train_labels_str,
                   val_feats,   val_labels_str, val_sids,
                   work_dir=args.out, n_jobs=args.n_jobs)
    print(f'\nDone. Results in {args.out}')


if __name__ == '__main__':
    main()

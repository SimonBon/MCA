"""
Handcrafted expression baseline for MIBI_TNBC.

For each cell, extracts the mean (and std) intensity of each marker within
the cell mask. This gives a simple 37-dim (or 74-dim with std) feature vector
per cell — the natural representation biologists use.

Runs the same LP / kNN / clustering / sample-integration evaluation as the
deep learning models and saves metrics.json to the output directory.

Usage (on cemm):
    python tools/baseline_expression.py \
        --h5     /nobackup/.../MIBI_TNBC.h5 \
        --markers /nobackup/.../used_markers.txt \
        --train   /nobackup/.../train.txt \
        --val     /nobackup/.../val.txt \
        --out     /nobackup/.../z_RUNS/MIBI_TNBC_ExprBaseline \
        --feat    mean          # 'mean' | 'mean+std'
        --patch_size 32
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, accuracy_score,
    normalized_mutual_info_score, adjusted_rand_score,
    silhouette_score, average_precision_score, confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("umap-learn not installed — skipping UMAP")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--h5',         required=True)
    p.add_argument('--markers',    required=True)
    p.add_argument('--train',      required=True)
    p.add_argument('--val',        required=True)
    p.add_argument('--out',        required=True)
    p.add_argument('--feat',       default='mean', choices=['mean', 'mean+std'])
    p.add_argument('--patch_size', type=int, default=32)
    p.add_argument('--ignore',          nargs='*', default=['Unidentified'])
    p.add_argument('--annotation_map',  default=None,
                   help='Comma-separated old:new pairs, e.g. "Cytotoxic CD8:CD8,TReg:Treg"')
    p.add_argument('--n_jobs',          type=int, default=8)
    p.add_argument('--lp_max_iter',type=int, default=5000)
    p.add_argument('--knn_k',      type=int, default=15)
    return p.parse_args()


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(h5_path, marker_indices, dim1, dim2, sample_ids,
                     patch_size, feat_type):
    """
    For each cell: read the patch, apply cell mask, compute mean (+ std).
    Returns features [N, D] and the sample_id array.
    """
    half = patch_size // 2
    features = []

    with h5py.File(h5_path, 'r') as f:
        # cache sample groups
        groups = {sid: f['data'][sid] for sid in np.unique(sample_ids)}

        for i, (d1, d2, sid) in enumerate(zip(dim1, dim2, sample_ids)):
            if i % 5000 == 0:
                print(f'  {i}/{len(dim1)}', flush=True)

            grp = groups[sid]
            H_img, W_img = grp['image'].shape[:2]
            r0, r1 = d1 - half, d1 + half
            c0, c1 = d2 - half, d2 + half
            # clamp to image bounds and zero-pad like dataset._get_patch
            r0c, r1c = max(0, r0), min(H_img, r1)
            c0c, c1c = max(0, c0), min(W_img, c1)
            chunk = grp['image'][r0c:r1c, c0c:c1c, :][:, :, marker_indices].astype(np.float32)
            patch = np.zeros((patch_size, patch_size, len(marker_indices)), dtype=np.float32)
            patch[r0c - r0:r1c - r0, c0c - c0:c1c - c0, :] = chunk
            chunk_m = grp['masks'][r0c:r1c, c0c:c1c]
            mask = np.zeros((patch_size, patch_size), dtype=chunk_m.dtype)
            mask[r0c - r0:r1c - r0, c0c - c0:c1c - c0] = chunk_m

            # keep only pixels belonging to the centre cell
            centre_label = mask[half, half]
            cell_mask = (mask == centre_label)   # [H, W]

            # mean per marker over masked pixels
            n_px = cell_mask.sum()
            if n_px == 0:
                cell_mask = np.ones_like(cell_mask)   # fallback: full patch
                n_px = cell_mask.sum()

            pixels = patch[cell_mask]   # [n_px, C]
            mean   = pixels.mean(axis=0)

            if feat_type == 'mean+std':
                std  = pixels.std(axis=0)
                feat = np.concatenate([mean, std])
            else:
                feat = mean

            features.append(feat)

    return np.array(features, dtype=np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load metadata ─────────────────────────────────────────────────────────
    print('Loading metadata ...')
    with h5py.File(args.h5, 'r') as f:
        all_markers  = f['marker_names'][:].astype(str)
        dim1         = f['coords']['DIM1'][:]
        dim2         = f['coords']['DIM2'][:]
        sample_ids   = f['coords']['sample_id'][:].astype(str)
        annotations  = f['annotation'][()].astype(str)

    used = np.loadtxt(args.markers, dtype=str, delimiter=',')
    marker2idx     = {m: i for i, m in enumerate(all_markers)}
    marker_indices = np.array([marker2idx[m] for m in used])

    # ── Filter ignored classes ────────────────────────────────────────────────
    keep = np.ones(len(annotations), dtype=bool)
    for cls in (args.ignore or []):
        keep &= (annotations != cls)
        print(f'Removing {(~keep).sum()} for {cls}')
    dim1, dim2, sample_ids, annotations = (
        dim1[keep], dim2[keep], sample_ids[keep], annotations[keep])

    # ── Annotation map ────────────────────────────────────────────────────────
    if args.annotation_map:
        amap = dict(pair.split(':') for pair in args.annotation_map.split(','))
        annotations = np.array([amap.get(a, a) for a in annotations])
        print(f'annotation_map applied: {amap}')

    # ── Train / val split ─────────────────────────────────────────────────────
    train_idx = np.loadtxt(args.train, dtype=int)
    val_idx   = np.loadtxt(args.val,   dtype=int)

    # remap to filtered indices
    orig_indices = np.where(keep)[0]
    orig2new     = {o: n for n, o in enumerate(orig_indices)}

    def remap(idx_arr):
        return np.array([orig2new[i] for i in idx_arr if i in orig2new])

    train_sel = remap(train_idx)
    val_sel   = remap(val_idx)

    # ── Extract features ──────────────────────────────────────────────────────
    print(f'\nExtracting {args.feat} features for {len(train_sel)+len(val_sel)} cells ...')
    all_sel = np.concatenate([train_sel, val_sel])
    all_feats = extract_features(
        args.h5, marker_indices,
        dim1[all_sel], dim2[all_sel], sample_ids[all_sel],
        args.patch_size, args.feat,
    )
    n_train = len(train_sel)
    train_feats  = all_feats[:n_train]
    val_feats    = all_feats[n_train:]
    train_labels = annotations[train_sel]
    val_labels   = annotations[val_sel]
    val_ids      = sample_ids[val_sel]

    print(f'Feature dim: {all_feats.shape[1]}')

    le = LabelEncoder().fit(np.concatenate([train_labels, val_labels]))
    train_y = le.transform(train_labels)
    val_y   = le.transform(val_labels)
    classes = list(le.classes_)
    n_classes = len(classes)

    metrics = {
        'classes': classes,
        'n_classes': n_classes,
        'feature_dim': int(all_feats.shape[1]),
        'feat_type': args.feat,
    }

    def save():
        with open(out_dir / 'metrics.json', 'w') as fp:
            json.dump(metrics, fp, indent=2)

    # ── 1. Linear Probe ───────────────────────────────────────────────────────
    print('\n=== 1. Linear Probe ===')
    clf = LogisticRegression(
        solver='lbfgs', penalty='l2', max_iter=args.lp_max_iter,
        class_weight='balanced', C=1, n_jobs=args.n_jobs, verbose=1, tol=1e-6,
    )
    clf.fit(train_feats, train_y)

    for split, feats, labels_enc, labels_str in [
        ('train', train_feats, train_y, train_labels),
        ('val',   val_feats,   val_y,   val_labels),
    ]:
        proba = clf.predict_proba(feats)
        pred  = clf.predict(feats)
        ap_per = {}
        for i, cls in enumerate(classes):
            binary = (labels_enc == i).astype(int)
            if binary.sum() > 0:
                ap_per[cls] = round(float(average_precision_score(binary, proba[:, i])), 4)
        entry = {
            'top1_accuracy':          float(accuracy_score(labels_enc, pred)),
            'top1_balanced_accuracy': float(balanced_accuracy_score(labels_enc, pred)),
            'f1':                     float(f1_score(labels_enc, pred, average='weighted')),
            'precision':              float(f1_score(labels_enc, pred, average='macro')),
        }
        if split == 'val':
            entry['mean_average_precision'] = float(np.mean(list(ap_per.values())))
            entry['per_class_ap'] = ap_per
        metrics.setdefault('linear_probe', {})[split] = entry
        print(f'  {split}: bal_acc={entry["top1_balanced_accuracy"]:.4f}')
    save()

    # ── 2. kNN ────────────────────────────────────────────────────────────────
    print(f'\n=== 2. kNN (k={args.knn_k}) ===')
    knn = KNeighborsClassifier(n_neighbors=args.knn_k, metric='cosine', n_jobs=args.n_jobs)
    knn.fit(train_feats, train_y)
    knn_pred = knn.predict(val_feats)
    metrics['knn'] = {'k': args.knn_k, 'val': {
        'top1_accuracy':          float(accuracy_score(val_y, knn_pred)),
        'top1_balanced_accuracy': float(balanced_accuracy_score(val_y, knn_pred)),
        'f1':                     float(f1_score(val_y, knn_pred, average='weighted')),
    }}
    print(f'  val bal_acc: {metrics["knn"]["val"]["top1_balanced_accuracy"]:.4f}')
    save()

    # ── 3. Clustering ─────────────────────────────────────────────────────────
    print('\n=== 3. Clustering ===')
    km = MiniBatchKMeans(n_clusters=n_classes, n_init=10, max_iter=300)
    km.fit(np.concatenate([train_feats, val_feats]))
    val_cl = km.predict(val_feats)
    metrics['clustering'] = {'method': f'MiniBatchKMeans(k={n_classes}, n_init=10)', 'val': {
        'nmi': float(normalized_mutual_info_score(val_labels, val_cl)),
        'ari': float(adjusted_rand_score(val_labels, val_cl)),
    }}
    print(f'  NMI={metrics["clustering"]["val"]["nmi"]:.4f}  ARI={metrics["clustering"]["val"]["ari"]:.4f}')
    save()

    # ── 4. Neighbourhood purity ───────────────────────────────────────────────
    print(f'\n=== 4. Neighbourhood Purity (k={args.knn_k}) ===')
    nn = KNeighborsClassifier(n_neighbors=args.knn_k, metric='cosine', n_jobs=args.n_jobs)
    nn.fit(val_feats, val_y)
    neigh_idx = nn.kneighbors(val_feats, return_distance=False)
    per_class = {}
    for cls_idx, cls in enumerate(classes):
        sel = np.where(val_y == cls_idx)[0]
        if len(sel) == 0:
            continue
        purity = np.mean([
            np.mean(val_y[neigh_idx[i]] == cls_idx) for i in sel
        ])
        expected = (val_y == cls_idx).mean()
        per_class[cls] = {
            'purity': float(purity),
            'expected_chance': float(expected),
            'purity_lift': float(purity / expected) if expected > 0 else 0,
            'n_cells': int(len(sel)),
        }
    mean_purity = float(np.mean([v['purity'] for v in per_class.values()]))
    mean_lift   = float(np.mean([v['purity_lift'] for v in per_class.values()]))
    metrics['neighbourhood_purity'] = {
        'k': args.knn_k,
        'mean_purity': mean_purity,
        'mean_lift': mean_lift,
        'per_class': per_class,
    }
    print(f'  mean purity={mean_purity:.4f}  mean lift={mean_lift:.2f}x')
    save()

    # ── 5. Sample integration ─────────────────────────────────────────────────
    print('\n=== 5. Sample Integration ===')
    n_sil   = min(10_000, len(val_feats))
    sil_idx = np.random.default_rng(42).choice(len(val_feats), n_sil, replace=False)
    unique_ids = np.unique(val_ids[sil_idx])
    if len(unique_ids) >= 2:
        sil_sample = float(silhouette_score(val_feats[sil_idx], val_ids[sil_idx], metric='cosine'))
        sample_integration = -sil_sample
    else:
        sample_integration = float('nan')
    metrics['sample_integration'] = {
        'score': sample_integration,
        'n_samples': n_sil,
        'metric': 'cosine',
        'note': '+1=fully mixed, -1=fully separated',
    }
    print(f'  sample integration: {sample_integration:.4f}')
    save()

    # ── 6. UMAP ───────────────────────────────────────────────────────────────
    if HAS_UMAP:
        print('\n=== 6. UMAP ===')
        reducer = umap_lib.UMAP(n_components=2, metric='cosine',
                                n_neighbors=15, min_dist=0.1,
                                n_jobs=args.n_jobs, verbose=True)
        reducer.fit(train_feats)
        val_emb = reducer.transform(val_feats)

        cmap = plt.get_cmap('tab20', n_classes)
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, cls in enumerate(classes):
            m = val_y == i
            ax.scatter(val_emb[m, 0], val_emb[m, 1], s=1, alpha=0.4,
                       color=cmap(i), label=cls, rasterized=True)
        ax.legend(markerscale=6, bbox_to_anchor=(1.02, 1),
                  loc='upper left', fontsize=8, frameon=False)
        ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
        ax.set_title('Val features — UMAP (cell type)')
        plt.tight_layout()
        plt.savefig(out_dir / 'umap.pdf', bbox_inches='tight')
        plt.close()

        unique_ids = np.unique(val_ids)
        cmap_s = plt.get_cmap('nipy_spectral', len(unique_ids))
        id2int = {s: i for i, s in enumerate(unique_ids)}
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        for i, sid in enumerate(unique_ids):
            m = np.array([id2int[s] for s in val_ids]) == i
            ax2.scatter(val_emb[m, 0], val_emb[m, 1], s=1, alpha=0.4,
                        color=cmap_s(i), label=str(sid), rasterized=True)
        if len(unique_ids) <= 30:
            ax2.legend(markerscale=6, bbox_to_anchor=(1.02, 1),
                       loc='upper left', fontsize=8, frameon=False, title='Sample ID')
        ax2.set_xlabel('UMAP 1'); ax2.set_ylabel('UMAP 2')
        ax2.set_title('Val features — UMAP (sample ID)')
        plt.tight_layout()
        plt.savefig(out_dir / 'umap_sample.pdf', bbox_inches='tight')
        plt.close()
        print(f'Saved UMAPs to {out_dir}')
        metrics['umap'] = {'saved': True, 'n_neighbors': 15, 'min_dist': 0.1, 'metric': 'cosine'}
    else:
        metrics['umap'] = {'saved': False}
    save()

    # ── Save feature arrays ───────────────────────────────────────────────────
    val_pred_lr  = clf.predict(val_feats)
    train_pred_lr = clf.predict(train_feats)
    np.savez_compressed(
        out_dir / 'val_results.npz',
        features=val_feats, labels_str=val_labels,
        labels_num=val_y, sample_ids=val_ids,
        top1_pred_lr=val_pred_lr, top1_pred_knn=knn_pred,
        classes=np.array(classes),
    )
    np.savez_compressed(
        out_dir / 'train_results.npz',
        features=train_feats, labels_str=train_labels,
        labels_num=train_y, sample_ids=sample_ids[train_sel],
        top1_pred_lr=train_pred_lr,
        classes=np.array(classes),
    )
    print(f'Saved feature arrays to {out_dir}')

    # ── Summary ───────────────────────────────────────────────────────────────
    lp  = metrics['linear_probe']['val']['top1_balanced_accuracy']
    knn_acc = metrics['knn']['val']['top1_balanced_accuracy']
    nmi = metrics['clustering']['val']['nmi']
    ari = metrics['clustering']['val']['ari']
    si  = metrics['sample_integration']['score']
    print(f"""
╔══════════════════════════════════════════════════╗
║         Expression Baseline Summary (Val)        ║
╠══════════════════════════════════════════════════╣
║  Features: {args.feat:<38}║
║  Linear Probe  bal-acc  {lp:.4f}                  ║
║  k-NN (k={args.knn_k:2d})    bal-acc  {knn_acc:.4f}                  ║
║  Clustering    NMI      {nmi:.4f}   ARI {ari:.4f} ║
║  Sample integration (+1=mixed) {si:.4f}              ║
╚══════════════════════════════════════════════════╝
""")
    print(f'Saved metrics to {out_dir}/metrics.json')


if __name__ == '__main__':
    main()

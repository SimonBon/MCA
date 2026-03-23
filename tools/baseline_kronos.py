"""
KRONOS baseline feature extractor for multiplexed imaging.

Loads the pretrained KRONOS ViT-S/16 model, extracts per-marker features
[C × embed_dim] from cell-centred patches, and evaluates with the same
LP/kNN/clustering/LISI pipeline as baseline_expression.py.

Patches are loaded on-the-fly via a PyTorch DataLoader (no upfront RAM dump).

KRONOS architecture:
  - ViT-S/16, processes each marker channel independently through patch_embed
  - Adds sinusoidal marker embeddings (by marker_id) + positional embeddings
  - Positional embedding is interpolated → works for any spatial input size
  - Output used: marker_features [B, C, 384] flattened → [B, C*384]
    (matches KRONOS cell phenotyping tutorial)

Per-marker normalisation:
  Mean and std are computed from a subsample of training cells. This is a
  reasonable approximation when KRONOS's own SPM-47M stats are unavailable.
  Pass --marker_meta_csv to use KRONOS's own stats if available.

Usage:
    python tools/baseline_kronos.py \\
        --h5             /nobackup/.../CODEX_cHL.h5 \\
        --markers        /nobackup/.../used_markers.txt \\
        --train          /nobackup/.../train.txt \\
        --val            /nobackup/.../test.txt \\
        --out            /nobackup/.../paper_clean/CODEX_cHL/KRONOS \\
        --checkpoint     /nobackup/.../kronos_vits16_model.pt \\
        --kronos_src     /home/sgutwein/src/KRONOS
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, accuracy_score,
    normalized_mutual_info_score, adjusted_rand_score,
    silhouette_score, average_precision_score,
)
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import LabelEncoder

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print('umap-learn not installed — skipping UMAP')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--h5',              required=True)
    p.add_argument('--markers',         required=True)
    p.add_argument('--train',           required=True)
    p.add_argument('--val',             required=True)
    p.add_argument('--out',             required=True)
    p.add_argument('--checkpoint',      required=True,
                   help='Local .pt file or "hf_hub:MahmoodLab/kronos"')
    p.add_argument('--kronos_src',      default=None,
                   help='Path to KRONOS repo (added to sys.path)')
    p.add_argument('--model_type',      default='vits16',
                   choices=['vits16', 'vitl16'])
    p.add_argument('--patch_size',      type=int, default=32)
    p.add_argument('--batch_size',      type=int, default=64)
    p.add_argument('--num_workers',     type=int, default=8,
                   help='DataLoader workers for patch loading')
    p.add_argument('--ignore',          nargs='*', default=['Unidentified'])
    p.add_argument('--annotation_map',  default=None,
                   help='Comma-separated old:new pairs e.g. "Cytotoxic CD8:CD8,TReg:Treg"')
    p.add_argument('--n_jobs',          type=int, default=8)
    p.add_argument('--lp_max_iter',     type=int, default=5000)
    p.add_argument('--lp_tol',          type=float, default=1e-6,
                   help='Convergence tolerance for sklearn lbfgs LP')
    p.add_argument('--lp_C',            type=float, default=1.0,
                   help='Regularisation strength for LP (smaller = stronger regularisation)')
    p.add_argument('--skip_extract',    action='store_true',
                   help='Skip GPU feature extraction if train/val_results.npz already exist')
    p.add_argument('--lp_subsample',    type=int, default=None,
                   help='Subsample N train cells for LP fitting (reduces memory/time for large feature dims)')
    p.add_argument('--knn_k',           type=int, default=15)
    p.add_argument('--marker_max_values', type=float, default=65535.0,
                   help='Divide raw intensities by this before normalisation (65535 uint16, 1.0 if already scaled)')
    p.add_argument('--norm_n_cells',    type=int, default=5000,
                   help='Number of train cells to subsample for per-marker normalisation stats')
    p.add_argument('--marker_meta_csv', default=None,
                   help='Optional KRONOS marker_meta.csv with marker_id, marker_mean, marker_std')
    p.add_argument('--hf_token',        default=None,
                   help='HuggingFace auth token (required if checkpoint is hf_hub:)')
    p.add_argument('--cache_dir',       default=None,
                   help='Cache dir for HF downloads')
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_kronos(args):
    if args.kronos_src:
        if args.kronos_src not in sys.path:
            sys.path.insert(0, args.kronos_src)

    from kronos import create_model_from_pretrained

    model, precision, embed_dim = create_model_from_pretrained(
        checkpoint_path=args.checkpoint,
        cfg={"model_type": args.model_type, "token_overlap": False},
        hf_auth_token=args.hf_token,
        cache_dir=args.cache_dir,
    )
    model.eval()
    return model, precision, embed_dim


# ── Dataset ───────────────────────────────────────────────────────────────────

class KronosDataset(torch.utils.data.Dataset):
    """Streams cell patches on the fly from H5. Opens H5 lazily per worker."""

    def __init__(self, h5_path, marker_indices, dim1, dim2, sample_ids, patch_size):
        self.h5_path       = h5_path
        self.marker_indices = np.array(marker_indices)
        self.dim1          = dim1
        self.dim2          = dim2
        self.sample_ids    = sample_ids
        self.patch_size    = patch_size
        self._h5           = None  # opened lazily in each worker

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')

    def __len__(self):
        return len(self.dim1)

    def __getitem__(self, idx):
        self._open()
        half = self.patch_size // 2
        d1, d2, sid = int(self.dim1[idx]), int(self.dim2[idx]), self.sample_ids[idx]
        grp = self._h5['data'][sid]
        H_img, W_img = grp['image'].shape[:2]

        r0, r1 = d1 - half, d1 + half
        c0, c1 = d2 - half, d2 + half
        r0c, r1c = max(0, r0), min(H_img, r1)
        c0c, c1c = max(0, c0), min(W_img, c1)

        chunk = grp['image'][r0c:r1c, c0c:c1c, :][:, :, self.marker_indices].astype(np.float32)
        patch = np.zeros((self.patch_size, self.patch_size, len(self.marker_indices)), dtype=np.float32)
        patch[r0c - r0:r1c - r0, c0c - c0:c1c - c0, :] = chunk
        patch = patch.transpose(2, 0, 1)  # [C, H, W]

        if 'masks' in grp:
            chunk_m = grp['masks'][r0c:r1c, c0c:c1c]
            mask = np.zeros((self.patch_size, self.patch_size), dtype=chunk_m.dtype)
            mask[r0c - r0:r1c - r0, c0c - c0:c1c - c0] = chunk_m
            centre    = mask[half, half]
            cell_mask = (mask == centre).astype(np.float32)
            if cell_mask.sum() == 0:
                cell_mask = np.ones((self.patch_size, self.patch_size), dtype=np.float32)
        else:
            cell_mask = np.ones((self.patch_size, self.patch_size), dtype=np.float32)

        return torch.from_numpy(patch), torch.from_numpy(cell_mask)


def make_loader(h5_path, marker_indices, dim1, dim2, sample_ids,
                patch_size, batch_size, num_workers, shuffle=False):
    ds = KronosDataset(h5_path, marker_indices, dim1, dim2, sample_ids, patch_size)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


# ── Per-marker normalisation stats (streaming) ────────────────────────────────

def compute_norm_stats_streaming(loader, marker_max_values):
    """
    Stream patches from loader, compute per-marker mean and std.
    patches are [B, C, H, W] float32; masks are [B, H, W] float32 binary.
    Stats are computed over cell-masked pixels only (mask > 0).
    """
    sum_vals  = None
    sum_sq    = None
    counts    = None

    for patches, masks in loader:
        # patches: [B, C, H, W], masks: [B, H, W]
        B, C, H, W = patches.shape
        if sum_vals is None:
            sum_vals = np.zeros(C, dtype=np.float64)
            sum_sq   = np.zeros(C, dtype=np.float64)
            counts   = np.zeros(C, dtype=np.float64)

        scaled = patches.numpy() / marker_max_values  # [B, C, H, W]
        m = masks.numpy()  # [B, H, W]
        for c in range(C):
            vals = scaled[:, c, :, :][m > 0]  # all cell pixels for this marker
            sum_vals[c] += vals.sum()
            sum_sq[c]   += (vals ** 2).sum()
            counts[c]   += len(vals)

    mean = (sum_vals / np.maximum(counts, 1)).astype(np.float32)
    var  = (sum_sq / np.maximum(counts, 1) - mean ** 2).astype(np.float32)
    std  = np.sqrt(np.maximum(var, 0)).astype(np.float32)
    std  = np.where(std < 1e-6, 1.0, std)
    return mean, std


# ── KRONOS feature extraction (streaming) ────────────────────────────────────

@torch.no_grad()
def extract_features(model, loader, mean, std, marker_ids,
                     marker_max_values, device, precision, total):
    """
    Stream patches from loader through KRONOS model.

    Returns:
        marker_features : [N, C * embed_dim]  (per-marker, flattened)
        patch_features  : [N, embed_dim]      (CLS token)
    """
    mean_t = torch.tensor(mean, dtype=precision)[None, :, None, None].to(device)
    std_t  = torch.tensor(std,  dtype=precision)[None, :, None, None].to(device)
    mids_t = torch.tensor(marker_ids, dtype=torch.int64).unsqueeze(0).to(device)

    all_marker_feats = []
    all_patch_feats  = []
    done = 0

    for patches, masks in loader:
        B = patches.shape[0]
        # Normalise on GPU: /max_values then (x - mean) / std
        batch = patches.to(device=device, dtype=precision)
        batch = batch / marker_max_values
        batch = (batch - mean_t) / std_t

        # Zero out non-cell pixels
        cmask = masks.to(device=device, dtype=precision).unsqueeze(1)  # [B, 1, H, W]
        batch = batch * cmask  # broadcast over C

        mids = mids_t.expand(B, -1)  # [B, C]
        patch_feats, marker_feats, _ = model(batch, marker_ids=mids)
        # marker_feats: [B, C, embed_dim] → [B, C*embed_dim]
        all_marker_feats.append(marker_feats.cpu().float().numpy().reshape(B, -1))
        all_patch_feats.append(patch_feats.cpu().float().numpy())

        done += B
        if done % (args_global.batch_size * 10) == 0 or done == total:
            print(f'  features {done}/{total}', flush=True)

    return (np.concatenate(all_marker_feats, axis=0),
            np.concatenate(all_patch_feats,  axis=0))


# ── LISI ──────────────────────────────────────────────────────────────────────

def compute_lisi(features, labels, n_neighbors=90, metric='cosine'):
    n = len(features)
    k = min(n_neighbors, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(features)
    _, indices = nbrs.kneighbors(features)
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    label2i = {l: i for i, l in enumerate(unique_labels)}
    label_idx = np.array([label2i[l] for l in labels])
    scores = []
    for i in range(n):
        counts = np.bincount(label_idx[indices[i]], minlength=n_labels)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        scores.append(1.0 / np.sum(probs ** 2))
    return float(np.mean(scores))


# ── Marker meta CSV ───────────────────────────────────────────────────────────

def load_marker_meta(csv_path, marker_names):
    import pandas as pd

    # Name aliases: our marker names → KRONOS metadata names (all uppercase)
    ALIASES = {
        'DAPI-01':      'DAPI',
        'DAPI':         'DAPI',
        'CYTOKERITIN':  'CYTOKERATIN',
        'CYTOKERATIN':  'CYTOKERATIN',
        'CYTOKERATIN18':'CYTOKERATIN',
        'GRANZYMEB':    'GZMB',
        'GRANZYMEБ':    'GZMB',
        'HLA-DR':       'HLA_DR',
        'IDO-1':        'IDO1',
        'IDO1':         'IDO1',
        'LAG-3':        'LAG3',
        'PD-1':         'PD1',
        'PD-L1':        'PDL1',
        'TIM-3':        'TIM3',
        'TCRB':         'TCR_B',
        'TCRΒ':         'TCR_B',
        'BCL-2':        'BCL2',
        'MUC-1':        'MUC1',
        'B-CATENIN':    'B-CATENIN',
        'COLLAGEN4':    'COLLAGEN',
        # IMC / NB aliases
        'CD8A':         'CD8',
        'KI-67':        'KI67',
        'CD274':        'PDL1',
        'CD279':        'PD1',
        # MIBI_TNBC aliases
        'IDO':          'IDO1',
        'BETA CATENIN': 'B-CATENIN',
    }

    df = pd.read_csv(csv_path)
    df['marker_name'] = df['marker_name'].str.upper()
    df = df.set_index('marker_name')

    marker_ids = []
    means = []
    stds  = []
    for i, m in enumerate(marker_names):
        key = ALIASES.get(m.upper(), m.upper())
        if key in df.index:
            marker_ids.append(int(df.loc[key, 'marker_id']))
            means.append(float(df.loc[key, 'marker_mean']))
            stds.append(float(df.loc[key, 'marker_std']))
        else:
            print(f'  WARNING: {m} (lookup: {key}) not in marker_meta.csv — using unknown placeholder id=5 + data stats')
            marker_ids.append(5)  # fixed gap ID not used by any SPM-47M marker
            means.append(None)
            stds.append(None)

    return marker_ids, means, stds


# ── Main ──────────────────────────────────────────────────────────────────────

args_global = None  # set in main() so extract_features can access batch_size for progress


def main():
    global args_global
    args = parse_args()
    args_global = args
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load KRONOS model ─────────────────────────────────────────────────────
    print('Loading KRONOS model...')
    model, precision, embed_dim = load_kronos(args)
    model.to(device)
    print(f'  embed_dim={embed_dim}  precision={precision}')

    # ── Load metadata (coords, labels — lightweight) ──────────────────────────
    print('Loading H5 metadata...')
    with h5py.File(args.h5, 'r') as f:
        all_markers  = f['marker_names'][:].astype(str)
        dim1         = f['coords']['DIM1'][:]
        dim2         = f['coords']['DIM2'][:]
        sample_ids   = f['coords']['sample_id'][:].astype(str)
        annotations  = f['annotation'][()].astype(str)

    used = np.loadtxt(args.markers, dtype=str, delimiter=',')
    marker2idx     = {m: i for i, m in enumerate(all_markers)}
    marker_indices = np.array([marker2idx[m] for m in used])
    print(f'  {len(used)} markers: {list(used)}')

    # ── Annotation mapping ────────────────────────────────────────────────────
    if args.annotation_map:
        amap = dict(pair.split(':') for pair in args.annotation_map.split(','))
        annotations = np.array([amap.get(a, a) for a in annotations])

    # ── Filter ignored classes ────────────────────────────────────────────────
    keep = np.ones(len(annotations), dtype=bool)
    for cls in (args.ignore or []):
        keep &= (annotations != cls)
    dim1, dim2, sample_ids, annotations = (
        dim1[keep], dim2[keep], sample_ids[keep], annotations[keep])
    print(f'  {keep.sum()} cells after filtering')

    # ── Train / val split ─────────────────────────────────────────────────────
    train_idx = np.loadtxt(args.train, dtype=int)
    val_idx   = np.loadtxt(args.val,   dtype=int)
    orig_indices = np.where(keep)[0]
    orig2new     = {o: n for n, o in enumerate(orig_indices)}
    train_sel = np.array([orig2new[i] for i in train_idx if i in orig2new])
    val_sel   = np.array([orig2new[i] for i in val_idx   if i in orig2new])

    train_labels = annotations[train_sel]
    val_labels   = annotations[val_sel]
    val_ids      = sample_ids[val_sel]

    le = LabelEncoder().fit(np.concatenate([train_labels, val_labels]))
    train_y = le.transform(train_labels)
    val_y   = le.transform(val_labels)
    classes = list(le.classes_)
    n_classes = len(classes)

    # ── Marker IDs and normalisation stats ────────────────────────────────────
    if args.marker_meta_csv:
        print('Loading marker metadata from CSV...')
        marker_ids, meta_means, meta_stds = load_marker_meta(args.marker_meta_csv, used)
    else:
        marker_ids = list(range(4, 4 + len(used)))
        meta_means = [None] * len(used)
        meta_stds  = [None] * len(used)

    needs_data_stats = any(m is None for m in meta_means)
    print(f'  marker_ids: {marker_ids}')

    # ── Compute normalisation stats from subsample of train ───────────────────
    if needs_data_stats:
        n_sub = min(args.norm_n_cells, len(train_sel))
        rng   = np.random.default_rng(42)
        sub   = rng.choice(len(train_sel), n_sub, replace=False)
        sub_sel = train_sel[sub]
        print(f'Computing normalisation stats from {n_sub} train cells (streaming)...')
        norm_loader = make_loader(
            args.h5, marker_indices,
            dim1[sub_sel], dim2[sub_sel], sample_ids[sub_sel],
            args.patch_size, batch_size=256, num_workers=args.num_workers,
        )
        data_mean, data_std = compute_norm_stats_streaming(norm_loader, args.marker_max_values)
        del norm_loader
    else:
        data_mean = data_std = None

    final_mean = np.array([
        m if m is not None else data_mean[i]
        for i, m in enumerate(meta_means)
    ], dtype=np.float32)
    final_std = np.array([
        s if s is not None else data_std[i]
        for i, s in enumerate(meta_stds)
    ], dtype=np.float32)
    np.save(out_dir / 'marker_norm_stats.npy', np.stack([final_mean, final_std]))

    train_npz = out_dir / 'train_results.npz'
    val_npz   = out_dir / 'val_results.npz'

    if args.skip_extract and train_npz.exists() and val_npz.exists():
        print('\nLoading cached features (--skip_extract)...')
        tr = np.load(train_npz, allow_pickle=True)
        vl = np.load(val_npz,   allow_pickle=True)
        train_feats   = tr['features']
        val_feats     = vl['features']
        train_patch_feats = np.zeros((len(train_feats), 1), dtype=np.float32)  # placeholder
        val_patch_feats   = np.zeros((len(val_feats),   1), dtype=np.float32)
        print(f'  train: {train_feats.shape}  val: {val_feats.shape}')
    else:
        # ── Extract features: train ───────────────────────────────────────────
        print(f'\nExtracting train features ({len(train_sel)} cells, batch={args.batch_size})...')
        train_loader = make_loader(
            args.h5, marker_indices,
            dim1[train_sel], dim2[train_sel], sample_ids[train_sel],
            args.patch_size, args.batch_size, args.num_workers,
        )
        train_marker_feats, train_patch_feats = extract_features(
            model, train_loader, final_mean, final_std, marker_ids,
            args.marker_max_values, device, precision, len(train_sel))
        del train_loader

        # ── Extract features: val ─────────────────────────────────────────────
        print(f'\nExtracting val features ({len(val_sel)} cells, batch={args.batch_size})...')
        val_loader = make_loader(
            args.h5, marker_indices,
            dim1[val_sel], dim2[val_sel], sample_ids[val_sel],
            args.patch_size, args.batch_size, args.num_workers,
        )
        val_marker_feats, val_patch_feats = extract_features(
            model, val_loader, final_mean, final_std, marker_ids,
            args.marker_max_values, device, precision, len(val_sel))
        del val_loader

        train_feats = train_marker_feats
        val_feats   = val_marker_feats

    train_feats = train_feats if isinstance(train_feats, np.ndarray) else np.array(train_feats)
    val_feats   = val_feats   if isinstance(val_feats,   np.ndarray) else np.array(val_feats)
    print(f'  Marker feature dim: {train_feats.shape[1]} ({len(used)} markers × {embed_dim})')
    print(f'  Patch (CLS) feature dim: {train_patch_feats.shape[1]}')

    metrics = {
        'model': 'KRONOS',
        'model_type': args.model_type,
        'checkpoint': args.checkpoint,
        'classes': classes,
        'n_classes': n_classes,
        'feature_dim': int(train_feats.shape[1]),
        'patch_size': args.patch_size,
        'marker_ids': marker_ids,
        'n_markers': len(used),
        'feature_type': 'marker_features_flattened',
    }

    def save():
        with open(out_dir / 'metrics.json', 'w') as fp:
            json.dump(metrics, fp, indent=2)

    # ── 1. Linear Probe ───────────────────────────────────────────────────────
    print('\n=== 1. Linear Probe ===')
    lp_train_feats, lp_train_y = train_feats, train_y
    if args.lp_subsample and args.lp_subsample < len(train_feats):
        rng = np.random.default_rng(42)
        lp_idx = rng.choice(len(train_feats), args.lp_subsample, replace=False)
        lp_train_feats = train_feats[lp_idx]
        lp_train_y     = train_y[lp_idx]
        print(f'  Subsampled to {args.lp_subsample} train cells for LP')
    clf = LogisticRegression(
        solver='lbfgs', penalty='l2', max_iter=args.lp_max_iter,
        class_weight='balanced', C=args.lp_C, n_jobs=args.n_jobs, verbose=1,
        tol=args.lp_tol,
    )
    print(f'  Fitting on {lp_train_feats.shape[0]} cells × {lp_train_feats.shape[1]} dims ...', flush=True)
    clf.fit(lp_train_feats, lp_train_y)
    print('  LP fit done.', flush=True)
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
    metrics['clustering'] = {'method': f'MiniBatchKMeans(k={n_classes})', 'val': {
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
        purity   = float(np.mean([np.mean(val_y[neigh_idx[i]] == cls_idx) for i in sel]))
        expected = float((val_y == cls_idx).mean())
        per_class[cls] = {
            'purity': purity, 'expected_chance': expected,
            'purity_lift': purity / expected if expected > 0 else 0,
            'n_cells': int(len(sel)),
        }
    mean_purity = float(np.mean([v['purity'] for v in per_class.values()]))
    metrics['neighbourhood_purity'] = {
        'k': args.knn_k, 'mean_purity': mean_purity, 'per_class': per_class}
    print(f'  mean purity={mean_purity:.4f}')
    save()

    # ── 5. Sample integration ─────────────────────────────────────────────────
    print('\n=== 5. Sample Integration ===')
    n_sil   = min(10_000, len(val_feats))
    sil_idx = np.random.default_rng(42).choice(len(val_feats), n_sil, replace=False)
    if len(np.unique(val_ids[sil_idx])) >= 2:
        sil_sample = float(silhouette_score(val_feats[sil_idx], val_ids[sil_idx], metric='cosine'))
    else:
        sil_sample = float('nan')
    metrics['sample_integration'] = {
        'score': -sil_sample, 'n_samples': n_sil, 'metric': 'cosine'}
    print(f'  sample integration: {-sil_sample:.4f}')
    save()

    # ── 6. cLISI / iLISI ─────────────────────────────────────────────────────
    print('\n=== 6. cLISI / iLISI ===')
    n_lisi  = min(10_000, len(val_feats))
    li_idx  = np.random.default_rng(42).choice(len(val_feats), n_lisi, replace=False)
    clisi   = compute_lisi(val_feats[li_idx], val_labels[li_idx])
    ilisi   = compute_lisi(val_feats[li_idx], val_ids[li_idx])
    metrics['cLISI'] = clisi
    metrics['iLISI'] = ilisi
    print(f'  cLISI={clisi:.4f}  iLISI={ilisi:.4f}')
    save()

    # ── 7. UMAP ───────────────────────────────────────────────────────────────
    if HAS_UMAP:
        print('\n=== 7. UMAP ===')
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
        ax.legend(markerscale=6, bbox_to_anchor=(1.02, 1), loc='upper left',
                  fontsize=8, frameon=False)
        ax.set_title('KRONOS val features — UMAP (cell type)')
        plt.tight_layout()
        plt.savefig(out_dir / 'umap.pdf', bbox_inches='tight')
        plt.close()

        unique_ids = np.unique(val_ids)
        cmap_s = plt.get_cmap('nipy_spectral', len(unique_ids))
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        for i, sid in enumerate(unique_ids):
            m = val_ids == sid
            ax2.scatter(val_emb[m, 0], val_emb[m, 1], s=1, alpha=0.4,
                        color=cmap_s(i), label=str(sid), rasterized=True)
        if len(unique_ids) <= 30:
            ax2.legend(markerscale=6, bbox_to_anchor=(1.02, 1), loc='upper left',
                       fontsize=8, frameon=False, title='Sample ID')
        ax2.set_title('KRONOS val features — UMAP (sample ID)')
        plt.tight_layout()
        plt.savefig(out_dir / 'umap_sample.pdf', bbox_inches='tight')
        plt.close()

        np.savez_compressed(
            out_dir / 'umap_embeddings.npz',
            embedding=val_emb, labels_str=val_labels,
            labels_num=val_y, sample_ids=val_ids,
        )
        metrics['umap'] = {'saved': True}
        print('  Saved UMAPs')
    save()

    # ── Save feature arrays ───────────────────────────────────────────────────
    np.savez_compressed(
        out_dir / 'train_results.npz',
        features=train_feats, labels_str=train_labels,
        labels_num=train_y, sample_ids=train_ids,
        classes=np.array(classes),
    )
    np.savez_compressed(
        out_dir / 'val_results.npz',
        features=val_feats, labels_str=val_labels,
        labels_num=val_y, sample_ids=val_ids,
        top1_pred_lr=clf.predict(val_feats), top1_pred_knn=knn_pred,
        classes=np.array(classes),
    )
    np.savez_compressed(
        out_dir / 'val_results_cls.npz',
        features=val_patch_feats, labels_str=val_labels,
        labels_num=val_y, sample_ids=val_ids, classes=np.array(classes),
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    lp      = metrics['linear_probe']['val']['top1_balanced_accuracy']
    knn_acc = metrics['knn']['val']['top1_balanced_accuracy']
    nmi     = metrics['clustering']['val']['nmi']
    ari     = metrics['clustering']['val']['ari']
    si      = metrics['sample_integration']['score']
    print(f"""
╔══════════════════════════════════════════════════╗
║            KRONOS Baseline Summary (Val)         ║
╠══════════════════════════════════════════════════╣
║  Model type: {args.model_type:<36}║
║  Feature dim: {train_feats.shape[1]:<35}║
║  Linear Probe  bal-acc  {lp:.4f}                  ║
║  kNN           bal-acc  {knn_acc:.4f}                  ║
║  Clustering    NMI      {nmi:.4f}                  ║
║  Clustering    ARI      {ari:.4f}                  ║
║  Sample integ. score    {si:.4f}                  ║
╚══════════════════════════════════════════════════╝
""")
    print(f'Results saved to {out_dir}')


if __name__ == '__main__':
    main()

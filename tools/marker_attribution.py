#!/usr/bin/env python3
"""Per-cell marker attribution via Integrated Gradients, visualised on the
training-run UMAP.

Uses MCIDataset (same class and val indices as the training eval hook) so
cell ordering is guaranteed to match umap_embeddings.npz.

Workflow:
  1. Load backbone  →  Config.fromfile registers MCIDataset in mmengine
  2. Build val dataset via MCIDataset (same ordering as training val hook)
  3. Compute IG attribution  [N, n_markers]  per cell
  4. Load umap_embeddings.npz  →  same N cells, same order  →  direct overlay
  5. Plot per-marker UMAP panels coloured by IG attribution
  6. Save attribution.npz

Usage:
    python tools/marker_attribution.py \\
        --model_dir  /nobackup/.../paper_clean/CODEX_cHL/CIM_Funnel_Large \\
        --h5         /nobackup/.../h5_files/CODEX_cHL/CODEX_cHL.h5 \\
        --markers    /nobackup/.../h5_files/CODEX_cHL/used_markers.txt \\
        --val        /nobackup/.../h5_files/CODEX_cHL/test.txt \\
        --umap_emb   /nobackup/.../paper_clean/CODEX_cHL/CIM_Funnel_Large/umap_embeddings.npz \\
        --out        /nobackup/.../marker_attribution/CODEX_cHL_CIM_Funnel \\
        --annotation_map "Cytotoxic CD8:CD8,TReg:Treg"

Outputs:
    attribution.npz          — [N, n_markers] IG matrix + labels + sample_ids
    umap_marker_influence.png — grid: one UMAP panel per marker
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

_TOOLS = Path(__file__).resolve().parent
_MCA   = _TOOLS.parent
_SRC   = _MCA.parent
for _p in [str(_MCA), str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Model + dataset loading ────────────────────────────────────────────────────

def load_backbone(model_dir, device):
    """Load backbone; Config.fromfile registers MCIDataset as a side-effect."""
    from src.utils import load_checkpoint
    result   = load_checkpoint(str(model_dir), device=device)
    backbone = result['model'].backbone.eval()
    return backbone


def build_val_dataset(h5, markers, val, patch_size, annotation_map, ignore):
    """Build MCIDataset from the mmengine registry (registered by load_backbone).

    Returns the dataset with pipeline=None so __getitem__ returns the raw dict.
    Cell ordering is identical to the training val hook.
    """
    from mmengine.registry import DATASETS
    MCIDataset = DATASETS.get('MCIDataset')

    ignore_list = [s.strip() for s in ignore.split(',') if s.strip()] if ignore else None

    dataset = MCIDataset(
        h5_filepath       = h5,
        patch_size        = patch_size,
        used_markers      = markers,
        used_indicies     = val,
        pipeline          = None,       # returns raw dict
        ignore_annotation = ignore_list,
        mask_patch        = True,
        annotation_map    = annotation_map,
    )
    return dataset


def collate(batch):
    """Convert list of MCIDataset dicts → (imgs [B,C,H,W], labels, sample_ids)."""
    imgs = []
    labels, sids = [], []
    for d in batch:
        img = torch.from_numpy(d['img']).permute(2, 0, 1).float()  # [C,H,W]
        img = img / (img.amax() + 1e-8)
        imgs.append(img)
        labels.append(d['annotation'])
        sids.append(d['sample_id'])
    return torch.stack(imgs), labels, sids


# ── Integrated Gradients ───────────────────────────────────────────────────────

def ig_batch(backbone, imgs, n_steps):
    """Integrated Gradients. Target: ||f(x)||². Baseline: zero patch.

    Returns attribution [B, C] — abs IG summed over spatial dims.
    """
    baseline   = torch.zeros_like(imgs)
    grad_accum = torch.zeros_like(imgs)

    for step in range(n_steps):
        alpha    = step / max(n_steps - 1, 1)
        x_interp = (baseline + alpha * (imgs - baseline)).detach().requires_grad_(True)
        emb      = backbone(x_interp)[0]
        emb.pow(2).sum().backward()
        grad_accum += x_interp.grad.detach()

    avg_grads   = grad_accum / n_steps
    attribution = ((imgs - baseline) * avg_grads).abs()
    return attribution.sum(dim=[-2, -1]).cpu()   # [B, C]


# ── Visualisation ──────────────────────────────────────────────────────────────

def plot_marker_panels(attr_matrix, marker_names, umap_coords, labels,
                       out_path, percentile=99):
    """Grid of UMAPs: one panel per marker, coloured by row-normalised IG.

    umap_coords: [N, 2]  — from umap_embeddings.npz, same cell order.
    """
    attr_norm = attr_matrix / (attr_matrix.sum(axis=1, keepdims=True) + 1e-8)

    n     = len(marker_names)
    ncols = 8
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.0))
    axes = axes.flatten()

    for i, name in enumerate(marker_names):
        ax   = axes[i]
        val  = attr_norm[:, i]
        vmax = np.percentile(val, percentile)
        sc   = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                          c=val, s=1, alpha=0.6, linewidths=0,
                          cmap='YlOrRd', vmin=0, vmax=vmax)
        ax.set_title(name, fontsize=7, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Marker attribution influence (IG) on original embedding UMAP\n'
                 '(row-normalised, per-marker colour scale)',
                 fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out_path}')


def plot_umap_groundtruth(umap_coords, labels, out_path):
    import matplotlib.cm as cm
    categories = sorted(set(labels))
    palette    = cm.get_cmap('tab20', len(categories))
    cmap_dict  = {c: palette(i) for i, c in enumerate(categories)}
    colors     = [cmap_dict[l] for l in labels]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
               c=colors, s=2, alpha=0.5, linewidths=0)
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=cmap_dict[c], markersize=6, label=c)
               for c in categories]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc='upper left',
              fontsize=7, frameon=False)
    ax.set_title('Ground-truth labels (original embedding UMAP)')
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved {out_path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir',      required=True)
    p.add_argument('--h5',             required=True)
    p.add_argument('--markers',        required=True)
    p.add_argument('--val',            required=True)
    p.add_argument('--umap_emb',       default=None,
                   help='Path to umap_embeddings.npz from the training run. '
                        'If omitted, UMAP is computed from backbone features.')
    p.add_argument('--out',            required=True)
    p.add_argument('--annotation_map', default=None,
                   help='e.g. "Cytotoxic CD8:CD8,TReg:Treg"')
    p.add_argument('--ignore',         default=None,
                   help='Comma-separated class names to ignore')
    p.add_argument('--patch_size',     type=int, default=22)
    p.add_argument('--n_steps',        type=int, default=20)
    p.add_argument('--batch_size',     type=int, default=64)
    p.add_argument('--max_cells',      type=int, default=None)
    p.add_argument('--n_workers',      type=int, default=8)
    p.add_argument('--device',         default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    out    = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f'Device: {device}')

    annotation_map = {}
    if args.annotation_map:
        for pair in args.annotation_map.split(','):
            k, v = pair.split(':')
            annotation_map[k.strip()] = v.strip()

    # ── 1. Load backbone (registers MCIDataset) ────────────────────────────
    print('Loading backbone...')
    backbone = load_backbone(args.model_dir, device)

    # ── 2. Build val dataset via MCIDataset (same order as training hook) ──
    print('Building val dataset (MCIDataset)...')
    dataset = build_val_dataset(
        h5           = args.h5,
        markers      = args.markers,
        val          = args.val,
        patch_size   = args.patch_size,
        annotation_map = annotation_map,
        ignore       = args.ignore,
    )
    marker_names = list(dataset.used_markers)
    print(f'  {len(dataset)} cells | {len(marker_names)} markers')
    print(f'  Classes: {sorted(set(dataset.annotation))}')

    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.n_workers,
        collate_fn  = collate,
        pin_memory  = device.type == 'cuda',
    )

    # ── 3. Compute IG + extract features ──────────────────────────────────
    all_attr, all_feats, all_labels, all_sids = [], [], [], []
    n_done = 0

    for imgs, labels, sids in tqdm(loader, desc='IG'):
        if args.max_cells and n_done >= args.max_cells:
            break
        imgs_gpu = imgs.to(device)
        attr = ig_batch(backbone, imgs_gpu, args.n_steps)
        with torch.no_grad():
            feats = backbone(imgs_gpu)[0].squeeze(-1).squeeze(-1)  # [B, D]
        all_attr.append(attr.numpy())
        all_feats.append(feats.cpu().numpy())
        all_labels.extend(labels)
        all_sids.extend(sids)
        n_done += len(labels)

    attr_matrix = np.concatenate(all_attr,  axis=0)   # [N, n_markers]
    feat_matrix = np.concatenate(all_feats, axis=0)   # [N, D]
    labels_arr  = np.array(all_labels)
    sids_arr    = np.array(all_sids)
    print(f'\nAttribution matrix: {attr_matrix.shape}')
    print(f'Feature matrix:     {feat_matrix.shape}')

    # ── 4. UMAP coords — load from file or compute from features ──────────
    if args.umap_emb:
        print(f'Loading UMAP embeddings from {args.umap_emb}...')
        emb_data   = np.load(args.umap_emb, allow_pickle=True)
        umap_all   = emb_data['embedding']      # [N_total, 2]
        labels_emb = emb_data['labels_str'].astype(str)

        # Apply same annotation_map + ignore to get matching subset
        ignore_set = set(s.strip() for s in args.ignore.split(',') if s.strip()) \
                     if args.ignore else set()
        labels_emb_mapped = np.array([annotation_map.get(l, l) for l in labels_emb])
        keep       = np.array([l not in ignore_set for l in labels_emb_mapped])
        umap_coords = umap_all[keep]

        if len(umap_coords) != len(attr_matrix):
            print(f'WARNING: UMAP has {len(umap_coords)} cells after filtering, '
                  f'attribution has {len(attr_matrix)}. Check --ignore / --annotation_map.')
        else:
            match = np.mean(labels_emb_mapped[keep] == labels_arr)
            print(f'  Label alignment: {match*100:.1f}% match (should be ~100%)')
    else:
        print('No --umap_emb provided — computing UMAP from backbone features...')
        from sklearn.decomposition import PCA
        import umap as umap_lib
        n_pca = min(50, feat_matrix.shape[1])
        print(f'  PCA {feat_matrix.shape[1]}→{n_pca}...')
        pca_feats = PCA(n_components=n_pca, random_state=42).fit_transform(feat_matrix)
        print(f'  UMAP {n_pca}→2 on {len(pca_feats)} cells...')
        umap_coords = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                                    random_state=42).fit_transform(pca_feats)
        print(f'  Done. UMAP shape: {umap_coords.shape}')

    # ── 5. Plot ────────────────────────────────────────────────────────────
    plot_umap_groundtruth(umap_coords, labels_arr, out / 'umap_groundtruth.png')
    plot_marker_panels(attr_matrix, marker_names, umap_coords, labels_arr,
                       out / 'umap_marker_influence.png')

    # ── 6. Save ────────────────────────────────────────────────────────────
    np.savez_compressed(
        out / 'attribution.npz',
        attribution  = attr_matrix,
        features     = feat_matrix,
        labels       = labels_arr,
        sample_ids   = sids_arr,
        marker_names = np.array(marker_names),
        umap_coords  = umap_coords,
    )
    print(f'\nDone. Outputs in {out}')


if __name__ == '__main__':
    main()

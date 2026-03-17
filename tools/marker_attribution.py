#!/usr/bin/env python3
"""Per-cell marker attribution via Integrated Gradients.

Computes a [N_cells × n_markers] importance matrix:
  For each cell, which markers drove its position in embedding space?

Loads data directly from HDF5 (no mmengine dataset pipeline) to avoid
module registration conflicts. Model is loaded via src.utils.load_checkpoint.

Attribution target: ||f(x)||²  — fully unsupervised, no labels used.
Attribution = (input - baseline) * mean_gradients  [Integrated Gradients]
Spatial dims summed → one importance value per marker per cell.

Usage (on cluster, GPU recommended):
    python tools/marker_attribution.py \\
        --model_dir /nobackup/.../paper_clean/CODEX_cHL/CIM_Funnel_Large \\
        --h5        /nobackup/.../h5_files/CODEX_cHL/CODEX_cHL.h5 \\
        --markers   /nobackup/.../h5_files/CODEX_cHL/used_markers.txt \\
        --val       /nobackup/.../h5_files/CODEX_cHL/test.txt \\
        --out       /nobackup/.../z_RUNS/marker_attribution/CODEX_cHL_CIM_Funnel \\
        --annotation_map "Cytotoxic CD8:CD8,TReg:Treg"

Outputs:
    attribution.npz   — [N, n_markers] importance matrix + labels + sample_ids
    marker_names.txt  — marker names in column order
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

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ── Model loading ──────────────────────────────────────────────────────────────

def load_backbone(model_dir, device):
    """Load backbone via src.utils.load_checkpoint (handles Config.fromfile)."""
    from src.utils import load_checkpoint
    result   = load_checkpoint(str(model_dir), device=device)
    backbone = result['model'].backbone.eval()
    return backbone


# ── HDF5 Dataset ───────────────────────────────────────────────────────────────

def decode(x):
    return x.decode() if isinstance(x, bytes) else x


class PatchDataset(Dataset):
    """Load cell patches directly from HDF5, mirroring MCIDataset behaviour."""

    def __init__(self, h5_path, marker_indices, indices_path,
                 patch_size, annotation_map, ignore):
        self.h5_path       = h5_path
        self.marker_indices = marker_indices
        self.patch_size    = patch_size
        self.half          = patch_size // 2
        self.annotation_map = annotation_map
        self.ignore        = set(ignore or [])
        self._h5           = None

        with h5py.File(h5_path, 'r') as f:
            all_sids  = np.array([decode(s) for s in f['coords']['sample_id'][:]])
            all_dim1  = f['coords']['DIM1'][:].astype(int)
            all_dim2  = f['coords']['DIM2'][:].astype(int)
            all_annot = np.array([decode(a) for a in f['annotation'][:]])

        # Filter to requested indices
        with open(indices_path) as fh:
            idx = np.array([int(l.strip()) for l in fh if l.strip()])

        self.dim1      = all_dim1[idx]
        self.dim2      = all_dim2[idx]
        self.sample_id = all_sids[idx]
        self.labels    = np.array([
            annotation_map.get(a, a) for a in all_annot[idx]
        ])

        # Filter ignored classes
        keep = np.array([l not in self.ignore for l in self.labels])
        self.dim1      = self.dim1[keep]
        self.dim2      = self.dim2[keep]
        self.sample_id = self.sample_id[keep]
        self.labels    = self.labels[keep]

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')

    def __len__(self):
        return len(self.dim1)

    def __getitem__(self, idx):
        self._open()
        d1, d2, sid = self.dim1[idx], self.dim2[idx], self.sample_id[idx]
        half = self.half
        ps   = self.patch_size

        grp   = self._h5['data'][sid]
        H, W  = grp['image'].shape[:2]
        r0, r1 = d1 - half, d1 + half
        c0, c1 = d2 - half, d2 + half
        pr0, pr1 = max(0, -r0), max(0, r1 - H)
        pc0, pc1 = max(0, -c0), max(0, c1 - W)
        r0c, r1c = max(0, r0), min(H, r1)
        c0c, c1c = max(0, c0), min(W, c1)

        chunk = grp['image'][r0c:r1c, c0c:c1c, :][:, :, self.marker_indices].astype(np.float32)
        patch = np.zeros((ps, ps, len(self.marker_indices)), dtype=np.float32)
        patch[pr0:pr0 + (r1c - r0c), pc0:pc0 + (c1c - c0c)] = chunk

        # mask to centre cell only
        mchunk = grp['masks'][r0c:r1c, c0c:c1c]
        msk    = np.zeros((ps, ps), dtype=mchunk.dtype)
        msk[pr0:pr0 + (r1c - r0c), pc0:pc0 + (c1c - c0c)] = mchunk
        centre = msk[half, half]
        patch *= (msk == centre)[:, :, None]

        # central crop to patch_size, [C, H, W], normalise to [0, 1]
        img = torch.from_numpy(patch).permute(2, 0, 1)  # [C, H, W]
        img = img / (img.amax() + 1e-8)
        return img, self.labels[idx], self.sample_id[idx]


# ── Integrated Gradients ───────────────────────────────────────────────────────

def ig_batch(backbone, imgs, n_steps):
    """
    Integrated Gradients for a batch.

    Target: ||backbone(x)||²  — fully unsupervised.
    Baseline: zero patch (no marker expression).

    Returns:
        attribution [B, C]  — absolute IG summed over spatial dims
    """
    baseline   = torch.zeros_like(imgs)
    grad_accum = torch.zeros_like(imgs)

    for step in range(n_steps):
        alpha    = step / max(n_steps - 1, 1)
        x_interp = (baseline + alpha * (imgs - baseline)).detach().requires_grad_(True)
        emb      = backbone(x_interp)[0]      # [B, D, 1, 1] or [B, D]
        emb.pow(2).sum().backward()
        grad_accum += x_interp.grad.detach()

    avg_grads   = grad_accum / n_steps
    attribution = ((imgs - baseline) * avg_grads).abs()  # [B, C, H, W]
    return attribution.sum(dim=[-2, -1]).cpu()            # [B, C]


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir',      required=True)
    p.add_argument('--h5',             required=True)
    p.add_argument('--markers',        required=True)
    p.add_argument('--val',            required=True)
    p.add_argument('--out',            required=True)
    p.add_argument('--annotation_map', default=None,
                   help='e.g. "Cytotoxic CD8:CD8,TReg:Treg"')
    p.add_argument('--ignore',         default='Unidentified',
                   help='Comma-separated class names to ignore')
    p.add_argument('--patch_size',     type=int, default=22)
    p.add_argument('--n_steps',        type=int, default=50)
    p.add_argument('--batch_size',     type=int, default=32)
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

    ignore = [s.strip() for s in args.ignore.split(',') if s.strip()]

    # ── Load backbone (registers all mmengine modules via Config.fromfile) ──
    print('Loading backbone...')
    backbone = load_backbone(args.model_dir, device)

    with open(args.markers) as f:
        marker_names = [l.strip() for l in f if l.strip()]
    print(f'{len(marker_names)} markers: {marker_names[:5]} ...')

    # ── Build marker index map from HDF5 ───────────────────────────────────
    with h5py.File(args.h5, 'r') as f:
        all_marker_names = [decode(m) for m in f['marker_names'][:]]
    m2i            = {m: i for i, m in enumerate(all_marker_names)}
    marker_indices = np.array([m2i[m] for m in marker_names])

    # ── Dataset + DataLoader ───────────────────────────────────────────────
    dataset = PatchDataset(
        h5_path        = args.h5,
        marker_indices = marker_indices,
        indices_path   = args.val,
        patch_size     = args.patch_size,
        annotation_map = annotation_map,
        ignore         = ignore,
    )
    print(f'{len(dataset)} cells | classes: {sorted(set(dataset.labels))}')

    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.n_workers,
        pin_memory  = device.type == 'cuda',
    )

    # ── Compute IG ─────────────────────────────────────────────────────────
    all_attr, all_feats, all_labels, all_sids = [], [], [], []
    n_done = 0

    for imgs, labels, sids in tqdm(loader, desc='IG'):
        if args.max_cells and n_done >= args.max_cells:
            break
        imgs_gpu = imgs.to(device)
        with torch.no_grad():
            feats = backbone(imgs_gpu)[0]          # [B, D, 1, 1] or [B, D]
            feats = feats.flatten(1).cpu().numpy() # [B, D]
        attr = ig_batch(backbone, imgs_gpu, args.n_steps)
        all_attr.append(attr.numpy())
        all_feats.append(feats)
        all_labels.extend(labels)
        all_sids.extend(sids)
        n_done += len(labels)

    attr_matrix = np.concatenate(all_attr,  axis=0)  # [N, n_markers]
    feat_matrix = np.concatenate(all_feats, axis=0)  # [N, D]
    print(f'\nAttribution matrix: {attr_matrix.shape}')
    print(f'Feature matrix:     {feat_matrix.shape}')

    # ── Save ───────────────────────────────────────────────────────────────
    np.savez_compressed(
        out / 'attribution.npz',
        attribution  = attr_matrix,
        features     = feat_matrix,
        labels       = np.array(all_labels),
        sample_ids   = np.array(all_sids),
        marker_names = np.array(marker_names),
    )
    with open(out / 'marker_names.txt', 'w') as f:
        f.write('\n'.join(marker_names))

    print(f'Saved {out}/attribution.npz')
    print(f'Labels: {sorted(set(all_labels))}')


if __name__ == '__main__':
    main()

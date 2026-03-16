#!/usr/bin/env python3
"""Per-cell marker attribution via Integrated Gradients.

Computes a [N_cells × n_markers] importance matrix:
  For each cell, which markers drove its position in embedding space?

The importance vectors are then saved for downstream analysis:
  - Cluster on attribution vectors (Leiden / K-means)
  - UMAP of attribution space
  - Heatmap of mean attribution per cluster
  - Optionally validate against ground-truth labels

Attribution target: ||f(x)||² (L2 norm of backbone embedding).
Fully unsupervised — no labels used during attribution.
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
    attribution.npz          — [N, n_markers] importance matrix + labels + sample_ids
    marker_names.txt         — marker names in column order
    (clustering/UMAP downstream in notebooks or a separate script)
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
from tqdm import tqdm

from mmengine.registry import DATASETS
from torch.utils.data import DataLoader

import src.models   # noqa: registers WideModel, CIM_Funnel, etc.
import src.VICReg   # noqa: registers MVVICReg

from src.utils import load_checkpoint


# ── Model loading ──────────────────────────────────────────────────────────────

def load_backbone(model_dir, device):
    result   = load_checkpoint(str(model_dir), device=device)
    backbone = result['model'].backbone.eval()
    return backbone


# ── Data ───────────────────────────────────────────────────────────────────────

def build_dataloader(h5, markers_path, indices_path, patch_size,
                     annotation_map, batch_size, n_workers):
    pipeline = [
        dict(type='C_CentralCutter', size=patch_size),
        dict(type='C_ToTensor'),
    ]
    dataset = DATASETS.build(dict(
        type='MCIDataset',
        h5_filepath=h5,
        used_markers=markers_path,
        patch_size=patch_size + 8,
        ignore_annotation=None,
        annotation_map=annotation_map,
        used_indicies=indices_path,
        pipeline=pipeline,
    ))

    def collate(batch):
        imgs       = torch.stack([b['inputs'][0][0] for b in batch])
        labels     = [b['data_samples']['annotation'][0] for b in batch]
        sample_ids = [b['data_samples']['sample_id'][0]  for b in batch]
        return imgs, labels, sample_ids

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=n_workers, collate_fn=collate)


# ── Integrated Gradients ───────────────────────────────────────────────────────

def ig_batch(backbone, imgs, n_steps, device):
    """
    Compute per-cell, per-marker IG attribution for a batch.

    Target: ||backbone(x)||²  — fully unsupervised.
    Baseline: zero patch (no marker expression).

    Returns:
        attribution [B, C]  — absolute attribution summed over spatial dims
    """
    B, C, H, W = imgs.shape
    baseline   = torch.zeros_like(imgs)
    grad_accum = torch.zeros_like(imgs)

    for step in range(n_steps):
        alpha   = step / max(n_steps - 1, 1)
        x_interp = (baseline + alpha * (imgs - baseline)).detach().requires_grad_(True)

        emb    = backbone(x_interp)[0]         # [B, D, 1, 1] or [B, D]
        scalar = emb.pow(2).sum()              # L2 norm² — unsupervised target
        scalar.backward()

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

    # ── Load backbone ──────────────────────────────────────────────────────
    print('Loading backbone...')
    backbone = load_backbone(args.model_dir, device)

    with open(args.markers) as f:
        marker_names = [l.strip() for l in f if l.strip()]
    print(f'{len(marker_names)} markers')

    # ── Dataloader ─────────────────────────────────────────────────────────
    dl = build_dataloader(
        args.h5, args.markers, args.val,
        patch_size=args.patch_size,
        annotation_map=annotation_map,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
    )

    # ── Compute IG for all cells ───────────────────────────────────────────
    all_attr, all_labels, all_sids = [], [], []
    n_done = 0

    for imgs, labels, sids in tqdm(dl, desc='IG'):
        if args.max_cells and n_done >= args.max_cells:
            break
        attr = ig_batch(backbone, imgs.to(device), args.n_steps, device)
        all_attr.append(attr.numpy())
        all_labels.extend(labels)
        all_sids.extend(sids)
        n_done += len(labels)

    attr_matrix = np.concatenate(all_attr, axis=0)  # [N, n_markers]
    print(f'\nAttribution matrix: {attr_matrix.shape}  ({n_done} cells × {len(marker_names)} markers)')

    # ── Save ───────────────────────────────────────────────────────────────
    np.savez_compressed(
        out / 'attribution.npz',
        attribution  = attr_matrix,
        labels       = np.array(all_labels),
        sample_ids   = np.array(all_sids),
        marker_names = np.array(marker_names),
    )
    with open(out / 'marker_names.txt', 'w') as f:
        f.write('\n'.join(marker_names))

    print(f'Saved {out}/attribution.npz')
    print(f'  Shape: {attr_matrix.shape}')
    print(f'  Labels: {sorted(set(all_labels))}')
    print('\nNext steps (notebook / separate script):')
    print('  1. sc.pp.neighbors() on attribution matrix → Leiden clustering')
    print('  2. UMAP of attribution space, colored by cluster then by GT label')
    print('  3. Heatmap: mean attribution per cluster × marker')


if __name__ == '__main__':
    main()

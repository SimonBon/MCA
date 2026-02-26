"""
Analyse gate values of WideModelAttentionGated on a val set.

Usage:
    python scripts/analyse_gates.py \
        --checkpoint z_RUNS/MIBI_TNBC_CIMATT_Gate_VICReg/last_checkpoint \
        --config    z_RUNS/MIBI_TNBC_CIMATT_Gate_VICReg/CIMATT_Gate_VICReg.py \
        --h5        <path/to/MIBI_TNBC.h5> \
        --markers   <path/to/used_markers.txt> \
        --val_idx   <path/to/val.txt> \
        --outdir    z_RUNS/MIBI_TNBC_CIMATT_Gate_VICReg/gate_analysis
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# ── make sure MCA src is importable ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src  # registers MODELS / DATASETS / HOOKS via __init__
from src.models import WideModelAttentionGated

from mmengine.config import Config
from mmengine.registry import MODELS, DATASETS
from mmcv.transforms import Compose
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load model from mmengine checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def load_model(config_path, checkpoint_path, device='cuda'):
    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)

    ckpt_file = Path(checkpoint_path)
    if ckpt_file.suffix == '':          # last_checkpoint is a text file with the path
        ckpt_file = Path(ckpt_file.read_text().strip())

    state = torch.load(ckpt_file, map_location='cpu')
    state_dict = state.get('state_dict', state)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Hook: capture gate values from WideModelAttentionGated.forward
# ─────────────────────────────────────────────────────────────────────────────
class GateCapture:
    """Registers a forward hook that saves gates [B, C] after sigmoid."""
    def __init__(self, backbone: WideModelAttentionGated):
        self.gates = []
        self._hook = backbone.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, output):
        # Re-compute gates from the stored intermediate — simpler: patch forward
        pass

    def remove(self):
        self._hook.remove()


def patch_backbone_to_capture_gates(backbone: WideModelAttentionGated):
    """
    Monkey-patch forward so it also returns gate values.
    Returns a list that will be filled with [B, C] tensors per batch.
    """
    captured = []
    original_forward = backbone.forward

    def new_forward(x, *args, **kwargs):
        import torch.nn.functional as F

        if backbone.input_norm:
            x = F.normalize(x, dim=1)

        x = backbone.stem(x)
        x = backbone.layers(x)

        B, CD, H, W = x.shape
        C = backbone.in_channels
        D = backbone.stem_width

        tokens = x.view(B, C, D, H, W).mean(dim=(-2, -1))       # [B, C, D]
        rel = tokens - tokens.mean(dim=1, keepdim=True)
        gates = torch.sigmoid(
            backbone.gate_proj(rel).squeeze(-1)
            / backbone.gate_temp.abs().clamp(min=1e-4)
        )                                                         # [B, C]
        captured.append(gates.detach().cpu())

        # continue with original logic
        tokens_gated = tokens * gates.unsqueeze(-1)
        tokens_t = backbone.attn_norm(tokens_gated).transpose(0, 1)
        attn_out, _ = backbone.channel_attn(tokens_t, tokens_t, tokens_t)
        tokens_gated = tokens_gated + attn_out.transpose(0, 1)
        tokens_gated = tokens_gated + backbone.ffn(backbone.ffn_norm(tokens_gated))
        correction = tokens_gated.view(B, CD, 1, 1)
        x = x + correction
        out = x.mean(dim=(-2, -1)).view(B, CD, 1, 1)
        return (out,)

    backbone.forward = new_forward
    return captured


# ─────────────────────────────────────────────────────────────────────────────
# Build a minimal val dataloader
# ─────────────────────────────────────────────────────────────────────────────
def build_val_loader(h5_path, markers_path, val_idx_path, patch_size=32, batch_size=256, num_workers=8):
    from src.dataset import MCIDataset  # noqa: ensure registered

    val_pipeline = [
        dict(type='CenterCrop', size=patch_size),
        dict(type='ToTensor'),
    ]

    dataset = DATASETS.build(dict(
        type='MCIDataset',
        h5_filepath=h5_path,
        patch_size=patch_size,
        used_markers=markers_path,
        used_indicies=val_idx_path,
        ignore_annotation=['Unidentified'],
        pipeline=val_pipeline,
    ))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda b: b,          # raw list of dicts
    )
    return loader, dataset


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
def plot_gate_heatmap(mean_gates, marker_names, class_names, outpath):
    """Heatmap: rows = cell types, cols = markers, values = mean gate."""
    fig, ax = plt.subplots(figsize=(max(12, len(marker_names) * 0.4),
                                    max(6,  len(class_names)  * 0.4)))
    sns.heatmap(
        mean_gates,
        xticklabels=marker_names,
        yticklabels=class_names,
        cmap='viridis',
        ax=ax,
        vmin=0, vmax=1,
        linewidths=0.3,
    )
    ax.set_title('Mean sigmoid gate value per cell type per marker')
    ax.set_xlabel('Marker')
    ax.set_ylabel('Cell type')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'Saved gate heatmap → {outpath}')


def plot_gate_variance_heatmap(std_gates, marker_names, class_names, outpath):
    """Heatmap of within-class gate std — high = fragmentation driver."""
    fig, ax = plt.subplots(figsize=(max(12, len(marker_names) * 0.4),
                                    max(6,  len(class_names)  * 0.4)))
    sns.heatmap(
        std_gates,
        xticklabels=marker_names,
        yticklabels=class_names,
        cmap='rocket_r',
        ax=ax,
        linewidths=0.3,
    )
    ax.set_title('Within-class gate std per marker (high = fragmentation driver)')
    ax.set_xlabel('Marker')
    ax.set_ylabel('Cell type')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'Saved gate variance heatmap → {outpath}')


def plot_gate_boxplots(all_gates, all_labels, marker_names, class_names, outdir):
    """One figure per marker: boxplot of gate distribution per cell type."""
    mdir = Path(outdir) / 'per_marker_boxplots'
    mdir.mkdir(exist_ok=True)
    for mi, mname in enumerate(marker_names):
        fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.6), 4))
        data = [all_gates[all_labels == c, mi] for c in class_names]
        ax.boxplot(data, labels=class_names, patch_artist=True)
        ax.set_title(f'Gate distribution for marker: {mname}')
        ax.set_ylabel('Sigmoid gate value')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        safe = mname.replace('/', '_').replace(' ', '_')
        fig.savefig(mdir / f'{safe}.png', dpi=100)
        plt.close(fig)
    print(f'Saved per-marker boxplots → {mdir}')


def plot_umap_colored_by_gates(umap_coords, all_gates, marker_names, outdir):
    """One UMAP per marker colored by gate value."""
    mdir = Path(outdir) / 'umap_per_marker'
    mdir.mkdir(exist_ok=True)
    # Show only top-N most variable markers to avoid clutter
    gate_var = all_gates.var(axis=0)
    top_idx  = np.argsort(gate_var)[::-1][:min(12, len(marker_names))]

    for mi in top_idx:
        mname = marker_names[mi]
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                        c=all_gates[:, mi], cmap='plasma', s=1, alpha=0.5,
                        vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, label='gate value')
        ax.set_title(f'UMAP coloured by gate — {mname}')
        ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
        plt.tight_layout()
        safe = mname.replace('/', '_').replace(' ', '_')
        fig.savefig(mdir / f'{safe}.png', dpi=120)
        plt.close(fig)
    print(f'Saved gate-coloured UMAPs → {mdir}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config',     required=True)
    parser.add_argument('--h5',         required=True)
    parser.add_argument('--markers',    required=True)
    parser.add_argument('--val_idx',    required=True)
    parser.add_argument('--outdir',     required=True)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────────
    print('Loading model …')
    model = load_model(args.config, args.checkpoint, device=args.device)

    # find backbone
    backbone = None
    for m in model.modules():
        if isinstance(m, WideModelAttentionGated):
            backbone = m
            break
    assert backbone is not None, 'Could not find WideModelAttentionGated in model.'

    captured_gates = patch_backbone_to_capture_gates(backbone)

    # ── marker names ──────────────────────────────────────────────────────────
    marker_names = Path(args.markers).read_text().strip().splitlines()
    print(f'Markers ({len(marker_names)}): {marker_names}')

    # ── val dataloader ────────────────────────────────────────────────────────
    print('Building val dataloader …')
    loader, dataset = build_val_loader(
        args.h5, args.markers, args.val_idx,
        patch_size=args.patch_size, batch_size=args.batch_size,
    )

    # ── inference ─────────────────────────────────────────────────────────────
    print('Running inference …')
    all_labels   = []
    all_sample   = []

    with torch.no_grad():
        for batch in tqdm(loader):
            imgs = torch.stack([b['inputs'][0] for b in batch]).float().to(args.device)
            model([imgs], mode='tensor')                          # gates captured as side-effect
            all_labels.extend([b['data_samples'].annotation for b in batch])
            all_sample.extend([b['data_samples'].sample_id  for b in batch])

    all_gates  = torch.cat(captured_gates, dim=0).numpy()        # [N, C]
    all_labels = np.array(all_labels)
    all_sample = np.array(all_sample)

    print(f'Collected gates: {all_gates.shape}  labels: {all_labels.shape}')
    np.save(outdir / 'gates.npy',  all_gates)
    np.save(outdir / 'labels.npy', all_labels)
    np.save(outdir / 'sample.npy', all_sample)

    # ── per-class statistics ──────────────────────────────────────────────────
    class_names = sorted(set(all_labels))
    mean_gates  = np.stack([all_gates[all_labels == c].mean(axis=0) for c in class_names])
    std_gates   = np.stack([all_gates[all_labels == c].std(axis=0)  for c in class_names])

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_gate_heatmap(mean_gates, marker_names, class_names,
                      outdir / 'gate_heatmap_mean.png')
    plot_gate_variance_heatmap(std_gates, marker_names, class_names,
                               outdir / 'gate_heatmap_std.png')
    plot_gate_boxplots(all_gates, all_labels, marker_names, class_names, outdir)

    # ── UMAP (reuse existing coords if available) ─────────────────────────────
    umap_file = Path(args.checkpoint).parent / 'umap_coords.npy'
    if not umap_file.exists():
        print('Computing UMAP …')
        from umap import UMAP
        # re-collect features via a second pass (gates are reset)
        captured_gates.clear()
        all_feats = []
        with torch.no_grad():
            for batch in tqdm(loader):
                imgs = torch.stack([b['inputs'][0] for b in batch]).float().to(args.device)
                feats = model([imgs], mode='tensor')[0].squeeze()
                all_feats.append(feats.cpu().numpy())
        all_feats = np.concatenate(all_feats, axis=0)
        umap_coords = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(all_feats)
        np.save(umap_file, umap_coords)
    else:
        print(f'Loading UMAP coords from {umap_file}')
        umap_coords = np.load(umap_file)

    if len(umap_coords) == len(all_gates):
        plot_umap_colored_by_gates(umap_coords, all_gates, marker_names, outdir)
    else:
        print(f'UMAP coord count ({len(umap_coords)}) != gate count ({len(all_gates)}), skipping UMAP plots.')

    print(f'\nDone. All outputs in {outdir}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Plot loss curve for any z_RUNS directory.

Usage:
    python tools/plot_loss.py z_RUNS/CODEX_cHL_CIM_VICReg_KRONOS18
    python tools/plot_loss.py z_RUNS/CODEX_cHL_CIM_VICReg_KRONOS18 --zoom 0.6 --smooth 30
"""
import argparse
import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def smooth(arr, w):
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode='same')


def plot_loss(work_dir, smooth_window=50, zoom_frac=0.5):
    records = []
    for path in glob.glob(os.path.join(work_dir, '**/scalars.json'), recursive=True):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if not records:
        print(f"No scalars.json found under {work_dir}")
        return

    records.sort(key=lambda r: r.get('step', 0))
    steps    = np.array([r['step']                  for r in records], dtype=float)
    loss     = np.array([r.get('loss',     np.nan)  for r in records])
    loss_inv = np.array([r.get('loss_inv', np.nan)  for r in records])
    loss_var = np.array([r.get('loss_var', np.nan)  for r in records])
    loss_cov = np.array([r.get('loss_cov', np.nan)  for r in records])
    lr       = np.array([r.get('lr',       np.nan)  for r in records])

    loss_s     = smooth(loss,     smooth_window)
    loss_inv_s = smooth(loss_inv, smooth_window)
    loss_var_s = smooth(loss_var, smooth_window)
    loss_cov_s = smooth(loss_cov, smooth_window)

    n      = len(steps)
    zoom_i = int(n * (1 - zoom_frac))
    alpha_r = 0.15

    fig, axes = plt.subplots(3, 1, figsize=(12, 12),
                             gridspec_kw={'hspace': 0.45})
    title_prefix = os.path.basename(os.path.normpath(work_dir))

    # ── Row 0: full curve, log y ──────────────────────────────────────────
    ax = axes[0]
    ax.plot(steps, loss,   color='steelblue', alpha=alpha_r, linewidth=0.5)
    ax.plot(steps, loss_s, color='steelblue', linewidth=1.8, label='total loss (smoothed)')
    ax.set_yscale('log')
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss (log scale)', color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax.set_title(f'{title_prefix} — full training (log scale) + LR')
    ax2 = ax.twinx()
    ax2.plot(steps, lr, color='tomato', linewidth=1.2, linestyle='--', alpha=0.8, label='LR')
    ax2.set_ylabel('learning rate', color='tomato')
    ax2.tick_params(axis='y', labelcolor='tomato')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    warmup_end = steps[np.argmax(lr)]
    ax.axvspan(steps[0], warmup_end, alpha=0.08, color='orange', label='warmup')

    # ── Row 1: zoomed total loss, linear y ───────────────────────────────
    ax = axes[1]
    zs, zl, zls = steps[zoom_i:], loss[zoom_i:], loss_s[zoom_i:]
    ax.plot(zs, zl,  color='steelblue', alpha=alpha_r, linewidth=0.5)
    ax.plot(zs, zls, color='steelblue', linewidth=1.8)
    lo, hi = np.nanpercentile(zls, 1), np.nanpercentile(zls, 99)
    pad = (hi - lo) * 0.15
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel('iteration')
    ax.set_ylabel('total loss')
    ax.set_title(f'Last {int(zoom_frac*100)}% — total loss (linear, clipped y-axis)')
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    # ── Row 2: component losses, zoomed ──────────────────────────────────
    ax = axes[2]
    ax.plot(steps[zoom_i:], loss_inv[zoom_i:], color='C0', alpha=alpha_r, linewidth=0.5)
    ax.plot(steps[zoom_i:], loss_var[zoom_i:], color='C1', alpha=alpha_r, linewidth=0.5)
    ax.plot(steps[zoom_i:], loss_cov[zoom_i:], color='C2', alpha=alpha_r, linewidth=0.5)
    ax.plot(steps[zoom_i:], loss_inv_s[zoom_i:], color='C0', linewidth=1.5, label='invariance')
    ax.plot(steps[zoom_i:], loss_var_s[zoom_i:], color='C1', linewidth=1.5, label='variance')
    ax.plot(steps[zoom_i:], loss_cov_s[zoom_i:], color='C2', linewidth=1.5, label='covariance')
    all_comp = np.concatenate([loss_inv_s[zoom_i:], loss_var_s[zoom_i:], loss_cov_s[zoom_i:]])
    lo2, hi2 = np.nanpercentile(all_comp, 1), np.nanpercentile(all_comp, 99)
    pad2 = (hi2 - lo2) * 0.15
    ax.set_ylim(lo2 - pad2, hi2 + pad2)
    ax.set_xlabel('iteration')
    ax.set_ylabel('component loss')
    ax.set_title(f'Last {int(zoom_frac*100)}% — VICReg components (smoothed)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    out = os.path.join(work_dir, 'loss_curve.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('work_dir', help='Path to z_RUNS/<run> directory')
    parser.add_argument('--smooth', type=int,   default=50,  help='Smoothing window (default: 50)')
    parser.add_argument('--zoom',   type=float, default=0.5, help='Fraction of training to zoom into (default: 0.5)')
    args = parser.parse_args()
    plot_loss(args.work_dir, smooth_window=args.smooth, zoom_frac=args.zoom)

#!/usr/bin/env python3
"""Compare IG attribution matrices across different n_steps vs the baseline (20 steps).

Metrics per n_steps:
  - Pearson r with baseline (mean over all cells)
  - Spearman rank correlation with baseline (per-cell top-marker rank agreement)
  - Top-1 marker agreement (fraction of cells where argmax matches baseline)
  - Runtime is not measured here — see SLURM logs

Usage:
    python tools/compare_nsteps.py \
        --sweep_dir /nobackup/.../marker_attribution/nsteps_sweep \
        --baseline  /nobackup/.../marker_attribution/CODEX_cHL_CIM_Funnel/attribution.npz \
        --out       /nobackup/.../marker_attribution/nsteps_sweep
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def load_attr(path):
    d = np.load(path, allow_pickle=True)
    attr = d['attribution'].astype(np.float32)
    # row-normalise
    attr = attr / (attr.sum(axis=1, keepdims=True) + 1e-8)
    return attr


def compare(attr, baseline):
    N = attr.shape[0]
    pearson_vals, spearman_vals, top1_agree = [], [], []

    for i in range(N):
        r, _  = pearsonr(attr[i], baseline[i])
        sr, _ = spearmanr(attr[i], baseline[i])
        pearson_vals.append(r)
        spearman_vals.append(sr)
        top1_agree.append(int(np.argmax(attr[i]) == np.argmax(baseline[i])))

    return (np.mean(pearson_vals), np.std(pearson_vals),
            np.mean(spearman_vals), np.std(spearman_vals),
            np.mean(top1_agree))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep_dir', required=True)
    p.add_argument('--baseline',  required=True)
    p.add_argument('--out',       required=True)
    args = p.parse_args()

    out      = Path(args.out)
    baseline = load_attr(args.baseline)
    print(f'Baseline: {baseline.shape}')

    sweep_dir = Path(args.sweep_dir)
    step_dirs = sorted(sweep_dir.glob('steps_*'),
                       key=lambda p: int(p.name.split('_')[1]))

    results = []
    for d in step_dirs:
        npz = d / 'attribution.npz'
        if not npz.exists():
            print(f'  Skipping {d.name} — no attribution.npz')
            continue
        n_steps = int(d.name.split('_')[1])
        attr    = load_attr(npz)
        pr, pr_std, sr, sr_std, top1 = compare(attr, baseline)
        results.append((n_steps, pr, pr_std, sr, sr_std, top1))
        print(f'  steps={n_steps:>2}  pearson={pr:.4f}±{pr_std:.4f}  '
              f'spearman={sr:.4f}±{sr_std:.4f}  top1_agree={top1*100:.1f}%')

    if not results:
        print('No results found.')
        return

    steps    = [r[0] for r in results]
    pearsons = [r[1] for r in results]
    spearmans= [r[3] for r in results]
    top1s    = [r[5] for r in results]

    # Add baseline point
    steps    += [20]; pearsons += [1.0]; spearmans += [1.0]; top1s += [1.0]
    order     = np.argsort(steps)
    steps     = [steps[i]     for i in order]
    pearsons  = [pearsons[i]  for i in order]
    spearmans = [spearmans[i] for i in order]
    top1s     = [top1s[i]     for i in order]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, vals, title, ylabel in zip(
        axes,
        [pearsons, spearmans, top1s],
        ['Pearson r vs baseline', 'Spearman r vs baseline', 'Top-1 marker agreement'],
        ['Pearson r', 'Spearman r', 'Fraction'],
    ):
        ax.plot(steps, vals, 'o-', color='steelblue', linewidth=2, markersize=6)
        ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8)
        ax.set_xlabel('n_steps (IG)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(steps)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle('IG attribution quality vs n_steps (baseline = 20 steps)',
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(out / 'nsteps_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out}/nsteps_comparison.png')


if __name__ == '__main__':
    main()

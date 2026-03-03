#!/usr/bin/env python3
"""Label efficiency evaluation for a trained model.

Loads pre-extracted features from train_results.npz / val_results.npz
(produced by EvaluateModelRich), fits a linear probe at several label
budgets, and saves results as JSON + PNG.

Two sampling modes (combinable):
  --fractions   fraction of each class in (0, 1], e.g. 0.01 0.05 0.1 0.25 0.5 1.0
  --n_per_class fixed number of cells per class, e.g. 5 10 50 100

Sampling is always per-class. For fractions, max(1, round(n_class * f))
cells are drawn so rare classes always get at least one sample.
For n_per_class, min(n, n_class) cells are drawn.

Usage:
    # fractions only (default)
    python tools/label_efficiency.py z_RUNS/CODEX_cHL_CIM_VICReg_KRONOS18

    # n_per_class only
    python tools/label_efficiency.py z_RUNS/CODEX_cHL_CIM_VICReg_KRONOS18 \\
        --n_per_class 5 10 50 100

    # both combined
    python tools/label_efficiency.py z_RUNS/CODEX_cHL_CIM_VICReg_KRONOS18 \\
        --fractions 0.01 0.1 0.5 1.0 --n_per_class 5 10 50 100 --n_repeats 3
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample_by_fraction(train_labels, frac, n_classes, rng):
    """Sample max(1, round(n_class * frac)) cells from each class."""
    indices = []
    for c in range(n_classes):
        class_idx = np.where(train_labels == c)[0]
        if len(class_idx) == 0:
            continue
        n_take = max(1, int(round(len(class_idx) * frac)))
        n_take = min(n_take, len(class_idx))
        indices.append(rng.choice(class_idx, n_take, replace=False))
    return np.concatenate(indices)


def _sample_by_n(train_labels, n, n_classes, rng):
    """Sample min(n, n_class) cells from each class."""
    indices = []
    for c in range(n_classes):
        class_idx = np.where(train_labels == c)[0]
        if len(class_idx) == 0:
            continue
        n_take = min(n, len(class_idx))
        indices.append(rng.choice(class_idx, n_take, replace=False))
    return np.concatenate(indices)


def _lp_metrics(train_feats, train_labels, val_feats, val_labels,
                n_classes, epochs, n_jobs):
    """Fit LP and return (bal_acc, mean_ap)."""
    clf = LogisticRegression(
        solver='lbfgs', penalty='l2', max_iter=epochs,
        class_weight='balanced', C=1, n_jobs=n_jobs, tol=1e-6,
    )
    clf.fit(train_feats, train_labels)
    val_pred  = clf.predict(val_feats)
    val_proba = clf.predict_proba(val_feats)

    bal_acc = float(balanced_accuracy_score(val_labels, val_pred))

    val_bin = label_binarize(val_labels, classes=list(range(n_classes)))
    if val_bin.shape[1] == 1:
        val_bin = np.hstack([1 - val_bin, val_bin])
    mean_ap = float(np.mean(average_precision_score(val_bin, val_proba, average=None)))

    return bal_acc, mean_ap


def _run_point(label, sample_fn, train_feats, train_labels, val_feats, val_labels,
               n_classes, n_repeats, epochs, n_jobs):
    """Run n_repeats LP evaluations for one budget point. Returns result dict."""
    bal_accs, mean_aps, n_labeled_list = [], [], []

    for rep in range(n_repeats):
        rng = np.random.default_rng(seed=42 + rep)
        idx = sample_fn(rng)
        ba, ap = _lp_metrics(
            train_feats[idx], train_labels[idx],
            val_feats, val_labels,
            n_classes, epochs, n_jobs,
        )
        bal_accs.append(ba)
        mean_aps.append(ap)
        n_labeled_list.append(len(idx))
        print(f"    rep {rep+1} (n={len(idx)}): bal_acc={ba:.4f}  mean_ap={ap:.4f}")

    print(f"    => bal_acc {np.mean(bal_accs):.4f}±{np.std(bal_accs):.4f}  "
          f"mean_ap {np.mean(mean_aps):.4f}±{np.std(mean_aps):.4f}")

    return {
        'label':        label,
        'n_labeled':    int(np.mean(n_labeled_list)),
        'n_repeats':    len(bal_accs),
        'bal_acc_mean': float(np.mean(bal_accs)),
        'bal_acc_std':  float(np.std(bal_accs)),
        'mean_ap_mean': float(np.mean(mean_aps)),
        'mean_ap_std':  float(np.std(mean_aps)),
        'bal_acc_runs': [round(v, 5) for v in bal_accs],
        'mean_ap_runs': [round(v, 5) for v in mean_aps],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run(work_dir, fractions, n_per_class, n_repeats, epochs, n_jobs):
    train_path = os.path.join(work_dir, 'train_results.npz')
    val_path   = os.path.join(work_dir, 'val_results.npz')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Not found: {train_path}\n"
                                "Run the model first (EvaluateModelRich saves this file).")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Not found: {val_path}")

    # validate fractions
    bad = [f for f in fractions if not (0 < f <= 1.0)]
    if bad:
        print(f"ERROR: --fractions must be in (0, 1]. Invalid values: {bad}\n"
              f"       To use a fixed count per class use --n_per_class instead.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading features from {work_dir} ...")
    train_npz = np.load(train_path, allow_pickle=True)
    val_npz   = np.load(val_path,   allow_pickle=True)

    train_feats  = train_npz['features']
    train_labels = train_npz['labels_num'].astype(int)
    val_feats    = val_npz['features']
    val_labels   = val_npz['labels_num'].astype(int)
    classes      = list(train_npz['classes'])
    n_classes    = len(classes)

    class_counts = np.bincount(train_labels, minlength=n_classes)
    print(f"Train: {len(train_feats)} cells  |  Val: {len(val_feats)} cells  |  "
          f"Classes ({n_classes}): {classes}")
    print(f"Train class counts: { {c: int(class_counts[i]) for i, c in enumerate(classes)} }")

    points = []   # list of result dicts, sorted by n_labeled at the end

    # ── N-per-class points ────────────────────────────────────────────────────
    for n in sorted(set(n_per_class)):
        label = f"{n}/cls"
        print(f"\n  [{label}] fixed {n} cells per class, {n_repeats} repeat(s):")
        result = _run_point(
            label=label,
            sample_fn=lambda rng, n_=n: _sample_by_n(train_labels, n_, n_classes, rng),
            train_feats=train_feats, train_labels=train_labels,
            val_feats=val_feats,     val_labels=val_labels,
            n_classes=n_classes, n_repeats=n_repeats, epochs=epochs, n_jobs=n_jobs,
        )
        points.append(result)
        
    # ── Fraction-based points ─────────────────────────────────────────────────
    for frac in sorted(set(fractions)):
        label    = f"{frac*100:.0f}%"
        # frac=1.0 → deterministic, only 1 repeat needed
        reps = 1 if frac >= 1.0 else n_repeats
        print(f"\n  [{label}] per-class fraction, {reps} repeat(s):")
        result = _run_point(
            label=label,
            sample_fn=lambda rng, f=frac: _sample_by_fraction(train_labels, f, n_classes, rng),
            train_feats=train_feats, train_labels=train_labels,
            val_feats=val_feats,     val_labels=val_labels,
            n_classes=n_classes, n_repeats=reps, epochs=epochs, n_jobs=n_jobs,
        )
        points.append(result)
        
    # sort by n_labeled for a clean plot
    points.sort(key=lambda p: p['n_labeled'])

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = os.path.join(work_dir, 'label_efficiency.json')
    payload  = {
        'classes':   classes,
        'n_classes': n_classes,
        'epochs':    epochs,
        'n_repeats': n_repeats,
        'points':    points,
    }
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    x    = np.array([p['n_labeled']    for p in points])
    ba_m = np.array([p['bal_acc_mean'] for p in points])
    ba_s = np.array([p['bal_acc_std']  for p in points])
    ap_m = np.array([p['mean_ap_mean'] for p in points])
    ap_s = np.array([p['mean_ap_std']  for p in points])
    lbls = [p['label'] for p in points]

    # squares for n/cls points, circles for fraction points
    markers = ['s' if p['label'].endswith('/cls') else 'o' for p in points]

    title = os.path.basename(os.path.normpath(work_dir))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Label Efficiency — {title}', fontsize=12)

    for ax, means, stds, ylabel, color in [
        (ax1, ba_m, ba_s, 'Balanced Accuracy (val)', 'steelblue'),
        (ax2, ap_m, ap_s, 'Mean Average Precision (val)', 'darkorange'),
    ]:
        ax.plot(x, means, '-', color=color, linewidth=1.5, zorder=1)
        for xi, yi, si, mk in zip(x, means, stds, markers):
            ax.plot(xi, yi, mk, color=color, markersize=7, zorder=2)
            if si > 0:
                ax.errorbar(xi, yi, yerr=si, fmt='none', color=color,
                            capsize=3, linewidth=1, zorder=2)
        ax.set_xscale('log')
        ax.set_xlabel('# labeled training cells')
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        for lbl, xi, yi in zip(lbls, x, means):
            ax.annotate(lbl, xy=(xi, yi), xytext=(0, 8),
                        textcoords='offset points', ha='center', fontsize=8)

    from matplotlib.lines import Line2D
    ax2.legend(handles=[
        Line2D([0], [0], marker='o', color='grey', linestyle='none', markersize=7, label='fraction'),
        Line2D([0], [0], marker='s', color='grey', linestyle='none', markersize=7, label='n/class'),
    ], fontsize=8, loc='lower right')

    plt.tight_layout()
    out_png = os.path.join(work_dir, 'label_efficiency.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Label efficiency evaluation from saved feature NPZs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # fractions only (default: 1%,5%,10%,25%,50%,100% per class)
  python tools/label_efficiency.py z_RUNS/CODEX_cHL_CIM_VICReg_KRONOS18

  # fixed count per class only
  python tools/label_efficiency.py z_RUNS/CODEX_cHL_CIM_VICReg_KRONOS18 \\
      --n_per_class 5 10 50 100

  # both combined
  python tools/label_efficiency.py z_RUNS/CODEX_cHL_CIM_VICReg_KRONOS18 \\
      --fractions 0.01 0.1 0.5 1.0 --n_per_class 5 10 50 100
""")
    parser.add_argument('work_dir',
                        help='Path to z_RUNS/<run> directory containing '
                             'train_results.npz and val_results.npz')
    parser.add_argument('--fractions', nargs='+', type=float,
                        default=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
                        metavar='F',
                        help='Per-class label fractions in (0,1] '
                             '(default: 0.01 0.05 0.1 0.25 0.5 1.0)')
    parser.add_argument('--n_per_class', nargs='+', type=int,
                        default=[],
                        metavar='N',
                        help='Fixed cells per class, e.g. 5 10 50 100')
    parser.add_argument('--n_repeats', type=int, default=3,
                        help='Repeats per point with different random seeds (default: 3)')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Max LP solver iterations (default: 2000)')
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='Parallel jobs for LP (default: 4)')
    args = parser.parse_args()

    run(
        work_dir=args.work_dir,
        fractions=args.fractions,
        n_per_class=args.n_per_class,
        n_repeats=args.n_repeats,
        epochs=args.epochs,
        n_jobs=args.n_jobs,
    )

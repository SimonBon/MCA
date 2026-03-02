#!/usr/bin/env python3
"""Label efficiency evaluation for a trained model.

Loads pre-extracted features from train_results.npz / val_results.npz
(produced by EvaluateModelRich), fits a linear probe at several label
fractions, and saves results as JSON + PNG.

Usage:
    python tools/label_efficiency.py z_RUNS/CODEX_cHL_CIM_VICReg_KRONOS18
    python tools/label_efficiency.py z_RUNS/CODEX_cHL_CIM_VICReg_KRONOS18 \\
        --fractions 0.01 0.05 0.1 0.25 0.5 1.0 --n_repeats 3 --n_jobs 8
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lp_metrics(train_feats, train_labels, val_feats, val_labels,
                n_classes, epochs, n_jobs):
    """Fit LP and return (bal_acc, mean_ap)."""
    clf = LogisticRegression(
        solver='lbfgs', penalty='l2', max_iter=epochs,
        class_weight='balanced', C=10, n_jobs=n_jobs,
    )
    clf.fit(train_feats, train_labels)
    val_pred  = clf.predict(val_feats)
    val_proba = clf.predict_proba(val_feats)

    bal_acc = float(balanced_accuracy_score(val_labels, val_pred))

    val_bin = label_binarize(val_labels, classes=list(range(n_classes)))
    # label_binarize returns (N,1) for binary — handle gracefully
    if val_bin.shape[1] == 1:
        val_bin = np.hstack([1 - val_bin, val_bin])
    mean_ap = float(np.mean(average_precision_score(val_bin, val_proba, average=None)))

    return bal_acc, mean_ap


# ── Main ──────────────────────────────────────────────────────────────────────

def run(work_dir, fractions, n_repeats, epochs, n_jobs):
    train_path = os.path.join(work_dir, 'train_results.npz')
    val_path   = os.path.join(work_dir, 'val_results.npz')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Not found: {train_path}\n"
                                "Run the model first (EvaluateModelRich saves this file).")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Not found: {val_path}")

    print(f"Loading features from {work_dir} ...")
    train_npz = np.load(train_path, allow_pickle=True)
    val_npz   = np.load(val_path,   allow_pickle=True)

    train_feats  = train_npz['features']
    train_labels = train_npz['labels_num'].astype(int)
    val_feats    = val_npz['features']
    val_labels   = val_npz['labels_num'].astype(int)
    classes      = list(train_npz['classes'])
    n_classes    = len(classes)

    print(f"Train: {len(train_feats)} cells  |  Val: {len(val_feats)} cells  |  "
          f"Classes ({n_classes}): {classes}")

    # ── Evaluate at each fraction ─────────────────────────────────────────────
    results = {}   # fraction -> {'bal_acc': [...], 'mean_ap': [...]}

    for frac in fractions:
        bal_accs, mean_aps = [], []
        n_labeled = int(round(len(train_feats) * frac))
        # Need at least 1 sample per class; skip if impossible
        if n_labeled < n_classes:
            print(f"  frac={frac:.2f}: only {n_labeled} samples < {n_classes} classes — skipping")
            continue

        label_str = f"{frac*100:.0f}%"
        print(f"\n  Fraction {label_str} ({n_labeled} samples), {n_repeats} repeat(s):")

        if frac >= 1.0:
            # Use full training set — no subsampling needed
            ba, ap = _lp_metrics(train_feats, train_labels,
                                  val_feats, val_labels,
                                  n_classes, epochs, n_jobs)
            bal_accs.append(ba)
            mean_aps.append(ap)
            print(f"    bal_acc={ba:.4f}  mean_ap={ap:.4f}")
        else:
            sss = StratifiedShuffleSplit(
                n_splits=n_repeats, train_size=frac, random_state=42)
            for rep, (idx, _) in enumerate(sss.split(train_feats, train_labels)):
                ba, ap = _lp_metrics(
                    train_feats[idx], train_labels[idx],
                    val_feats, val_labels,
                    n_classes, epochs, n_jobs,
                )
                bal_accs.append(ba)
                mean_aps.append(ap)
                print(f"    rep {rep+1}: bal_acc={ba:.4f}  mean_ap={ap:.4f}")

        results[frac] = {
            'n_labeled':   n_labeled,
            'n_repeats':   len(bal_accs),
            'bal_acc_mean': float(np.mean(bal_accs)),
            'bal_acc_std':  float(np.std(bal_accs)),
            'mean_ap_mean': float(np.mean(mean_aps)),
            'mean_ap_std':  float(np.std(mean_aps)),
            'bal_acc_runs': [round(v, 5) for v in bal_accs],
            'mean_ap_runs': [round(v, 5) for v in mean_aps],
        }
        print(f"    => bal_acc {np.mean(bal_accs):.4f}±{np.std(bal_accs):.4f}  "
              f"mean_ap {np.mean(mean_aps):.4f}±{np.std(mean_aps):.4f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = os.path.join(work_dir, 'label_efficiency.json')
    payload  = {
        'classes':   classes,
        'n_classes': n_classes,
        'epochs':    epochs,
        'n_repeats': n_repeats,
        'fractions': {
            str(frac): results[frac] for frac in sorted(results)
        },
    }
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    sorted_fracs = sorted(results)
    x    = np.array([results[f]['n_labeled'] for f in sorted_fracs])
    ba_m = np.array([results[f]['bal_acc_mean'] for f in sorted_fracs])
    ba_s = np.array([results[f]['bal_acc_std']  for f in sorted_fracs])
    ap_m = np.array([results[f]['mean_ap_mean'] for f in sorted_fracs])
    ap_s = np.array([results[f]['mean_ap_std']  for f in sorted_fracs])

    title = os.path.basename(os.path.normpath(work_dir))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Label Efficiency — {title}', fontsize=12)

    for ax, means, stds, ylabel, color in [
        (ax1, ba_m, ba_s, 'Balanced Accuracy (val)', 'steelblue'),
        (ax2, ap_m, ap_s, 'Mean Average Precision (val)', 'darkorange'),
    ]:
        ax.plot(x, means, 'o-', color=color, linewidth=2, markersize=6)
        if n_repeats > 1:
            ax.fill_between(x, means - stds, means + stds,
                            alpha=0.20, color=color)
        ax.set_xscale('log')
        ax.set_xlabel('# labeled training cells')
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        # annotate each point with the fraction label
        for frac, xi, yi in zip(sorted_fracs, x, means):
            ax.annotate(f'{frac*100:.0f}%', xy=(xi, yi),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=8)

    plt.tight_layout()
    out_png = os.path.join(work_dir, 'label_efficiency.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Label efficiency evaluation from saved feature NPZs.')
    parser.add_argument('work_dir',
                        help='Path to z_RUNS/<run> directory containing '
                             'train_results.npz and val_results.npz')
    parser.add_argument('--fractions', nargs='+', type=float,
                        default=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
                        help='Label fractions to evaluate (default: 0.01 0.05 0.1 0.25 0.5 1.0)')
    parser.add_argument('--n_repeats', type=int, default=3,
                        help='Stratified repeats per fraction (default: 3; '
                             'fraction=1.0 always uses 1 repeat)')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Max LP solver iterations (default: 2000)')
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='Parallel jobs for LP (default: 4)')
    args = parser.parse_args()

    run(
        work_dir=args.work_dir,
        fractions=args.fractions,
        n_repeats=args.n_repeats,
        epochs=args.epochs,
        n_jobs=args.n_jobs,
    )
